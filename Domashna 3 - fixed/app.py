from flask import Flask, render_template, request, jsonify
import sqlite3
import plotly.graph_objects as go
import ta
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)



def get_stock_data(page=1, table="stock_data", limit=10):
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()

    offset = (page - 1) * limit
    query = f"SELECT DISTINCT Код_на_издавач FROM {table} LIMIT {limit} OFFSET {offset}"
    cursor.execute(query)

    rows = cursor.fetchall()
    stock_data = [{'Код_на_издавач': row[0]} for row in rows]

    conn.close()
    return stock_data



def get_all_stock_data(table="stock_data"):
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()


    query = f"SELECT * FROM {table}"
    cursor.execute(query)

    rows = cursor.fetchall()


    stock_data = [
        {
            'Код_на_издавач': row[0],
            'Датум': row[1],
            'Цена_на_последна_трансакција': row[2],
            'Макс': row[3],
            'Мин': row[4],
            'Просечна_цена': row[5],
            'Промет_во_БЕСТ_во_денари': row[6],
            'Купен_промет_во_денари': row[7],
            'Количина': row[8],
            'Промет_во_Бест_во_денари_друга': row[9]
        }
        for row in rows
    ]

    conn.close()
    return stock_data



def get_total_issuers_count(table="stock_data"):
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()

    query = f"SELECT COUNT(DISTINCT Код_на_издавач) FROM {table}"
    cursor.execute(query)

    count = cursor.fetchone()[0]
    conn.close()
    return count


@app.route('/')
def home():
    page = request.args.get('page', default=1, type=int)
    limit = 10
    total_issuers = get_total_issuers_count()
    total_pages = (total_issuers + limit - 1) // limit

    stock_data = get_stock_data(page=page, limit=limit)
    return render_template('index.html', stock_data=stock_data, page=page, total_pages=total_pages)


@app.route('/analysis')
def analysis():

    issuer = request.args.get('issuer', default='', type=str).strip()
    page = request.args.get('page', default=1, type=int)
    limit = 10  

    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()

    query = "SELECT * FROM stock_data WHERE 1=1"
    params = []

    if issuer:
        query += " AND Код_на_издавач = ?"
        params.append(issuer)

    count_query = f"SELECT COUNT(*) FROM stock_data WHERE 1=1"
    if issuer:
        count_query += " AND Код_на_издавач = ?"
    cursor.execute(count_query, [issuer] if issuer else [])
    total_rows = cursor.fetchone()[0]
    total_pages = (total_rows + limit - 1) // limit

    query += " LIMIT ? OFFSET ?"
    params.extend([limit, (page - 1) * limit])

    cursor.execute(query, params)
    rows = cursor.fetchall()

    stock_data = [
        {
            'Код_на_издавач': row[0],
            'Датум': row[1],
            'Цена_на_последна_трансакција': row[2],
            'Макс': row[3],
            'Мин': row[4],
            'Просечна_цена': row[5],
            'Промет_во_БЕСТ_во_денари': row[6],
            'Купен_промет_во_денари': row[7],
            'Количина': row[8],
            'Промет_во_Бест_во_денари_друга': row[9],
        }
        for row in rows
    ]

    conn.close()

    return render_template(
        'analysis.html',
        stock_data=stock_data,
        page=page,
        total_pages=total_pages,
        issuer=issuer,
        max=max,
        min=min,
    )

@app.route('/issuer/<issuer_code>')
def issuer_details(issuer_code):
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()

    query = "SELECT * FROM stock_data WHERE Код_на_издавач = ?"
    cursor.execute(query, (issuer_code,))
    rows = cursor.fetchall()

    stock_data = [
        {
            'Код_на_издавач': row[0],
            'Датум': row[1],
            'Цена_на_последна_трансакција': row[2],
            'Макс': row[3],
            'Мин': row[4],
            'Просечна_цена': row[5],
            'Промет_во_БЕСТ_во_денари': row[6],
            'Купен_промет_во_денари': row[7],
            'Количина': row[8],
            'Промет_во_Бест_во_денари_друга': row[9],
        }
        for row in rows
    ]

    conn.close()
    return render_template('issuer.html', issuer_code=issuer_code, stock_data=stock_data)

def calculate_technical_indicators(df):

    df['Цена_на_последна_трансакција'] = df['Цена_на_последна_трансакција'].str.replace(',', '').astype(float)
    df['Мак_'] = df['Мак_'].str.replace(',', '').astype(float)
    df['Мин_'] = df['Мин_'].str.replace(',', '').astype(float)

    df['Датум'] = pd.to_datetime(df['Датум'], errors='coerce')

    df = df.dropna(subset=['Цена_на_последна_трансакција', 'Мак_', 'Мин_', 'Датум'])

    df = df.sort_values('Датум')

    row_count = len(df)
    print(f"Number of rows available after cleaning: {row_count}")

    if row_count < 3:
        df['InsufficientData'] = True
        return df

    df['SMA10'] = df['Цена_на_последна_трансакција'].rolling(window=min(10, row_count)).mean()
    df['SMA50'] = df['Цена_на_последна_трансакција'].rolling(window=min(50, row_count)).mean()
    df['EMA10'] = df['Цена_на_последна_трансакција'].ewm(span=min(10, row_count), adjust=False).mean()
    df['EMA50'] = df['Цена_на_последна_трансакција'].ewm(span=min(50, row_count), adjust=False).mean()

    df['RSI'] = ta.momentum.RSIIndicator(df['Цена_на_последна_трансакција'], window=min(14, row_count)).rsi()
    df['MACD'] = ta.trend.MACD(df['Цена_на_последна_трансакција']).macd()
    df['CCI'] = ta.trend.CCIIndicator(
        high=df['Мак_'], low=df['Мин_'], close=df['Цена_на_последна_трансакција'], window=min(20, row_count)
    ).cci()

    if row_count >= 14:
        try:
            df['ADX'] = ta.trend.ADXIndicator(
                high=df['Мак_'], low=df['Мин_'], close=df['Цена_на_последна_трансакција'], window=14
            ).adx()
        except IndexError:
            print("IndexError during ADX calculation: Not enough data for ADX.")
            df['ADX'] = None
    else:
        df['ADX'] = None

    if row_count < 14:

        df['Signal'] = 'Hold'
        df['Price_Change'] = df['Цена_на_последна_трансакција'].diff()

        df.loc[
            (df['Price_Change'] > 0) |
            (df['Цена_на_последна_трансакција'] < df['Цена_на_последна_трансакција'].mean() * 0.95),
            'Signal'
        ] = 'Buy'

        df.loc[
            (df['Price_Change'] < 0) |
            (df['Цена_на_последна_трансакција'] > df['Цена_на_последна_трансакција'].mean() * 1.05),
            'Signal'
        ] = 'Sell'
        print("Fallback signals applied for small dataset.")
    else:

        df['Signal'] = 'Hold'
        df.loc[
            (df['RSI'] < 40) & (df['Цена_на_последна_трансакција'] > df['SMA10']) & (df['SMA10'].notnull()),
            'Signal'
        ] = 'Buy'
        df.loc[
            (df['RSI'] > 60) & (df['Цена_на_последна_трансакција'] < df['SMA10']) & (df['SMA10'].notnull()),
            'Signal'
        ] = 'Sell'

    buy_signals_count = df[df['Signal'] == 'Buy'].shape[0]
    sell_signals_count = df[df['Signal'] == 'Sell'].shape[0]
    print(f"Buy signals: {buy_signals_count}, Sell signals: {sell_signals_count}")

    df['InsufficientData'] = False
    return df

@app.route('/issuer/<issuer_code>/graph')
def issuer_graph(issuer_code):
    conn = sqlite3.connect('stock_data.db')
    query = """
        SELECT Датум, Цена_на_последна_трансакција, Мак_, Мин_ 
        FROM stock_data 
        WHERE Код_на_издавач = ? 
        ORDER BY Датум ASC
    """
    df = pd.read_sql_query(query, conn, params=(issuer_code,))
    conn.close()

    df = calculate_technical_indicators(df)

    if 'InsufficientData' in df and df['InsufficientData'].iloc[0]:
        return f"<h3>Insufficient data for issuer {issuer_code}. Please upload more data to perform technical analysis.</h3>"

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Датум'], y=df['Цена_на_последна_трансакција'], name='Price'))
    fig.add_trace(go.Scatter(x=df['Датум'], y=df['SMA10'], name='SMA10'))
    fig.add_trace(go.Scatter(x=df['Датум'], y=df['SMA50'], name='SMA50'))
    fig.add_trace(go.Scatter(x=df['Датум'], y=df['EMA10'], name='EMA10'))
    fig.add_trace(go.Scatter(x=df['Датум'], y=df['EMA50'], name='EMA50'))

    buy_signals = df[df['Signal'] == 'Buy']
    sell_signals = df[df['Signal'] == 'Sell']

    fig.add_trace(go.Scatter(
        x=buy_signals['Датум'], y=buy_signals['Цена_на_последна_трансакција'],
        mode='markers', name='Buy Signal', marker=dict(color='green', size=10)
    ))
    fig.add_trace(go.Scatter(
        x=sell_signals['Датум'], y=sell_signals['Цена_на_последна_трансакција'],
        mode='markers', name='Sell Signal', marker=dict(color='red', size=10)
    ))

    fig.update_layout(
        title=f'Technical Analysis for {issuer_code}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
    )

    return fig.to_html(full_html=False)

def fetch_data(issuer_code):
    conn = sqlite3.connect('stock_data.db')
    query = "SELECT Датум, Цена_на_последна_трансакција FROM stock_data WHERE Код_на_издавач = ? ORDER BY Датум"
    df = pd.read_sql_query(query, conn, params=(issuer_code,))
    conn.close()

    df['Датум'] = pd.to_datetime(df['Датум'])

    df.set_index('Датум', inplace=True)

    df['Цена_на_последна_трансакција'] = df['Цена_на_последна_трансакција'] \
        .str.replace('.', '', regex=False) \
        .str.replace(',', '.', regex=False) \
        .astype(float)

    df = df.resample('W').mean()  

    df.dropna(inplace=True)

    print(f"Number of rows after aggregation: {len(df)}")
    return df

def train_lstm(df):

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)

    train_size = int(len(scaled_data) * 0.7)
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:]

    def create_sequences(data, sequence_length=50):
        if len(data) <= sequence_length:  
            return np.empty((0, sequence_length, 1)), np.empty((0,))

        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        X = np.array(X)
        y = np.array(y)

        X = X.reshape((X.shape[0], X.shape[1], 1)) 

        return X, y

    sequence_length = 50
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_val, y_val = create_sequences(val_data, sequence_length)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_val: {X_val.shape}")
    print(f"Shape of y_val: {y_val.shape}")

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=50, callbacks=[early_stop])
    else:
        model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=[early_stop])

    return model, scaler, sequence_length

@app.route('/issuer/<issuer_code>/predict', methods=['GET'])
def predict_and_display(issuer_code):

    df = fetch_data(issuer_code)
    if len(df) < 100:
        return f"<h3>Not enough data to train the model for {issuer_code}. Please add more historical data.</h3>"

    model, scaler, sequence_length = train_lstm(df)

    scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))  
    X_test, y_test = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_test.append(scaled_data[i - sequence_length:i])
        y_test.append(scaled_data[i])
    X_test, y_test = np.array(X_test), np.array(y_test)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  
    y_test = scaler.inverse_transform(y_test)  

    prediction_dates = df.index[-len(predictions):]
    y_test = y_test[-len(predictions):]

    downsample_rate = 1
    prediction_dates = prediction_dates[::downsample_rate]
    y_test = y_test[::downsample_rate]
    predictions = predictions[::downsample_rate]

    print(f"Prediction dates: {len(prediction_dates)}")
    print(f"Predicted values: {predictions.shape}")
    print(f"Actual values: {y_test.shape}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prediction_dates,
        y=y_test.flatten(),
        mode='lines',
        name='Actual Prices',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=prediction_dates,
        y=predictions.flatten(),
        mode='lines',
        name='Predicted Prices',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f'Stock Price Prediction for {issuer_code}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
    )

    graph_html = fig.to_html(full_html=False)

    return render_template(
        'issuer.html',
        issuer_code=issuer_code,
        predicted_price=predictions[-1, 0],
        graph_html=graph_html
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)
