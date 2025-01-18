# controllers/main_controller.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from flask import Blueprint, render_template, request, jsonify
from plotly.subplots import make_subplots

from models.stock_model import (
    get_stock_data,
    get_all_stock_data,
    get_total_issuers_count,
    get_filtered_data_for_analysis,
    get_issuer_details,
    get_issuer_data_for_graph,
    fetch_data
)

main_blueprint = Blueprint('main_blueprint', __name__)


@main_blueprint.route('/')
def home():
    page = request.args.get('page', default=1, type=int)
    limit = 10
    total_issuers = get_total_issuers_count()
    total_pages = (total_issuers + limit - 1) // limit
    stock_data_page = get_stock_data(page=page, limit=limit)
    return render_template('index.html', stock_data=stock_data_page, page=page, total_pages=total_pages)


@main_blueprint.route('/analysis')
def analysis():
    issuer = request.args.get('issuer', default='', type=str).strip()
    page = request.args.get('page', default=1, type=int)
    limit = 10
    stock_data, total_rows, total_pages = get_filtered_data_for_analysis(issuer=issuer, page=page, limit=limit)
    return render_template(
        'analysis.html',
        stock_data=stock_data,
        page=page,
        total_pages=total_pages,
        issuer=issuer,
        max=max,
        min=min
    )


@main_blueprint.route('/issuer/<issuer_code>')
def issuer_details(issuer_code):
    stock_data = get_issuer_details(issuer_code)
    return render_template('issuer.html', issuer_code=issuer_code, stock_data=stock_data)


@main_blueprint.route('/issuer/<issuer_code>/graph')
def issuer_graph(issuer_code):
    df = get_issuer_data_for_graph(issuer_code)

    # Determine which strategy to use
    chosen_strategy = request.args.get('strategy', 'full').lower()

    # Prepare data for the microservice
    data_payload = {
        'issuer_data': df.reset_index().to_dict(orient='records'),
        'strategy': chosen_strategy
    }

    try:
        # Call the strategy microservice
        response = requests.post(
            'http://strategy_service:5003/analyze',  # Ensure this matches the microservice's endpoint
            json=data_payload
        )
        response.raise_for_status()
        analyzed_data = response.json()

        # Convert back to DataFrame
        df = pd.DataFrame(analyzed_data)

        # Convert `Датум` back to datetime
        df['Датум'] = pd.to_datetime(df['Датум'])

        if 'InsufficientData' in df and df['InsufficientData'].iloc[0]:
            return f"<h3>Insufficient data for issuer {issuer_code}. Please upload more data to perform technical strategies.</h3>"

        # Create subplots: 4 rows, 1 column
        #   Row 1: Price + MAs, Buy/Sell signals
        #   Row 2: RSI
        #   Row 3: MACD
        #   Row 4: ADX & CCI
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,  # all subplots share the same x-axis
            vertical_spacing=0.02,
            row_heights=[0.4, 0.15, 0.15, 0.3]
        )

        # --- Row 1: Price & MAs ---
        fig.add_trace(
            go.Scatter(
                x=df['Датум'],
                y=df['Цена_на_последна_трансакција'],
                name='Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Plot SMA/EMA lines if they exist in df
        if 'SMA10' in df:
            fig.add_trace(
                go.Scatter(
                    x=df['Датум'],
                    y=df['SMA10'],
                    name='SMA10',
                    line=dict(color='orange')
                ),
                row=1, col=1
            )
        if 'SMA50' in df:
            fig.add_trace(
                go.Scatter(
                    x=df['Датум'],
                    y=df['SMA50'],
                    name='SMA50',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
        if 'EMA10' in df:
            fig.add_trace(
                go.Scatter(
                    x=df['Датум'],
                    y=df['EMA10'],
                    name='EMA10',
                    line=dict(color='purple')
                ),
                row=1, col=1
            )
        if 'EMA50' in df:
            fig.add_trace(
                go.Scatter(
                    x=df['Датум'],
                    y=df['EMA50'],
                    name='EMA50',
                    line=dict(color='green')
                ),
                row=1, col=1
        )

        # Buy/Sell signals, if present
        if 'Signal' in df:
            buy_signals = df[df['Signal'] == 'Buy']
            sell_signals = df[df['Signal'] == 'Sell']
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['Датум'],
                    y=buy_signals['Цена_на_последна_трансакција'],
                    mode='markers',
                    marker=dict(color='green', size=10),
                    name='Buy Signal'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['Датум'],
                    y=sell_signals['Цена_на_последна_трансакција'],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Sell Signal'
                ),
                row=1, col=1
            )

        # --- Row 2: RSI ---
        if 'RSI' in df:
            fig.add_trace(
                go.Scatter(
                    x=df['Датум'],
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color='magenta')
                ),
                row=2, col=1
            )

        # --- Row 3: MACD ---
        if 'MACD' in df:
            fig.add_trace(
                go.Scatter(
                    x=df['Датум'],
                    y=df['MACD'],
                    name='MACD',
                    line=dict(color='black')
                ),
                row=3, col=1
            )

        # --- Row 4: ADX + CCI on the same row (you could separate them if you prefer) ---
        if 'ADX' in df:
            fig.add_trace(
                go.Scatter(
                    x=df['Датум'],
                    y=df['ADX'],
                    name='ADX',
                    line=dict(color='teal')
                ),
                row=4, col=1
            )
        if 'CCI' in df:
            fig.add_trace(
                go.Scatter(
                    x=df['Датум'],
                    y=df['CCI'],
                    name='CCI',
                    line=dict(color='gold')
                ),
                row=4, col=1
            )

        # Layout
        fig.update_layout(
            title=f"Technical Analysis for {issuer_code} ({chosen_strategy} strategy)",
            template="plotly_white",
            height=900,
            xaxis=dict(
                title="Date",
                tickangle=-45,
                tickformat="%b %d, %Y",
                showgrid=True
            )
        )

        # You can label Y-axes for each row if desired:
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="ADX & CCI", row=4, col=1)

        # Return minimal HTML so it can be embedded in iframe
        return fig.to_html(full_html=False)

    except requests.RequestException as e:
        return f"<h3>Error communicating with the strategy service: {e}</h3>"

    except Exception as e:
        return f"<h3>An unexpected error occurred: {e}</h3>"


@main_blueprint.route('/issuer/<issuer_code>/predict', methods=['GET'])
def predict_and_display(issuer_code):
    # Fetch issuer data
    df = fetch_data(issuer_code)
    if len(df) < 100:
        return f"<h3>Not enough data to train the model for {issuer_code}. Please add more historical data.</h3>"

    # Convert the DataFrame to a format suitable for the API call
    data_payload = {
        'issuer_data': df.reset_index().assign(
            Датум=lambda x: x['Датум'].dt.strftime('%Y-%m-%d')
        ).to_dict(orient='records')  # Convert Timestamps to strings
    }

    try:
        # Call the prediction microservice
        response = requests.post('http://prediction_service:5002/predict', json=data_payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the microservice response
        response_data = response.json()

        if 'predictions' not in response_data or 'dates' not in response_data:
            return f"<h3>Invalid response from prediction service for {issuer_code}.</h3>"

        predictions = response_data['predictions']
        actual_prices = response_data.get('actual_prices', [])
        prediction_dates = response_data['dates']

        # Plot with Plotly
        fig = go.Figure()
        if actual_prices:
            fig.add_trace(go.Scatter(
                x=prediction_dates,
                y=actual_prices,
                mode='lines',
                name='Actual Prices',
                line=dict(color='blue')
            ))
        fig.add_trace(go.Scatter(
            x=prediction_dates,
            y=predictions,
            mode='lines',
            name='Predicted Prices',
            line=dict(color='red')
        ))
        fig.update_layout(
            title=f'Stock Price Prediction for {issuer_code}',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            height=600
        )

        return render_template(
            'issuer.html',
            issuer_code=issuer_code,
            predicted_price=predictions[-1],
            graph_html=fig.to_html(full_html=False)
        )

    except requests.RequestException as e:
        return f"<h3>Error communicating with the prediction service: {e}</h3>"
    except Exception as e:
        return f"<h3>An unexpected error occurred: {e}</h3>"
