# services/analysis_service.py

import pandas as pd
import ta

def calculate_technical_indicators(df):
    """
    Calculate 5 oscillators and 5 moving averages and generate buy/sell signals,
    mirroring the logic from the original code.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns like 'Цена_на_последна_трансакција',
        'Мак_', 'Мин_', and 'Датум'.

    Returns
    -------
    df : pd.DataFrame
        The same DataFrame with added columns for indicators and 'Signal'.
        If there is insufficient data (< 3 rows), the 'InsufficientData' column is set to True.
    """
    # Remove commas from numeric columns and convert to float
    df['Цена_на_последна_трансакција'] = (
        df['Цена_на_последна_трансакција']
        .str.replace(',', '')
        .astype(float)
    )
    df['Мак_'] = df['Мак_'].str.replace(',', '').astype(float)
    df['Мин_'] = df['Мин_'].str.replace(',', '').astype(float)

    # Convert 'Датум' to datetime
    df['Датум'] = pd.to_datetime(df['Датум'], errors='coerce')

    # Drop rows with missing values
    df = df.dropna(subset=['Цена_на_последна_трансакција', 'Мак_', 'Мин_', 'Датум'])

    # Ensure the data is sorted by date
    df = df.sort_values('Датум')

    # Check the number of rows
    row_count = len(df)
    print(f"Number of rows available after cleaning: {row_count}")

    # If fewer than 3 rows, cannot do analysis
    if row_count < 3:
        df['InsufficientData'] = True
        return df

    # Calculate moving averages
    df['SMA10'] = df['Цена_на_последна_трансакција'].rolling(window=min(10, row_count)).mean()
    df['SMA50'] = df['Цена_на_последна_трансакција'].rolling(window=min(50, row_count)).mean()
    df['EMA10'] = df['Цена_на_последна_трансакција'].ewm(span=min(10, row_count), adjust=False).mean()
    df['EMA50'] = df['Цена_на_последна_трансакција'].ewm(span=min(50, row_count), adjust=False).mean()

    # Calculate oscillators
    df['RSI'] = ta.momentum.RSIIndicator(
        df['Цена_на_последна_трансакција'],
        window=min(14, row_count)
    ).rsi()
    df['MACD'] = ta.trend.MACD(
        df['Цена_на_последна_трансакција']
    ).macd()
    df['CCI'] = ta.trend.CCIIndicator(
        high=df['Мак_'],
        low=df['Мин_'],
        close=df['Цена_на_последна_трансакција'],
        window=min(20, row_count)
    ).cci()

    # Handle ADX Calculation only if row_count >= 14
    if row_count >= 14:
        try:
            df['ADX'] = ta.trend.ADXIndicator(
                high=df['Мак_'],
                low=df['Мин_'],
                close=df['Цена_на_последна_трансакција'],
                window=14
            ).adx()
        except IndexError:
            print("IndexError during ADX calculation: Not enough data for ADX.")
            df['ADX'] = None
    else:
        df['ADX'] = None

    # Generate signals
    if row_count < 14:
        # Fallback approach for small datasets
        df['Signal'] = 'Hold'
        df['Price_Change'] = df['Цена_на_последна_трансакција'].diff()

        # Buy Signal
        df.loc[
            (df['Price_Change'] > 0) |
            (df['Цена_на_последна_трансакција'] < df['Цена_на_последна_трансакција'].mean() * 0.95),
            'Signal'
        ] = 'Buy'

        # Sell Signal
        df.loc[
            (df['Price_Change'] < 0) |
            (df['Цена_на_последна_трансакција'] > df['Цена_на_последна_трансакција'].mean() * 1.05),
            'Signal'
        ] = 'Sell'
        print("Fallback signals applied for small dataset.")
    else:
        # Standard logic
        df['Signal'] = 'Hold'
        df.loc[
            (df['RSI'] < 40) &
            (df['Цена_на_последна_трансакција'] > df['SMA10']) &
            (df['SMA10'].notnull()),
            'Signal'
        ] = 'Buy'
        df.loc[
            (df['RSI'] > 60) &
            (df['Цена_на_последна_трансакција'] < df['SMA10']) &
            (df['SMA10'].notnull()),
            'Signal'
        ] = 'Sell'

    # Count signals for debugging
    buy_signals_count = df[df['Signal'] == 'Buy'].shape[0]
    sell_signals_count = df[df['Signal'] == 'Sell'].shape[0]
    print(f"Buy signals: {buy_signals_count}, Sell signals: {sell_signals_count}")

    df['InsufficientData'] = False
    return df
