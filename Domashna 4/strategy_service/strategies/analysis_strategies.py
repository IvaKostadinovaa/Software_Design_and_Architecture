# services/analysis_strategies.py

from abc import ABC, abstractmethod
import pandas as pd
import ta


class AnalysisStrategy(ABC):
    """
    Abstract base class for different technical strategies strategies.
    Each strategy must implement 'perform_analysis(df)' which:
      1) Cleans the data (numeric conversions, etc.).
      2) Calculates any needed indicators.
      3) Generates 'Signal' buy/sell/hold columns.
      4) Adds a 'InsufficientData' flag if data is too small.
    Returns the modified DataFrame.
    """

    @abstractmethod
    def perform_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class RSIOnlyStrategy(AnalysisStrategy):
    """
    Strategy focusing solely on RSI to generate Buy/Sell signals.
    """

    def perform_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean numeric columns
        df['Цена_на_последна_трансакција'] = df['Цена_на_последна_трансакција'].str.replace(',', '').astype(float)
        df['Мак_'] = df['Мак_'].str.replace(',', '').astype(float)
        df['Мин_'] = df['Мин_'].str.replace(',', '').astype(float)

        df['Датум'] = pd.to_datetime(df['Датум'], errors='coerce')
        df = df.dropna(subset=['Цена_на_последна_трансакција', 'Мак_', 'Мин_', 'Датум'])
        df = df.sort_values('Датум')

        if len(df) < 3:
            df['InsufficientData'] = True
            return df

        # Calculate RSI
        window = min(14, len(df))
        df['RSI'] = ta.momentum.RSIIndicator(df['Цена_на_последна_трансакција'], window=window).rsi()

        # Generate signals
        if len(df) < 14:
            df['Signal'] = 'Hold'
            df['Price_Change'] = df['Цена_на_последна_трансакција'].diff()
            df.loc[df['Price_Change'] > 0, 'Signal'] = 'Buy'
            df.loc[df['Price_Change'] < 0, 'Signal'] = 'Sell'
        else:
            df['Signal'] = 'Hold'
            df.loc[df['RSI'] < 40, 'Signal'] = 'Buy'
            df.loc[df['RSI'] > 60, 'Signal'] = 'Sell'

        df['InsufficientData'] = False
        return df


class MacdOnlyStrategy(AnalysisStrategy):
    """
    Strategy focusing on MACD for Buy/Sell signals.
    """

    def perform_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean numeric columns
        df['Цена_на_последна_трансакција'] = df['Цена_на_последна_трансакција'].str.replace(',', '').astype(float)
        df['Мак_'] = df['Мак_'].str.replace(',', '').astype(float)
        df['Мин_'] = df['Мин_'].str.replace(',', '').astype(float)

        df['Датум'] = pd.to_datetime(df['Датум'], errors='coerce')
        df = df.dropna(subset=['Цена_на_последна_трансакција', 'Мак_', 'Мин_', 'Датум'])
        df = df.sort_values('Датум')

        if len(df) < 3:
            df['InsufficientData'] = True
            return df

        # Calculate MACD
        df['MACD'] = ta.trend.MACD(df['Цена_на_последна_трансакција']).macd()

        if len(df) < 14:
            # fallback for small dataset
            df['Signal'] = 'Hold'
            df['Price_Change'] = df['Цена_на_последна_трансакција'].diff()
            df.loc[df['Price_Change'] > 0, 'Signal'] = 'Buy'
            df.loc[df['Price_Change'] < 0, 'Signal'] = 'Sell'
        else:
            # Example logic: If MACD > 0 => 'Buy', else => 'Sell'
            df['Signal'] = 'Sell'
            df.loc[df['MACD'] > 0, 'Signal'] = 'Buy'

        df['InsufficientData'] = False
        return df


class AdxOnlyStrategy(AnalysisStrategy):
    """
    Strategy focusing on ADX (Average Directional Index).
    Generally, ADX is used to measure trend strength, not directly Buy/Sell.
    But we'll create a simple example approach:
      - If ADX > 25, we do 'Buy' (assuming a strong uptrend), else 'Sell'.
    """

    def perform_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean numeric columns
        df['Цена_на_последна_трансакција'] = df['Цена_на_последна_трансакција'].str.replace(',', '').astype(float)
        df['Мак_'] = df['Мак_'].str.replace(',', '').astype(float)
        df['Мин_'] = df['Мин_'].str.replace(',', '').astype(float)

        df['Датум'] = pd.to_datetime(df['Датум'], errors='coerce')
        df = df.dropna(subset=['Цена_на_последна_трансакција', 'Мак_', 'Мин_', 'Датум'])
        df = df.sort_values('Датум')

        if len(df) < 3:
            df['InsufficientData'] = True
            return df

        row_count = len(df)
        if row_count >= 14:
            df['ADX'] = ta.trend.ADXIndicator(
                high=df['Мак_'],
                low=df['Мин_'],
                close=df['Цена_на_последна_трансакција'],
                window=14
            ).adx()
        else:
            df['ADX'] = None

        if row_count < 14:
            df['Signal'] = 'Hold'
            df['Price_Change'] = df['Цена_на_последна_трансакција'].diff()
            df.loc[df['Price_Change'] > 0, 'Signal'] = 'Buy'
            df.loc[df['Price_Change'] < 0, 'Signal'] = 'Sell'
        else:
            df['Signal'] = 'Sell'
            df.loc[df['ADX'] > 25, 'Signal'] = 'Buy'

        df['InsufficientData'] = False
        return df


class CciOnlyStrategy(AnalysisStrategy):
    """
    Strategy focusing on CCI (Commodity Channel Index).
    We'll do a simple approach: If CCI < -100 => Buy, If CCI > 100 => Sell, else Hold.
    """

    def perform_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean numeric columns
        df['Цена_на_последна_трансакција'] = df['Цена_на_последна_трансакција'].str.replace(',', '').astype(float)
        df['Мак_'] = df['Мак_'].str.replace(',', '').astype(float)
        df['Мин_'] = df['Мин_'].str.replace(',', '').astype(float)

        df['Датум'] = pd.to_datetime(df['Датум'], errors='coerce')
        df = df.dropna(subset=['Цена_на_последна_трансакција', 'Мак_', 'Мин_', 'Датум'])
        df = df.sort_values('Датум')

        if len(df) < 3:
            df['InsufficientData'] = True
            return df

        window = min(20, len(df))
        df['CCI'] = ta.trend.CCIIndicator(
            high=df['Мак_'],
            low=df['Мин_'],
            close=df['Цена_на_последна_трансакција'],
            window=window
        ).cci()

        if len(df) < 20:
            # fallback signals
            df['Signal'] = 'Hold'
            df['Price_Change'] = df['Цена_на_последна_трансакција'].diff()
            df.loc[df['Price_Change'] > 0, 'Signal'] = 'Buy'
            df.loc[df['Price_Change'] < 0, 'Signal'] = 'Sell'
        else:
            df['Signal'] = 'Hold'
            df.loc[df['CCI'] < -100, 'Signal'] = 'Buy'
            df.loc[df['CCI'] > 100, 'Signal'] = 'Sell'

        df['InsufficientData'] = False
        return df


class FullIndicatorStrategy(AnalysisStrategy):
    """
    Calculates SMA/EMA, RSI, MACD, ADX, CCI.
    Falls back to 'InsufficientData' if anything goes wrong
    or there's not enough data for an indicator (like ADX).
    """

    def perform_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean numeric columns
        df['Цена_на_последна_трансакција'] = df['Цена_на_последна_трансакција'].str.replace(',', '').astype(float)
        df['Мак_'] = df['Мак_'].str.replace(',', '').astype(float)
        df['Мин_'] = df['Мин_'].str.replace(',', '').astype(float)

        df['Датум'] = pd.to_datetime(df['Датум'], errors='coerce')
        df = df.dropna(subset=['Цена_на_последна_трансакција', 'Мак_', 'Мин_', 'Датум'])
        df = df.sort_values('Датум')

        if len(df) < 3:
            # Mark insufficient if fewer than 3 rows
            df['InsufficientData'] = True
            return df

        row_count = len(df)

        # Moving Averages
        df['SMA10'] = df['Цена_на_последна_трансакција'].rolling(window=min(10, row_count)).mean()
        df['SMA50'] = df['Цена_на_последна_трансакција'].rolling(window=min(50, row_count)).mean()
        df['EMA10'] = df['Цена_на_последна_трансакција'].ewm(span=min(10, row_count), adjust=False).mean()
        df['EMA50'] = df['Цена_на_последна_трансакција'].ewm(span=min(50, row_count), adjust=False).mean()

        # RSI
        try:
            df['RSI'] = ta.momentum.RSIIndicator(
                df['Цена_на_последна_трансакција'], window=min(14, row_count)
            ).rsi()
        except (ValueError, IndexError) as e:
            df['InsufficientData'] = True
            return df

        # MACD
        try:
            df['MACD'] = ta.trend.MACD(df['Цена_на_последна_трансакција']).macd()
        except (ValueError, IndexError) as e:
            df['InsufficientData'] = True
            return df

        # CCI
        try:
            df['CCI'] = ta.trend.CCIIndicator(
                high=df['Мак_'],
                low=df['Мин_'],
                close=df['Цена_на_последна_трансакција'],
                window=min(20, row_count)
            ).cci()
        except (ValueError, IndexError) as e:
            df['InsufficientData'] = True
            return df

        # ADX: needs at least 14 rows
        if row_count >= 14:
            try:
                df['ADX'] = ta.trend.ADXIndicator(
                    high=df['Мак_'],
                    low=df['Мин_'],
                    close=df['Цена_на_последна_трансакција'],
                    window=14
                ).adx()
            except (ValueError, IndexError) as e:
                # If ADX can't be calculated, mark insufficient
                df['InsufficientData'] = True
                return df
        else:
            # Not enough rows for ADX
            df['ADX'] = None

        # Generate signals
        if row_count < 14:
            # fallback signals
            df['Signal'] = 'Hold'
            df['Price_Change'] = df['Цена_на_последна_трансакција'].diff()
            df.loc[df['Price_Change'] > 0, 'Signal'] = 'Buy'
            df.loc[df['Price_Change'] < 0, 'Signal'] = 'Sell'
        else:
            df['Signal'] = 'Hold'
            # Example logic combining RSI + SMA10
            df.loc[
                (df['RSI'] < 40) & (df['Цена_на_последна_трансакција'] > df['SMA10']),
                'Signal'
            ] = 'Buy'
            df.loc[
                (df['RSI'] > 60) & (df['Цена_на_последна_трансакција'] < df['SMA10']),
                'Signal'
            ] = 'Sell'

        df['InsufficientData'] = False
        return df