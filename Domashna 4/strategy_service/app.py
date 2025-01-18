from flask import Flask, request, jsonify
from strategies.analysis_strategies import (
    RSIOnlyStrategy,
    MacdOnlyStrategy,
    AdxOnlyStrategy,
    CciOnlyStrategy,
    FullIndicatorStrategy
)
import pandas as pd

app = Flask(__name__)

# Strategy Mapping
STRATEGIES = {
    "rsi": RSIOnlyStrategy,
    "macd": MacdOnlyStrategy,
    "adx": AdxOnlyStrategy,
    "cci": CciOnlyStrategy,
    "full": FullIndicatorStrategy
}

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Parse input data
        data = request.json
        if 'issuer_data' not in data or 'strategy' not in data:
            return jsonify({'error': 'Missing issuer_data or strategy parameter'}), 400

        # Load data into a DataFrame
        df = pd.DataFrame(data['issuer_data'])
        df['Датум'] = pd.to_datetime(df['Датум'])
        df = df.sort_values('Датум')

        # Get the strategy
        strategy_name = data['strategy'].lower()
        strategy_class = STRATEGIES.get(strategy_name)

        if not strategy_class:
            return jsonify({'error': f"Strategy '{strategy_name}' not supported"}), 400

        # Perform analysis
        strategy = strategy_class()
        result_df = strategy.perform_analysis(df)

        # Convert `Датум` to string for JSON compatibility
        result_df['Датум'] = result_df['Датум'].dt.strftime('%Y-%m-%d')

        # Convert DataFrame to JSON and return
        return jsonify(result_df.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003)
