import numpy as np
from flask import Flask, request, jsonify
from prediction.model import train_lstm
import pandas as pd
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        data = request.json
        issuer_data = data['issuer_data']

        # Convert to DataFrame
        df = pd.DataFrame(issuer_data)
        df['Датум'] = pd.to_datetime(df['Датум'])
        df.set_index('Датум', inplace=True)

        # Train the LSTM model
        model, scaler, sequence_length = train_lstm(df[['Цена_на_последна_трансакција']])

        # Prepare test data
        scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))
        X_test, y_test = [], []
        for i in range(sequence_length, len(scaled_data)):
            X_test.append(scaled_data[i - sequence_length:i])
            y_test.append(scaled_data[i])
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Predict
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions).flatten()
        actual_prices = scaler.inverse_transform(y_test).flatten()

        # Dates for predictions
        prediction_dates = df.index[-len(predictions):].strftime('%Y-%m-%d').tolist()

        # Return the response
        return jsonify({
            "predictions": predictions.tolist(),
            "actual_prices": actual_prices.tolist(),
            "dates": prediction_dates
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002)