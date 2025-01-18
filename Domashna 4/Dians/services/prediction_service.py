# services/prediction_service.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def train_lstm(df):
    """
    Train an LSTM model on the provided DataFrame (weekly-resampled),
    exactly replicating the logic from the original code.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by date with a single 'Цена_на_последна_трансакција' column.

    Returns
    -------
    model : keras.Model
        Trained LSTM model
    scaler : MinMaxScaler
        Fitted scaler used for normalizing data
    sequence_length : int
        The lookback window used for training
    """
    # Convert your single column to a numpy array
    values = df.values  # shape (num_samples, 1)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)

    # Split data: 70% training, 30% validation
    train_size = int(len(scaled_data) * 0.7)
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:]

    # Helper function to create sequences
    def create_sequences(data, sequence_length=50):
        if len(data) <= sequence_length:  # Ensure enough data
            return np.empty((0, sequence_length, 1)), np.empty((0,))

        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        X = np.array(X)
        y = np.array(y)
        # Reshape X to (batch_size, time_steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    sequence_length = 50  # Lookback window
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_val, y_val = create_sequences(val_data, sequence_length)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_val: {X_val.shape}")
    print(f"Shape of y_val: {y_val.shape}")

    # Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    if X_val.size > 0 and y_val.size > 0:  # If we have validation data
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=32,
            epochs=50,
            callbacks=[early_stop]
        )
    else:
        # No validation data, train without it
        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            callbacks=[early_stop]
        )

    return model, scaler, sequence_length
