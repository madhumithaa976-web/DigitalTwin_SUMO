# lstm_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_lstm_model(timesteps, features):
    """
    Build Improved LSTM model for Traffic Prediction

    Parameters:
        timesteps (int): Number of past time steps (e.g., 10)
        features (int): Number of input features 
                        (Vehicle_Count, Average_Speed, Waiting_Time = 3)

    Returns:
        Compiled Keras LSTM model
    """

    model = Sequential()

    # Explicit Input Layer (Best Practice)
    model.add(Input(shape=(timesteps, features)))

    # First LSTM Layer
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    # Second LSTM Layer
    model.add(LSTM(32))
    model.add(Dropout(0.2))

    # Dense Layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(features))  # Output: predict 3 values

    # Compile Model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model
