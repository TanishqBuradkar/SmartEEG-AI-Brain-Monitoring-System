import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from collections import deque, Counter
import joblib

# ======================================
# LOAD TRAINED MODEL
# ======================================

model = joblib.load("brain_model.pkl")

# ======================================
# SERIAL CONNECTION
# ======================================

ser = serial.Serial('COM3', 115200)

# ======================================
# VARIABLES
# ======================================

signal_data = deque([0]*300, maxlen=300)

samples = []
predictions = []

labels = {
    0: "Relaxation",
    1: "Focus",
    2: "Drowsiness"
}

current_state = "Analyzing..."

# ======================================
# MATPLOTLIB SETUP
# ======================================

plt.ion()

fig, ax = plt.subplots(figsize=(12,6))

line, = ax.plot(signal_data)

ax.set_title("Real-Time EEG Signal Analysis")

ax.set_xlabel("Samples")
ax.set_ylabel("EEG Amplitude")

# ======================================
# MAIN LOOP
# ======================================

while True:

    try:

        raw = ser.readline().decode(
            'utf-8',
            errors='ignore'
        ).strip()

        if raw.isdigit():

            value = int(raw)

            # Store live signal
            signal_data.append(value)

            # Store for AI analysis
            samples.append(value)

            # ==================================
            # UPDATE LIVE GRAPH
            # ==================================

            line.set_ydata(signal_data)
            line.set_xdata(range(len(signal_data)))

            ax.relim()
            ax.autoscale_view()

            # ==================================
            # AI ANALYSIS
            # ==================================

            if len(samples) >= 1024:

                eeg = np.array(samples)

                # Signal validation
                if np.std(eeg) < 5:

                    current_state = "No EEG Signal"

                    samples = []

                    continue

                # FFT
                yf = np.abs(fft(eeg))

                # Match feature size
                features = yf[:988]

                X = pd.DataFrame([features])

                # Prediction
                prediction = model.predict(X)[0]

                predictions.append(prediction)

                # Stable prediction
                if len(predictions) >= 5:

                    stable_prediction = Counter(
                        predictions
                    ).most_common(1)[0][0]

                    current_state = labels[
                        stable_prediction
                    ]

                    predictions = []

                samples = []

            # ==================================
            # DISPLAY TEXT
            # ==================================

            ax.set_title(
                f"Real-Time EEG Analysis | "
                f"Mental State: {current_state}"
            )

            plt.draw()
            plt.pause(0.01)

    except Exception as e:

        print("Error:", e)