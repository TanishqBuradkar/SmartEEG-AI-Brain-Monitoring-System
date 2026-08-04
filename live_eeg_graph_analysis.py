import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft
from collections import deque, Counter
import joblib

# ===============================
# LOAD MODEL
# ===============================

model = joblib.load("brain_model.pkl")

# ===============================
# SERIAL CONNECTION
# ===============================

ser = serial.Serial('COM3', 115200)

# ===============================
# VARIABLES
# ===============================

graph_data = deque([0]*200, maxlen=200)

samples = []
predictions = []

mental_state = "Analyzing..."

labels = {
    0: "Relaxation",
    1: "Focus",
    2: "Drowsiness"
}

# ===============================
# GRAPH SETUP
# ===============================

fig, ax = plt.subplots(figsize=(12,6))

line, = ax.plot(graph_data)

ax.set_ylim(0, 4095)

ax.set_title("Real-Time EEG Signal")

ax.set_xlabel("Samples")

ax.set_ylabel("Amplitude")

# ===============================
# UPDATE FUNCTION
# ===============================

def update(frame):

    global mental_state
    global samples
    global predictions

    try:

        raw = ser.readline().decode(
            'utf-8',
            errors='ignore'
        ).strip()

        if raw.isdigit():

            value = int(raw)

            # Add live data
            graph_data.append(value)

            samples.append(value)

            # =======================
            # AI ANALYSIS
            # =======================

            if len(samples) >= 256:

                eeg = np.array(samples)

                # Signal validation
                if np.std(eeg) > 5:

                    # FFT
                    yf = np.abs(fft(eeg))

                    # Match feature count
                    features = yf[:988]

                    X = pd.DataFrame([features])

                    # Prediction
                    pred = model.predict(X)[0]

                    predictions.append(pred)

                    # Stable prediction
                    if len(predictions) >= 5:

                        stable_pred = Counter(
                            predictions
                        ).most_common(1)[0][0]

                        mental_state = labels[
                            stable_pred
                        ]

                        predictions = []

                else:

                    mental_state = "No EEG Signal"

                samples = []

            # =======================
            # UPDATE GRAPH
            # =======================

            line.set_ydata(graph_data)

            ax.set_title(
                f"Live EEG Signal | Mental State: {mental_state}"
            )

    except Exception as e:

        print(e)

    return line,

# ===============================
# ANIMATION
# ===============================

ani = FuncAnimation(
    fig,
    update,
    interval=10,
    blit=False
)

plt.show()