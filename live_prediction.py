import pandas as pd
import joblib
import time
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("mental-state.csv")

# Features
X = data.drop("Label", axis=1)

# Labels
y = data["Label"]

# Encode labels
encoder = LabelEncoder()
encoder.fit(y)

# Load trained model
model = joblib.load("brain_model.pkl")

print("\n===================================")
print(" LIVE EEG BRAIN HEALTH MONITOR ")
print("===================================\n")

while True:

    # Random EEG sample
    sample = X.sample(1)

    # Predict
    pred = model.predict(sample)

    # Convert label
    state = encoder.inverse_transform(pred)[0]
    state = str(state).lower()

    # Default outputs
    alert = "No Danger Detected"

    # Mental State Logic
    if "focus" in state:

        mental_state = "Focused"
        health = "Brain Activity Normal"

    elif "relax" in state:

        mental_state = "Relaxed"
        health = "Brain Calm and Stable"

    elif "drows" in state or "sleep" in state:

        mental_state = "Drowsiness"
        health = "Low Attention Detected"

        alert = "⚠ ALERT: User appears drowsy"

    else:

        mental_state = state
        health = "Brain Activity Stable"

    # Additional anomaly check
    if sample.mean(axis=1).values[0] > 1000:

        alert = "🚨 ALERT: Abnormal Brain Activity Detected"

    # Display
    print("Mental State :", mental_state)
    print("Brain Health :", health)
    print("Alert Status :", alert)
    print("--------------------------------------")

    time.sleep(2)