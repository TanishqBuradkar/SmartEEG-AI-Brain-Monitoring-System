import pandas as pd

# Load dataset
data = pd.read_csv("mental-state.csv")

# Select frequency columns
freq_cols = [col for col in data.columns if "freq_" in col]

# Create empty lists
alpha_cols = []
beta_cols = []
theta_cols = []
delta_cols = []
gamma_cols = []

# Separate frequencies
for col in freq_cols:

    try:
        freq = int(col.split("_")[1])

        if 0 <= freq <= 4:
            delta_cols.append(col)

        elif 4 < freq <= 8:
            theta_cols.append(col)

        elif 8 < freq <= 13:
            alpha_cols.append(col)

        elif 13 < freq <= 30:
            beta_cols.append(col)

        elif freq > 30:
            gamma_cols.append(col)

    except:
        pass

# Create new brainwave features
brainwaves = pd.DataFrame()

brainwaves["Delta"] = data[delta_cols].mean(axis=1)
brainwaves["Theta"] = data[theta_cols].mean(axis=1)
brainwaves["Alpha"] = data[alpha_cols].mean(axis=1)
brainwaves["Beta"] = data[beta_cols].mean(axis=1)
brainwaves["Gamma"] = data[gamma_cols].mean(axis=1)

# Add labels
brainwaves["Label"] = data["Label"]

# Save new dataset
brainwaves.to_csv("brainwave_dataset.csv", index=False)

print("\nConverted Brainwave Dataset:\n")
print(brainwaves.head())

print("\nSaved as brainwave_dataset.csv")