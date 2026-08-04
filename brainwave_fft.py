import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

ser = serial.Serial('COM3', 115200)

samples = []

plt.ion()

fig, ax = plt.subplots()

while True:

    try:
        raw = ser.readline().decode().strip()

        if raw.isdigit():

            value = int(raw)
            samples.append(value)

            if len(samples) >= 128:

                eeg = np.array(samples)

                yf = np.abs(fft(eeg))

                xf = np.linspace(0, 50, len(yf)//2)

                ax.clear()

                ax.plot(xf, yf[:len(yf)//2])

                ax.set_title("EEG Frequency Spectrum")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Amplitude")

                plt.pause(0.01)

                samples = []

    except:
        pass