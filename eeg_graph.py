import serial
import matplotlib.pyplot as plt
from collections import deque

ser = serial.Serial('COM3', 115200)

data = deque([0]*100)

plt.ion()

fig, ax = plt.subplots()
line, = ax.plot(data)

ax.set_title("Real-Time EEG Signal")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

while True:

    try:
        raw = ser.readline().decode().strip()

        if raw.isdigit():

            value = int(raw)

            data.append(value)
            data.popleft()

            line.set_ydata(data)
            line.set_xdata(range(len(data)))

            ax.relim()
            ax.autoscale_view()

            plt.draw()
            plt.pause(0.01)

    except:
        pass