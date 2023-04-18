import nidaqmx
import torch
import numpy as np
from nidaqmx.constants import AcquisitionType
from collections import deque
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from LSTMmodel import GaitLSTM
from utils import buffer_data


# The number of IMU (It can only 1, 2, 4)
num_IMU = 2

# Define ML model parameter
input_size = num_IMU*6
hidden_size = 128
num_layers = 3
output_size = 1
dropout_prob = 0.4

# Load the trained LSTM model
model = GaitLSTM(input_size, hidden_size, num_layers, output_size, dropout_prob)
model.load_state_dict(torch.load('LSTMmodel_'+str(num_IMU)+'IMU.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Buffer to store the incoming data
window_size = 200
buffer = deque(maxlen=window_size)

device_name = "Dev1"
channels = "ai0,ai1,ai2,ai3,ai4,ai5"

# Real-time plotting
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r-')

def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 5)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(estimated_speed)
    if len(xdata) > 100:
        xdata.pop(0)
        ydata.pop(0)
        ax.set_xlim(xdata[0], xdata[-1])
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(f"{device_name}/{channels}")
    task.timing.cfg_samp_clk_timing(1000, sample_mode=AcquisitionType.CONTINUOUS)

    frame = 0
    while True:
        frame += 1
        sample = np.array(task.read(number_of_samples_per_channel=1))  ####
        
        # Preprocess the incoming data
        scaler = StandardScaler()
        preprocessed_sample = scaler.fit_transform(sample)

        # Buffer the preprocessed data and create segments when the buffer is full
        segment = buffer_data(buffer, preprocessed_sample, window_size)

        if segment is not None:
            # Reshape the segment to match the input shape of the LSTM model
            segment = segment.reshape(1, window_size, -1)
            
            # Convert the segment to a PyTorch tensor
            input_data = torch.tensor(segment, dtype=torch.float32).to(device)

            # Estimate gait speed using the LSTM model
            with torch.no_grad():
                estimated_speed = model(input_data).item()
            
            print(f"Estimated gait speed: {estimated_speed}")
            ani.event_source.start()
            plt.pause(0.01)

plt.show()