import nidaqmx
import torch
import numpy as np
import time
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from collections import deque
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from LSTMmodel import GaitLSTM
from utils import buffer_data

# The number of IMU (It can only 1, 2, 4)
num_IMU = 1

Gyro = True  
Acc = False

# Define ML model parameter
input_size = num_IMU*(Gyro+Acc)*3
hidden_size = 128
num_layers = 3
output_size = 1
dropout_prob = 0.4

# Load the trained LSTM model
model = GaitLSTM(input_size, hidden_size, num_layers, output_size, dropout_prob)
model.load_state_dict(torch.load('LSTMmodel_'+str(num_IMU)+'IMU_gyro.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Buffer to store the incoming data
window_size = 200
buffer = deque(maxlen=window_size)


# Replace device_name is the name of your device, and channel_name is the name of your input channel.
device_name = "Dev1"
channel_names = ["ai7", "ai8", "ai9"]

ylabel = ['Gyro X', 'Gyro Y', 'Gyro Z']

gyro_range = 2000 # Choose the range you are using: 250 dps, 500 dps, 1000 dps, 2000 dps
analog_output_range = 5
gyro_sensitivity = gyro_range / (2 * analog_output_range)

# Create a task
with nidaqmx.Task() as task:
    # Configure the analog input channel
    for channel_name in channel_names:
        task.ai_channels.add_ai_voltage_chan(f'{device_name}/{channel_name}',
                                             terminal_config=TerminalConfiguration.RSE)

    # Set the sample rate and number of samples
    sample_rate = 1000  # 1000 samples per second
    num_samples = 100  # Acquire 10 samples
    task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=num_samples,
                                     sample_mode=AcquisitionType.CONTINUOUS)
    
    while True:
        try:
            start = time.time()
            
            data = np.array(task.read(number_of_samples_per_channel=num_samples, timeout=1)) * gyro_sensitivity
            
            # Preprocess the incoming data
            scaler = StandardScaler()
            preprocessed_data = scaler.fit_transform(data)
            
            # Buffer the preprocessed data and create segments when the buffer is full
            segment = buffer_data(buffer, preprocessed_data, window_size)

            
            if segment is not None:
                # Reshape the segment to match the input shape of the LSTM model
                segment = segment.reshape(1, window_size, -1)
                
                # Convert the segment to a PyTorch tensor
                input_data = torch.tensor(segment, dtype=torch.float32).to(device)
                # Estimate gait speed using the LSTM model
                with torch.no_grad():
                    estimated_speed = model(input_data).item()
                
                print(f"Estimated gait speed: {estimated_speed}")
            
        except KeyboardInterrupt:
            print("Stopping data acquisition...")
            task.stop()
            break    