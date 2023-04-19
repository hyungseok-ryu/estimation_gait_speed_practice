import nidaqmx
import time
import numpy as np
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import matplotlib.pyplot as plt

##########################################################
#######  Press Ctrl + C to stop data acquisition  ########
##########################################################


# Replace device_name is the name of your device, and channel_name is the name of your input channel.
device_name = "Dev1"
channel_names = ["ai7", "ai8", "ai9"]

label_IMU = ['Gyro X', 'Gyro Y', 'Gyro Z']
color_IMU = ['b', 'r', 'g']

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
    
    # Set up the real-time plot
    plt.ion()
    fig, ax = plt.subplots(1, 1, sharex=True)
    x_data = np.arange(0, num_samples)/sample_rate
    y_data = [np.zeros(num_samples) for _ in channel_names]
    ax.set_xlabel('Time (s)')
    ax.set_ylim([-1500, 1500])
    ax.set_title('Real-time Data Acquisition')
    plt.show(block=False)
    # Start the task
    task.start()

    # x_data ,y_data = [], []
    # Continuously read and plot data
    while True:
        try:
            start = time.time()
            
            data = np.array(task.read(number_of_samples_per_channel=num_samples, timeout=1)) * gyro_sensitivity
            
            
            for i, channel_data in enumerate(data):
                y_data[i] = np.concatenate((y_data[i], channel_data))
                x_data_i = np.arange(len(y_data[i])) / sample_rate
                ax.plot(x_data_i, y_data[i], color_IMU[i])
                ax.legend(label_IMU)
                ax.set_xlim(left=max(0, np.max(x_data_i)-0.5), right=np.max(x_data_i)+0.5)
                # ax[i].relim()
                # ax[i].autoscale_view()

            plt.draw()
            plt.pause(0.0001)

            print('time : ', time.time()-start)
            
            
        except KeyboardInterrupt:
            print("Stopping data acquisition...")
            task.stop()
            break

    # Close the plot
    plt.ioff()
    plt.show()