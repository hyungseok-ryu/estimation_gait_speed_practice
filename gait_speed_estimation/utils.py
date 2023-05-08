import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def DatasetFromPath(data_path, IMU_locations):
    walking_conditions = ['treadmill']
    sensors = ['imu', 'gcLeft', 'conditions']
    subjects_data = []  # Initialize an empty dictionary to store DataFrames for each subject
    
    # subjects = ['AB18', 'AB19']
    # for subject in subjects:
    for subject in os.listdir(data_path):
        subject_path = os.path.join(data_path, subject)
        if os.path.isdir(subject_path):
            imu_data = []
            gcLeft_data = []
            condition_data = []
            for measurement in os.listdir(subject_path):
                measurement_path = os.path.join(subject_path, measurement)
                if (measurement != "osimxml") & (os.path.isdir(measurement_path)):
                    for condition in walking_conditions:
                        condition_path = os.path.join(measurement_path, condition)
                        if os.path.isdir(condition_path):
                            for sensor in sensors:
                                sensor_path = os.path.join(condition_path, sensor)
                                if os.path.isdir(sensor_path):
                                    for data_file in os.listdir(sensor_path):
                                        if data_file[-3:] == "csv":
                                            data_file_path = os.path.join(sensor_path, data_file)
                                            if os.path.isfile(data_file_path):
                                                data = pd.read_csv(data_file_path)
                                                if sensor == "imu":
                                                    filtered_columns = [col for col in data.columns if any(col.startswith(location) for location in IMU_locations)]
                                                    filtered_columns.append('Header')
                                                    data = data[filtered_columns]
                                                    imu_data.append(data)
                                                # elif sensor == "gcLeft":
                                                #     gcLeft_data.append(data)
                                                elif sensor == "conditions":
                                                    data['Trial'] = [subject + '_' + data_file[:-4]] * len(data)
                                                    condition_data.append(data)
            
            for i in range(len(imu_data)):
                imu_data[i] = imu_data[i].merge(condition_data[i], on='Header', how='left')  # Merge imu_data and condition_data based on the 'Header' column
            if imu_data:
                combined_data = pd.concat(imu_data, ignore_index=True)

            subjects_data.append(combined_data)
            # print("Subject : ", subject)
            # print("Number of NaN values in dataset : ", np.isnan(combined_data['foot_Gyro_Z']).sum())
    dataset = pd.concat(subjects_data, ignore_index=True)
    dataset = dataset.drop(columns=['Header'])
    # Preprocessing Data
    scaler = StandardScaler()
    dataset[dataset.columns.difference(['Speed', 'Trial'])] = scaler.fit_transform(dataset[dataset.columns.difference(['Speed', 'Trial'])])
    
    return dataset

def segment_data(data, window_size, stride=1):
    segments = []
    output = []
    
    data['Trial'] = data['Trial'].apply(lambda x: str(x) if not isinstance(x, str) else x)
    
    # Split 'trial' column into 'subject' and 'trial'
    data['subject'] = data['Trial'].apply(lambda x: x.split('_')[0])
    data['trial'] = data['Trial'].apply(lambda x: '_'.join(x.split('_')[1:]))

    # Drop the original 'Trial' column
    data = data.drop(columns=['Trial'])

    # Group the dataset by 'subject' and 'trial'
    grouped_dataset = data.groupby(['subject', 'trial'])

    for (_, _), group in grouped_dataset:
        group = group.reset_index(drop=True)
        
        for start in range(0, len(group) - window_size, stride):
            end = start + window_size
            segment = group.iloc[start:end].drop(columns=['subject', 'trial', 'Speed']).values
            speed = group.iloc[start:end]['Speed']
            
            segments.append(segment)
            output.append(speed)
            
    return np.array(segments), np.array(output)