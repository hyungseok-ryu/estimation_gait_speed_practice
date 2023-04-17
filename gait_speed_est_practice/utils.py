import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def DatasetFromPath(IMU_data_path, processed_data_path):
    
    conditions = ['OG_dt_control', 'OG_st_control', 'OG_dt_fatigue', 'OG_st_fatigue']
    
    Input_IMU = ['LL', 'LF', 'RL', 'RF']
    
    dataset_list = []
    for condition in conditions:
        condition_path_IMU = os.path.join(IMU_data_path, condition)
        condition_path_processed = os.path.join(processed_data_path, condition)
        if (os.path.isdir(condition_path_IMU)) & (os.path.isdir(condition_path_processed)):
            X_list = []
            for subject in os.listdir(condition_path_IMU):
                subject_path_IMU = os.path.join(condition_path_IMU, subject)
                subject_path_processed = os.path.join(condition_path_processed, subject)
                if (os.path.isdir(subject_path_IMU)) & (os.path.isdir(subject_path_processed)):
                    data_list = []
                    for IMU in Input_IMU:
                        data_file_IMU = f'{IMU}.csv'
                        data_file_path_IMU = os.path.join(subject_path_IMU, data_file_IMU)
                        data_file_path_processed = os.path.join(subject_path_processed, 'aggregate_params.csv')
                        if os.path.isfile(data_file_path_IMU):
                            data_IMU = pd.read_csv(data_file_path_IMU, 
                                                   usecols=['GyrX', 
                                                            'GyrY',
                                                            'GyrZ', 
                                                            'AccX',
                                                            'AccY', 
                                                            'AccZ'])
                            data_IMU = data_IMU.add_suffix('_'+IMU)
                            data_list.append(data_IMU)
                        
                        if os.path.isfile(data_file_path_processed):
                            data_speed = pd.read_csv(data_file_path_processed,
                                                     usecols=['speed_avg'])
                            
                    data = pd.concat(data_list, axis=1)
                    data['gait_speed'] = [data_speed.iloc[0, 0]] * len(data)
                    data['subject'] = [int(subject[-2:])] * len(data)
                    X_list.append(data)
            
            X = pd.concat(X_list, axis=0)
            X['condition'] = [condition] * len(X)
            X['condition'] = [condition] * len(X)
            dataset_list.append(X)
    
    Dataset = pd.concat(dataset_list, axis=0)
    
    # Preprocess data
    scaler = StandardScaler()
    Dataset[Dataset.columns.difference(['gait_speed', 'subject', 'condition'])] = scaler.fit_transform(Dataset[Dataset.columns.difference(['gait_speed', 'subject', 'condition'])])

    return Dataset

def segment_data(data, window_size, stride):
    segments = []
    output = []
    
    for condition in data['condition'].unique():
        subset = data[data['condition'] == condition]
        
        for start in range(0, len(subset) - window_size, stride):
            end = start + window_size
            segment = subset.iloc[start:end, :24].values
            # gait_speed = subset.iloc[start:end]['gait_speed'].mean() 
            gait_speed = subset.iloc[start:end]['gait_speed']
            
            segments.append(segment)
            output.append(gait_speed)
    
    return np.array(segments), np.array(output)