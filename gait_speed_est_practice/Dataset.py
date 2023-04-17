
#######################################################################################################################################
# 
# IMU Data :                   timestamp  GyrX    GyrY    GyrZ    AccX    AccY    AccZ
# Preprocessed Data :        stride_lengths_avg    clearances_min_avg       clearances_max_avg
#                            stride_times_avg      swing_times_avg          stance_times_avg
#                            stance_ratios_avg     cadence_avg              speed_avg
#                            stride_lengths_CV     clearances_min_CV        clearances_max_CV
#                            stride_times_CV       swing_times_CV           stance_times_CV
#
# 18-2 = 16 subjects
# Consisting of 6-minute walks under single- (st) and dual-task (dt)
# Conditions in non-fatigued (control) and fatigued (fatigue) states
# 9 units on head (HE), chest(ST), lower back(SA), wrists(LW, RW), legs(LL, RL), and feet(LF, RF)
#######################################################################################################################################

# Use only Control Data
# Use Legs and Feet Data

from torch.utils.data import Dataset
import torch

class GaitDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

