import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class PowerDataset(Dataset):
    def __init__(self, csv_file, seq_len=96, overlap=0):

        df = pd.read_csv(csv_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        
        power_data = df['power'].values
        
        self.scaler = MinMaxScaler()
        power_normalized = self.scaler.fit_transform(power_data.reshape(-1, 1)).flatten()
        
        self.sequences = []
        step = seq_len - overlap  
        
        for i in range(0, len(power_normalized) - seq_len + 1, step):
            sequence = power_normalized[i:i + seq_len]
            self.sequences.append(sequence)
     
    def __getitem__(self, index):
        power_seq = torch.tensor(self.sequences[index], dtype=torch.float32).view(-1, 1)
        output_data = {"power": power_seq}
        return output_data
    
    def __len__(self):
        return len(self.sequences)
    
    def get_scaler(self):
        return self.scaler

def create_power_dataloader(csv_file, batch_size=4, shuffle=True, seq_len=96, overlap=0):
    dataset = PowerDataset(csv_file, seq_len, overlap)
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader, dataset