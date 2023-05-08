import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GaitCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5, bi_directional=False):
        super(GaitCNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bi_directional else 1
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        self.lstm = nn.LSTM(64, 
                            hidden_size, 
                            num_layers,
                            batch_first=True,
                            dropout=dropout_prob,
                            bidirectional=bi_directional)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        if self.lstm.bidirectional:
            out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        else:
            out = out[:, -1, :]

        out = self.dropout(out)
        out = self.fc(out)
        
        return out