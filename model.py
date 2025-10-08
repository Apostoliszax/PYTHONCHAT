import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.dropout1(x)
        x = self.act2(self.fc2(x))
        x = self.dropout2(x)
        x = self.act3(self.fc3(x))
        x = self.dropout3(x)
        x = self.out(x)
        return x
