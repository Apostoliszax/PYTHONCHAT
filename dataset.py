import torch
from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.x_data = X
        self.y_data = y
    def __getitem__(self, index):
        return torch.tensor(self.x_data[index], dtype=torch.float32), torch.tensor(self.y_data[index], dtype=torch.long)
    def __len__(self):
        return len(self.x_data)

def CreateDataloader(X, y, batchSize):
    dataset = ChatDataset(X, y)
    return DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=0)