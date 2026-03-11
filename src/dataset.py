import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MusicDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    


def load_dataloaders(dataset_path, batch_size=32, shuffle=True):
    data = torch.load(dataset_path)
    dataset = MusicDataset(data["inputs"], data["targets"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader