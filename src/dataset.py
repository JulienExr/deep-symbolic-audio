import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MusicDataset(Dataset):
    def __init__(self, inputs, targets, emotions=None):
        self.inputs = inputs
        self.targets = targets
        self.emotions = emotions

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.emotions is not None:
            return self.inputs[idx], self.targets[idx], self.emotions[idx]
        return self.inputs[idx], self.targets[idx]
    


def load_dataloaders(dataset_path, batch_size=32, shuffle=True, require_emotions=False):
    data = torch.load(dataset_path)
    emotions = data.get("emotions")
    if require_emotions and emotions is None:
        raise ValueError(f"Le dataset {dataset_path} ne contient pas de labels d'emotion.")
    dataset = MusicDataset(data["inputs"], data["targets"], emotions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=4, pin_memory=True, persistent_workers=True
                            )
    return dataloader
