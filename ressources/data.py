import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch

class CanopyHightDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.targets = data["rh98"]
        self.embeddings = data.iloc[:, 3:]

        assert len(self.targets) == len(self.embeddings), f"{len(self.targets)=}, {len(self.embeddings)=}"

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        embs = self.embeddings.iloc[item, :]
        target = self.targets.iloc[item]

        embs_tensor = torch.from_numpy(embs.array.to_numpy().astype(np.float32))
        targets_tensor = torch.from_numpy(np.array([target]).astype(np.float32))
        return embs_tensor, targets_tensor


def get_training_loader(data: pd.DataFrame, batch_size: int):
    training_data = CanopyHightDataset(data)
    training_dataloader = DataLoader(training_data, batch_size)
    print(f"Validation data samples: {len(training_data)} in {len(training_dataloader)} batches")
    return training_dataloader


def get_validation_loader(data: pd.DataFrame, batch_size: int):
    val_data = CanopyHightDataset(data)
    val_dataloader = DataLoader(val_data, batch_size)
    print(f"Training data samples: {len(val_data)} in {len(val_dataloader)} batches")
    return val_dataloader
