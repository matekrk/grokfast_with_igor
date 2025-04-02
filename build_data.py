import torch
from torch.utils.data import Dataset, DataLoader


class TransformerDataset(Dataset):
    def __init__(self, token_array, labels=None):
        """
        token_array: numpy array where each column is a sequence of token IDs for one example
        labels: optional array of labels for each example
        """
        self.data = torch.tensor(token_array.T)  # Transpose to make each row an example
        self.labels = None if labels is None else torch.tensor(labels)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]


# Create dataset and dataloader
dataset = TransformerDataset(token_array, labels)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Usage in training loop
for batch in dataloader:
    if len(batch) == 2:
        inputs, targets = batch
    else:
        inputs = batch
    # Forward pass, loss calculation, etc.