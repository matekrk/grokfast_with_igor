import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import random


class ModularArithmeticDataset(Dataset):
    """
    Dataset for modular arithmetic tasks.

    Supports various operations:
    - 'add': (a + b) % m
    - 'subtract': (a - b) % m
    - 'multiply': (a * b) % m
    - 'power': (a ^ b) % m (modular exponentiation)
    """

    def __init__(self, modulus, op='add', num_samples=10000, max_int=None, seed=42):
        """
        Initialize the modular arithmetic dataset.

        Args:
            modulus (int): The modulus for arithmetic operations.
            op (str): The operation to perform ('add', 'subtract', 'multiply', 'power').
            num_samples (int): Number of samples to generate.
            max_int (int): Maximum integer for operands (defaults to modulus-1).
            seed (int): Random seed for reproducibility.
        """
        self.modulus = modulus
        self.op = op
        self.num_samples = num_samples
        self.max_int = max_int if max_int is not None else modulus - 1

        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Generate the dataset
        self.data = self._generate_data()

    def _generate_data(self):
        """Generate pairs of integers and their modular operation result."""
        data = []

        for _ in range(self.num_samples):
            a = random.randint(0, self.max_int)
            b = random.randint(0, self.max_int)

            if self.op == 'add':
                result = (a + b) % self.modulus
            elif self.op == 'subtract':
                result = (a - b) % self.modulus
            elif self.op == 'multiply':
                result = (a * b) % self.modulus
            elif self.op == 'power':
                # Modular exponentiation using Python's pow function
                result = pow(a, b, self.modulus)
            else:
                raise ValueError(f"Unsupported operation: {self.op}")

            data.append((a, b, result))

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        a, b, result = self.data[idx]

        # Convert to format suitable for the transformer
        # Input sequence: [a, b]
        # Target: result
        inputs = torch.tensor([a, b], dtype=torch.long)
        target = torch.tensor(result, dtype=torch.long)

        return inputs, target


class SequenceModularArithmeticDataset(Dataset):
    """
    Dataset for modular arithmetic tasks presented as token sequences.

    Format for Transformer models: [a, b, =, result]
    This version encodes the problem as a sequence prediction task.
    warning data examples are given as rows [:-1] the input, [-1] the output need to be transposed withing forward
    """

    def __init__(self, modulus, op='add', num_samples=10000, max_int=None, seed=42):
        """
        Initialize the sequence modular arithmetic dataset.

        Args:
            modulus (int): The modulus for arithmetic operations.
            op (str): The operation to perform ('add', 'subtract', 'multiply', 'power').
            num_samples (int): Number of samples to generate.
            max_int (int): Maximum integer for operands (defaults to modulus-1).
            seed (int): Random seed for reproducibility.
        """
        self.modulus = modulus
        self.op = op
        self.num_samples = num_samples
        self.max_int = max_int if max_int is not None else modulus - 1

        # Special tokens
        self.eq_token = self.modulus  # '=' token
        self.op_token = self.modulus + 1  # operation token

        # Total vocabulary size: 0 to modulus-1, plus special tokens
        self.vocab_size = self.modulus + 2

        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Generate the dataset
        self.data = self._generate_data()

    def _generate_data(self):
        """Generate pairs of integers and their modular operation result as sequences."""
        data = []

        for a in range(self.modulus):
            for b in range(1, self.modulus):
                # a = random.randint(0, self.max_int)
                # b = random.randint(0, self.max_int)

                if self.op == 'add':
                    result = (a + b) % self.modulus
                    op_symbol = self.op_token  # Addition token
                elif self.op == 'subtract':
                    result = (a - b) % self.modulus
                    op_symbol = self.op_token  # Subtraction token
                elif self.op == 'multiply':
                    result = (a * b) % self.modulus
                    op_symbol = self.op_token  # Multiplication token
                elif self.op == 'power':
                    result = pow(a, b, self.modulus)
                    op_symbol = self.op_token  # Power token
                else:
                    raise ValueError(f"Unsupported operation: {self.op}")

                # Create sequence [a, op, b, =, result]
                sequence = [a, op_symbol, b, self.eq_token, result]
                data.append(sequence)

        self.num_samples = len(data)
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = self.data[idx]

        # Convert to tensor format
        # Input: all tokens except the last one [a, op, b, =]
        # Target: the last token [result]
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target = torch.tensor(sequence[-1], dtype=torch.long)

        return input_seq, target


def create_modular_dataloaders(modulus, op='add', batch_size=32,
                               train_ratio=0.5, sequence_format=True, seed=42,
                               dataset_split_indices=None):
    """
    Create train and test dataloaders for modular arithmetic tasks.

    Args:
        modulus (int): The modulus for arithmetic operations.
        op (str): The operation to perform ('add', 'subtract', 'multiply', 'power').
        num_samples (int): Total number of samples to generate.
        batch_size (int): Batch size for DataLoader.
        train_ratio (float): Ratio of training samples (0-1).
        sequence_format (bool): If True, use SequenceModularArithmeticDataset.
                               If False, use ModularArithmeticDataset.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, test_loader, vocab_size)
    """
    # Set seed for reproducible splits
    torch.manual_seed(seed)

    # info Create all examples dataset from scratch
    if sequence_format:
        dataset = SequenceModularArithmeticDataset(
            modulus=modulus,
            op=op,
            seed=seed
        )
        vocab_size = dataset.vocab_size
    else:
        dataset = ModularArithmeticDataset(
            modulus=modulus,
            op=op,
            seed=seed
        )
        vocab_size = modulus

    # Split into train and test sets
    # info split randomly using seed
    if dataset_split_indices is None:
        if train_ratio >= 1. or train_ratio < 0.:
            raise ValueError(f"train_ratio must be >= 0 and < 1, but got {train_ratio}")
        num_samples = len(dataset.data)
        train_size = int(train_ratio * num_samples)
        test_size = num_samples - train_size

        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size]
        )
    else:
        train_dataset = Subset(dataset, dataset_split_indices['train_indices'])
        test_dataset = Subset(dataset, dataset_split_indices['test_indices'])

    split_indices = {
        'train_indices': train_dataset.indices,
        'test_indices': test_dataset.indices,
        'seed': seed,
        'train_ratio': train_ratio,
    }
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, vocab_size, split_indices
