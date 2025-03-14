For; the; `train_dataloader_state` and `eval_dataloader_state`;
parameters, you; need; to; capture; enough; information; to; resume;
iterating from the correct; position.However, this is challenging;
because; PyTorch; DataLoaders; don't have a simple built-in mechanism '
'to save and restore their state.; Here's what should be included in these
dataloader states:
### Train DataLoader State
```python
train_dataloader_state = {
    # Current position information
    'batch_idx': current_batch_index,  # Which batch we're on
    'epoch': current_epoch,  # Current epoch

    # Sampler state (critical for reproducibility)
    'sampler_iter_state': get_sampler_iter_state(train_loader),

    # DataLoader configuration (for recreation)
    'batch_size': train_loader.batch_size,
    'shuffle': True,  # or whatever your setting is
    'num_workers': train_loader.num_workers,
    'pin_memory': train_loader.pin_memory,
    'drop_last': train_loader.drop_last
}
```

### Eval DataLoader State
```python
eval_dataloader_state = {
    # Similar structure but typically simpler since evaluation is often sequential
    'batch_idx': 0,  # Usually reset for each evaluation

    # DataLoader configuration
    'batch_size': eval_loader.batch_size,
    'shuffle': False,  # Typically False for eval
    'num_workers': eval_loader.num_workers,
    'pin_memory': eval_loader.pin_memory,
    'drop_last': eval_loader.drop_last
}
```

### Capturing the Sampler Iterator State

# The; most; challenging; part is capturing; the; state; of; the; iterator from the sampler,
# which; controls; the; order; of; samples.Here's a helper function that attempts to '
# capture this:



def get_sampler_iter_state(dataloader):
    """
    Try to capture the state of a dataloader's sampler iterator.
    This is inherently difficult in PyTorch.
    """
    # Check if we have access to the sampler
    if not hasattr(dataloader, 'sampler'):
        return None

    sampler = dataloader.sampler

    # Check if sampler is a RandomSampler (most common with shuffle=True)
    if hasattr(sampler, 'generator'):
        # If it has a generator, we can save its state
        return {
            'generator_state': sampler.generator.get_state() if sampler.generator else None
        }

    # For SequentialSampler, we don't need state
    return None


```

### Restoring the DataLoader State

# When; resuming, you; 'll need to:
#
# 1.; Recreate; the; DataLoader with the same configuration
# 2.; Try; to; restore; the; sampler; state if applicable
# 3.; Skip; to; the; correct; batch if resuming; within; an; epoch



def recreate_dataloader_with_state(dataset, state):
    """Recreate a dataloader and try to restore its state"""
    if state is None:
        return DataLoader(dataset, batch_size=32, shuffle=True)  # Default

    # Create new dataloader with same config
    dataloader = DataLoader(
        dataset,
        batch_size=state.get('batch_size', 32),
        shuffle=state.get('shuffle', True),
        num_workers=state.get('num_workers', 0),
        pin_memory=state.get('pin_memory', False),
        drop_last=state.get('drop_last', False)
    )

    # Try to restore sampler state if possible
    sampler_state = state.get('sampler_iter_state')
    if sampler_state and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'generator'):
        if dataloader.sampler.generator and 'generator_state' in sampler_state:
            dataloader.sampler.generator.set_state(sampler_state['generator_state'])

    return dataloader, state.get('batch_idx', 0)



# Even with all these efforts, completely restoring a DataLoader's exact iteration state is difficult due to PyTorch's implementation details.The most reliable approach is to:

# 1.; Always finish epochs; completely(save; checkpoints; at; epoch; boundaries)
# 2.; Use; fixed; seeds for all random operations
# 3.; Save; dataset; split; indices

# If; you; must; resume; within; an; epoch, your; best; option is to; skip; to; the;
# approximately; correct; batch; index, understanding; that; the; exact; samples; might; differ\
# ; slightly; due to; shuffling; variations.

# For; your; specific; use; case with modular arithmetic, where the order of training examples
# isn't intrinsically meaningful, focusing on saving the dataset split indices and RNG states
# for reproducibility is likely more important than trying to resume from the exact batch
# position.
def get_sampler_iter_state(dataloader):
    """
    Try to capture the state of a dataloader's sampler iterator.
    This is inherently difficult in PyTorch.
    """
    # Check if we have access to the sampler
    if not hasattr(dataloader, 'sampler'):
        return None

    sampler = dataloader.sampler

    # Check if sampler is a RandomSampler (most common with shuffle=True)
    if hasattr(sampler, 'generator'):
        # If it has a generator, we can save its state
        return {
            'generator_state': sampler.generator.get_state() if sampler.generator else None
        }

    # For SequentialSampler, we don't need state
    return None

# saving the original parameters used to create dataset
dataset_params = {
    'modulus': 97,
    'operation': 'multiply',
    'num_samples': 10000,
    'max_int': None,
    'seed': 42
}

# Include in checkpoint metadata or state
ckpt_manager.save_checkpoint(
    extra_data={'dataset_params': dataset_params}
)


# WHEN RESUMING TRAINING

# recreate the original dataset
# Load parameters from checkpoint
dataset_params = state_info['extra_data']['dataset_params']

# Recreate the same dataset
full_dataset = ModularArithmeticDataset(
    modulus=dataset_params['modulus'],
    op=dataset_params['operation'],
    num_samples=dataset_params['num_samples'],
    seed=dataset_params['seed']
)


# apply the same datasplit
# Get indices from the loaded checkpoint
split_indices = torch.load(split_indices_path)
train_indices = split_indices['train_indices']
test_indices = split_indices['test_indices']

# Create subset datasets with these exact indices
from torch.utils.data import Subset
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# restore random generators' states
# Set RNG states from checkpoint
random.setstate(checkpoint['rng_states']['python'])
np.random.set_state(checkpoint['rng_states']['numpy'])
torch.set_rng_state(checkpoint['rng_states']['torch'])
if torch.cuda.is_available() and checkpoint['rng_states']['torch_cuda'] is not None:
    torch.cuda.set_rng_state_all(checkpoint['rng_states']['torch_cuda'])


# the whole example
def resume_training_new_process(checkpoint_path):
    """Resume training in a new process"""
    # Load checkpoint & state info
    checkpoint = torch.load(checkpoint_path)
    state_path = str(checkpoint_path).replace(".pt", "_state.json")
    with open(state_path, "r") as f:
        state_info = json.load(f)

    # Get dataset parameters
    dataset_params = state_info.get("extra_data", {}).get("dataset_params", {})
    if not dataset_params:
        raise ValueError("Dataset parameters not found in checkpoint")

    # Recreate the full dataset
    full_dataset = ModularArithmeticDataset(
        modulus=dataset_params.get("modulus", 97),
        op=dataset_params.get("operation", "add"),
        num_samples=dataset_params.get("num_samples", 10000),
        seed=dataset_params.get("seed", 42)
    )

    # Get split indices path
    split_indices_path = state_info.get("dataset_split_indices_file")
    if not split_indices_path or not os.path.exists(split_indices_path):
        raise ValueError("Dataset split indices not found")

    # Load split indices
    split_indices = torch.load(split_indices_path)

    # Recreate train/test datasets
    train_dataset = Subset(full_dataset, split_indices["train_indices"])
    test_dataset = Subset(full_dataset, split_indices["test_indices"])

    # Create dataloaders
    batch_size = dataset_params.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Restore RNG states
    if "rng_states" in checkpoint:
    # (restore RNG states as shown earlier)

    # Continue with model creation, loading weights, etc.
    # ...

    return model, train_loader, test_loader, checkpoint["step"], checkpoint["epoch"]


