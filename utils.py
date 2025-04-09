import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

# info asynchronous unlinking of several files; perhaps the direct use of os.remove,
#  as well as asynchronous actions may speed up?
async def remove_files_async(file_list):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(
            executor,
            lambda: [os.remove(f) for f in file_list if os.path.exists(f)]
        )

### info to execute
# file_list = [f"checkpoint_{i}.pth" for i in range(100)]
# asyncio.run(remove_files_async(file_list))


# info some simple dataloader and sampler functions

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

def init_train_dataloader_state(dataloader):
    return {
        # Current position information
        'batch_idx': 0,  # Which batch we're on
        'epoch': 0,  # Current epoch
        # Sampler state (critical for reproducibility)
        'sampler_iter_state': get_sampler_iter_state(dataloader),
        # DataLoader configuration (for recreation)
        'batch_size': dataloader.batch_size,
        'shuffle': True,  # or whatever your setting is
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
        'drop_last': dataloader.drop_last
    }

def init_val_dataloader_state(dataloader):
    return {
        # Similar structure but typically simpler since evaluation is often sequential
        'batch_idx': 0,  # Usually reset for each evaluation
        # DataLoader configuration
        'batch_size': dataloader.batch_size,
        'shuffle': False,  # Typically False for eval
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
        'drop_last': dataloader.drop_last
    }
