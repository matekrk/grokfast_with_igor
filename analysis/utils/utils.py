import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


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


import json


def clean_for_json(obj):
    """
    Recursively clean a Python object by removing or replacing elements
    that are not JSON serializable.
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items() if is_json_serializable(k)}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj if is_json_serializable(item)]
    elif is_json_serializable(obj):
        return obj
    else:
        # Replace with string representation, None, or exclude
        return str(obj)  # or return None, or other default value


def is_json_serializable(obj):
    """
    Check if an object is JSON serializable.
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


# Example usage
def save_clean_json(data, filename):
    cleaned_data = clean_for_json(data)
    with open(filename, 'w') as f:
        json.dump(cleaned_data, f, indent=4)


class FittingScore:
    def __init__(self):
        self.min_train_loss = +1.e+6
        self.min_eval_loss = +1.e+6
        self.max_train_loss = -1.0e+6
        self.max_eval_loss = -1.0e+6
        self.epsilon = 1.0e-6

    def normalize(self, loss, min_loss, max_loss):
        # return (loss - self.min_train_loss) / (self.max_train_loss - self.min_train_loss + self.epsilon)
        return (loss - min_loss) / max((max_loss - min_loss), self.epsilon)

    def __call__(self, train_loss, train_accu, eval_loss, eval_accu):
        return self.fitting_score(train_loss, train_accu, eval_loss, eval_accu)

    def fitting_score(self, train_loss, train_accu, eval_loss, eval_accu):
        self.min_train_loss = max(-1.e+6, min(train_loss, self.min_train_loss))
        self.min_eval_loss = max(-1.e+6, min(eval_loss, self.min_eval_loss))
        self.max_train_loss = min(+1.e+6, max(train_loss, self.max_train_loss))
        self.max_eval_loss = min(+1.e+6, max(eval_loss, self.max_eval_loss))
        return ((1. - np.abs(train_accu - eval_accu)) *
                (1. - np.abs(self.normalize(train_loss, min_loss=self.min_train_loss, max_loss=self.max_train_loss) -
                             self.normalize(eval_loss, min_loss=self.min_eval_loss, max_loss=self.max_eval_loss))))
