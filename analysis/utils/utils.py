import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
import torch


def create_model(embedding, num_layers, heads_per_layer, batch_size,
                 operation, num_tokens, seq_len, ratio,
                 criterion, device,
                 optimizer_name, scheduler_name,
                 learning_rate, weight_decay,
                 base_dir="results", init_xavier=False,
                 enable_head_outputs=True,
                 ):
    # info lazy import to prevent circular import error
    from analysis.models.analysis_transformer import Decoder

    ff = datetime.now().strftime("%f")
    xavier = '_xavier' if init_xavier else ''
    sched = '_sched' if init_xavier else '_nosched'
    optim = f'_{optimizer_name}' if optimizer_name else ''
    sched = f'_{scheduler_name}' if scheduler_name else ''
    lr = f'_lr{learning_rate}' if learning_rate > 0 else ''
    wd = f'_wd{weight_decay:.1g}' if weight_decay > 0 else ''
    id = f"l{num_layers}_h{heads_per_layer}_e{embedding}_b{batch_size}_{operation[:4]}-{num_tokens - 2}{xavier}{optim}{lr}{wd}{sched}_r{ratio}_{ff}"

    # info save_dir
    save_dir = Path(base_dir) / f"{id}"
    save_dir.mkdir(exist_ok=True)

    # info checkpoint saves
    checkpoint_dir = Path(save_dir) / f"checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # info stats directory
    stats_dir = Path(save_dir) / "stats"
    stats_dir.mkdir(exist_ok=True)

    # When creating Block instances, pass the flag
    if enable_head_outputs:
        model = Decoder(
            dim=embedding,
            num_layers=num_layers,
            num_heads=heads_per_layer,
            num_tokens=num_tokens,
            seq_len=seq_len,
            criterion=criterion,
            device=device,
            id=id,
            save_dir=save_dir,
            checkpoint_dir=checkpoint_dir,
            store_head_outputs=True  # Pass this to Decoder
        )
    else:
        # Original instantiation
        model = Decoder(
            dim=embedding,
            num_layers=num_layers,
            num_heads=heads_per_layer,
            num_tokens=num_tokens,
            seq_len=seq_len,
            criterion=criterion,
            device=device,
            id=id,
            save_dir=save_dir,
            checkpoint_dir=checkpoint_dir
        )
    model.to(device)
    if init_xavier:
        model.apply_xavier_init()

    return model, save_dir, checkpoint_dir, stats_dir


def create_optimizer(model, config):
    optimizer = None
    if config['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        ValueError("Optimizer '{}' not recognized".format(config['optimizer']))
    return optimizer


def create_scheduler(config):
    if config['scheduler'] is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            config.optimizer, lambda update: 1 if update > 10 else update / 10
        )
    else:
        scheduler = None
    return scheduler


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


def find_closest_jump(jump_epochs, target_epoch):
    """Find the jump epoch closest to the target epoch"""
    return jump_epochs[np.argmin(np.abs(np.array(jump_epochs) - target_epoch))]


def get_class_name(obj, shorten=False):
    """
    Get the class name of an object, class, or method.

    Args:
        obj: An instance, class reference, or method
        shorten: If True, shortens the name by removing common prefixes/underscores

    Returns:
        str: The class name (shortened if requested)
    """
    # Handle different types of objects to extract class name
    if hasattr(obj, '__self__') and obj.__self__ is not None:
        # Bound method (accessed through an instance)
        class_name = obj.__self__.__class__.__name__
    elif hasattr(obj, '__qualname__'):
        # Function or unbound method
        qualname = obj.__qualname__
        # Extract class name from qualname (Format: ClassName.method_name)
        if '.' in qualname:
            class_name = qualname.split('.')[0]
        else:
            # Handle functions not attached to classes
            return "Not a class method"
    elif isinstance(obj, type):
        # If obj is a class
        class_name = obj.__name__
    else:
        # If obj is an instance
        class_name = obj.__class__.__name__

    # Shorten the name if requested
    if shorten:
        # Remove common prefixes in PyTorch
        if class_name.startswith('torch_'):
            class_name = class_name[6:]
        elif class_name.startswith('nn_'):
            class_name = class_name[3:]

        # Replace double underscores with single
        class_name = class_name.replace('__', '_')

        # Remove trailing underscores
        class_name = class_name.rstrip('_')

    return class_name


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


def get_memory_pressure():
    # System memory pressure
    sys_mem = psutil.virtual_memory()
    sys_pressure = sys_mem.percent
    sys_available_gb = sys_mem.available / (1024 ** 3)

    # Process-specific memory
    process = psutil.Process()
    process_memory_gb = process.memory_info().rss / (1024 ** 3)

    # GPU memory if applicable
    gpu_pressure = None
    if torch.cuda.is_available():
        gpu_pressure = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0

    return {
        "system_percent": sys_pressure,
        "available_gb": sys_available_gb,
        "process_gb": process_memory_gb,
        "gpu_percent": gpu_pressure
    }


def find_non_serializable_objects(obj, path=""):
    """
    Recursively find and report non-JSON-serializable objects.
    Use this to diagnose JSON serialization errors.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_non_serializable_objects(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_non_serializable_objects(v, f"{path}[{i}]")
    else:
        try:
            json.dumps(obj)
        except (TypeError, OverflowError):
            print(f"Non-serializable object at {path}: {type(obj)} - {str(obj)[:100]}")


def debug_grokking_step(grokking_step, location="Unknown", logger=None):
    """Helper to debug grokking_step type issues"""
    debug_msg = [
        f"[DEBUG] grokking_step at {location}:",
        f"  Type: {type(grokking_step)}",
        f"  Value: {grokking_step}"
    ]

    if isinstance(grokking_step, list):
        debug_msg.append(f"  List length: {len(grokking_step)}")
        debug_msg.append(
            f"  List items: {', '.join(str(x) for x in grokking_step[:5])}{' ...' if len(grokking_step) > 5 else ''}")

    # Print to console
    print("\t".join(debug_msg))

    # Log if logger is available
    if logger:
        for line in debug_msg:
            logger.log_data('debug_grokking', f'{location}_info', line)

from pathlib import Path
import re
from datetime import datetime


def find_best_matching_file(directory, pattern, criteria='newest'):
    """
    Find the best matching file in directory based on pattern and criteria.

    Args:
        directory: Directory to search in
        pattern: Glob pattern to match files (e.g., "data_*.csv")
        criteria: Method to select best match if multiple files match:
            - 'newest': Most recently modified file
            - 'largest': Largest file by size
            - 'longest_name': File with longest name
            - 'largest_number': File with largest number in name
            - 'most_similar': Most similar to pattern (Levenshtein distance)

    Returns:
        Path object of best matching file or None if no match
    """
    directory = Path(directory)
    matching_files = list(directory.glob(pattern))

    if not matching_files:
        return None

    # If only one file matches, return it
    if len(matching_files) == 1:
        return matching_files[0]

    # Select best match based on criteria
    if criteria == 'newest':
        return max(matching_files, key=lambda f: f.stat().st_mtime)

    elif criteria == 'largest':
        return max(matching_files, key=lambda f: f.stat().st_size)

    elif criteria == 'longest_name':
        return max(matching_files, key=lambda f: len(f.name))

    elif criteria == 'largest_number':
        def extract_number(filename):
            # Extract all numbers from filename and return the largest
            numbers = re.findall(r'\d+', filename.name)
            return max([int(n) for n in numbers]) if numbers else 0

        return max(matching_files, key=extract_number)

    elif criteria == 'most_similar':
        # Levenshtein distance (needs python-Levenshtein or similar library)
        try:
            from Levenshtein import distance
            # Extract the base pattern without wildcards for comparison
            base = pattern.replace('*', '').replace('?', '')
            return min(matching_files, key=lambda f: distance(base, f.name))
        except ImportError:
            print("For 'most_similar' criteria, install Levenshtein: pip install python-Levenshtein")
            # Fall back to newest if Levenshtein not available
            return max(matching_files, key=lambda f: f.stat().st_mtime)

    # Default to newest if criteria not recognized
    return max(matching_files, key=lambda f: f.stat().st_mtime)


from pathlib import Path
import re
from datetime import datetime


def find_matching_files(directory, pattern, criteria='newest', return_sorted=False):
    """
    Find matching files in directory based on pattern and criteria.

    Args:
        directory: Directory to search in (str or Path object)
        pattern: Glob pattern to match files (e.g., "data_*.csv")
        criteria: Method to select/sort matches:
            - 'newest': Most recently modified file(s)
            - 'oldest': Oldest modified file(s)
            - 'largest': Largest file(s) by size
            - 'smallest': Smallest file(s) by size
            - 'longest_name': File(s) with longest name
            - 'largest_number': File(s) with largest number in name
            - 'most_similar': Most similar to pattern (Levenshtein distance)
        return_sorted: If True, returns sorted list of all matches
                       If False, returns only the best match (default)

    Returns:
        If return_sorted=False: Path object of best matching file or None
        If return_sorted=True: List of Path objects sorted by criteria or empty list
    """
    directory = Path(directory)
    matching_files = list(directory.glob(pattern))

    if not matching_files:
        return [] if return_sorted else None

    # If only one file matches and not requesting sorted list
    if len(matching_files) == 1 and not return_sorted:
        return matching_files[0]

    # Define key functions for different criteria
    key_functions = {
        'newest': lambda f: f.stat().st_mtime,
        'oldest': lambda f: -f.stat().st_mtime,  # Negative for reverse sorting
        'largest': lambda f: f.stat().st_size,
        'smallest': lambda f: -f.stat().st_size,  # Negative for reverse sorting
        'longest_name': lambda f: len(f.name),
        'shortest_name': lambda f: -len(f.name),  # Negative for reverse sorting
        'largest_number': lambda f: max([int(n) for n in re.findall(r'\d+', f.name)] or [0]),
        'smallest_number': lambda f: -max([int(n) for n in re.findall(r'\d+', f.name)] or [0])
    }

    # Handle 'most_similar' criteria separately
    if criteria == 'most_similar' or criteria == 'least_similar':
        try:
            from Levenshtein import distance
            base = pattern.replace('*', '').replace('?', '')
            reverse = (criteria == 'least_similar')
            sorted_files = sorted(matching_files,
                                  key=lambda f: distance(base, f.name),
                                  reverse=reverse)
        except ImportError:
            print("For similarity criteria, install Levenshtein: pip install python-Levenshtein")
            # Fall back to newest
            sorted_files = sorted(matching_files,
                                  key=lambda f: f.stat().st_mtime,
                                  reverse=True)
    else:
        # Get the appropriate key function
        key_func = key_functions.get(criteria, key_functions['newest'])
        reverse = not criteria.startswith('smallest') and not criteria.startswith('oldest')
        sorted_files = sorted(matching_files, key=key_func, reverse=reverse)

    # Return either the sorted list or just the best match
    return sorted_files if return_sorted else sorted_files[0]

"""
# Example usage:
if __name__ == "__main__":
    # Example 1: Get just the best match
    data_dir = Path.home() / "projects" / "data"
    best_file = find_matching_files(data_dir, "model_*.pt", criteria="newest")
    print(f"Best match: {best_file}")

    # Example 2: Get all matching files, sorted by criterion
    checkpoint_dir = Path("./checkpoints")
    all_checkpoints = find_matching_files(checkpoint_dir, "model_epoch_*.pt",
                                          criteria="largest_number",
                                          return_sorted=True)
    print("All checkpoints in order:")
    for i, checkpoint in enumerate(all_checkpoints, 1):
        print(f"{i}. {checkpoint.name}")

    # Example 3: Load the top 3 largest models
    if all_checkpoints:
        top_3_models = all_checkpoints[:3]
        print(f"\nTop 3 models with highest epoch numbers: {[m.name for m in top_3_models]}")

        # You could then load them for ensemble predictions
        # models = [torch.load(model_path) for model_path in top_3_models]
"""
"""
# Example usage:
if __name__ == "__main__":
    # Example 1: Find the newest log file
    log_file = find_best_matching_file("/path/to/logs", "app_*.log", criteria="newest")
    print(f"Latest log file: {log_file}")

    # Example 2: Find the dataset with largest version number
    dataset = find_best_matching_file("./datasets", "data_v*.csv", criteria="largest_number")
    print(f"Latest dataset version: {dataset}")

    # Example 3: Find the largest checkpoint file
    checkpoint = find_best_matching_file("./models/checkpoints", "model_*.pt", criteria="largest")
    print(f"Largest checkpoint: {checkpoint}")
"""

def read_json(file_path):
    """
    Read JSON data from a file.

    Args:
        file_path: Path to JSON file (str or Path object)

    Returns:
        The parsed JSON data
    """
    file_path = Path(file_path)  # Convert to Path object if it isn't already

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {file_path}")
        return None
