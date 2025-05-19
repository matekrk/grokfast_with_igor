import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from analysis.utils.utils import find_non_serializable_objects


class DataLogger:
    def __init__(self, id=None, **kwargs):
        self.logs = defaultdict(lambda: defaultdict(list))
        self.id = id


    def log_data(self, category, key, value):
        """
        Log data under a specific category and key.

        :param category: The category or subset of data logs.
        :param key: The key within the category to store the value.
        :param value: The value to log.
        """
        self.logs[category][key].append(value)

    def update_category(self, category, data_dict):
        """
        Add or update a whole sub-dictionary for a specific category.

        :param category: The category to update.
        :param data_dict: The dictionary containing data to add or update.
        """
        for key in data_dict:
            self.log_data(category, key, data_dict[key])

    def update_category_means(self, category, data_dict):
        self.update_category_with_lists(category, data_dict)

    def value_in_category_key(self, category, key, value):
        return True if value in self.logs[category][key] else False

    def value_in_category(self, category, value):
        return True in self.logs[category].values()

    def update_category_with_lists(self, category, data_dict):
        for key in data_dict:
            self.log_data(category, key, data_dict[key])

    def get_length(self, category, key):
        return len(self.logs[category][key])

    def get_all_logs(self):
        return self.logs.copy()

    def get_logs(self, category=None):
        """
        Retrieve logs for a specific category or all logs if no category is specified.

        :param category: The category to retrieve logs for. If None, retrieve all logs.
        :return: The logs for the specified category or all logs.
        """
        if category:
            return self.logs.get(category, {})
        return self.logs

    def get_last_value(self, category, key):
        category_dict = self.logs.get(category, {})
        if key in category_dict:
            return category_dict[key][-1]
        return None

    def save_logs_to_file(self, file_path, categories=None, binary_handler=None):
        """
        Save logs to file with comprehensive type handling for scientific computing types.

        Args:
            file_path: Path where to save the logs
            categories: Optional list of categories to save (saves all if None)
            binary_handler: Optional function to handle binary data conversion

        Returns:
            str: Path where logs were saved
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine what data to save
        if categories:
            data_to_save = {cat: self.logs[cat] for cat in categories if cat in self.logs}
        else:
            data_to_save = dict(self.logs)

        # Enhanced numeric type handling function that catches ALL numpy/torch numeric types
        def deep_convert_numeric_types(item):
            """Recursively convert numpy/torch types to Python native types"""
            # Handle numpy/torch scalar types
            if hasattr(item, 'item'):
                return item.item()  # This handles both numpy and torch scalar types

            # Handle numpy arrays
            elif isinstance(item, np.ndarray):
                return item.tolist()

            # Handle torch tensors
            elif isinstance(item, torch.Tensor):
                return item.detach().cpu().numpy().tolist()

            # Handle dictionaries
            elif isinstance(item, dict):
                return {k: deep_convert_numeric_types(v) for k, v in item.items()}

            # Handle lists and tuples
            elif isinstance(item, (list, tuple)):
                return [deep_convert_numeric_types(i) for i in item]

            # Handle numpy scalar types explicitly
            elif type(item).__module__ == 'numpy':
                return item.item() if hasattr(item, 'item') else float(item)

            # Return other types unchanged
            return item

        # Use our deep conversion function if no custom handler provided
        if binary_handler is None:
            binary_handler = deep_convert_numeric_types

        # Process data through binary handler
        processed_data = {}
        binary_paths = {}

        for cat, cat_data in data_to_save.items():
            find_non_serializable_objects(cat_data, cat)
            processed_data[cat] = {}
            for key, values in cat_data.items():
                # Convert all values using our deep conversion function
                try:
                    processed_values = deep_convert_numeric_types(values)
                    processed_data[cat][key] = processed_values
                except Exception as e:
                    print(f"Warning: Error processing {cat}/{key}: {e}")
                    # Fall back to string representation for problematic values
                    processed_data[cat][key] = [str(v) for v in values]

        # Save main data file
        try:
            with open(path, 'w') as f:
                json.dump({
                    "data": processed_data,
                    "binary_paths": binary_paths,
                    "id": self.id,
                    "save_time": datetime.now().isoformat()
                }, f, indent=2)
        except TypeError as e:
            # If we still get JSON serialization errors, try emergency fallback
            print(f"JSON serialization error: {e}")
            print("Attempting emergency type conversion...")

            # Emergency string conversion for all values
            emergency_data = {}
            for cat, cat_data in processed_data.items():
                emergency_data[cat] = {}
                for key, values in cat_data.items():
                    try:
                        # Try each value individually to find problematic ones
                        safe_values = []
                        for i, v in enumerate(values):
                            try:
                                # Test if this value is JSON serializable
                                json.dumps(v)
                                safe_values.append(v)
                            except (TypeError, OverflowError):
                                # Convert to string if not serializable
                                print(f"Converting non-serializable value at {cat}/{key}[{i}] to string")
                                safe_values.append(str(v))
                        emergency_data[cat][key] = safe_values
                    except Exception:
                        # Last resort: pure string conversion
                        emergency_data[cat][key] = [str(v) for v in values]

            # Save with emergency converted data
            with open(path, 'w') as f:
                json.dump({
                    "data": emergency_data,
                    "binary_paths": binary_paths,
                    "id": self.id,
                    "save_time": datetime.now().isoformat(),
                    "emergency_conversion": True
                }, f, indent=2)

        return str(path)

    def load_logs_from_file(self, file_path, merge=False):
        """
        Load logs from a file, with support for binary data.

        Args:
            file_path: Path to the saved logs
            merge: Whether to merge with existing logs (True) or replace (False)

        Returns:
            bool: Success status
        """
        path = Path(file_path)

        try:
            # Load main data file
            if path.suffix.lower() == '.npz':
                # NPZ format
                with np.load(path, allow_pickle=True) as data:
                    loaded_data = data['data'].item()
                    binary_paths = data.get('binary_paths', {}).item() if 'binary_paths' in data else {}
            else:
                # JSON format
                with open(path, 'r') as f:
                    file_data = json.load(f)
                    loaded_data = file_data["data"]
                    binary_paths = file_data.get("binary_paths", {})

            # Handle binary data references
            for cat, cat_data in loaded_data.items():
                for key, values in cat_data.items():
                    # Check if we have binary references
                    if binary_paths and cat in binary_paths and key in binary_paths[cat]:
                        # Load binary data
                        bin_path = binary_paths[cat][key]
                        with np.load(bin_path, allow_pickle=True) as bin_data:
                            bin_values = bin_data['data']
                            # Replace references with actual data
                            loaded_data[cat][key] = bin_values

            # Update or replace existing logs
            if not merge:
                self.logs.clear()

            # Merge loaded data with existing logs
            for cat, cat_data in loaded_data.items():
                for key, values in cat_data.items():
                    if merge:
                        self.logs[cat][key].extend(values)
                    else:
                        self.logs[cat][key] = values

            return True

        except Exception as e:
            print(f"Error loading logs from {file_path}: {e}")
            return False

    def save_binary_data(self, category, key, data, file_path=None):
        """
        Save binary data (numpy arrays, tensors) separately.

        Args:
            category: The category for the data
            key: The key for the data
            data: The binary data to save
            file_path: Optional custom path, default uses category/key

        Returns:
            str: Path where binary data was saved
        """
        # If no path provided, generate one based on category and key
        if file_path is None:
            file_path = f"logs/{self.id}/{category}_{key}.npz"

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert torch tensor to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # Save as compressed numpy file
        np.savez_compressed(path, data=data)

        # Store metadata reference in logs
        self.log_data(category, f"{key}_binary_ref", str(path))

        return str(path)

    def clear_category(self, category):
        """
        Clear all data for a specific category.

        Args:
            category: The category to clear

        Returns:
            bool: True if category existed and was cleared, False otherwise
        """
        if category in self.logs:
            del self.logs[category]
            return True
        return False

    def clear_logs(self, category=None):
        """
        Clear logs for a specific category or all logs if no category is specified.

        :param category: The category to clear logs for. If None, clear all logs.
        """
        if category:
            if category in self.logs:
                del self.logs[category]
        else:
            self.logs.clear()
