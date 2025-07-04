# analysis/core/json_safe_analyzer.py
"""
JSON-safe analyzer utility for handling numpy types and complex data structures
"""
import json
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Union
from datetime import datetime


class JSONSafeAnalyzer:
    """
    Mixin class providing JSON-safe data handling for all analyzer classes
    Can be used via inheritance or composition
    """

    def _make_json_safe(self, obj: Any) -> Any:
        """
        Recursively convert all data types to JSON-safe equivalents

        Args:
            obj: Any Python object that might contain numpy types

        Returns:
            JSON-serializable equivalent
        """
        # Handle numpy integer types
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.integer)):
            return int(obj)

        # Handle numpy floating types
        elif isinstance(obj, (np.float64, np.float32, np.float16, np.floating)):
            return float(obj)

        # Handle numpy boolean types
        elif isinstance(obj, np.bool_):
            return bool(obj)

        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle PyTorch tensors
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()

        # Handle dictionaries recursively
        elif isinstance(obj, dict):
            return {str(k): self._make_json_safe(v) for k, v in obj.items()}

        # Handle lists and tuples recursively
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(item) for item in obj]

        # Handle sets
        elif isinstance(obj, set):
            return [self._make_json_safe(item) for item in obj]

        # Handle datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()

        # Handle Path objects
        elif isinstance(obj, Path):
            return str(obj)

        # Return as-is for already JSON-safe types
        return obj

    def save_json_safe(self, data: Any, filepath: Union[str, Path],
                       indent: int = 2, add_metadata: bool = True) -> bool:
        """
        Save data to JSON file with automatic type conversion

        Args:
            data: Data to save
            filepath: Output file path
            indent: JSON indentation
            add_metadata: Whether to add saving metadata

        Returns:
            bool: Success status
        """
        try:
            # Convert to JSON-safe format
            safe_data = self._make_json_safe(data)

            # Add metadata if requested
            if add_metadata:
                safe_data = {
                    'data': safe_data,
                    'metadata': {
                        'saved_at': datetime.now().isoformat(),
                        'analyzer_class': self.__class__.__name__,
                        'json_safe_version': '1.0'
                    }
                }

            # Ensure directory exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save with JSON safety
            with open(filepath, 'w') as f:
                json.dump(safe_data, f, indent=indent)

            print(f"✅ JSON-safe save successful: {filepath}")
            return True

        except Exception as e:
            print(f"❌ JSON-safe save failed for {filepath}: {e}")
            return False

    def load_json_safe(self, filepath: Union[str, Path]) -> Union[Dict, None]:
        """
        Load JSON data with error handling

        Args:
            filepath: File to load

        Returns:
            Loaded data or None if failed
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Handle metadata wrapper
            if isinstance(data, dict) and 'data' in data and 'metadata' in data:
                print(f"📊 Loaded data from {data['metadata'].get('saved_at', 'unknown time')}")
                return data['data']

            return data

        except Exception as e:
            print(f"❌ Failed to load {filepath}: {e}")
            return None

    def validate_json_safety(self, data: Any) -> Dict[str, Any]:
        """
        Validate that data can be JSON serialized without errors

        Args:
            data: Data to validate

        Returns:
            Validation report
        """
        try:
            # Test original data
            json.dumps(data)
            original_safe = True
        except Exception as e:
            original_safe = False
            original_error = str(e)

        try:
            # Test converted data
            safe_data = self._make_json_safe(data)
            json.dumps(safe_data)
            converted_safe = True
        except Exception as e:
            converted_safe = False
            converted_error = str(e)

        return {
            'original_json_safe': original_safe,
            'converted_json_safe': converted_safe,
            'original_error': original_error if not original_safe else None,
            'converted_error': converted_error if not converted_safe else None,
            'data_type': type(data).__name__,
            'needs_conversion': not original_safe and converted_safe
        }
