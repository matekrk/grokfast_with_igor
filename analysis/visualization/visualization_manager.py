import importlib
import importlib.util
import json
import os
from pathlib import Path

from matplotlib import pyplot as plt


class VisualizationManager:
    """
    Manages visualizations and provides a unified interface for visualization operations.
    This class handles the registration, discovery, and execution of visualization methods.
    """

    def __init__(self, save_dir, logger=None):
        """
        Initialize the visualization manager.

        Args:
            logger: The logger object containing the data to visualize
        """
        self.save_dir = Path(save_dir)  # fixme yes
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.available_fields_cache = None  # fixme is it still needed ?
        self.visualizers = []  # List to store registered visualizers
        self.data = None  # Storage for logger data

    def set_logger(self, logger):
        """
        Set or update the logger instance.

        Args:
            logger: The logger object containing the data to visualize

        Returns:
            self: For method chaining
        """
        self.logger = logger
        # Invalidate cache when logger changes
        self.available_fields_cache = None
        return self

    def set_data(self, logger_data):
        """
        Set the logger data for visualization.

        Args:
            logger_data: The logger data (either DataLogger instance or dictionary)

        Returns:
            None
        """
        # Handle both DataLogger instances and dictionaries
        if hasattr(logger_data, 'data') and isinstance(logger_data['data'], dict):
            # It's a DataLogger instance
            self.data = logger_data.data
        elif isinstance(logger_data, dict):
            # It's already a dictionary
            self.data = logger_data
        else:
            # Unknown type
            print(f"\tWarning: Unsupported data type {type(logger_data)}. Data not set.")
            self.data = None


    def register_visualizer(self, visualizer):
        """
        Register a visualizer with the manager.

        Args:
            visualizer: The visualizer to register

        Returns:
            bool: True if registration was successful, False otherwise
        """
        # Check if visualizer is already registered
        if visualizer in self.visualizers:
            return False

        # Check if visualizer has required interface
        required_methods = ['can_visualize', 'get_supported_fields', 'visualize', 'save']
        for method in required_methods:
            if not hasattr(visualizer, method) or not callable(getattr(visualizer, method)):
                print(f"Visualizer {visualizer} missing required method: {method}")
                return False

        # Add to visualizers list
        self.visualizers.append(visualizer)
        return True

    def register_visualizers_from_directory(self, directory_path):
        """
        Automatically discover and register visualizers from a directory.

        Args:
            directory_path: Path to directory containing visualizer modules

        Returns:
            int: Number of visualizers registered
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")

        count = 0
        for file_path in directory.glob("*_visualizer.py"):
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for visualizer classes in the module
            for name in dir(module):
                obj = getattr(module, name)
                try:
                    # Check if it's a non-abstract BaseVisualizer subclass
                    if (isinstance(obj, type) and
                            hasattr(obj, '__mro__') and
                            'BaseVisualizer' in [c.__name__ for c in obj.__mro__] and
                            obj.__name__ != 'BaseVisualizer'):
                        # Instantiate and register
                        visualizer = obj()
                        self.register_visualizer(visualizer)
                        count += 1
                except (TypeError, AttributeError):
                    continue

        return count

    def get_available_fields(self, logger_data=None):
        """
        Get a list of all available fields in the logger data that can be visualized.

        Args:
            logger_data: Optional logger data (either DataLogger instance or dictionary).
                        If None, uses the last set data

        Returns:
            list: List of available fields that can be visualized
        """
        if logger_data is None:
            logger_data = self.data

        if logger_data is None:
            return []

        # Handle both DataLogger instances and dictionaries
        data_dict = None
        if hasattr(logger_data, 'logs') and isinstance(logger_data.logs, dict):
            # It's a DataLogger instance
            data_dict = logger_data.logs
        elif isinstance(logger_data, dict):
            # It's already a dictionary
            data_dict = logger_data
        else:
            # Unknown type
            return []

        available_fields = []

        # Directly add all top-level keys as available fields
        available_fields.extend(data_dict.keys())

        # For each visualizer, get the fields it supports
        for visualizer in self.visualizers:
            supported_fields = visualizer.get_supported_fields(data_dict)
            available_fields.extend(supported_fields)

        # Remove duplicates and sort
        return sorted(list(set(available_fields)))


    def can_visualize_field(self, field_path):
        """
        Check if a specific field can be visualized by any registered visualizer.

        Args:
            field_path: The path to the field in dot notation (e.g., 'training.accuracy')

        Returns:
            bool: True if the field can be visualized, False otherwise
        """
        # Ensure available fields are cached
        fields = self.get_available_fields()

        # Check if this field is in the available fields
        return field_path in fields

    def get_field_data(self, field_path, logger_data=None):
        """
        Get data for a specific field path.

        Args:
            field_path: Path to the field (e.g., "training.accuracy")
            logger_data: Optional logger data (either DataLogger instance or dictionary).
                        If None, uses the last set data

        Returns:
            The data at the specified field path, or None if not found
        """
        if logger_data is None:
            logger_data = self.data

        if logger_data is None:
            return None

        # Handle both DataLogger instances and dictionaries
        data_dict = None
        if hasattr(logger_data, 'logs') and isinstance(logger_data.logs, dict):
            # It's a DataLogger instance
            data_dict = logger_data.logs
        elif isinstance(logger_data, dict):
            # It's already a dictionary
            data_dict = logger_data
        else:
            # Unknown type
            return None

        # Handle wildcard paths (e.g., "training.*")
        if field_path.endswith('.*'):
            prefix = field_path[:-2]
            result = {}
            for key, value in data_dict.items():
                if key.startswith(prefix):
                    result[key] = value
            return result if result else None

        # Handle direct path
        if field_path in data_dict:
            return data_dict[field_path]

        # Handle nested paths (e.g., "category.key" or "category.0.key")
        if '.' in field_path:
            parts = field_path.split('.')
            current = data_dict

            for i, part in enumerate(parts):
                # Try as integer if it's a number
                if part.isdigit():
                    part = int(part)

                # Check if we can access this part
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif isinstance(current, (list, tuple)) and isinstance(part, int) and 0 <= part < len(current):
                    current = current[part]
                else:
                    # Try alternative ways to find the data
                    if i == 0:  # Only for first level
                        # Look for similar category names
                        for category in data_dict:
                            if part.lower() in category.lower():
                                remaining_path = '.'.join(parts[i + 1:])
                                if remaining_path:
                                    # Recursively look up the remaining path
                                    sub_result = self.get_field_data(f"{category}.{remaining_path}", data_dict)
                                    if sub_result is not None:
                                        return sub_result
                                else:
                                    return data_dict[category]
                    return None

            return current

        # Look for fields with similar names
        for key in data_dict:
            if field_path.lower() in key.lower():
                return data_dict[key]

        # Special handling for common field patterns
        if field_path == "training":
            # For "training", return the whole training data if it exists
            if "training" in data_dict:
                return data_dict["training"]
            # Or look for keys that have "training" in them
            training_data = {}
            for key in data_dict:
                if "training" in key.lower():
                    training_data[key] = data_dict[key]
            if training_data:
                return training_data

        elif field_path == "attention_patterns":
            # For "attention_patterns", look for keys that might contain attention patterns
            for key in data_dict:
                if "attention" in key.lower() and "pattern" in key.lower():
                    return data_dict[key]

        elif field_path == "circuit_tracker.circuit_history":
            # Special handling for circuit history
            if "circuit_tracker" in data_dict and "circuit_history" in data_dict["circuit_tracker"]:
                return data_dict["circuit_tracker"]["circuit_history"]

            # Try to find any circuit history data
            for key in data_dict:
                if "circuit" in key.lower() and "history" in key.lower():
                    return data_dict[key]

        # Field not found
        return None

    def find_compatible_visualizer(self, field_path, visualizer_name=None):
        """
        Find a compatible visualizer for a given field.

        Args:
            field_path: The path to the field in dot notation
            visualizer_name: Optional name of specific visualizer to use

        Returns:
            BaseVisualizer: A compatible visualizer, or None if not found
        """
        data = self.get_field_data(field_path)
        if data is None:
            return None

        # If specific visualizer requested, try to find it
        if visualizer_name:
            visualizer = next((v for v in self.visualizers if v.name == visualizer_name), None)
            if visualizer and visualizer.can_visualize(data):
                return visualizer
            return None

        # Otherwise, find the first compatible visualizer
        return next((v for v in self.visualizers if v.can_visualize(data)), None)

    def visualize_field(self, field_path, visualizer_name=None, config=None):
        """
        Visualize a specific field using an appropriate visualizer.

        Args:
            field_path: The path to the field in dot notation
            visualizer_name: Optional name of specific visualizer to use
            config: Optional visualization configuration

        Returns:
            object: The visualization object, or None if visualization failed
        """
        # Get the data
        data = self.get_field_data(field_path)
        if data is None:
            raise ValueError(f"Field {field_path} not found in logger data")

        # Find appropriate visualizer
        visualizer = self.find_compatible_visualizer(field_path, visualizer_name)
        if not visualizer:
            raise ValueError(
                f"No compatible visualizer found for field {field_path}" +
                (f" with name {visualizer_name}" if visualizer_name else "")
            )

        # Create visualization
        return visualizer.visualize(data, config)

    def save_visualization(self, visualization, path, visualizer_name=None):
        """
        Save a visualization to disk.

        Args:
            visualization: The visualization object to save
            path: Path where to save the visualization
            visualizer_name: Optional name of visualizer to use for saving

        Returns:
            str: The path where the visualization was saved
        """
        # Find the visualizer that created this visualization
        if visualizer_name:
            visualizer = next((v for v in self.visualizers if v.name == visualizer_name), None)
            if not visualizer:
                raise ValueError(f"Visualizer with name {visualizer_name} not found")
        else:
            # Use the first visualizer that accepts this visualization type
            visualizer = next((v for v in self.visualizers
                               if hasattr(v, 'can_save') and v.can_save(visualization)),
                              self.visualizers[0] if self.visualizers else None)

        if not visualizer:
            raise ValueError("No visualizer available to save the visualization")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save the visualization
        return visualizer.save(visualization, path)

    def visualize_and_save(self, field_path, save_path, visualizer_name=None, config=None):
        """
        Visualize a field and save it in one operation.

        Args:
            field_path: The path to the field in dot notation
            save_path: Path where to save the visualization
            visualizer_name: Optional name of specific visualizer to use
            config: Optional visualization configuration

        Returns:
            tuple: (visualization object, saved path)
        """
        visualization = self.visualize_field(field_path, visualizer_name, config)
        saved_path = self.save_visualization(visualization, save_path, visualizer_name)
        return visualization, saved_path

    def get_visualizer_info(self):
        """
        Get information about registered visualizers.

        Returns:
            list: List of dictionaries with visualizer information
        """
        return [
            {
                'name': v.name,
                'description': v.description
            }
            for v in self.visualizers
        ]

    def export_available_fields(self, path):
        """
        Export the list of available fields to a JSON file.

        Args:
            path: Path where to save the JSON file

        Returns:
            str: The path where the file was saved
        """
        fields = self.get_available_fields()

        with open(path, 'w') as f:
            json.dump(fields, f, indent=2)

        return path

    # Add to the existing VisualizationManager class
    def register_visualizers_from_directory(self, directory_path):
        """
        Automatically discover and register visualizers from a directory.

        Args:
            directory_path: Path to directory containing visualizer modules

        Returns:
            int: Number of visualizers registered
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")

        count = 0
        for file_path in directory.glob("*_visualizer.py"):
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for visualizer classes in the module
            for name in dir(module):
                obj = getattr(module, name)
                try:
                    # Check if it's a non-abstract BaseVisualizer subclass
                    if (isinstance(obj, type) and
                            hasattr(obj, '__mro__') and
                            'BaseVisualizer' in [c.__name__ for c in obj.__mro__] and
                            obj.__name__ != 'BaseVisualizer'):
                        # Instantiate and register
                        visualizer = obj()
                        self.register_visualizer(visualizer)
                        count += 1
                except (TypeError, AttributeError):
                    continue

        return count

    # Add to the existing VisualizationManager class
    def export_available_fields(self, path):
        """
        Export the list of available fields to a JSON file.

        Args:
            path: Path where to save the JSON file

        Returns:
            str: The path where the file was saved
        """
        fields = self.get_available_fields()

        with open(path, 'w') as f:
            json.dump(fields, f, indent=2)

        return path

    def list_registered_visualizers(self):
        """
        Get a list of all registered visualizers.

        Returns:
            list: Information about registered visualizers
        """
        visualizer_info = []
        for v in self.visualizers:
            visualizer_info.append({
                'name': v.name if hasattr(v, 'name') else v.__class__.__name__,
                'description': v.description if hasattr(v, 'description') else "No description available",
                'type': v.__class__.__name__
            })
        return visualizer_info

    def find_compatible_visualizers(self, field_path):
        """
        Find visualizers compatible with a specific field.

        Args:
            field_path: Path to the field

        Returns:
            list: Compatible visualizers
        """
        if self.data is None:
            return []

        field_data = self.get_field_data(field_path)
        if field_data is None:
            return []

        compatible = []
        for visualizer in self.visualizers:
            if visualizer.can_visualize({field_path: field_data}):
                compatible.append(visualizer)

        return compatible

    def save_visualization_results(self, results, file_path):
        """
        Save visualization results to a JSON file.

        Args:
            results: The results dictionary from create_example_visualizations
            file_path: Path to save the results

        Returns:
            str: The path where results were saved
        """
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        return file_path

    def load_visualization_results(self, file_path):
        """
        Load visualization results from a JSON file.

        Args:
            file_path: Path to the results file

        Returns:
            dict: The loaded results
        """
        with open(file_path, 'r') as f:
            return json.load(f)


    def create_specified_visualizations(self, logger_data, output_dir=None, fields_to_visualize=None,
                                        visualizer_mapping=None, discover_fields=True):
        """
        Create visualizations for the given logger data with flexible field selection.

        Args:
            logger_data: The logger data dictionary (from logger.logs or loaded from file)
            output_dir: Directory where to save the visualizations
            fields_to_visualize: List of specific fields to visualize (if None, discovers fields)
            visualizer_mapping: Optional dict mapping fields to specific visualizer names
            discover_fields: Whether to automatically discover additional fields

        Returns:
            dict: Paths to the saved visualizations
        """
        if output_dir is None:
            output_dir = self.save_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        results = {}

        # Set the data for visualization
        self.set_data(logger_data)

        # Determine fields to visualize
        target_fields = []

        # 1. Add explicitly specified fields
        if fields_to_visualize is not None:
            target_fields.extend(fields_to_visualize)

        # 2. Discover additional fields if requested
        if discover_fields:
            # Get available fields from all visualizers
            available_fields = self.get_available_fields()

            # Add fields that aren't already in target_fields
            for field in available_fields:
                if field not in target_fields:
                    target_fields.append(field)

        # If no fields specified or discovered, use some common defaults
        if not target_fields:
            target_fields = [
                "training", "evaluation", "circuit_analysis", "attention_patterns",
                "mlp_sparsity", "phase_transitions", "grokking_phases"
            ]

        # For each target field, create a visualization
        for field in target_fields:
            # Get the field data
            field_data = self.get_field_data(field)
            if field_data is None:
                print(f"Field '{field}' not found or empty - skipping")
                continue

            # Determine which visualizer to use
            if visualizer_mapping and field in visualizer_mapping:
                # Use specified visualizer if provided
                visualizer_name = visualizer_mapping[field]
                visualizer = next((v for v in self.visualizers if v.name == visualizer_name), None)
                if visualizer is None:
                    print(f"Specified visualizer '{visualizer_name}' for field '{field}' not found")
                    continue
            else:
                # Find compatible visualizers
                compatible_visualizers = [v for v in self.visualizers
                                          if v.can_visualize({field: field_data})]

                if not compatible_visualizers:
                    print(f"No compatible visualizer found for '{field}'")
                    continue

                # Use the first compatible visualizer
                visualizer = compatible_visualizers[0]

            try:
                # Create and save the visualization
                vis = visualizer.visualize({field: field_data})
                filename = f"{field.replace('.', '_')}.png"
                path = os.path.join(output_dir, filename)
                visualizer.save(vis, path)

                results[field] = path
                print(f"Created visualization for '{field}' using {visualizer.name}")
            except Exception as e:
                print(f"Error creating visualization for '{field}': {e}")

        return results

    def create_example_visualizations(self, logger_data, output_dir=None, fields_to_visualize=None,
                                      visualizer_mapping=None, discover_fields=True):
        """
        Create visualizations for the given logger data with intelligent visualizer selection.

        Args:
            logger_data: The logger data dictionary or DataLogger instance
            output_dir: Directory where to save the visualizations
            fields_to_visualize: List of specific fields to visualize (if None, discovers fields)
            visualizer_mapping: Optional dict mapping fields to specific visualizer names
            discover_fields: Whether to automatically discover additional fields

        Returns:
            dict: Paths to the saved visualizations
        """
        if output_dir is None:
            output_dir = self.save_dir

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Set the data for visualization
        self.set_data(logger_data)

        # Determine fields to visualize
        target_fields = []

        # 1. Add explicitly specified fields
        if fields_to_visualize is not None:
            target_fields.extend(fields_to_visualize)

        # 2. Discover additional fields if requested
        if discover_fields:
            # Get available fields from all visualizers
            available_fields = self.get_available_fields()

            # Add fields that aren't already in target_fields
            for field in available_fields:
                if field not in target_fields:
                    target_fields.append(field)

        # If no fields specified or discovered, use some common defaults
        if not target_fields:
            target_fields = [
                "training", "evaluation", "circuit_analysis", "attention_patterns",
                "mlp_sparsity", "phase_transitions", "grokking_phases"
            ]

        # For each target field, create a visualization
        for field in target_fields:
            # Get the field data
            field_data = self.get_field_data(field)
            if field_data is None:
                print(f"Field '{field}' not found or empty - skipping")
                continue

            # field_content = {field: field_data}

            # Determine which visualizer to use
            selected_visualizer = None

            # 1. Use specified visualizer if provided in mapping
            if visualizer_mapping and field in visualizer_mapping:
                visualizer_name = visualizer_mapping[field]
                selected_visualizer = next(
                    (v for v in self.visualizers if getattr(v, 'name', v.__class__.__name__) == visualizer_name), None)
                if selected_visualizer is None:
                    print(f"Specified visualizer '{visualizer_name}' for field '{field}' not found")

            # 2. If no mapping or specified visualizer not found, intelligently select best visualizer
            if selected_visualizer is None:
                # Get all compatible visualizers with their compatibility scores
                compatibility_scores = []

                for visualizer in self.visualizers:
                    try:
                        # Simple compatibility check
                        if visualizer.can_visualize(field_data):
                            # Calculate a score based on specialized field checks
                            score = self._calculate_visualizer_compatibility(visualizer, field, field_data)
                            compatibility_scores.append((visualizer, score))
                    except Exception as e:
                        print(f"Error checking compatibility for {visualizer.__class__.__name__}: {e}")

                # Sort by score (highest first)
                compatibility_scores.sort(key=lambda x: x[1], reverse=True)

                if compatibility_scores:
                    selected_visualizer = compatibility_scores[0][0]
                    # Print debug info about selection
                    print(
                        f"Selected {selected_visualizer.__class__.__name__} for field '{field}' (score: {compatibility_scores[0][1]:.2f})")
                    if len(compatibility_scores) > 1:
                        print(
                            f"  Other options: {', '.join(f'{v.__class__.__name__}({s:.2f})' for v, s in compatibility_scores[1:])}")
                else:
                    print(f"No compatible visualizer found for '{field}'")
                    continue

            # 3. Create and save the visualization with the selected visualizer
            try:
                # Create visualization
                # vis = selected_visualizer.visualize(field_content)
                vis = selected_visualizer.visualize(field_data)

                # Generate a unique filename
                vis_type = selected_visualizer.__class__.__name__.replace('Visualizer', '').lower()
                filename = f"{field.replace('.', '_')}_{vis_type}.png"
                save_path = output_path / filename

                # Save visualization
                selected_visualizer.save(vis, str(save_path))

                # Close matplotlib figure if applicable
                if hasattr(vis, 'clf'):
                    vis.clf()
                if hasattr(vis, 'close'):
                    plt.close(vis)

                results[field] = str(save_path)
                print(f"Created visualization for '{field}' saved to {save_path}")
            except Exception as e:
                print(f"Error creating visualization for '{field}': {e}")

        return results

    def _calculate_visualizer_compatibility_previous(self, visualizer, field, field_data):
        """
        Calculate a compatibility score between a visualizer and field data.
        Higher scores indicate better compatibility.

        Args:
            visualizer: The visualizer to check
            field: The field name
            field_data: The field data

        Returns:
            float: Compatibility score between 0 and 1
        """
        visualizer_name = visualizer.__class__.__name__.lower()
        field_name = field.lower()

        # Base score from visualizer.can_visualize() is 0.5
        base_score = 0.5

        # Check specific visualizer types against field characteristics
        if 'timeseries' in visualizer_name:
            # Time series visualizer is ideal for training metrics over time
            if 'training' in field_name or 'evaluation' in field_name:
                base_score += 0.4
            # Look for lists of numeric values that could be time series
            if isinstance(field_data, dict) and any(
                    isinstance(field_data.get(k), list) and
                    len(field_data.get(k)) > 0 and
                    all(isinstance(x, (int, float)) for x in field_data.get(k)[:5])
                    for k in field_data
            ):
                base_score += 0.3

        elif 'heatmap' in visualizer_name:
            # Heatmap visualizer is ideal for attention patterns
            if 'attention' in field_name and 'pattern' in field_name:
                base_score += 0.4
            # Look for 2D array-like data
            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if (isinstance(v, (list, tuple)) and len(v) > 0 and
                            isinstance(v[0], (list, tuple))):
                        base_score += 0.3
                        break

        elif 'network' in visualizer_name or 'graph' in visualizer_name:
            # Network graph visualizer is ideal for circuit relationships
            if 'circuit' in field_name or 'graph' in field_name or 'network' in field_name:
                base_score += 0.4
            # Connections or relationships
            if ('connections' in field_name or 'interactions' in field_name or
                    'relationship' in field_name):
                base_score += 0.3

        elif 'phase' in visualizer_name or 'transition' in visualizer_name:
            # Phase transition visualizer
            if 'phase' in field_name or 'transition' in field_name or 'grokking' in field_name:
                base_score += 0.4

        elif 'sparse' in visualizer_name or 'sparsity' in visualizer_name:
            # Sparsity visualizer
            if 'spars' in field_name or 'activation' in field_name or 'neuron' in field_name:
                base_score += 0.4

        # Cap the score at 1.0
        return min(1.0, base_score)

    def _calculate_visualizer_compatibility(self, visualizer, field, field_data):
        """Calculate a compatibility score between a visualizer and field data."""
        visualizer_name = visualizer.__class__.__name__.lower()
        field_name = field.lower()

        # Start with no score - must explicitly match to get points
        base_score = 0.0

        # Strong preference for exact category matching
        if 'circuit' in field_name and 'circuit' in visualizer_name:
            base_score += 0.8
        elif 'timeseries' in visualizer_name and ('training' in field_name or 'evaluation' in field_name):
            base_score += 0.8
        elif 'heatmap' in visualizer_name and 'attention' in field_name:
            base_score += 0.8
        elif 'phase' in visualizer_name and ('phase' in field_name or 'transition' in field_name):
            base_score += 0.8
        elif 'network' in visualizer_name and ('circuit' in field_name or 'interaction' in field_name):
            base_score += 0.8
        elif 'sparsity' in visualizer_name and ('sparsity' in field_name or 'mlp' in field_name):
            base_score += 0.8

        # Add secondary data structure matching (this is a fallback)
        if base_score < 0.5:
            # Then check the data structure
            if 'timeseries' in visualizer_name:
                # Check for time series data structure
                if isinstance(field_data, dict) and 'epoch' in field_data:
                    base_score += 0.5
                elif isinstance(field_data, (list, tuple)) and all(isinstance(x, (int, float)) for x in field_data[:5]):
                    base_score += 0.4

            # todo add checks for other data types
            # Add similar checks for other visualizer types
            # ...

        # Cap the score at 1.0
        return min(1.0, base_score)
