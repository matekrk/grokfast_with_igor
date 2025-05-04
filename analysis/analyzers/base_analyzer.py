# analyzers/base_analyzer.py
from pathlib import Path

import torch


class BaseAnalyzer:
    """Base class for all analysis components"""

    def __init__(self, model, save_dir, logger=None):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger if logger else (model.logger if hasattr(model, 'logger') else None)

        # Create analysis directory
        self.analysis_dir = self.save_dir / self._get_analysis_dir_name()
        self.analysis_dir.mkdir(exist_ok=True, parents=True)

        # Initialize analysis history
        self.analysis_history = []

    def _get_analysis_dir_name(self):
        """Return the name for this analyzer's directory"""
        # Default implementation uses class name
        return self.__class__.__name__.lower()

    def analyze(self, *args, **kwargs):
        """Main analysis method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

    def save_results(self, results, filename):
        """Save analysis results to disk"""
        import json
        import numpy as np

        # Helper function to make results JSON serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            else:
                return str(obj)

        # Create serializable copy of results
        serializable_results = make_serializable(results)

        # Save to file
        with open(self.analysis_dir / filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def log_metric(self, category, key, value):
        """Log a metric if logger is available"""
        if self.logger:
            self.logger.log_data(category, key, value)