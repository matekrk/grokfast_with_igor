# analysis/core/metrics_tracker.py
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MetricsTracker:
    """
    Track and analyze training metrics over time

    This class provides functionality to track, aggregate, and visualize
    various metrics during model training, such as loss, accuracy,
    and custom metrics.
    """

    def __init__(self, id=None):
        """
        Initialize the metrics tracker

        Args:
            id: Optional identifier for this tracker
        """
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.id = id

    def add_metric(self, category, name, value, step=None, epoch=None):
        """
        Add a metric value to the tracker

        Args:
            category: Category of the metric (e.g., 'training', 'evaluation')
            name: Name of the metric (e.g., 'loss', 'accuracy')
            value: Value of the metric
            step: Optional training step
            epoch: Optional training epoch
        """
        self.metrics[category][name].append(value)

        # Track steps and epochs if provided
        if step is not None:
            if 'step' not in self.metrics[category]:
                self.metrics[category]['step'] = []

            # Make sure we have the same number of steps as values
            while len(self.metrics[category]['step']) < len(self.metrics[category][name]) - 1:
                self.metrics[category]['step'].append(None)

            self.metrics[category]['step'].append(step)

        if epoch is not None:
            if 'epoch' not in self.metrics[category]:
                self.metrics[category]['epoch'] = []

            # Make sure we have the same number of epochs as values
            while len(self.metrics[category]['epoch']) < len(self.metrics[category][name]) - 1:
                self.metrics[category]['epoch'].append(None)

            self.metrics[category]['epoch'].append(epoch)

    def get_metrics(self, category=None):
        """
        Get all tracked metrics or metrics for a specific category

        Args:
            category: Optional category to filter by

        Returns:
            dict: Dictionary of metrics
        """
        if category:
            return dict(self.metrics.get(category, {}))
        return {cat: dict(metrics) for cat, metrics in self.metrics.items()}

    def get_metric_values(self, category, name):
        """
        Get values for a specific metric

        Args:
            category: Category of the metric
            name: Name of the metric

        Returns:
            list: Values for the metric
        """
        return self.metrics.get(category, {}).get(name, [])

    def get_last_value(self, category, name):
        """
        Get the most recent value for a metric

        Args:
            category: Category of the metric
            name: Name of the metric

        Returns:
            The most recent value, or None if the metric doesn't exist
        """
        values = self.get_metric_values(category, name)
        return values[-1] if values else None

    def get_mean(self, category, name, window=None):
        """
        Get the mean of a metric over a window

        Args:
            category: Category of the metric
            name: Name of the metric
            window: Number of most recent values to consider, or None for all

        Returns:
            float: Mean value, or None if the metric doesn't exist
        """
        values = self.get_metric_values(category, name)
        if not values:
            return None

        if window is not None:
            values = values[-window:]

        return np.mean(values)

    def get_dataframe(self, category, pivot=False):
        """
        Convert metrics for a category to a pandas DataFrame

        Args:
            category: Category of metrics to convert
            pivot: If True, pivot the DataFrame to have epochs/steps as index

        Returns:
            pandas.DataFrame: DataFrame containing the metrics
        """
        if category not in self.metrics:
            return pd.DataFrame()

        data = {name: values.copy() for name, values in self.metrics[category].items()}

        # Ensure all lists have the same length
        max_len = max(len(values) for values in data.values())
        for name, values in data.items():
            if len(values) < max_len:
                data[name] = values + [None] * (max_len - len(values))

        df = pd.DataFrame(data)

        if pivot and ('epoch' in df.columns or 'step' in df.columns):
            # Use epoch or step as index
            index_col = 'epoch' if 'epoch' in df.columns else 'step'
            df = df.set_index(index_col)

        return df

    def plot_metric(self, category, name, ax=None, window=None, **kwargs):
        """
        Plot a metric over time

        Args:
            category: Category of the metric
            name: Name of the metric
            ax: Optional matplotlib axis to plot on
            window: Optional smoothing window size
            **kwargs: Additional arguments to pass to plot

        Returns:
            matplotlib.axes.Axes: The plot axes
        """
        values = self.get_metric_values(category, name)
        if not values:
            return None

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        x = None
        if 'epoch' in self.metrics[category]:
            x = self.metrics[category]['epoch']
        elif 'step' in self.metrics[category]:
            x = self.metrics[category]['step']

        # Apply smoothing if requested
        if window is not None and window > 1:
            values = pd.Series(values).rolling(window=window, min_periods=1).mean().values

        if x is not None and len(x) == len(values):
            ax.plot(x, values, **kwargs)
            ax.set_xlabel('Epoch' if 'epoch' in self.metrics[category] else 'Step')
        else:
            ax.plot(values, **kwargs)
            ax.set_xlabel('Update')

        ax.set_ylabel(name)
        ax.set_title(f'{name} over time ({category})')

        return ax

    def plot_metrics(self, category, names=None, fig=None, window=None):
        """
        Plot multiple metrics from a category

        Args:
            category: Category of metrics to plot
            names: List of metric names to plot, or None for all
            fig: Optional matplotlib figure to plot on
            window: Optional smoothing window size

        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        if category not in self.metrics:
            return None

        if names is None:
            # Exclude 'step' and 'epoch' from automatic plotting
            names = [name for name in self.metrics[category].keys()
                     if name not in ('step', 'epoch')]

        if not names:
            return None

        if fig is None:
            fig, axes = plt.subplots(len(names), 1, figsize=(10, 4 * len(names)), squeeze=False)
            axes = axes.flatten()
        else:
            axes = fig.subplots(len(names), 1, squeeze=False).flatten()

        for i, name in enumerate(names):
            self.plot_metric(category, name, ax=axes[i], window=window, label=name)
            axes[i].legend()

        fig.tight_layout()
        return fig

    def clear(self, category=None, name=None):
        """
        Clear metrics data

        Args:
            category: Optional category to clear, or None for all
            name: Optional metric name to clear, or None for all in the category
        """
        if category is None:
            # Clear all metrics
            self.metrics = defaultdict(lambda: defaultdict(list))
        elif name is None:
            # Clear all metrics in a category
            self.metrics[category] = defaultdict(list)
        elif category in self.metrics and name in self.metrics[category]:
            # Clear a specific metric
            self.metrics[category][name] = []