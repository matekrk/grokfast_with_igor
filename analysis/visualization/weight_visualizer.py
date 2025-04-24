# analysis/visualization/weight_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_weight_trajectory(weight_tracker, components=[0, 1], fig=None,
                           highlight_epochs=None, title=None):
    """
    Plot the model's trajectory in weight space

    Args:
        weight_tracker: WeightSpaceTracker instance
        components: Which PCA components to plot
        fig: Optional existing figure
        highlight_epochs: Optional dict mapping epoch types to lists of epochs to highlight
        title: Optional title for the plot

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        ax = fig.subplots(1, 1)

    # Plot the trajectory
    weight_tracker.plot_trajectory(components=components, ax=ax,
                                   highlight_epochs=highlight_epochs)

    # Set title if provided
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_weight_dynamics(weight_tracker, fig=None, smooth_window=5, title=None):
    """
    Plot velocity and acceleration over time

    Args:
        weight_tracker: WeightSpaceTracker instance
        fig: Optional existing figure
        smooth_window: Window size for smoothing
        title: Optional title for the plot

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        ax = fig.subplots(1, 1)

    # Plot dynamics
    weight_tracker.plot_dynamics(ax=ax, smooth_window=smooth_window)

    # Set title if provided
    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig