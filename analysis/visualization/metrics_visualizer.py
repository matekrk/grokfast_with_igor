# analysis/visualization/metrics_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_metrics(model, metrics=None, fig=None, logx=False,
                 include_training=True, include_evaluation=True):
    """
    Plot training and evaluation metrics

    Args:
        model: The model with attached logger
        metrics: List of metrics to plot, or None for defaults
        fig: Optional existing figure
        logx: Whether to use log scale on x axis
        include_training: Whether to include training metrics
        include_evaluation: Whether to include evaluation metrics

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    if not hasattr(model, 'logger'):
        print("Model has no logger")
        return None

    # Default metrics
    if metrics is None:
        metrics = ['accuracy', 'loss']

    # Determine how many subplots we need
    n_plots = len(metrics)

    # Create figure if not provided
    if fig is None:
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), squeeze=False)
        axes = axes.flatten()
    else:
        axes = fig.subplots(n_plots, 1, squeeze=False).flatten()

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Training data
        if include_training and 'training' in model.logger.logs and metric in model.logger.logs['training']:
            trn_data = model.logger.logs['training']
            ax.plot(trn_data['epoch'], trn_data[metric],
                    label=f'Training {metric}', color='blue', alpha=0.7)

        # Evaluation data
        if include_evaluation and 'evaluation' in model.logger.logs and metric in model.logger.logs['evaluation']:
            eval_data = model.logger.logs['evaluation']
            ax.plot(eval_data['epoch'], eval_data[metric],
                    label=f'Evaluation {metric}', color='orange')

        ax.set_title(f'{metric.capitalize()} over time')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()

        if logx:
            ax.set_xscale('log')

    fig.tight_layout()
    return fig


def plot_comparison(models, metric='accuracy', fig=None, dataset='evaluation', logx=False):
    """
    Compare the same metric across multiple models

    Args:
        models: List of models to compare
        metric: Metric to plot
        fig: Optional existing figure
        dataset: Which dataset to plot ('training' or 'evaluation')
        logx: Whether to use log scale on x axis

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    # Create figure if not provided
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax = fig.subplots(1, 1)

    # Plot each model
    for model in models:
        if not hasattr(model, 'logger') or dataset not in model.logger.logs:
            continue

        data = model.logger.logs[dataset]
        if metric in data:
            label = model.id if hasattr(model, 'id') else f"Model {id(model)}"
            ax.plot(data['epoch'], data[metric], label=label)

    ax.set_title(f'{metric.capitalize()} Comparison ({dataset})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.capitalize())
    ax.legend()

    if logx:
        ax.set_xscale('log')

    fig.tight_layout()
    return fig