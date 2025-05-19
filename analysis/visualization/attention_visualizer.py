# analysis/visualization/attention_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_attention_patterns(model, input_data=None, fig=None):
    """
    Visualize attention patterns for a model

    Args:
        model: The model
        input_data: Optional input data to visualize attention for
        fig: Optional existing figure

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    if input_data is not None:
        # Run a forward pass to get attention patterns
        device = next(model.parameters()).device
        input_data = input_data.to(device)

        # Forward pass with attention storage
        _ = model(input_data, store_attention=True)

    # Get attention patterns
    patterns = model.get_attention_patterns()

    if not patterns:
        print("No attention patterns available")
        return None

    # Create a figure with a subplot for each layer and head
    n_layers = model.num_layers
    n_heads = model.num_heads

    if fig is None:
        fig, axes = plt.subplots(n_layers, n_heads,
                                 figsize=(n_heads * 3, n_layers * 3))
    else:
        axes = fig.subplots(n_layers, n_heads)

    # Handle case of single subplot
    if n_layers == 1 and n_heads == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = axes.reshape(1, -1)
    elif n_heads == 1:
        axes = axes.reshape(-1, 1)

    # Plot each attention pattern
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            pattern_key = f'layer_{layer_idx}_head_{head_idx}'

            if pattern_key in patterns:
                ax = axes[layer_idx, head_idx]

                pattern = patterns[pattern_key].cpu().numpy()

                # Create a mask for lower triangle only (including diagonal)
                mask = np.triu(np.ones_like(pattern, dtype=bool), k=1)

                sns.heatmap(pattern, cmap='viridis', mask=mask,
                            cbar=False, xticklabels=False, yticklabels=False, ax=ax)

                ax.set_title(f'L{layer_idx}, H{head_idx}')

    plt.suptitle('Attention Patterns', y=1.02)
    fig.tight_layout()
    return fig


def plot_attention_entropy(model, fig=None, sort_by_entropy=True):
    """
    Visualize attention entropy for each head

    Args:
        model: The model
        fig: Optional existing figure
        sort_by_entropy: Whether to sort heads by entropy

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    if not hasattr(model, 'logger') or 'attention_entropy' not in model.logger.logs:
        print("No attention entropy data available")
        return None

    entropy_data = model.logger.logs['attention_entropy']

    # Filter out epochs and other non-head columns
    head_cols = [col for col in entropy_data.keys()
                 if col.startswith('layer_') and 'head_' in col]

    if not head_cols:
        print("No head entropy data found")
        return None

    # Extract most recent entropy values
    head_entropies = {col: entropy_data[col][-1] for col in head_cols}

    # Create dataframe
    df = pd.DataFrame(list(head_entropies.items()), columns=['Head', 'Entropy'])

    # Sort if requested
    if sort_by_entropy:
        df = df.sort_values('Entropy')

    # Create figure if needed
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        ax = fig.subplots(1, 1)

    # Create bar chart
    sns.barplot(x='Head', y='Entropy', data=df, ax=ax)
    ax.set_title('Attention Head Entropy (lower = more specialized)')
    ax.set_xlabel('Head')
    ax.set_ylabel('Entropy')
    ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()
    return fig
