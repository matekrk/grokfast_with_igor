# analysis/visualization/model_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path


def visualize_model_analysis(model, epoch, eval_loader=None,
                             include_metrics=None, save_path=None, logx=False):
    """
    Create a comprehensive visualization of model analysis

    Args:
        model: The model to visualize
        epoch: Current training epoch
        eval_loader: Optional evaluation data loader
        include_metrics: List of metrics to include, or None for all
        save_path: Path to save the figure, or None to display
        logx: Whether to use log scale on x axis

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    if include_metrics is None:
        include_metrics = ['accuracy', 'loss', 'attention', 'attribution',
                           'weight_norms', 'entropy']

    # Determine number of subplots needed
    n_plots = sum([
        'accuracy' in include_metrics or 'loss' in include_metrics,
        'attention' in include_metrics,
        'attribution' in include_metrics or 'cross_attribution' in include_metrics,
        'entropy' in include_metrics,
        'weight_norms' in include_metrics,
        'grokking_phases' in include_metrics
    ])

    # Create figure
    fig = plt.figure(figsize=(12, 6 * n_plots))
    plt.subplots_adjust(hspace=0.4)

    # Current subplot index
    subplot_idx = 1

    # 1. Accuracy and Loss
    if 'accuracy' in include_metrics or 'loss' in include_metrics:
        metrics = []
        if 'accuracy' in include_metrics:
            metrics.append('accuracy')
        if 'loss' in include_metrics:
            metrics.append('loss')

        ax = fig.add_subplot(n_plots, 1, subplot_idx)
        subplot_idx += 1

        # Plot metrics
        if 'training' in model.logger.logs and 'evaluation' in model.logger.logs:
            for metric in metrics:
                if metric in model.logger.logs['training'] and metric in model.logger.logs['evaluation']:
                    trn_data = model.logger.logs['training']
                    ax.plot(trn_data['epoch'], trn_data[metric],
                            label=f'Training {metric}', alpha=0.6)

                    eval_data = model.logger.logs['evaluation']
                    ax.plot(eval_data['epoch'], eval_data[metric],
                            label=f'Evaluation {metric}')

        ax.set_title('Training and Evaluation Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()

        if logx:
            ax.set_xscale('log')

    # 2. Attention Patterns
    if 'attention' in include_metrics and eval_loader is not None:
        ax = fig.add_subplot(n_plots, 1, subplot_idx)
        subplot_idx += 1

        # Get a sample input
        sample_input = next(iter(eval_loader))[0]

        # Forward pass with attention storage
        _ = model(sample_input, store_attention=True)

        # Get attention patterns
        patterns = model.get_attention_patterns()

        if patterns:
            # Pick a representative head (e.g., layer 0, head 0)
            pattern_key = f'layer_0_head_0'
            if pattern_key in patterns:
                pattern = patterns[pattern_key].cpu().numpy()

                # Create a mask for lower triangle only (including diagonal)
                mask = np.triu(np.ones_like(pattern, dtype=bool), k=1)

                sns.heatmap(pattern, cmap='viridis', mask=mask, ax=ax)
                ax.set_title(f'Attention Pattern - Layer 0, Head 0 (Epoch {epoch})')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')

    # 3. Head Attribution
    if ('attribution' in include_metrics or 'cross_attribution' in include_metrics) and eval_loader is not None:
        ax = fig.add_subplot(n_plots, 1, subplot_idx)
        subplot_idx += 1

        # Get head attribution
        attribution = model.analyze_head_attribution(eval_loader)

        if attribution:
            # Convert to DataFrame for plotting
            attr_data = []
            for head, score in attribution.items():
                attr_data.append({
                    'Head': head,
                    'Attribution': score
                })

            attr_df = pd.DataFrame(attr_data)
            attr_df = attr_df.sort_values('Attribution', ascending=False)

            # Plot attribution
            sns.barplot(x='Head', y='Attribution', data=attr_df, ax=ax)
            ax.set_title(f'Head Attribution (Epoch {epoch})')
            ax.set_xlabel('Head')
            ax.set_ylabel('Attribution Score')
            ax.tick_params(axis='x', rotation=45)

        # Add cross-attribution if requested
        if 'cross_attribution' in include_metrics:
            # Cross-attribution would be plotted here if available
            pass

    # 4. Attention Entropy
    if 'entropy' in include_metrics and 'attention_entropy' in model.logger.logs:
        ax = fig.add_subplot(n_plots, 1, subplot_idx)
        subplot_idx += 1

        # Extract entropy data
        entropy_data = model.logger.logs['attention_entropy']

        # Filter out epochs and other non-head columns
        head_cols = [col for col in entropy_data.keys()
                     if col.startswith('layer_') and 'head_' in col]

        if head_cols:
            # Extract most recent entropy values
            head_entropies = {col: entropy_data[col][-1] for col in head_cols}

            # Create dataframe
            df = pd.DataFrame(list(head_entropies.items()), columns=['Head', 'Entropy'])
            df = df.sort_values('Entropy')

            # Plot entropy
            sns.barplot(x='Head', y='Entropy', data=df, ax=ax)
            ax.set_title(f'Attention Head Entropy (Epoch {epoch})')
            ax.set_xlabel('Head')
            ax.set_ylabel('Entropy (lower = more specialized)')
            ax.tick_params(axis='x', rotation=45)

    # 5. Weight Norms
    if 'weight_norms' in include_metrics and 'weight_norms' in model.logger.logs:
        ax = fig.add_subplot(n_plots, 1, subplot_idx)
        subplot_idx += 1

        # Extract weight norm data
        weight_data = model.logger.logs['weight_norms']

        # Filter interesting norms (e.g., head combined norms)
        head_norm_cols = [col for col in weight_data.keys()
                          if 'head_' in col and 'combined' in col]

        if head_norm_cols and 'epoch' in weight_data:
            # Plot head norms over time
            for col in head_norm_cols:
                ax.plot(weight_data['epoch'], weight_data[col], label=col)

            ax.set_title('Head Weight Norms Over Time')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Norm')
            ax.legend()

            if logx:
                ax.set_xscale('log')

    # 6. Grokking Phases
    if 'grokking_phases' in include_metrics and 'grokking_phases' in model.logger.logs:
        ax = fig.add_subplot(n_plots, 1, subplot_idx)
        subplot_idx += 1

        # Plot evaluation accuracy
        if 'evaluation' in model.logger.logs and 'accuracy' in model.logger.logs['evaluation']:
            eval_data = model.logger.logs['evaluation']
            ax.plot(eval_data['epoch'], eval_data['accuracy'],
                    label='Evaluation accuracy', color='blue')

        # Add training accuracy if available
        if 'training' in model.logger.logs and 'accuracy' in model.logger.logs['training']:
            trn_data = model.logger.logs['training']
            ax.plot(trn_data['epoch'], trn_data['accuracy'],
                    label='Training accuracy', color='green', alpha=0.6)

        # Add grokking point if available
        grokking_phases = model.logger.logs['grokking_phases']
        if 'grokking_step' in grokking_phases and grokking_phases['grokking_step']:
            grokking_step = grokking_phases['grokking_step']
            ax.axvline(x=grokking_step, color='red', linestyle='--',
                       label=f'Grokking: Epoch {grokking_step}')

        ax.set_title('Accuracy and Grokking Phases')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()

        if logx:
            ax.set_xscale('log')

    # Add a title for the whole figure
    if hasattr(model, 'id'):
        fig.suptitle(f"Model Analysis: {model.id} (Epoch {epoch})", fontsize=16, y=0.99)
    else:
        fig.suptitle(f"Model Analysis (Epoch {epoch})", fontsize=16, y=0.99)

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    return fig