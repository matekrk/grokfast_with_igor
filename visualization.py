import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec


# This is a standalone replacement section for the attention patterns visualization
# Replace the entire "Attention Patterns" section with this code

def create_attention_patterns_plot(fig, gs, row, col, model, eval_loader):
    """
    Create attention patterns visualization in a separate function
    to avoid variable scope issues.

    Parameters:
    -----------
    fig : matplotlib Figure
        The figure to add the subplot to
    gs : GridSpec
        The grid specification
    row, col : int
        The row and column in the grid for this subplot
    model : Decoder
        The model to visualize attention for
    eval_loader : DataLoader
        Evaluation data loader for input samples

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Create the subplot
        ax_attn = fig.add_subplot(gs[row, col])

        # Get sample input for visualization
        sample_input = next(iter(eval_loader))[0]
        if hasattr(model, 'device'):
            sample_input = sample_input.to(model.device)

        # Forward pass with attention storage
        _ = model(sample_input, store_attention=True)

        # Get attention patterns
        patterns = model.get_attention_patterns()

        if not patterns:
            ax_attn.text(0.5, 0.5, 'No attention patterns available',
                         ha='center', va='center', fontsize=12)
            return True

        # Create a composite visualization of attention patterns
        n_layers = model.num_layers
        n_heads = model.num_heads

        # Remove the original axis to make room for our grid
        ax_attn.remove()

        # Get the position for the title
        pos = gs[row, col].get_position(fig)

        # Import necessary module
        # from matplotlib.gridspec import GridSpecFromSubplotSpec

        # Create a new GridSpecFromSubplotSpec
        inner_gs = GridSpecFromSubplotSpec(
            n_layers, n_heads,
            subplot_spec=gs[row, col]
        )

        # Now create all the subplots in the grid
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                pattern_key = f'layer_{layer_idx}_head_{head_idx}'

                if pattern_key in patterns:
                    inner_ax = fig.add_subplot(inner_gs[layer_idx, head_idx])

                    # Plot attention heatmap
                    # Create mask for lower triangle only (including diagonal)
                    mask = np.triu(np.ones_like(patterns[pattern_key], dtype=bool), k=1)
                    sns.heatmap(patterns[pattern_key], cmap='viridis', mask=mask,
                                cbar=False, xticklabels=False, yticklabels=False, ax=inner_ax)

                    inner_ax.set_title(f'layer {layer_idx} :: head {head_idx}', fontsize=8)
                    inner_ax.set_xlabel('Key position', fontsize=8)
                    inner_ax.set_ylabel('Query position', fontsize=8)

        # Add overall title for attention patterns
        plt.figtext(
            pos.x0 + pos.width / 2,
            pos.y1 + 0.02,
            'Attention Patterns',
            ha='center', fontsize=12
        )

        return True

    except Exception as e:
        print(f"Error creating attention patterns plot: {str(e)}")
        return False

def visualize_model_analysis(
        model,
        epoch,
        eval_loader=None,
        include_metrics=None,  # List of metrics to include or None for all
        save_path=None,  # Path to save the figure or None to display
        fig_size=(20, 16),  # Figure size
        title_prefix=None,  # Custom title prefix
        logx=False,
):
    """
    Comprehensive visualization of model analysis metrics in a single figure.

    Parameters:
    -----------
    model : Decoder
        The transformer model with analysis capabilities
    epoch : int
        Current training epoch
    eval_loader : DataLoader, optional
        Evaluation data loader (required for some metrics)
    include_metrics : list, optional
        List of metrics to include. If None, include all available metrics.
        Possible values: 'attention', 'attribution', 'cross_attribution', 'entropy',
                         'weight_norms', 'accuracy', 'loss', 'grokking'
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed
    fig_size : tuple, optional
        Size of the figure (width, height)
    title_prefix : str, optional
        Prefix for the figure title

    Returns:
    --------
    fig : matplotlib.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # Set default metrics if None
    if include_metrics is None:
        include_metrics = ['attention', 'attribution', 'cross_attribution',
                           'entropy', 'weight_norms', 'accuracy', 'loss', 'grokking']

    # Check what data is available in the logger
    available_data = {}
    logger = model.logger

    # Get logs from logger for different categories
    if 'training' in logger.logs and 'evaluation' in logger.logs:
        available_data['training'] = logger.get_logs('training')
        available_data['evaluation'] = logger.get_logs('evaluation')

    if 'weight_norms' in logger.logs:
        available_data['weight_norms'] = logger.get_logs('weight_norms')

    if 'head_attribution' in logger.logs:
        available_data['head_attribution'] = logger.get_logs('head_attribution')

    if 'attention_entropy' in logger.logs:
        available_data['attention_entropy'] = logger.get_logs('attention_entropy')

    if 'grokking_phases' in logger.logs:
        available_data['grokking_phases'] = logger.get_logs('grokking_phases')

    # Count how many plots we need
    plot_count = 0
    metrics_plot_added = False

    for metric in include_metrics:
        # For accuracy and loss, only add one plot total since they share axes
        if metric in ['accuracy', 'loss'] and not metrics_plot_added and ('training' in available_data or 'evaluation' in available_data):
            plot_count += 1
            metrics_plot_added = True
        elif metric == 'weight_norms' and 'weight_norms' in available_data:
            plot_count += 1
        elif metric == 'attribution' and 'head_attribution' in available_data:
            plot_count += 1
        elif metric == 'cross_attribution' and eval_loader is not None:
            plot_count += 1
        elif metric == 'entropy' and 'attention_entropy' in available_data:
            plot_count += 1
        elif metric == 'attention' and eval_loader is not None:
            plot_count += 1
        elif metric == 'grokking' and 'grokking_phases' in available_data:
            # warning check first if the data is correct (not None)
            plot_count += 1

    # Ensure at least one plot
    plot_count = max(plot_count, 1)

    # Calculate grid layout
    if plot_count <= 3:
        rows, cols = plot_count, 1
    else:
        # For more than 3 plots, create a more balanced grid
        # Calculate the minimum number of rows needed
        rows = (plot_count + 1) // 2
        cols = min(2, plot_count)  # Use at most 2 columns

        # Ensure we have enough cells in the grid
        while rows * cols < plot_count:
            rows += 1

    # Create figure and grid
    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.27)

    # Track current position in grid
    current_plot = 0

    # 1. Accuracy and Loss Plot
    if ('accuracy' in include_metrics or 'loss' in include_metrics) and \
            ('training' in available_data or 'evaluation' in available_data):
        ax_metrics = fig.add_subplot(gs[current_plot // cols, current_plot % cols])
        current_plot += 1

        # Plot training metrics if available
        # warning the following approach toi putting 'train loss'/'eval loss' labels is probably better:
        """
        # Add y-axis label
        ax.set_ylabel('Variable Values')  # A common label for both
        # Add annotations for each line
        ax.annotate('First Variable', xy=(-0.15, 0.25), xycoords='axes fraction', 
            color='blue', rotation=90, va='center')
        ax.annotate('Second Variable', xy=(-0.15, 0.75), xycoords='axes fraction', 
            color='red', rotation=90, va='center')
        """
        if 'training' in available_data:
            train_data = pd.DataFrame(available_data['training'])
            if 'accuracy' in train_data and 'accuracy' in include_metrics:
                sns.lineplot(data=train_data, x='epoch', y='accuracy', color='blue',
                             label='Train Accuracy', ax=ax_metrics)

            if 'loss' in train_data and 'loss' in include_metrics:
                ax_loss = ax_metrics.twinx() if 'accuracy' in include_metrics else ax_metrics
                sns.lineplot(data=train_data, x='epoch', y='loss', color='red',
                             label='Train Loss', ax=ax_loss)
                ax_loss.set_ylabel('train loss', color='red')
                ax_loss.yaxis.set_label_coords(1.05, 0.25)
                ax_loss.tick_params(axis='y', labelcolor='red')

        # Plot evaluation metrics if available
        if 'evaluation' in available_data:
            eval_data = pd.DataFrame(available_data['evaluation'])
            if 'accuracy' in eval_data and 'accuracy' in include_metrics:
                sns.lineplot(data=eval_data, x='epoch', y='accuracy', color='green',
                             label='Eval Accuracy', ax=ax_metrics)

            if 'loss' in eval_data and 'loss' in include_metrics:
                ax_loss = ax_metrics.twinx() if 'accuracy' in include_metrics else ax_metrics
                sns.lineplot(data=eval_data, x='epoch', y='loss', color='orange',
                             label='Eval Loss', ax=ax_loss)
                ax_loss.set_ylabel('eval loss', color='orange')
                ax_loss.yaxis.get_label().set_color('orange')
                ax_loss.yaxis.set_label_coords(1.05, 0.75)
                ax_loss.tick_params(axis='y', labelcolor='orange')

        # Add all grokking points if available top the drawing
        if 'grokking_phases' in available_data and 'grokking_step' in available_data['grokking_phases']:
            grokking_step = available_data['grokking_phases']['grokking_step']
            try:
                if isinstance(grokking_step, list):
                    for step in grokking_step:
                        if step is not None and not (isinstance(step, float) and np.isnan(step)):
                            step = float(step)
                        ax_metrics.axvline(x=step, color='purple', linestyle='--')
            except (ValueError, TypeError):
                pass


        # Combine legends if we have both accuracy and loss
        if 'accuracy' in include_metrics and 'loss' in include_metrics:
            lines1, labels1 = ax_metrics.get_legend_handles_labels()
            lines2, labels2 = ax_loss.get_legend_handles_labels()
            ax_metrics.legend(lines1 + lines2, labels1 + labels2, loc='best')

        ax_metrics.set_title('Training and Evaluation Metrics')
        # Remove x-axis label when not at the bottom row
        if current_plot // cols < rows - 1:
            ax_metrics.set_xlabel('')
        else:
            ax_metrics.set_xlabel('Epoch')

        if 'accuracy' in include_metrics:
            ax_metrics.set_ylabel('Accuracy')

        if logx:
            ax_metrics.set_xscale('log')
            ax_loss.set_xscale('log')

    # 2. Weight Norms Plot
    if 'weight_norms' in include_metrics and 'weight_norms' in available_data:
        ax_norms = fig.add_subplot(gs[current_plot // cols, current_plot % cols])
        current_plot += 1

        norms_data = pd.DataFrame(available_data['weight_norms'])

        # Select columns for head norms
        head_cols = [col for col in norms_data.columns
                     if col.startswith('head_') and 'combined' in col]

        # Select columns for MLP norms
        mlp_cols = [col for col in norms_data.columns
                    if col.startswith('mlp_') and 'combined' in col]

        # Plot head norms
        if head_cols:
            norm_data_melted = pd.melt(norms_data, id_vars=['epoch'],
                                       value_vars=head_cols,
                                       var_name='Component', value_name='Norm')
            sns.lineplot(data=norm_data_melted, x='epoch', y='Norm',
                         hue='Component', palette='viridis', ax=ax_norms)

        # Plot MLP norms
        if mlp_cols:
            mlp_data_melted = pd.melt(norms_data, id_vars=['epoch'],
                                      value_vars=mlp_cols,
                                      var_name='Component', value_name='Norm')
            sns.lineplot(data=mlp_data_melted, x='epoch', y='Norm',
                         hue='Component', palette='rocket', ax=ax_norms,
                         linestyle='--')

        # Add all grokking points if available top the drawing
        if 'grokking_phases' in available_data and 'grokking_step' in available_data['grokking_phases']:
            grokking_step = available_data['grokking_phases']['grokking_step']
            try:
                if isinstance(grokking_step, list):
                    for step in grokking_step:
                        if step is not None and not (isinstance(step, float) and np.isnan(step)):
                            step = float(step)
                        ax_norms.axvline(x=step, color='purple', linestyle='--')
            except (ValueError, TypeError):
                pass

        ax_norms.set_title('Weight Norms Over Training')
        # Remove x-axis label when not at the bottom row
        if current_plot // cols < rows - 1:
            ax_norms.set_xlabel('')
        else:
            ax_norms.set_xlabel('Epoch')

        ax_norms.set_ylabel('Norm Value')

        # Adjust legend to be more compact
        if head_cols or mlp_cols:
            ax_norms.legend(fontsize='small', title_fontsize='small',
                            ncol=2)  #, loc='upper right')

    # 3. Head Attribution Plot
    if 'attribution' in include_metrics and 'head_attribution' in available_data:
        ax_attr = fig.add_subplot(gs[current_plot // cols, current_plot % cols])
        current_plot += 1

        attr_data = pd.DataFrame(available_data['head_attribution'])

        # Get the latest attribution scores
        latest_idx = attr_data['epoch'].idxmax()
        latest_attr = attr_data.iloc[latest_idx].drop('epoch')

        # Create a bar plot of attribution scores
        sns.barplot(x=latest_attr.index, y=latest_attr.values, ax=ax_attr)

        ax_attr.set_title(f'Head Attribution at Epoch {int(attr_data.iloc[latest_idx]["epoch"])}')
        ax_attr.set_xlabel('Head')
        ax_attr.set_ylabel('Attribution Score')
        ax_attr.tick_params(axis='x', rotation=45)

    # 4. Cross-Attribution Plot
    if 'cross_attribution' in include_metrics and eval_loader is not None:
        ax_cross = fig.add_subplot(gs[current_plot // cols, current_plot % cols])
        current_plot += 1

        # Generate cross-attribution data if not cached
        cross_attribution = model.analyze_head_cross_attribution(eval_loader)

        # Convert to DataFrame
        df = pd.DataFrame(cross_attribution)
        df.columns = ['layer_f', 'head_f', 'layer_s', 'head_s', 'cross_attn_score']
        df["first"] = df["layer_f"].astype(int).astype(str) + ':' + df["head_f"].astype(int).astype(str)
        df["second"] = df["layer_s"].astype(int).astype(str) + ':' + df["head_s"].astype(int).astype(str)
        df = df[["first", "second", "cross_attn_score"]]
        pivot_df = pd.pivot_table(df, values='cross_attn_score', index=['first'], columns=['second'])

        # Fill NaN values with mean
        mean_val = df["cross_attn_score"].mean()
        pivot_df = pivot_df.fillna(mean_val)

        # Create mask for lower triangle only (including diagonal)
        mask = np.triu(np.ones_like(pivot_df, dtype=bool), k=1)

        # Create heatmap with mask for upper triangle
        sns.heatmap(pivot_df, annot=True, fmt=".2f", square=True, cmap="mako",
                   mask=mask, ax=ax_cross)

        ax_cross.set_title(f'Cross-Attribution Scores at Epoch {epoch}')
        ax_cross.tick_params(axis='x', rotation=45)
        ax_cross.tick_params(axis='y', rotation=0)
        ax_cross.set_xlabel('')
        ax_cross.set_ylabel('')

    # 5. Entropy Plot
    if 'entropy' in include_metrics and 'attention_entropy' in available_data:
        ax_entropy = fig.add_subplot(gs[current_plot // cols, current_plot % cols])
        current_plot += 1

        entropy_data = pd.DataFrame(available_data['attention_entropy'])

        # Get the latest entropy values
        latest_idx = entropy_data['epoch'].idxmax()
        latest_entropy = entropy_data.iloc[latest_idx].drop('epoch')

        # Create a bar plot of entropy values
        sns.barplot(x=latest_entropy.index, y=latest_entropy.values, color='skyblue', ax=ax_entropy)

        ax_entropy.set_title(f'Attention Entropy at Epoch {int(entropy_data.iloc[latest_idx]["epoch"])}')
        ax_entropy.set_xlabel('Head')
        ax_entropy.set_ylabel('Entropy (lower = more specialized)')
        ax_entropy.tick_params(axis='x', rotation=45)

    # 6. Attention patterns as a call
    if 'attention' in include_metrics and eval_loader is not None and current_plot < rows * cols:
        row, col = current_plot // cols, current_plot % cols
        success = create_attention_patterns_plot(
            fig=fig,
            gs=gs,
            row=row,
            col=col,
            model=model,
            eval_loader=eval_loader
        )
        if success:
            current_plot += 1
    # 7. Grokking Analysis
    if 'grokking' in include_metrics and 'grokking_phases' in available_data:
        ax_grok = fig.add_subplot(gs[current_plot // cols, current_plot % cols])
        current_plot += 1

        # Get training and evaluation data
        if 'training' in available_data and 'evaluation' in available_data:
            train_data = pd.DataFrame(available_data['training'])
            eval_data = pd.DataFrame(available_data['evaluation'])

            # Plot accuracy curves
            if 'accuracy' in train_data.columns and 'accuracy' in eval_data.columns:
                sns.lineplot(data=train_data, x='epoch', y='accuracy',
                             color='blue', label='Train', ax=ax_grok)
                sns.lineplot(data=eval_data, x='epoch', y='accuracy',
                             color='green', label='Eval', ax=ax_grok)

            # Highlight grokking phases if available
            phases = available_data['grokking_phases']

            if 'pre_grokking_steps' in phases and len(phases['pre_grokking_steps']) > 0:
                ax_grok.axvspan(min(phases['pre_grokking_steps']), max(phases['pre_grokking_steps']),
                                alpha=0.15, color='red', label='Pre-Grokking')

            if 'transition_steps' in phases and len(phases['transition_steps']) > 0:
                ax_grok.axvspan(min(phases['transition_steps']), max(phases['transition_steps']),
                                alpha=0.2, color='yellow', label='Transition')

            if 'post_grokking_steps' in phases and len(phases['post_grokking_steps']) > 0:
                ax_grok.axvspan(min(phases['post_grokking_steps']), max(phases['post_grokking_steps']),
                                alpha=0.2, color='green', label='Post-Grokking')

            if 'grokking_step' in phases and phases['grokking_step'] is not None:
                try:
                    grokking_step = phases['grokking_step']
                    # If it's a list with one element, extract that element
                    if isinstance(grokking_step, list) and len(grokking_step) == 1:
                        grokking_step = grokking_step[0]

                    # Check if it's a valid numeric value
                    if grokking_step is not None and not (isinstance(grokking_step, float) and np.isnan(grokking_step)):
                        grokking_step_value = float(grokking_step)
                        ax_grok.axvline(x=grokking_step_value, color='r', linestyle='--',
                                       label=f'Grokking Point: {int(grokking_step_value)}')
                except (ValueError, TypeError):
                    print(f"Warning: Invalid grokking_step value in phases: {phases['grokking_step']} [line {530}]")

        ax_grok.set_title('Grokking Analysis')
        ax_grok.set_xlabel('Epoch')
        ax_grok.set_ylabel('Accuracy')
        ax_grok.legend(loc='best')

    # Add title to the whole figure
    if title_prefix:
        title = f"{title_prefix} - Model Analysis at Epoch {epoch}"
    else:
        title = f"Model Analysis at Epoch {epoch} - {model.id}"

    plt.suptitle(title, fontsize=14, y=0.98)

    # Adjust layout with reduced space for the title
    # plt.tight_layout(rect=[0, 0, 1, 0.97])  # Reduced from 0.97 to 0.95

    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_attributions(attribution, epoch, title):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    heads = list(attribution.keys())
    scores = [attribution[h] for h in heads]
    plt.bar(heads, scores)
    plt.xlabel('Attention Head')
    plt.ylabel('Attribution Score (decrease in accuracy when masked)')
    plt.title(f'Head Attribution at Step {epoch}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_cross_attributions(attribution, epoch, title):
    # numpy array to Dataframe
    df = pd.DataFrame(attribution)
    df.columns = ['layer_f', 'head_f', 'layer_s', 'head_s', 'cross_attn_score']
    df["first"] = df["layer_f"].astype(int).astype(str) + ':' + df["head_f"].astype(int).astype(str)
    df["second"] = df["layer_s"].astype(int).astype(str) + ':' + df["head_s"].astype(int).astype(str)
    df = df[["first", "second", "cross_attn_score"]]
    pivot_df = pd.pivot_table(df, values='cross_attn_score', index=['first'], columns=['second'])
    mean_val = df["cross_attn_score"].mean()
    pivot_df = pivot_df.fillna(mean_val)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", square=True, cmap="mako", ax=ax)
    plt.title(f'{title}')
    plt.tight_layout()
    plt.show()

def plot_entropy(entropies, epoch, title):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    heads = list(entropies.keys())
    entropy_vals = [entropies[h] for h in heads]
    plt.bar(heads, entropy_vals, color='skyblue')
    plt.xlabel('Attention Head')
    plt.ylabel('Entropy')
    plt.title(f'{title} at Step {epoch} (lower = more specialized)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
