import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import wasserstein_distance
from scipy.interpolate import interp1d


def normalize(signal):
    """Normalize signal to [0,1] range"""
    if np.max(signal) == np.min(signal):
        return np.zeros_like(signal)
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))


def smooth_signal(signal, window_length=11, polyorder=3):
    if len(signal) < window_length:
        # Not enough data points, use simpler smoothing
        return np.convolve(signal, np.ones(min(len(signal), 3)) / min(len(signal), 3), mode='same')
    return savgol_filter(signal, window_length, polyorder)


def detect_grokking_multi_metric(model, min_epoch=250, min_acc_threshold=0.2):
    """
    Identify grokking transitions using multiple metrics

    This implementation uses the DataLogger from the model
    and creates DataFrames for analysis and visualization.

    Parameters:
    -----------
    model : Decoder
        The transformer model with analysis capabilities

    Returns:
    --------
    dict
        Grokking detection results including DataFrames for visualization
    """
    # Get data from logger
    logger = model.logger
    evl_log = logger.get_logs("evaluation")
    trn_log = logger.get_logs("training")

    # Convert to DataFrames for easier manipulation
    evl_df = pd.DataFrame(evl_log)
    trn_df = pd.DataFrame(trn_log)

    # Filter out early epochs to avoid detecting transient initial instability
    if 'epoch' in evl_df.columns:
        evl_df = evl_df[evl_df['epoch'] >= min_epoch]
        # Apply the same filtering to training DataFrame to keep them aligned
        if 'epoch' in trn_df.columns:
            trn_df = trn_df[trn_df['epoch'] >= min_epoch]

    if len(evl_df) < 10:  # Need enough data points after filtering
        print(f"detect_grokking_multi_metric(): not enough data points after epoch {min_epoch} for reliable grokking detection")
        return None

    # Also ensure validation accuracy has reached a minimum threshold
    if 'accuracy' in evl_df.columns:
        max_acc = evl_df['accuracy'].max()
        if max_acc < min_acc_threshold:
            print(f"detect_grokking_multi_metric(): maximum validation accuracy ({max_acc:.3f}) is below threshold ({min_acc_threshold})")
            return None

    # Ensure we have enough data
    smoothing_window = min(11, len(evl_df) // 5) if len(evl_df) > 20 else 3
    if len(evl_df) < smoothing_window:
        print("Not enough data points for reliable grokking detection")
        return None

    # Create a metrics DataFrame
    metrics_data = {
        'step': evl_df['epoch'].values
    }


    # 1. Validation Accuracy Gradient Analysis
    #
    # The function calculates the gradient of validation accuracy using np.gradient().
    # This captures the rate of change in accuracy, which is a primary indicator of grokking.
    # The signal is smoothed using a Savitzky-Golay filter to reduce noise.
    # This metric is weighted highest (0.30) since it's typically the most reliable indicator.
    # used with 30% weight for the final score
    # -------------------------------
    acc_gradient = np.gradient(evl_df['accuracy'].values)
    acc_gradient_smoothed = smooth_signal(acc_gradient, window_length=smoothing_window)
    metrics_data['acc_gradient'] = normalize(acc_gradient_smoothed)

    # 2. Training vs. Validation Gap Analysis
    #
    # The implementation tracks the difference between training and validation accuracy.
    # More importantly, it calculates the rate at which this gap closes with np.gradient(acc_gap).
    # When the gap suddenly narrows, it's a strong signal of grokking.
    # This metric is weighted second highest (0.25) since it's a key grokking characteristic.
    # -----------------------------
    if 'accuracy' in trn_df.columns and len(trn_df) == len(evl_df):
        acc_gap = trn_df['accuracy'].values - evl_df['accuracy'].values
        # The gap closing (negative gradient) indicates grokking
        acc_gap_gradient = -np.gradient(acc_gap)
        acc_gap_smoothed = smooth_signal(acc_gap_gradient, window_length=smoothing_window)
        metrics_data['gap_gradient'] = normalize(acc_gap_smoothed)
    else:
        metrics_data['gap_gradient'] = np.zeros_like(metrics_data['acc_gradient'])

    # 3. Extract Weight Norms from Logger and perform Weight Norm Analysis
    #
    # The code tracks Frobenius norms of different model components (attention heads, MLP weights).
    # It calculates the rate of change in these norms over time.
    # Rapid changes in weight norms often indicate a transition in learning regime.
    # The track_weight_norms() function collects norms from each layer and head.
    # This metric has a 0.15 weight since it's more of a supplementary signal.
    # ----------------------------------
    weight_norm_signal = np.zeros_like(metrics_data['acc_gradient'])

    # Check if we have weight norm data in the logger
    if 'weight_norms' in logger.logs:
        weight_logs = logger.get_logs('weight_norms')
        if weight_logs and 'epoch' in weight_logs:
            # Create dataframe from weight logs
            weight_df = pd.DataFrame(weight_logs)
            # weight_norm_cols = [col for col in weight_df.columns
            #                     if col.endswith('_norm') and col != 'epoch']
            # Update column selection to match track_metrics_for_grokking() naming
            weight_norm_cols = [col for col in weight_df.columns
                                if ('head_' in col or 'mlp_' in col) and col != 'epoch']

            if weight_norm_cols:
                # Calculate the rate of change for each weight norm
                weight_gradients = []
                for col in weight_norm_cols:
                    grad = np.gradient(weight_df[col].values)
                    weight_gradients.append(np.abs(grad))  # Take absolute value

                # Average the gradients
                avg_weight_gradient = np.mean(weight_gradients, axis=0)

                # Match to evaluation steps if needed
                if len(weight_df) != len(evl_df):
                    # Interpolate to match steps
                    if len(weight_df) > 1:
                        f = interp1d(weight_df['epoch'].values, avg_weight_gradient,
                                     bounds_error=False, fill_value='extrapolate')
                        weight_norm_signal = f(metrics_data['step'])
                else:
                    weight_norm_signal = avg_weight_gradient

                # Smooth and normalize
                weight_norm_signal = smooth_signal(weight_norm_signal, window_length=smoothing_window)
                weight_norm_signal = normalize(weight_norm_signal)

    metrics_data['weight_norm_gradient'] = weight_norm_signal

    # 4.  Attention Entropy Dynamics
    #
    # The implementation tracks entropy of attention distributions for each head.
    # It focuses on the rate of decrease in entropy, which signals head specialization.
    # When heads rapidly become more specialized (lower entropy), it's often a grokking sign.
    # The metric is negated so that decreasing entropy (increasing specialization) produces positive peaks.
    # This has a weight of 0.15 as it's a strong indicator but can be noisy.
    # ----------------------------
    entropy_signal = np.zeros_like(metrics_data['acc_gradient'])

    # Check for entropy data in logger
    if 'attention_entropy' in logger.logs:
        entropy_logs = logger.get_logs('attention_entropy')
        if entropy_logs and 'epoch' in entropy_logs:
            # Create dataframe for entropy data
            entropy_df = pd.DataFrame(entropy_logs)
            # Update to match pattern from compute_attention_entropy()
            entropy_cols = [col for col in entropy_df.columns
                            if col.startswith('layer_') and 'head_' in col and col != 'epoch']

            if entropy_cols:
                # Calculate gradient of entropies (negative entropy indicates specialization)
                entropy_gradients = []
                for col in entropy_cols:
                    # Decreasing entropy indicates specialization
                    grad = -np.gradient(entropy_df[col].values)
                    entropy_gradients.append(grad)

                # Average the entropy gradients
                avg_entropy_gradient = np.mean(entropy_gradients, axis=0)

                # Match to evaluation steps if needed
                if len(entropy_df) != len(evl_df):
                    if len(entropy_df) > 1:
                        f = interp1d(entropy_df['epoch'].values, avg_entropy_gradient,
                                     bounds_error=False, fill_value='extrapolate')
                        entropy_signal = f(metrics_data['step'])
                else:
                    entropy_signal = avg_entropy_gradient

                # Smooth and normalize
                entropy_signal = smooth_signal(entropy_signal, window_length=smoothing_window)
                entropy_signal = normalize(entropy_signal)

    metrics_data['entropy_gradient'] = entropy_signal

    # 5. Attribution Score Transitions
    #
    # The code measures how the importance of different heads changes over time.
    # It uses Wasserstein distance to quantify changes in the attribution distribution.
    # Large distances indicate dramatic shifts in which heads are important to model performance.
    # These transitions often align with grokking points.
    # This metric has a weight of 0.15 as it's informative but can be computationally expensive.
    # -------------------------------
    attribution_signal = np.zeros_like(metrics_data['acc_gradient'])

    # Check for attribution data in logger
    if 'head_attribution' in logger.logs:
        attribution_logs = logger.get_logs('head_attribution')
        if attribution_logs and 'epoch' in attribution_logs:
            # Create dataframe for attribution data
            attribution_df = pd.DataFrame(attribution_logs)
            # attribution_cols = [col for col in attribution_df.columns
            #                     if 'head' in col and col != 'epoch']
            attribution_cols = [col for col in attribution_df.columns
                                if col.startswith('layer_') and 'head_' in col and col != 'epoch']

            if attribution_cols and len(attribution_df) > 1:
                # Pre-allocate distances array with zeros
                distances = np.zeros(len(attribution_df))
                valid_indices = []
                valid_distances = []

                # Calculate Wasserstein distances between consecutive attribution distributions
                for i in range(1, len(attribution_df)):
                    prev_dist = attribution_df.loc[i - 1, attribution_cols].values
                    curr_dist = attribution_df.loc[i, attribution_cols].values

                    # Normalize to create proper distributions
                    prev_total = np.sum(prev_dist)
                    curr_total = np.sum(curr_dist)

                    if prev_total > 0 and curr_total > 0:
                        prev_dist = prev_dist / prev_total
                        curr_dist = curr_dist / curr_total

                        dist = wasserstein_distance(prev_dist, curr_dist)
                        distances[i] = dist
                        valid_indices.append(i)
                        valid_distances.append(dist)

                # If we have at least one valid distance, fill in missing values
                if valid_distances:
                    # Use valid indices and distances to fill in missing values
                    if len(valid_indices) < len(attribution_df):
                        # We have missing values to interpolate
                        valid_epochs = attribution_df.iloc[valid_indices]['epoch'].values

                        if len(valid_epochs) > 1:  # Need at least 2 points for interpolation
                            # Create interpolation function from valid points
                            f = interp1d(
                                valid_epochs,
                                valid_distances,
                                bounds_error=False,
                                fill_value=(valid_distances[0], valid_distances[-1])  # Extrapolate with edge values
                            )

                            # Apply to all epochs
                            all_epochs = attribution_df['epoch'].values
                            distances = f(all_epochs)
                        else:
                            # Not enough valid points for interpolation, use the constant value
                            distances = np.ones(len(attribution_df)) * valid_distances[0]

                    # Match to evaluation steps if needed
                    if len(attribution_df) != len(evl_df):
                        f = interp1d(
                            attribution_df['epoch'].values,
                            distances,
                            bounds_error=False,
                            fill_value='extrapolate'
                        )
                        attribution_signal = f(metrics_data['step'])
                    else:
                        attribution_signal = distances

                    # Smooth and normalize
                    attribution_signal = smooth_signal(attribution_signal, window_length=smoothing_window)
                    attribution_signal = normalize(attribution_signal)

    metrics_data['attribution_changes'] = attribution_signal

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Combine all metrics with appropriate weighting
    #
    # Each metric is normalized to a [0,1] range to ensure fair comparison.
    # Metrics are weighted according to their reliability and smoothed to reduce noise.
    # The combined signal is analyzed using find_peaks() to identify potential grokking points.
    # Results include the primary grokking step plus any secondary transition points.
    # --------------------------------------------
    weights = {
        'acc_gradient': 0.30,
        'gap_gradient': 0.25,
        'weight_norm_gradient': 0.15,
        'entropy_gradient': 0.15,
        'attribution_changes': 0.15
    }

    # Calculate combined signal
    combined_signal = np.zeros_like(metrics_data['acc_gradient'])
    for metric, weight in weights.items():
        combined_signal += metrics_data[metric] * weight

    metrics_df['combined_signal'] = combined_signal

    # When finding peaks, add a time weighting factor that increases with epoch
    combined_signal = metrics_df['combined_signal'].values

    # Create a time weighting factor that increases with epoch
    # This will make later peaks more significant
    time_factor = np.linspace(0.5, 1.0, len(combined_signal))
    weighted_combined_signal = combined_signal * time_factor

    # Find peaks in the weighted combined signal
    peak_height = 0.4  # Threshold for significance
    min_distance = max(5, len(weighted_combined_signal) // 20)  # Minimum distance between peaks
    peak_indices = find_peaks(
        weighted_combined_signal,
        height=peak_height,
        distance=min_distance  # Enforce minimum distance between peaks
    )[0]

    # Filter peaks using contextual validation
    valid_indices = []
    for idx in peak_indices:
        if is_real_grokking_peak(metrics_df=metrics_df, evl_df=evl_df, peak_idx=idx, window_size=11):
            valid_indices.append(idx)

    peak_indices = np.array(valid_indices)

    # Prepare the result dictionary
    result = {
        'metrics_df': metrics_df,
        'weights': weights,
        'evl_df': evl_df,
        'trn_df': trn_df
    }

    if len(peak_indices) > 0:
        # Get and sort peaks by height
        peak_heights = combined_signal[peak_indices]
        sorted_indices = np.argsort(-peak_heights)  # Descending order
        peak_indices = peak_indices[sorted_indices]
        peak_heights = peak_heights[sorted_indices]

        # Get the top peaks
        grokking_steps = metrics_df['step'].values[peak_indices]
        result['grokking_steps'] = grokking_steps.tolist()
        result['peak_heights'] = peak_heights.tolist()
        result['primary_grokking_step'] = int(grokking_steps[0])

        # Create a DataFrame for grokking points
        grokking_points = pd.DataFrame({
            'step': grokking_steps,
        'height': peak_heights,
            'rank': range(1, len(grokking_steps) + 1)
        })
        result['grokking_points_df'] = grokking_points
    else:
        # No clear peaks found
        result['grokking_steps'] = []
        result['peak_heights'] = []
        result['primary_grokking_step'] = None
        result['grokking_points_df'] = pd.DataFrame(columns=['step', 'height', 'rank'])

    return result


def visualize_grokking_detection_seaborn(detection_result, model=None, save_path=None):
    """
    Visualize grokking detection results using seaborn and DataFrames

    Parameters:
    -----------
    detection_result : dict
        Output from detect_grokking_multi_metric
    model : Decoder, optional
        The transformer model for additional context
    save_path : str, optional
        Path to save the figure, if None, the figure is displayed

    Returns:
    --------
    fig : matplotlib.Figure
        The created figure
    """
    if detection_result is None:
        print("No detection results to visualize")
        return None

    # Extract DataFrames
    metrics_df = detection_result['metrics_df']
    evl_df = detection_result['evl_df']
    trn_df = detection_result['trn_df']
    weights = detection_result['weights']

    # Set up the figure with a grid spec
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

    # 1. Combined signal plot
    ax1 = fig.add_subplot(gs[0, 0])

    # Plot combined signal
    sns.lineplot(
        data=metrics_df, x='step', y='combined_signal',
        color='blue', linewidth=2, ax=ax1
    )

    # Add grokking points if any were detected
    if 'grokking_points_df' in detection_result and not detection_result['grokking_points_df'].empty:
        grokking_df = detection_result['grokking_points_df']

        # Add vertical lines for grokking points
        for i, row in grokking_df.iterrows():
            linestyle = '--' if row['rank'] == 1 else ':'
            linewidth = 2 if row['rank'] == 1 else 1
            label = f"Primary Grokking: Step {int(row['step'])}" if row['rank'] == 1 else None

            ax1.axvline(x=row['step'], color='red', linestyle=linestyle,
                        linewidth=linewidth, label=label)

    ax1.set_title('Combined Grokking Detection Signal', fontsize=14)
    ax1.set_ylabel('Signal Strength')
    # ax1.legend(loc='upper right')

    # 2. Component metrics plot
    ax2 = fig.add_subplot(gs[1, 0])

    # Melt the metrics DataFrame for seaborn
    metric_cols = [col for col in metrics_df.columns
                   if col not in ['step', 'combined_signal']]

    metrics_melted = pd.melt(
        metrics_df,
        id_vars=['step'],
        value_vars=metric_cols,
        var_name='Metric',
        value_name='Value'
    )

    # Add weight information to the metric names
    metrics_melted['Metric_with_weight'] = metrics_melted['Metric'].apply(
        lambda x: f"{x} (w={weights.get(x, 0)})"
    )

    # Plot all metrics
    sns.lineplot(
        data=metrics_melted,
        x='step',
        y='Value',
        hue='Metric_with_weight',
        palette='deep',
        ax=ax2
    )

    ax2.set_title('Component Metrics', fontsize=14)
    ax2.set_ylabel('Normalized Signal')

    # 3. Accuracy plot
    ax3 = fig.add_subplot(gs[2, 0])

    # Plot validation accuracy
    sns.lineplot(
        data=evl_df, x='epoch', y='accuracy',
        color='blue', label='Validation Accuracy', ax=ax3
    )

    # Add training accuracy if available
    if 'accuracy' in trn_df.columns:
        sns.lineplot(
            data=trn_df, x='epoch', y='accuracy',
            color='green', label='Training Accuracy', ax=ax3
        )

    # Add grokking points to accuracy plot too
    if 'grokking_points_df' in detection_result and not detection_result['grokking_points_df'].empty:
        for i, row in detection_result['grokking_points_df'].iterrows():
            ax3.axvline(x=row['step'], color='red', linestyle='--', linewidth=1)

    ax3.set_title('Model Accuracy', fontsize=14)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Accuracy')

    # Overall title
    if model is not None:
        plt.suptitle(f"Grokking Analysis for {model.id}", fontsize=16, y=0.98)
    else:
        plt.suptitle("Multi-Metric Grokking Detection", fontsize=16, y=0.98)

    # plt.tight_layout()
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def track_metrics_for_grokking(epoch, model, train_loader, eval_loader):
    """
    Track all metrics needed for grokking detection and store in model's logger

    Parameters:
    -----------
    model : Decoder
        The transformer model
    train_loader : DataLoader
        Training data loader
    eval_loader : DataLoader
        Evaluation data loader
    """
    model.current_epoch = epoch

    # 1. Track weight norms
    head_norms = {}
    mlp_norms = {}

    for layer_idx, layer in enumerate(model.layers):
        # Get head norms
        layer_head_norms = layer.get_head_norms()
        for head_name, norm in layer_head_norms.items():
            head_norms[f'head_{layer_idx}_{head_name}'] = norm

        # Get MLP norms
        layer_mlp_norms = layer.get_mlp_norms()
        for mlp_name, norm in layer_mlp_norms.items():
            mlp_norms[f'mlp_{layer_idx}_{mlp_name}'] = norm

    # Log weight norms
    weight_norms_data = {'epoch': model.current_epoch, **head_norms, **mlp_norms}
    model.log_stats('weight_norms', weight_norms_data)

    # 2. Track head attribution
    attribution = model.analyze_head_attribution(eval_loader)
    attribution_data = {'epoch': model.current_epoch, **attribution}
    model.log_stats('head_attribution', attribution_data)

    # 3. Track attention entropy
    entropies = model.compute_attention_entropy(eval_loader)
    entropy_data = {'epoch': model.current_epoch, **entropies}
    model.log_stats('attention_entropy', entropy_data)


def is_real_grokking_peak(metrics_df, evl_df, peak_idx, window_size=10):
    """
    Check if a peak is likely to be a real grokking transition
    by examining the context around the peak.

    Parameters:
    -----------
    metrics_df : DataFrame
        DataFrame with metrics
    evl_df : DataFrame
        DataFrame with evaluation data including accuracy
    peak_idx : int
        Index of the potential peak
    window_size : int
        Size of the window to check before and after peak

    Returns:
    --------
    bool
        True if the peak passes validation criteria
    """
    # Skip if we don't have enough context
    if peak_idx < window_size or peak_idx >= len(metrics_df) - window_size:
        return False

    # Get the step for this peak
    peak_step = metrics_df['step'].iloc[peak_idx]

    # Find surrounding steps in evl_df to check accuracy changes
    # Get closest steps before and after
    pre_steps = evl_df[evl_df['epoch'] < peak_step].tail(window_size)
    post_steps = evl_df[evl_df['epoch'] >= peak_step].head(window_size)

    # Make sure we have enough data
    if len(pre_steps) < window_size / 2 or len(post_steps) < window_size / 2:
        return False

    # Get pre/post peak accuracy
    pre_acc = pre_steps['accuracy'].mean()
    post_acc = post_steps['accuracy'].mean()

    # Check for significant accuracy improvement
    acc_improvement = post_acc - pre_acc
    if acc_improvement < pre_acc * 0.05:  # Require at least 5% improvement
        return False

    # Look for decreasing gap between train and val accuracy
    if 'gap_gradient' in metrics_df.columns:
        gap_value = metrics_df['gap_gradient'].iloc[peak_idx - window_size:peak_idx + window_size].mean()
        if gap_value < 0:  # Gap should be closing (positive gap_gradient)
            return False

    return True

def analyze_grokking_transitions(model, train_loader, eval_loader):
    """
    Comprehensive analysis of grokking transitions

    Parameters:
    -----------
    model : Decoder
        The transformer model
    train_loader : DataLoader
        Training data loader
    eval_loader : DataLoader
        Evaluation data loader

    Returns:
    --------
    dict
        Analysis results
    """
    # 1. Detect grokking points
    detection_result = detect_grokking_multi_metric(model)

    # Exit if no detection results
    if detection_result is None or 'primary_grokking_step' not in detection_result:
        print("\tNo clear grokking transition detected")
        return None

    # 2. Visualize detection
    fig = visualize_grokking_detection_seaborn(
        detection_result,
        model=model,
        save_path=f"{model.save_dir}/grokking_detection_{model.current_epoch}.png"
    )
    # plt.show()

    # 3. Analyze before, during, and after grokking
    if 'primary_grokking_step' in detection_result and detection_result['primary_grokking_step'] is not None:
        grokking_step = detection_result['primary_grokking_step']
        # Find the index of this step in the metrics dataframe
        grokking_indices = np.where(detection_result['metrics_df']['step'] == grokking_step)[0]

        # Make sure we found a matching index
        if len(grokking_indices) > 0:
            grokking_idx = grokking_indices[0]

        # todo make the window have the same number of setps before- and after- the grokking point
        window_size = min(len(detection_result['metrics_df']) // 10, 5)

        # Define time windows
        pre_start = max(0, grokking_idx - 2 * window_size)
        pre_end = max(0, grokking_idx - window_size)

        during_start = max(0, grokking_idx - window_size)
        during_end = min(len(detection_result['metrics_df']), grokking_idx + window_size)

        post_start = min(len(detection_result['metrics_df']), grokking_idx + window_size)
        post_end = min(len(detection_result['metrics_df']), grokking_idx + 2 * window_size)

        # Steps in each phase
        pre_steps = detection_result['metrics_df']['step'].iloc[pre_start:pre_end].values
        during_steps = detection_result['metrics_df']['step'].iloc[during_start:during_end].values
        post_steps = detection_result['metrics_df']['step'].iloc[post_start:post_end].values

        # Log phases to model logger
        phases = {
            'pre_grokking_steps': pre_steps.tolist(),
            'transition_steps': during_steps.tolist(),
            'post_grokking_steps': post_steps.tolist(),
            'grokking_step': grokking_step
        }

        if phases['grokking_step'] not in model.logger.logs['grokking_phases']['grokking_step']:
            # info log only if the value found for the grokking step is not already logged
            # todo prepeare a logger method that would do it on its own?
            model.log_stats('grokking_phases', phases)

        # Create a DataFrame for visualization
        # warning does it need to be added in case it was added previously (as above)?,
        #  perhaps a DataFrame would not add double lines?
        phase_df = pd.DataFrame({
            'step': np.concatenate([pre_steps, during_steps, post_steps]),
            'phase': np.concatenate([
                ['pre_grokking'] * len(pre_steps),
                ['transition'] * len(during_steps),
                ['post_grokking'] * len(post_steps)
            ])
        })

        detection_result['phase_df'] = phase_df

        # 4. Visualize head importance changes across phases
        if 'head_attribution' in model.logger.logs:
            attr_df = pd.DataFrame(model.logger.get_logs('head_attribution'))

            # Filter for the relevant steps
            phase_steps = np.concatenate([pre_steps, during_steps, post_steps])
            filtered_attr = attr_df[attr_df['epoch'].isin(phase_steps)].copy()

            # Add phase information
            filtered_attr['phase'] = 'unknown'
            filtered_attr.loc[filtered_attr['epoch'].isin(pre_steps), 'phase'] = 'pre_grokking'
            filtered_attr.loc[filtered_attr['epoch'].isin(during_steps), 'phase'] = 'transition'
            filtered_attr.loc[filtered_attr['epoch'].isin(post_steps), 'phase'] = 'post_grokking'

            # Create a new figure for head attribution
            attr_fig, attr_ax = plt.subplots(figsize=(12, 8))

            # Get head columns
            head_cols = [col for col in attr_df.columns if col.startswith('layer_') and col != 'epoch']

            # Melt for seaborn
            attr_melted = pd.melt(
                filtered_attr,
                id_vars=['epoch', 'phase'],
                value_vars=head_cols,
                var_name='Head',
                value_name='Attribution'
            )

            # Plot head attributions by phase
            sns.barplot(
                data=attr_melted,
                x='Head',
                y='Attribution',
                hue='phase',
                palette={'pre_grokking': 'blue', 'transition': 'orange', 'post_grokking': 'green'},
                ax=attr_ax
            )

            attr_ax.set_title(f'Head Attribution by Grokking Phase for model {model.plot_prefix}, grokking epoch {grokking_step}', fontsize=14)
            attr_ax.set_xlabel('Attention Head')
            attr_ax.set_ylabel('Attribution Score')
            attr_ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(f'{model.save_dir}/head_attribution_by_phase_{model.plot_prefix}_{model.current_epoch}.png')

            detection_result['attribution_by_phase'] = attr_melted
        else:
            print(f"Warning: Grokking step {grokking_step} not found in metrics dataframe")
    else:
        print("\tNo clear grokking transition detected for phase analysis")
        # Create empty phase information
        phases = {
            'pre_grokking_steps': [],
            'transition_steps': [],
            'post_grokking_steps': [],
            'grokking_step': None
        }
        if phases['grokking_step'] is not None:
            model.log_stats('grokking_phases', phases)

    return detection_result
