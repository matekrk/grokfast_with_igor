# analysis/analyzers/grokking_detector.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import wasserstein_distance
from scipy.interpolate import interp1d
# from pathlib import Path

from analysis.analyzers.base_analyzer import BaseAnalyzer


class GrokkingDetector(BaseAnalyzer):
    """
    Detector for grokking transitions in transformer training

    This class implements multi-metric detection of grokking
    transitions, using validation accuracy, training-validation gap,
    weight norms, attention entropy, and attribution scores.
    """

    def _get_analysis_dir_name(self):
        return "grokking_detection"

    def detect_grokking(self, min_epoch=250, min_acc_threshold=0.2, plot=True, save_path=None):
        """
        Identify grokking transitions using multiple metrics

        Args:
            min_epoch: Minimum epoch to consider (avoid early instability)
            min_acc_threshold: Minimum validation accuracy to be considered grokking
            plot: Whether to generate visualizations
            save_path: Path to save visualizations, or None to use default

        Returns:
            dict: Grokking detection results including DataFrames for visualization
        """
        # Get data from logger
        if not self.logger:
            print("No logger available for grokking detection")
            return None

        evl_log = self.logger.get_logs("evaluation")
        trn_log = self.logger.get_logs("training")

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
            print(f"Not enough data points for reliable grokking detection (need at least 10, got {len(evl_df)})")
            return None

        # Also ensure validation accuracy has reached a minimum threshold
        if 'accuracy' in evl_df.columns:
            max_acc = evl_df['accuracy'].max()
            if max_acc < min_acc_threshold:
                print(f"Maximum validation accuracy ({max_acc:.3f}) is below threshold ({min_acc_threshold})")
                return None

        # Ensure we have enough data
        smoothing_window = min(11, len(evl_df) // 5) if len(evl_df) > 20 else 3
        if len(evl_df) < smoothing_window:
            print("Not enough data points for reliable grokking detection")
            return None

        # Create a metrics DataFrame for grokking detection analysis
        metrics_data = {
            'step': evl_df['epoch'].values
        }

        # 1. Validation Accuracy Gradient Analysis
        metrics_data['acc_gradient'] = self._compute_acc_gradient(evl_df, smoothing_window)

        # 2. Training vs. Validation Gap Analysis
        metrics_data['gap_gradient'] = self._compute_gap_gradient(trn_df, evl_df, smoothing_window)

        # 3. Weight Norm Analysis
        metrics_data['weight_norm_gradient'] = self._compute_weight_norm_gradient(smoothing_window)

        # 4. Attention Entropy Dynamics
        metrics_data['entropy_gradient'] = self._compute_entropy_gradient(smoothing_window)

        # 5. Attribution Score Transitions
        metrics_data['attribution_changes'] = self._compute_attribution_changes(smoothing_window)

        # Create metrics DataFrame
        metrics_df = pd.DataFrame(metrics_data)

        # Combine all metrics with appropriate weighting
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

        # Create a time weighting factor that increases with epoch
        # This makes later peaks more significant
        time_factor = np.linspace(0.5, 1.0, len(combined_signal))
        weighted_combined_signal = combined_signal * time_factor

        # Find peaks in the weighted combined signal
        peak_height = 0.4  # Threshold for significance
        min_distance = max(5, len(weighted_combined_signal) // 20)  # Minimum distance between peaks
        peak_indices = find_peaks(
            weighted_combined_signal,
            height=peak_height,
            distance=min_distance
        )[0]

        # Filter peaks using contextual validation
        valid_indices = []
        for idx in peak_indices:
            if self._is_real_grokking_peak(metrics_df, evl_df, idx, window_size=11):
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

        # Generate visualization if requested
        if plot:
            self._visualize_grokking_detection(result, save_path=save_path)

        # Log the detected grokking points
        if self.logger and 'primary_grokking_step' in result and result['primary_grokking_step'] is not None:
            phases = {
                'grokking_step': result['primary_grokking_step'],
                'all_grokking_steps': result.get('grokking_steps', [])
            }
            self.logger.log_data('grokking_phases', 'grokking_step', result['primary_grokking_step'])
            self.logger.log_data('grokking_phases', 'all_grokking_steps', result.get('grokking_steps', []))

        return result

    def _normalize(self, signal):
        """Normalize signal to [0,1] range"""
        if np.max(signal) == np.min(signal):
            return np.zeros_like(signal)
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    def _smooth_signal(self, signal, window_length=11, polyorder=3):
        """Apply smoothing to a signal"""
        if len(signal) < window_length or polyorder >= window_length:
            # Not enough data points, use simpler smoothing
            return np.convolve(signal, np.ones(min(len(signal), 3)) / min(len(signal), 3), mode='same')
        return savgol_filter(signal, window_length, polyorder)

    def _compute_acc_gradient(self, evl_df, smoothing_window):
        """Compute validation accuracy gradient"""
        # Calculate gradient of validation accuracy
        acc_gradient = np.gradient(evl_df['accuracy'].values)
        acc_gradient_smoothed = self._smooth_signal(acc_gradient, window_length=smoothing_window)
        return self._normalize(acc_gradient_smoothed)

    def _compute_gap_gradient(self, trn_df, evl_df, smoothing_window):
        """Compute training-validation gap gradient"""
        if 'accuracy' in trn_df.columns and len(trn_df) == len(evl_df):
            acc_gap = trn_df['accuracy'].values - evl_df['accuracy'].values
            # The gap closing (negative gradient) indicates grokking
            acc_gap_gradient = -np.gradient(acc_gap)
            acc_gap_smoothed = self._smooth_signal(acc_gap_gradient, window_length=smoothing_window)
            return self._normalize(acc_gap_smoothed)
        return np.zeros_like(evl_df['accuracy'].values)

    def _compute_weight_norm_gradient(self, smoothing_window):
        """Compute weight norm gradient"""
        if not self.logger or 'weight_norms' not in self.logger.logs:
            return np.zeros_like(self.logger.get_logs('evaluation')['accuracy'])

        weight_logs = self.logger.get_logs('weight_norms')
        if weight_logs and 'epoch' in weight_logs:
            # Create dataframe from weight logs
            weight_df = pd.DataFrame(weight_logs)
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
                eval_df = pd.DataFrame(self.logger.get_logs('evaluation'))
                if len(weight_df) != len(eval_df):
                    # Interpolate to match steps
                    if len(weight_df) > 1:
                        f = interp1d(weight_df['epoch'].values, avg_weight_gradient,
                                     bounds_error=False, fill_value='extrapolate')
                        weight_norm_signal = f(eval_df['epoch'].values)

                        # Smooth and normalize
                        weight_norm_signal = self._smooth_signal(weight_norm_signal, window_length=smoothing_window)
                        return self._normalize(weight_norm_signal)

                # If we get here, lengths match or interpolation failed
                weight_norm_signal = self._smooth_signal(avg_weight_gradient, window_length=smoothing_window)
                return self._normalize(weight_norm_signal)

        # Default to zeros if we couldn't compute
        return np.zeros_like(self.logger.get_logs('evaluation')['accuracy'])

    def _compute_entropy_gradient(self, smoothing_window):
        """Compute attention entropy gradient"""
        if not self.logger or 'attention_entropy' not in self.logger.logs:
            return np.zeros_like(self.logger.get_logs('evaluation')['accuracy'])

        entropy_logs = self.logger.get_logs('attention_entropy')
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
                eval_df = pd.DataFrame(self.logger.get_logs('evaluation'))
                if len(entropy_df) != len(eval_df):
                    if len(entropy_df) > 1:
                        f = interp1d(entropy_df['epoch'].values, avg_entropy_gradient,
                                     bounds_error=False, fill_value='extrapolate')
                        entropy_signal = f(eval_df['epoch'].values)

                        # Smooth and normalize
                        entropy_signal = self._smooth_signal(entropy_signal, window_length=smoothing_window)
                        return self._normalize(entropy_signal)

                # If we get here, lengths match or interpolation failed
                entropy_signal = self._smooth_signal(avg_entropy_gradient, window_length=smoothing_window)
                return self._normalize(entropy_signal)

        # Default to zeros if we couldn't compute
        return np.zeros_like(self.logger.get_logs('evaluation')['accuracy'])

    def _compute_attribution_changes(self, smoothing_window):
        """Compute head attribution changes"""
        if not self.logger or 'head_attribution' not in self.logger.logs:
            return np.zeros_like(self.logger.get_logs('evaluation')['accuracy'])

        attribution_logs = self.logger.get_logs('head_attribution')
        if attribution_logs and 'epoch' in attribution_logs:
            # Create dataframe for attribution data
            attribution_df = pd.DataFrame(attribution_logs)
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
                    eval_df = pd.DataFrame(self.logger.get_logs('evaluation'))
                    if len(attribution_df) != len(eval_df):
                        f = interp1d(
                            attribution_df['epoch'].values,
                            distances,
                            bounds_error=False,
                            fill_value='extrapolate'
                        )
                        attribution_signal = f(eval_df['epoch'].values)
                    else:
                        attribution_signal = distances

                    # Smooth and normalize
                    attribution_signal = self._smooth_signal(attribution_signal, window_length=smoothing_window)
                    return self._normalize(attribution_signal)

        # Default to zeros if we couldn't compute
        return np.zeros_like(self.logger.get_logs('evaluation')['accuracy'])

    def _is_real_grokking_peak(self, metrics_df, evl_df, peak_idx, window_size=10):
        """
        Check if a peak is likely to be a real grokking transition
        by examining the context around the peak.

        Args:
            metrics_df: DataFrame with metrics
            evl_df: DataFrame with evaluation data including accuracy
            peak_idx: Index of the potential peak
            window_size: Size of the window to check before and after peak

        Returns:
            bool: True if the peak passes validation criteria
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

    def _visualize_grokking_detection(self, detection_result, save_path=None):
        """
        Visualize grokking detection results

        Args:
            detection_result: Output from detect_grokking
            save_path: Path to save the figure, or None to use default
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
        if hasattr(self.model, 'id'):
            plt.suptitle(f"Grokking Analysis for {self.model.id}", fontsize=16, y=0.98)
        else:
            plt.suptitle("Multi-Metric Grokking Detection", fontsize=16, y=0.98)

        # Save or show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            # Use default path
            plt.savefig(self.analysis_dir / f"grokking_detection.png", bbox_inches='tight', dpi=300)

        plt.close(fig)

        return fig

    def track_metrics_for_grokking(self, epoch, train_loader, eval_loader):
        """
        Track metrics needed for grokking detection and store in model's logger

        Args:
            epoch: Current training epoch
            train_loader: Training data loader
            eval_loader: Evaluation data loader
        """
        if hasattr(self.model, 'current_epoch'):
            self.model.current_epoch = epoch

        # 1. Track weight norms
        head_norms = {}
        mlp_norms = {}

        for layer_idx, layer in enumerate(self.model.layers):
            # Get head norms
            layer_head_norms = layer.get_head_norms()
            for head_name, norm in layer_head_norms.items():
                head_norms[f'head_{layer_idx}_{head_name}'] = norm

            # Get MLP norms
            layer_mlp_norms = layer.get_mlp_norms()
            for mlp_name, norm in layer_mlp_norms.items():
                mlp_norms[f'mlp_{layer_idx}_{mlp_name}'] = norm

        # Log weight norms
        weight_norms_data = {'epoch': epoch, **head_norms, **mlp_norms}
        self.logger.log_data('weight_norms', weight_norms_data)

        # 2. Track head attribution
        attribution = self.model.analyze_head_attribution(eval_loader)
        attribution_data = {'epoch': epoch, **attribution}
        self.logger.log_data('head_attribution', attribution_data)

        # 3. Track attention entropy
        entropies = self.model.compute_attention_entropy(eval_loader)
        entropy_data = {'epoch': epoch, **entropies}
        self.logger.log_data('attention_entropy', entropy_data)

    def analyze_grokking_transitions(self, train_loader, eval_loader, save_path_prefix=""):
        """
        Comprehensive analysis of grokking transitions

        Args:
            train_loader: Training data loader
            eval_loader: Evaluation data loader
            save_path_prefix: Prefix for saved files

        Returns:
            dict: Analysis results
        """
        # 1. Detect grokking points
        detection_result = self.detect_grokking(plot=True)

        # Exit if no detection results
        if detection_result is None or 'primary_grokking_step' not in detection_result:
            print("No clear grokking transition detected")
            return None

        # 2. Analyze before, during, and after grokking
        if 'primary_grokking_step' in detection_result and detection_result['primary_grokking_step'] is not None:
            grokking_step = detection_result['primary_grokking_step']
            # Find the index of this step in the metrics dataframe
            grokking_indices = np.where(detection_result['metrics_df']['step'] == grokking_step)[0]

            # Make sure we found a matching index
            if len(grokking_indices) > 0:
                grokking_idx = grokking_indices[0]

                # Make the window have the same number of steps before and after the grokking point
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

                # Check if we already logged this grokking step
                if not self.logger.value_in_category_key('grokking_phases', 'grokking_step', grokking_step):
                    self.logger.update_category('grokking_phases', phases)

                # Create a DataFrame for visualization
                phase_df = pd.DataFrame({
                    'step': np.concatenate([pre_steps, during_steps, post_steps]),
                    'phase': np.concatenate([
                        ['pre_grokking'] * len(pre_steps),
                        ['transition'] * len(during_steps),
                        ['post_grokking'] * len(post_steps)
                    ])
                })

                detection_result['phase_df'] = phase_df

                # 3. Visualize head importance changes across phases
                self._visualize_head_attribution_phases(pre_steps, during_steps, post_steps, save_path_prefix)

            else:
                print(f"Warning: Grokking step {grokking_step} not found in metrics dataframe")
        else:
            print("No clear grokking transition detected for phase analysis")
            # Create empty phase information
            phases = {
                'pre_grokking_steps': [],
                'transition_steps': [],
                'post_grokking_steps': [],
                'grokking_step': None
            }
            self.logger.update_category('grokking_phases', phases)

        return detection_result

    def _visualize_head_attribution_phases(self, pre_steps, during_steps, post_steps, save_path_prefix=""):
        """
        Visualize head attribution changes across grokking phases

        Args:
            pre_steps: Steps in pre-grokking phase
            during_steps: Steps in transition phase
            post_steps: Steps in post-grokking phase
            save_path_prefix: Prefix for saved files
        """
        if not self.logger or 'head_attribution' not in self.logger.logs:
            return

        attr_logs = self.logger.get_logs('head_attribution')
        if not attr_logs:
            return

        # Convert to DataFrame
        attr_df = pd.DataFrame(attr_logs)

        # Filter for the relevant steps
        phase_steps = np.concatenate([pre_steps, during_steps, post_steps])
        filtered_attr = attr_df[attr_df['epoch'].isin(phase_steps)].copy()

        if filtered_attr.empty:
            return

        # Add phase information
        filtered_attr['phase'] = 'unknown'
        filtered_attr.loc[filtered_attr['epoch'].isin(pre_steps), 'phase'] = 'pre_grokking'
        filtered_attr.loc[filtered_attr['epoch'].isin(during_steps), 'phase'] = 'transition'
        filtered_attr.loc[filtered_attr['epoch'].isin(post_steps), 'phase'] = 'post_grokking'

        # Create a new figure for head attribution
        attr_fig, attr_ax = plt.subplots(figsize=(12, 8))

        # Get head columns
        head_cols = [col for col in attr_df.columns
                     if col.startswith('layer_') and 'head_' in col and col != 'epoch']

        if not head_cols:
            plt.close(attr_fig)
            return

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

        # Get model ID for title if available
        model_id = self.model.id if hasattr(self.model, 'id') else "model"

        # Get grokking epoch if available
        grokking_epoch = None
        if self.logger and 'grokking_phases' in self.logger.logs and 'grokking_step' in self.logger.logs[
            'grokking_phases']:
            grokking_epoch = self.logger.logs['grokking_phases']['grokking_step']

        title = f'Head Attribution by Grokking Phase for {model_id}'
        if grokking_epoch:
            title += f', grokking epoch {grokking_epoch}'

        attr_ax.set_title(title, fontsize=14)
        attr_ax.set_xlabel('Attention Head')
        attr_ax.set_ylabel('Attribution Score')
        attr_ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save the figure
        save_path = self.analysis_dir / f"{save_path_prefix}head_attribution_by_phase.png"
        plt.savefig(save_path)
        plt.close(attr_fig)

        return attr_melted