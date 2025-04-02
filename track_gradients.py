from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def analyze_gradient_flow(model, loss, optimizer):
    """
    Analyze gradient flow through the network.

    Args:
        model: The transformer model
        loss: The current loss value
        optimizer: The optimizer

    Returns:
        dict: Gradient statistics
    """
    # Calculate gradients but don't apply them yet
    loss.backward(retain_graph=True)

    gradient_stats = {}
    layer_grad_norms = []
    head_grad_norms = []
    mlp_grad_norms = []

    # Analyze gradients for each component
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            weight_norm = param.norm().item()
            grad_weight_ratio = grad_norm / weight_norm if weight_norm > 0 else 0

            gradient_stats[f"{name}_grad_norm"] = grad_norm
            gradient_stats[f"{name}_weight_norm"] = weight_norm
            gradient_stats[f"{name}_ratio"] = grad_weight_ratio

            # Categorize by component type
            if 'layers' in name:
                layer_idx = int(name.split('.')[1])

                if 'attn' in name:
                    if 'out_proj' in name and 'weight' in name:
                        head_grad_norms.append((layer_idx, grad_norm))
                elif 'mlp' in name and 'weight' in name:
                    mlp_grad_norms.append((layer_idx, grad_norm))

                if layer_idx >= len(layer_grad_norms):
                    layer_grad_norms.extend([0] * (layer_idx + 1 - len(layer_grad_norms)))
                layer_grad_norms[layer_idx] += grad_norm

    # Store aggregated stats
    gradient_stats['layer_grad_norms'] = layer_grad_norms
    gradient_stats['head_grad_norms'] = sorted(head_grad_norms)
    gradient_stats['mlp_grad_norms'] = sorted(mlp_grad_norms)

    # Calculate overall gradient norm
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm()
            total_grad_norm += param_norm.item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    gradient_stats['total_grad_norm'] = total_grad_norm

    # Clear gradients so we don't interfere with actual training
    optimizer.zero_grad()

    return gradient_stats


class WeightSpaceTracker:
    """Track model's trajectory in weight space to identify attraction/repulsion patterns"""

    def __init__(self, model, save_dir, pca_components=50, snapshot_freq=10):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.pca_components = pca_components
        self.snapshot_freq = snapshot_freq

        # Storage for weight snapshots and trajectories
        self.weight_snapshots = []
        self.weight_timestamps = []
        self.flattened_weights = []
        self.pca = None
        self.pca_fitted = False

    def take_snapshot(self, epoch):
        """Take a snapshot of the current model weights"""
        if epoch % self.snapshot_freq != 0:
            return

        # Flatten all weights into a single vector
        flattened = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:  # Only consider weight matrices, not biases
                flattened.append(param.detach().cpu().view(-1))

        flattened_vector = torch.cat(flattened).numpy()
        self.flattened_weights.append(flattened_vector)
        self.weight_timestamps.append(epoch)

        # Store only if significant change occurred
        if len(self.weight_snapshots) == 0 or np.linalg.norm(
                flattened_vector - self.flattened_weights[-2]) > 1e-6:
            self.weight_snapshots.append({
                'epoch': epoch,
                'state_dict': {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            })

    def analyze_trajectory(self):
        """Analyze the trajectory of weights in PCA space"""
        if len(self.flattened_weights) < 3:
            return None  # Need at least 3 points for meaningful analysis

        # Convert to numpy array
        weight_matrix = np.stack(self.flattened_weights)

        # Fit PCA on the weights
        from sklearn.decomposition import PCA
        n_components = min(self.pca_components, weight_matrix.shape[0], weight_matrix.shape[1])
        self.pca = PCA(n_components=n_components)
        projected = self.pca.fit_transform(weight_matrix)
        self.pca_fitted = True

        # Calculate trajectory statistics
        trajectory_stats = {
            'epochs': self.weight_timestamps,
            'pca_trajectory': projected,
            'explained_variance': self.pca.explained_variance_ratio_,
            'velocities': [],
            'accelerations': []
        }

        # Calculate velocities (first derivatives)
        for i in range(1, len(projected)):
            velocity = projected[i] - projected[i - 1]
            speed = np.linalg.norm(velocity)
            trajectory_stats['velocities'].append({
                'epoch': self.weight_timestamps[i],
                'vector': velocity,
                'magnitude': speed
            })

        # Calculate accelerations (second derivatives)
        for i in range(1, len(trajectory_stats['velocities'])):
            v_curr = trajectory_stats['velocities'][i]['vector']
            v_prev = trajectory_stats['velocities'][i - 1]['vector']
            acceleration = v_curr - v_prev
            acc_magnitude = np.linalg.norm(acceleration)

            # Calculate component parallel to velocity (tangential acceleration)
            v_norm = v_curr / np.linalg.norm(v_curr) if np.linalg.norm(v_curr) > 0 else 0
            parallel_component = np.dot(acceleration, v_norm) * v_norm

            # Calculate component perpendicular to velocity (normal acceleration)
            perpendicular_component = acceleration - parallel_component

            trajectory_stats['accelerations'].append({
                'epoch': self.weight_timestamps[i + 1],
                'vector': acceleration,
                'magnitude': acc_magnitude,
                'tangential': np.linalg.norm(parallel_component),
                'normal': np.linalg.norm(perpendicular_component)
            })

        return trajectory_stats

    def visualize_trajectory(self, selected_dims=[0, 1], highlight_epochs=None):
        """
        Visualize the weight trajectory in PCA space.

        Args:
            selected_dims: Which PCA dimensions to plot
            highlight_epochs: List of epochs to highlight as potential transition points

        Returns:
            Path to saved visualization
        """
        if not self.pca_fitted:
            self.analyze_trajectory()
            if not self.pca_fitted:
                return None

        import matplotlib.pyplot as plt

        # Get the trajectory in selected dimensions
        trajectory = self.pca.transform(np.stack(self.flattened_weights))

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot trajectory
        ax1.plot(trajectory[:, selected_dims[0]], trajectory[:, selected_dims[1]], 'o-', alpha=0.7)

        # Add arrows to show direction
        for i in range(1, len(trajectory)):
            ax1.arrow(
                trajectory[i - 1, selected_dims[0]], trajectory[i - 1, selected_dims[1]],
                (trajectory[i, selected_dims[0]] - trajectory[i - 1, selected_dims[0]]) * 0.9,
                (trajectory[i, selected_dims[1]] - trajectory[i - 1, selected_dims[1]]) * 0.9,
                head_width=0.01, head_length=0.02, fc='blue', ec='blue', alpha=0.7
            )

        # Highlight specific epochs
        min_epoch_to_mark = 20
        if highlight_epochs:
            if isinstance(highlight_epochs, list):
                for epoch in highlight_epochs:
                    if epoch in self.weight_timestamps and epoch >= min_epoch_to_mark:
                        idx = self.weight_timestamps.index(epoch)
                        ax1.plot(
                            trajectory[idx, selected_dims[0]],
                            trajectory[idx, selected_dims[1]],
                            'ro', markersize=10, alpha=0.7
                        )
                        ax1.text(
                            trajectory[idx, selected_dims[0]] + 0.02,
                            trajectory[idx, selected_dims[1]] + 0.02,
                            f'grok {epoch}', fontsize=9
                        )
            elif isinstance(highlight_epochs, dict):
                for key, val in highlight_epochs.items():
                    if key == 'grok':
                        color, off_x, off_y = 'ro', -0.01, -0.01
                    else:
                        color, off_x, off_y = 'go', -0.01, +0.01
                    for epoch in highlight_epochs[key]:
                        if epoch in self.weight_timestamps and epoch >= min_epoch_to_mark:
                            idx = self.weight_timestamps.index(epoch)
                            ax1.plot(
                                trajectory[idx, selected_dims[0]] + off_x,
                                trajectory[idx, selected_dims[1]] + off_y,
                                color, markersize=10, alpha=0.7
                            )
                            fstr = f'grok {epoch}' if key == 'grok' else f'{epoch}'
                            ax1.text(
                                trajectory[idx, selected_dims[0]] - off_x,
                                trajectory[idx, selected_dims[1]] - off_y,
                                fstr, fontsize=9
                            )

        ax1.set_xlabel(
            f'PCA Dimension {selected_dims[0] + 1} ({self.pca.explained_variance_ratio_[selected_dims[0]]:.2%} var)')
        ax1.set_ylabel(
            f'PCA Dimension {selected_dims[1] + 1} ({self.pca.explained_variance_ratio_[selected_dims[1]]:.2%} var)')
        ax1.set_title('Weight Space Trajectory')

        # Plot speed and acceleration magnitudes
        stats = self.analyze_trajectory()
        epochs = [v['epoch'] for v in stats['velocities']]
        speeds = [v['magnitude'] for v in stats['velocities']]

        acc_epochs = [a['epoch'] for a in stats['accelerations']]
        tangential = [a['tangential'] for a in stats['accelerations']]
        normal = [a['normal'] for a in stats['accelerations']]

        ax2.plot(epochs, speeds, 'b-', label='Speed')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Speed', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        ax3 = ax2.twinx()
        ax3.plot(acc_epochs, tangential, 'r-', label='Tangential Acc.')
        ax3.plot(acc_epochs, normal, 'g-', label='Normal Acc.')
        ax3.set_ylabel('Acceleration', color='r')
        ax3.tick_params(axis='y', labelcolor='r')

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax2.set_title('Weight Space Velocity and Acceleration')

        if highlight_epochs:
            if isinstance(highlight_epochs, list):
                for epoch in highlight_epochs:
                    if epoch in self.weight_timestamps and epoch >= min_epoch_to_mark:
                        idx = self.weight_timestamps.index(epoch)
                        ax1.plot(
                            trajectory[idx, selected_dims[0]],
                            trajectory[idx, selected_dims[1]],
                            'ro', markersize=10, alpha=0.7
                        )
                        ax1.text(
                            trajectory[idx, selected_dims[0]] + 0.02,
                            trajectory[idx, selected_dims[1]] + 0.02,
                            f'grok {epoch}', fontsize=9
                        )
            elif isinstance(highlight_epochs, dict):
                for key, val in highlight_epochs.items():
                    if key == 'grok':
                        color, off_x, off_y = 'ro', +0.02, -0.03
                    else:
                        color, off_x, off_y = 'go', +0.02, +0.03
                    for epoch in highlight_epochs[key]:
                        if epoch in self.weight_timestamps and epoch >= min_epoch_to_mark:
                            idx = self.weight_timestamps.index(epoch)
                            ax1.plot(
                                trajectory[idx, selected_dims[0]],
                                trajectory[idx, selected_dims[1]] + off_y,
                                color, markersize=10, alpha=0.7
                            )
                            fstr = f'grok {epoch}' if key == 'grok' else f'{epoch}'
                            ax1.text(
                                trajectory[idx, selected_dims[0]] + off_x,
                                trajectory[idx, selected_dims[1]] + off_y,
                                fstr, fontsize=9
                            )

        # Highlight grokking transitions if provided
        if highlight_epochs:
            if isinstance(highlight_epochs, list):
                for epoch in highlight_epochs:
                    ax2.axvline(x=epoch, color='purple', linestyle='--', alpha=0.5)
            elif isinstance(highlight_epochs, dict):
                for key, val in highlight_epochs.items():
                    color = 'purple' if key == 'grok' else 'green'
                    for epoch in highlight_epochs[key]:
                        ax2.axvline(x=epoch, color=color, linestyle='--', alpha=0.5)

        plt.tight_layout()
        save_path = self.save_dir / f'weight_trajectory_dims_x{selected_dims[0]}_y{selected_dims[1]}_{self.weight_timestamps[-1]}.png'
        plt.savefig(save_path)
        plt.close()

        return save_path


def analyze_loss_curvature(model, inputs, targets, criterion, step_size=0.1, n_steps=10):
    """
    Analyze the curvature of the loss landscape around the current weights.

    Args:
        model: The transformer model
        inputs, targets: A batch of data
        criterion: Loss function
        step_size: Size of steps in random directions
        n_steps: Number of steps to take

    Returns:
        dict: Curvature statistics
    """
    # Store original parameters
    original_params = [p.detach().clone() for p in model.parameters()]

    # Calculate loss at current position
    with torch.no_grad():
        outputs = model(inputs)
        base_loss = criterion(outputs, targets).item()

    # Generate two random directions
    direction1, direction2 = [], []
    for p in model.parameters():
        d1 = torch.randn_like(p)
        d2 = torch.randn_like(p)

        # Normalize
        d1 = d1 / d1.norm() * p.norm()
        d2 = d2 / d2.norm() * p.norm()

        direction1.append(d1)
        direction2.append(d2)

    # Sample the loss landscape
    steps = np.linspace(-step_size * n_steps, step_size * n_steps, 2 * n_steps + 1)
    landscape = np.zeros((len(steps), len(steps)))

    for i, alpha in enumerate(steps):
        for j, beta in enumerate(steps):
            # Update model parameters
            with torch.no_grad():
                for p, p0, d1, d2 in zip(model.parameters(), original_params, direction1, direction2):
                    p.copy_(p0 + alpha * d1 + beta * d2)

            # Calculate loss
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets).item()
                landscape[i, j] = loss

    # Restore original parameters
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), original_params):
            p.copy_(p0)

    # Calculate curvature statistics
    from scipy.linalg import eigh
    from scipy.ndimage import gaussian_filter

    # Smooth the landscape
    smoothed = gaussian_filter(landscape, sigma=1.0)

    # Finite difference Hessian at the center
    center_idx = n_steps
    dx = step_size

    # Central finite difference for second derivatives
    dxx = (smoothed[center_idx + 1, center_idx] - 2 * smoothed[center_idx, center_idx] + smoothed[
        center_idx - 1, center_idx]) / dx ** 2
    dyy = (smoothed[center_idx, center_idx + 1] - 2 * smoothed[center_idx, center_idx] + smoothed[
        center_idx, center_idx - 1]) / dx ** 2
    dxy = (smoothed[center_idx + 1, center_idx + 1] - smoothed[center_idx + 1, center_idx - 1] - smoothed[
        center_idx - 1, center_idx + 1] + smoothed[center_idx - 1, center_idx - 1]) / (4 * dx ** 2)

    # Construct Hessian
    hessian = np.array([[dxx, dxy], [dxy, dyy]])

    # Eigenvalues of Hessian
    eigvals, eigvecs = eigh(hessian)

    # Results
    curvature = {
        'base_loss': base_loss,
        'landscape': landscape,
        'steps': steps,
        'hessian': hessian,
        'eigenvalues': eigvals,
        'eigenvectors': eigvecs,
        'max_curvature': max(abs(eigvals)),
        'min_curvature': min(abs(eigvals)),
        'condition_number': max(abs(eigvals)) / (min(abs(eigvals)) + 1e-10)
    }

    return curvature


class CyclicBehaviorDetector:
    """Detect and analyze cyclic behavior in training metrics"""

    def __init__(self, window_size=100, min_cycles=2):
        self.window_size = window_size
        self.min_cycles = min_cycles
        self.metrics = defaultdict(list)
        self.cycle_info = {}

    def add_metric(self, name, value, epoch):
        """Add a metric measurement"""
        self.metrics[name].append((epoch, value))

        # Limit history length
        hist_len = 5
        if len(self.metrics[name]) > self.window_size * hist_len:
            self.metrics[name] = self.metrics[name][-self.window_size * hist_len:]

    def has_enough_data(self):
        if len(self.metrics[list(self.metrics.keys())[0]]) >= self.window_size:
            return True
        else:
            return False

    def detect_cycles(self):
        """Detect cycles in all tracked metrics"""
        results = {}

        for name, values in self.metrics.items():
            if len(values) < self.window_size:
                continue

            # Extract epochs and values
            epochs, metric_values = zip(*values)
            epochs = np.array(epochs)
            metric_values = np.array(metric_values)

            # Find peaks and troughs
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(metric_values)
            troughs, _ = find_peaks(-metric_values)

            if len(peaks) < self.min_cycles or len(troughs) < self.min_cycles:
                continue

            # Calculate cycle lengths between peaks
            peak_epochs = epochs[peaks]
            peak_intervals = np.diff(peak_epochs)

            # Calculate central tendency of intervals
            median_interval = np.median(peak_intervals)
            mean_interval = np.mean(peak_intervals)
            std_interval = np.std(peak_intervals)

            # Check if consistent cycle length
            is_consistent = std_interval / mean_interval < 0.25  # Less than 25% variation

            # Calculate cycle amplitude
            amplitudes = []
            for i in range(min(len(peaks), len(troughs))):
                if i < len(peaks) and i < len(troughs):
                    peak_val = metric_values[peaks[i]]
                    trough_val = metric_values[troughs[i]]
                    amplitudes.append(peak_val - trough_val)

            mean_amplitude = np.mean(amplitudes) if amplitudes else 0

            # Save cycle information
            results[name] = {
                'median_cycle_length': median_interval,
                'mean_cycle_length': mean_interval,
                'std_cycle_length': std_interval,
                'mean_amplitude': mean_amplitude,
                'is_consistent': is_consistent,
                'peak_epochs': peak_epochs.tolist(),
                'num_detected_cycles': len(peaks) - 1
            }

        self.cycle_info = results
        return results

    def visualize_cycles(self, save_dir):
        """Visualize detected cycles for all metrics"""
        if not self.cycle_info:
            self.detect_cycles()

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        for name, cycle_data in self.cycle_info.items():
            epochs, values = zip(*self.metrics[name])

            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 2, figure=fig)

            # Plot the metric values
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(epochs, values, 'b-')

            # Mark detected peaks
            if 'peak_epochs' in cycle_data:
                peak_indices = [epochs.index(e) for e in cycle_data['peak_epochs'] if e in epochs]
                peak_values = [values[i] for i in peak_indices]
                ax1.plot(cycle_data['peak_epochs'], peak_values, 'ro')

            ax1.set_title(f"{name} Over Time")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel(name)

            # Add cycle length text
            if 'median_cycle_length' in cycle_data:
                ax1.text(
                    0.02, 0.95,
                    f"Median cycle: {cycle_data['median_cycle_length']:.1f} epochs\n"
                    f"Mean cycle: {cycle_data['mean_cycle_length']:.1f} ± {cycle_data['std_cycle_length']:.1f} epochs\n"
                    f"Consistent: {'Yes' if cycle_data['is_consistent'] else 'No'}\n"
                    f"Mean amplitude: {cycle_data['mean_amplitude']:.4f}",
                    transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

            # Plot the FFT to visualize frequency components
            ax2 = fig.add_subplot(gs[1, 0])
            from scipy.fft import fft, fftfreq

            # Compute FFT
            N = len(values)
            yf = fft(values)
            xf = fftfreq(N, d=1)  # Assuming 1 epoch intervals

            # Only plot positive frequencies up to Nyquist frequency
            pos_mask = xf > 0
            xf = xf[pos_mask]
            yf = np.abs(yf[pos_mask])

            # Convert frequency to cycle length (in epochs)
            cycle_lengths = 1 / xf

            # Plot
            ax2.plot(cycle_lengths, yf)
            ax2.set_xlim([0, min(500, max(cycle_lengths))])  # Limit to reasonable range
            ax2.set_xlabel("Cycle Length (epochs)")
            ax2.set_ylabel("Magnitude")
            ax2.set_title("Frequency Analysis")

            # Add vertical line at detected cycle length
            if 'median_cycle_length' in cycle_data:
                ax2.axvline(x=cycle_data['median_cycle_length'], color='r', linestyle='--')

            # Plot autocorrelation to detect periodicity
            ax3 = fig.add_subplot(gs[1, 1])
            from statsmodels.tsa.stattools import acf

            # Compute autocorrelation
            lag_max = min(200, len(values) // 2)
            acf_values = acf(values, nlags=lag_max)

            # Plot
            ax3.plot(range(lag_max + 1), acf_values)
            ax3.set_xlabel("Lag (epochs)")
            ax3.set_ylabel("Autocorrelation")
            ax3.set_title("Autocorrelation Analysis")

            # Add vertical lines at multiples of detected cycle length
            if 'median_cycle_length' in cycle_data:
                cycle_len = int(round(cycle_data['median_cycle_length']))
                for i in range(1, 5):
                    if i * cycle_len <= lag_max:
                        ax3.axvline(x=i * cycle_len, color='r', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(save_dir / f"{name}_cycle_analysis_{epochs[-1]}.png")
            plt.close()

        return save_dir


