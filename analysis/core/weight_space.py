# analysis/core/weight_space.py
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging


class WeightSpaceTracker:
    """
    Basic tracker for model's trajectory in weight space

    This class provides functionality to track and analyze a model's
    trajectory in weight space during training.
    """

    def __init__(self, model, save_dir=None, pca_components=20):
        """
        Initialize the weight space tracker

        Args:
            model: The model to track
            save_dir: Directory to save results, or None
            pca_components: Number of PCA components to use for visualization
        """
        ###                                                               todo is in follow
        self.model = model                                              # fixme yes
        self.save_dir = Path(save_dir) if save_dir else None            # fixme yes
        self.save_dir.mkdir(exist_ok=True, parents=True)                # warning ^^^ no mkdir() there!

        self.pca_components = pca_components                            # fixme yes
        self.snapshots = []                                             # warning ??? todo weight_snapshots?
        self.epochs = []                                                # warning ???
        self.flattened_weights = []                                     # fixme yes
        self.pca = None                                                 # fixme yes
        self.pca_fitted = False                                         # fixme yes

    def take_snapshot(self, epoch, flatten=True):                       # fixme yes
        """
        Take a snapshot of the current model weights

        Args:
            epoch: Current training epoch
            flatten: Whether to also compute flattened representation

        Returns:
            dict: The snapshot data
        """
        # Get state dict
        state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        # Create snapshot
        snapshot = {
            'epoch': epoch,
            'state_dict': state_dict
        }

        # Append to list
        self.snapshots.append(snapshot)
        self.epochs.append(epoch)

        # Compute flattened weights if requested
        if flatten:
            flattened = self._flatten_weights(state_dict)
            self.flattened_weights.append(flattened)
            # Reset PCA since we have new data
            self.pca_fitted = False

        return snapshot

    def _flatten_weights(self, state_dict=None):
        """
        Flatten model weights into a single vector

        Args:
            state_dict: Optional state dict to flatten, or None for current model

        Returns:
            numpy.ndarray: Flattened weights
        """
        if state_dict is None:
            state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        # Extract weight matrices (skip biases, batch norm params, etc)
        flattened = []
        for name, param in state_dict.items():
            if 'weight' in name and len(param.shape) >= 2:  # Only weight matrices
                flattened.append(param.view(-1).numpy())

        return np.concatenate(flattened)

    def _ensure_pca_fitted(self):
        """Ensure PCA is fitted to the current weights"""
        if not self.pca_fitted and len(self.flattened_weights) >= 3:
            # Convert to numpy array
            weight_matrix = np.vstack(self.flattened_weights)

            # Fit PCA
            n_components = min(self.pca_components, weight_matrix.shape[0], weight_matrix.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(weight_matrix)
            self.pca_fitted = True

            # Log variance explained
            var_explained = self.pca.explained_variance_ratio_
            logging.info(f"PCA variance explained: {var_explained.sum():.2%}")

    def project_weights(self, weights=None, components=None):
        """
        Project weights onto PCA components

        Args:
            weights: Optional weights to project, or None for all tracked weights
            components: Optional subset of components to use

        Returns:
            numpy.ndarray: Projected weights
        """
        self._ensure_pca_fitted()

        if not self.pca_fitted:
            return None

        if weights is None:
            weights = self.flattened_weights
        elif not isinstance(weights, list):
            weights = [weights]

        # Project onto PCA components
        projected = self.pca.transform(np.vstack(weights))

        # Subset of components if requested
        if components is not None:
            projected = projected[:, components]

        return projected

    def get_trajectory(self, components=None):
        """
        Get the model's trajectory in PCA space

        Args:
            components: Optional subset of components to use

        Returns:
            tuple: (epochs, trajectory)
        """
        if not self.flattened_weights:
            return None, None

        # Ensure PCA is fitted
        self._ensure_pca_fitted()

        if not self.pca_fitted:
            return self.epochs, None

        # Project all weights
        trajectory = self.project_weights(components=components)

        return self.epochs, trajectory

    def plot_trajectory(self, components=[0, 1], ax=None, highlight_epochs=None):
        """
        Plot the model's trajectory in PCA space

        Args:
            components: Which PCA components to plot
            ax: Optional matplotlib axis to plot on
            highlight_epochs: Optional dict mapping epoch types to lists of epochs to highlight

        Returns:
            matplotlib.axes.Axes: The plot axes
        """
        epochs, trajectory = self.get_trajectory(components=components)

        if trajectory is None or len(trajectory) < 2:
            logging.warning("Not enough snapshots to plot trajectory")
            return None

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        # Plot the trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', alpha=0.6, markersize=5)

        # Add arrows to show direction
        for i in range(1, len(trajectory)):
            ax.arrow(
                trajectory[i - 1, 0], trajectory[i - 1, 1],
                (trajectory[i, 0] - trajectory[i - 1, 0]) * 0.9,
                (trajectory[i, 1] - trajectory[i - 1, 1]) * 0.9,
                head_width=0.01, head_length=0.02, fc='blue', ec='blue', alpha=0.6
            )

        # Highlight specific epochs if requested
        if highlight_epochs:
            for etype, ep_list in highlight_epochs.items():
                color = 'red' if etype == 'jumps' else 'green'
                marker = 'o'

                for e in ep_list:
                    if e in epochs:
                        idx = epochs.index(e)
                        ax.plot(trajectory[idx, 0], trajectory[idx, 1],
                                marker=marker, color=color, markersize=8)
                        ax.text(trajectory[idx, 0], trajectory[idx, 1], str(e),
                                fontsize=8, ha='center', va='bottom')

        # Add axis labels with variance explained
        var_explained = self.pca.explained_variance_ratio_
        ax.set_xlabel(f'PC{components[0] + 1} ({var_explained[components[0]]:.2%} var)')
        ax.set_ylabel(f'PC{components[1] + 1} ({var_explained[components[1]]:.2%} var)')
        ax.set_title('Weight Space Trajectory')

        return ax

    def compute_velocity(self, smooth_window=None):
        """
        Compute velocity (change in weights) between snapshots

        Args:
            smooth_window: Optional window size for smoothing

        Returns:
            tuple: (epochs, velocities)
        """
        if len(self.flattened_weights) < 2:
            return [], []

        # Compute velocities between consecutive snapshots
        velocities = []
        for i in range(1, len(self.flattened_weights)):
            v = self.flattened_weights[i] - self.flattened_weights[i - 1]
            velocities.append(np.linalg.norm(v))

        # Epochs for velocities (one fewer than snapshots)
        vel_epochs = self.epochs[1:]

        # Apply smoothing if requested
        if smooth_window and smooth_window > 1 and len(velocities) >= smooth_window:
            import pandas as pd
            velocities = pd.Series(velocities).rolling(window=smooth_window, min_periods=1).mean().values

        return vel_epochs, velocities

    def compute_acceleration(self, smooth_window=None):
        """
        Compute acceleration (change in velocity) between snapshots

        Args:
            smooth_window: Optional window size for smoothing

        Returns:
            tuple: (epochs, accelerations)
        """
        vel_epochs, velocities = self.compute_velocity()

        if len(velocities) < 2:
            return [], []

        # Compute accelerations
        accelerations = []
        for i in range(1, len(velocities)):
            a = velocities[i] - velocities[i - 1]
            accelerations.append(a)

        # Epochs for accelerations (two fewer than snapshots)
        acc_epochs = vel_epochs[1:]

        # Apply smoothing if requested
        if smooth_window and smooth_window > 1 and len(accelerations) >= smooth_window:
            import pandas as pd
            accelerations = pd.Series(accelerations).rolling(window=smooth_window, min_periods=1).mean().values

        return acc_epochs, accelerations

    def plot_dynamics(self, ax=None, smooth_window=5):
        """
        Plot velocity and acceleration over time

        Args:
            ax: Optional matplotlib axis to plot on
            smooth_window: Window size for smoothing

        Returns:
            matplotlib.axes.Axes: The plot axes
        """
        vel_epochs, velocities = self.compute_velocity(smooth_window)
        acc_epochs, accelerations = self.compute_acceleration(smooth_window)

        if not velocities:
            logging.warning("Not enough snapshots to plot dynamics")
            return None

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        # Plot velocity
        ax.plot(vel_epochs, velocities, 'b-', label='Velocity')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Velocity (Weight Change)', color='b')
        ax.tick_params(axis='y', labelcolor='b')

        # Plot acceleration on second y-axis if available
        if accelerations:
            ax2 = ax.twinx()
            ax2.plot(acc_epochs, accelerations, 'r-', label='Acceleration')
            ax2.set_ylabel('Acceleration (Velocity Change)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

        ax.set_title('Weight Space Dynamics')

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        if accelerations:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax.legend(loc='upper left')

        return ax

    def save_snapshot(self, snapshot, filename=None):
        """
        Save a snapshot to disk

        Args:
            snapshot: The snapshot to save
            filename: Optional filename, or None for automatic naming

        Returns:
            pathlib.Path: Path to the saved file
        """
        if self.save_dir is None:
            logging.warning("No save directory specified")
            return None

        if filename is None:
            filename = f"snapshot_epoch_{snapshot['epoch']}.pt"

        save_path = self.save_dir / filename
        torch.save(snapshot, save_path)

        return save_path

    def load_snapshot(self, epoch=None, filename=None):
        """
        Load a snapshot from disk

        Args:
            epoch: Optional epoch to load, or None for latest
            filename: Optional filename to load, or None for automatic naming

        Returns:
            dict: The loaded snapshot
        """
        if self.save_dir is None:
            logging.warning("No save directory specified")
            return None

        if filename is None:
            if epoch is None:
                # Find the latest snapshot
                snapshots = list(self.save_dir.glob("snapshot_epoch_*.pt"))
                if not snapshots:
                    logging.warning("No snapshots found")
                    return None

                # Sort by epoch (extracted from filename)
                snapshots.sort(key=lambda p: int(p.stem.split('_')[-1]))
                filename = snapshots[-1].name
            else:
                filename = f"snapshot_epoch_{epoch}.pt"

        load_path = self.save_dir / filename
        if not load_path.exists():
            logging.warning(f"Snapshot file not found: {load_path}")
            return None

        snapshot = torch.load(load_path)

        return snapshot

    def clear(self):
        """Clear all tracked data"""
        self.snapshots = []
        self.epochs = []
        self.flattened_weights = []
        self.pca = None
        self.pca_fitted = False