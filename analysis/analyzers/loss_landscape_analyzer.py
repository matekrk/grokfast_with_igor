# analyzers/loss_landscape_analyzer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy.linalg import eigh
from pathlib import Path

from analysis.analyzers.base_analyzer import BaseAnalyzer


class LossLandscapeAnalyzer(BaseAnalyzer):
    """Analyzer for loss landscape curvature and properties"""

    def _get_analysis_dir_name(self):
        return "loss_landscape"

    def analyze_loss_curvature(self, inputs, targets, criterion,
                               step_size=0.1, n_steps=10, n_directions=2):
        """
        Analyze the curvature of the loss landscape around current weights

        Args:
            inputs, targets: A batch of data
            criterion: Loss function
            step_size: Size of steps in random directions
            n_steps: Number of steps to take
            n_directions: Number of random directions to sample

        Returns:
            dict: Curvature statistics
        """
        # Store original parameters
        original_params = [p.detach().clone() for p in self.model.parameters()]

        # Calculate loss at current position
        with torch.no_grad():
            outputs = self.model(inputs)
            base_loss = criterion(outputs, targets).item()

        # Generate random orthogonal directions
        directions = []
        for _ in range(n_directions):
            direction = []
            for p in self.model.parameters():
                d = torch.randn_like(p)
                # Normalize
                d = d / d.norm() * p.norm() if p.norm() > 0 else d
                direction.append(d)
            directions.append(direction)

        # Sample the loss landscape for each pair of directions
        landscapes = []

        for i in range(0, n_directions, 2):
            if i + 1 < n_directions:  # Make sure we have a pair
                direction1 = directions[i]
                direction2 = directions[i + 1]

                # Sample grid of points
                steps = np.linspace(-step_size * n_steps, step_size * n_steps, 2 * n_steps + 1)
                landscape = np.zeros((len(steps), len(steps)))

                for j, alpha in enumerate(steps):
                    for k, beta in enumerate(steps):
                        # Update model parameters
                        with torch.no_grad():
                            for p, p0, d1, d2 in zip(self.model.parameters(), original_params, direction1, direction2):
                                p.copy_(p0 + alpha * d1 + beta * d2)

                        # Calculate loss
                        with torch.no_grad():
                            outputs = self.model(inputs)
                            loss = criterion(outputs, targets).item()
                            landscape[j, k] = loss

                # Store the landscape
                landscapes.append({
                    'landscape': landscape,
                    'steps': steps,
                    'direction1': direction1,
                    'direction2': direction2
                })

        # Restore original parameters
        with torch.no_grad():
            for p, p0 in zip(self.model.parameters(), original_params):
                p.copy_(p0)

        # Calculate curvature statistics from the first landscape
        if landscapes:
            # Smooth the landscape
            smoothed = gaussian_filter(landscapes[0]['landscape'], sigma=1.0)

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
        else:
            # Default values if no landscape was computed
            hessian = np.array([[0, 0], [0, 0]])
            eigvals = np.array([0, 0])
            eigvecs = np.array([[1, 0], [0, 1]])
            smoothed = np.zeros((2 * n_steps + 1, 2 * n_steps + 1))

        # Store in history
        analysis_result = {
            'base_loss': base_loss,
            'landscapes': landscapes,
            'hessian': hessian.tolist(),
            'eigenvalues': eigvals.tolist(),
            'eigenvectors': eigvecs.tolist(),
            'max_curvature': max(abs(eigvals)) if len(eigvals) > 0 else 0,
            'min_curvature': min(abs(eigvals)) if len(eigvals) > 0 else 0,
            'condition_number': max(abs(eigvals)) / (min(abs(eigvals)) + 1e-10) if len(eigvals) > 0 else 0,
            'smoothed_landscape': smoothed.tolist() if isinstance(smoothed, np.ndarray) else smoothed
        }

        self.analysis_history.append(analysis_result)

        # Log key metrics
        if self.logger:
            self.logger.log_data('loss_landscape', 'max_curvature', analysis_result['max_curvature'])
            self.logger.log_data('loss_landscape', 'condition_number', analysis_result['condition_number'])

        return analysis_result

    def visualize_landscape(self, analysis_result, name, epoch):
        """Create visualization of the loss landscape"""
        if not analysis_result.get('landscapes', []):
            return

        for i, landscape_data in enumerate(analysis_result['landscapes']):
            landscape = landscape_data['landscape']
            steps = landscape_data['steps']

            fig, ax = plt.subplots(figsize=(10, 8))

            # Create a heatmap of the loss landscape
            im = ax.imshow(landscape, cmap='viridis',
                           extent=[steps[0], steps[-1], steps[0], steps[-1]],
                           origin='lower', aspect='auto')

            # Add contour lines
            contours = ax.contour(steps, steps, landscape, colors='white', alpha=0.8)
            ax.clabel(contours, inline=True, fontsize=8)

            # Mark the center point
            ax.plot(0, 0, 'ro', markersize=8)

            # Add a colorbar
            plt.colorbar(im, ax=ax)

            # Add eigenvalue information
            if 'eigenvalues' in analysis_result:
                eigvals = analysis_result['eigenvalues']
                ax.text(0.05, 0.95,
                        f"λ1 = {eigvals[0]:.4f}\nλ2 = {eigvals[1]:.4f}\nCond. = {analysis_result['condition_number']:.4f}",
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Use formatted epoch for title
            epoch_str = f"{epoch:.1f}" if isinstance(epoch, float) else f"{epoch}"
            ax.set_title(f'Loss Landscape at Epoch {epoch_str} (Direction {i + 1})')
            ax.set_xlabel('Direction 1')
            ax.set_ylabel('Direction 2')

            plt.tight_layout()
            plt.savefig(self.analysis_dir / f"{name}_landscape_{i}.png")
            plt.close(fig)

    def analyze_jump(self, jump_epoch, pre_jump_snapshot, jump_snapshot, post_jump_snapshot,
                     inputs, targets, criterion):
        """Analyze loss landscape before, during, and after a jump"""
        jump_id = f"jump_{jump_epoch}"

        # Prepare the results structure
        results = {
            'jump_epoch': jump_epoch,
            'pre_jump_epoch': pre_jump_snapshot['epoch'],
            'jump_epoch': jump_snapshot['epoch'],
            'post_jump_epoch': post_jump_snapshot['epoch'],
            'landscape_analysis': {}
        }

        # Store original model state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        try:
            # Analyze each state (pre-jump, jump, post-jump)
            for state_name, snapshot in [
                ('pre_jump', pre_jump_snapshot),
                ('jump', jump_snapshot),
                ('post_jump', post_jump_snapshot)
            ]:
                # Load the state
                self.model.load_state_dict(snapshot['state_dict'])

                # Analyze curvature
                curvature = self.analyze_loss_curvature(
                    inputs=inputs,
                    targets=targets,
                    criterion=criterion
                )

                results['landscape_analysis'][state_name] = curvature

                # Create visualization
                self.visualize_landscape(
                    curvature,
                    f"{jump_id}_{state_name}",
                    snapshot['epoch']
                )

            # Create comparative visualization
            self._visualize_landscape_comparison(results, jump_id)

        finally:
            # Restore original model state
            self.model.load_state_dict(original_state)

        # Save the results
        if self.logger:
            self.logger.log_data('loss_landscape', f'jump_{jump_epoch}_eigenvalues',
                                 results['landscape_analysis']['jump']['eigenvalues'])
            self.logger.log_data('loss_landscape', f'jump_{jump_epoch}_condition_number',
                                 results['landscape_analysis']['jump']['condition_number'])

        return results

    def _visualize_landscape_comparison(self, results, jump_id):
        """Create a comparative visualization of landscapes before/during/after jump"""
        # Extract eigenvalues
        before_eigs = results['landscape_analysis']['pre_jump']['eigenvalues']
        jump_eigs = results['landscape_analysis']['jump']['eigenvalues']
        after_eigs = results['landscape_analysis']['post_jump']['eigenvalues']

        # Plot eigenvalue comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot eigenvalues
        epochs = [results['pre_jump_epoch'], results['jump_epoch'], results['post_jump_epoch']]
        # Format the epoch values for display
        epoch_labels = [f"{e:.1f}" if isinstance(e, float) else f"{e}" for e in epochs]

        eig1 = [before_eigs[0], jump_eigs[0], after_eigs[0]]
        eig2 = [before_eigs[1], jump_eigs[1], after_eigs[1]]

        ax1.plot(range(len(epochs)), eig1, 'o-', label='λ1')
        ax1.plot(range(len(epochs)), eig2, 'o-', label='λ2')
        ax1.axvline(x=1, color='r', linestyle='--')  # Mark the jump point
        ax1.set_xticks(range(len(epochs)))
        ax1.set_xticklabels(epoch_labels)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Hessian Eigenvalues Around Jump')
        ax1.legend()

        # Plot condition number
        conds = [
            results['landscape_analysis']['pre_jump']['condition_number'],
            results['landscape_analysis']['jump']['condition_number'],
            results['landscape_analysis']['post_jump']['condition_number']
        ]

        ax2.plot(range(len(epochs)), conds, 'o-')
        ax2.axvline(x=1, color='r', linestyle='--')  # Mark the jump point
        ax2.set_xticks(range(len(epochs)))
        ax2.set_xticklabels(epoch_labels)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Condition Number')
        ax2.set_title('Loss Landscape Condition Number')

        # Add labels clarifying the states
        for i, state in enumerate(['Before Jump', 'Jump Point', 'After Jump']):
            ax1.annotate(state, xy=(i, min(eig1[i], eig2[i])),
                         xytext=(i, min(eig1[i], eig2[i]) - 0.001),
                         ha='center', rotation=90, fontsize=8)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / f"{jump_id}_eigenvalue_comparison.png")
        plt.close(fig)

        # Combine landscapes in a single figure
        if all('landscapes' in results['landscape_analysis'][state] and
               results['landscape_analysis'][state]['landscapes']
               for state in ['pre_jump', 'jump', 'post_jump']):

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            for i, (state, ax) in enumerate(zip(
                    ['pre_jump', 'jump', 'post_jump'],
                    axes
            )):
                if 'smoothed_landscape' in results['landscape_analysis'][state]:
                    landscape = np.array(results['landscape_analysis'][state]['smoothed_landscape'])
                    im = ax.imshow(landscape, cmap='viridis', origin='lower', aspect='auto')

                    # Use formatted epoch for title
                    epoch = epochs[i]
                    epoch_str = f"{epoch:.1f}" if isinstance(epoch, float) else f"{epoch}"
                    state_labels = ['Before Jump', 'At Jump', 'After Jump']
                    ax.set_title(f"{state_labels[i]} (Epoch {epoch_str})")

                    plt.colorbar(im, ax=ax)

            plt.tight_layout()
            plt.savefig(self.analysis_dir / f"{jump_id}_landscape_comparison.png")
            plt.close(fig)