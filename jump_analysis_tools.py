import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
# from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.linalg import eigh


class JumpAnalysisTools:
    """Analysis tools specifically for studying model behavior around jump events"""

    def __init__(self, model, save_dir, logger=None):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger if logger else model.logger if hasattr(model, 'logger') else None

        # Create analysis directories
        self.loss_landscape_dir = self.save_dir / "loss_landscape"
        self.loss_landscape_dir.mkdir(exist_ok=True, parents=True)

        self.attribution_dir = self.save_dir / "attribution_analysis"
        self.attribution_dir.mkdir(exist_ok=True, parents=True)

        self.attention_dir = self.save_dir / "attention_analysis"
        self.attention_dir.mkdir(exist_ok=True, parents=True)

        # Initialize analysis storage
        self.loss_curvature_history = []
        self.attribution_history = []
        self.attention_entropy_history = []

    def analyze_loss_landscape_around_jump(self, jump_epoch, inputs, targets, criterion,
                                           weight_tracker, step_size=0.1, n_steps=10, n_directions=4,
                                           ensure_after_snapshot=True):
        """
        Analyze loss landscape around a detected jump

        Args:
            jump_epoch: The epoch of the jump
            inputs, targets: A batch of data for loss calculation
            criterion: Loss function
            weight_tracker: EnhancedWeightSpaceTracker instance with snapshots
            step_size: Size of steps in random directions
            n_steps: Number of steps to take in each direction
            n_directions: Number of random directions to sample
            ensure_after_snapshot: If True, creates a new snapshot for after state if needed

        Returns:
            dict: Loss landscape analysis around the jump
        """
        jump_id = f"jump_{jump_epoch}"

        # # info find the jump index in the weight_timestamps
        jump_idx = None
        for i, epoch in enumerate(weight_tracker.weight_timestamps):
            if epoch == jump_epoch:
                jump_idx = i
                break

        if jump_idx is None:
            print(f"No snapshot found for jump at epoch {jump_epoch}")
            return None

        # info get snapshots before, at, and after the jump
        before_idx = max(0, jump_idx - 1)

        # info check if we have a valid "after" snapshot
        after_idx = min(len(weight_tracker.weight_timestamps) - 1, jump_idx + 1)
        after_epoch = weight_tracker.weight_timestamps[after_idx]

        # info if after_idx points to the same epoch as jump_idx, we need to create a new snapshot
        if after_epoch == jump_epoch and ensure_after_snapshot:
            print(f"Creating additional 'after jump' snapshot for epoch {jump_epoch}")

            # info store current model state
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            # info create a new snapshot labeled as "after jump"
            after_epoch = jump_epoch + 0.1  # Use fraction to indicate it's right after jump
            weight_tracker.take_snapshot(epoch=after_epoch, force=True)

            # info find the new snapshot
            after_idx = len(weight_tracker.weight_timestamps) - 1

            # info restore current state
            self.model.load_state_dict(current_state)

        before_snapshot = weight_tracker.weight_snapshots[before_idx]
        jump_snapshot = weight_tracker.weight_snapshots[jump_idx]
        after_snapshot = weight_tracker.weight_snapshots[after_idx]
        # info then analyzing a jump
        before_epoch = before_snapshot['epoch']
        jump_epoch = jump_snapshot['epoch']
        after_epoch = after_snapshot['epoch']
        print(f"Landscape: analyzing jump at {jump_epoch} with before={before_epoch}, after={after_epoch}")

        # info analyze the loss landscape around each of these points
        results = {
            'jump_epoch': jump_epoch,
            'before_epoch': before_snapshot['epoch'],
            'jump_epoch': jump_snapshot['epoch'],
            'after_epoch': after_snapshot['epoch'],
            'landscape_analysis': {}
        }

        # info store original model state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        try:
            # info analyze each state (before, jump, after)
            for state_name, snapshot in [
                ('before', before_snapshot),
                ('jump', jump_snapshot),
                ('after', after_snapshot)
            ]:
                # info log which snapshot we're analyzing
                print(f"Analyzing {state_name} state at epoch {snapshot['epoch']}")

                # info load the state
                self.model.load_state_dict(snapshot['state_dict'])

                # info analyze curvature at this point
                curvature = self.analyze_loss_curvature(
                    inputs=inputs,
                    targets=targets,
                    criterion=criterion,
                    step_size=step_size,
                    n_steps=n_steps,
                    n_directions=n_directions
                )

                results['landscape_analysis'][state_name] = curvature

                # info create visualization
                self._visualize_loss_landscape(
                    curvature,
                    f"{jump_id}_{state_name}",
                    snapshot['epoch']
                )

            # info create comparative visualization
            self._visualize_landscape_comparison(results, jump_id)

        finally:
            # info restore original model state
            self.model.load_state_dict(original_state)

        # info save the results
        if self.logger:
            self.logger.log_data('loss_landscape', f'jump_{jump_epoch}_eigenvalues',
                                 results['landscape_analysis']['jump']['eigenvalues'].tolist())
            self.logger.log_data('loss_landscape', f'jump_{jump_epoch}_condition_number',
                                 results['landscape_analysis']['jump']['condition_number'])

        return results

    def analyze_loss_curvature(self, inputs, targets, criterion, step_size=0.1, n_steps=10, n_directions=2):
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
        # info store original parameters
        original_params = [p.detach().clone() for p in self.model.parameters()]

        # info calculate loss at current position
        with torch.no_grad():
            outputs = self.model(inputs)
            base_loss = criterion(outputs, targets).item()

        # info generate random orthogonal directions
        directions = []
        for _ in range(n_directions):
            direction = []
            for p in self.model.parameters():
                d = torch.randn_like(p)
                # Normalize
                d = d / d.norm() * p.norm() if p.norm() > 0 else d
                direction.append(d)
            directions.append(direction)

        # info sample the loss landscape for each pair of directions
        landscapes = []

        for i in range(0, n_directions, 2):
            if i + 1 < n_directions:  # Make sure we have a pair
                direction1 = directions[i]
                direction2 = directions[i + 1]

                # info sample grid of points
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

                # info store the landscape
                landscapes.append({
                    'landscape': landscape,
                    'steps': steps,
                    'direction1': direction1,
                    'direction2': direction2
                })

        # info restore original parameters
        with torch.no_grad():
            for p, p0 in zip(self.model.parameters(), original_params):
                p.copy_(p0)

        # info calculate curvature statistics from the first landscape
        if landscapes:
            # info smooth the landscape
            smoothed = gaussian_filter(landscapes[0]['landscape'], sigma=1.0)

            # info finite difference Hessian at the center
            center_idx = n_steps
            dx = step_size

            # info central finite difference for second derivatives
            dxx = (smoothed[center_idx + 1, center_idx] - 2 * smoothed[center_idx, center_idx] + smoothed[
                center_idx - 1, center_idx]) / dx ** 2
            dyy = (smoothed[center_idx, center_idx + 1] - 2 * smoothed[center_idx, center_idx] + smoothed[
                center_idx, center_idx - 1]) / dx ** 2
            dxy = (smoothed[center_idx + 1, center_idx + 1] - smoothed[center_idx + 1, center_idx - 1] - smoothed[
                center_idx - 1, center_idx + 1] + smoothed[center_idx - 1, center_idx - 1]) / (4 * dx ** 2)

            # info construct Hessian
            hessian = np.array([[dxx, dxy], [dxy, dyy]])

            # info eigenvalues of Hessian
            eigvals, eigvecs = eigh(hessian)
        else:
            # info default values if no landscape was computed
            hessian = np.array([[0, 0], [0, 0]])
            eigvals = np.array([0, 0])
            eigvecs = np.array([[1, 0], [0, 1]])
            smoothed = np.zeros((2 * n_steps + 1, 2 * n_steps + 1))

        # info store in history
        self.loss_curvature_history.append({
            'eigenvalues': eigvals,
            'max_curvature': max(abs(eigvals)) if len(eigvals) > 0 else 0,
            'min_curvature': min(abs(eigvals)) if len(eigvals) > 0 else 0,
            'condition_number': max(abs(eigvals)) / (min(abs(eigvals)) + 1e-10) if len(eigvals) > 0 else 0
        })

        # info results
        curvature = {
            'base_loss': base_loss,
            'landscapes': landscapes,
            'hessian': hessian,
            'eigenvalues': eigvals,
            'eigenvectors': eigvecs,
            'max_curvature': max(abs(eigvals)) if len(eigvals) > 0 else 0,
            'min_curvature': min(abs(eigvals)) if len(eigvals) > 0 else 0,
            'condition_number': max(abs(eigvals)) / (min(abs(eigvals)) + 1e-10) if len(eigvals) > 0 else 0,
            'smoothed_landscape': smoothed
        }

        return curvature

    def _visualize_loss_landscape(self, curvature, name, epoch):
        """Create visualization of the loss landscape"""
        if not curvature['landscapes']:
            return

        for i, landscape_data in enumerate(curvature['landscapes']):
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
            if 'eigenvalues' in curvature:
                eigvals = curvature['eigenvalues']
                ax.text(0.05, 0.95,
                        f"λ1 = {eigvals[0]:.4f}\nλ2 = {eigvals[1]:.4f}\nCond. = {curvature['condition_number']:.4f}",
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Use formatted epoch for title
            epoch_str = f"{epoch:.1f}" if isinstance(epoch, float) else f"{epoch}"
            ax.set_title(f'{self.model.plot_prefix}: Loss Landscape at Epoch {epoch_str} (Direction {i + 1})')
            ax.set_xlabel('Direction 1')
            ax.set_ylabel('Direction 2')

            plt.tight_layout()
            plt.savefig(self.loss_landscape_dir / f"{name}_landscape_{i}.png")
            plt.close(fig)

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
        ax1.set_title(f'Hessian Eigenvalues Around Jump {self.model.plot_prefix}')
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
        ax2.set_title(f'Loss Landscape Condition Number {self.model.plot_prefix}')

        # Add labels clarifying the states
        for i, state in enumerate(['Before Jump', 'Jump Point', 'After Jump']):
            ax1.annotate(state, xy=(i, min(eig1[i], eig2[i])),
                         xytext=(i, min(eig1[i], eig2[i]) - 0.001),
                         ha='center', rotation=90, fontsize=8)

        plt.tight_layout()
        plt.savefig(self.loss_landscape_dir / f"{jump_id}_eigenvalue_comparison.png")
        plt.close(fig)

        # Combine landscapes in a single figure
        if results['landscape_analysis']['pre_jump']['landscapes'] and \
                results['landscape_analysis']['jump']['landscapes'] and \
                results['landscape_analysis']['post_jump']['landscapes']:

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            for i, (state, ax) in enumerate(zip(
                    ['pre_jump', 'jump', 'post_jump'],
                    axes
            )):
                landscape = results['landscape_analysis'][state]['smoothed_landscape']
                im = ax.imshow(landscape, cmap='viridis', origin='lower', aspect='auto')

                # Use formatted epoch for title
                epoch = epochs[i]
                epoch_str = f"{epoch:.1f}" if isinstance(epoch, float) else f"{epoch}"
                state_labels = ['Before Jump', 'At Jump', 'After Jump']
                ax.set_title(f"{state_labels[i]} (Epoch {epoch_str}) {self.model.plot_prefix}")

                plt.colorbar(im, ax=ax)

            plt.tight_layout()
            plt.savefig(self.loss_landscape_dir / f"{jump_id}_landscape_comparison.png")
            plt.close(fig)


    def analyze_head_attribution_around_jump(self, jump_epoch, eval_loader, weight_tracker, ensure_after_snapshot=True):
        """
        Analyze head attribution before, during, and after a jump

        Args:
            jump_epoch: The epoch of the jump
            eval_loader: Evaluation data loader
            weight_tracker: EnhancedWeightSpaceTracker instance with snapshots
            ensure_after_snapshot: If True, creates a new snapshot for after state if needed

        Returns:
            dict: Attribution analysis around the jump
        """
        jump_id = f"jump_{jump_epoch}"

        # Find the jump index in the weight_timestamps
        jump_idx = None
        for i, epoch in enumerate(weight_tracker.weight_timestamps):
            if epoch == jump_epoch:
                jump_idx = i
                break

        if jump_idx is None:
            print(f"No snapshot found for jump at epoch {jump_epoch}")
            return None

        # Get snapshots before, at, and after the jump
        before_idx = max(0, jump_idx - 1)

        # Check if we have a valid "after" snapshot
        after_idx = min(len(weight_tracker.weight_timestamps) - 1, jump_idx + 1)
        after_epoch = weight_tracker.weight_timestamps[after_idx]

        # If after_idx points to the same epoch as jump_idx, we need to create a new snapshot
        if after_epoch == jump_epoch and ensure_after_snapshot:
            print(f"Creating additional 'after jump' snapshot for epoch {jump_epoch}")

            # Store current model state
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            # Create a new snapshot labeled as "after jump"
            after_epoch = jump_epoch + 0.1  # Use fraction to indicate it's right after jump
            weight_tracker.take_snapshot(epoch=after_epoch, force=True)

            # Find the new snapshot
            after_idx = len(weight_tracker.weight_timestamps) - 1

            # Restore current state
            self.model.load_state_dict(current_state)

        before_snapshot = weight_tracker.weight_snapshots[before_idx]
        jump_snapshot = weight_tracker.weight_snapshots[jump_idx]
        after_snapshot = weight_tracker.weight_snapshots[after_idx]
        # When analyzing a jump
        before_epoch = before_snapshot['epoch']
        jump_epoch = jump_snapshot['epoch']
        after_epoch = after_snapshot['epoch']
        print(f"Head attribution: analyzing jump at {jump_epoch} with before={before_epoch}, after={after_epoch}")

        # Analyze head attribution for each snapshot
        results = {
            'jump_epoch': jump_epoch,
            'before_epoch': before_snapshot['epoch'],
            'jump_epoch': jump_snapshot['epoch'],
            'after_epoch': after_snapshot['epoch'],
            'attribution_analysis': {}
        }

        # Store original model state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        try:
            # Analyze each state (before, jump, after)
            for state_name, snapshot in [
                ('before', before_snapshot),
                ('jump', jump_snapshot),
                ('after', after_snapshot)
            ]:
                # Log which snapshot we're analyzing
                print(f"Analyzing {state_name} state attribution at epoch {snapshot['epoch']}")

                # Load the state
                self.model.load_state_dict(snapshot['state_dict'])

                # Analyze head attribution
                attribution = self.model.analyze_head_attribution(eval_loader)

                results['attribution_analysis'][state_name] = attribution

            # Create visualizations
            self._visualize_attribution_comparison(results, jump_id)

        finally:
            # Restore original model state
            self.model.load_state_dict(original_state)

        # Save the results
        if self.logger:
            for state_name in ['before', 'jump', 'after']:
                for head, score in results['attribution_analysis'][state_name].items():
                    self.logger.log_data('head_attribution',
                                         f'jump_{jump_epoch}_{state_name}_{head}',
                                         float(score))

        return results

    def _visualize_attribution_comparison(self, results, jump_id):
        """Create a comparative visualization of head attribution before/during/after jump"""
        # Extract attribution data
        before_attr = results['attribution_analysis']['before']
        jump_attr = results['attribution_analysis']['jump']
        after_attr = results['attribution_analysis']['after']

        # Convert to DataFrame for easier plotting
        heads = list(before_attr.keys())

        attr_data = []
        for head in heads:
            attr_data.append({
                'head': head,
                'before': before_attr[head],
                'jump': jump_attr[head],
                'after': after_attr[head]
            })

        attr_df = pd.DataFrame(attr_data)

        # Calculate change in attribution
        attr_df['before_to_jump'] = attr_df['jump'] - attr_df['before']
        attr_df['jump_to_after'] = attr_df['after'] - attr_df['jump']

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 1. Attribution values
        attr_melted = pd.melt(
            attr_df,
            id_vars=['head'],
            value_vars=['before', 'jump', 'after'],
            var_name='state',
            value_name='attribution'
        )

        # Remap state labels to be more descriptive
        state_labels = {
            'before': f'Before Jump (Epoch {results["before_epoch"]})',
            'jump': f'At Jump (Epoch {results["jump_epoch"]})',
            'after': f'After Jump (Epoch {results["after_epoch"]})'
        }
        attr_melted['state_label'] = attr_melted['state'].map(state_labels)

        sns.barplot(
            data=attr_melted,
            x='head',
            y='attribution',
            hue='state_label',
            ax=ax1
        )

        ax1.set_title(f'Head Attribution Around Jumpm {self.model.plot_prefix}')
        ax1.set_xlabel('Head')
        ax1.set_ylabel('Attribution Score')
        ax1.legend(title='State')
        ax1.tick_params(axis='x', rotation=45)

        # 2. Attribution changes
        change_melted = pd.melt(
            attr_df,
            id_vars=['head'],
            value_vars=['before_to_jump', 'jump_to_after'],
            var_name='transition',
            value_name='change'
        )

        # Remap transition labels
        transition_labels = {
            'before_to_jump': f'Before → Jump',
            'jump_to_after': f'Jump → After'
        }
        change_melted['transition_label'] = change_melted['transition'].map(transition_labels)

        sns.barplot(
            data=change_melted,
            x='head',
            y='change',
            hue='transition_label',
            ax=ax2
        )

        ax2.set_title(f'Head Attribution Changes: {self.model.plot_prefix}')
        ax2.set_xlabel('Head')
        ax2.set_ylabel('Attribution Change')
        ax2.legend(title='Transition')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.attribution_dir / f"{jump_id}_attribution_comparison.png")
        plt.close(fig)

        # Create a heatmap of the attribution matrix
        fig, ax = plt.subplots(figsize=(10, 8))

        # Reshape data for heatmap
        heatmap_data = attr_df[['head', 'before', 'jump', 'after']].set_index('head')

        # Rename columns for clarity
        heatmap_data.columns = [
            f'Before (Epoch {results["before_epoch"]})',
            f'Jump (Epoch {results["jump_epoch"]})',
            f'After (Epoch {results["after_epoch"]})'
        ]

        sns.heatmap(
            heatmap_data,
            cmap='viridis',
            annot=True,
            fmt=".3f",
            ax=ax
        )

        ax.set_title(f'Head Attribution Comparison Around Jump at Epoch {results["jump_epoch"]} {self.model.plot_prefix}')

        plt.tight_layout()
        plt.savefig(self.attribution_dir / f"{jump_id}_attribution_heatmap.png")
        plt.close(fig)

    def analyze_attention_patterns_around_jump(self, jump_epoch, eval_loader, weight_tracker,
                                               ensure_after_snapshot=True):
        """
        Analyze attention patterns and entropy before, during, and after a jump

        Args:
            jump_epoch: The epoch of the jump
            eval_loader: Evaluation data loader
            weight_tracker: EnhancedWeightSpaceTracker instance with snapshots
            ensure_after_snapshot: If True, creates a new snapshot for after state if needed

        Returns:
            dict: Attention pattern analysis around the jump
        """
        jump_id = f"jump_{jump_epoch}"

        # Find the jump index in the weight_timestamps
        jump_idx = None
        for i, epoch in enumerate(weight_tracker.weight_timestamps):
            if epoch == jump_epoch:
                jump_idx = i
                break

        if jump_idx is None:
            print(f"No snapshot found for jump at epoch {jump_epoch}")
            return None

        # Get snapshots before, at, and after the jump
        before_idx = max(0, jump_idx - 1)

        # Check if we have a valid "after" snapshot
        after_idx = min(len(weight_tracker.weight_timestamps) - 1, jump_idx + 1)
        after_epoch = weight_tracker.weight_timestamps[after_idx]

        # If after_idx points to the same epoch as jump_idx, we need to create a new snapshot
        if after_epoch == jump_epoch and ensure_after_snapshot:
            print(f"Creating additional 'after jump' snapshot for epoch {jump_epoch}")

            # Store current model state
            current_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            # Create a new snapshot labeled as "after jump"
            after_epoch = jump_epoch + 0.1  # Use fraction to indicate it's right after jump
            weight_tracker.take_snapshot(epoch=after_epoch, force=True)

            # Find the new snapshot
            after_idx = len(weight_tracker.weight_timestamps) - 1

            # Restore current state
            self.model.load_state_dict(current_state)

        before_snapshot = weight_tracker.weight_snapshots[before_idx]
        jump_snapshot = weight_tracker.weight_snapshots[jump_idx]
        after_snapshot = weight_tracker.weight_snapshots[after_idx]
        # When analyzing a jump
        before_epoch = before_snapshot['epoch']
        jump_epoch = jump_snapshot['epoch']
        after_epoch = after_snapshot['epoch']
        print(f"Attention: analyzing jump at {jump_epoch} with before={before_epoch}, after={after_epoch}")

        # Analyze attention patterns for each snapshot
        results = {
            'jump_epoch': jump_epoch,
            'before_epoch': before_snapshot['epoch'],
            'jump_epoch': jump_snapshot['epoch'],
            'after_epoch': after_snapshot['epoch'],
            'attention_analysis': {}
        }

        # Store original model state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        try:
            # Analyze each state (before, jump, after)
            for state_name, snapshot in [
                ('before', before_snapshot),
                ('jump', jump_snapshot),
                ('after', after_snapshot)
            ]:
                # Log which snapshot we're analyzing
                print(f"Analyzing {state_name} state attention at epoch {snapshot['epoch']}")

                # Load the state
                self.model.load_state_dict(snapshot['state_dict'])

                # Analyze attention entropy
                entropy = self.model.compute_attention_entropy(eval_loader)

                # Get attention patterns
                sample_input = next(iter(eval_loader))[0]
                if hasattr(self.model, 'device'):
                    sample_input = sample_input.to(self.model.device)

                # Forward pass with attention storage
                _ = self.model(sample_input, store_attention=True)

                # Get attention patterns
                patterns = self.model.get_attention_patterns()

                results['attention_analysis'][state_name] = {
                    'entropy': entropy,
                    'patterns': {k: v.cpu().numpy() for k, v in patterns.items()} if patterns else {}
                }

            # Create visualizations
            self._visualize_attention_comparison(results, jump_id)

        finally:
            # Restore original model state
            self.model.load_state_dict(original_state)

        # Save the results
        if self.logger:
            for state_name in ['before', 'jump', 'after']:
                for head, entropy_val in results['attention_analysis'][state_name]['entropy'].items():
                    self.logger.log_data('attention_entropy',
                                         f'jump_{jump_epoch}_{state_name}_{head}',
                                         float(entropy_val))

        return results

    def _visualize_attention_comparison(self, results, jump_id):
        """Create a comparative visualization of attention patterns and entropy before/during/after jump"""
        # Extract entropy data
        before_entropy = results['attention_analysis']['before']['entropy']
        jump_entropy = results['attention_analysis']['jump']['entropy']
        after_entropy = results['attention_analysis']['after']['entropy']

        # Convert to DataFrame for easier plotting
        heads = list(before_entropy.keys())

        entropy_data = []
        for head in heads:
            entropy_data.append({
                'head': head,
                'before': before_entropy[head],
                'jump': jump_entropy[head],
                'after': after_entropy[head]
            })

        entropy_df = pd.DataFrame(entropy_data)

        # Calculate change in entropy
        entropy_df['before_to_jump'] = entropy_df['jump'] - entropy_df['before']
        entropy_df['jump_to_after'] = entropy_df['after'] - entropy_df['jump']

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # 1. Entropy values
        entropy_melted = pd.melt(
            entropy_df,
            id_vars=['head'],
            value_vars=['before', 'jump', 'after'],
            var_name='state',
            value_name='entropy'
        )

        # Remap state labels to be more descriptive
        state_labels = {
            'before': f'Before Jump (Epoch {results["before_epoch"]})',
            'jump': f'At Jump (Epoch {results["jump_epoch"]})',
            'after': f'After Jump (Epoch {results["after_epoch"]})'
        }
        entropy_melted['state_label'] = entropy_melted['state'].map(state_labels)

        sns.barplot(
            data=entropy_melted,
            x='head',
            y='entropy',
            hue='state_label',
            ax=ax1
        )

        ax1.set_title(f'Attention Entropy Around Jump {self.model.plot_prefix}')
        ax1.set_xlabel('Head')
        ax1.set_ylabel('Entropy (lower = more specialized)')
        ax1.legend(title='State')
        ax1.tick_params(axis='x', rotation=45)

        # 2. Entropy changes
        change_melted = pd.melt(
            entropy_df,
            id_vars=['head'],
            value_vars=['before_to_jump', 'jump_to_after'],
            var_name='transition',
            value_name='change'
        )

        # Remap transition labels
        transition_labels = {
            'before_to_jump': f'Before → Jump',
            'jump_to_after': f'Jump → After'
        }
        change_melted['transition_label'] = change_melted['transition'].map(transition_labels)

        sns.barplot(
            data=change_melted,
            x='head',
            y='change',
            hue='transition_label',
            ax=ax2
        )

        ax2.set_title(f'Attention Entropy Changes {self.model.plot_prefix}')
        ax2.set_xlabel('Head')
        ax2.set_ylabel('Entropy Change')
        ax2.legend(title='Transition')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.attention_dir / f"{jump_id}_entropy_comparison.png")
        plt.close(fig)

        # Visualize attention patterns for each head
        pattern_sets = [
            ('before', results['attention_analysis']['before']['patterns']),
            ('jump', results['attention_analysis']['jump']['patterns']),
            ('after', results['attention_analysis']['after']['patterns'])
        ]

        # Get all heads that have patterns
        all_pattern_heads = set()
        for _, patterns in pattern_sets:
            all_pattern_heads.update(patterns.keys())

        # Create visualization for each head
        for head in all_pattern_heads:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            for i, (state_name, patterns) in enumerate(pattern_sets):
                ax = axes[i]

                if head in patterns:
                    pattern = patterns[head]
                    # Create a mask for lower triangle only (including diagonal)
                    mask = np.triu(np.ones_like(pattern, dtype=bool), k=1)
                    sns.heatmap(pattern, cmap='viridis', mask=mask,
                                cbar=False, xticklabels=False, yticklabels=False, ax=ax)
                else:
                    ax.text(0.5, 0.5, 'No pattern available',
                            ha='center', va='center')

                # Use more descriptive labels
                epoch = results[f"{state_name}_epoch"]
                epoch_str = f"{epoch:.1f}" if isinstance(epoch, float) else f"{epoch}"
                state_labels = ['Before Jump', 'At Jump', 'After Jump']
                ax.set_title(f"{state_labels[i]} (Epoch {epoch_str})")

            plt.suptitle(f'Attention Pattern Comparison for {head} {self.model.plot_prefix}', y=1.05)
            plt.tight_layout()

            # Save the figure
            head_safe = head.replace(':', '_').replace('.', '_')
            plt.savefig(self.attention_dir / f"{jump_id}_{head_safe}_pattern_comparison.png")
            plt.close(fig)

    def analyze_jump_with_snapshots(self, jump_epoch, pre_jump_snapshot, jump_snapshot,
                                    post_jump_snapshot, inputs, targets, criterion):
        """
        Analyze a jump using explicit pre-jump, jump, and post-jump snapshots

        Args:
            jump_epoch: The epoch of the jump
            pre_jump_snapshot: Snapshot from just before the jump
            jump_snapshot: Snapshot at the jump
            post_jump_snapshot: Snapshot after the jump
            inputs, targets: Batch of data for loss calculation
            criterion: Loss function

        Returns:
            dict: Analysis results around the jump
        """
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
                # Log which snapshot we're analyzing
                print(f"Analyzing {state_name} state at epoch {snapshot['epoch']}")

                # Load the state
                self.model.load_state_dict(snapshot['state_dict'])

                # Analyze curvature at this point
                curvature = self.analyze_loss_curvature(
                    inputs=inputs,
                    targets=targets,
                    criterion=criterion
                )

                results['landscape_analysis'][state_name] = curvature

                # Create visualization
                self._visualize_loss_landscape(
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
                                 results['landscape_analysis']['jump']['eigenvalues'].tolist())
            self.logger.log_data('loss_landscape', f'jump_{jump_epoch}_condition_number',
                                 results['landscape_analysis']['jump']['condition_number'])

        return results