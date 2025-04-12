import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


class EnhancedWeightSpaceTracker:
    """Enhanced tracker for model's trajectory in weight space with jump detection and analysis"""

    def __init__(self, model, save_dir, logger=None, pca_components=50, snapshot_freq=10,
                 sliding_window_size=5, dense_sampling=True, jump_detection_window=100,
                 jump_threshold=1.0):
        # Existing initialization code...
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.pca_components = pca_components
        self.snapshot_freq = snapshot_freq
        self.jump_detection_window = jump_detection_window
        self.jump_threshold = jump_threshold
        self.logger = logger if logger else model.logger if hasattr(model, 'logger') else None

        # Add sliding window parameters
        self.sliding_window_size = sliding_window_size  # Number of recent epochs to keep
        self.dense_sampling = dense_sampling  # Whether to sample more densely between snapshots

        # Storage for weight snapshots and trajectories
        self.weight_snapshots = []
        self.weight_timestamps = []
        self.flattened_weights = []
        self.velocities = []
        self.accelerations = []
        self.pca = None
        self.pca_fitted = False

        # todo perhaps it would be better to track ALL the snapshots, but then remove these that were NOT used?

        # info sliding window of recent states (more frequent than main snapshots)
        self.recent_snapshots = []  # info shall contain (epoch, state_dict, flattened_vector) tuples
        self.last_jump_epoch = None  # info track when the last jump occurred

        # Jump detection
        self.detected_jumps = []
        self.jump_analysis = {}
        self.pending_jumps = []  # info track jumps that need analysis

        # Create directories for detailed analysis
        self.jump_dir = self.save_dir / "jump_analysis"
        self.jump_dir.mkdir(exist_ok=True, parents=True)

        # label height for "jump at {epoch}"
        self.label_height = 0.0
        self.label_height_shift = 0.09

        # Counter for jump analysis
        self.jump_counter = 0

    def take_snapshot(self, epoch, force=False):
        """Take a snapshot of the current model weights with improved jump detection"""
        # Always flatten weights for potential sliding window storage
        flattened = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:  # Only consider weight matrices, not biases
                flattened.append(param.detach().cpu().view(-1))

        flattened_vector = torch.cat(flattened).numpy()

        # info heck if we should add this to the sliding window (more frequent than main snapshots)
        #  either we're doing dense sampling or this is a regular snapshot point
        should_add_to_window = self.dense_sampling or epoch % self.snapshot_freq == 0

        if should_add_to_window:
            # Store state in sliding window
            state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.recent_snapshots.append((epoch, state_dict, flattened_vector))

            # info maintain window size
            while len(self.recent_snapshots) > self.sliding_window_size:
                self.recent_snapshots.pop(0)

        # info regular snapshot logic - only at snapshot_freq intervals or when forced
        if not force and epoch % self.snapshot_freq != 0:
            return False

        # info store only  in main trajectory if this is the first snapshot or significant change occurred
        should_store = False
        if len(self.flattened_weights) == 0:
            should_store = True
        else:
            # info calculate and store velocity (change in weights)
            velocity = flattened_vector - self.flattened_weights[-1]
            velocity_norm = np.linalg.norm(velocity)
            self.velocities.append((epoch, velocity, velocity_norm))

            # info calculate acceleration if we have at least two velocities
            if len(self.velocities) >= 2:
                prev_velocity = self.velocities[-2][1]
                acceleration = velocity - prev_velocity
                acceleration_norm = np.linalg.norm(acceleration)
                self.accelerations.append((epoch, acceleration, acceleration_norm))

            # info decide whether to store based on change
            should_store = force or velocity_norm > 1e-6

            # info check for jumps - requires some history
            if len(self.velocities) > self.jump_detection_window:
                jump_detected = self._check_for_jumps(epoch, velocity_norm)

                # info if a jump is detected, queue it for analysis with pre-jump state
                if jump_detected:
                    # Find the most recent pre-jump snapshot from sliding window
                    pre_jump_snapshot = self._find_pre_jump_snapshot(epoch)
                    if pre_jump_snapshot:
                        pre_jump_epoch, pre_jump_state, pre_jump_vector = pre_jump_snapshot
                        # info store this explicitly as a "pre-jump" snapshot
                        self.pending_jumps.append({
                            'jump_epoch': epoch,
                            'pre_jump_epoch': pre_jump_epoch,
                            'pre_jump_state': pre_jump_state,
                            'pre_jump_vector': pre_jump_vector
                        })

        if should_store:
            self.flattened_weights.append(flattened_vector)
            self.weight_timestamps.append(epoch)

            # info store model state dictionary for later detailed analysis
            self.weight_snapshots.append({
                'epoch': epoch,
                'state_dict': {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            })

            # info log the snapshot event
            if self.logger:
                self.logger.log_data('weight_space_tracking', 'snapshot_epochs', epoch)

            return True

        return False


    def _find_pre_jump_snapshot(self, jump_epoch):
        """Find the most recent snapshot before the jump epoch"""
        # Sort snapshots by epoch in descending order
        sorted_snapshots = sorted(self.recent_snapshots, key=lambda x: x[0], reverse=True)

        # info find the most recent snapshot that's before the jump epoch
        for snapshot in sorted_snapshots:
            if snapshot[0] < jump_epoch:
                return snapshot

        # info if no suitable snapshot found, return None
        return None

    def analyze_pending_jumps(self, inputs, targets, criterion, jump_analyzer):
        """Analyze any pending jumps with pre-jump states"""
        if not self.pending_jumps:
            return None

        results = []

        for pending_jump in self.pending_jumps:
            jump_epoch = pending_jump['jump_epoch']
            pre_jump_epoch = pending_jump['pre_jump_epoch']
            pre_jump_state = pending_jump['pre_jump_state']

            # Find the jump snapshot
            jump_snapshot = None
            for snapshot in self.weight_snapshots:
                if snapshot['epoch'] == jump_epoch:
                    jump_snapshot = snapshot
                    break

            if jump_snapshot is None:
                print(f"Warning: Jump snapshot for epoch {jump_epoch} not found")
                continue

            # Find the next snapshot after the jump
            post_jump_snapshot = None
            for snapshot in self.weight_snapshots:
                if snapshot['epoch'] > jump_epoch:
                    post_jump_snapshot = snapshot
                    break

            # If no post-jump snapshot exists, create a simulated one at jump_epoch + 0.1
            if post_jump_snapshot is None:
                # Use current model state for post-jump (will be saved as jump_epoch + 0.1)
                post_jump_snapshot = {
                    'epoch': jump_epoch + 0.1,
                    'state_dict': {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                }

            # Create a "pre-jump" snapshot explicitly
            pre_jump_snapshot = {
                'epoch': pre_jump_epoch,
                'state_dict': pre_jump_state
            }

            # Now we have balanced pre_jump, jump, and post_jump snapshots
            print(f"Analyzing jump at {jump_epoch} with pre={pre_jump_epoch}, post={post_jump_snapshot['epoch']}")

            # Perform the analysis with these snapshots
            result = jump_analyzer.analyze_jump_with_snapshots(
                jump_epoch=jump_epoch,
                pre_jump_snapshot=pre_jump_snapshot,
                jump_snapshot=jump_snapshot,
                post_jump_snapshot=post_jump_snapshot,
                inputs=inputs,
                targets=targets,
                criterion=criterion
            )

            results.append(result)

        # Clear pending jumps after analysis
        self.pending_jumps = []

        return results


    def _check_for_jumps(self, epoch, current_velocity_norm):
        """Check if a jump has occurred based on velocity patterns"""
        # info get recent velocity norms
        recent_norms = [v[2] for v in self.velocities[-self.jump_detection_window:]]
        mean_norm = np.mean(recent_norms[:-1])  # info mean of all except current
        std_norm = np.std(recent_norms[:-1])  # info std of all except current

        # info a jump is detected if velocity is significantly higher than recent history
        #  using Z-score threshold for detection
        z_score = (current_velocity_norm - mean_norm) / (std_norm + 1e-8)

        # info set a minimum absolute threshold as well as a relative threshold
        # info adjust this based to typical problem velocity
        # fixme todo grow if there was a jump detected at last step (in detection window[-1]?)
        #  how to decrease it then? Perhaps decrease if in the current there is no jump then set to minimum (or standard) value?
        minimum_velocity_threshold = 0.5

        # info check both relative significance (z-score) AND absolute magnitude
        is_jump = (z_score > self.jump_threshold) and (current_velocity_norm > minimum_velocity_threshold)

        # info mpose minimum distance between detected jumps (at least 5 epochs)
        if is_jump and self.last_jump_epoch is not None:
            # todo fixme the same might be done here, but it probably does not work?
            min_jump_distance = 5
            if epoch - self.last_jump_epoch < min_jump_distance:
                is_jump = False  # Too close to the last jump, ignore

        # info add an additional filter to avoid detecting jumps when velocity is consistently low
        if is_jump and mean_norm < 0.1 and current_velocity_norm < 0.5:  # Adjust thresholds as needed
            is_jump = False  # Likely noise, not a significant jump

        if is_jump:
            # info update last jump epoch
            self.last_jump_epoch = epoch

            # info mark this epoch as a jump point
            self.detected_jumps.append({
                'epoch': epoch,
                'velocity_norm': current_velocity_norm,
                'z_score': z_score,
                'mean_norm': mean_norm,
                'std_norm': std_norm
            })

            # info log the jump
            if self.logger:
                self.logger.log_data('weight_space_jumps', 'jump_epochs', epoch)
                self.logger.log_data('weight_space_jumps', 'jump_velocity_norms', current_velocity_norm)
                self.logger.log_data('weight_space_jumps', 'jump_z_scores', z_score)

            return True

        return False

    def _analyze_jump(self, jump_epoch, z_score):
        """Perform detailed analysis of a detected jump"""
        self.jump_counter += 1
        jump_id = f"jump_{self.jump_counter}_{jump_epoch}"

        # Find the index of the jump epoch in our weight snapshots
        jump_indices = [i for i, e in enumerate(self.weight_timestamps) if e == jump_epoch]

        if not jump_indices:
            print(f"Warning: Jump detected at epoch {jump_epoch} but no corresponding weight snapshot found")
            return

        jump_idx = jump_indices[0]

        # We need snapshots before and after the jump for comparison
        if jump_idx == 0 or jump_idx >= len(self.weight_snapshots) - 1:
            print(f"Warning: Jump at index {jump_idx} is at the boundary of available snapshots")
            return

        # Get snapshots before and after jump
        before_snapshot = self.weight_snapshots[jump_idx - 1]
        jump_snapshot = self.weight_snapshots[jump_idx]
        after_snapshot = self.weight_snapshots[jump_idx + 1]

        # Detailed analysis of weights
        self.jump_analysis[jump_id] = {
            'epoch': jump_epoch,
            'z_score': z_score,
            'before_epoch': before_snapshot['epoch'],
            'after_epoch': after_snapshot['epoch'],
            'weight_changes': {}
        }

        # Analyze each layer's weights
        for name, jump_param in jump_snapshot['state_dict'].items():
            if 'weight' not in name:
                continue

            before_param = before_snapshot['state_dict'][name]
            after_param = after_snapshot['state_dict'][name]

            # Calculate changes
            before_to_jump = (jump_param - before_param).abs().mean().item()
            jump_to_after = (after_param - jump_param).abs().mean().item()

            # Store layer-specific changes
            self.jump_analysis[jump_id]['weight_changes'][name] = {
                'before_to_jump': before_to_jump,
                'jump_to_after': jump_to_after,
                'ratio': jump_to_after / (before_to_jump + 1e-8),
                'before_norm': before_param.norm().item(),
                'jump_norm': jump_param.norm().item(),
                'after_norm': after_param.norm().item()
            }

        # Log jump details
        if self.logger:
            self.logger.log_data('weight_space_jumps', 'analyzed_jumps', jump_id)

        # Save analysis to disk
        self._save_jump_analysis(jump_id)

    def _save_jump_analysis(self, jump_id):
        """Save the jump analysis to disk"""
        analysis = self.jump_analysis[jump_id]

        # Save as JSON
        import json
        json_path = self.jump_dir / f"{jump_id}_analysis.json"

        # Make the analysis JSON-serializable
        serializable_analysis = {
            'epoch': analysis['epoch'],
            'z_score': float(analysis['z_score']),
            'before_epoch': analysis['before_epoch'],
            'after_epoch': analysis['after_epoch'],
            'weight_changes': {}
        }

        for layer_name, changes in analysis['weight_changes'].items():
            serializable_analysis['weight_changes'][layer_name] = {
                k: float(v) for k, v in changes.items()
            }

        with open(json_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)

        # Create visualization of the jump
        self._visualize_jump(jump_id)

    def _visualize_jump(self, jump_id):
        """Create visualizations for the jump"""
        analysis = self.jump_analysis[jump_id]

        # Create PCA visualization if we have enough data
        if len(self.flattened_weights) >= 3:
            self._ensure_pca_fitted()

            # Get the trajectory in selected dimensions
            projected = self.pca.transform(np.stack(self.flattened_weights))

            # Find epochs around the jump
            jump_epoch = analysis['epoch']
            before_epoch = analysis['before_epoch']
            after_epoch = analysis['after_epoch']

            # Find indices in weight_timestamps
            jump_idx = self.weight_timestamps.index(jump_epoch)
            before_idx = self.weight_timestamps.index(before_epoch)
            after_idx = self.weight_timestamps.index(after_epoch)

            # Create figure for PCA visualization
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot full trajectory with lower alpha
            ax.plot(projected[:, 0], projected[:, 1], 'o-', alpha=0.3, color='blue')

            # Highlight the section around the jump
            window_size = 5
            start_idx = max(0, before_idx - window_size)
            end_idx = min(len(projected), after_idx + window_size)

            ax.plot(projected[start_idx:end_idx, 0], projected[start_idx:end_idx, 1], 'o-', alpha=0.8, color='blue')

            # Mark the jump specifically
            ax.plot(projected[jump_idx, 0], projected[jump_idx, 1], 'ro', markersize=10)
            ax.text(projected[jump_idx, 0], projected[jump_idx, 1], f"Jump {jump_epoch}", fontsize=12)

            # Mark before and after points
            ax.plot(projected[before_idx, 0], projected[before_idx, 1], 'go', markersize=8)
            ax.text(projected[before_idx, 0], projected[before_idx, 1], f"Before {before_epoch}", fontsize=10)

            ax.plot(projected[after_idx, 0], projected[after_idx, 1], 'mo', markersize=8)
            ax.text(projected[after_idx, 0], projected[after_idx, 1], f"After {after_epoch}", fontsize=10)

            # Add arrows to show direction
            for i in range(start_idx + 1, end_idx):
                ax.arrow(
                    projected[i - 1, 0], projected[i - 1, 1],
                    (projected[i, 0] - projected[i - 1, 0]) * 0.9,
                    (projected[i, 1] - projected[i - 1, 1]) * 0.9,
                    head_width=0.5, head_length=1.0, fc='blue', ec='blue', alpha=0.7
                )

            # Set labels with variance explained
            ax.set_xlabel(f'PCA Dimension 1 ({self.pca.explained_variance_ratio_[0]:.2%} var)')
            ax.set_ylabel(f'PCA Dimension 2 ({self.pca.explained_variance_ratio_[1]:.2%} var)')
            ax.set_title(f'Weight Space Trajectory Around Jump at Epoch {jump_epoch}')

            # Save the figure
            plt.tight_layout()
            fig.savefig(self.jump_dir / f"{jump_id}_pca_trajectory.png")
            plt.close(fig)

            # Create layer-specific analysis
            self._visualize_layer_changes(jump_id)

    def _visualize_layer_changes(self, jump_id):
        """Visualize changes in each layer across the jump"""
        analysis = self.jump_analysis[jump_id]

        # Extract layer data
        layers = list(analysis['weight_changes'].keys())
        before_to_jump = [analysis['weight_changes'][layer]['before_to_jump'] for layer in layers]
        jump_to_after = [analysis['weight_changes'][layer]['jump_to_after'] for layer in layers]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # Plot changes by layer
        x = np.arange(len(layers))
        width = 0.35

        ax1.bar(x - width / 2, before_to_jump, width, label='Before to Jump')
        ax1.bar(x + width / 2, jump_to_after, width, label='Jump to After')

        ax1.set_ylabel('Average Weight Change')
        ax1.set_title(f'Weight Changes by Layer at Jump {analysis["epoch"]}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers, rotation=90)
        ax1.legend()

        # Plot norm changes
        norms_before = [analysis['weight_changes'][layer]['before_norm'] for layer in layers]
        norms_jump = [analysis['weight_changes'][layer]['jump_norm'] for layer in layers]
        norms_after = [analysis['weight_changes'][layer]['after_norm'] for layer in layers]

        ax2.plot(x, norms_before, 'o-', label='Before')
        ax2.plot(x, norms_jump, 'o-', label='Jump')
        ax2.plot(x, norms_after, 'o-', label='After')

        ax2.set_ylabel('Weight Norm')
        ax2.set_title(f'Weight Norms by Layer at Jump {analysis["epoch"]}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layers, rotation=90)
        ax2.legend()

        # Save the figure
        plt.tight_layout()
        fig.savefig(self.jump_dir / f"{jump_id}_layer_changes.png")
        plt.close(fig)

    def _ensure_pca_fitted(self):
        """Ensure PCA is fitted to the current weights"""
        if not self.pca_fitted and len(self.flattened_weights) >= 3:
            # Convert to numpy array
            weight_matrix = np.stack(self.flattened_weights)

            # Fit PCA on the weights
            n_components = min(self.pca_components, weight_matrix.shape[0], weight_matrix.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(weight_matrix)
            self.pca_fitted = True

    def analyze_trajectory(self):
        """Analyze the trajectory of weights in PCA space"""
        self._ensure_pca_fitted()

        if not self.pca_fitted:
            return None

        # Convert to numpy array
        weight_matrix = np.stack(self.flattened_weights)

        # Project onto PCA space
        projected = self.pca.transform(weight_matrix)

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
            parallel_component = np.dot(acceleration, v_norm) * v_norm if np.ndim(v_norm) > 0 else 0

            # Calculate component perpendicular to velocity (normal acceleration)
            perpendicular_component = acceleration - parallel_component if np.ndim(
                parallel_component) > 0 else acceleration

            trajectory_stats['accelerations'].append({
                'epoch': self.weight_timestamps[i + 1],
                'vector': acceleration,
                'magnitude': acc_magnitude,
                'tangential': np.linalg.norm(parallel_component) if np.ndim(parallel_component) > 0 else 0,
                'normal': np.linalg.norm(perpendicular_component)
            })

        return trajectory_stats

    def get_jump_summary(self):
        """Get a summary of all detected jumps"""
        if not self.detected_jumps:
            return None

        jump_df = pd.DataFrame(self.detected_jumps)
        return jump_df

    def visualize_jumps_timeline(self):
        """Create a timeline visualization of all detected jumps"""
        if not self.detected_jumps:
            return None

        jump_df = pd.DataFrame(self.detected_jumps)

        # Get evaluation accuracy data if available
        eval_data = None
        train_data = None

        if self.logger and 'evaluation' in self.logger.logs and 'accuracy' in self.logger.logs['evaluation']:
            eval_epochs = self.logger.logs['evaluation']['epoch']
            eval_accs = self.logger.logs['evaluation']['accuracy']
            eval_data = pd.DataFrame({'epoch': eval_epochs, 'accuracy': eval_accs})

        if self.logger and 'training' in self.logger.logs and 'accuracy' in self.logger.logs['training']:
            train_epochs = self.logger.logs['training']['epoch']
            train_accs = self.logger.logs['training']['accuracy']
            train_data = pd.DataFrame({'epoch': train_epochs, 'accuracy': train_accs})

        # Create the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Plot accuracy if available
        if eval_data is not None:
            sns.lineplot(data=eval_data, x='epoch', y='accuracy', ax=ax1, label='Eval Accuracy', color='blue')

        if train_data is not None:
            sns.lineplot(data=train_data, x='epoch', y='accuracy', ax=ax1, label='Train Accuracy', color='green')

        # Add vertical lines for jumps
        for _, jump in jump_df.iterrows():
            self.label_height += self.label_height_shift
            if self.label_height > 1. - self.label_height_shift:
                self.label_height = self.label_height_shift
            ax1.axvline(x=jump['epoch'], color='red', linestyle='--',  lw=1.25, alpha=0.7)
            ax1.text(jump['epoch'], self.label_height, f"{int(jump['epoch'])}", rotation=90, alpha=0.9)

        ax1.set_title('Accuracy and Weight Space Jumps')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.05)

        # Plot jump velocities
        sns.scatterplot(data=jump_df, x='epoch', y='velocity_norm', ax=ax2, color='red', s=100)

        # Add velocity plot if we have enough data
        if self.velocities:
            vel_epochs = [v[0] for v in self.velocities]
            vel_norms = [v[2] for v in self.velocities]
            ax2.plot(vel_epochs, vel_norms, 'b-', alpha=0.5)

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Velocity Norm')
        ax2.set_title('Weight Space Velocities and Detected Jumps')

        plt.tight_layout()
        fig.savefig(self.save_dir / f"weight_space_jumps_timeline.png")
        plt.close(fig)

        return self.save_dir / f"weight_space_jumps_timeline.png"

    def visualize_trajectory(self, selected_dims=[0, 1], highlight_epochs=None):
        """Visualize the weight trajectory in PCA space with detected jumps highlighted"""
        self._ensure_pca_fitted()

        if not self.pca_fitted:
            return None

        # Get the trajectory in selected dimensions
        trajectory = self.pca.transform(np.stack(self.flattened_weights))

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot trajectory
        ax1.plot(trajectory[:, selected_dims[0]], trajectory[:, selected_dims[1]], 'o-', markersize=6, alpha=0.6)

        # Add arrows to show direction
        for i in range(1, len(trajectory)):
            ax1.arrow(
                trajectory[i - 1, selected_dims[0]], trajectory[i - 1, selected_dims[1]],
                (trajectory[i, selected_dims[0]] - trajectory[i - 1, selected_dims[0]]) * 0.9,
                (trajectory[i, selected_dims[1]] - trajectory[i - 1, selected_dims[1]]) * 0.9,
                head_width=0.01, head_length=0.02, fc='blue', ec='blue', alpha=0.7
            )

        # Highlight jumps
        min_epoch_to_mark = 20
        jump_epochs = [jump['epoch'] for jump in self.detected_jumps]

        offsets = [[+0.02, +0.02], [+0.02, -0.02], [-0.05, +0.02], [-0.05, -0.02]]
        for epoch in jump_epochs:
            if epoch in self.weight_timestamps and epoch >= min_epoch_to_mark:
                idx = self.weight_timestamps.index(epoch)
                ax1.plot(
                    trajectory[idx, selected_dims[0]],
                    trajectory[idx, selected_dims[1]],
                    'ro', markersize=7, alpha=0.7
                )
                off_x, off_y = offsets[np.random.randint(len(offsets))]
                ax1.text(
                    trajectory[idx, selected_dims[0]] + off_x,
                    trajectory[idx, selected_dims[1]] + off_y,
                    f'{epoch}', fontsize=9
                )

        # Highlight specific epochs if provided
        if highlight_epochs:
            if isinstance(highlight_epochs, list):
                for epoch in highlight_epochs:
                    if epoch not in jump_epochs:
                        if epoch in self.weight_timestamps and epoch >= min_epoch_to_mark:
                            idx = self.weight_timestamps.index(epoch)
                            ax1.plot(
                                trajectory[idx, selected_dims[0]],
                                trajectory[idx, selected_dims[1]],
                                'go', markersize=10, alpha=0.7
                            )
                            off_x, off_y = offsets[np.random.randint(len(offsets))]
                            ax1.text(
                                trajectory[idx, selected_dims[0]] + off_x,
                                trajectory[idx, selected_dims[1]] + off_y,
                                f'Epoch {epoch}', fontsize=9
                            )
            elif isinstance(highlight_epochs, dict):
                for key, vals in highlight_epochs.items():
                    color = 'ro' if key == 'grok' else 'go'
                    for epoch in vals:
                        if epoch not in jump_epochs:
                            if epoch in self.weight_timestamps and epoch >= min_epoch_to_mark:
                                idx = self.weight_timestamps.index(epoch)
                                ax1.plot(
                                    trajectory[idx, selected_dims[0]],
                                    trajectory[idx, selected_dims[1]],
                                    color, markersize=10, alpha=0.7
                                )
                                label = f'Grok {epoch}' if key == 'grok' else f'{epoch}'
                                off_x, off_y = offsets[np.random.randint(len(offsets))]
                                ax1.text(
                                    trajectory[idx, selected_dims[0]] + off_x,
                                    trajectory[idx, selected_dims[1]] + off_y,
                                    label, fontsize=9
                                )

        ax1.set_xlabel(
            f'PCA Dimension {selected_dims[0] + 1} ({self.pca.explained_variance_ratio_[selected_dims[0]]:.2%} var)')
        ax1.set_ylabel(
            f'PCA Dimension {selected_dims[1] + 1} ({self.pca.explained_variance_ratio_[selected_dims[1]]:.2%} var)')
        ax1.set_title('Weight Space Trajectory')

        # Plot speed and acceleration magnitudes
        stats = self.analyze_trajectory()
        if stats:
            epochs = [v['epoch'] for v in stats['velocities']]
            speeds = [v['magnitude'] for v in stats['velocities']]

            ax2.plot(epochs, speeds, 'b-', label='Speed')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Speed', color='b')
            ax2.tick_params(axis='y', labelcolor='b')

            if stats['accelerations']:
                acc_epochs = [a['epoch'] for a in stats['accelerations']]
                tangential = [a['tangential'] for a in stats['accelerations']]
                normal = [a['normal'] for a in stats['accelerations']]

                ax3 = ax2.twinx()
                ax3.plot(acc_epochs, tangential, 'r-', label='Tangential Acc.')
                ax3.plot(acc_epochs, normal, 'g-', label='Normal Acc.')
                ax3.set_ylabel('Acceleration', color='r')
                ax3.tick_params(axis='y', labelcolor='r')

                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax3.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            # Highlight jumps
            for epoch in jump_epochs:
                ax2.axvline(x=epoch, color='purple', linestyle='--', lw=1.0, alpha=0.5)

            ax2.set_title(f'Weight Space Velocity and Acceleration: {self.model.plot_prefix}')

        plt.tight_layout()
        save_path = self.save_dir / f'weight_trajectory_with_jumps_dims_{selected_dims[0]}_{selected_dims[1]}.png'
        plt.savefig(save_path)
        plt.close()

        return save_path