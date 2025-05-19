from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA


# todo let EnhancedWeightSpaceTracker inherit from analysis.core.weight_space.WeightSpaceTracker
class EnhancedWeightSpaceTracker:  # (WeightSpaceTracker):
    """Enhanced tracker for model's trajectory in weight space with jump detection and analysis"""

    def __init__(self, model, save_dir=None, pca_components=50, logger=None, snapshot_freq=10,
                 sliding_window_size=5, dense_sampling=True, jump_detection_window=100,
                 jump_threshold=1.0):
        # info initialization code...
        # info is in base
        self.model = model  # fixme yes
        self.save_dir = Path(save_dir)  # fixme yes
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.pca_components = pca_components  # fixme yes
        self.snapshot_freq = snapshot_freq
        self.jump_detection_window = jump_detection_window
        self.jump_threshold = jump_threshold
        self.logger = logger if logger else model.logger if hasattr(model, 'logger') else None

        # Add sliding window parameters
        self.sliding_window_size = sliding_window_size  # Number of recent epochs to keep
        self.dense_sampling = dense_sampling  # Whether to sample more densely between snapshots

        # info storage for weight snapshots and trajectories
        self.weight_snapshots = []
        self.weight_timestamps = []
        self.flattened_weights = []  # fixme yes
        self.velocities = []
        self.accelerations = []
        self.pca = None  # fixme yes
        self.pca_fitted = False  # fixme yes

        # todo perhaps it would be better to track ALL the snapshots, but then remove these that were NOT used?

        # info sliding window of recent states (more frequent than main snapshots)
        self.recent_snapshots = []  # info shall contain (epoch, state_dict, flattened_vector) tuples
        self.last_jump_epoch = None  # info track when the last jump occurred

        # info jump detection
        self.detected_jumps = []
        self.jump_analysis = {}
        self.pending_jumps = []  # info track jumps that need analysis

        # info create directories for detailed analysis
        self.jump_dir = self.save_dir / "jump_analysis"
        self.jump_dir.mkdir(exist_ok=True, parents=True)

        # info label height for "jump at {epoch}"
        self.label_height = 0.0
        self.label_height_shift = 0.09

        # info counter for jump analysis
        self.jump_counter = 0

    def take_snapshot(self, epoch, force=False):
        """Take a snapshot of the current model weights with improved jump detection"""
        # info always flatten weights for potential sliding window storage
        flattened = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:  # info anly consider weight matrices, not biases
                flattened.append(param.detach().cpu().view(-1))

        flattened_vector = torch.cat(flattened).numpy()
        # info get the current dictionary for better and more in-depth analysis
        current_state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        # info now check for group velocities, not only the whole model
        # todo is it possible to visualize all using the same pca plot?
        group_velocities = self.track_parameter_group_velocities(epoch, current_state_dict)
        if group_velocities and self.logger:
            # info log the group velocities
            for group, velocity in group_velocities['group_velocities'].items():
                self.logger.log_data('weight_velocities', f'{group}_velocity', velocity)

            # info log particularly interesting head velocities if significant
            if 'head_velocities' in group_velocities:
                top_head_velocities = sorted(
                    group_velocities['head_velocities'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]  # info top 5 changing heads

                for head, velocity in top_head_velocities:
                    if velocity > 0.1:  # Only log if change is significant
                        self.logger.log_data('weight_velocities', f'{head}_velocity', velocity)

        # info track spectral properties on regular snapshot intervals
        if epoch % self.snapshot_freq == 0 or force:
            spectral_stats = self.track_spectral_properties(epoch, current_state_dict)
            if spectral_stats and self.logger:
                # info log key spectral properties
                for component, stats in spectral_stats.items():
                    self.logger.log_data('spectral_properties',
                                         f'{component}_condition',
                                         stats['condition_number'])
                    self.logger.log_data('spectral_properties',
                                         f'{component}_rank',
                                         stats['effective_rank'])

        ###############################################################################################################
        # info check if we should add this to the sliding window (more frequent than main snapshots)
        #  either we're doing dense sampling or this is a regular snapshot point
        should_add_to_window = self.dense_sampling or epoch % self.snapshot_freq == 0

        if should_add_to_window:
            # info store state in sliding window
            saved_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.recent_snapshots.append((epoch, saved_state_dict, flattened_vector))

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
                    # info find the most recent pre-jump snapshot from sliding window  # fixme LAST??
                    pre_jump_snapshot = self._find_pre_jump_snapshot(epoch)
                    if pre_jump_snapshot:
                        pre_jump_epoch, pre_jump_state, pre_jump_vector = pre_jump_snapshot
                        # info store this explicitly for later analysis
                        self.pending_jumps.append({
                            'jump_epoch': epoch,
                            'pre_jump_epoch': pre_jump_epoch,
                            'pre_jump_state': pre_jump_state,
                            'pre_jump_vector': pre_jump_vector
                        })
                    print([np.linalg.norm(self.flattened_weights[-k - 1] - self.flattened_weights[-k - 2]) for k in
                           range(len(self.recent_snapshots) - 1)])

        if should_store:
            self.flattened_weights.append(flattened_vector)
            self.weight_timestamps.append(epoch)

            # info store model state dictionary for later detailed analysis
            self.weight_snapshots.append({
                'epoch': epoch,
                'state_dict': {k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()}
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

    # Add to EnhancedWeightSpaceTracker class
    def get_window_snapshots(self, center_epoch, window_size=5):
        """
        Get an array of snapshots centered around a specific epoch with the given window size

        Args:
            center_epoch: The center epoch to get snapshots around
            window_size: How many snapshots to get before and after (total will be 2*window_size+1)

        Returns:
            list: List of snapshot dictionaries ordered by epoch
        """
        # info get all available snapshots
        all_snapshots = self.weight_snapshots.copy()

        # info sort by epoch distance from center
        all_snapshots.sort(key=lambda s: abs(s['epoch'] - center_epoch))

        # info separate into before, at, and after center
        center_snapshot = None
        before_snapshots = []
        after_snapshots = []

        for snapshot in all_snapshots:
            if snapshot['epoch'] == center_epoch:
                center_snapshot = snapshot
            elif snapshot['epoch'] < center_epoch:
                before_snapshots.append(snapshot)
            else:
                after_snapshots.append(snapshot)

        # info sort by epoch
        before_snapshots.sort(key=lambda s: s['epoch'], reverse=True)  # Newest first
        after_snapshots.sort(key=lambda s: s['epoch'])  # Oldest first

        # info take up to window_size before and after
        before_snapshots = before_snapshots[:window_size]
        after_snapshots = after_snapshots[:window_size]

        # info combine in chronological order
        result = before_snapshots[::-1]  # Reverse to get chronological order
        if center_snapshot:
            result.append(center_snapshot)
        result.extend(after_snapshots)

        return result

    def analyze_extended_jump(self, jump_epoch, window_size=5):
        """
        Analyze a jump with a wider time window to capture gradual transformer changes

        Args:
            jump_epoch: The epoch of the jump to analyze
            window_size: Number of snapshots to analyze before and after the jump

        Returns:
            dict: Extended analysis results
        """
        # info get snapshots around the jump
        snapshots = self.get_window_snapshots(jump_epoch, window_size)

        if len(snapshots) < 3:
            print(f"\tEnhancedWeightSpaceTracker.analyze_extended_jump Not enough snapshots around jump at epoch {jump_epoch}")
            return None

        # info calculate change trajectory
        epochs = [s['epoch'] for s in snapshots]
        changes = []

        # info compare each consecutive pair of snapshots
        for i in range(1, len(snapshots)):
            prev_state = snapshots[i - 1]['state_dict']
            curr_state = snapshots[i]['state_dict']

            # info calculate total change
            total_change = 0
            for name, curr_param in curr_state.items():
                if name in prev_state:
                    prev_param = prev_state[name]
                    change = torch.norm(curr_param - prev_param).item()
                    total_change += change

            changes.append(total_change)

        # info create layer and head-specific change trajectories
        layer_trajectories = defaultdict(list)
        head_trajectories = defaultdict(list)

        # info for each layer and head, track changes over time
        for i in range(1, len(snapshots)):
            prev_state = snapshots

    def _find_closest_snapshot(self, epoch):
        # info find the index in recent_snapshots that is for epoch closest to one in param
        best_idx = np.argmin([abs((self.recent_snapshots[:][k][0] - epoch)) for k in range(len(self.recent_snapshots))])
        return [self.recent_snapshots[:][k][0] for k in range(len(self.recent_snapshots))][best_idx]

    def get_snapshot(self, epoch):
        # info return index in recent snapshots with that of epoch in param fixme if epoch not on recent_snapshots, add searching on weight_snapshots
        best_idx = [self.recent_snapshots[:][k][0] for k in range(len(self.recent_snapshots))].index(epoch)
        return self.recent_snapshots[best_idx]

    def analyze_pending_jumps(self, inputs, targets, criterion, jump_analyzer,
                              eval_loader=None, optimizer=None, mini_train_steps=10):
        """
        Analyze pending jumps with option to briefly train for post-jump state

        Args:
            inputs: Batch of inputs for analysis
            targets: Batch of targets
            criterion: Loss function
            jump_analyzer: Jump analyzer object
            optimizer: Optional optimizer for mini-training
            mini_train_steps: Number of steps to train for post-jump state

        Returns:
            list: Analysis results for pending jumps
        """
        results = []
        if not self.pending_jumps:
            return results

        # info store current model to restore after tarining
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # info get epoch numbers for epochs in jump window
        for pending_jump in self.pending_jumps:
            jump_epoch = pending_jump['jump_epoch']
            print(f"\tEnhancedWeightSpaceTracker.analyze_pending_jumps: analyzing jump at epoch {jump_epoch}")

            # info get pre jump epoch and state
            # fixme mini_train_steps
            pre_jump_epoch = self._find_closest_snapshot(jump_epoch - mini_train_steps)
            pre_jump_snapshot = self.get_snapshot(pre_jump_epoch)
            # fixme or as below?
            # pre_jump_epoch = pending_jump['pre_jump_epoch']
            # pre_jump_state = pending_jump['pre_jump_state']
            # info get current snapshot warning get_snapshot() operates only on self.recent_snapshots # a jesli wywolane gdzie indziej???
            jump_snapshot = self.get_snapshot(jump_epoch)

            # info reformat pre_ and jump_ snapshots to {'epoch': int, 'state_dict': {dict: 29}} format
            pre_jump_snapshot = {'epoch': pre_jump_epoch,
                                 'state_dict': pre_jump_snapshot[1]}
            jump_snapshot = {'epoch': jump_epoch,
                             'state_dict': jump_snapshot[1]}
            # info create true post-jump state using short training
            if optimizer is not None:
                # info load the jump state
                self.model.load_state_dict(jump_snapshot['state_dict'])
                # info perform some traing epochs
                self.model.train()
                if eval_loader is not None:
                    for _ in range(mini_train_steps):
                        train_loss = 0.0
                        train_total = 0
                        for this_input, this_target in eval_loader:
                            optimizer.zero_grad()
                            # fixme how to get more than one input? pass dataloader?
                            this_output = self.model(this_input)
                            loss = criterion(this_output, this_target)
                            loss.backward()
                            optimizer.step()
                            train_total += targets.size(0)
                            train_loss += loss.item() * targets.size(0)
                        train_loss = train_loss / train_total
                        pass
                else:
                    for _ in range(2 * mini_train_steps):
                        optimizer.zero_grad()
                        # fixme how to get more than one input? pass dataloader?
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                self.model.eval()
                # info get the post jump state
                post_jump_snapshot = {
                    'epoch': jump_epoch + mini_train_steps / 100,
                    'state_dict': {k: v.clone() for k, v in self.model.state_dict().items()}
                }
            else:
                post_jump_snapshot = {'epoch': jump_epoch,
                                      'state_dict': jump_snapshot[1]}

            # info perform detailed jump characterization
            jump_char = self.characterize_jump(
                jump_epoch,
                pre_jump_snapshot,
                jump_snapshot,
                post_jump_snapshot
            )

            # info log the key metrics
            if self.logger:
                self.logger.log_data('jump_analysis', f'jump_{jump_epoch}_total_magnitude',
                                     jump_char['total_magnitude']['pre_to_jump'])
                self.logger.log_data('jump_analysis', f'jump_{jump_epoch}_symmetry_ratio',
                                     jump_char['total_magnitude']['symmetry_ratio'])

                # info log top changing components
                for component_type, changes in jump_char['top_changing_components'].items():
                    if changes:  # info if there are any changes for this component
                        top_param, top_change = changes[0]  # Get the most changed parameter
                        self.logger.log_data('jump_analysis',
                                             f'jump_{jump_epoch}_{component_type}_top_change',
                                             top_change['pre_to_jump'])

                # info log top changing layers and heads
                for layer in jump_char['top_layers']:
                    self.logger.log_data('jump_analysis',
                                         f'jump_{jump_epoch}_top_layer_{layer}',
                                         jump_char['layer_changes'][layer]['pre_to_jump'])

                for head in jump_char['top_heads']:
                    self.logger.log_data('jump_analysis',
                                         f'jump_{jump_epoch}_top_head_{head}',
                                         jump_char['head_changes'][head]['pre_to_jump'])

            print(f"\tEnhancedWeightSpaceTracker.analyze_pending_jumps Jump at {jump_epoch} characterized:")
            print(f"\t\t  - Pre-jump epoch: {pre_jump_epoch}")
            print(f"\t\t  - Post-jump epoch: {post_jump_snapshot['epoch']}")
            print(f"\t\t  - Total change magnitude: {jump_char['total_magnitude']['pre_to_jump']:.4f}")
            print(f"\t\t  - Top changing layers: {', '.join(jump_char['top_layers'])}")
            print(f"\t\t  - Top changing heads: {', '.join(jump_char['top_heads'])}")

            ########################################################################################################
            # info now we have balanced pre_jump, jump, and post_jump snapshots
            print(f"\tEnhancedWeightSpaceTracker.analyze_pending_jumps Analyzing jump at {jump_epoch} with pre={pre_jump_epoch}, post={post_jump_snapshot['epoch']}")

            # info analysis with these snapshots
            analyzer_result = {}
            if jump_analyzer is not None:
                analyzer_result = jump_analyzer.analyze_jump_with_snapshots(
                    jump_epoch=jump_epoch,
                    pre_jump_snapshot=pre_jump_snapshot,
                    jump_snapshot=jump_snapshot,
                    post_jump_snapshot=post_jump_snapshot,
                    inputs=inputs,
                    targets=targets,
                    eval_loader=eval_loader,
                    criterion=criterion
                )

            # info combine all the results
            # fixme add 'pre_jump_snapshot', 'jump_snapshot', 'post_jump_snapshot'
            combined_result = {
                'jump_epoch': jump_epoch,
                'analyzer_result': analyzer_result,
                'characterization': jump_char,
                'pre_jump_snapshot': pre_jump_snapshot,
                'jump_snapshot': jump_snapshot,
                'post_jump_snapshot': post_jump_snapshot,
            }

        results.append(combined_result)

        # info clear pending jumps after analysis
        self.pending_jumps = []
        # info restore original model
        self.model.load_state_dict(original_state)

        return results

    def characterize_jump(self, jump_epoch, pre_jump_snapshot, jump_snapshot, post_jump_snapshot, window_size=5):
        """
        Enhanced characterization of a jump to better capture transformer structural changes
        by analyzing within a wider window before and after the jump.
        """
        # Get the jump type
        jump_idx = None
        for i, jump in enumerate(self.detected_jumps):
            if jump['epoch'] == jump_epoch:
                jump_idx = i
                break

        jump_type = self.detected_jumps[jump_idx]['jump_type'] if jump_idx is not None else "unknown"

        # Define the analysis window
        pre_epoch = pre_jump_snapshot['epoch']
        jump_epoch = jump_snapshot['epoch']
        post_epoch = post_jump_snapshot['epoch']

        # For gradual jumps, we want to compare states further apart
        if jump_type == "gradual":
            # Try to find snapshots further back and ahead
            pre_state = self._find_snapshot_at_epoch_offset(pre_epoch, -window_size)
            if pre_state is None:
                pre_state = pre_jump_snapshot['state_dict']
            else:
                pre_state = pre_state['state_dict']

            post_state = self._find_snapshot_at_epoch_offset(post_epoch, window_size)
            if post_state is None:
                post_state = post_jump_snapshot['state_dict']
            else:
                post_state = post_state['state_dict']
        else:
            pre_state = pre_jump_snapshot['state_dict']
            post_state = post_jump_snapshot['state_dict']

        jump_state = jump_snapshot['state_dict']

        # Component changes tracking
        component_changes = defaultdict(dict)
        total_pre_to_jump = 0
        total_jump_to_post = 0

        # Layer and head specific changes
        layer_changes = defaultdict(lambda: {'pre_to_jump': 0, 'jump_to_post': 0})
        head_changes = defaultdict(lambda: {'pre_to_jump': 0, 'jump_to_post': 0})

        # Track structural changes using SVD analysis
        svd_changes = {}

        # Analyze each parameter
        for name, jump_param in jump_state.items():
            if name not in pre_state or name not in post_state:
                continue

            pre_param = pre_state[name]
            post_param = post_state[name]

            # Compute standard changes
            pre_to_jump_change = torch.norm(jump_param - pre_param).item()
            jump_to_post_change = torch.norm(post_param - jump_param).item()

            # Track total change
            total_pre_to_jump += pre_to_jump_change
            total_jump_to_post += jump_to_post_change

            # Enhanced analysis for weight matrices
            if len(jump_param.shape) >= 2 and min(jump_param.shape) > 1:
                # Perform SVD to analyze structural changes
                try:
                    # Pre-jump to jump change
                    if pre_param.shape == jump_param.shape:
                        pre_U, pre_S, pre_V = torch.svd(pre_param)
                        jump_U, jump_S, jump_V = torch.svd(jump_param)

                        # Compare singular values
                        sing_change = torch.norm(jump_S - pre_S).item() / torch.norm(pre_S).item()

                        # Compare principal directions (first few right singular vectors)
                        num_vectors = min(3, pre_V.shape[1])
                        direction_change = 0
                        for i in range(num_vectors):
                            # Account for sign ambiguity in SVD
                            sim = max(
                                torch.abs(torch.dot(pre_V[:, i], jump_V[:, i])).item(),
                                torch.abs(torch.dot(pre_V[:, i], -jump_V[:, i])).item()
                            )
                            direction_change += 1.0 - sim
                        direction_change /= num_vectors

                        svd_changes[name] = {
                            'singular_value_change': sing_change,
                            'direction_change': direction_change
                        }
                except:
                    # SVD may fail for some tensors
                    pass

            # Categorize by component type
            component_type = self._categorize_parameter(name)
            component_changes[component_type][name] = {
                'pre_to_jump': pre_to_jump_change,
                'jump_to_post': jump_to_post_change,
                'relative_change': pre_to_jump_change / (torch.norm(pre_param).item() + 1e-8)
            }

            # Extract layer and head info
            if 'layers.' in name:
                self._extract_layer_head_changes(
                    name, pre_param, jump_param, post_param,
                    layer_changes, head_changes
                )

        # Normalize layer and head changes
        self._normalize_changes(layer_changes, head_changes)

        # Find top changing components
        top_changing = self._find_top_changing_components(component_changes)

        # Calculate spectral changes
        pre_spectral = self.track_spectral_properties(jump_epoch - 1, pre_state)
        jump_spectral = self.track_spectral_properties(jump_epoch, jump_state)

        # Build comprehensive characterization with enhanced structural info
        characterization = {
            'jump_epoch': jump_epoch,
            'jump_type': jump_type,
            'pre_epoch': pre_jump_snapshot['epoch'],
            'post_epoch': post_jump_snapshot['epoch'],
            'total_magnitude': {
                'pre_to_jump': total_pre_to_jump,
                'jump_to_post': total_jump_to_post,
                'symmetry_ratio': total_pre_to_jump / (total_jump_to_post + 1e-8)
            },
            'component_changes': {k: v for k, v in component_changes.items()},
            'top_changing_components': top_changing,
            'layer_changes': dict(sorted(layer_changes.items(), key=lambda x: x[1]['pre_to_jump'], reverse=True)),
            'head_changes': dict(sorted(head_changes.items(), key=lambda x: x[1]['pre_to_jump'], reverse=True)),
            'top_layers': [l[0] for l in
                           sorted(layer_changes.items(), key=lambda x: x[1]['pre_to_jump'], reverse=True)[:3]],
            'top_heads': [h[0] for h in
                          sorted(head_changes.items(), key=lambda x: x[1]['pre_to_jump'], reverse=True)[:3]],
            'svd_changes': svd_changes,
            'spectral_changes': {
                component: {
                    'condition_number_change': jump_spectral.get(component, {}).get('condition_number', 0) -
                                               pre_spectral.get(component, {}).get('condition_number', 0),
                    'effective_rank_change': jump_spectral.get(component, {}).get('effective_rank', 0) -
                                             pre_spectral.get(component, {}).get('effective_rank', 0)
                }
                for component in jump_spectral if component in pre_spectral
            }
        }

        return characterization

    def _find_snapshot_at_epoch_offset(self, base_epoch, offset):
        """Find a snapshot at base_epoch + offset, or the closest available"""
        target_epoch = base_epoch + offset

        # Sort snapshots by epoch distance from target
        sorted_snapshots = sorted(self.weight_snapshots,
                                  key=lambda s: abs(s['epoch'] - target_epoch))

        # Return the closest one that's in the right direction
        for snapshot in sorted_snapshots:
            # For negative offset, want snapshot earlier than base_epoch
            if offset < 0 and snapshot['epoch'] < base_epoch:
                return snapshot
            # For positive offset, want snapshot later than base_epoch
            elif offset > 0 and snapshot['epoch'] > base_epoch:
                return snapshot

        # If no suitable snapshot found, return None
        return None

    def _categorize_parameter(self, name):
        """Categorize parameter by type with more specific categorization"""
        if 'attn' in name:
            if 'in_proj' in name:
                # Further distinguish QKV components where possible
                # Detect possible separate Q,K,V components (depends on PyTorch version)
                if 'q_proj' in name:
                    return 'attention_query'
                elif 'k_proj' in name:
                    return 'attention_key'
                elif 'v_proj' in name:
                    return 'attention_value'
                else:
                    return 'attention_qkv'
            elif 'out_proj' in name:
                return 'attention_output'
        elif 'mlp' in name:
            if any(up_term in name for up_term in ['0.weight', 'up_proj', 'c_fc']):
                return 'mlp_up'
            elif any(down_term in name for down_term in ['2.weight', 'down_proj', 'c_proj']):
                return 'mlp_down'
            else:
                return 'mlp_other'
        elif any(norm_term in name for norm_term in ['ln', 'norm', 'layer_norm']):
            return 'layernorm'
        elif 'embed' in name:
            return 'embedding'
        else:
            return 'other'

    def _extract_layer_head_changes(self, name, pre_param, jump_param, post_param,
                                    layer_changes, head_changes):
        """Extract layer and head specific changes with enhanced detection"""
        parts = name.split('.')
        try:
            # Extract layer index
            if 'layers.' in name:
                layer_idx = int(parts[parts.index('layers') + 1])
            else:
                # Try other common patterns
                for i, part in enumerate(parts):
                    if part.isdigit() and i > 0 and parts[i - 1] in ['layer', 'transformer', 'block']:
                        layer_idx = int(part)
                        break
                else:
                    return  # No layer found

            layer_key = f'layer_{layer_idx}'

            # Track layer changes
            pre_to_jump = torch.norm(jump_param - pre_param).item()
            jump_to_post = torch.norm(post_param - jump_param).item()

            layer_changes[layer_key]['pre_to_jump'] += pre_to_jump
            layer_changes[layer_key]['jump_to_post'] += jump_to_post

            # Enhanced head detection
            if (('attn' in name or 'attention' in name) and
                    hasattr(self.model, 'dim') and hasattr(self.model, 'num_heads')):
                self._analyze_attention_head_changes(
                    name, layer_idx, pre_param, jump_param, post_param, head_changes
                )
        except:
            pass

    def _analyze_attention_head_changes(self, name, layer_idx, pre_param, jump_param, post_param, head_changes):
        """Detailed analysis of attention head changes with better detection"""
        model = self.model
        head_dim = model.dim // model.num_heads

        # For different attention implementations:

        # 1. Standard PyTorch MultiheadAttention (QKV stacked in in_proj_weight)
        if 'in_proj_weight' in name:
            dim = model.dim
            for h in range(model.num_heads):
                start = h * head_dim
                end = (h + 1) * head_dim

                # Extract head-specific weights for Q, K, V
                q_pre = pre_param[:dim][start:end]
                q_jump = jump_param[:dim][start:end]
                q_post = post_param[:dim][start:end]

                k_pre = pre_param[dim:2 * dim][start:end]
                k_jump = jump_param[dim:2 * dim][start:end]
                k_post = post_param[dim:2 * dim][start:end]

                v_pre = pre_param[2 * dim:][start:end]
                v_jump = jump_param[2 * dim:][start:end]
                v_post = post_param[2 * dim:][start:end]

                # Compute head-specific changes
                head_key = f'layer_{layer_idx}_head_{h}'
                head_changes[head_key]['pre_to_jump'] += (
                        torch.norm(q_jump - q_pre).item() +
                        torch.norm(k_jump - k_pre).item() +
                        torch.norm(v_jump - v_pre).item()
                )
                head_changes[head_key]['jump_to_post'] += (
                        torch.norm(q_post - q_jump).item() +
                        torch.norm(k_post - k_jump).item() +
                        torch.norm(v_post - v_jump).item()
                )

        # 2. Separate QKV projections
        elif any(qkv in name for qkv in ['q_proj', 'k_proj', 'v_proj']):
            for h in range(model.num_heads):
                start = h * head_dim
                end = (h + 1) * head_dim

                # Take the relevant slice of the weight matrix
                try:
                    head_pre = pre_param[:, start:end] if pre_param.dim() > 1 else pre_param[start:end]
                    head_jump = jump_param[:, start:end] if jump_param.dim() > 1 else jump_param[start:end]
                    head_post = post_param[:, start:end] if post_param.dim() > 1 else post_param[start:end]

                    head_key = f'layer_{layer_idx}_head_{h}'

                    head_changes[head_key]['pre_to_jump'] += torch.norm(head_jump - head_pre).item()
                    head_changes[head_key]['jump_to_post'] += torch.norm(head_post - head_jump).item()
                except:
                    pass

        # 3. Output projection - common across architectures
        elif 'out_proj.weight' in name or any(term in name for term in ['attn_output', 'o_proj']):
            for h in range(model.num_heads):
                start = h * head_dim
                end = (h + 1) * head_dim

                try:
                    head_key = f'layer_{layer_idx}_head_{h}'

                    # Output projection maps from head dimension back to model dimension
                    # The slice depends on the matrix orientation
                    if pre_param.shape[1] >= end:  # Check if we can slice on dim 1
                        head_changes[head_key]['pre_to_jump'] += torch.norm(
                            jump_param[:, start:end] - pre_param[:, start:end]
                        ).item()
                        head_changes[head_key]['jump_to_post'] += torch.norm(
                            post_param[:, start:end] - jump_param[:, start:end]
                        ).item()
                    elif pre_param.shape[0] >= end:  # Try dim 0 instead
                        head_changes[head_key]['pre_to_jump'] += torch.norm(
                            jump_param[start:end] - pre_param[start:end]
                        ).item()
                        head_changes[head_key]['jump_to_post'] += torch.norm(
                            post_param[start:end] - jump_param[start:end]
                        ).item()
                except:
                    pass

    def _normalize_changes(self, layer_changes, head_changes):
        """Normalize changes to highlight relative importance"""
        if layer_changes:
            max_layer_change = max(item['pre_to_jump'] for item in layer_changes.values())
            for layer, changes in layer_changes.items():
                changes['normalized_change'] = changes['pre_to_jump'] / (max_layer_change + 1e-8)

                # Additional relative-to-size normalization
                layer_idx = int(layer.split('_')[1])
                param_count = 0
                for name, param in self.model.named_parameters():
                    if f'layers.{layer_idx}.' in name:
                        param_count += param.numel()

                if param_count > 0:
                    changes['size_normalized_change'] = changes['pre_to_jump'] / (param_count ** 0.5)

        if head_changes:
            max_head_change = max(item['pre_to_jump'] for item in head_changes.values())
            for head, changes in head_changes.items():
                changes['normalized_change'] = changes['pre_to_jump'] / (max_head_change + 1e-8)

    def _find_top_changing_components(self, component_changes):
        """Find top changing components by category with enhanced filters"""
        top_changing = {}

        for component_type, params in component_changes.items():
            # Calculate average change magnitude for this component type
            avg_change = np.mean([v['pre_to_jump'] for v in params.values()]) if params else 0

            # Filter for parameters with significant changes (above average)
            significant_params = {k: v for k, v in params.items() if v['pre_to_jump'] > avg_change}

            # Sort by change magnitude
            sorted_params = sorted(significant_params.items(), key=lambda x: x[1]['pre_to_jump'], reverse=True)

            # Take top 5 or fewer if not enough
            top_changing[component_type] = sorted_params[:min(5, len(sorted_params))]

        return top_changing

    def _check_for_jumps(self, epoch, current_velocity_norm):
        """
        Improved jump detection that accounts for transformer's multi-step changes
        """
        # Get recent velocity norms with wider window
        recent_norms = [v[2] for v in self.velocities[-self.jump_detection_window:]]

        # Use exponential moving average for smoother baseline
        if not hasattr(self, 'ema_norm'):
            self.ema_norm = np.mean(recent_norms[:min(10, len(recent_norms))])

        # Update EMA (alpha=0.9 for strong smoothing)
        self.ema_norm = 0.9 * self.ema_norm + 0.1 * current_velocity_norm

        # Calculate robust statistics
        median_norm = np.median(recent_norms[:-1])  # Median is more robust than mean
        q75 = np.percentile(recent_norms[:-1], 75)
        q25 = np.percentile(recent_norms[:-1], 25)
        iqr = q75 - q25  # Interquartile range for robust variability measure

        # Calculate robust z-score using median and IQR
        robust_z_score = (current_velocity_norm - median_norm) / (
                iqr + 1e-8) * 0.7413  # Scale to approximate standard z-score

        # Define minimum velocity threshold based on model size
        model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        size_factor = np.log10(model_size) / 6.0  # Normalize by log of parameter count
        minimum_velocity_threshold = 0.3 * size_factor  # Scale threshold by model size

        # Multi-level jump detection:
        jump_threshold_multiplier_strong = 1.25  # 1.5
        jump_threshold_multiplier_medium = 0.7 * 3.0 / 5.0  # 0.7
        minimum_velocity_threshold_multiplier = 2.0 * (3.0 / 5.0)  # 2.0
        # 1. Strong jumps: High z-score and absolute magnitude
        is_strong_jump = ((robust_z_score > self.jump_threshold * jump_threshold_multiplier_strong)
                          and (
                                      current_velocity_norm > minimum_velocity_threshold * minimum_velocity_threshold_multiplier))

        # 2. Medium jumps: Moderate z-score sustained over multiple steps
        if hasattr(self, 'sustained_counter') and self.sustained_counter > 0:
            self.sustained_counter -= 1
            is_medium_jump = (
                                         robust_z_score > self.jump_threshold * jump_threshold_multiplier_medium) and self.sustained_counter == 0
        else:
            self.sustained_counter = 0
            is_medium_jump = False

        if robust_z_score > self.jump_threshold * jump_threshold_multiplier_medium and self.sustained_counter == 0:
            self.sustained_counter = 3  # Count 3 consecutive epochs with elevated velocity

        # 3. Gradual jumps: Consistent deviation from baseline
        if not hasattr(self, 'cumulative_deviation'):
            self.cumulative_deviation = 0.0

        # Update cumulative deviation from EMA baseline
        deviation = current_velocity_norm - self.ema_norm
        self.cumulative_deviation = jump_threshold_multiplier_medium * self.cumulative_deviation + 0.3 * deviation

        # Detect gradual jump when cumulative deviation crosses threshold
        is_gradual_jump = self.cumulative_deviation > minimum_velocity_threshold * 1.5

        # Reset once detected
        if is_gradual_jump:
            self.cumulative_deviation = 0.0

        # Combine jump detection criteria
        is_jump = is_strong_jump or is_medium_jump or is_gradual_jump

        # Enforce minimum distance between detected jumps (increased to 15 epochs)
        if is_jump and self.last_jump_epoch is not None:
            min_jump_distance = 15
            if epoch - self.last_jump_epoch < min_jump_distance:
                is_jump = False  # Too close to the last jump, ignore

        if is_jump:
            # Tag the type of jump detected
            jump_type = "strong" if is_strong_jump else "medium" if is_medium_jump else "gradual"

            # Update last jump epoch
            self.last_jump_epoch = epoch

            # info mark this epoch as a jump point
            self.detected_jumps.append({
                'epoch': epoch,
                'jump_type': jump_type,
                'velocity_norm': current_velocity_norm,
                'z_score': robust_z_score,
                'median_norm': median_norm,
                'iqr': iqr,
                'cumulative_deviation': getattr(self, 'cumulative_deviation', 0.0)
            })

            # info log the jump
            if self.logger:
                self.logger.log_data('weight_space_jumps', 'jump_epochs', epoch)
                # self.logger.log_data('weight_space_jumps', 'jump_velocity_norms', current_velocity_norm)
                self.logger.log_data('weight_space_jumps', 'jump_z_scores', robust_z_score)
                self.logger.log_data('weight_space_jumps', 'jump_types', jump_type)

            return True

        return False

    def _analyze_jump(self, jump_epoch, z_score):
        """Perform detailed analysis of a detected jump"""
        self.jump_counter += 1
        jump_id = f"jump_{self.jump_counter}_{jump_epoch}"

        # Find the index of the jump epoch in our weight snapshots
        jump_indices = [i for i, e in enumerate(self.weight_timestamps) if e == jump_epoch]

        if not jump_indices:
            print(f"\tEnhancedWeightSpaceTracker._analyze_jump: Warning: Jump detected at epoch {jump_epoch} but no corresponding weight snapshot found")
            return

        jump_idx = jump_indices[0]

        # We need snapshots before and after the jump for comparison
        if jump_idx == 0 or jump_idx >= len(self.weight_snapshots) - 1:
            print(f"\tEnhancedWeightSpaceTracker._analyze_jump Warning: Jump at index {jump_idx} is at the boundary of available snapshots")
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

            plt.suptitle(f"{self.model.plot_prefix}")
            # Save the figure
            plt.tight_layout()
            fig.savefig(self.jump_dir / f"{jump_id}_pca_trajectory.png")
            plt.close(fig)

            # Create layer-specific analysis
            self._visualize_layer_changes(jump_id)

    def visualize_jump_characterization(self, jump_char, save_prefix=None):
        """Enhanced visualization of jump characterization with structural changes"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        # Create directory for visualizations
        viz_dir = self.save_dir / "jump_characterization"
        viz_dir.mkdir(exist_ok=True, parents=True)

        # File name prefix (based on jump epoch)
        prefix = save_prefix or f"jump_{jump_char['jump_epoch']}"

        # Add SVD-based structural change visualization
        if 'svd_changes' in jump_char and jump_char['svd_changes']:
            fig_svd, ax_svd = plt.subplots(figsize=(12, 8))

            # Extract structural changes
            svd_data = []
            for name, changes in jump_char['svd_changes'].items():
                # Simplify parameter name for display
                short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
                if len(short_name) > 20:
                    short_name = short_name[:18] + '..'

                svd_data.append({
                    'Parameter': short_name,
                    'Singular Value Change': changes['singular_value_change'],
                    'Direction Change': changes['direction_change']
                })

            svd_df = pd.DataFrame(svd_data)

            # Sort by total change
            svd_df['Total Change'] = svd_df['Singular Value Change'] + svd_df['Direction Change']
            svd_df = svd_df.sort_values('Total Change', ascending=False).head(15)  # Top 15 parameters

            # Plot as grouped bar chart
            svd_melted = pd.melt(
                svd_df,
                id_vars=['Parameter'],
                value_vars=['Singular Value Change', 'Direction Change'],
                var_name='Change Type',
                value_name='Magnitude'
            )

            # Create grouped bar plot
            sns.barplot(
                data=svd_melted,
                x='Parameter',
                y='Magnitude',
                hue='Change Type',
                ax=ax_svd
            )

            ax_svd.set_title(
                f"Structural Changes at Jump {jump_char['jump_epoch']} ({jump_char['jump_type']} jump)\t{self.model.plot_prefix}")
            ax_svd.set_xlabel('Parameter')
            ax_svd.set_ylabel('Change Magnitude')
            ax_svd.set_xticklabels(ax_svd.get_xticklabels(), rotation=45, ha='right')

            plt.suptitle(f"{self.model.plot_prefix}")
            plt.tight_layout()
            plt.savefig(viz_dir / f"{prefix}_structural_changes.png")
            plt.close(fig_svd)

        ###############################################################################################

        # info 1. component Change Visualization
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        # info extract component-level changes
        component_data = []
        for component, value in jump_char['total_magnitude'].items():
            if component != 'symmetry_ratio':  # info skip derived metrics
                component_data.append({
                    'Component': 'Total',
                    'Change Type': component,
                    'Magnitude': value
                })

        # info add component-type level changes
        for component_type, params in jump_char['component_changes'].items():
            if params:  # If there are changes for this component
                # info sum up changes for this component type
                pre_to_jump = sum(v['pre_to_jump'] for v in params.values())
                jump_to_post = sum(v['jump_to_post'] for v in params.values())

                component_data.append({
                    'Component': component_type,
                    'Change Type': 'pre_to_jump',
                    'Magnitude': pre_to_jump
                })
                component_data.append({
                    'Component': component_type,
                    'Change Type': 'jump_to_post',
                    'Magnitude': jump_to_post
                })

        df = pd.DataFrame(component_data)

        # info create grouped bar chart
        sns.barplot(x='Component', y='Magnitude', hue='Change Type', data=df, ax=ax1)
        ax1.set_title(f'Component Changes at Jump Epoch {jump_char["jump_epoch"]}')
        ax1.set_ylabel('Change Magnitude')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

        plt.suptitle(f"{self.model.plot_prefix}")
        plt.tight_layout()
        plt.savefig(viz_dir / f"{prefix}_component_changes.png")
        plt.close(fig1)

        # info 2. Layer and Head Change Visualization
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 6))

        # info extract layer-level changes
        layer_data = []
        for layer, changes in jump_char['layer_changes'].items():
            layer_data.append({
                'Layer': layer,
                'Change Type': 'pre_to_jump',
                'Magnitude': changes['pre_to_jump'],
                'Normalized': changes.get('normalized_change', 0)
            })
            layer_data.append({
                'Layer': layer,
                'Change Type': 'jump_to_post',
                'Magnitude': changes['jump_to_post']
            })

        # info convert to DataFrame and sort by pre_to_jump magnitude
        layer_df = pd.DataFrame(layer_data)
        layer_df = layer_df.sort_values(by=['Change Type', 'Magnitude'], ascending=[True, False])

        # info only show top 8 layers for clarity
        top_layers = layer_df[layer_df['Change Type'] == 'pre_to_jump'].head(8)['Layer'].unique()
        layer_df_filtered = layer_df[layer_df['Layer'].isin(top_layers)]

        # info create layer bar chart
        sns.barplot(x='Layer', y='Magnitude', hue='Change Type', data=layer_df_filtered, ax=ax2)
        ax2.set_title(f'Layer Changes at Jump Epoch {jump_char["jump_epoch"]} {self.model.plot_prefix}')
        ax2.set_ylabel('Change Magnitude')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        # info extract head-level changes
        head_data = []
        for head, changes in jump_char['head_changes'].items():
            head_data.append({
                'Head': head,
                'Change Type': 'pre_to_jump',
                'Magnitude': changes['pre_to_jump'],
                'Normalized': changes.get('normalized_change', 0)
            })
            head_data.append({
                'Head': head,
                'Change Type': 'jump_to_post',
                'Magnitude': changes['jump_to_post']
            })

        # info convert to DataFrame and sort by pre_to_jump magnitude
        head_df = pd.DataFrame(head_data)
        head_df = head_df.sort_values(by=['Change Type', 'Magnitude'], ascending=[True, False])

        # info only show top 8 heads for clarity
        top_heads = head_df[head_df['Change Type'] == 'pre_to_jump'].head(8)['Head'].unique()
        head_df_filtered = head_df[head_df['Head'].isin(top_heads)]

        # info create head bar chart
        sns.barplot(x='Head', y='Magnitude', hue='Change Type', data=head_df_filtered, ax=ax3)
        ax3.set_title(f'Head Changes at Jump Epoch {jump_char["jump_epoch"]} {self.model.plot_prefix}')
        ax3.set_ylabel('Change Magnitude')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')

        plt.suptitle(f"{self.model.plot_prefix}")
        plt.tight_layout()
        plt.savefig(viz_dir / f"{prefix}_layer_head_changes.png")
        plt.close(fig2)

        # info 3. Spectral Change Visualization (if available)
        if 'spectral_changes' in jump_char and jump_char['spectral_changes']:
            fig3, ax4 = plt.subplots(figsize=(12, 6))

            # info extract spectral changes
            spectral_data = []
            for component, changes in jump_char['spectral_changes'].items():
                for metric, value in changes.items():
                    spectral_data.append({
                        'Component': component,
                        'Metric': metric,
                        'Change': value
                    })

            # info convert to DataFrame
            spectral_df = pd.DataFrame(spectral_data)

            # info filter to include only significant changes
            spectral_df = spectral_df[abs(spectral_df['Change']) > 0.01]

            if not spectral_df.empty:
                # info create spectral change bar chart
                sns.barplot(x='Component', y='Change', hue='Metric', data=spectral_df, ax=ax4)
                ax4.set_title(f'Spectral Changes at Jump Epoch {jump_char["jump_epoch"]} {self.model.plot_prefix}')
                ax4.set_ylabel('Change Value')
                ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

                plt.suptitle(f"{self.model.plot_prefix}")
                plt.tight_layout()
                plt.savefig(viz_dir / f"{prefix}_spectral_changes.png")

            plt.close(fig3)
        ###############################################################################################################
        # Add a new visualization for size-normalized changes
        if any('size_normalized_change' in changes for changes in jump_char['layer_changes'].values()):
            fig_size, ax_size = plt.subplots(figsize=(10, 6))

            # Extract data
            size_data = []
            for layer, changes in jump_char['layer_changes'].items():
                if 'size_normalized_change' in changes:
                    size_data.append({
                        'Layer': layer,
                        'Size-Normalized Change': changes['size_normalized_change']
                    })

            if size_data:
                size_df = pd.DataFrame(size_data)

                # Sort by normalized change
                size_df = size_df.sort_values('Size-Normalized Change', ascending=False)

                # Plot bar chart
                sns.barplot(
                    data=size_df,
                    x='Layer',
                    y='Size-Normalized Change',
                    ax=ax_size
                )

                ax_size.set_title(f"Parameter Size-Normalized Changes at Jump {jump_char['jump_epoch']}")
                ax_size.set_xlabel('Layer')
                ax_size.set_ylabel('Size-Normalized Change')

                plt.suptitle(f"{self.model.plot_prefix}")
                plt.tight_layout()
                plt.savefig(viz_dir / f"{prefix}_size_normalized_changes.png")
                plt.close(fig_size)

        return viz_dir

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
        plt.suptitle(f"{self.model.plot_prefix}")
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

    def track_parameter_group_velocities(self, epoch, current_state_dict):
        """
        Track velocities for different parameter groups to identify which
        components are changing most rapidly.
        """
        if not hasattr(self, 'prev_state_dict') or self.prev_state_dict is None:
            self.prev_state_dict = {k: v.clone().detach() for k, v in current_state_dict.items()}
            self.group_velocities_history = defaultdict(list)
            return None

        # Define parameter groups
        groups = {
            'attention_query': [],
            'attention_key': [],
            'attention_value': [],
            'attention_output': [],
            'mlp_up': [],
            'mlp_down': [],
            'layernorm': [],
            'embedding': []
        }

        # Track per-head and per-layer metrics
        head_velocities = defaultdict(list)
        layer_velocities = defaultdict(list)

        # Calculate velocities by group
        for name, param in current_state_dict.items():
            if name not in self.prev_state_dict:
                continue

            # Calculate velocity norm
            velocity = param - self.prev_state_dict[name]
            velocity_norm = torch.norm(velocity).item()

            # Get layer index if applicable
            layer_idx = None
            if 'layers.' in name:
                parts = name.split('.')
                try:
                    layer_idx = int(parts[1])
                    layer_velocities[f'layer_{layer_idx}'].append(velocity_norm)
                except:
                    pass

            # Categorize by parameter type
            if 'attn' in name and 'in_proj' in name:
                # Split query/key/value based on dimension
                if hasattr(self.model, 'dim') and hasattr(self.model, 'num_heads'):
                    dim = self.model.dim
                    # Standard MultiheadAttention implementation has QKV stacked
                    if 'weight' in name:
                        q_weights = velocity[:dim]
                        k_weights = velocity[dim:2 * dim]
                        v_weights = velocity[2 * dim:]

                        groups['attention_query'].append(torch.norm(q_weights).item())
                        groups['attention_key'].append(torch.norm(k_weights).item())
                        groups['attention_value'].append(torch.norm(v_weights).item())

                        # Track per-head velocities if possible
                        head_dim = dim // self.model.num_heads
                        for h in range(self.model.num_heads):
                            start = h * head_dim
                            end = (h + 1) * head_dim
                            if layer_idx is not None:
                                head_velocities[f'layer_{layer_idx}_head_{h}_q'].append(
                                    torch.norm(q_weights[start:end]).item())
                                head_velocities[f'layer_{layer_idx}_head_{h}_k'].append(
                                    torch.norm(k_weights[start:end]).item())
                                head_velocities[f'layer_{layer_idx}_head_{h}_v'].append(
                                    torch.norm(v_weights[start:end]).item())
                else:
                    groups['attention_query'].append(velocity_norm)
            elif 'attn' in name and 'out_proj' in name:
                groups['attention_output'].append(velocity_norm)

                # Track per-head output velocities
                if 'weight' in name and hasattr(self.model, 'dim') and hasattr(self.model, 'num_heads'):
                    head_dim = self.model.dim // self.model.num_heads
                    for h in range(self.model.num_heads):
                        start = h * head_dim
                        end = (h + 1) * head_dim
                        if layer_idx is not None:
                            head_velocity = torch.norm(velocity[:, start:end]).item()
                            head_velocities[f'layer_{layer_idx}_head_{h}_out'].append(head_velocity)
            elif 'mlp.0' in name or 'mlp.up_proj' in name:
                groups['mlp_up'].append(velocity_norm)
            elif 'mlp.2' in name or 'mlp.down_proj' in name:
                groups['mlp_down'].append(velocity_norm)
            elif 'ln' in name or 'norm' in name:
                groups['layernorm'].append(velocity_norm)
            elif 'embed' in name:
                groups['embedding'].append(velocity_norm)

        # Calculate average velocity for each group
        group_velocities = {k: np.mean(v) if v else 0.0 for k, v in groups.items()}
        layer_group_velocities = {k: np.mean(v) if v else 0.0 for k, v in layer_velocities.items()}
        head_group_velocities = {k: np.mean(v) if v else 0.0 for k, v in head_velocities.items()}

        # Store history
        for k, v in group_velocities.items():
            self.group_velocities_history[k].append((epoch, v))
        for k, v in layer_group_velocities.items():
            self.group_velocities_history[k].append((epoch, v))
        for k, v in head_group_velocities.items():
            self.group_velocities_history[k].append((epoch, v))

        # Update previous state
        self.prev_state_dict = {k: v.clone().detach() for k, v in current_state_dict.items()}

        return {
            'group_velocities': group_velocities,
            'layer_velocities': layer_group_velocities,
            'head_velocities': head_group_velocities
        }

    def track_spectral_properties(self, epoch, current_state_dict):
        """
        Track spectral properties (singular values) of weight matrices to understand
        how the representation space is evolving.
        """
        spectral_stats = {}

        # Track layer-wise stats
        for layer_idx in range(self.model.num_layers):
            # Attention weights
            for component in ['in_proj_weight', 'out_proj.weight']:
                key = f'layers.{layer_idx}.attn.{component}'
                if key in current_state_dict:
                    matrix = current_state_dict[key].detach().cpu().numpy()

                    # Special handling for in_proj_weight (QKV stacked)
                    if 'in_proj_weight' in key and hasattr(self.model, 'dim'):
                        dim = self.model.dim
                        q_weights = matrix[:dim]
                        k_weights = matrix[dim:2 * dim]
                        v_weights = matrix[2 * dim:]

                        # Analyze each component separately
                        for name, weights in [('query', q_weights), ('key', k_weights), ('value', v_weights)]:
                            try:
                                u, s, vh = np.linalg.svd(weights, full_matrices=False)
                                spectral_stats[f'layer_{layer_idx}_attn_{name}'] = {
                                    'singular_values': s,
                                    'condition_number': s[0] / (s[-1] + 1e-8),
                                    'effective_rank': np.sum(s) / (s[0] + 1e-8)
                                }
                            except:
                                pass
                    else:
                        try:
                            u, s, vh = np.linalg.svd(matrix, full_matrices=False)
                            spectral_stats[f'layer_{layer_idx}_attn_output'] = {
                                'singular_values': s,
                                'condition_number': s[0] / (s[-1] + 1e-8),
                                'effective_rank': np.sum(s) / (s[0] + 1e-8)
                            }
                        except:
                            pass

            # MLP weights
            for idx, component in [(0, 'up'), (2, 'down')]:
                key = f'layers.{layer_idx}.mlp.{idx}.weight'
                if key in current_state_dict:
                    matrix = current_state_dict[key].detach().cpu().numpy()
                    try:
                        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
                        spectral_stats[f'layer_{layer_idx}_mlp_{component}'] = {
                            'singular_values': s,
                            'condition_number': s[0] / (s[-1] + 1e-8),
                            'effective_rank': np.sum(s) / (s[0] + 1e-8)
                        }
                    except:
                        pass

        # Store history if not already tracking
        if not hasattr(self, 'spectral_history'):
            self.spectral_history = defaultdict(list)

        # Track key metrics over time
        for component, stats in spectral_stats.items():
            self.spectral_history[f'{component}_condition'].append((epoch, stats['condition_number']))
            self.spectral_history[f'{component}_rank'].append((epoch, stats['effective_rank']))

        return spectral_stats

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
            ax1.axvline(x=jump['epoch'], color='red', linestyle='--', lw=1.25, alpha=0.7)
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

        plt.suptitle(f"{self.model.plot_prefix}")
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
                                f'epoch {epoch}', fontsize=9
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
                                label = f'grok {epoch}' if key == 'grok' else f'{epoch}'
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

            ax2.set_title(f'Weight Space Velocity and Acceleration')

        plt.suptitle(f"{self.model.plot_prefix}")
        plt.tight_layout()
        save_path = self.save_dir / f'weight_trajectory_with_jumps_dims_{selected_dims[0]}_{selected_dims[1]}.png'
        plt.savefig(save_path)
        plt.close()

        return save_path

    def analyze_phase_weight_spaces(self, phase_analyzer, eval_loader=None, n_components=3):
        """
        Analyze weight spaces at key phase transition points using PCA/SVD.

        Args:
            phase_analyzer: PhaseTransitionAnalyzer containing phase information
            eval_loader: Optional evaluation data loader for functional tests
            n_components: Number of principal components to analyze

        Returns:
            dict: Analysis results by phase and transition
        """
        # Ensure PCA is fitted
        self._ensure_pca_fitted()

        if not self.pca_fitted:
            print("\tEnhancedWeightSpaceTracker.analyze_phase_weight_spaces Not enough data to fit PCA. Need at least 3 weight snapshots.")
            return None

        # Get phase structure and transitions from analyzer
        transitions = phase_analyzer.detected_transitions
        # warning this dictionary is redundant - phase structure could simply be a list,
        #  but then reduce the `if phase_structure and 'phases' in phase_structure:`
        #  to simple looking through the list
        phase_structure = {}
        phase_structure['phases'] = phase_analyzer.phase_structure

        # Also check for grokking points
        grokking_epochs = []
        if hasattr(phase_analyzer.model, 'logger') and phase_analyzer.model.logger:
            logger = phase_analyzer.model.logger
            if 'grokking_phases' in logger.logs and 'grokking_step' in logger.logs['grokking_phases']:
                grokking_step = logger.logs['grokking_phases']['grokking_step']
                if isinstance(grokking_step, list):
                    grokking_epochs.extend(grokking_step)
                else:
                    grokking_epochs.append(grokking_step)

        # Initialize results
        analysis_results = {
            'transitions': {},
            'phases': {},
            'grokking_points': {},
            'functional_analysis': {}
        }

        # Analyze transition points
        for transition in transitions:
            transition_epoch = transition['epoch']
            # Find closest weight snapshot
            closest_snapshot_idx = self._find_closest_snapshot_idx(transition_epoch)
            if closest_snapshot_idx is not None:
                analysis_results['transitions'][transition_epoch] = self._analyze_weight_snapshot(
                    closest_snapshot_idx, transition_epoch, n_components,
                    label=f"Transition: {', '.join(transition['transition_types'])}"
                )

        # Analyze phase midpoints
        if phase_structure and 'phases' in phase_structure:
            for i, phase in enumerate(phase_structure['phases']):
                phase_start = phase['start_epoch']
                phase_end = phase['end_epoch']
                phase_mid = (phase_start + phase_end) // 2

                closest_snapshot_idx = self._find_closest_snapshot_idx(phase_mid)
                if closest_snapshot_idx is not None:
                    phase_classification = phase.get('classification', 'unknown')
                    analysis_results['phases'][f"phase_{i + 1}"] = self._analyze_weight_snapshot(
                        closest_snapshot_idx, phase_mid, n_components,
                        label=f"Phase {i + 1}: {phase_classification.title()}"
                    )

        # Analyze grokking points
        for i, grok_epoch in enumerate(grokking_epochs):
            closest_snapshot_idx = self._find_closest_snapshot_idx(grok_epoch)
            if closest_snapshot_idx is not None:
                analysis_results['grokking_points'][f"grokking_{i + 1}"] = self._analyze_weight_snapshot(
                    closest_snapshot_idx, grok_epoch, n_components,
                    label=f"Grokking Point {i + 1}"
                )

        # Perform functional analysis if eval_loader is provided
        if eval_loader is not None:
            # Test model function at different key points
            key_epochs = []
            # Add phase transition points
            key_epochs.extend([t['epoch'] for t in transitions])
            # Add grokking points
            key_epochs.extend(grokking_epochs)
            # Add initial and final states
            if self.weight_timestamps:
                key_epochs.extend([self.weight_timestamps[0], self.weight_timestamps[-1]])

            key_epochs = sorted(list(set(key_epochs)))  # Remove duplicates and sort

            # Perform functional analysis at each key epoch
            for epoch in key_epochs:
                closest_snapshot_idx = self._find_closest_snapshot_idx(epoch)
                if closest_snapshot_idx is not None:
                    func_analysis = self._perform_functional_analysis(
                        closest_snapshot_idx, epoch, eval_loader
                    )
                    analysis_results['functional_analysis'][epoch] = func_analysis

        # Create comprehensive visualization
        self._visualize_phase_weight_spaces(analysis_results)

        return analysis_results

    def _find_closest_snapshot_idx(self, target_epoch):
        """Find the index of the snapshot closest to the target epoch"""
        if not self.weight_timestamps:
            return None

        distances = [abs(epoch - target_epoch) for epoch in self.weight_timestamps]
        closest_idx = distances.index(min(distances))
        return closest_idx

    def _analyze_weight_snapshot(self, snapshot_idx, target_epoch, n_components=3, label=None):
        """Perform detailed PCA and SVD analysis on a specific weight snapshot"""
        # Get the flattened weights
        flattened_weights = self.flattened_weights[snapshot_idx]

        # Get the projection in PCA space
        pca_coords = self.pca.transform([flattened_weights])[0]

        # Get the actual snapshot
        snapshot = self.weight_snapshots[snapshot_idx]
        state_dict = snapshot['state_dict']

        # Perform SVD on major weight matrices
        svd_results = {}

        # Analyze key weight matrices (attention and MLP)
        for name, param in state_dict.items():
            # Skip non-weight parameters and small matrices
            if 'weight' not in name or len(param.shape) < 2 or min(param.shape) < 3:
                continue

            try:
                # Perform SVD
                u, s, vh = torch.linalg.svd(param, full_matrices=False)

                # Calculate key metrics
                total_var = torch.sum(s ** 2).item()
                explained_var = [(sing ** 2 / total_var).item() for sing in s[:10]]  # First 10 components
                condition_number = (s[0] / s[-1]).item() if s[-1] > 0 else float('inf')
                effective_rank = torch.sum(s) / s[0] if s[0] > 0 else 0
                effective_rank = effective_rank.item()

                svd_results[name] = {
                    'singular_values': s.detach().cpu().numpy()[:10].tolist(),  # First 10
                    'explained_variance': explained_var,
                    'condition_number': condition_number,
                    'effective_rank': effective_rank
                }
            except:
                # Skip matrices that cause SVD to fail
                continue

        # Calculate weight norms by layer
        layer_norms = {}
        for layer_idx in range(self.model.num_layers):
            # Attention weights
            qkv_norm = 0
            out_norm = 0
            mlp_norm = 0

            for name, param in state_dict.items():
                if f'layers.{layer_idx}.attn.in_proj' in name:
                    qkv_norm += torch.norm(param).item()
                elif f'layers.{layer_idx}.attn.out_proj' in name:
                    out_norm += torch.norm(param).item()
                elif f'layers.{layer_idx}.mlp.' in name and 'weight' in name:
                    mlp_norm += torch.norm(param).item()

            layer_norms[f'layer_{layer_idx}'] = {
                'attention_qkv_norm': qkv_norm,
                'attention_out_norm': out_norm,
                'mlp_norm': mlp_norm,
                'total_norm': qkv_norm + out_norm + mlp_norm
            }

        return {
            'epoch': target_epoch,
            'snapshot_epoch': self.weight_timestamps[snapshot_idx],
            'distance_to_target': abs(self.weight_timestamps[snapshot_idx] - target_epoch),
            'pca_coordinates': pca_coords[:n_components].tolist(),
            'pca_explained_variance': self.pca.explained_variance_ratio_[:n_components].tolist(),
            'layer_weight_norms': layer_norms,
            'svd_analysis': svd_results,
            'label': label
        }

    def _perform_functional_analysis(self, snapshot_idx, target_epoch, eval_loader):
        """Analyze model function by testing on the evaluation data"""
        # Store original state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        try:
            # Load the snapshot state
            snapshot = self.weight_snapshots[snapshot_idx]
            self.model.load_state_dict(snapshot['state_dict'])

            # Evaluate performance
            self.model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in eval_loader:
                    inputs = inputs.to(next(self.model.parameters()).device)
                    targets = targets.to(next(self.model.parameters()).device)

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            accuracy = correct / total if total > 0 else 0

            # Get attention patterns for a sample input
            sample_input = next(iter(eval_loader))[0].to(next(self.model.parameters()).device)
            _ = self.model(sample_input, store_attention=True)
            attention_patterns = self.model.get_attention_patterns()

            # Calculate entropy of attention patterns
            attention_entropy = {}
            for head_name, pattern in attention_patterns.items():
                # Calculate entropy
                probs = pattern.flatten()
                probs = probs / (torch.sum(probs) + 1e-10)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                attention_entropy[head_name] = entropy

            return {
                'epoch': target_epoch,
                'accuracy': accuracy,
                'attention_entropy': attention_entropy,
                'attention_entropy_avg': sum(attention_entropy.values()) / len(
                    attention_entropy) if attention_entropy else 0
            }

        finally:
            # Restore original state
            self.model.load_state_dict(original_state)

    def _visualize_phase_weight_spaces(self, analysis_results):
        """Create visualizations of weight spaces across different phases"""
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Create PCA trajectory visualization
        fig1 = plt.figure(figsize=(12, 10))
        ax1 = fig1.add_subplot(111, projection='3d')

        # Plot full PCA trajectory as background
        if self.pca_fitted and len(self.flattened_weights) >= 3:
            projected = self.pca.transform(np.stack(self.flattened_weights))
            ax1.plot(projected[:, 0], projected[:, 1], projected[:, 2],
                     'o-', alpha=0.2, color='gray', markersize=3)

        # Plot transitions in red
        for epoch, data in analysis_results['transitions'].items():
            coords = data['pca_coordinates']
            ax1.scatter(coords[0], coords[1], coords[2], color='red', s=100, alpha=0.8)
            ax1.text(coords[0], coords[1], coords[2], f"T:{epoch}", color='red')

        # Plot phase midpoints in blue
        for phase_id, data in analysis_results['phases'].items():
            coords = data['pca_coordinates']
            ax1.scatter(coords[0], coords[1], coords[2], color='blue', s=100, alpha=0.8)
            ax1.text(coords[0], coords[1], coords[2], phase_id, color='blue')

        # Plot grokking points in green
        for grok_id, data in analysis_results['grokking_points'].items():
            coords = data['pca_coordinates']
            ax1.scatter(coords[0], coords[1], coords[2], color='green', s=100, alpha=0.8)
            ax1.text(coords[0], coords[1], coords[2], f"G:{data['epoch']}", color='green')

        # Set labels with variance explained
        explained_var = self.pca.explained_variance_ratio_[:3] * 100
        ax1.set_xlabel(f'PC1 ({explained_var[0]:.1f}%)')
        ax1.set_ylabel(f'PC2 ({explained_var[1]:.1f}%)')
        ax1.set_zlabel(f'PC3 ({explained_var[2]:.1f}%)')
        ax1.set_title('Weight Space Trajectory with Phase Transitions')

        plt.tight_layout()
        plt.suptitle(f"{self.model.plot_prefix}")
        plt.savefig(self.save_dir / "phase_weight_spaces_3d.png")
        plt.close(fig1)

        # 2. Create layer norm comparison across phases
        if analysis_results['phases']:
            # Collect data for all layers
            layer_data = []
            for phase_id, data in analysis_results['phases'].items():
                for layer_id, norms in data['layer_weight_norms'].items():
                    layer_data.append({
                        'phase': phase_id,
                        'layer': layer_id,
                        'attn_qkv': norms['attention_qkv_norm'],
                        'attn_out': norms['attention_out_norm'],
                        'mlp': norms['mlp_norm'],
                        'total': norms['total_norm'],
                        'epoch': data['epoch']
                    })

            # Create visualization
            if layer_data:
                import pandas as pd
                import seaborn as sns

                df = pd.DataFrame(layer_data)

                fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

                # Plot attention QKV norms
                sns.barplot(x='layer', y='attn_qkv', hue='phase', data=df, ax=axes[0, 0])
                axes[0, 0].set_title('Attention QKV Weight Norms by Phase')

                # Plot attention output norms
                sns.barplot(x='layer', y='attn_out', hue='phase', data=df, ax=axes[0, 1])
                axes[0, 1].set_title('Attention Output Weight Norms by Phase')

                # Plot MLP norms
                sns.barplot(x='layer', y='mlp', hue='phase', data=df, ax=axes[1, 0])
                axes[1, 0].set_title('MLP Weight Norms by Phase')

                # Plot total norms
                sns.barplot(x='layer', y='total', hue='phase', data=df, ax=axes[1, 1])
                axes[1, 1].set_title('Total Layer Weight Norms by Phase')

                plt.suptitle(f"{self.model.plot_prefix}")
                plt.tight_layout()
                plt.savefig(self.save_dir / "phase_layer_norms_comparison.png")
                plt.close(fig2)

        # 3. Create SVD analysis visualization
        svd_data = []
        for key_type in ['transitions', 'phases', 'grokking_points']:
            for key, data in analysis_results[key_type].items():
                if 'svd_analysis' in data:
                    for layer_name, svd_results in data['svd_analysis'].items():
                        if 'effective_rank' in svd_results and 'condition_number' in svd_results:
                            svd_data.append({
                                'key_type': key_type,
                                'key': key,
                                'epoch': data['epoch'],
                                'layer': layer_name,
                                'effective_rank': svd_results['effective_rank'],
                                'condition_number': min(svd_results['condition_number'], 1000),
                                # Cap for better visualization
                                'explained_var_ratio': svd_results['explained_variance'][0]  # Top component
                            })

        if svd_data:
            import pandas as pd
            import seaborn as sns

            df = pd.DataFrame(svd_data)

            # Focus on specific layer types for clarity
            df['layer_type'] = df['layer'].apply(lambda x:
                                                 'attn_in' if 'attn.in_proj' in x else
                                                 'attn_out' if 'attn.out_proj' in x else
                                                 'mlp_up' if 'mlp.0.weight' in x else
                                                 'mlp_down' if 'mlp.2.weight' in x else 'other'
                                                 )

            # Filter only main component types
            df_filtered = df[df['layer_type'].isin(['attn_in', 'attn_out', 'mlp_up', 'mlp_down'])]

            fig3, axes = plt.subplots(2, 1, figsize=(14, 10))

            # Effective rank by epoch and layer type
            sns.lineplot(x='epoch', y='effective_rank', hue='layer_type',
                         data=df_filtered, marker='o', ax=axes[0])
            axes[0].set_title('Effective Rank Across Learning Phases')

            # Condition number by epoch and layer type
            sns.lineplot(x='epoch', y='condition_number', hue='layer_type',
                         data=df_filtered, marker='o', ax=axes[1])
            axes[1].set_title('Condition Number Across Learning Phases')

            # Add vertical lines for phase transitions
            for transition_epoch in analysis_results['transitions'].keys():
                axes[0].axvline(x=transition_epoch, color='r', linestyle='--', alpha=0.5)
                axes[1].axvline(x=transition_epoch, color='r', linestyle='--', alpha=0.5)

            # Add vertical lines for grokking points
            for grok_data in analysis_results['grokking_points'].values():
                grok_epoch = grok_data['epoch']
                axes[0].axvline(x=grok_epoch, color='g', linestyle='-', alpha=0.5)
                axes[1].axvline(x=grok_epoch, color='g', linestyle='-', alpha=0.5)

            plt.tight_layout()
            plt.suptitle(f"{self.model.plot_prefix}")
            plt.savefig(self.save_dir / "phase_svd_analysis.png")
            plt.close(fig3)

        # 4. Create functional analysis visualization if available
        if analysis_results['functional_analysis']:
            func_data = []
            for epoch, data in analysis_results['functional_analysis'].items():
                func_data.append({
                    'epoch': epoch,
                    'accuracy': data['accuracy'],
                    'attn_entropy_avg': data['attention_entropy_avg']
                })

            if func_data:
                import pandas as pd

                df = pd.DataFrame(func_data).sort_values('epoch')

                fig4, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

                # Plot accuracy
                axes[0].plot(df['epoch'], df['accuracy'], 'bo-')
                axes[0].set_ylabel('Accuracy')
                axes[0].set_title('Model Accuracy Across Learning Phases')

                # Plot attention entropy
                axes[1].plot(df['epoch'], df['attn_entropy_avg'], 'ro-')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Avg Attention Entropy')
                axes[1].set_title('Attention Pattern Entropy Across Learning Phases')

                # Add vertical lines for phase transitions
                for transition_epoch in analysis_results['transitions'].keys():
                    axes[0].axvline(x=transition_epoch, color='r', linestyle='--', alpha=0.5)
                    axes[1].axvline(x=transition_epoch, color='r', linestyle='--', alpha=0.5)

                # Add vertical lines for grokking points
                for grok_data in analysis_results['grokking_points'].values():
                    grok_epoch = grok_data['epoch']
                    axes[0].axvline(x=grok_epoch, color='g', linestyle='-', alpha=0.5)
                    axes[1].axvline(x=grok_epoch, color='g', linestyle='-', alpha=0.5)

                plt.tight_layout()
                plt.suptitle(f"{self.model.plot_prefix}")
                plt.savefig(self.save_dir / "phase_functional_analysis.png")
                plt.close(fig4)

    def cleanup(self):
        """Release memory held by various analyzers"""
        # Clear cached activations
        self.layer_activations = {}

        # Clear large stored tensors
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, dict) and any(isinstance(v, (torch.Tensor, np.ndarray))
                                              for v in attr.values() if hasattr(attr, 'values')):
                setattr(self, attr_name, {})

        # Call torch.cuda.empty_cache() if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Call cleanup on child analyzers
        for analyzer_name in ['mlp_sparsity_tracker', 'circuit_class_analyzer', 'interaction_analyzer']:
            if hasattr(self, analyzer_name):
                analyzer = getattr(self, analyzer_name)
                if hasattr(analyzer, 'cleanup'):
                    analyzer.cleanup()