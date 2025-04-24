# analyzers/attention_analyzer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from analysis.analyzers.base_analyzer import BaseAnalyzer


class AttentionAnalyzer(BaseAnalyzer):
    """Analyzer for attention patterns and entropy"""

    def _get_analysis_dir_name(self):
        return "attention_analysis"

    def analyze_attention_patterns_around_jump(self, jump_epoch, eval_loader, weight_tracker,
                                               ensure_after_snapshot=True):
        """
        Analyze attention patterns and entropy before, during, and after a jump

        Args:
            jump_epoch: The epoch of the jump
            eval_loader: Evaluation data loader
            weight_tracker: EnhancedWeightSpaceTracker instance
            ensure_after_snapshot: Whether to create a post-jump snapshot if needed

        Returns:
            dict: Attention analysis results
        """
        jump_id = f"jump_{jump_epoch}"

        # Find the jump snapshots
        snapshots = self._find_jump_snapshots(jump_epoch, weight_tracker, ensure_after_snapshot)
        if snapshots is None:
            return None

        pre_jump_snapshot, jump_snapshot, post_jump_snapshot = snapshots

        # Analyze attention for each snapshot
        results = {
            'jump_epoch': jump_epoch,
            'pre_jump_epoch': pre_jump_snapshot['epoch'],
            'jump_epoch': jump_snapshot['epoch'],
            'post_jump_epoch': post_jump_snapshot['epoch'],
            'attention_analysis': {}
        }

        # Store original model state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        try:
            # Analyze each state
            for state_name, snapshot in [
                ('pre_jump', pre_jump_snapshot),
                ('jump', jump_snapshot),
                ('post_jump', post_jump_snapshot)
            ]:
                # Load the state
                self.model.load_state_dict(snapshot['state_dict'])

                # Analyze attention entropy
                entropy = self.model.compute_attention_entropy(eval_loader)

                # Get attention patterns from a sample
                sample_input = next(iter(eval_loader))[0]
                device = next(self.model.parameters()).device
                sample_input = sample_input.to(device)

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

        # Log attention entropy changes
        if self.logger:
            for state_name in ['pre_jump', 'jump', 'post_jump']:
                for head, entropy_val in results['attention_analysis'][state_name]['entropy'].items():
                    self.logger.log_data('attention_entropy',
                                         f'jump_{jump_epoch}_{state_name}_{head}',
                                         float(entropy_val))

        return results

        # analyzers/attention_analyzer.py (continued)

    def _find_jump_snapshots(self, jump_epoch, weight_tracker, ensure_after_snapshot=True):
        """Find snapshots before, at, and after a jump"""
        # Find the jump index in weight_tracker snapshots
        jump_idx = None
        for i, epoch in enumerate(weight_tracker.weight_timestamps):
            if epoch == jump_epoch:
                jump_idx = i
                break

        if jump_idx is None:
            print(f"No snapshot found for jump at epoch {jump_epoch}")
            return None

        # Get snapshots around the jump
        before_idx = max(0, jump_idx - 1)

        # Check for after snapshot
        after_idx = min(len(weight_tracker.weight_timestamps) - 1, jump_idx + 1)
        after_epoch = weight_tracker.weight_timestamps[after_idx]

        # Handle case where after_idx points to jump_epoch
        if after_epoch == jump_epoch and ensure_after_snapshot:
            print(f"No post-jump snapshot available for epoch {jump_epoch}")
            return None

        pre_jump_snapshot = weight_tracker.weight_snapshots[before_idx]
        jump_snapshot = weight_tracker.weight_snapshots[jump_idx]
        post_jump_snapshot = weight_tracker.weight_snapshots[after_idx]

        return pre_jump_snapshot, jump_snapshot, post_jump_snapshot

    def _visualize_attention_comparison(self, results, jump_id):
        """Create visualizations comparing attention before/during/after a jump"""
        # Extract entropy data
        before_entropy = results['attention_analysis']['pre_jump']['entropy']
        jump_entropy = results['attention_analysis']['jump']['entropy']
        after_entropy = results['attention_analysis']['post_jump']['entropy']

        # Convert to DataFrame for plotting
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

        # 1. Entropy values visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Melt dataframe for seaborn
        entropy_melted = pd.melt(
            entropy_df,
            id_vars=['head'],
            value_vars=['before', 'jump', 'after'],
            var_name='state',
            value_name='entropy'
        )

        # Remap state labels
        state_labels = {
            'before': f'Before Jump (Epoch {results["pre_jump_epoch"]})',
            'jump': f'At Jump (Epoch {results["jump_epoch"]})',
            'after': f'After Jump (Epoch {results["post_jump_epoch"]})'
        }
        entropy_melted['state_label'] = entropy_melted['state'].map(state_labels)

        # Create barplot
        sns.barplot(
            data=entropy_melted,
            x='head',
            y='entropy',
            hue='state_label',
            ax=ax1
        )

        ax1.set_title('Attention Entropy Around Jump')
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

        # Create barplot
        sns.barplot(
            data=change_melted,
            x='head',
            y='change',
            hue='transition_label',
            ax=ax2
        )

        ax2.set_title('Attention Entropy Changes')
        ax2.set_xlabel('Head')
        ax2.set_ylabel('Entropy Change')
        ax2.legend(title='Transition')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / f"{jump_id}_entropy_comparison.png")
        plt.close(fig)

        # 3. Visualize attention patterns for each head
        self._visualize_attention_patterns(results, jump_id)

    def _visualize_attention_patterns(self, results, jump_id):
        """Create visualizations of attention patterns for each head"""
        pattern_sets = [
            ('pre_jump', results['attention_analysis']['pre_jump']['patterns']),
            ('jump', results['attention_analysis']['jump']['patterns']),
            ('post_jump', results['attention_analysis']['post_jump']['patterns'])
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

            plt.suptitle(f'Attention Pattern Comparison for {head}', y=1.05)
            plt.tight_layout()

            # Save the figure
            head_safe = head.replace(':', '_').replace('.', '_')
            plt.savefig(self.analysis_dir / f"{jump_id}_{head_safe}_pattern_comparison.png")
            plt.close(fig)

    def analyze(self, eval_loader, sample_input=None):
        """
        General attention analysis method

        Args:
            eval_loader: Evaluation data loader
            sample_input: Optional sample input for pattern extraction

        Returns:
            dict: Attention analysis results
        """
        # Analyze attention entropy
        entropy = self.model.compute_attention_entropy(eval_loader)

        # Extract patterns if sample input provided
        patterns = {}
        if sample_input is not None:
            device = next(self.model.parameters()).device
            sample_input = sample_input.to(device)

            # Forward pass with attention storage
            _ = self.model(sample_input, store_attention=True)

            # Get attention patterns
            patterns = self.model.get_attention_patterns()
            patterns = {k: v.cpu().numpy() for k, v in patterns.items()} if patterns else {}

        # Calculate average entropy per layer
        layer_entropies = {}
        for head_name, entropy_val in entropy.items():
            # Extract layer from head name (assuming format 'layer_X_head_Y')
            parts = head_name.split('_')
            if len(parts) >= 4 and parts[0] == 'layer':
                layer_idx = parts[1]
                layer_key = f'layer_{layer_idx}'

                if layer_key not in layer_entropies:
                    layer_entropies[layer_key] = []

                layer_entropies[layer_key].append(entropy_val)

        # Average entropy per layer
        avg_layer_entropy = {layer: sum(values) / len(values)
                             for layer, values in layer_entropies.items()}

        # Log metrics
        if self.logger:
            for head, entropy_val in entropy.items():
                self.logger.log_data('attention_entropy', head, entropy_val)

            for layer, avg_entropy in avg_layer_entropy.items():
                self.logger.log_data('attention_entropy', f'avg_{layer}', avg_entropy)

        return {
            'entropy': entropy,
            'patterns': patterns,
            'layer_avg_entropy': avg_layer_entropy
        }
    