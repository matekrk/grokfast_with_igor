# analyzers/attribution_analyzer.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.analyzers.base_analyzer import BaseAnalyzer


class AttributionAnalyzer(BaseAnalyzer):
    """Analyzer for attention head attribution around jumps"""

    def _get_analysis_dir_name(self):
        return "attribution_analysis"

    def analyze_head_attribution_around_jump(self, jump_epoch, eval_loader, weight_tracker, ensure_after_snapshot=True):
        """
        Analyze head attribution before, during, and after a jump

        Args:
            jump_epoch: The epoch of the jump
            eval_loader: Evaluation data loader
            weight_tracker: EnhancedWeightSpaceTracker instance
            ensure_after_snapshot: Whether to create a post-jump snapshot if needed

        Returns:
            dict: Attribution analysis results
        """
        jump_id = f"jump_{jump_epoch}"

        # Find snapshots around the jump
        snapshots = self._find_jump_snapshots(jump_epoch, weight_tracker, ensure_after_snapshot)
        if snapshots is None:
            return None

        pre_jump_snapshot, jump_snapshot, post_jump_snapshot = snapshots

        # Analyze attribution for each snapshot
        results = {
            'jump_epoch': jump_epoch,
            'pre_jump_epoch': pre_jump_snapshot['epoch'],
            'jump_epoch': jump_snapshot['epoch'],
            'post_jump_epoch': post_jump_snapshot['epoch'],
            'attribution_analysis': {}
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

                # Analyze head attribution
                attribution = self.model.analyze_head_attribution(eval_loader)

                results['attribution_analysis'][state_name] = attribution

            # Create visualizations
            self._visualize_attribution_comparison(results, jump_id)

        finally:
            # Restore original model state
            self.model.load_state_dict(original_state)

        # Log attribution changes
        if self.logger:
            for state_name in ['pre_jump', 'jump', 'post_jump']:
                for head, score in results['attribution_analysis'][state_name].items():
                    self.logger.log_data('head_attribution',
                                         f'jump_{jump_epoch}_{state_name}_{head}',
                                         float(score))

        return results

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

    def _visualize_attribution_comparison(self, results, jump_id):
        """Create visualizations comparing attribution before/during/after a jump"""
        # Extract attribution data
        before_attr = results['attribution_analysis']['pre_jump']
        jump_attr = results['attribution_analysis']['jump']
        after_attr = results['attribution_analysis']['post_jump']

        # Convert to DataFrame
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

        # 1. Attribution values visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Melt dataframe for seaborn
        attr_melted = pd.melt(
            attr_df,
            id_vars=['head'],
            value_vars=['before', 'jump', 'after'],
            var_name='state',
            value_name='attribution'
        )

        # Remap state labels
        state_labels = {
            'before': f'Before Jump (Epoch {results["pre_jump_epoch"]})',
            'jump': f'At Jump (Epoch {results["jump_epoch"]})',
            'after': f'After Jump (Epoch {results["post_jump_epoch"]})'
        }
        attr_melted['state_label'] = attr_melted['state'].map(state_labels)

        # Create barplot
        sns.barplot(
            data=attr_melted,
            x='head',
            y='attribution',
            hue='state_label',
            ax=ax1
        )

        ax1.set_title('Head Attribution Around Jump')
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

        # Create barplot
        sns.barplot(
            data=change_melted,
            x='head',
            y='change',
            hue='transition_label',
            ax=ax2
        )

        ax2.set_title('Head Attribution Changes')
        ax2.set_xlabel('Head')
        ax2.set_ylabel('Attribution Change')
        ax2.legend(title='Transition')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / f"{jump_id}_attribution_comparison.png")
        plt.close(fig)

        # 3. Heatmap visualization
        fig, ax = plt.subplots(figsize=(10, 8))

        # Reshape data for heatmap
        heatmap_data = attr_df[['head', 'before', 'jump', 'after']].set_index('head')

        # Rename columns for clarity
        heatmap_data.columns = [
            f'Before (Epoch {results["pre_jump_epoch"]})',
            f'Jump (Epoch {results["jump_epoch"]})',
            f'After (Epoch {results["post_jump_epoch"]})'
        ]

        # Create heatmap
        sns.heatmap(
            heatmap_data,
            cmap='viridis',
            annot=True,
            fmt=".3f",
            ax=ax
        )

        ax.set_title(f'Head Attribution Comparison Around Jump at Epoch {results["jump_epoch"]}')

        plt.tight_layout()
        plt.savefig(self.analysis_dir / f"{jump_id}_attribution_heatmap.png")
        plt.close(fig)

    def analyze(self, eval_loader, snapshots=None):
        """
        General attribution analysis method

        Args:
            eval_loader: Evaluation data loader
            snapshots: Optional list of model snapshots to analyze

        Returns:
            dict: Attribution analysis results
        """
        results = {}

        # If snapshots provided, analyze each
        if snapshots:
            original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            try:
                for snapshot in snapshots:
                    epoch = snapshot['epoch']
                    self.model.load_state_dict(snapshot['state_dict'])

                    # Analyze attribution
                    attribution = self.model.analyze_head_attribution(eval_loader)
                    results[f'epoch_{epoch}'] = attribution
            finally:
                # Restore original state
                self.model.load_state_dict(original_state)
        else:
            # Just analyze current model state
            attribution = self.model.analyze_head_attribution(eval_loader)
            results['current'] = attribution

        return results
