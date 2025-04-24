# analyzers/jump_analysis_manager.py
import torch
from pathlib import Path

from analysis.analyzers.loss_landscape_analyzer import LossLandscapeAnalyzer
from analysis.analyzers.attribution_analyzer import AttributionAnalyzer
from analysis.analyzers.attention_pattern_analyzer import AttentionAnalyzer


class JumpAnalysisManager:
    """
    Manager class that coordinates all jump analyzers

    This class provides a unified interface to run all analyses on jump events,
    delegating to specialized analyzers for each aspect (loss landscape,
    head attribution, attention patterns).
    """

    def __init__(self, model, save_dir, logger=None):
        """
        Initialize the jump analysis manager

        Args:
            model: The transformer model to analyze
            save_dir: Directory for saving analysis results
            logger: Optional logger instance
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger if logger else (model.logger if hasattr(model, 'logger') else None)

        # Initialize component analyzers
        self.loss_analyzer = LossLandscapeAnalyzer(model, save_dir, logger)
        self.attribution_analyzer = AttributionAnalyzer(model, save_dir, logger)
        self.attention_analyzer = AttentionAnalyzer(model, save_dir, logger)

        # Track analysis history
        self.analyzed_jumps = []

    def analyze_jump_with_snapshots(self, jump_epoch, pre_jump_snapshot, jump_snapshot,
                                    post_jump_snapshot, inputs, targets, criterion, eval_loader=None):
        """
        Comprehensive analysis of a jump using model snapshots

        Args:
            jump_epoch: The epoch at which the jump occurred
            pre_jump_snapshot: Model snapshot from before the jump
            jump_snapshot: Model snapshot at the jump
            post_jump_snapshot: Model snapshot from after the jump
            inputs, targets: Data batch for loss landscape analysis
            criterion: Loss function
            eval_loader: Evaluation data loader for attribution and attention analysis

        Returns:
            dict: Comprehensive jump analysis results
        """
        jump_id = f"jump_{jump_epoch}"
        print(f"JumpAnalysisManager:analyze_jump_with_snapshots(): starting comprehensive analysis of jump at epoch {jump_epoch}")

        # Prepare the complete results structure
        results = {
            'jump_epoch': jump_epoch,
            'pre_jump_epoch': pre_jump_snapshot['epoch'],
            'jump_epoch': jump_snapshot['epoch'],
            'post_jump_epoch': post_jump_snapshot['epoch'],
        }

        # 1. Analyze loss landscape
        print(f"  Analyzing loss landscape...")
        landscape_results = self.loss_analyzer.analyze_jump(
            jump_epoch, pre_jump_snapshot, jump_snapshot, post_jump_snapshot,
            inputs, targets, criterion
        )
        results['loss_landscape'] = landscape_results

        # Only proceed with attribution and attention analysis if eval_loader is provided
        if eval_loader is not None:
            # 2. Analyze head attribution
            print(f"  Analyzing head attribution...")
            # Create a snapshot-ready format for the attribution analyzer
            attribution_results = self._analyze_with_snapshots(
                self.attribution_analyzer,
                eval_loader,
                pre_jump_snapshot, jump_snapshot, post_jump_snapshot
            )
            results['head_attribution'] = attribution_results

            # 3. Analyze attention patterns
            print(f"  Analyzing attention patterns...")
            attention_results = self._analyze_with_snapshots(
                self.attention_analyzer,
                eval_loader,
                pre_jump_snapshot, jump_snapshot, post_jump_snapshot,
                sample_input=next(iter(eval_loader))[0]
            )
            results['attention_patterns'] = attention_results

        # Store the analysis in history
        self.analyzed_jumps.append({
            'epoch': jump_epoch,
            'results': results
        })

        print(f"Completed analysis of jump at epoch {jump_epoch}")
        return results

    def _analyze_with_snapshots(self, analyzer, eval_loader,
                                pre_snapshot, jump_snapshot, post_snapshot, **kwargs):
        """
        Helper method to run analysis with multiple snapshots

        Args:
            analyzer: The analyzer to use
            eval_loader: Evaluation data loader
            pre_snapshot, jump_snapshot, post_snapshot: Model snapshots
            **kwargs: Additional arguments for the analyzer

        Returns:
            dict: Analysis results for each snapshot
        """
        # Store original model state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        results = {}

        try:
            # Analyze each state
            for state_name, snapshot in [
                ('pre_jump', pre_snapshot),
                ('jump', jump_snapshot),
                ('post_jump', post_snapshot)
            ]:
                # Load the state
                self.model.load_state_dict(snapshot['state_dict'])

                # Run analysis
                analysis_result = analyzer.analyze(eval_loader, **kwargs)
                results[state_name] = analysis_result

        finally:
            # Restore original model state
            self.model.load_state_dict(original_state)

        return results

    def analyze_head_attribution_around_jump(self, jump_epoch, eval_loader, weight_tracker):
        """
        Analyze head attribution around a jump (convenience method)

        Args:
            jump_epoch: The epoch of the jump
            eval_loader: Evaluation data loader
            weight_tracker: Weight tracker with snapshots

        Returns:
            dict: Head attribution analysis results
        """
        return self.attribution_analyzer.analyze_head_attribution_around_jump(
            jump_epoch, eval_loader, weight_tracker
        )

    def analyze_attention_patterns_around_jump(self, jump_epoch, eval_loader, weight_tracker):
        """
        Analyze attention patterns around a jump (convenience method)

        Args:
            jump_epoch: The epoch of the jump
            eval_loader: Evaluation data loader
            weight_tracker: Weight tracker with snapshots

        Returns:
            dict: Attention pattern analysis results
        """
        return self.attention_analyzer.analyze_attention_patterns_around_jump(
            jump_epoch, eval_loader, weight_tracker
        )

    def analyze_loss_curvature(self, inputs, targets, criterion):
        """
        Analyze loss landscape curvature (convenience method)

        Args:
            inputs, targets: Data batch
            criterion: Loss function

        Returns:
            dict: Loss landscape curvature analysis
        """
        return self.loss_analyzer.analyze_loss_curvature(
            inputs, targets, criterion
        )

    def get_summary(self):
        """
        Get a summary of all analyzed jumps

        Returns:
            list: Summary of all analyzed jumps
        """
        summary = []

        for jump in self.analyzed_jumps:
            epoch = jump['epoch']
            results = jump['results']

            # Extract key metrics
            jump_summary = {
                'epoch': epoch,
                'pre_epoch': results.get('pre_jump_epoch'),
                'post_epoch': results.get('post_jump_epoch'),
            }

            # Add loss landscape metrics if available
            if 'loss_landscape' in results:
                landscape = results['loss_landscape']
                if 'landscape_analysis' in landscape and 'jump' in landscape['landscape_analysis']:
                    jump_data = landscape['landscape_analysis']['jump']
                    jump_summary['max_curvature'] = jump_data.get('max_curvature')
                    jump_summary['condition_number'] = jump_data.get('condition_number')

            # Add head attribution change metrics if available
            if 'head_attribution' in results:
                attribution = results['head_attribution']
                if all(k in attribution for k in ['pre_jump', 'jump', 'post_jump']):
                    # Find the head with the biggest change
                    max_change_head = None
                    max_change = 0

                    for head in attribution['pre_jump'].keys():
                        if head in attribution['jump'] and head in attribution['post_jump']:
                            pre_to_jump = abs(attribution['jump'][head] - attribution['pre_jump'][head])
                            if pre_to_jump > max_change:
                                max_change = pre_to_jump
                                max_change_head = head

                    if max_change_head:
                        jump_summary['max_attribution_change_head'] = max_change_head
                        jump_summary['max_attribution_change'] = max_change

            # Add attention metrics if available
            if 'attention_patterns' in results:
                attention = results['attention_patterns']
                if all(k in attention for k in ['pre_jump', 'jump', 'post_jump']):
                    # Find the head with the biggest entropy change
                    max_entropy_change_head = None
                    max_entropy_change = 0

                    for head in attention['pre_jump']['entropy'].keys():
                        if head in attention['jump']['entropy'] and head in attention['post_jump']['entropy']:
                            pre_to_jump = abs(attention['jump']['entropy'][head] -
                                              attention['pre_jump']['entropy'][head])
                            if pre_to_jump > max_entropy_change:
                                max_entropy_change = pre_to_jump
                                max_entropy_change_head = head

                    if max_entropy_change_head:
                        jump_summary['max_entropy_change_head'] = max_entropy_change_head
                        jump_summary['max_entropy_change'] = max_entropy_change

            summary.append(jump_summary)

        return summary
