from analysis.analyzers.base_analyzer import BaseAnalyzer
# from analysis.visualization.model_visualizer import visualize_model_analysis
# from pathlib import Path


class PhaseTransitionAnalyzer(BaseAnalyzer):
    """
    Analyzer that combines multiple analysis methods to detect and characterize
    phase transitions in transformer learning.

    This analyzer integrates circuit analysis, weight space tracking, attention patterns,
    and eigenvalue analysis to provide a comprehensive view of learning dynamics.
    """

    def __init__(self, model, save_dir, logger=None,
                 circuit_tracker=None, weight_tracker=None):
        """
        Initialize the phase transition analyzer.

        Args:
            model: The transformer model to analyze
            save_dir: Directory to save analysis results
            logger: Optional logger instance
            circuit_tracker: Optional ContinuousCircuitTracker instance
            weight_tracker: Optional EnhancedWeightSpaceTracker instance
        """
        super().__init__(model, save_dir, logger)

        # Create directory for phase transition analysis
        self.phase_dir = self.analysis_dir / "phase_transitions"
        self.phase_dir.mkdir(exist_ok=True, parents=True)

        # Initialize trackers if not provided
        self.circuit_tracker = circuit_tracker
        self.weight_tracker = weight_tracker

        # Initialize phase transition detection
        self.detected_transitions = []
        self.phase_structure = {}

        # For tracking behavioral metrics alongside structural changes
        self.performance_history = []

        # Integration with other analyzers
        self._initialize_related_analyzers()

    def _get_analysis_dir_name(self):
        return "phase_transitions"

    def _initialize_related_analyzers(self):
        """Initialize connections to other analyzers"""
        # Import here to avoid circular dependencies
        from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
        from analysis.analyzers.circuit_analyzer import CircuitAnalyzer
        from analysis.analyzers.attention_pattern_analyzer import AttentionAnalyzer
        from analysis.analyzers.attribution_analyzer import AttributionAnalyzer

        # Create circuit tracker if not provided
        if self.circuit_tracker is None:
            # Import ContinuousCircuitTracker
            from analysis.analyzers.continuous_circuit_tracker import ContinuousCircuitTracker
            self.circuit_tracker = ContinuousCircuitTracker(
                model=self.model,
                save_dir=self.save_dir,
                logger=self.logger
            )

        # Create weight tracker if not provided
        if self.weight_tracker is None:
            self.weight_tracker = EnhancedWeightSpaceTracker(
                model=self.model,
                save_dir=self.save_dir,
                logger=self.logger
            )

        # Initialize other analyzers we'll use
        self.circuit_analyzer = CircuitAnalyzer(
            model=self.model,
            save_dir=self.save_dir,
            logger=self.logger
        )

        self.attention_analyzer = AttentionAnalyzer(
            model=self.model,
            save_dir=self.save_dir,
            logger=self.logger
        )

        self.attribution_analyzer = AttributionAnalyzer(
            model=self.model,
            save_dir=self.save_dir,
            logger=self.logger
        )

    def analyze(self, epoch, eval_loader, baseline_acc=None):
        """
        Analyze the current model state for phase transitions.

        Args:
            epoch: Current training epoch
            eval_loader: Evaluation data loader
            baseline_acc: Optional baseline accuracy

        Returns:
            dict: Analysis results
        """
        # Track weight space state with the weight tracker
        took_snapshot = self.weight_tracker.take_snapshot(epoch=epoch)

        # Check if we need to sample circuits
        # We'll conditionally sample based on the circuit tracker's sampling frequency
        # to avoid duplicate sampling
        should_sample_circuits = (epoch % self.circuit_tracker.sampling_freq == 0)

        # Only sample circuits if we need to and haven't already
        circuit_state = None
        if should_sample_circuits:
            # Check if this epoch has already been sampled - important for epoch 0
            already_sampled = False
            if self.circuit_tracker.circuit_history['epochs']:
                already_sampled = epoch in self.circuit_tracker.circuit_history['epochs']

            # Only sample if we haven't already
            if not already_sampled:
                circuit_state = self.circuit_tracker.sample_circuits(
                    epoch=epoch,
                    eval_loader=eval_loader,
                    baseline_acc=baseline_acc
                )
            else:
                # If already sampled, get the existing state
                epoch_idx = self.circuit_tracker.circuit_history['epochs'].index(epoch)
                circuit_state = {
                    'epoch': epoch,
                    'active_heads': self.circuit_tracker.circuit_history['active_heads'][epoch_idx],
                    'head_attributions': self.circuit_tracker.circuit_history['head_attributions'][epoch_idx]
                    if epoch_idx < len(self.circuit_tracker.circuit_history['head_attributions']) else {},
                    'active_circuits': self.circuit_tracker.circuit_history['active_circuits'][epoch_idx]
                    if epoch_idx < len(self.circuit_tracker.circuit_history['active_circuits']) else [],
                    'circuit_strengths': self.circuit_tracker.circuit_history['circuit_strengths'][epoch_idx]
                    if epoch_idx < len(self.circuit_tracker.circuit_history['circuit_strengths']) else {}
                }

        # Track performance for correlating with transitions
        if baseline_acc is not None:
            # Check for duplicates in performance history
            already_recorded = any(p.get('epoch') == epoch for p in self.performance_history)

            if not already_recorded:
                self.performance_history.append({
                    'epoch': epoch,
                    'accuracy': baseline_acc
                })

        # Check for phase transitions in circuit structure
        new_transitions = []

        # Only perform transition detection periodically to avoid overhead
        # Make sure we have enough history before trying to detect transitions
        if epoch % 50 == 0 and epoch > 0 and circuit_state is not None and len(
                self.circuit_tracker.circuit_history['epochs']) >= 3:
            # Analyze phase structure
            phase_analysis = self.circuit_tracker.analyze_phase_structure()
            if 'transitions' in phase_analysis and phase_analysis['transitions']:
                new_transitions = phase_analysis['transitions']
                self.phase_structure = phase_analysis.get('phases', {})  # fixme self.phase_structure['phases'] <- ?

                # Update global transitions list
                for transition in new_transitions:
                    if transition not in self.detected_transitions:
                        self.detected_transitions.append(transition)

                # Log transition detection
                print(f"Phase transition analysis at epoch {epoch} detected {len(new_transitions)} transitions")

                # Log transitions to logger if available
                if self.logger and new_transitions:
                    for i, transition in enumerate(new_transitions):
                        self.logger.log_data('phase_transitions',
                                             f'transition_{epoch}_{i}_epoch',
                                             transition['epoch'])
                        self.logger.log_data('phase_transitions',
                                             f'transition_{epoch}_{i}_types',
                                             '+'.join(transition['transition_types']))

        # Perform detailed analysis if we just detected a transition
        transition_details = {}
        for transition in new_transitions:
            transition_epoch = transition['epoch']

            # Skip if this transition is too far in the past
            if abs(transition_epoch - epoch) > 100:
                continue

            # Perform detailed analysis
            transition_details[transition_epoch] = self._analyze_transition_details(
                transition=transition,
                eval_loader=eval_loader
            )

            # Visualize this transition
            self._visualize_transition(transition, transition_details[transition_epoch])

        # Return the analysis results
        return {
            'epoch': epoch,
            'weight_snapshot_taken': took_snapshot,
            'circuit_state': circuit_state,
            'new_transitions': new_transitions,
            'transition_details': transition_details
        }

    def _analyze_transition_details(self, transition, eval_loader):
        """
        Perform detailed analysis of a specific transition.

        Args:
            transition: The transition to analyze
            eval_loader: Evaluation data loader

        Returns:
            dict: Detailed analysis of the transition
        """
        transition_epoch = transition['epoch']
        print(f"Performing detailed analysis of transition at epoch {transition_epoch}")

        details = {
            'epoch': transition_epoch,
            'transition_types': transition['transition_types'],
            'window': transition['window'],
            'attribution_analysis': {},
            'attention_analysis': {},
            'circuit_analysis': {},
            'performance_correlation': {}
        }

        # Get detailed attribution analysis
        attribution_results = self.attribution_analyzer.analyze(eval_loader)
        if attribution_results:
            details['attribution_analysis'] = attribution_results

        # Get detailed attention analysis
        attention_results = self.attention_analyzer.analyze(eval_loader,
                                                            sample_input=next(iter(eval_loader))[0])
        if attention_results:
            details['attention_analysis'] = attention_results

        # Get detailed circuit analysis
        circuit_results = self.circuit_analyzer.identify_circuits(eval_loader)
        if circuit_results:
            details['circuit_analysis'] = circuit_results

        # Analyze performance correlation with this transition
        if self.performance_history:
            # Find performance before and after the transition
            pre_perf = [p['accuracy'] for p in self.performance_history
                        if p['epoch'] < transition_epoch]
            post_perf = [p['accuracy'] for p in self.performance_history
                         if p['epoch'] > transition_epoch]

            if pre_perf and post_perf:
                avg_pre = sum(pre_perf[-min(5, len(pre_perf)):]) / min(5, len(pre_perf))
                avg_post = sum(post_perf[:min(5, len(post_perf))]) / min(5, len(post_perf))

                # Calculate performance change
                perf_change = avg_post - avg_pre
                perf_change_pct = perf_change / (avg_pre + 1e-8) * 100

                details['performance_correlation'] = {
                    'pre_transition_avg': avg_pre,
                    'post_transition_avg': avg_post,
                    'absolute_change': perf_change,
                    'percent_change': perf_change_pct,
                    'correlation_type': 'positive' if perf_change > 0.01 else
                    'negative' if perf_change < -0.01 else
                    'neutral'
                }

        return details

    def _visualize_transition(self, transition, details):
        """
        Create visualizations for a detected phase transition.

        Args:
            transition: The transition data
            details: Detailed analysis of the transition
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        # import numpy as np
        import pandas as pd

        transition_epoch = transition['epoch']

        # 1. Create timeline visualization showing this transition
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        # Plot performance history
        if self.performance_history:
            epochs = [p['epoch'] for p in self.performance_history]
            accuracy = [p['accuracy'] for p in self.performance_history]

            ax1.plot(epochs, accuracy, 'b-', marker='.', label='Accuracy')   # fixme marker = '1', '2'?

            # Mark the transition
            ax1.axvline(x=transition_epoch, color='r', linestyle='--',
                        label=f'Transition at Epoch {transition_epoch}')

            # Highlight transition window
            window_start, window_end = transition['window']
            ax1.axvspan(window_start, window_end, alpha=0.2, color='yellow',
                        label='Transition Window')

            # Annotate transition types
            transition_label = ', '.join(transition['transition_types'])
            ax1.annotate(transition_label,
                         xy=(transition_epoch, max(accuracy) * 0.9),
                         xytext=(transition_epoch + 20, max(accuracy) * 0.9),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                         fontsize=10)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Phase Transition Timeline - Epoch {transition_epoch}')
        ax1.legend()

        plt.tight_layout()
        plt.savefig(self.phase_dir / f"transition_timeline_epoch_{transition_epoch}.png")
        plt.close(fig1)

        # 2. Create head attribution comparison
        if 'attribution_analysis' in details and details['attribution_analysis']:
            attribution = details['attribution_analysis'].get('current', {})

            if attribution:
                fig2, ax2 = plt.subplots(figsize=(12, 6))

                # Convert to DataFrame for plotting
                attr_data = []
                for head, score in attribution.items():
                    attr_data.append({
                        'Head': head,
                        'Attribution': score
                    })

                attr_df = pd.DataFrame(attr_data)
                attr_df = attr_df.sort_values('Attribution', ascending=False)

                # Plot top 10 heads
                top_heads = attr_df.head(10)
                sns.barplot(x='Head', y='Attribution', data=top_heads, ax=ax2)
                ax2.set_title(f'Head Attribution at Transition Epoch {transition_epoch}')
                ax2.set_xlabel('Head')
                ax2.set_ylabel('Attribution Score')
                ax2.tick_params(axis='x', rotation=45)

                plt.tight_layout()
                plt.savefig(self.phase_dir / f"transition_attribution_epoch_{transition_epoch}.png")
                plt.close(fig2)

        # 3. Create circuit network visualization
        if 'circuit_analysis' in details and details['circuit_analysis']:
            circuits = details['circuit_analysis']

            if 'circuits' in circuits and circuits['circuits']:
                # Import networkx if available
                import networkx as nx
                try:

                    fig3, ax3 = plt.subplots(figsize=(12, 10))

                    # Create a graph representation
                    G = nx.Graph()

                    # Add nodes for significant heads
                    for head, score in circuits['significant_heads'].items():
                        G.add_node(head, size=score, type='head')

                    # Add edges for active circuits
                    for circuit in circuits['circuits']:
                        if '+' in circuit:
                            head1, head2 = circuit.split('+')
                        else:
                            head1, head2 = circuit  # Handle tuple format

                        # Get circuit data
                        if circuit in circuits['circuits']:
                            strength = circuits['circuits'][circuit]['circuit_strength']
                            G.add_edge(head1, head2, weight=strength)

                    # Set positions using spring layout
                    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

                    # Get node attributes for visualization
                    node_sizes = [G.nodes[n].get('size', 0.0) * 3000 + 100 for n in G.nodes()]

                    # Color nodes by layer
                    node_colors = []
                    for n in G.nodes():
                        layer = int(n.split('_')[1])
                        # Create a colormap based on layer
                        node_colors.append(plt.cm.viridis(layer / max(1, self.model.num_layers - 1)))

                    # Draw nodes
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

                    # Draw edges with width proportional to circuit strength
                    edge_widths = [G[u][v]['weight'] * 5 + 1 for u, v in G.edges()]
                    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')

                    # Draw labels
                    nx.draw_networkx_labels(G, pos, font_size=8)

                    # Add legend
                    plt.title(f'Circuit Network at Transition Epoch {transition_epoch}')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(self.phase_dir / f"transition_circuit_network_epoch_{transition_epoch}.png")
                    plt.close(fig3)
                except ImportError:
                    print("NetworkX not available for circuit visualization")

        # 4. Create attention pattern visualization if available
        if 'attention_analysis' in details and 'patterns' in details['attention_analysis']:
            patterns = details['attention_analysis']['patterns']

            if patterns:
                num_patterns = len(patterns)
                max_heads_to_show = min(6, num_patterns)  # Limit to 6 heads for clarity

                # Select heads to visualize - prioritize heads from layer 0
                selected_patterns = {}
                layer0_patterns = {k: v for k, v in patterns.items() if 'layer_0' in k}

                if layer0_patterns:
                    selected_patterns.update(list(layer0_patterns.items())[:max_heads_to_show])

                # Fill remaining slots with other patterns if needed
                remaining_slots = max_heads_to_show - len(selected_patterns)
                if remaining_slots > 0:
                    other_patterns = {k: v for k, v in patterns.items() if k not in selected_patterns}
                    selected_patterns.update(list(other_patterns.items())[:remaining_slots])

                # Create figure with 2x3 grid of pattern heatmaps
                fig4, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()

                for i, (pattern_name, pattern) in enumerate(selected_patterns.items()):
                    if i < len(axes):
                        sns.heatmap(pattern, cmap='viridis', ax=axes[i])
                        axes[i].set_title(pattern_name)

                # Hide unused axes
                for i in range(len(selected_patterns), len(axes)):
                    axes[i].axis('off')

                plt.suptitle(f'Attention Patterns at Transition Epoch {transition_epoch}')
                plt.tight_layout()
                plt.savefig(self.phase_dir / f"transition_attention_patterns_epoch_{transition_epoch}.png")
                plt.close(fig4)

    def analyze_with_weight_space_jumps(self, jump_results, eval_loader):
        """
        Compare phase transitions with detected weight space jumps.

        Args:
            jump_results: Results from weight_tracker.analyze_pending_jumps()
            eval_loader: Evaluation data loader

        Returns:
            dict: Comparison results
        """
        if not self.detected_transitions or not jump_results:
            return {"status": "No transitions or jumps detected yet"}

        # Extract jump epochs
        jump_epochs = [result['jump_epoch'] for result in jump_results]

        # Extract transition epochs
        transition_epochs = [t['epoch'] for t in self.detected_transitions]

        comparison = {
            'jumps': jump_epochs,
            'transitions': transition_epochs,
            'correlation': [],
            'aligned_events': [],
            'jump_without_transition': [],
            'transition_without_jump': []
        }

        # Find correlations between jumps and transitions
        for jump_epoch in jump_epochs:
            # Find the closest transition
            closest_transition = min(transition_epochs,
                                     key=lambda x: abs(x - jump_epoch)) if transition_epochs else None

            if closest_transition is not None:
                # Calculate distance
                distance = abs(jump_epoch - closest_transition)

                # Is this a close match? (within 5 epochs)
                is_aligned = distance <= 5

                correlation = {
                    'jump_epoch': jump_epoch,
                    'transition_epoch': closest_transition,
                    'distance': distance,
                    'aligned': is_aligned
                }

                comparison['correlation'].append(correlation)

                if is_aligned:
                    comparison['aligned_events'].append((jump_epoch, closest_transition))
                else:
                    comparison['jump_without_transition'].append(jump_epoch)

        # Find transitions without corresponding jumps
        for transition_epoch in transition_epochs:
            closest_jump = min(jump_epochs, key=lambda x: abs(x - transition_epoch)) if jump_epochs else None

            if closest_jump is not None:
                distance = abs(transition_epoch - closest_jump)

                # If not closely aligned, this is a transition without a jump
                if distance > 5:
                    comparison['transition_without_jump'].append(transition_epoch)
            else:
                comparison['transition_without_jump'].append(transition_epoch)

        # Analyze correlation numerically
        if comparison['correlation']:
            avg_distance = sum(c['distance'] for c in comparison['correlation']) / len(comparison['correlation'])
            alignment_rate = len(comparison['aligned_events']) / len(comparison['correlation'])

            comparison['summary'] = {
                'average_distance': avg_distance,
                'alignment_rate': alignment_rate,
                'conclusion': "Strong correlation" if alignment_rate > 0.7 else
                "Moderate correlation" if alignment_rate > 0.4 else
                "Weak correlation"
            }

        # Perform detailed analysis on aligned events
        if comparison['aligned_events']:
            aligned_analysis = []

            for jump_epoch, transition_epoch in comparison['aligned_events']:
                # Find the corresponding jump and transition data
                jump_data = next((j for j in jump_results if j['jump_epoch'] == jump_epoch), None)
                transition_data = next((t for t in self.detected_transitions if t['epoch'] == transition_epoch), None)

                if jump_data and transition_data:
                    # Compare characteristics
                    analysis = {
                        'jump_epoch': jump_epoch,
                        'transition_epoch': transition_epoch,
                        'jump_data': {
                            'magnitude': jump_data.get('characterization', {}).get('total_magnitude', {}).get(
                                'pre_to_jump', 0),
                            'top_layers': jump_data.get('characterization', {}).get('top_layers', []),
                            'top_heads': jump_data.get('characterization', {}).get('top_heads', [])
                        },
                        'transition_data': {
                            'types': transition_data.get('transition_types', []),
                            'head_change_ratio': transition_data.get('head_change_ratio', 0),
                            'circuit_change_ratio': transition_data.get('circuit_change_ratio', 0)
                        }
                    }

                    # Analyze if the jump and transition affect the same components
                    # This would involve checking if the top changing heads in the jump
                    # match those that form or dissolve circuits in the transition

                    # (This would require additional analysis and data that we may not have here)

                    aligned_analysis.append(analysis)

            comparison['aligned_analysis'] = aligned_analysis

            # Create visualization comparing jumps and transitions
            self._visualize_jump_transition_comparison(comparison)

        return comparison

    def _visualize_jump_transition_comparison(self, comparison):
        """
        Create visualizations comparing weight space jumps and phase transitions.

        Args:
            comparison: The comparison results
        """
        import matplotlib.pyplot as plt

        # 1. Create timeline showing jumps and transitions
        fig1, ax1 = plt.subplots(figsize=(14, 6))

        # Plot performance history if available
        if self.performance_history:
            epochs = [p['epoch'] for p in self.performance_history]
            accuracy = [p['accuracy'] for p in self.performance_history]

            ax1.plot(epochs, accuracy, 'b-', alpha=0.5, label='Accuracy')

        # Mark jumps
        for jump_epoch in comparison['jumps']:
            ax1.axvline(x=jump_epoch, color='r', linestyle='--', alpha=0.7)

        # Mark transitions
        for transition_epoch in comparison['transitions']:
            ax1.axvline(x=transition_epoch, color='g', linestyle='-', alpha=0.7)

        # Create custom legend entries
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='r', linestyle='--', label='Weight Space Jump'),
            Line2D([0], [0], color='g', linestyle='-', label='Phase Transition')
        ]

        if self.performance_history:
            legend_elements.insert(0, Line2D([0], [0], color='b', label='Accuracy'))

        ax1.legend(handles=legend_elements)

        # Mark aligned events
        for jump_epoch, transition_epoch in comparison.get('aligned_events', []):
            mid_point = (jump_epoch + transition_epoch) / 2
            text_y = 0.9 if self.performance_history else 0.7
            ax1.annotate('Aligned',
                         xy=(mid_point, text_y),
                         xycoords=('data', 'axes fraction'),
                         horizontalalignment='center',
                         arrowprops=dict(arrowstyle='->'))

        ax1.set_xlabel('Epoch')
        if self.performance_history:
            ax1.set_ylabel('Accuracy')
        ax1.set_title('Comparison of Weight Space Jumps and Phase Transitions')

        plt.tight_layout()
        plt.savefig(self.phase_dir / "jump_transition_comparison.png")
        plt.close(fig1)

        # 2. Create scatter plot of jump epochs vs transition epochs
        if comparison['correlation']:
            fig2, ax2 = plt.subplots(figsize=(10, 8))

            jump_epochs = [c['jump_epoch'] for c in comparison['correlation']]
            transition_epochs = [c['transition_epoch'] for c in comparison['correlation']]
            distances = [c['distance'] for c in comparison['correlation']]

            # Determine point sizes based on distance (closer = larger)
            point_sizes = [max(20, 200 / (d + 1)) for d in distances]

            # Determine colors based on alignment
            colors = ['green' if c['aligned'] else 'red' for c in comparison['correlation']]

            # Create scatter plot
            ax2.scatter(jump_epochs, transition_epochs, s=point_sizes, c=colors, alpha=0.7)

            # Add identity line
            min_epoch = min(min(jump_epochs), min(transition_epochs))
            max_epoch = max(max(jump_epochs), max(transition_epochs))
            ax2.plot([min_epoch, max_epoch], [min_epoch, max_epoch], 'k--', alpha=0.5)

            # Add annotations for aligned events
            for i, corr in enumerate(comparison['correlation']):
                if corr['aligned']:
                    ax2.annotate(f"Epoch {corr['jump_epoch']}",
                                 xy=(corr['jump_epoch'], corr['transition_epoch']),
                                 xytext=(10, 10),
                                 textcoords="offset points",
                                 fontsize=8)

            ax2.set_xlabel('Weight Space Jump Epochs')
            ax2.set_ylabel('Phase Transition Epochs')
            ax2.set_title('Correlation Between Jumps and Transitions')

            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                       markersize=10, label='Aligned (≤5 epochs)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=10, label='Not Aligned (>5 epochs)'),
                Line2D([0], [0], color='k', linestyle='--', label='Perfect Alignment')
            ]
            ax2.legend(handles=legend_elements)

            plt.tight_layout()
            plt.savefig(self.phase_dir / "jump_transition_correlation.png")
            plt.close(fig2)

        # 3. Create comparison of characteristics for aligned events
        if 'aligned_analysis' in comparison and comparison['aligned_analysis']:
            fig3, ax3 = plt.subplots(figsize=(12, len(comparison['aligned_analysis']) * 2))

            # Prepare data for table
            data = []
            for analysis in comparison['aligned_analysis']:
                jump_epoch = analysis['jump_epoch']
                transition_epoch = analysis['transition_epoch']

                jump_info = f"Epoch {jump_epoch}\n" + \
                            f"Magnitude: {analysis['jump_data']['magnitude']:.3f}\n" + \
                            f"Top Layers: {', '.join(analysis['jump_data']['top_layers'][:3])}\n" + \
                            f"Top Heads: {', '.join(analysis['jump_data']['top_heads'][:3])}"

                transition_info = f"Epoch {transition_epoch}\n" + \
                                  f"Types: {', '.join(analysis['transition_data']['types'])}\n" + \
                                  f"Head Change: {analysis['transition_data']['head_change_ratio']:.2f}\n" + \
                                  f"Circuit Change: {analysis['transition_data']['circuit_change_ratio']:.2f}"

                data.append([jump_info, transition_info])

            # Hide axes
            ax3.axis('tight')
            ax3.axis('off')

            # Create table
            table = ax3.table(cellText=data,
                              colLabels=["Weight Space Jump", "Phase Transition"],
                              loc='center',
                              cellLoc='center')

            # Adjust table appearance
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            plt.title('Characteristics of Aligned Jump-Transition Events')
            plt.tight_layout()
            plt.savefig(self.phase_dir / "aligned_event_characteristics.png")
            plt.close(fig3)

    def get_learning_phase_summary(self):
        """
        Generate a comprehensive summary of learning phases and transitions.

        Returns:
            dict: Learning phase summary
        """
        if not self.detected_transitions:
            return {"status": "No transitions detected yet"}

        # Get the phase structure from circuit tracker
        phase_structure = self.circuit_tracker.analyze_phase_structure()

        summary = {
            'transitions': self.detected_transitions,
            'phases': phase_structure.get('phases', []),
        }

        # Add grokking correlation if available
        if self.logger and 'grokking_phases' in self.logger.logs:
            grokking_data = self.logger.logs['grokking_phases']
            if 'grokking_step' in grokking_data and grokking_data['grokking_step']:
                grokking_epochs = grokking_data['grokking_step']

                # Handle both single epoch and list of epochs
                if not isinstance(grokking_epochs, list):
                    grokking_epochs = [grokking_epochs]

                # Track correlations for each grokking epoch
                grokking_correlations = []

                for grokking_epoch in grokking_epochs:
                    # Find closest transition to this grokking epoch
                    closest_transition = None
                    min_distance = float('inf')

                    for transition in self.detected_transitions:
                        transition_epoch = transition['epoch']
                        distance = abs(transition_epoch - grokking_epoch)
                        if distance < min_distance:
                            min_distance = distance
                            closest_transition = transition

                    if closest_transition:
                        # Determine which phase contains this grokking epoch
                        grokking_phase = None
                        for phase in summary.get('phases', []):
                            if phase['start_epoch'] <= grokking_epoch <= phase['end_epoch']:
                                grokking_phase = phase
                                break

                        # Store correlation for this grokking epoch
                        grokking_correlations.append({
                            'grokking_epoch': grokking_epoch,
                            'closest_transition': closest_transition,
                            'distance': min_distance,
                            'aligned': min_distance <= 20,  # Consider aligned if within 20 epochs
                            'grokking_phase': grokking_phase
                        })

                # Store all correlations
                summary['grokking_correlations'] = grokking_correlations

                # For backward compatibility, also store the first correlation
                # under the old key 'grokking_correlation'
                if grokking_correlations:
                    summary['grokking_correlation'] = grokking_correlations[0]

        # Generate insights about the learning process
        insights = []

        # How many distinct phases were detected
        if 'phases' in summary and summary['phases']:
            num_phases = len(summary['phases'])
            insights.append(f"Detected {num_phases} distinct learning phases")

            # Characterize each phase
            for i, phase in enumerate(summary['phases']):
                if 'classification' in phase:
                    insights.append(f"Phase {i + 1} (epochs {phase['start_epoch']}-{phase['end_epoch']}): " +
                                    f"{phase['classification'].title()} phase")

        # Add correlation with grokking - handle multiple grokking epochs
        if 'grokking_correlations' in summary:
            for corr in summary['grokking_correlations']:
                if corr.get('aligned'):
                    grokking_epoch = corr['grokking_epoch']
                    transition = corr['closest_transition']

                    insights.append(f"Grokking at epoch {grokking_epoch} aligns with a " +
                                    f"phase transition at epoch {transition['epoch']} " +
                                    f"characterized by {', '.join(transition['transition_types'])}")

                    if corr.get('grokking_phase') and 'classification' in corr['grokking_phase']:
                        insights.append(f"Grokking at epoch {grokking_epoch} occurs during a " +
                                        f"{corr['grokking_phase']['classification']} phase")

        summary['insights'] = insights

        # Create summary visualization
        self._visualize_learning_phases(summary)

        return summary

    def _visualize_learning_phases(self, summary):
        """
        Create a visualization summarizing the learning phases.

        Args:
            summary: The learning phase summary
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Get phases and transitions
        phases = summary.get('phases', [])
        transitions = summary.get('transitions', [])

        # Plot performance history in top subplot if available
        if self.performance_history:
            epochs = [p['epoch'] for p in self.performance_history]
            accuracy = [p['accuracy'] for p in self.performance_history]

            ax1.plot(epochs, accuracy, 'b-', label='Accuracy')

            # Mark transitions
            for transition in transitions:
                epoch = transition['epoch']
                ax1.axvline(x=epoch, color='r', linestyle='--', alpha=0.7)

        # Mark grokking points if available
        if 'grokking_correlations' in summary:
            for i, corr in enumerate(summary['grokking_correlations']):
                grokking_epoch = corr['grokking_epoch']
                ax1.axvline(x=grokking_epoch, color='g', linestyle='-', linewidth=2,
                            label='Grokking' if i == 0 else None)  # Only add to legend once

                # Add text annotation
                max_y = max(accuracy) if self.performance_history else 1.0
                y_position = max_y * (0.9 - i * 0.05)  # Stagger annotations to avoid overlap
                ax1.text(grokking_epoch + 5, y_position, f'Grokking {i + 1}',
                         fontsize=12, color='green')

        # Add legend to top subplot
        if self.performance_history or 'grokking_correlations' in summary:
            ax1.legend()

        ax1.set_title('Learning Phases and Transitions')
        ax1.set_xlabel('Epoch')
        if self.performance_history:
            ax1.set_ylabel('Accuracy')

        # Create phase visualization in bottom subplot
        if phases:
            # Define colors for different phase classifications
            phase_colors = {
                'exploration': 'lightblue',
                'consolidation': 'lightgreen',
                'stability': 'lightyellow',
                'pruning': 'salmon',
                'transition': 'lightgray'
            }

            # Plot phases as colored blocks
            for i, phase in enumerate(phases):
                start = phase['start_epoch']
                end = phase['end_epoch']

                classification = phase.get('classification', 'transition')
                color = phase_colors.get(classification, 'lightgray')

                # Draw rectangle for this phase
                rect = Rectangle((start, 0), end - start, 1,
                                 facecolor=color, alpha=0.7, edgecolor='black')
                ax2.add_patch(rect)

                # Add phase label
                ax2.text((start + end) / 2, 0.5, f"Phase {i + 1}\n{classification.title()}",
                         ha='center', va='center', fontsize=10)

            # Set limits
            min_epoch = min(p['start_epoch'] for p in phases)
            max_epoch = max(p['end_epoch'] for p in phases)
            ax2.set_xlim(min_epoch, max_epoch)
            ax2.set_ylim(0, 1)

            # Remove y-axis ticks and labels
            ax2.set_yticks([])
            ax2.set_ylabel('Phases')

        # Ensure both subplots have same x-axis limits
        if self.performance_history and phases:
            min_x = min(min(epochs), min(p['start_epoch'] for p in phases))
            max_x = max(max(epochs), max(p['end_epoch'] for p in phases))
            ax1.set_xlim(min_x, max_x)
            ax2.set_xlim(min_x, max_x)

        plt.tight_layout()
        plt.savefig(self.phase_dir / "learning_phases_summary.png")
        plt.close(fig)