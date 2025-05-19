# enhanced_phase_analyzer.py
import numpy as np
import torch

from analysis.analyzers.attention_mlp_interaction_analyzer import AttentionMLPInteractionAnalyzer
from analysis.analyzers.circuit_class_attribution import CircuitClassAttribution
from analysis.analyzers.mlp_sparsity_tracker import MLPSparsityTracker
from analysis.analyzers.phase_transition_analyzer import PhaseTransitionAnalyzer


class EnhancedPhaseAnalyzer(PhaseTransitionAnalyzer):
    """Extends phase transition analysis with circuit and sparsity tracking"""

    def __init__(self, model, save_dir, logger=None,
                 circuit_tracker=None, weight_tracker=None):
        super().__init__(model, save_dir, logger, circuit_tracker, weight_tracker)

        # Additional analysis components
        self.mlp_sparsity_tracker = MLPSparsityTracker(
            model=model,
            save_dir=save_dir / "sparsity_analysis",
            logger=logger
        )

        self.circuit_class_analyzer = CircuitClassAttribution(
            model=model,
            save_dir=save_dir / "circuit_class_analysis",
            circuit_tracker=circuit_tracker,
            logger=logger
        )

        self.interaction_analyzer = AttentionMLPInteractionAnalyzer(
            model=model,
            save_dir=save_dir / "interaction_analysis",
            logger=logger
        )

        # Storage for enhanced analysis results
        self.enhanced_analysis_history = {}

    def analyze(self, epoch, eval_loader, baseline_acc=None):
        """Enhanced analysis including sparsity and circuit-class relationships"""
        # First perform standard phase analysis
        results = super().analyze(epoch, eval_loader, baseline_acc)

        # Add enhanced analysis at key points
        if self._is_analysis_point(epoch):
            # print(f"Performing enhanced analysis at epoch {epoch}...")

            # Track MLP sparsity patterns
            sparsity_results = self.mlp_sparsity_tracker.track_sparsity_evolution(
                epoch, eval_loader)

            # Analyze circuit-class relationships
            class_circuit_results = self.circuit_class_analyzer.analyze_epoch(
                epoch, eval_loader)

            # Analyze attention-MLP interactions
            interaction_results = self.interaction_analyzer.analyze_interactions(
                epoch, eval_loader)

            # Correlate with phase transitions
            enhanced_insights = {}
            if results.get('new_transitions'):
                enhanced_insights = self._analyze_transitions_with_circuits(
                    results['new_transitions'],
                    sparsity_results,
                    class_circuit_results,
                    interaction_results
                )

            # Store enhanced analysis results
            self.enhanced_analysis_history[epoch] = {
                'sparsity_analysis': sparsity_results,
                'class_circuit_analysis': class_circuit_results,
                'interaction_analysis': interaction_results,
                'insights': enhanced_insights
            }

            # Add to results
            results['enhanced_analysis'] = {
                'sparsity_analysis': sparsity_results,
                'class_circuit_analysis': class_circuit_results,
                'interaction_analysis': interaction_results,
                'insights': enhanced_insights
            }
        else:
            print(f"\tEnhancedPhaseAnalyzer.analyze()\tno enhanced analysis at epoch {epoch}...")

        if epoch % 100 == 0:
            self.cleanup()

        return results

    def _is_analysis_point(self, epoch):
        """Determine if this epoch should have enhanced analysis"""
        # Always analyze at detected transitions
        if self.detected_transitions:
            for transition in self.detected_transitions:
                if transition['epoch'] == epoch:
                    return True

        # Always analyze at grokking points
        if self.logger and 'grokking_phases' in self.logger.logs:
            grokking_step = self.logger.logs['grokking_phases'].get('grokking_step')
            if grokking_step:
                if isinstance(grokking_step, list) and epoch in grokking_step:
                    return True
                elif epoch == grokking_step:
                    return True

        # Analyze periodically (less frequently than main analysis)
        analyze_interval = getattr(self, 'analyze_interval', 100)
        return epoch % analyze_interval == 0

    def _analyze_transitions_with_circuits(self, transitions, sparsity_results,
                                           class_circuit_results, interaction_results):
        """Correlate phase transitions with circuit and sparsity changes"""
        insights = {}

        for transition in transitions:
            transition_epoch = transition['epoch']
            transition_type = '+'.join(transition['transition_types'])

            # Check for correlation with sparsity patterns
            if sparsity_results:
                sparsity_insights = self._correlate_sparsity_with_transition(
                    transition, sparsity_results)
                insights[f'transition_{transition_epoch}_sparsity'] = sparsity_insights

            # Check for correlation with circuit-class relationships
            if class_circuit_results:
                circuit_insights = self._correlate_circuits_with_transition(
                    transition, class_circuit_results)
                insights[f'transition_{transition_epoch}_circuits'] = circuit_insights

            # Check for correlation with attention-MLP interactions
            if interaction_results:
                interaction_insights = self._correlate_interactions_with_transition(
                    transition, interaction_results)
                insights[f'transition_{transition_epoch}_interactions'] = interaction_insights

        return insights

    def _correlate_sparsity_with_transition(self, transition, sparsity_results):
        """Correlate sparsity patterns with a phase transition"""

        def safe_mean(values, default=0.0):
            """Calculate mean safely even with empty arrays"""
            if not values or len(values) == 0:
                return default
            return np.mean(values)

        # Extract relevant information
        transition_epoch = transition['epoch']

        # Get recent sparsity history
        sparsity_history = {}
        for epoch, data in self.mlp_sparsity_tracker.sparsity_history.items():
            if abs(epoch - transition_epoch) <= 100:  # Look at nearby epochs
                sparsity_history[epoch] = data

        # Skip if not enough history
        if len(sparsity_history) < 2:
            return {"message": "Not enough sparsity history to correlate with transition"}

        # Analyze sparsity changes around transition
        pre_epochs = [e for e in sparsity_history.keys() if e < transition_epoch]
        post_epochs = [e for e in sparsity_history.keys() if e > transition_epoch]

        # Skip if not enough data on both sides
        if not pre_epochs or not post_epochs:
            return {"message": "Need sparsity data from both before and after transition"}

        # Get data from before and after transition
        pre_data = sparsity_history[max(pre_epochs)]
        post_data = sparsity_history[min(post_epochs)]

        # Compare average sparsity
        sparsity_changes = {}
        for layer in pre_data['avg_sparsity'].keys():
            if layer in post_data['avg_sparsity']:
                pre_sparsity = pre_data['avg_sparsity'][layer]
                post_sparsity = post_data['avg_sparsity'][layer]

                change = post_sparsity - pre_sparsity
                percent_change = (change / pre_sparsity) * 100 if pre_sparsity > 0 else float('inf')

                sparsity_changes[layer] = {
                    'pre_sparsity': pre_sparsity,
                    'post_sparsity': post_sparsity,
                    'absolute_change': change,
                    'percent_change': percent_change
                }

        # Compare neuron selectivity
        selectivity_changes = {}
        if 'selectivity_summary' in pre_data and 'selectivity_summary' in post_data:
            for layer in pre_data['selectivity_summary'].keys():
                if layer in post_data['selectivity_summary']:
                    pre_selective = pre_data['selectivity_summary'][layer]['selective_neurons']
                    post_selective = post_data['selectivity_summary'][layer]['selective_neurons']

                    change = post_selective - pre_selective
                    percent_change = (change / pre_selective) * 100 if pre_selective > 0 else float('inf')

                    selectivity_changes[layer] = {
                        'pre_selective': pre_selective,
                        'post_selective': post_selective,
                        'absolute_change': change,
                        'percent_change': percent_change
                    }

        # Generate insights
        insights = {
            'sparsity_changes': sparsity_changes,
            'selectivity_changes': selectivity_changes
        }

        # Add high-level summary
        values = [data['percent_change'] for data in sparsity_changes.values()]
        avg_sparsity_change = safe_mean(values)
        insights['summary'] = {
            'avg_sparsity_change_percent': avg_sparsity_change,
            'interpretation': "increasing_sparsity" if avg_sparsity_change > 5 else
            "decreasing_sparsity" if avg_sparsity_change < -5 else
            "stable_sparsity"
        }

        return insights

    def _correlate_circuits_with_transition(self, transition, circuit_results):
        """Correlate circuit-class relationships with a phase transition"""
        # Generate insights based on circuit specialization
        if 'class_specific_circuits' in circuit_results:
            class_circuits = circuit_results['class_specific_circuits']

            insights = {
                'specialized_percentage': class_circuits['specialized_percentage'],
                'class_distribution': {
                    'class_count': len(class_circuits['class_to_circuits']),
                    'circuits_per_class': {
                        class_id: len(circuits)
                        for class_id, circuits in class_circuits['class_to_circuits'].items()
                    }
                }
            }

            # Add interpretation
            if class_circuits['specialized_percentage'] > 70:
                insights['interpretation'] = "highly_specialized_circuits"
            elif class_circuits['specialized_percentage'] > 30:
                insights['interpretation'] = "moderately_specialized_circuits"
            else:
                insights['interpretation'] = "general_purpose_circuits"

            return insights

        return {"message": "No circuit-class analysis available"}

    def _correlate_interactions_with_transition(self, transition, interaction_results):
        """Correlate attention-MLP interactions with a phase transition"""
        # Generate insights based on correlation strength
        if 'layer_correlations' in interaction_results:
            correlations = interaction_results['layer_correlations']

            avg_correlation = np.mean(list(correlations.values()))
            insights = {
                'avg_correlation': avg_correlation,
                'layer_correlations': correlations
            }

            # Check for cross-component circuits
            if hasattr(self.interaction_analyzer, 'identify_cross_component_circuits'):
                cross_circuits = self.interaction_analyzer.identify_cross_component_circuits()

                if cross_circuits:
                    insights['cross_component_circuits'] = {
                        'count': len(cross_circuits),
                        'avg_strength': np.mean([c['strength'] for c in cross_circuits])
                    }

            # Add interpretation
            if avg_correlation > 0.6:
                insights['interpretation'] = "strong_attn_mlp_coordination"
            elif avg_correlation > 0.3:
                insights['interpretation'] = "moderate_attn_mlp_coordination"
            else:
                insights['interpretation'] = "weak_attn_mlp_coordination"

            return insights

        return {"message": "No interaction analysis available"}

    def get_enhanced_learning_phase_summary(self):
        """Generate a comprehensive summary with enhanced insights"""
        # First get standard summary
        summary = super().get_learning_phase_summary()

        # Add enhanced insights
        enhanced_insights = []

        # Analyze sparsity evolution
        sparsity_evolution = self._analyze_sparsity_evolution()
        if sparsity_evolution:
            enhanced_insights.extend(sparsity_evolution)

        # Analyze circuit specialization
        circuit_specialization = self._analyze_circuit_specialization()
        if circuit_specialization:
            enhanced_insights.extend(circuit_specialization)

        # Analyze attention-MLP coordination
        attn_mlp_coordination = self._analyze_attn_mlp_coordination()
        if attn_mlp_coordination:
            enhanced_insights.extend(attn_mlp_coordination)

        # Add enhanced insights to summary
        summary['enhanced_insights'] = enhanced_insights

        return summary

    def _analyze_sparsity_evolution(self):
        """Analyze how sparsity evolves through learning phases"""
        if not self.mlp_sparsity_tracker.sparsity_history:
            return []

        insights = []

        # Get phase boundaries
        if 'phases' in self.phase_structure:
            phases = self.phase_structure['phases']

            # Calculate average sparsity for each phase
            phase_sparsity = {}
            for i, phase in enumerate(phases):
                phase_epochs = [e for e in self.mlp_sparsity_tracker.sparsity_history.keys()
                                if phase['start_epoch'] <= e <= phase['end_epoch']]

                if phase_epochs:
                    # Average sparsity across layers and epochs in this phase
                    phase_avg = {}
                    for epoch in phase_epochs:
                        epoch_data = self.mlp_sparsity_tracker.sparsity_history[epoch]
                        for layer, sparsity in epoch_data['avg_sparsity'].items():
                            if layer not in phase_avg:
                                phase_avg[layer] = []
                            phase_avg[layer].append(sparsity)

                    # Compute average for each layer
                    phase_sparsity[f"phase_{i + 1}"] = {
                        'avg_sparsity': {layer: np.mean(values) for layer, values in phase_avg.items()},
                        'phase_class': phase.get('classification', 'unknown')
                    }

            # Analyze trends across phases
            if len(phase_sparsity) >= 2:
                phase_ids = sorted(phase_sparsity.keys())

                # Check for trend in sparsity
                first_phase = phase_sparsity[phase_ids[0]]
                last_phase = phase_sparsity[phase_ids[-1]]

                # Calculate average sparsity across layers
                first_avg = np.mean(list(first_phase['avg_sparsity'].values()))
                last_avg = np.mean(list(last_phase['avg_sparsity'].values()))

                sparsity_change = last_avg - first_avg

                # Generate insight
                if sparsity_change > 0.1:
                    insights.append(
                        f"MLP representations become increasingly sparse throughout training, " +
                        f"with sparsity increasing by {sparsity_change:.1%} from early to late phases."
                    )
                elif sparsity_change < -0.1:
                    insights.append(
                        f"MLP representations become less sparse throughout training, " +
                        f"with sparsity decreasing by {abs(sparsity_change):.1%} from early to late phases."
                    )
                else:
                    insights.append(
                        "MLP representation sparsity remains relatively stable throughout training phases."
                    )

                # Check for correlation between phase type and sparsity
                exploration_phases = [
                    phase_id for phase_id, data in phase_sparsity.items()
                    if data['phase_class'] == 'exploration'
                ]

                consolidation_phases = [
                    phase_id for phase_id, data in phase_sparsity.items()
                    if data['phase_class'] == 'consolidation'
                ]

                if exploration_phases and consolidation_phases:
                    # Compare average sparsity
                    exploration_sparsity = np.mean([
                        np.mean(list(phase_sparsity[phase_id]['avg_sparsity'].values()))
                        for phase_id in exploration_phases
                    ])

                    consolidation_sparsity = np.mean([
                        np.mean(list(phase_sparsity[phase_id]['avg_sparsity'].values()))
                        for phase_id in consolidation_phases
                    ])

                    diff = consolidation_sparsity - exploration_sparsity

                    if diff > 0.15:
                        insights.append(
                            f"Consolidation phases show significantly higher MLP sparsity " +
                            f"({consolidation_sparsity:.1%}) compared to exploration phases " +
                            f"({exploration_sparsity:.1%}), suggesting feature refinement."
                        )
                    elif diff < -0.15:
                        insights.append(
                            f"Exploration phases show higher MLP sparsity " +
                            f"({exploration_sparsity:.1%}) compared to consolidation phases " +
                            f"({consolidation_sparsity:.1%}), suggesting feature diversification."
                        )

        return insights

    def _analyze_circuit_specialization(self):
        """Analyze how circuits specialize throughout learning"""
        insights = []

        # Check for circuit-class analysis history
        circuit_epochs = [
            epoch for epoch, data in self.enhanced_analysis_history.items()
            if data.get('class_circuit_analysis') is not None
        ]

        if len(circuit_epochs) < 2:
            return []

        # Extract specialization percentages over time
        specialization_trend = []
        for epoch in sorted(circuit_epochs):
            analysis = self.enhanced_analysis_history[epoch]['class_circuit_analysis']
            if 'class_specific_circuits' in analysis:
                circuits = analysis['class_specific_circuits']
                if 'specialized_percentage' in circuits:
                    specialization_trend.append((epoch, circuits['specialized_percentage']))

        if len(specialization_trend) >= 2:
            # Check for trend in specialization
            first_epoch, first_spec = specialization_trend[0]
            last_epoch, last_spec = specialization_trend[-1]

            spec_change = last_spec - first_spec

            # Generate insight
            if spec_change > 15:
                insights.append(
                    f"Circuits become increasingly specialized for specific classes over time, " +
                    f"with specialization increasing from {first_spec:.1f}% to {last_spec:.1f}%."
                )
            elif spec_change < -15:
                insights.append(
                    f"Circuits become more general-purpose over time, " +
                    f"with specialization decreasing from {first_spec:.1f}% to {last_spec:.1f}%."
                )
            else:
                insights.append(
                    f"Circuit specialization remains relatively stable around {np.mean([s for _, s in specialization_trend]):.1f}% " +
                    f"throughout training."
                )

        # Check for correlation with grokking
        if self.logger and 'grokking_phases' in self.logger.logs:
            grokking_step = self.logger.logs['grokking_phases'].get('grokking_step')
            # Add debug logging here
            from analysis.utils.utils import debug_grokking_step
            debug_grokking_step(grokking_step, "EnhancedPhaseAnalyzer._analyze_circuit_specialization", self.logger)

            if grokking_step:
                # Handle both list and scalar cases for grokking_step
                grokking_steps = grokking_step if isinstance(grokking_step, list) else [grokking_step]

                for step in grokking_steps:
                    # Find closest circuit analysis to this grokking point
                    try:
                        closest_epoch = min(circuit_epochs, key=lambda x: abs(x - step))

                        if abs(closest_epoch - step) <= 50:  # Within a reasonable window
                            analysis = self.enhanced_analysis_history[closest_epoch]['class_circuit_analysis']
                            if 'class_specific_circuits' in analysis:
                                circuits = analysis['class_specific_circuits']
                                spec_percentage = circuits.get('specialized_percentage', 0)

                                specialization_level = 'strong' if spec_percentage > 60 else 'moderate' if spec_percentage > 30 else 'limited'
                                insights.append(
                                    f"At the grokking transition (epoch {step}), " +
                                    f"{spec_percentage:.1f}% of circuits were specialized for specific classes, " +
                                    f"suggesting {specialization_level} circuit specialization during grokking."
                                )
                    except (TypeError, ValueError) as e:
                        # Log the error but continue
                        print(f"\tEnhancedPhaseAnalyzer_analyze_circuit_specialization():\tError analyzing circuit specialization for grokking step {step}: {e}")
                        # You might want to add more diagnostic information here
                        print(f"\t\tTypes: grokking_step={type(step)}, circuit_epochs={type(circuit_epochs)}")
                        print(
                            f"\t\tValues: step={step}, circuit_epochs={circuit_epochs[:5] if len(circuit_epochs) > 5 else circuit_epochs}")

        return insights

    def _analyze_attn_mlp_coordination(self):
        """Analyze how attention and MLP components coordinate"""
        insights = []

        # Check for interaction analysis history
        interaction_epochs = [
            epoch for epoch, data in self.enhanced_analysis_history.items()
            if data.get('interaction_analysis') is not None
        ]

        if len(interaction_epochs) < 2:
            return []

        # Extract coordination metrics over time
        coordination_trend = []
        for epoch in sorted(interaction_epochs):
            analysis = self.enhanced_analysis_history[epoch]['interaction_analysis']
            if 'layer_correlations' in analysis:
                correlations = analysis['layer_correlations']
                avg_correlation = np.mean(list(correlations.values()))
                coordination_trend.append((epoch, avg_correlation))

        if len(coordination_trend) >= 2:
            # Check for trend in coordination
            first_epoch, first_coord = coordination_trend[0]
            last_epoch, last_coord = coordination_trend[-1]

            coord_change = last_coord - first_coord

            # Generate insight
            if coord_change > 0.15:
                insights.append(
                    f"Attention-MLP coordination strengthens significantly throughout training, " +
                    f"with average correlation increasing from {first_coord:.2f} to {last_coord:.2f}."
                )
            elif coord_change < -0.15:
                insights.append(
                    f"Attention-MLP coordination weakens throughout training, " +
                    f"with average correlation decreasing from {first_coord:.2f} to {last_coord:.2f}."
                )
            else:
                insights.append(
                    f"Attention-MLP coordination remains relatively stable " +
                    f"at {np.mean([c for _, c in coordination_trend]):.2f} throughout training."
                )

        # Check for cross-component circuits
        for epoch in sorted(interaction_epochs):
            analysis = self.enhanced_analysis_history[epoch]['interaction_analysis']

            if 'cross_component_circuits' in analysis.get('insights', {}):
                cross_circuits = analysis['insights']['cross_component_circuits']

                if cross_circuits['count'] > 0:
                    insights.append(
                        f"At epoch {epoch}, found {cross_circuits['count']} circuits spanning " +
                        f"attention and MLP components with average strength {cross_circuits['avg_strength']:.2f}, " +
                        f"indicating unified computational paths across component types."
                    )
                    break  # Just report the first occurrence

        return insights

    def _process_tensor_for_storage(self, tensor):
        """Convert tensor to memory-efficient format for storage"""
        if tensor is None:
            return None

        # Detach from computation graph and move to CPU
        if isinstance(tensor, torch.Tensor):
            # Use float16 for storage of activation patterns
            if tensor.dim() > 1 and tensor.shape[0] > 10:
                # For large tensors, use half precision
                return tensor.detach().cpu().half()
            return tensor.detach().cpu()
        return tensor

    def _cleanup(self):
        """Clean up resources to prevent memory leaks"""
        self.mlp_sparsity_tracker.cleanup()
        self.interaction_analyzer.cleanup()
        # Call parent cleanup if available
        if hasattr(super(), 'cleanup'):
            super().cleanup()

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
