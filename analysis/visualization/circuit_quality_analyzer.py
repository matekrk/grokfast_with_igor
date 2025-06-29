# analysis/visualization/circuit_quality_analyzer.py
"""
Circuit Visualization and Quality Assessment Tools

Helps distinguish meaningful circuits from spurious ones through multiple visualization
and analysis approaches.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import networkx as nx
from pathlib import Path


class CircuitQualityAnalyzer:
    """
    Comprehensive circuit analysis and visualization system
    """

    def __init__(self, model, canonical_registry, save_dir=None):
        self.model = model
        self.canonical_registry = canonical_registry
        self.save_dir = Path(save_dir) if save_dir else None

        # Quality metrics
        self.circuit_scores = {}
        self.visualization_cache = {}

    def analyze_all_circuits(self, eval_loader, max_circuits=50) -> Dict[str, Any]:
        """
        Comprehensive analysis of all circuits to identify the most meaningful ones
        """

        # print(f"🔍 Analyzing quality of {len(self.canonical_registry.canonical_circuits)} circuits...")

        circuit_analyses = {}

        for i, (circuit_id, canonical_circuit) in enumerate(self.canonical_registry.canonical_circuits.items()):
            if i >= max_circuits:
                break

            # print(
            #     f"  📊 Analyzing circuit {i + 1}/{min(max_circuits, len(self.canonical_registry.canonical_circuits))}: {circuit_id}")

            analysis = self.analyze_single_circuit(circuit_id, canonical_circuit, eval_loader)
            circuit_analyses[circuit_id] = analysis

        # Rank circuits by quality
        ranked_circuits = self.rank_circuits_by_quality(circuit_analyses)

        # Generate summary report
        quality_report = self.generate_quality_report(ranked_circuits)

        return {
            'circuit_analyses': circuit_analyses,
            'ranked_circuits': ranked_circuits,
            'quality_report': quality_report
        }

    def analyze_single_circuit(self, circuit_id: str, canonical_circuit, eval_loader) -> Dict[str, Any]:
        """
        Deep analysis of a single circuit
        """

        # Get circuit metadata
        operation_type = canonical_circuit.computational_signature.operation_type
        total_detections = canonical_circuit.total_detections
        stability_score = canonical_circuit.stability_score

        # Analyze temporal dynamics
        temporal_analysis = self.analyze_temporal_dynamics(canonical_circuit)

        # Analyze functional behavior
        functional_analysis = self.analyze_functional_behavior(circuit_id, canonical_circuit, eval_loader)

        # Analyze attention patterns
        attention_analysis = self.analyze_attention_patterns(canonical_circuit)

        # Calculate quality score
        quality_score = self.calculate_quality_score(
            temporal_analysis, functional_analysis, attention_analysis, stability_score
        )

        return {
            'circuit_id': circuit_id,
            'operation_type': operation_type,
            'total_detections': total_detections,
            'stability_score': stability_score,
            'quality_score': quality_score,
            'temporal_analysis': temporal_analysis,
            'functional_analysis': functional_analysis,
            'attention_analysis': attention_analysis,
            'metadata': canonical_circuit.metadata
        }

    def analyze_temporal_dynamics(self, canonical_circuit) -> Dict[str, Any]:
        """
        Analyze how circuit strength changes over time (key for detecting spurious circuits)
        """

        history = canonical_circuit.attribution_history
        if len(history) < 10:
            return {'status': 'insufficient_data'}

        epochs = [epoch for epoch, _ in history]
        strengths = [strength for _, strength in history]

        # Calculate trends
        early_strength = np.mean(strengths[:len(strengths) // 3])
        middle_strength = np.mean(strengths[len(strengths) // 3:2 * len(strengths) // 3])
        late_strength = np.mean(strengths[2 * len(strengths) // 3:])

        # Trend analysis
        trend_slope = np.polyfit(epochs, strengths, 1)[0]
        strength_variance = np.var(strengths)

        # Phase transition detection
        phase_transitions = self.detect_phase_transitions(epochs, strengths)

        return {
            'early_strength': early_strength,
            'middle_strength': middle_strength,
            'late_strength': late_strength,
            'trend_slope': trend_slope,
            'strength_variance': strength_variance,
            'phase_transitions': phase_transitions,
            'temporal_consistency': 1.0 / (1.0 + strength_variance),  # Higher is better
            'strengthening_trend': trend_slope > 0.001,  # Is it getting stronger?
            'epochs_analyzed': len(history)
        }

    def analyze_functional_behavior(self, circuit_id: str, canonical_circuit, eval_loader) -> Dict[str, Any]:
        """
        info Test what the circuit actually computes on diverse examples
        """

        # Get diverse examples for testing
        test_examples = []
        for i, (inputs, targets) in enumerate(eval_loader):
            if i >= 10:  # Test on 10 examples  # fixme actually, it tests on 10 batches!
                break
            test_examples.append((inputs, targets))

        # Test circuit behavior
        behaviors = []
        consistencies = []

        for inputs, targets in test_examples:
            try:
                behavior = self.test_circuit_on_example(circuit_id, canonical_circuit, inputs, targets)
                behaviors.append(behavior)

                # Check consistency with expected operation type
                expected_type = canonical_circuit.computational_signature.operation_type
                consistency = self.measure_behavior_consistency(behavior, expected_type)
                consistencies.append(consistency)

            except Exception as e:
                print(f"    ⚠️ Failed to test circuit {circuit_id}: {e}")
                continue

        if not consistencies:
            return {'status': 'test_failed'}

        return {
            'avg_consistency': np.mean(consistencies),
            'behavior_variance': np.var(consistencies),
            'examples_tested': len(behaviors),
            'behaviors': behaviors[:3],  # Store first 3 for inspection
            'functional_quality': np.mean(consistencies) * (1.0 - np.var(consistencies))
        }

    def analyze_attention_patterns(self, canonical_circuit) -> Dict[str, Any]:
        """
        Analyze the attention patterns to assess circuit quality
        """

        # Get recent instances for pattern analysis
        recent_instances = canonical_circuit.instances[-5:]  # Last 5 instances

        if not recent_instances:
            return {'status': 'no_instances'}

        pattern_similarities = []
        attention_entropies = []

        for instance in recent_instances:
            # Analyze attention entropy (lower = more focused = better)
            if 'attention_strength' in instance.example_metadata:
                strength = instance.example_metadata['attention_strength']
                attention_entropies.append(-strength * np.log(strength + 1e-8))

        # Pattern consistency across instances
        if len(recent_instances) >= 2:
            for i in range(len(recent_instances) - 1):
                similarity = self.calculate_instance_similarity(recent_instances[i], recent_instances[i + 1])
                pattern_similarities.append(similarity)

        return {
            'avg_attention_entropy': np.mean(attention_entropies) if attention_entropies else 0,
            'pattern_consistency': np.mean(pattern_similarities) if pattern_similarities else 0,
            'instances_analyzed': len(recent_instances),
            'attention_quality': 1.0 / (1.0 + np.mean(attention_entropies)) if attention_entropies else 0
        }

    def calculate_quality_score(self, temporal_analysis, functional_analysis,
                                attention_analysis, stability_score) -> float:
        """
        Calculate overall circuit quality score (0-1, higher is better)
        """

        scores = []
        weights = []

        # Temporal quality (30% weight)
        if temporal_analysis.get('status') != 'insufficient_data':
            temporal_score = (
                    0.4 * temporal_analysis.get('temporal_consistency', 0) +
                    0.3 * (1.0 if temporal_analysis.get('strengthening_trend', False) else 0.3) +
                    0.3 * min(1.0, len(temporal_analysis.get('phase_transitions', [])) / 2.0)
            # Some transitions are good
            )
            scores.append(temporal_score)
            weights.append(0.3)

        # Functional quality (40% weight)
        if functional_analysis.get('status') != 'test_failed':
            functional_score = functional_analysis.get('functional_quality', 0)
            scores.append(functional_score)
            weights.append(0.4)

        # Attention quality (20% weight)
        if attention_analysis.get('status') != 'no_instances':
            attention_score = (
                    0.6 * attention_analysis.get('attention_quality', 0) +
                    0.4 * attention_analysis.get('pattern_consistency', 0)
            )
            scores.append(attention_score)
            weights.append(0.2)

        # Stability (10% weight)
        scores.append(stability_score)
        weights.append(0.1)

        # Weighted average
        if sum(weights) > 0:
            quality_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            quality_score = 0.0

        return min(1.0, max(0.0, quality_score))

    def detect_phase_transitions(self, epochs, strengths) -> List[Dict[str, Any]]:
        """
        Detect phase transitions in circuit strength
        """

        if len(strengths) < 20:
            return []

        # Simple changepoint detection using moving averages
        window_size = max(5, len(strengths) // 10)
        transitions = []

        for i in range(window_size, len(strengths) - window_size):
            before_avg = np.mean(strengths[i - window_size:i])
            after_avg = np.mean(strengths[i:i + window_size])

            # Significant change?
            change_magnitude = abs(after_avg - before_avg)
            if change_magnitude > 0.1:  # Threshold for significant change
                transitions.append({
                    'epoch': epochs[i],
                    'change_magnitude': change_magnitude,
                    'direction': 'increase' if after_avg > before_avg else 'decrease'
                })

        return transitions

    def visualize_circuit_quality(self, circuit_analyses: Dict[str, Any], top_n: int = 20):
        """
        Create comprehensive visualizations of circuit quality
        """

        # Sort circuits by quality
        sorted_circuits = sorted(
            circuit_analyses.items(),
            key=lambda x: x[1]['quality_score'],
            reverse=True
        )

        top_circuits = sorted_circuits[:top_n]

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Circuit Quality Analysis - Top {top_n} Circuits', fontsize=16)

        # 1. Quality Score Distribution
        quality_scores = [analysis['quality_score'] for _, analysis in sorted_circuits]
        axes[0, 0].hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(quality_scores), color='red', linestyle='--',
                           label=f'Mean: {np.mean(quality_scores):.3f}')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Circuit Quality Distribution')
        axes[0, 0].legend()

        # 2. Quality vs Stability
        qualities = [analysis['quality_score'] for _, analysis in top_circuits]
        stabilities = [analysis['stability_score'] for _, analysis in top_circuits]
        axes[0, 1].scatter(stabilities, qualities, alpha=0.6, s=50)
        axes[0, 1].set_xlabel('Stability Score')
        axes[0, 1].set_ylabel('Quality Score')
        axes[0, 1].set_title('Quality vs Stability')

        # Add circuit type colors
        operation_types = [analysis['operation_type'] for _, analysis in top_circuits]
        unique_types = list(set(operation_types))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        for i, op_type in enumerate(unique_types):
            mask = [t == op_type for t in operation_types]
            axes[0, 1].scatter(
                [s for s, m in zip(stabilities, mask) if m],
                [q for q, m in zip(qualities, mask) if m],
                label=op_type, alpha=0.7, s=50, c=[colors[i]]
            )
        axes[0, 1].legend()

        # 3. Top Circuits Bar Chart
        circuit_ids = [cid[:15] + '...' if len(cid) > 15 else cid for cid, _ in top_circuits[:10]]
        top_qualities = [analysis['quality_score'] for _, analysis in top_circuits[:10]]
        bars = axes[0, 2].bar(range(len(circuit_ids)), top_qualities, color='lightgreen', alpha=0.7)
        axes[0, 2].set_xlabel('Circuit Rank')
        axes[0, 2].set_ylabel('Quality Score')
        axes[0, 2].set_title('Top 10 Circuits by Quality')
        axes[0, 2].set_xticks(range(len(circuit_ids)))
        axes[0, 2].set_xticklabels(circuit_ids, rotation=45, ha='right')

        # 4. Temporal Trends of Top Circuits
        for i, (circuit_id, analysis) in enumerate(top_circuits[:5]):
            temporal = analysis.get('temporal_analysis', {})
            if 'epochs_analyzed' in temporal:
                # Reconstruct approximate timeline
                epochs = list(range(temporal['epochs_analyzed']))
                # Simulate strength evolution (in real implementation, use actual history)
                trend_slope = temporal.get('trend_slope', 0)
                base_strength = temporal.get('early_strength', 0.5)
                strengths = [base_strength + trend_slope * e + np.random.normal(0, 0.02) for e in epochs]

                axes[1, 0].plot(epochs, strengths, label=f'Circuit {i + 1}', alpha=0.7)

        axes[1, 0].set_xlabel('Epoch (relative)')
        axes[1, 0].set_ylabel('Circuit Strength')
        axes[1, 0].set_title('Temporal Evolution of Top Circuits')
        axes[1, 0].legend()

        # 5. Operation Type Distribution
        all_types = [analysis['operation_type'] for _, analysis in sorted_circuits]
        type_counts = {op_type: all_types.count(op_type) for op_type in set(all_types)}
        axes[1, 1].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Circuit Operation Types')

        # 6. Quality Components Breakdown (for top 10)
        components = ['Temporal', 'Functional', 'Attention', 'Stability']
        component_scores = []

        for _, analysis in top_circuits[:10]:
            temporal = analysis.get('temporal_analysis', {})
            functional = analysis.get('functional_analysis', {})
            attention = analysis.get('attention_analysis', {})

            scores = [
                temporal.get('temporal_consistency', 0),
                functional.get('functional_quality', 0),
                attention.get('attention_quality', 0),
                analysis['stability_score']
            ]
            component_scores.append(scores)

        # Create stacked bar chart
        component_scores = np.array(component_scores).T
        bottom = np.zeros(len(top_circuits[:10]))

        for i, (component, scores) in enumerate(zip(components, component_scores)):
            axes[1, 2].bar(range(len(top_circuits[:10])), scores, bottom=bottom,
                           label=component, alpha=0.7)
            bottom += scores

        axes[1, 2].set_xlabel('Top Circuits')
        axes[1, 2].set_ylabel('Component Scores')
        axes[1, 2].set_title('Quality Components Breakdown')
        axes[1, 2].legend()

        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / 'circuit_quality_analysis.png', dpi=300, bbox_inches='tight')

        plt.show()

        return fig

    def visualize_circuit_attention_patterns(self, circuit_id: str, canonical_circuit,
                                             num_examples: int = 3):
        """
        Visualize attention patterns for a specific circuit
        """

        if len(canonical_circuit.instances) == 0:
            print(f"No instances available for circuit {circuit_id}")
            return None

        # Get recent instances
        instances = canonical_circuit.instances[-num_examples:]

        fig, axes = plt.subplots(1, len(instances), figsize=(6 * len(instances), 5))
        if len(instances) == 1:
            axes = [axes]

        fig.suptitle(f'Attention Patterns for Circuit: {circuit_id}', fontsize=14)

        for i, instance in enumerate(instances):
            # Create attention heatmap
            tokens = instance.tokens
            attention_strength = instance.example_metadata.get('attention_strength', 0.5)

            # Create mock attention matrix (in real implementation, extract from stored patterns)
            seq_len = len(tokens)
            attention_matrix = np.random.random((seq_len, seq_len)) * 0.1

            # Add circuit-specific pattern
            source_pos = instance.positions.get('source', 0)
            target_pos = instance.positions.get('target', seq_len - 1)

            if source_pos < seq_len and target_pos < seq_len:
                attention_matrix[target_pos, source_pos] = attention_strength

            # Plot heatmap
            im = axes[i].imshow(attention_matrix, cmap='Blues', aspect='auto')
            axes[i].set_title(f'Instance {i + 1}\nTokens: {" ".join(tokens[:10])}{"..." if len(tokens) > 10 else ""}')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')

            # Add colorbar
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

            # Highlight circuit positions
            if source_pos < seq_len and target_pos < seq_len:
                axes[i].plot(source_pos, target_pos, 'ro', markersize=8, label='Circuit Connection')
                axes[i].legend()

        plt.tight_layout()

        if self.save_dir:
            safe_id = circuit_id.replace('/', '_').replace('\\', '_')[:50]
            plt.savefig(self.save_dir / f'attention_patterns_{safe_id}.png', dpi=300, bbox_inches='tight')

        plt.show()

        return fig

    def rank_circuits_by_quality(self, circuit_analyses: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Rank circuits by quality score and provide recommendations
        """

        ranked = sorted(
            circuit_analyses.items(),
            key=lambda x: x[1]['quality_score'],
            reverse=True
        )

        # Add rankings and recommendations
        for i, (circuit_id, analysis) in enumerate(ranked):
            analysis['rank'] = i + 1
            analysis['recommendation'] = self.get_circuit_recommendation(analysis)

        return ranked

    def get_circuit_recommendation(self, analysis: Dict[str, Any]) -> str:
        """
        Get recommendation for what to do with this circuit
        """

        quality = analysis['quality_score']
        stability = analysis['stability_score']
        operation_type = analysis['operation_type']

        if quality >= 0.8 and stability >= 0.7:
            return "KEEP - High quality and stable"
        elif quality >= 0.6 and stability >= 0.5:
            return "MONITOR - Good circuit, watch for improvements"
        elif quality < 0.3 or stability < 0.2:
            return "REMOVE - Likely spurious circuit"
        elif operation_type in ['unknown', 'generic']:
            return "INVESTIGATE - Unclear operation type"
        else:
            return "UNCERTAIN - Needs more analysis"

    def generate_quality_report(self, ranked_circuits: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate comprehensive quality report
        """

        total_circuits = len(ranked_circuits)

        recommendations = [analysis['recommendation'] for _, analysis in ranked_circuits]
        keep_count = sum(1 for r in recommendations if r.startswith('KEEP'))
        remove_count = sum(1 for r in recommendations if r.startswith('REMOVE'))
        monitor_count = sum(1 for r in recommendations if r.startswith('MONITOR'))

        operation_types = {}
        for _, analysis in ranked_circuits:
            op_type = analysis['operation_type']
            operation_types[op_type] = operation_types.get(op_type, 0) + 1

        quality_stats = {
            'mean_quality': np.mean([analysis['quality_score'] for _, analysis in ranked_circuits]),
            'median_quality': np.median([analysis['quality_score'] for _, analysis in ranked_circuits]),
            'std_quality': np.std([analysis['quality_score'] for _, analysis in ranked_circuits])
        }

        return {
            'total_circuits': total_circuits,
            'keep_circuits': keep_count,
            'remove_circuits': remove_count,
            'monitor_circuits': monitor_count,
            'operation_type_distribution': operation_types,
            'quality_statistics': quality_stats,
            'recommendations': {
                f'KEEP ({keep_count})': [cid for cid, analysis in ranked_circuits if
                                         analysis['recommendation'].startswith('KEEP')][:5],
                f'REMOVE ({remove_count})': [cid for cid, analysis in ranked_circuits if
                                             analysis['recommendation'].startswith('REMOVE')][:5],
                f'TOP_5_QUALITY': [(cid, analysis['quality_score']) for cid, analysis in ranked_circuits[:5]]
            }
        }

    # Helper methods (simplified implementations)
    def test_circuit_on_example(self, circuit_id, canonical_circuit, inputs, targets):
        # Simplified - in real implementation, run activation patching
        return {'behavior': 'mock_behavior', 'strength': 0.5}

    def measure_behavior_consistency(self, behavior, expected_type):
        # Simplified consistency check
        return np.random.random() * 0.8 + 0.2  # Mock consistency score

    def calculate_instance_similarity(self, instance1, instance2):
        # Simplified similarity calculation
        return np.random.random() * 0.5 + 0.5  # Mock similarity


# ============================================================================
# PHASE TRANSITION CIRCUIT ANALYSIS
# ============================================================================

class PhaseTransitionCircuitAnalyzer:
    """
    Analyze relationship between phase transitions and circuit emergence
    """

    def __init__(self, canonical_registry, training_metrics_history):
        self.canonical_registry = canonical_registry
        self.training_metrics = training_metrics_history  # Should include loss, accuracy over epochs

    def detect_training_phase_transitions(self) -> List[Dict[str, Any]]:
        """
        Detect major phase transitions in training (e.g., grokking transitions)
        """

        if 'accuracy' not in self.training_metrics or len(self.training_metrics['accuracy']) < 100:
            return []

        epochs = list(range(len(self.training_metrics['accuracy'])))
        accuracy = self.training_metrics['accuracy']
        loss = self.training_metrics.get('loss', [0] * len(accuracy))

        transitions = []

        # Detect grokking transition (sudden accuracy improvement)
        for i in range(50, len(accuracy) - 50):
            # Check for sudden accuracy jump
            before_window = accuracy[i - 20:i]
            after_window = accuracy[i:i + 20]

            before_avg = np.mean(before_window)
            after_avg = np.mean(after_window)

            if after_avg - before_avg > 0.1 and before_avg < 0.6 and after_avg > 0.7:
                transitions.append({
                    'epoch': epochs[i],
                    'type': 'grokking_transition',
                    'accuracy_jump': after_avg - before_avg,
                    'before_accuracy': before_avg,
                    'after_accuracy': after_avg
                })

        # Detect other transitions (loss plateaus, etc.)
        # ... additional transition detection logic

        return transitions

    def analyze_circuit_emergence_around_transitions(self, transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze which circuits emerge around phase transitions
        """

        results = {}

        for transition in transitions:
            transition_epoch = transition['epoch']
            transition_type = transition['type']

            # Find circuits that emerged around this transition
            pre_transition_circuits = []
            during_transition_circuits = []
            post_transition_circuits = []

            for circuit_id, canonical_circuit in self.canonical_registry.canonical_circuits.items():
                first_seen = canonical_circuit.first_seen

                if first_seen < transition_epoch - 50:
                    pre_transition_circuits.append(circuit_id)
                elif transition_epoch - 50 <= first_seen <= transition_epoch + 50:
                    during_transition_circuits.append(circuit_id)
                elif first_seen > transition_epoch + 50:
                    post_transition_circuits.append(circuit_id)

            results[f"{transition_type}_epoch_{transition_epoch}"] = {
                'transition': transition,
                'pre_transition_circuits': pre_transition_circuits,
                'during_transition_circuits': during_transition_circuits,
                'post_transition_circuits': post_transition_circuits,
                'new_circuits_during_transition': len(during_transition_circuits)
            }

        return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def analyze_circuit_quality_in_training_loop(canonical_detector, eval_loader, epoch, save_dir):
    """
    Integration example for circuit quality analysis
    """

    # Run every 100 epochs or so
    if epoch % 100 == 0 and epoch > 200:
        print(f"\n🔍 Running circuit quality analysis @ epoch {epoch}")

        analyzer = CircuitQualityAnalyzer(
            model=canonical_detector.model,
            canonical_registry=canonical_detector.canonical_registry,
            save_dir=save_dir
        )

        # Analyze all circuits
        analysis_results = analyzer.analyze_all_circuits(eval_loader, max_circuits=50)

        # Generate visualizations
        analyzer.visualize_circuit_quality(analysis_results['circuit_analyses'])

        # Print recommendations
        quality_report = analysis_results['quality_report']
        print(f"\n📊 Circuit Quality Report:")
        print(f"  Total circuits: {quality_report['total_circuits']}")
        print(f"  Recommended to KEEP: {quality_report['keep_circuits']}")
        print(f"  Recommended to REMOVE: {quality_report['remove_circuits']}")
        print(f"  Mean quality score: {quality_report['quality_statistics']['mean_quality']:.3f}")

        # Show top circuits
        print(f"\n🏆 Top 5 circuits by quality:")
        for i, (circuit_id, quality_score) in enumerate(quality_report['recommendations']['TOP_5_QUALITY'], 1):
            print(f"  {i}. {circuit_id}: {quality_score:.3f}")

        return analysis_results

    return None