# analysis/temporal/temporal_circuit_emergence_analyzer.py
"""
Temporal Circuit Emergence Analyzer - Integrated Solution

Combines CircuitEvolutionAnalyzer (core tracking) with specialized emergence analysis.
Simple integration that extends existing functionality without breaking compatibility.

Key Features:
- Extends CircuitEvolutionAnalyzer for full compatibility
- Adds specialized emergence pattern detection
- Provides cascade and dependency chain analysis
- Maintains all existing research methods
- Easy integration with existing tracker/registry
"""

from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field

# Import the fixed core analyzer
from analysis.core.circuit_evolution_analyzer import CircuitEvolutionAnalyzer
from analysis.core.circuit_schema import (
    EmergencePhase, EvolutionPattern, LearningPhase, InteractionType,
    CircuitLevel, GrokkingPhase, CircuitType, InteractionEvent
)


@dataclass
class EmergenceCascade:
    """ info represents a cascade of circuit emergence events"""
    cascade_id: str
    trigger_circuit: str
    trigger_epoch: int
    enabled_circuits: List[Tuple[str, int]] = field(default_factory=list)  # (circuit_id, epoch)
    cascade_strength: float = 0.0
    temporal_span: int = 0
    cascade_type: str = "linear"  # linear, exponential, wave

    def get_cascade_rate(self) -> float:
        """Calculate rate of circuit emergence in cascade"""
        if self.temporal_span == 0:
            return 0.0
        return len(self.enabled_circuits) / self.temporal_span


@dataclass
class DependencyChain:
    """ info represents a chain of circuit dependencies"""
    chain_id: str
    circuits: List[str] = field(default_factory=list)
    formation_epochs: List[int] = field(default_factory=list)
    chain_strength: float = 0.0
    chain_type: str = "sequential"  # sequential, parallel, hierarchical

    def get_formation_rate(self) -> float:
        """Calculate rate of dependency chain formation"""
        if len(self.formation_epochs) < 2:
            return 0.0
        return len(self.circuits) / (max(self.formation_epochs) - min(self.formation_epochs))


class TemporalCircuitEmergenceAnalyzer(CircuitEvolutionAnalyzer):
    """
    info INTEGRATED: Combines evolution tracking with specialized emergence analysis

    Extends CircuitEvolutionAnalyzer with:
    - Emergence cascade detection
    - Dependency chain analysis
    - Temporal pattern recognition
    - Grokking transition analysis
    """

    def __init__(self, evolution_tracker=None, enhanced_registry=None, storage_dir: Path = None):
        """Initialize with all existing functionality plus emergence analysis"""
        # Initialize parent class
        super().__init__(evolution_tracker, enhanced_registry, storage_dir)

        # Add emergence-specific analysis components
        self.emergence_cascades: Dict[str, EmergenceCascade] = {}
        self.dependency_chains: Dict[str, DependencyChain] = {}
        self.temporal_patterns: Dict[str, Any] = {}
        self.grokking_transitions: Dict[str, Dict[str, Any]] = {}

        # Analysis caches for performance
        self._emergence_order_cache: Optional[List[Tuple[str, int]]] = None
        self._dependency_graph_cache: Optional[Dict[str, Set[str]]] = None

        print("✅ Temporal Circuit Emergence Analyzer initialized")
        print("   📊 Core evolution analysis available")
        print("   📊 Specialized emergence analysis added")

    # ============================================================================
    # EMERGENCE PATTERN ANALYSIS - New Specialized Methods
    # ============================================================================

    def track_emergence_order(self) -> List[Tuple[str, int]]:
        """
        Analyze temporal order of circuit emergence to identify enabling relationships

        Returns:
            List of (circuit_id, emergence_epoch) sorted by emergence time
        """
        if self._emergence_order_cache is not None:
            return self._emergence_order_cache

        emergence_order = []

        # Extract emergence order from evolution tracker data
        for circuit_id, emergence_epoch in self.emergence_epochs.items():
            emergence_order.append((circuit_id, emergence_epoch))

        # Sort by emergence epoch
        emergence_order.sort(key=lambda x: x[1])

        self._emergence_order_cache = emergence_order

        print(f"📊 Emergence order tracked: {len(emergence_order)} circuits")
        return emergence_order

    def detect_emergence_cascades(self, cascade_window: int = 10) -> Dict[str, EmergenceCascade]:
        """
        Detect cascades where one circuit emergence triggers others

        Args:
            cascade_window: Epoch window to consider for cascade detection
        """
        emergence_order = self.track_emergence_order()
        cascades = {}

        for i, (trigger_circuit, trigger_epoch) in enumerate(emergence_order):
            # Look for circuits that emerged shortly after this one
            enabled_circuits = []

            for j in range(i + 1, len(emergence_order)):
                other_circuit, other_epoch = emergence_order[j]

                # If within cascade window, consider it part of cascade
                if other_epoch <= trigger_epoch + cascade_window:
                    enabled_circuits.append((other_circuit, other_epoch))
                else:
                    break  # Outside window

            # Only create cascade if there are enabled circuits
            if enabled_circuits:
                cascade_id = f"cascade_{trigger_circuit}_{trigger_epoch}"
                temporal_span = max([epoch for _, epoch in enabled_circuits]) - trigger_epoch

                cascade = EmergenceCascade(
                    cascade_id=cascade_id,
                    trigger_circuit=trigger_circuit,
                    trigger_epoch=trigger_epoch,
                    enabled_circuits=enabled_circuits,
                    cascade_strength=len(enabled_circuits) / cascade_window,
                    temporal_span=temporal_span,
                    cascade_type=self._classify_cascade_type(enabled_circuits, trigger_epoch)
                )

                cascades[cascade_id] = cascade

        self.emergence_cascades = cascades

        print(f"🌊 Emergence cascades detected: {len(cascades)}")
        for cascade_id, cascade in cascades.items():
            print(f"   {cascade_id}: {len(cascade.enabled_circuits)} circuits, "
                  f"rate: {cascade.get_cascade_rate():.3f}")

        return cascades

    def _classify_cascade_type(self, enabled_circuits: List[Tuple[str, int]], trigger_epoch: int) -> str:
        """Classify the type of emergence cascade"""
        if not enabled_circuits:
            return "none"

        epochs = [epoch for _, epoch in enabled_circuits]
        epoch_diffs = [epoch - trigger_epoch for epoch in epochs]

        # Check if emergence rate increases (exponential)
        if len(epoch_diffs) > 2:
            early_rate = sum(1 for diff in epoch_diffs if diff <= 3)
            late_rate = sum(1 for diff in epoch_diffs if diff > 3)

            if early_rate > late_rate * 1.5:
                return "exponential"
            elif late_rate > early_rate * 1.5:
                return "wave"

        return "linear"

    def analyze_dependency_chains(self) -> Dict[str, DependencyChain]:
        """
        Analyze chains of circuit dependencies using interaction events
        """
        # Build dependency graph from interactions
        dependency_graph = defaultdict(list)
        circuit_epochs = {}

        # Extract dependencies from interaction events
        for event in self.interaction_log:
            if event.interaction_type == InteractionType.ENABLES:
                dependency_graph[event.source_circuit].append(event.target_circuit)
                circuit_epochs[event.source_circuit] = event.epoch
                circuit_epochs[event.target_circuit] = event.epoch

        # Find dependency chains using DFS
        chains = {}
        visited = set()

        def build_chain(start_circuit, current_chain, current_epochs):
            if start_circuit in visited:
                return

            visited.add(start_circuit)
            current_chain.append(start_circuit)
            if start_circuit in circuit_epochs:
                current_epochs.append(circuit_epochs[start_circuit])

            # Continue chain with enabled circuits
            if start_circuit in dependency_graph:
                for enabled_circuit in dependency_graph[start_circuit]:
                    build_chain(enabled_circuit, current_chain.copy(), current_epochs.copy())
            else:
                # End of chain - save if meaningful
                if len(current_chain) > 1:
                    chain_id = f"chain_{'_'.join(current_chain[:2])}_{len(current_chain)}"
                    chain_strength = len(current_chain) / max(current_epochs) if current_epochs else 0

                    chain = DependencyChain(
                        chain_id=chain_id,
                        circuits=current_chain,
                        formation_epochs=current_epochs,
                        chain_strength=chain_strength,
                        chain_type=self._classify_chain_type(current_chain, current_epochs)
                    )
                    chains[chain_id] = chain

        # Start from circuits with no prerequisites
        all_targets = set()
        for enabled_list in dependency_graph.values():
            all_targets.update(enabled_list)

        root_circuits = set(dependency_graph.keys()) - all_targets

        for root in root_circuits:
            build_chain(root, [], [])

        self.dependency_chains = chains

        print(f"🔗 Dependency chains analyzed: {len(chains)}")
        for chain_id, chain in chains.items():
            print(f"   {chain_id}: {len(chain.circuits)} circuits, "
                  f"rate: {chain.get_formation_rate():.3f}")

        return chains

    def _classify_chain_type(self, circuits: List[str], epochs: List[int]) -> str:
        """Classify the type of dependency chain"""
        if len(epochs) < 2:
            return "single"

        # Check temporal pattern
        time_gaps = [epochs[i + 1] - epochs[i] for i in range(len(epochs) - 1)]
        avg_gap = np.mean(time_gaps)
        std_gap = np.std(time_gaps)

        # Sequential: consistent gaps
        if std_gap < avg_gap * 0.3:
            return "sequential"
        # Parallel: many circuits emerge at similar times
        elif len(set(epochs)) < len(epochs) * 0.5:
            return "parallel"
        else:
            return "hierarchical"

    # ============================================================================
    # GROKKING TRANSITION ANALYSIS - Specialized Methods
    # ============================================================================

    def analyze_grokking_transitions(self, accuracy_history: List[float]) -> Dict[str, Any]:
        """
        Analyze how circuit emergence relates to grokking transitions

        Args:
            accuracy_history: Training accuracy over epochs
        """
        grokking_transitions = {}

        # Detect grokking points
        grokking_epochs = self._detect_grokking_epochs(accuracy_history)

        for grok_epoch in grokking_epochs:
            # Analyze circuit changes around grokking
            pre_grok_circuits = self._get_circuits_in_window(grok_epoch - 20, grok_epoch)
            post_grok_circuits = self._get_circuits_in_window(grok_epoch, grok_epoch + 20)

            transition_analysis = {
                'grokking_epoch': grok_epoch,
                'pre_grokking_circuits': pre_grok_circuits,
                'post_grokking_circuits': post_grok_circuits,
                'circuit_emergence_rate_change': self._calculate_emergence_rate_change(
                    pre_grok_circuits, post_grok_circuits
                ),
                'new_circuit_types': self._identify_new_circuit_types(
                    pre_grok_circuits, post_grok_circuits
                ),
                'circuit_transformations': self._analyze_circuit_transformations(
                    grok_epoch, pre_grok_circuits, post_grok_circuits
                )
            }

            grokking_transitions[f"grokking_{grok_epoch}"] = transition_analysis

        self.grokking_transitions = grokking_transitions

        print(f"🚀 Grokking transitions analyzed: {len(grokking_transitions)}")
        for transition_id, analysis in grokking_transitions.items():
            print(f"   {transition_id}: "
                  f"{len(analysis['pre_grokking_circuits'])} → {len(analysis['post_grokking_circuits'])} circuits")

        return grokking_transitions

    def _detect_grokking_epochs(self, accuracy_history: List[float]) -> List[int]:
        """Detect epochs where grokking occurred"""
        grokking_epochs = []

        if len(accuracy_history) < 20:
            return grokking_epochs

        # Look for sudden accuracy improvements
        for i in range(10, len(accuracy_history) - 10):
            before_window = accuracy_history[i - 10:i]
            after_window = accuracy_history[i:i + 10]

            before_avg = np.mean(before_window)
            after_avg = np.mean(after_window)

            # Grokking: sudden improvement from low to high accuracy
            if (after_avg - before_avg > 0.15 and
                    before_avg < 0.6 and
                    after_avg > 0.7):
                grokking_epochs.append(i)

        return grokking_epochs

    def _get_circuits_in_window(self, start_epoch: int, end_epoch: int) -> List[str]:
        """Get circuits that emerged in the specified epoch window"""
        circuits = []
        for circuit_id, emergence_epoch in self.emergence_epochs.items():
            if start_epoch <= emergence_epoch <= end_epoch:
                circuits.append(circuit_id)
        return circuits

    def _calculate_emergence_rate_change(self, pre_circuits: List[str], post_circuits: List[str]) -> float:
        """Calculate change in circuit emergence rate"""
        pre_rate = len(pre_circuits) / 20  # per epoch
        post_rate = len(post_circuits) / 20  # per epoch

        if pre_rate == 0:
            return float('inf') if post_rate > 0 else 0

        return (post_rate - pre_rate) / pre_rate

    def _identify_new_circuit_types(self, pre_circuits: List[str], post_circuits: List[str]) -> List[str]:
        """Identify new types of circuits that emerged post-grokking"""
        # Simple heuristic: look for new circuit name patterns
        pre_patterns = set()
        post_patterns = set()

        for circuit in pre_circuits:
            pattern = circuit.split('_')[0] if '_' in circuit else circuit
            pre_patterns.add(pattern)

        for circuit in post_circuits:
            pattern = circuit.split('_')[0] if '_' in circuit else circuit
            post_patterns.add(pattern)

        return list(post_patterns - pre_patterns)

    def _analyze_circuit_transformations(self, grok_epoch: int, pre_circuits: List[str],
                                         post_circuits: List[str]) -> Dict[str, Any]:
        """Analyze how circuits transform during grokking"""
        transformations = {
            'disappeared_circuits': set(pre_circuits) - set(post_circuits),
            'new_circuits': set(post_circuits) - set(pre_circuits),
            'persistent_circuits': set(pre_circuits) & set(post_circuits),
            'transformation_ratio': len(set(post_circuits) - set(pre_circuits)) / max(len(pre_circuits), 1)
        }

        return {k: list(v) if isinstance(v, set) else v for k, v in transformations.items()}

    # ============================================================================
    # TEMPORAL PATTERN RECOGNITION
    # ============================================================================

    def detect_temporal_patterns(self) -> Dict[str, Any]:
        """
        Detect temporal patterns in circuit emergence and evolution
        """
        emergence_order = self.track_emergence_order()

        patterns = {
            'emergence_waves': self._detect_emergence_waves(emergence_order),
            'periodic_emergence': self._detect_periodic_emergence(emergence_order),
            'emergence_acceleration': self._detect_emergence_acceleration(emergence_order),
            'circuit_lifetime_patterns': self._analyze_circuit_lifetimes()
        }

        self.temporal_patterns = patterns

        print("📈 Temporal patterns detected:")
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and 'count' in pattern_data:
                print(f"   {pattern_type}: {pattern_data['count']} instances")
            elif isinstance(pattern_data, list):
                print(f"   {pattern_type}: {len(pattern_data)} instances")

        return patterns

    def _detect_emergence_waves(self, emergence_order: List[Tuple[str, int]]) -> Dict[str, Any]:
        """Detect waves of circuit emergence"""
        if len(emergence_order) < 5:
            return {'count': 0, 'waves': []}

        epochs = [epoch for _, epoch in emergence_order]

        # Find clusters of emergence
        waves = []
        current_wave = []
        wave_threshold = 10  # epochs

        for i, epoch in enumerate(epochs):
            if not current_wave or epoch - current_wave[-1][1] <= wave_threshold:
                current_wave.append(emergence_order[i])
            else:
                if len(current_wave) >= 3:  # Minimum wave size
                    waves.append(current_wave)
                current_wave = [emergence_order[i]]

        # Add final wave if significant
        if len(current_wave) >= 3:
            waves.append(current_wave)

        return {
            'count': len(waves),
            'waves': waves,
            'avg_wave_size': np.mean([len(wave) for wave in waves]) if waves else 0
        }

    def _detect_periodic_emergence(self, emergence_order: List[Tuple[str, int]]) -> Dict[str, Any]:
        """Detect periodic patterns in circuit emergence"""
        if len(emergence_order) < 10:
            return {'periodic': False, 'period': None}

        epochs = [epoch for _, epoch in emergence_order]
        gaps = [epochs[i + 1] - epochs[i] for i in range(len(epochs) - 1)]

        # Look for recurring gap patterns
        gap_counts = {}
        for gap in gaps:
            gap_range = (gap // 5) * 5  # Group into 5-epoch ranges
            gap_counts[gap_range] = gap_counts.get(gap_range, 0) + 1

        # Find most common gap
        if gap_counts:
            most_common_gap = max(gap_counts.items(), key=lambda x: x[1])
            periodicity = most_common_gap[1] / len(gaps)

            return {
                'periodic': periodicity > 0.3,
                'period': most_common_gap[0],
                'periodicity_strength': periodicity
            }

        return {'periodic': False, 'period': None}

    def _detect_emergence_acceleration(self, emergence_order: List[Tuple[str, int]]) -> Dict[str, Any]:
        """Detect acceleration/deceleration in circuit emergence"""
        if len(emergence_order) < 6:
            return {'acceleration': 'unknown', 'trend': None}

        epochs = [epoch for _, epoch in emergence_order]

        # Calculate emergence rate over time windows
        window_size = len(epochs) // 3
        early_rate = window_size / (epochs[window_size - 1] - epochs[0]) if epochs[window_size - 1] > epochs[0] else 0
        late_rate = window_size / (epochs[-1] - epochs[-window_size]) if epochs[-1] > epochs[-window_size] else 0

        acceleration_ratio = late_rate / early_rate if early_rate > 0 else float('inf')

        if acceleration_ratio > 1.5:
            trend = 'accelerating'
        elif acceleration_ratio < 0.67:
            trend = 'decelerating'
        else:
            trend = 'stable'

        return {
            'acceleration': trend,
            'trend': acceleration_ratio,
            'early_rate': early_rate,
            'late_rate': late_rate
        }

    def _analyze_circuit_lifetimes(self) -> Dict[str, Any]:
        """Analyze patterns in circuit lifetimes"""
        lifetimes = []

        for circuit_id in self.emergence_epochs:
            if circuit_id in self.circuit_metadata:
                metadata = self.circuit_metadata[circuit_id]
                if hasattr(metadata, 'first_detected') and hasattr(metadata, 'last_seen'):
                    lifetime = metadata.last_seen - metadata.first_detected
                    lifetimes.append(lifetime)

        if not lifetimes:
            return {'avg_lifetime': 0, 'patterns': 'insufficient_data'}

        return {
            'avg_lifetime': np.mean(lifetimes),
            'std_lifetime': np.std(lifetimes),
            'max_lifetime': max(lifetimes),
            'lifetime_distribution': 'short' if np.mean(lifetimes) < 50 else 'long'
        }

    # ============================================================================
    # INTEGRATED REPORTING
    # ============================================================================

    def generate_temporal_emergence_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report combining evolution tracking with emergence analysis
        """
        print("\n📊 Generating Temporal Emergence Report")
        print("=" * 50)

        # Get base evolution analysis from parent class
        base_report = self.generate_comprehensive_report()

        # Add temporal emergence analysis
        emergence_report = {
            'base_evolution_analysis': base_report,
            'temporal_emergence_analysis': {
                'emergence_cascades': {
                    cascade_id: {
                        'trigger_circuit': cascade.trigger_circuit,
                        'trigger_epoch': cascade.trigger_epoch,
                        'enabled_count': len(cascade.enabled_circuits),
                        'cascade_rate': cascade.get_cascade_rate(),
                        'cascade_type': cascade.cascade_type
                    }
                    for cascade_id, cascade in self.emergence_cascades.items()
                },
                'dependency_chains': {
                    chain_id: {
                        'circuit_count': len(chain.circuits),
                        'formation_rate': chain.get_formation_rate(),
                        'chain_type': chain.chain_type,
                        'chain_strength': chain.chain_strength
                    }
                    for chain_id, chain in self.dependency_chains.items()
                },
                'temporal_patterns': self.temporal_patterns,
                'grokking_transitions': self.grokking_transitions
            },
            'integrated_insights': {
                'total_cascades': len(self.emergence_cascades),
                'total_dependency_chains': len(self.dependency_chains),
                'temporal_complexity': self._calculate_temporal_complexity(),
                'emergence_efficiency': self._calculate_emergence_efficiency()
            }
        }

        # Save report
        if self.storage_dir:
            report_path = self.storage_dir / "temporal_emergence_report.json"
            import json
            with open(report_path, 'w') as f:
                json.dump(emergence_report, f, indent=2, default=str)
            print(f"✅ Report saved to {report_path}")

        return emergence_report

    def _calculate_temporal_complexity(self) -> float:
        """Calculate a metric for temporal complexity of circuit emergence"""
        cascade_complexity = len(self.emergence_cascades) * 0.3
        chain_complexity = len(self.dependency_chains) * 0.4
        pattern_complexity = len(self.temporal_patterns) * 0.3

        return cascade_complexity + chain_complexity + pattern_complexity

    def _calculate_emergence_efficiency(self) -> float:
        """Calculate emergence efficiency (circuits emerged / total epochs)"""
        if not self.emergence_epochs:
            return 0.0

        total_circuits = len(self.emergence_epochs)
        epoch_span = max(self.emergence_epochs.values()) - min(self.emergence_epochs.values())

        return total_circuits / max(epoch_span, 1)


# ============================================================================
# FACTORY FUNCTION FOR EASY INTEGRATION
# ============================================================================

def create_temporal_emergence_analyzer(evolution_tracker=None, enhanced_registry=None,
                                       storage_dir: Path = None) -> TemporalCircuitEmergenceAnalyzer:
    """
    Factory function to create integrated temporal emergence analyzer

    Combines:
    - All CircuitEvolutionAnalyzer functionality
    - Specialized emergence pattern detection
    - Temporal dynamics analysis
    - Grokking transition analysis
    """
    analyzer = TemporalCircuitEmergenceAnalyzer(
        evolution_tracker=evolution_tracker,
        enhanced_registry=enhanced_registry,
        storage_dir=storage_dir
    )

    print("🚀 Temporal Circuit Emergence Analyzer created")
    print("   ✅ Core evolution tracking integrated")
    print("   ✅ Emergence cascade detection ready")
    print("   ✅ Dependency chain analysis ready")
    print("   ✅ Temporal pattern recognition ready")
    print("   ✅ Grokking transition analysis ready")

    return analyzer


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_integrated_analysis():
    """Example showing how to use the integrated analyzer"""

    # Create analyzer (would use your existing tracker and registry)
    analyzer = create_temporal_emergence_analyzer(
        evolution_tracker=None,  # Your CircuitEvolutionTracker
        enhanced_registry=None,  # Your EnhancedCircuitRegistry
        storage_dir=Path("temporal_emergence_results")
    )

    # Run emergence analysis
    print("\n🔬 Running Integrated Analysis")

    # 1. Track emergence order
    emergence_order = analyzer.track_emergence_order()

    # 2. Detect emergence cascades
    cascades = analyzer.detect_emergence_cascades(cascade_window=15)

    # 3. Analyze dependency chains
    chains = analyzer.analyze_dependency_chains()

    # 4. Detect temporal patterns
    patterns = analyzer.detect_temporal_patterns()

    # 5. Analyze grokking transitions (if you have accuracy data)
    # accuracy_history = [...]  # Your training accuracy
    # grokking_analysis = analyzer.analyze_grokking_transitions(accuracy_history)

    # 6. Generate comprehensive report
    report = analyzer.generate_temporal_emergence_report()

    print("✅ Integrated analysis complete!")
    return analyzer, report


if __name__ == "__main__":
    # Run example
    analyzer, report = example_integrated_analysis()