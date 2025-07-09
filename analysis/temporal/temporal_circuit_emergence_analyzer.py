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

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np

# Import the fixed core analyzer
from analysis.core.circuit_evolution_analyzer import CircuitEvolutionAnalyzer
from analysis.core.circuit_schema import (
    InteractionType, EmergencePattern, InteractionEvent, CircuitInteractionType
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

    def __init__(self, evolution_tracker=None, enhanced_registry=None,
                 logger=None, storage_dir: Path = None):
        """Initialize with all existing functionality plus emergence analysis"""
        # Initialize parent class
        super().__init__(evolution_tracker, enhanced_registry, storage_dir)

        # 🆕 ADD: UnifiedLogger for proper logging
        if logger is not None:
            self.logger = logger
        else:
            from analysis.core.unified_logger import UnifiedLogger
            self.logger = UnifiedLogger("TemporalEmergenceAnalyzer")

        # Original temporal emergence components
        self.emergence_cascades: Dict[str, EmergenceCascade] = {}
        self.dependency_chains: Dict[str, DependencyChain] = {}
        self.temporal_patterns: Dict[str, Any] = {}
        self.grokking_transitions: Dict[str, Dict[str, Any]] = {}

        # 🆕 MOVED: Components from CircuitEmergenceAnalyzer
        self.emergence_patterns: Dict[str, EmergencePattern] = {}
        self.prerequisite_map: Dict[str, Set[str]] = defaultdict(set)
        self.dependency_strengths: Dict[Tuple[str, str], float] = {}

        # Analysis caches for performance
        self._emergence_order_cache: Optional[List[Tuple[str, int]]] = None
        self._dependency_graph_cache: Optional[Dict[str, Set[str]]] = None

        self.logger.info("Enhanced Temporal Circuit Emergence Analyzer initialized")
        self.logger.info("✅ Core evolution analysis available")
        self.logger.info("✅ Specialized emergence analysis added")
        self.logger.info("✅ Dependency tracking integrated")

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

        self.logger.info(f"📊 Emergence order tracked: {len(emergence_order)} circuits")
        return emergence_order

    def identify_emergence_cascades(self, canonical_registry,
                                    interaction_events: List[InteractionEvent]) -> Dict[str, EmergenceCascade]:
        """Identify cascades where one circuit triggers formation of others"""
        cascades = {}
        emergence_order = self.track_emergence_order()

        # Group circuits by emergence epoch windows
        epoch_windows = defaultdict(list)
        window_size = 5  # Epochs

        for circuit_id, epoch in emergence_order:
            window = epoch // window_size
            epoch_windows[window].append((circuit_id, epoch))

        # Analyze each window for cascade patterns
        for window, circuits in epoch_windows.items():
            if len(circuits) < 2:
                continue

            # Sort by epoch within window
            circuits.sort(key=lambda x: x[1])

            # Find potential trigger circuits (earliest in window)
            trigger_candidates = circuits[:len(circuits) // 3 + 1]

            for trigger_circuit, trigger_epoch in trigger_candidates:
                cascade = self._analyze_potential_cascade(trigger_circuit, trigger_epoch,
                                                          circuits, interaction_events)
                if cascade and len(cascade.enabled_circuits) > 0:
                    cascades[cascade.cascade_id] = cascade

        self.emergence_cascades = cascades
        return cascades

    def _classify_emergence_pattern(self, circuit_id: str, epoch: int,
                                    all_events: List[Tuple[str, int]], index: int) -> EmergencePattern:
        """Classify the emergence pattern for a circuit"""
        # Look at temporal context
        window_size = 10
        start_idx = max(0, index - window_size)
        end_idx = min(len(all_events), index + window_size + 1)

        nearby_events = all_events[start_idx:end_idx]
        nearby_epochs = [e[1] for e in nearby_events]

        if len(nearby_epochs) < 3:
            return EmergencePattern.SUDDEN

        # Calculate local emergence density
        epoch_range = max(nearby_epochs) - min(nearby_epochs)
        if epoch_range == 0:
            return EmergencePattern.SUDDEN

        density = len(nearby_events) / epoch_range

        # Classify based on density and position
        if density > 0.5:
            return EmergencePattern.CASCADING
        elif index < len(all_events) * 0.3:
            return EmergencePattern.GRADUAL
        elif index > len(all_events) * 0.7:
            return EmergencePattern.REINFORCING
        else:
            return EmergencePattern.SUDDEN

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

        self.logger.info(f"🔗 Dependency chains analyzed: {len(chains)}")
        for chain_id, chain in chains.items():
            self.logger.info(f"   {chain_id}: {len(chain.circuits)} circuits, "
                  f"rate: {chain.get_formation_rate():.3f}")

        return chains

    def build_dependency_graph(self, canonical_registry,
                               interaction_events: List[InteractionEvent]) -> Dict[str, Set[str]]:
        """Build complete dependency graph from interaction events"""
        if self._dependency_graph_cache is not None:
            return self._dependency_graph_cache

        dependency_graph = defaultdict(set)

        # Add dependencies from interaction events
        for event in interaction_events:
            if event.event_type == CircuitInteractionType.PREREQUISITE:
                dependency_graph[event.source_circuit].add(event.target_circuit)
            elif event.event_type == CircuitInteractionType.ENABLES:
                dependency_graph[event.source_circuit].add(event.target_circuit)

        # Add temporal dependencies (earlier circuits may enable later ones)
        emergence_order = self.track_emergence_order()
        circuit_epochs = {circuit_id: epoch for circuit_id, epoch in emergence_order}

        for circuit_id, epoch in emergence_order:
            # Find earlier circuits that might be prerequisites
            for other_id, other_epoch in emergence_order:
                if other_epoch < epoch and other_epoch >= epoch - 10:  # Within reasonable window
                    # Check for interaction evidence
                    interaction_strength = self._calculate_interaction_strength(
                        other_id, circuit_id, interaction_events)
                    if interaction_strength > 0.3:
                        dependency_graph[other_id].add(circuit_id)

        self._dependency_graph_cache = dict(dependency_graph)
        return self._dependency_graph_cache

    def _calculate_interaction_strength(self, source: str, target: str,
                                        interaction_events: List[InteractionEvent]) -> float:
        """Calculate interaction strength between two circuits"""
        total_strength = 0.0
        count = 0

        for event in interaction_events:
            if event.source_circuit == source and event.target_circuit == target:
                total_strength += event.interaction_strength
                count += 1

        return total_strength / count if count > 0 else 0.0


    def calculate_dependency_strength(self, source_circuit: str, target_circuit: str) -> float:
        """
        🆕 MOVED: Calculate strength of dependency relationship
        Originally from CircuitEmergenceAnalyzer
        """
        base_strength = self.dependency_strengths.get((source_circuit, target_circuit), 0.0)

        # TODO: Add more sophisticated strength calculation
        # Could include temporal proximity, functional similarity, etc.

        return base_strength * 0.8  # Placeholder adjustment

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

    def analyze_grokking_transitions(self, circuit_history: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze circuit behavior during grokking transitions"""
        grokking_analysis = {}

        for circuit_id, history in circuit_history.items():
            if not history:
                continue

            # Extract strength trajectory
            epochs = [item.get('epoch', 0) for item in history]
            strengths = [item.get('strength', 0.0) for item in history]

            if len(epochs) < 10:  # Need sufficient data
                continue

            # Detect grokking transition
            transition_info = self._detect_grokking_transition(epochs, strengths)

            # Analyze pre/post grokking behavior
            pre_grok_analysis = self._analyze_pre_grokking(epochs, strengths, transition_info)
            post_grok_analysis = self._analyze_post_grokking(epochs, strengths, transition_info)

            grokking_analysis[circuit_id] = {
                'transition_detected': transition_info['detected'],
                'transition_epoch': transition_info.get('epoch'),
                'transition_strength': transition_info.get('strength_change'),
                'pre_grokking': pre_grok_analysis,
                'post_grokking': post_grok_analysis,
                'grokking_acceleration': transition_info.get('acceleration', 0.0)
            }

        self.grokking_transitions = grokking_analysis
        return grokking_analysis

    def _detect_grokking_transition(self, epochs: List[int], strengths: List[float]) -> Dict[str, Any]:
        """Detect grokking transition point using strength trajectory analysis"""
        if len(strengths) < 10:
            return {'detected': False}

        # Calculate derivatives to find acceleration points
        derivatives = np.gradient(strengths)
        second_derivatives = np.gradient(derivatives)

        # Find points of maximum acceleration (grokking onset)
        acceleration_threshold = np.std(second_derivatives) * 2
        significant_accelerations = np.where(second_derivatives > acceleration_threshold)[0]

        if len(significant_accelerations) == 0:
            return {'detected': False}

        # Find the most significant acceleration point
        max_acceleration_idx = significant_accelerations[np.argmax(second_derivatives[significant_accelerations])]

        # Validate transition by checking strength change
        pre_strength = np.mean(strengths[:max_acceleration_idx])
        post_strength = np.mean(strengths[max_acceleration_idx:])
        strength_change = post_strength - pre_strength

        if strength_change > 0.1:  # Significant positive change
            return {
                'detected': True,
                'epoch': epochs[max_acceleration_idx],
                'index': max_acceleration_idx,
                'strength_change': strength_change,
                'acceleration': second_derivatives[max_acceleration_idx]
            }

        return {'detected': False}

    def _analyze_pre_grokking(self, epochs: List[int], strengths: List[float],
                              transition_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze circuit behavior before grokking"""
        if not transition_info.get('detected'):
            return {'phase': 'unknown'}

        transition_idx = transition_info['index']
        pre_strengths = strengths[:transition_idx]

        if len(pre_strengths) < 3:
            return {'phase': 'insufficient_data'}

        return {
            'phase': 'memorization',
            'average_strength': np.mean(pre_strengths),
            'strength_stability': np.std(pre_strengths),
            'trend': np.polyfit(range(len(pre_strengths)), pre_strengths, 1)[0],
            'duration': transition_idx
        }

    def _analyze_post_grokking(self, epochs: List[int], strengths: List[float],
                               transition_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze circuit behavior after grokking"""
        if not transition_info.get('detected'):
            return {'phase': 'unknown'}

        transition_idx = transition_info['index']
        post_strengths = strengths[transition_idx:]

        if len(post_strengths) < 3:
            return {'phase': 'insufficient_data'}

        return {
            'phase': 'generalization',
            'average_strength': np.mean(post_strengths),
            'strength_stability': np.std(post_strengths),
            'trend': np.polyfit(range(len(post_strengths)), post_strengths, 1)[0],
            'duration': len(post_strengths)
        }

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

        self.logger.info("📈 Temporal patterns detected:")
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and 'count' in pattern_data:
                self.logger.info(f"   {pattern_type}: {pattern_data['count']} instances")
            elif isinstance(pattern_data, list):
                self.logger.info(f"   {pattern_type}: {len(pattern_data)} instances")

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

    def analyze_prerequisite_chains(self) -> Dict[str, List[str]]:
        """🆕 NEW: Analyze prerequisite relationships between circuits"""
        self.logger.info("Analyzing prerequisite chains")

        chains = {}

        for circuit_id in self.prerequisite_map:
            chain = self._trace_prerequisite_chain(circuit_id, set())
            if len(chain) > 1:
                chains[circuit_id] = chain
                self.logger.debug(f"  Chain: {circuit_id} ← {len(chain) - 1} prerequisites")

        self.logger.info(f"🔗 Found {len(chains)} prerequisite chains")
        return chains

    def _trace_prerequisite_chain(self, circuit_id: str, visited: set) -> List[str]:
        """🆕 NEW: Trace prerequisite chain with cycle detection"""
        if circuit_id in visited:
            return []  # Avoid cycles

        visited.add(circuit_id)
        chain = [circuit_id]

        for prereq in self.prerequisite_map.get(circuit_id, set()):
            prereq_chain = self._trace_prerequisite_chain(prereq, visited.copy())
            chain.extend(prereq_chain)

        return chain

    def record_prerequisite_relationship(self, dependent_circuit: str, prerequisite_circuit: str,
                                         strength: float = 1.0):
        """🆕 NEW: Record prerequisite relationship"""
        self.prerequisite_map[dependent_circuit].add(prerequisite_circuit)
        self.dependency_strengths[(prerequisite_circuit, dependent_circuit)] = strength

        self.logger.debug(f"Recorded: {prerequisite_circuit} → {dependent_circuit} (strength: {strength:.3f})")

    def get_dependency_statistics(self) -> Dict[str, Any]:
        """🆕 NEW: Get quantitative dependency metrics"""
        all_circuits = set(self.prerequisite_map.keys())
        for prereqs in self.prerequisite_map.values():
            all_circuits.update(prereqs)

        stats = {
            'total_circuits': len(all_circuits),
            'circuits_with_prerequisites': len(self.prerequisite_map),
            'average_prerequisites': np.mean(
                [len(prereqs) for prereqs in self.prerequisite_map.values()]) if self.prerequisite_map else 0,
            'max_prerequisites': max(
                [len(prereqs) for prereqs in self.prerequisite_map.values()]) if self.prerequisite_map else 0,
            'total_dependencies': sum(len(prereqs) for prereqs in self.prerequisite_map.values()),
            'dependency_density': len(self.dependency_strengths) / (len(all_circuits) ** 2) if all_circuits else 0
        }

        self.logger.info("📊 Dependency Statistics:")
        self.logger.info(f"   Total circuits: {stats['total_circuits']}")
        self.logger.info(f"   With prerequisites: {stats['circuits_with_prerequisites']}")
        self.logger.info(f"   Avg prerequisites: {stats['average_prerequisites']:.2f}")

        return stats

    # ============================================================================
    # INTEGRATED REPORTING
    # ============================================================================

    def generate_temporal_emergence_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report combining evolution tracking with emergence analysis
        """
        self.logger.info("\n📊 Generating Temporal Emergence Report")
        self.logger.info("=" * 50)

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
            self.logger.info(f"✅ Report saved to {report_path}")

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

    def get_dependency_statistics(self) -> Dict[str, Any]:
        """
        🆕 MOVED: Get statistics about dependency relationships
        Originally from CircuitEmergenceAnalyzer
        """
        all_circuits = set(self.prerequisite_map.keys())
        for prereqs in self.prerequisite_map.values():
            all_circuits.update(prereqs)

        stats = {
            'total_circuits': len(all_circuits),
            'circuits_with_prerequisites': len(self.prerequisite_map),
            'average_prerequisites': np.mean(
                [len(prereqs) for prereqs in self.prerequisite_map.values()]) if self.prerequisite_map else 0,
            'max_prerequisites': max(
                [len(prereqs) for prereqs in self.prerequisite_map.values()]) if self.prerequisite_map else 0,
            'total_dependencies': sum(len(prereqs) for prereqs in self.prerequisite_map.values()),
            'dependency_density': len(self.dependency_strengths) / (len(all_circuits) ** 2) if all_circuits else 0
        }

        self.logger.info("📊 Dependency Statistics:")
        self.logger.info(f"   Total circuits: {stats['total_circuits']}")
        self.logger.info(f"   Circuits with prerequisites: {stats['circuits_with_prerequisites']}")
        self.logger.info(f"   Average prerequisites: {stats['average_prerequisites']:.2f}")
        self.logger.info(f"   Dependency density: {stats['dependency_density']:.3f}")

        return stats


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
