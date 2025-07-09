# analysis/temporal/circuit_emergence_analyzer.py
"""
Phase 1.3: Enhanced Temporal Analysis Framework

Implements circuit emergence analysis and dependency tracking for studying
how circuits form, interact, and build upon each other during learning.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

import numpy as np

# Import the enhanced schema from phase 1.2
from analysis.core.circuit_schema import (
    CircuitInteractionType, EmergencePattern, InteractionEvent
)
from analysis.temporal.temporal_circuit_emergence_analyzer import EmergenceCascade, DependencyChain


class CircuitEmergenceAnalyzer:
    """Analyzes temporal dynamics of circuit emergence and interactions"""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Analysis results storage
        self.emergence_patterns: Dict[str, EmergencePattern] = {}
        self.emergence_cascades: Dict[str, EmergenceCascade] = {}
        self.dependency_chains: Dict[str, DependencyChain] = {}
        self.grokking_transitions: Dict[str, Dict[str, Any]] = {}

        # Analysis caches
        self._emergence_order_cache: Optional[List[Tuple[str, int]]] = None
        self._dependency_graph_cache: Optional[Dict[str, Set[str]]] = None

    def track_emergence_order(self, canonical_registry) -> List[Tuple[str, int]]:
        """Analyze which circuits enable others and build temporal dependency graphs"""
        if self._emergence_order_cache is not None:
            return self._emergence_order_cache

        emergence_events = []

        # Extract emergence events from canonical registry
        for circuit_id, canonical_circuit in canonical_registry.canonical_circuits.items():
            first_epoch = canonical_circuit.first_seen
            emergence_events.append((circuit_id, first_epoch))

        # Sort by emergence epoch
        emergence_events.sort(key=lambda x: x[1])

        # Analyze emergence patterns
        for i, (circuit_id, epoch) in enumerate(emergence_events):
            pattern = self._classify_emergence_pattern(circuit_id, epoch, emergence_events, i)
            self.emergence_patterns[circuit_id] = pattern

        self._emergence_order_cache = emergence_events
        return emergence_events

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

    def identify_emergence_cascades(self, canonical_registry,
                                    interaction_events: List[InteractionEvent]) -> Dict[str, EmergenceCascade]:
        """Identify cascades where one circuit triggers formation of others"""
        cascades = {}
        emergence_order = self.track_emergence_order(canonical_registry)

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

    def _analyze_potential_cascade(self, trigger_circuit: str, trigger_epoch: int,
                                   window_circuits: List[Tuple[str, int]],
                                   interaction_events: List[InteractionEvent]) -> Optional[EmergenceCascade]:
        """Analyze if a circuit triggers a cascade"""
        enabled_circuits = []

        # Find circuits that emerged after trigger
        for circuit_id, epoch in window_circuits:
            if epoch > trigger_epoch and circuit_id != trigger_circuit:
                # Check for enabling relationships in interaction events
                enabling_strength = 0.0
                for event in interaction_events:
                    if (event.source_circuit == trigger_circuit and
                            event.target_circuit == circuit_id and
                            event.event_type == CircuitInteractionType.ENABLES):
                        enabling_strength = max(enabling_strength, event.interaction_strength)

                if enabling_strength > 0.3:  # Threshold for significant enabling
                    enabled_circuits.append((circuit_id, epoch))

        if len(enabled_circuits) < 1:
            return None

        # Calculate cascade properties
        cascade_id = f"cascade_{trigger_circuit}_{trigger_epoch}"
        temporal_span = max(epoch for _, epoch in enabled_circuits) - trigger_epoch
        cascade_strength = np.mean([0.5] + [0.8 for _ in enabled_circuits])  # Simplified

        return EmergenceCascade(
            cascade_id=cascade_id,
            trigger_circuit=trigger_circuit,
            trigger_epoch=trigger_epoch,
            enabled_circuits=enabled_circuits,
            cascade_strength=cascade_strength,
            temporal_span=temporal_span,
            cascade_type="linear"
        )

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
        emergence_order = self.track_emergence_order(canonical_registry)
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

    def analyze_dependency_chains(self, dependency_graph: Dict[str, Set[str]]) -> Dict[str, DependencyChain]:
        """Identify linear dependency chains"""
        chains = {}
        visited = set()

        for start_circuit in dependency_graph:
            if start_circuit in visited:
                continue

            # Follow chain from this starting point
            chain = self._trace_dependency_chain(start_circuit, dependency_graph, visited)

            if len(chain) > 2:  # Only keep meaningful chains
                chain_id = f"chain_{start_circuit}_{len(chain)}"

                # Get formation epochs for chain circuits
                emergence_order = self._emergence_order_cache or []
                circuit_epochs = {circuit_id: epoch for circuit_id, epoch in emergence_order}
                formation_epochs = [circuit_epochs.get(cid, 0) for cid in chain]

                dependency_chain = DependencyChain(
                    chain_id=chain_id,
                    circuits=chain,
                    formation_epochs=formation_epochs,
                    chain_strength=self._calculate_chain_strength(chain, dependency_graph),
                    chain_type="sequential"
                )

                chains[chain_id] = dependency_chain

        self.dependency_chains = chains
        return chains

    def _trace_dependency_chain(self, start: str, dependency_graph: Dict[str, Set[str]],
                                visited: set) -> List[str]:
        """Trace a linear dependency chain starting from a circuit"""
        chain = [start]
        visited.add(start)
        current = start

        while current in dependency_graph:
            # Find next circuit in chain (prefer single dependency)
            dependents = [dep for dep in dependency_graph[current] if dep not in visited]

            if len(dependents) == 1:
                next_circuit = dependents[0]
                chain.append(next_circuit)
                visited.add(next_circuit)
                current = next_circuit
            else:
                break  # Multiple dependencies or end of chain

        return chain

    def _calculate_chain_strength(self, chain: List[str],
                                  dependency_graph: Dict[str, Set[str]]) -> float:
        """Calculate overall strength of dependency chain"""
        if len(chain) < 2:
            return 0.0

        strengths = []
        for i in range(len(chain) - 1):
            source = chain[i]
            target = chain[i + 1]
            if source in dependency_graph and target in dependency_graph[source]:
                strengths.append(1.0)  # Simplified strength calculation
            else:
                strengths.append(0.0)

        return np.mean(strengths) if strengths else 0.0

    def generate_temporal_report(self) -> Dict[str, Any]:
        """Generate comprehensive temporal analysis report"""
        return {
            'emergence_patterns': {
                pattern.value: [cid for cid, p in self.emergence_patterns.items() if p == pattern]
                for pattern in EmergencePattern
            },
            'emergence_cascades': {
                cid: {
                    'trigger_circuit': cascade.trigger_circuit,
                    'trigger_epoch': cascade.trigger_epoch,
                    'enabled_count': len(cascade.enabled_circuits),
                    'cascade_rate': cascade.get_cascade_rate(),
                    'temporal_span': cascade.temporal_span
                }
                for cid, cascade in self.emergence_cascades.items()
            },
            'dependency_chains': {
                cid: {
                    'length': len(chain.circuits),
                    'formation_rate': chain.get_formation_rate(),
                    'chain_strength': chain.chain_strength,
                    'circuits': chain.circuits
                }
                for cid, chain in self.dependency_chains.items()
            },
            'grokking_transitions': self.grokking_transitions,
            'summary': {
                'total_circuits_analyzed': len(self.emergence_patterns),
                'cascades_detected': len(self.emergence_cascades),
                'dependency_chains': len(self.dependency_chains),
                'grokking_transitions_detected': len([
                    t for t in self.grokking_transitions.values()
                    if t.get('transition_detected', False)
                ])
            }
        }

    def save_analysis_results(self):
        """Save temporal analysis results"""
        report = self.generate_temporal_report()

        results_file = self.storage_dir / "temporal_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Temporal analysis results saved to {results_file}")
        print(f"   📊 {report['summary']['total_circuits_analyzed']} circuits analyzed")
        print(f"   📊 {report['summary']['cascades_detected']} emergence cascades detected")
        print(f"   📊 {report['summary']['dependency_chains']} dependency chains identified")
        print(f"   📊 {report['summary']['grokking_transitions_detected']} grokking transitions found")


class DependencyTracker:
    """Specialized tracker for circuit dependencies and prerequisite relationships"""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.prerequisite_map: Dict[str, Set[str]] = defaultdict(set)
        self.dependency_strengths: Dict[Tuple[str, str], float] = {}
        self.temporal_dependencies: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    def track_prerequisite_formation(self, circuit_id: str, epoch: int,
                                     existing_circuits: List[str],
                                     interaction_events: List[InteractionEvent]):
        """Track when prerequisites are established for a circuit"""
        circuit_prerequisites = set()

        # Analyze interaction events to find prerequisites
        for event in interaction_events:
            if (event.target_circuit == circuit_id and
                    event.event_type in [CircuitInteractionType.PREREQUISITE,
                                         CircuitInteractionType.ENABLES]):
                circuit_prerequisites.add(event.source_circuit)
                self.dependency_strengths[(event.source_circuit, circuit_id)] = event.interaction_strength

        # Add temporal prerequisites (circuits that must exist before this one)
        for existing_circuit in existing_circuits:
            # Check if temporal relationship suggests prerequisite
            temporal_strength = self._calculate_temporal_prerequisite_strength(
                existing_circuit, circuit_id, epoch)
            if temporal_strength > 0.4:
                circuit_prerequisites.add(existing_circuit)
                self.dependency_strengths[(existing_circuit, circuit_id)] = temporal_strength

        self.prerequisite_map[circuit_id] = circuit_prerequisites

        # Track temporal evolution
        for prereq in circuit_prerequisites:
            self.temporal_dependencies[circuit_id].append((epoch, prereq))

    def _calculate_temporal_prerequisite_strength(self, existing_circuit: str,
                                                  new_circuit: str, epoch: int) -> float:
        """Calculate likelihood that existing circuit is prerequisite for new one"""
        # Simplified heuristic - in real implementation, this would use:
        # - Circuit type compatibility
        # - Functional relationship analysis
        # - Pattern similarity
        # - Temporal proximity

        # For now, use a simple temporal decay model
        base_strength = 0.5
        temporal_decay = 0.05  # Strength decreases with time gap

        # This would need access to emergence epochs of existing circuits
        # temporal_gap = epoch - existing_circuit_epoch
        # strength = base_strength * np.exp(-temporal_decay * temporal_gap)

        # Simplified version
        return base_strength * 0.8  # Placeholder

    def analyze_prerequisite_chains(self) -> Dict[str, List[str]]:
        """Analyze chains of prerequisites"""
        chains = {}

        for circuit_id in self.prerequisite_map:
            chain = self._trace_prerequisite_chain(circuit_id, set())
            if len(chain) > 1:
                chains[circuit_id] = chain

        return chains

    def _trace_prerequisite_chain(self, circuit_id: str, visited: set) -> List[str]:
        """Trace chain of prerequisites for a circuit"""
        if circuit_id in visited:
            return []  # Avoid cycles

        visited.add(circuit_id)
        chain = [circuit_id]

        # Add prerequisites
        for prereq in self.prerequisite_map.get(circuit_id, []):
            prereq_chain = self._trace_prerequisite_chain(prereq, visited.copy())
            chain.extend(prereq_chain)

        return chain

    def get_dependency_statistics(self) -> Dict[str, Any]:
        """Get statistics about dependency relationships"""
        all_circuits = set(self.prerequisite_map.keys())
        for prereqs in self.prerequisite_map.values():
            all_circuits.update(prereqs)

        return {
            'total_circuits': len(all_circuits),
            'circuits_with_prerequisites': len(self.prerequisite_map),
            'average_prerequisites': np.mean(
                [len(prereqs) for prereqs in self.prerequisite_map.values()]) if self.prerequisite_map else 0,
            'max_prerequisites': max(
                [len(prereqs) for prereqs in self.prerequisite_map.values()]) if self.prerequisite_map else 0,
            'total_dependencies': sum(len(prereqs) for prereqs in self.prerequisite_map.values()),
            'dependency_density': len(self.dependency_strengths) / (len(all_circuits) ** 2) if all_circuits else 0
        }


# Integration function for easy usage
def create_temporal_analysis_system(storage_dir: Path) -> Tuple[CircuitEmergenceAnalyzer, DependencyTracker]:
    """Create complete temporal analysis system"""
    emergence_analyzer = CircuitEmergenceAnalyzer(storage_dir / "emergence")
    dependency_tracker = DependencyTracker(storage_dir / "dependencies")

    print("🔧 Enhanced temporal analysis framework initialized")
    print("   ✅ Circuit emergence analyzer")
    print("   ✅ Dependency tracker")
    print("   ✅ Grokking transition analysis")
    print("   ✅ Emergence cascade detection")
    print("   ✅ Dependency chain analysis")

    return emergence_analyzer, dependency_tracker
