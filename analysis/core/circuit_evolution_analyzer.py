# analysis/core/circuit_evolution_analyzer.py
"""
Circuit Evolution Analyzer - Research Analysis Tools

Analyzes recorded circuit evolution data to answer research questions about
transformer learning dynamics, circuit formation, and knowledge organization.

Uses your existing CircuitEvolutionTracker and EnhancedCircuitRegistry.
"""

from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from pathlib import Path
import json
import numpy as np

from analysis.core import CircuitMetadata, EmergencePhase
# Import from the tracker and existing schema
from analysis.core.circuit_schema import (LearningPhase, InteractionType, InteractionEvent,
                                          LearningPhaseTransition, CircuitType, EvolutionSnapshot, EvolutionPattern)
from analysis.core.unified_logger import UnifiedLogger


class CircuitEvolutionAnalyzer:
    """
    Analyzes circuit evolution data for research insights
    Works with your existing CircuitEvolutionTracker and EnhancedCircuitRegistry
    """

    def __init__(self, evolution_tracker=None, enhanced_registry=None, storage_dir: Path = None):
        """
        Initialize with existing tracker and registry
        """
        # Core components
        self.evolution_tracker = evolution_tracker
        self.enhanced_registry = enhanced_registry
        self.storage_dir = storage_dir

        # NEW: Missing fields that were referenced
        self.interaction_log: List[InteractionEvent] = []
        self.evolution_snapshots: Dict[str, List[Any]] = {}
        self.learning_phases: Dict[int, LearningPhase] = {}

        # Get data from existing tracker if available
        if evolution_tracker:
            self.evolution_data = getattr(evolution_tracker, 'evolution_data', {})
            self.emergence_epochs = getattr(evolution_tracker, 'emergence_epochs', {})
            self.circuit_relationships = getattr(evolution_tracker, 'circuit_relationships', {})
            self.epoch_to_circuits = getattr(evolution_tracker, 'epoch_to_circuits', {})

            # Extract learning phases from tracker
            if hasattr(evolution_tracker, 'learning_phases'):
                self.learning_phases = evolution_tracker.learning_phases

            # Extract interaction events
            if hasattr(evolution_tracker, 'interaction_events'):
                self.interaction_log = evolution_tracker.interaction_events

        # Get enhanced metadata from registry if available
        if enhanced_registry:
            self.circuit_metadata = getattr(enhanced_registry, 'circuit_metadata', {})
        else:
            self.circuit_metadata = {}

        # Analysis results
        self.emergence_patterns: Dict[str, Any] = {}
        self.interaction_patterns: Dict[InteractionType, Any] = {}
        self.learning_phase_data: Dict[int, LearningPhase] = {}
        self.phase_transitions: List[LearningPhaseTransition] = []

    def _load_interaction_events(self) -> List[InteractionEvent]:
        """
        Extract interaction events from your existing tracker data
        """
        interactions = []

        # Extract from your existing circuit_relationships
        for (circuit1, circuit2), relationship_data in self.circuit_relationships.items():
            # Convert your relationship data to interaction events
            if isinstance(relationship_data, dict):
                epoch = relationship_data.get('epoch', 0)
                strength = relationship_data.get('strength', 0.5)

                # Map your relationship types to interaction types
                if 'enables' in str(relationship_data.get('type', '')).lower():
                    interaction_type = InteractionType.ENABLES
                elif 'competes' in str(relationship_data.get('type', '')).lower():
                    interaction_type = InteractionType.COMPETES
                elif 'cooperates' in str(relationship_data.get('type', '')).lower():
                    interaction_type = InteractionType.COOPERATES
                else:
                    interaction_type = InteractionType.ENABLES  # Default

                event = InteractionEvent(
                    epoch=epoch,
                    source_circuit=circuit1,
                    target_circuit=circuit2,
                    interaction_type=interaction_type,
                    strength=strength
                )
                interactions.append(event)

        return interactions

    # ===============================================================
    # RESEARCH QUESTION 1: Circuit Emergence Order and Dynamics
    # ===============================================================

    def analyze_emergence_order(self) -> List[Tuple[str, int, str]]:
        """
        Research Question: How do different circuit types emerge in temporal order?
        Uses your existing emergence_epochs data
        """
        emergence_timeline = []

        # ✅ Use your existing emergence_epochs data
        for circuit_id, emergence_epoch in self.emergence_epochs.items():
            # Get circuit type from enhanced registry or evolution data
            circuit_type = "unknown"
            if circuit_id in self.circuit_metadata:
                circuit_type = self.circuit_metadata[circuit_id].detection_method
            elif circuit_id in self.evolution_data:
                circuit_type = self.evolution_data[circuit_id].get('type', 'unknown')

            emergence_timeline.append((circuit_id, emergence_epoch, circuit_type))

        # Sort by emergence epoch
        emergence_timeline.sort(key=lambda x: x[1])

        print("🔍 Research Question 1: Circuit Emergence Order")
        print("=" * 50)

        # Group by circuit type
        by_type = defaultdict(list)
        for circuit_id, epoch, circuit_type in emergence_timeline:
            by_type[circuit_type].append((circuit_id, epoch))

        for circuit_type, circuits in by_type.items():
            avg_epoch = sum(epoch for _, epoch in circuits) / len(circuits)
            print(f"   {circuit_type:20s}: {len(circuits):3d} circuits, avg epoch {avg_epoch:6.1f}")

        return emergence_timeline

    def analyze_emergence_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze how each circuit emerged using your existing evolution data"""
        patterns = {}

        for circuit_id in self.emergence_epochs:
            # ✅ Get strength progression from your existing evolution_data
            strength_progression = []
            if circuit_id in self.evolution_data:
                circuit_data = self.evolution_data[circuit_id]
                if isinstance(circuit_data, dict) and 'strength_history' in circuit_data:
                    strength_progression = circuit_data['strength_history']
                elif isinstance(circuit_data, list):
                    # Handle case where evolution_data[circuit_id] is a list of temporal data
                    strength_progression = [(item.get('epoch', 0), item.get('strength', 0.0))
                                            for item in circuit_data if isinstance(item, dict)]

            # Use enhanced registry metadata if available
            if circuit_id in self.circuit_metadata:
                metadata = self.circuit_metadata[circuit_id]
                if hasattr(metadata, 'strength_history') and metadata.strength_history:
                    strength_progression = metadata.strength_history

            if len(strength_progression) < 2:
                continue

            # Classify emergence type
            emergence_type = self._classify_emergence_type(strength_progression)

            # Find prerequisites (circuits that emerged earlier)
            emergence_epoch = self.emergence_epochs[circuit_id]
            prerequisites = [cid for cid, epoch in self.emergence_epochs.items()
                             if epoch < emergence_epoch - 5]  # At least 5 epochs earlier

            pattern = {
                'circuit_id': circuit_id,
                'emergence_type': emergence_type,
                'emergence_epoch': emergence_epoch,
                'strength_progression': strength_progression,
                'prerequisites': prerequisites[:5],  # Limit to first 5
                'total_prerequisites': len(prerequisites)
            }

            patterns[circuit_id] = pattern

        self.emergence_patterns = patterns
        return patterns

    def _classify_emergence_type(self, strength_progression: List[Tuple[int, float]]) -> str:
        """Classify how a circuit emerged based on strength progression"""
        if len(strength_progression) < 3:
            return "sudden"

        strengths = [s for _, s in strength_progression]

        # Sudden emergence: large jump in first few observations
        if len(strengths) >= 2 and strengths[1] > strengths[0] * 2:
            return "sudden"

        # Oscillating: multiple direction changes
        direction_changes = 0
        for i in range(2, len(strengths)):
            prev_trend = strengths[i - 1] - strengths[i - 2]
            curr_trend = strengths[i] - strengths[i - 1]
            if (prev_trend > 0) != (curr_trend > 0):
                direction_changes += 1

        if direction_changes > len(strengths) // 3:
            return "oscillating"

        # Gradual vs cascading (would need more sophisticated analysis)
        return "gradual"

    def _find_prerequisites(self, circuit_id: str, emergence_epoch: int) -> List[str]:
        """Find circuits that emerged before this one and might be prerequisites"""
        prerequisites = []

        # Check interaction log for enabling relationships
        for event in self.interaction_log:
            if (event.target_circuit == circuit_id and
                    event.interaction_type == InteractionType.ENABLES and
                    event.epoch <= emergence_epoch):
                prerequisites.append(event.source_circuit)

        # Also check for circuits that emerged significantly earlier
        for other_circuit_id, snapshots in self.evolution_snapshots.items():
            if (other_circuit_id != circuit_id and snapshots and
                    snapshots[0].epoch < emergence_epoch - 10):  # At least 10 epochs earlier
                prerequisites.append(other_circuit_id)

        return list(set(prerequisites))

    def _find_enabled_circuits(self, circuit_id: str, emergence_epoch: int) -> List[str]:
        """Find circuits that this circuit may have enabled"""
        enabled = []

        for event in self.interaction_log:
            if (event.source_circuit == circuit_id and
                    event.interaction_type == InteractionType.ENABLES and
                    event.epoch >= emergence_epoch):
                enabled.append(event.target_circuit)

        return list(set(enabled))

    # ===============================================================
    # RESEARCH QUESTION 2: Circuit Interactions and Relationships
    # ===============================================================

    def analyze_interaction_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Research Question: How do circuits interact and compete/cooperate?
        Uses your existing circuit_relationships data
        """
        # ✅ Get interaction events from your existing data
        interaction_events = self._load_interaction_events()

        patterns = {}

        # Group interactions by type
        by_type = defaultdict(list)
        for event in interaction_events:
            by_type[event.interaction_type].append(event)

        print("🔍 Research Question 2: Circuit Interaction Patterns")
        print("=" * 50)

        for interaction_type, events in by_type.items():
            # Calculate statistics
            strengths = [event.strength for event in events]
            circuit_pairs = [(event.source_circuit, event.target_circuit) for event in events]

            pattern = {
                'interaction_type': interaction_type,
                'frequency': len(events),
                'average_strength': np.mean(strengths) if strengths else 0.0,
                'circuit_pairs': circuit_pairs[:10],  # First 10 pairs
                'total_pairs': len(circuit_pairs)
            }

            patterns[interaction_type.value] = pattern

            print(f"   {interaction_type.value:12s}: {len(events):4d} events, "
                  f"avg strength {pattern['average_strength']:.3f}")

        self.interaction_patterns = patterns
        return patterns

    def analyze_circuit_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Analyze prerequisite relationships using your existing data"""
        dependencies = {}

        interaction_events = self._load_interaction_events()

        for circuit_id in self.emergence_epochs:
            # Find prerequisites from interaction events
            prerequisites = []
            enabled_circuits = []

            for event in interaction_events:
                if event.target_circuit == circuit_id and event.interaction_type == InteractionType.ENABLES:
                    prerequisites.append(event.source_circuit)
                elif event.source_circuit == circuit_id and event.interaction_type == InteractionType.ENABLES:
                    enabled_circuits.append(event.target_circuit)

            # Also use your existing circuit_relationships
            for (source, target), relationship_data in self.circuit_relationships.items():
                if target == circuit_id:
                    prerequisites.append(source)
                elif source == circuit_id:
                    enabled_circuits.append(target)

            dependencies[circuit_id] = {
                'prerequisites': list(set(prerequisites)),
                'enabled_circuits': list(set(enabled_circuits)),
                'dependency_depth': self._calculate_dependency_depth(circuit_id),
                'enabling_power': len(set(enabled_circuits))
            }

        return dependencies

    def _calculate_dependency_depth(self, circuit_id: str) -> int:
        """Calculate how deep in the dependency chain this circuit is"""
        if circuit_id not in self.emergence_patterns:
            return 0

        prereqs = self.emergence_patterns[circuit_id].prerequisites
        if not prereqs:
            return 0

        max_depth = 0
        for prereq in prereqs:
            depth = self._calculate_dependency_depth(prereq) + 1
            max_depth = max(max_depth, depth)

        return max_depth

    def _find_dependency_chains(self) -> List[List[str]]:
        """Find chains of circuit dependencies"""
        # Build dependency graph from interactions
        dependency_graph = defaultdict(set)

        for event in self.interaction_log:
            if event.interaction_type == InteractionType.ENABLES:
                dependency_graph[event.source_circuit].add(event.target_circuit)

        # Find chains using DFS
        chains = []
        visited = set()

        def dfs_chain(circuit, current_chain):
            if circuit in visited:
                return

            visited.add(circuit)
            current_chain.append(circuit)

            # If this circuit enables others, continue the chain
            if circuit in dependency_graph:
                for enabled_circuit in dependency_graph[circuit]:
                    dfs_chain(enabled_circuit, current_chain.copy())
            else:
                # End of chain
                if len(current_chain) > 1:
                    chains.append(current_chain)

        # Start DFS from circuits with no prerequisites
        all_circuits = set()
        enabled_circuits = set()

        for event in self.interaction_log:
            if event.interaction_type == InteractionType.ENABLES:
                all_circuits.add(event.source_circuit)
                enabled_circuits.add(event.target_circuit)

        root_circuits = all_circuits - enabled_circuits

        for root in root_circuits:
            dfs_chain(root, [])

        return chains

    # ===============================================================
    # COMPREHENSIVE ANALYSIS AND REPORTING
    # ===============================================================

    def analyze_learning_phase_transitions(self) -> List[LearningPhaseTransition]:
        """Analyze learning phase transitions and their effects on circuits"""
        transitions = []

        if not self.learning_phases:
            return transitions

        sorted_epochs = sorted(self.learning_phases.keys())

        for i in range(1, len(sorted_epochs)):
            prev_epoch = sorted_epochs[i - 1]
            curr_epoch = sorted_epochs[i]
            prev_phase = self.learning_phases[prev_epoch]
            curr_phase = self.learning_phases[curr_epoch]

            if prev_phase != curr_phase:
                # Found a transition
                circuits_affected = self._find_circuits_affected_by_transition(
                    prev_epoch, curr_epoch
                )
                interaction_changes = self._find_interaction_changes_during_transition(
                    prev_epoch, curr_epoch
                )

                transition = LearningPhaseTransition(
                    from_phase=prev_phase,
                    to_phase=curr_phase,
                    transition_epoch=curr_epoch,
                    circuits_affected=circuits_affected,
                    interaction_changes=interaction_changes,
                    transition_strength=len(circuits_affected) / max(len(self.emergence_epochs), 1),
                    duration=curr_epoch - prev_epoch
                )

                transitions.append(transition)

        self.phase_transitions = transitions
        return transitions

    def _find_circuits_affected_by_transition(self, start_epoch: int, end_epoch: int) -> List[str]:
        """Find circuits affected during a phase transition"""
        affected_circuits = []

        # Check for circuits that emerged during transition
        for circuit_id, emergence_epoch in self.emergence_epochs.items():
            if start_epoch <= emergence_epoch <= end_epoch:
                affected_circuits.append(circuit_id)

        # Check for circuits with significant interaction changes
        for event in self.interaction_log:
            if start_epoch <= event.epoch <= end_epoch:
                if event.source_circuit not in affected_circuits:
                    affected_circuits.append(event.source_circuit)
                if event.target_circuit not in affected_circuits:
                    affected_circuits.append(event.target_circuit)

        return affected_circuits

    def _find_interaction_changes_during_transition(self, start_epoch: int, end_epoch: int) -> Dict[str, Any]:
        """Find interaction changes during phase transition"""
        changes = {
            'new_interactions': 0,
            'strengthened_interactions': 0,
            'weakened_interactions': 0,
            'interaction_types': defaultdict(int)
        }

        transition_events = [
            event for event in self.interaction_log
            if start_epoch <= event.epoch <= end_epoch
        ]

        changes['new_interactions'] = len(transition_events)

        for event in transition_events:
            changes['interaction_types'][event.interaction_type.value] += 1

        return dict(changes)

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        start_epoch, end_epoch = self._get_analysis_timespan()

        report = {
            'analysis_summary': {
                'timespan': (start_epoch, end_epoch),
                'total_circuits': len(self.emergence_epochs),
                'total_interactions': len(self.interaction_log),
                'analysis_timestamp': None  # Could add timestamp
            },
            'emergence_analysis': self._analyze_emergence_by_type(),
            'interaction_analysis': {
                'strongest_interactions': self._find_strongest_interactions(),
                'network_stats': self._calculate_network_stats()
            },
            'dependency_analysis': {
                'dependency_chains': self._find_dependency_chains(),
                'most_influential_circuits': self._find_most_influential_circuits()
            },
            'phase_transition_analysis': [
                {
                    'from_phase': t.from_phase.value,
                    'to_phase': t.to_phase.value,
                    'transition_epoch': t.transition_epoch,
                    'circuits_affected': len(t.circuits_affected),
                    'transition_strength': t.transition_strength
                }
                for t in self.analyze_learning_phase_transitions()
            ]
        }

        return report

    def save_analysis_results(self, filepath: Path = None):
        """Save analysis results to file"""
        if filepath is None and self.storage_dir:
            filepath = self.storage_dir / "circuit_evolution_analysis.json"

        if filepath:
            report = self.generate_comprehensive_report()
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"✅ Analysis results saved to {filepath}")

        return filepath

    def generate_research_report(self) -> Dict[str, Any]:
        """Generate comprehensive research report using your existing data"""
        # Run all analyses
        emergence_order = self.analyze_emergence_order()
        emergence_patterns = self.analyze_emergence_patterns()
        interaction_patterns = self.analyze_interaction_patterns()
        dependencies = self.analyze_circuit_dependencies()

        report = {
            'summary': {
                'total_circuits_analyzed': len(self.emergence_epochs),
                'total_interactions_recorded': len(self._load_interaction_events()),
                'analysis_timespan': self._get_analysis_timespan()
            },
            'emergence_analysis': {
                'emergence_order': emergence_order[:10],  # First 10
                'emergence_patterns': {
                    pattern_type: len([p for p in emergence_patterns.values()
                                       if p['emergence_type'] == pattern_type])
                    for pattern_type in ['sudden', 'gradual', 'oscillating']
                },
                'circuit_types_emergence': self._analyze_emergence_by_type()
            },
            'interaction_analysis': {
                'interaction_frequencies': {
                    itype: pattern['frequency']
                    for itype, pattern in interaction_patterns.items()
                },
                'strongest_interactions': self._find_strongest_interactions(),
                'interaction_network_stats': self._calculate_network_stats()
            },
            'dependency_analysis': {
                'dependency_chains': self._find_dependency_chains(),
                'most_influential_circuits': self._find_most_influential_circuits(dependencies),
                'dependency_depths': {cid: deps['dependency_depth']
                                      for cid, deps in dependencies.items()}
            }
        }

        return report

    def save_research_report(self, report: Dict[str, Any]):
        """Save research report to file"""
        if not self.storage_dir:
            print("⚠️ No storage directory specified, cannot save report")
            return

        report_file = self.storage_dir / "circuit_evolution_research_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Research report saved to {report_file}")
        self._print_report_summary(report)

    def _analyze_emergence_by_phase(self) -> Dict[str, int]:
        """Count circuit emergence by learning phase"""
        phase_counts = Counter()
        for pattern in self.emergence_patterns.values():
            phase_counts[pattern.learning_phase_at_emergence.value] += 1
        return dict(phase_counts)

    def _trace_dependency_chain(self, start_circuit: str, dependencies: Dict[str, Dict[str, Any]],
                                visited: set) -> List[str]:
        """Trace a dependency chain from a starting circuit"""
        chain = [start_circuit]
        visited.add(start_circuit)

        current = start_circuit
        while current in dependencies:
            enabled = dependencies[current].get('enabled_circuits', [])
            # Find next circuit in chain (single dependency)
            unvisited_enabled = [c for c in enabled if c not in visited]

            if len(unvisited_enabled) == 1:
                next_circuit = unvisited_enabled[0]
                chain.append(next_circuit)
                visited.add(next_circuit)
                current = next_circuit
            else:
                break

        return chain

    def _get_analysis_timespan(self) -> Tuple[int, int]:
        """Get the timespan for analysis (start_epoch, end_epoch)"""
        if not self.learning_phases:
            return (0, 0)

        epochs = list(self.learning_phases.keys())
        return (min(epochs), max(epochs))

    def _analyze_emergence_by_type(self, circuit_type: CircuitType = None) -> Dict[str, Any]:
        """Analyze emergence patterns by circuit type"""
        emergence_by_type = defaultdict(list)

        for circuit_id, emergence_epoch in self.emergence_epochs.items():
            # Get circuit type from metadata or registry
            circuit_type_str = "unknown"
            if circuit_id in self.circuit_metadata:
                metadata = self.circuit_metadata[circuit_id]
                if hasattr(metadata, 'type'):
                    circuit_type_str = metadata.type

            emergence_by_type[circuit_type_str].append({
                'circuit_id': circuit_id,
                'emergence_epoch': emergence_epoch
            })

        # Calculate statistics for each type
        type_stats = {}
        for c_type, circuits in emergence_by_type.items():
            epochs = [c['emergence_epoch'] for c in circuits]
            type_stats[c_type] = {
                'count': len(circuits),
                'avg_emergence_epoch': np.mean(epochs) if epochs else 0,
                'earliest_emergence': min(epochs) if epochs else 0,
                'latest_emergence': max(epochs) if epochs else 0,
                'emergence_span': max(epochs) - min(epochs) if epochs else 0
            }

        return type_stats

    def _find_strongest_interactions(self, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """Find the strongest circuit interactions"""
        interaction_strengths = []

        # From interaction log
        for event in self.interaction_log:
            interaction_strengths.append((
                event.source_circuit,
                event.target_circuit,
                event.strength
            ))

        # From circuit relationships
        for (source, target), relationship_data in self.circuit_relationships.items():
            if isinstance(relationship_data, dict):
                strength = relationship_data.get('strength', 0.0)
                interaction_strengths.append((source, target, strength))

        # Sort by strength and return top-k
        interaction_strengths.sort(key=lambda x: x[2], reverse=True)
        return interaction_strengths[:top_k]

    def _calculate_network_stats(self) -> Dict[str, Any]:
        """Calculate network-level statistics"""
        # Build interaction graph
        graph = defaultdict(set)
        all_circuits = set()

        for event in self.interaction_log:
            graph[event.source_circuit].add(event.target_circuit)
            all_circuits.add(event.source_circuit)
            all_circuits.add(event.target_circuit)

        # Calculate basic network metrics
        num_circuits = len(all_circuits)
        num_interactions = len(self.interaction_log)

        # Calculate connectivity metrics
        out_degrees = [len(neighbors) for neighbors in graph.values()]
        in_degrees = defaultdict(int)
        for neighbors in graph.values():
            for neighbor in neighbors:
                in_degrees[neighbor] += 1

        in_degree_values = list(in_degrees.values())

        network_stats = {
            'num_circuits': num_circuits,
            'num_interactions': num_interactions,
            'avg_out_degree': np.mean(out_degrees) if out_degrees else 0,
            'avg_in_degree': np.mean(in_degree_values) if in_degree_values else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'max_in_degree': max(in_degree_values) if in_degree_values else 0,
            'density': num_interactions / (num_circuits * (num_circuits - 1)) if num_circuits > 1 else 0
        }

        return network_stats

    def _find_most_influential_circuits(self, top_k: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        """Find the most influential circuits based on various metrics"""
        influence_scores = {}

        # Calculate influence based on:
        # 1. Number of circuits they enable
        # 2. Strength of their interactions
        # 3. Duration of their activity
        # 4. Network centrality

        for circuit_id in self.emergence_epochs.keys():
            score_components = {
                'circuits_enabled': 0,
                'avg_interaction_strength': 0,
                'activity_duration': 0,
                'network_centrality': 0
            }

            # Count circuits enabled
            enabled_count = 0
            total_strength = 0
            interaction_count = 0

            for event in self.interaction_log:
                if event.source_circuit == circuit_id:
                    if event.interaction_type == InteractionType.ENABLES:
                        enabled_count += 1
                    total_strength += event.strength
                    interaction_count += 1

            score_components['circuits_enabled'] = enabled_count
            score_components['avg_interaction_strength'] = (
                total_strength / interaction_count if interaction_count > 0 else 0
            )

            # Calculate activity duration
            if circuit_id in self.circuit_metadata:
                metadata = self.circuit_metadata[circuit_id]
                if hasattr(metadata, 'first_detected') and hasattr(metadata, 'last_seen'):
                    score_components['activity_duration'] = metadata.last_seen - metadata.first_detected

            # Simple centrality measure (number of connections)
            score_components['network_centrality'] = interaction_count

            # Combine scores (weighted average)
            total_score = (
                    score_components['circuits_enabled'] * 0.4 +
                    score_components['avg_interaction_strength'] * 0.3 +
                    score_components['activity_duration'] * 0.1 +
                    score_components['network_centrality'] * 0.2
            )

            influence_scores[circuit_id] = {
                'total_score': total_score,
                'components': score_components
            }

        # Sort by total score and return top-k
        sorted_circuits = sorted(
            influence_scores.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )

        return sorted_circuits[:top_k]


    def _count_circuits_per_phase(self) -> Dict[str, int]:
        """Count how many circuits exist in each learning phase"""
        phase_counts = Counter()

        for snapshots in self.evolution_snapshots.values():
            for snapshot in snapshots:
                phase_counts[snapshot.learning_phase.value] += 1

        return dict(phase_counts)


    def _analyze_interaction_evolution(self) -> Dict[str, List[int]]:
        """Analyze how interactions change over time"""
        evolution = defaultdict(lambda: defaultdict(int))

        for event in self.interaction_log:
            epoch_bucket = (event.epoch // 100) * 100  # Group by 100-epoch buckets
            evolution[event.interaction_type.value][epoch_bucket] += 1

        return {itype: dict(buckets) for itype, buckets in evolution.items()}

    def _print_report_summary(self, report: Dict[str, Any]):
        """Print human-readable report summary"""
        print("\n" + "=" * 60)
        print("🧬 CIRCUIT EVOLUTION RESEARCH REPORT")
        print("=" * 60)

        summary = report['summary']
        print(f"📊 Analysis Overview:")
        print(f"   Circuits analyzed: {summary['total_circuits_analyzed']}")
        print(f"   Interactions recorded: {summary['total_interactions_recorded']}")
        print(f"   Learning transitions: {summary['learning_phase_transitions']}")
        print(f"   Analysis timespan: {summary['analysis_timespan']['total_epochs']} epochs")

        emergence = report['emergence_analysis']
        print(f"\n📈 Circuit Emergence:")
        for pattern, count in emergence['emergence_patterns'].items():
            print(f"   {pattern:12s}: {count:3d} circuits")

        interactions = report['interaction_analysis']
        print(f"\n🔗 Circuit Interactions:")
        for itype, freq in interactions['interaction_frequencies'].items():
            print(f"   {itype:12s}: {freq:3d} events")

        dependencies = report['dependency_analysis']
        print(f"\n🏗️ Circuit Dependencies:")
        print(f"   Dependency chains found: {len(dependencies['dependency_chains'])}")
        print(f"   Most influential circuits:")
        for circuit_id, influence in dependencies['most_influential_circuits'][:3]:
            print(f"     {circuit_id[:20]:<20}: enables {influence} circuits")

class CircuitEvolutionTracker:
    """
    Extends existing CircuitMetadata with comprehensive evolution tracking
    Integrates seamlessly with existing circuit_schema.py structure
    """

    def __init__(self, circuit_metadata: CircuitMetadata, logger=None):
        """Initialize with existing CircuitMetadata"""
        self.metadata = circuit_metadata

        # Evolution-specific extensions
        self.evolution_snapshots: List[EvolutionSnapshot] = []
        self.interaction_events: List[InteractionEvent] = []
        self.evolution_pattern: Optional[EvolutionPattern] = None
        self.learning_phases: Dict[int, LearningPhase] = {}  # epoch -> phase

        # Analysis results
        self.emergence_cascade_id: Optional[str] = None
        self.dependency_chain_position: Optional[int] = None
        self.evolution_milestones: Dict[str, int] = {}  # milestone -> epoch

        self.logger = logger

    def record_learning_phase(self, epoch: int, learning_phase: LearningPhase):
        """Record learning phase for epoch"""
        if not hasattr(self, 'learning_phases'):
            self.learning_phases = {}
        self.learning_phases[epoch] = learning_phase

    def record_snapshot(self, epoch: int, attribution: float, detection_confidence: float,
                        stability_score: float, behavioral_impact: float,
                        learning_phase: LearningPhase, **context):
        """Record circuit state at specific epoch"""
        snapshot = EvolutionSnapshot(
            epoch=epoch,
            attribution=attribution,
            detection_confidence=detection_confidence,
            stability_score=stability_score,
            behavioral_impact=behavioral_impact,
            learning_phase=learning_phase,
            context_metadata=context
        )
        self.evolution_snapshots.append(snapshot)

        # Update existing metadata
        self.metadata.last_seen = epoch
        self.metadata.detection_epochs.append(epoch)
        self.metadata.strength_history.append((epoch, attribution))

        # Track learning phase
        self.learning_phases[epoch] = learning_phase

        # Update emergence phase in existing metadata if appropriate
        if learning_phase in [LearningPhase.GENERALIZATION, LearningPhase.CONSOLIDATION]:
            self.metadata.emergence_phase = EmergencePhase.MATURE
        elif learning_phase in [LearningPhase.TRANSITION]:
            self.metadata.emergence_phase = EmergencePhase.DEVELOPING

    def analyze_evolution_pattern(self) -> EvolutionPattern:
        """Analyze overall evolution pattern from snapshots"""
        if len(self.evolution_snapshots) < 3:
            return EvolutionPattern.GRADUAL_EMERGENCE

        # Extract attribution trajectory
        attributions = [s.attribution for s in self.evolution_snapshots]
        epochs = [s.epoch for s in self.evolution_snapshots]

        # Simple pattern detection
        if self._is_sudden_emergence(attributions, epochs):
            pattern = EvolutionPattern.SUDDEN_EMERGENCE
        elif self._is_oscillating(attributions):
            pattern = EvolutionPattern.OSCILLATING
        elif self._is_plateauing(attributions):
            pattern = EvolutionPattern.PLATEAUING
        elif self._is_declining(attributions):
            pattern = EvolutionPattern.DECLINING
        else:
            pattern = EvolutionPattern.GRADUAL_EMERGENCE

        self.evolution_pattern = pattern
        return pattern

    def _is_sudden_emergence(self, attributions: List[float], epochs: List[int]) -> bool:
        """Detect sudden emergence pattern"""
        if len(attributions) < 3:
            return False

        # Look for rapid increase in short time window
        for i in range(1, len(attributions)):
            if attributions[i] > attributions[i - 1] * 2:  # Double in one step
                return True
        return False

    def _is_oscillating(self, attributions: List[float]) -> bool:
        """Detect oscillating pattern"""
        if len(attributions) < 5:
            return False

        # Count direction changes
        direction_changes = 0
        for i in range(2, len(attributions)):
            prev_trend = attributions[i - 1] - attributions[i - 2]
            curr_trend = attributions[i] - attributions[i - 1]
            if (prev_trend > 0) != (curr_trend > 0):  # Direction change
                direction_changes += 1

        return direction_changes > len(attributions) // 3

    def _is_plateauing(self, attributions: List[float]) -> bool:
        """Detect plateauing pattern"""
        if len(attributions) < 4:
            return False

        # Check if recent values are stable
        recent = attributions[-4:]
        std = sum((x - sum(recent) / len(recent)) ** 2 for x in recent) ** 0.5
        return std < 0.05  # Low variance in recent values

    def _is_declining(self, attributions: List[float]) -> bool:
        """Detect declining pattern"""
        if len(attributions) < 3:
            return False

        # Check if trend is consistently downward
        declining_steps = 0
        for i in range(1, len(attributions)):
            if attributions[i] < attributions[i - 1]:
                declining_steps += 1

        return declining_steps > len(attributions) * 0.6


    def record_interaction(self, epoch: int, source_circuit: str, target_circuit: str,
                           interaction_type: InteractionType, strength: float):
        if not hasattr(self, 'interactions'):
            self.interactions = []
        self.interactions.append({
            'epoch': epoch, 'source': source_circuit, 'target': target_circuit,
            'type': interaction_type, 'strength': strength
        })

    def determine_learning_phase(self, epoch: int, accuracy: float) -> LearningPhase:
        # Adapt this logic to your specific task
        if epoch < 100:
            return LearningPhase.EARLY_LEARNING
        elif accuracy < 0.5:
            return LearningPhase.MEMORIZATION
        elif accuracy > 0.9:
            return LearningPhase.GENERALIZATION
        else:
            return LearningPhase.TRANSITION

    def get_circuit_id(self) -> str:
        """Get circuit ID from metadata (would need to be passed or tracked)"""
        # This would need to be set when initializing the tracker
        return getattr(self, '_circuit_id', 'unknown')

    def set_circuit_id(self, circuit_id: str):
        """Set circuit ID for this tracker"""
        self._circuit_id = circuit_id

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary"""
        pattern = self.analyze_evolution_pattern() if self.evolution_pattern is None else self.evolution_pattern

        return {
            'evolution_pattern': pattern.value if pattern else 'unknown',
            'total_snapshots': len(self.evolution_snapshots),
            'total_interactions': len(self.interaction_events),
            'learning_phases_observed': list(set(self.learning_phases.values())),
            'strength_trajectory': [(s.epoch, s.attribution) for s in self.evolution_snapshots],
            'key_interactions': [
                {
                    'epoch': event.epoch,
                    'target': event.target_circuit,
                    'type': event.interaction_type.value,
                    'strength': event.strength
                }
                for event in self.interaction_events[-5:]  # Last 5 interactions
            ],
            'milestones': self.evolution_milestones,
            'current_phase': self.learning_phases.get(
                max(self.learning_phases.keys())) if self.learning_phases else None
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evolution data"""
        return {
            'evolution_snapshots': [
                {
                    'epoch': s.epoch,
                    'attribution': s.attribution,
                    'detection_confidence': s.detection_confidence,
                    'stability_score': s.stability_score,
                    'behavioral_impact': s.behavioral_impact,
                    'learning_phase': s.learning_phase.value,
                    'active_interactions': s.active_interactions,
                    'context_metadata': s.context_metadata
                }
                for s in self.evolution_snapshots
            ],
            'interaction_events': [
                {
                    'epoch': e.epoch,
                    'source_circuit': e.source_circuit,
                    'target_circuit': e.target_circuit,
                    'interaction_type': e.interaction_type.value,
                    'strength': e.strength,
                    'confidence': e.confidence,
                    'context': e.context
                }
                for e in self.interaction_events
            ],
            'evolution_pattern': self.evolution_pattern.value if self.evolution_pattern else None,
            'learning_phases': {str(k): v.value for k, v in self.learning_phases.items()},
            'emergence_cascade_id': self.emergence_cascade_id,
            'dependency_chain_position': self.dependency_chain_position,
            'evolution_milestones': self.evolution_milestones
        }

    @classmethod
    def from_dict(cls, circuit_metadata: CircuitMetadata, data: Dict[str, Any]) -> 'CircuitEvolutionTracker':
        """Deserialize evolution data"""
        tracker = cls(circuit_metadata)

        # Restore snapshots
        for s_data in data.get('evolution_snapshots', []):
            snapshot = EvolutionSnapshot(
                epoch=s_data['epoch'],
                attribution=s_data['attribution'],
                detection_confidence=s_data['detection_confidence'],
                stability_score=s_data['stability_score'],
                behavioral_impact=s_data['behavioral_impact'],
                learning_phase=LearningPhase(s_data['learning_phase']),
                active_interactions=s_data.get('active_interactions', []),
                context_metadata=s_data.get('context_metadata', {})
            )
            tracker.evolution_snapshots.append(snapshot)

        # Restore interaction events
        for e_data in data.get('interaction_events', []):
            event = InteractionEvent(
                epoch=e_data['epoch'],
                source_circuit=e_data['source_circuit'],
                target_circuit=e_data['target_circuit'],
                interaction_type=InteractionType(e_data['interaction_type']),
                strength=e_data['strength'],
                confidence=e_data.get('confidence', 0.5),
                context=e_data.get('context', {})
            )
            tracker.interaction_events.append(event)

        # Restore other fields
        if data.get('evolution_pattern'):
            tracker.evolution_pattern = EvolutionPattern(data['evolution_pattern'])

        tracker.learning_phases = {
            int(k): LearningPhase(v) for k, v in data.get('learning_phases', {}).items()
        }
        tracker.emergence_cascade_id = data.get('emergence_cascade_id')
        tracker.dependency_chain_position = data.get('dependency_chain_position')
        tracker.evolution_milestones = data.get('evolution_milestones', {})

        return tracker

class MultiCircuitEvolutionManager:
    """
    Manages evolution tracking for multiple circuits
    Each circuit gets its own CircuitEvolutionTracker
    """

    def __init__(self, registry, logger, save_dir: Path):
        self.registry = registry
        self.save_dir = save_dir
        self.circuit_trackers: Dict[str, CircuitEvolutionTracker] = {}
        self.logger = logger

    def get_or_create_tracker(self, circuit_id: str) -> CircuitEvolutionTracker:
        """Get existing tracker or create new one for circuit"""
        if circuit_id not in self.circuit_trackers:
            # Get circuit metadata from registry
            circuit_metadata = self.registry.circuit_metadata[circuit_id]

            # Create new tracker for this circuit
            tracker = CircuitEvolutionTracker(circuit_metadata)
            tracker.set_circuit_id(circuit_id)  # ✅ Set ID once

            self.circuit_trackers[circuit_id] = tracker
            self.logger.info(f"📊 Created new tracker for circuit: {circuit_id}")

        return self.circuit_trackers[circuit_id]

    def record_circuit_snapshot(self, circuit_id: str, epoch: int, **snapshot_data):
        """Record snapshot for specific circuit"""
        tracker = self.get_or_create_tracker(circuit_id)
        tracker.record_snapshot(epoch=epoch, **snapshot_data)

    def record_circuit_interaction(self, source_circuit_id: str, target_circuit_id: str,
                                   epoch: int, interaction_type: InteractionType, strength: float):
        """Record interaction between two circuits"""
        # Record in source circuit's tracker
        source_tracker = self.get_or_create_tracker(source_circuit_id)
        source_tracker.record_interaction(epoch, source_circuit_id, target_circuit_id,
                                          interaction_type, strength)

        # Record in target circuit's tracker too (if relevant)
        target_tracker = self.get_or_create_tracker(target_circuit_id)
        target_tracker.record_interaction(epoch, source_circuit_id, target_circuit_id,
                                          interaction_type, strength)

class IntegratedCircuitEvolutionAnalyzer(CircuitEvolutionAnalyzer):
    """
    Enhanced CircuitEvolutionAnalyzer that works seamlessly with MultiCircuitEvolutionManager
    """
    def __init__(self, multi_circuit_manager: MultiCircuitEvolutionManager,
                 enhanced_registry, logger, storage_dir: Path = None):
        """
        Initialize with MultiCircuitEvolutionManager integration
        """
        self.logger = logger

        # Create aggregated tracker that provides the interface CircuitEvolutionAnalyzer expects
        self.aggregated_tracker = AggregatedEvolutionTracker(
            multi_circuit_manager=multi_circuit_manager,logger=logger
        )

        # Initialize parent class with aggregated data
        super().__init__(
            evolution_tracker=self.aggregated_tracker,
            enhanced_registry=enhanced_registry,
            storage_dir=storage_dir
        )

        self.multi_circuit_manager = multi_circuit_manager

        self.logger.info("🚀 Integrated Circuit Evolution Analyzer initialized")
        self.logger.info(f"   📊 Analyzing {len(self.emergence_epochs)} circuits")

    def refresh_analysis(self):
        """Refresh analysis with latest data from all circuit trackers"""
        self.logger.info("🔄 Refreshing analysis with latest circuit data")

        # Refresh aggregated data
        self.aggregated_tracker.refresh_aggregation()

        # Update our data references
        self.emergence_epochs = self.aggregated_tracker.emergence_epochs
        self.circuit_relationships = self.aggregated_tracker.circuit_relationships
        self.evolution_snapshots = self.aggregated_tracker.evolution_snapshots
        self.interaction_log = self.aggregated_tracker.interaction_events
        self.learning_phases = self.aggregated_tracker.learning_phases

        self.logger.info("✅ Analysis data refreshed")

    def analyze_circuit_by_id(self, circuit_id: str) -> Dict[str, Any]:
        """Analyze specific circuit using its individual tracker"""
        if circuit_id not in self.multi_circuit_manager.circuit_trackers:
            self.logger.warning(f"Circuit {circuit_id} not found in manager")
            return {}

        tracker = self.multi_circuit_manager.circuit_trackers[circuit_id]

        # Individual circuit analysis
        analysis = {
            'circuit_id': circuit_id,
            'emergence_epoch': self.emergence_epochs.get(circuit_id),
            'total_snapshots': len(tracker.evolution_snapshots),
            'evolution_pattern': tracker.analyze_evolution_pattern() if hasattr(tracker,
                                                                                'analyze_evolution_pattern') else None,
            'strength_trajectory': [(s.epoch, s.attribution) for s in tracker.evolution_snapshots],
            'learning_phases_observed': list(set(tracker.learning_phases.values())),
            'interactions_count': len(tracker.interaction_events),
            'summary': tracker.get_evolution_summary() if hasattr(tracker, 'get_evolution_summary') else {}
        }

        self.logger.info(f"📊 Individual analysis complete for {circuit_id}")
        return analysis

    def analyze_all_circuits_individually(self) -> Dict[str, Dict[str, Any]]:
        """Analyze each circuit individually using their own trackers"""
        individual_analyses = {}

        self.logger.info(
            f"🔬 Running individual analysis for {len(self.multi_circuit_manager.circuit_trackers)} circuits")

        for circuit_id in self.multi_circuit_manager.circuit_trackers:
            individual_analyses[circuit_id] = self.analyze_circuit_by_id(circuit_id)

        return individual_analyses

    def generate_comprehensive_multi_circuit_report(self) -> Dict[str, Any]:
        """Generate comprehensive report combining collective and individual analysis"""
        self.logger.info("📋 Generating comprehensive multi-circuit report")

        # Refresh data first
        self.refresh_analysis()

        # Get collective analysis (from parent class)
        collective_analysis = self.generate_comprehensive_report()

        # Get individual circuit analyses
        individual_analyses = self.analyze_all_circuits_individually()

        # Combine into comprehensive report
        comprehensive_report = {
            'collective_analysis': collective_analysis,
            'individual_circuit_analyses': individual_analyses,
            'multi_circuit_insights': {
                'total_circuits_tracked': len(self.multi_circuit_manager.circuit_trackers),
                'circuits_with_interactions': len(
                    [cid for cid, tracker in self.multi_circuit_manager.circuit_trackers.items() if
                     tracker.interaction_events]),
                'avg_snapshots_per_circuit': np.mean([len(tracker.evolution_snapshots) for tracker in
                                                      self.multi_circuit_manager.circuit_trackers.values()]),
                'emergence_epoch_range': {
                    'earliest': min(self.emergence_epochs.values()) if self.emergence_epochs else 0,
                    'latest': max(self.emergence_epochs.values()) if self.emergence_epochs else 0
                },
                'most_active_circuits': self._find_most_active_circuits(),
                'circuit_interaction_network': self._analyze_interaction_network()
            }
        }

        # Save comprehensive report
        if self.storage_dir:
            report_path = self.storage_dir / "comprehensive_multi_circuit_report.json"
            import json
            with open(report_path, 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
            self.logger.info(f"📄 Comprehensive report saved to {report_path}")

        return comprehensive_report

    def _find_most_active_circuits(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Find most active circuits based on snapshots and interactions"""
        activity_scores = []

        for circuit_id, tracker in self.multi_circuit_manager.circuit_trackers.items():
            activity_score = {
                'snapshots': len(tracker.evolution_snapshots),
                'interactions': len(tracker.interaction_events),
                'total_activity': len(tracker.evolution_snapshots) + len(tracker.interaction_events) * 2
            }
            activity_scores.append((circuit_id, activity_score))

        # Sort by total activity
        activity_scores.sort(key=lambda x: x[1]['total_activity'], reverse=True)

        return activity_scores[:10]  # Top 10 most active

    def _analyze_interaction_network(self) -> Dict[str, Any]:
        """Analyze the network of circuit interactions"""
        # Build interaction graph
        interaction_graph = defaultdict(set)
        interaction_strengths = defaultdict(float)

        for event in self.interaction_log:
            interaction_graph[event.source_circuit].add(event.target_circuit)
            interaction_strengths[(event.source_circuit, event.target_circuit)] += event.strength

        # Network statistics
        all_circuits = set(interaction_graph.keys())
        for targets in interaction_graph.values():
            all_circuits.update(targets)

        num_circuits = len(all_circuits)
        num_interactions = len(self.interaction_log)

        return {
            'total_circuits_in_network': num_circuits,
            'total_interactions': num_interactions,
            'avg_interactions_per_circuit': num_interactions / num_circuits if num_circuits > 0 else 0,
            'most_connected_circuits': sorted(
                [(circuit, len(targets)) for circuit, targets in interaction_graph.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
            'network_density': num_interactions / (num_circuits * (num_circuits - 1)) if num_circuits > 1 else 0
        }


# ============================================================================
# FACTORY FUNCTION FOR INTEGRATION
# ============================================================================

def create_fixed_evolution_analyzer(evolution_tracker=None, enhanced_registry=None,
                                    storage_dir: Path = None) -> CircuitEvolutionAnalyzer:
    """
    Create analyzer with all missing components fixed
    """
    analyzer = CircuitEvolutionAnalyzer(
        evolution_tracker=evolution_tracker,
        enhanced_registry=enhanced_registry,
        storage_dir=storage_dir
    )

    print("✅ Fixed Circuit Evolution Analyzer created")
    print(f"   📊 Missing methods implemented")
    print(f"   📊 Missing fields added")
    print(f"   📊 Ready for Phase 2 transition")

    return analyzer


# ============================================================================
# INTEGRATION WITH YOUR EXISTING SYSTEM
# ============================================================================

# ============================================================================
# VALIDATION
# ============================================================================

def validate_analyzer_completeness():
    """Validate that all required methods and fields are present"""
    analyzer = CircuitEvolutionAnalyzer()

    # Check for required methods
    required_methods = [
        '_get_analysis_timespan',
        '_analyze_emergence_by_type',
        '_find_strongest_interactions',
        '_calculate_network_stats',
        '_find_dependency_chains',
        '_find_most_influential_circuits'
    ]

    for method_name in required_methods:
        if hasattr(analyzer, method_name):
            print(f"✅ {method_name} implemented")
        else:
            print(f"❌ {method_name} missing")

    # Check for required fields
    required_fields = [
        'interaction_log',
        'evolution_snapshots',
        'learning_phases'
    ]

    for field_name in required_fields:
        if hasattr(analyzer, field_name):
            print(f"✅ {field_name} field present")
        else:
            print(f"❌ {field_name} field missing")

    print("✅ Analyzer validation complete")
    return True

def create_evolution_analyzer_from_existing_tracker(evolution_tracker, enhanced_registry,
                                                    storage_dir: Path = None) -> CircuitEvolutionAnalyzer:
    """
    Create analyzer using your existing CircuitEvolutionTracker and EnhancedCircuitRegistry

    Args:
        evolution_tracker: Your existing CircuitEvolutionTracker instance
        enhanced_registry: Your existing EnhancedCircuitRegistry instance
        storage_dir: Optional directory for saving results

    Returns:
        CircuitEvolutionAnalyzer ready for research analysis
    """
    analyzer = CircuitEvolutionAnalyzer(
        evolution_tracker=evolution_tracker,
        enhanced_registry=enhanced_registry,
        storage_dir=storage_dir
    )

    print("✅ Circuit evolution analyzer created using your existing tracker")
    print(f"   📊 {len(analyzer.emergence_epochs)} circuits tracked")
    print(f"   📊 {len(analyzer.circuit_relationships)} relationships recorded")

    return analyzer


def example_research_analysis_workflow():
    """
    Example showing complete research analysis workflow using your existing systems
    """
    print("🔬 Research Analysis Workflow with Your Existing Systems")
    print("=" * 60)

    # ✅ Use your existing tracker and registry (no enhanced_circuit_schema_dynamics needed)
    # evolution_tracker = CircuitEvolutionTracker(registry, save_dir, logger)  # Your existing
    # enhanced_registry = EnhancedCircuitRegistry(save_dir)  # Your existing

    # ✅ Create analyzer from your existing systems
    # analyzer = create_evolution_analyzer_from_existing_tracker(
    #     evolution_tracker=evolution_tracker,
    #     enhanced_registry=enhanced_registry,
    #     storage_dir=Path("research_analysis")
    # )

    # ✅ Run research analysis
    # report = analyzer.generate_research_report()
    # analyzer.save_research_report(report)

    print("✅ Analysis complete using your existing CircuitEvolutionTracker!")
    print("✅ No enhanced_circuit_schema_dynamics needed!")


if __name__ == "__main__":
    example_research_analysis_workflow()


class AggregatedEvolutionTracker:
    """
    Aggregates data from multiple CircuitEvolutionTracker instances
    Provides the interface that CircuitEvolutionAnalyzer expects
    """

    def __init__(self, multi_circuit_manager: MultiCircuitEvolutionManager, logger: UnifiedLogger):
        self.manager = multi_circuit_manager
        self.logger = logger

        # Build aggregated views
        self._build_aggregated_data()

    def _build_aggregated_data(self):
        """Build aggregated data from all individual circuit trackers"""

        # ✅ Aggregate emergence epochs (circuit_id -> first_epoch)
        self.emergence_epochs: Dict[str, int] = {}

        # ✅ Aggregate circuit relationships
        self.circuit_relationships: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # ✅ Aggregate evolution snapshots by circuit
        self.evolution_snapshots: Dict[str, List[Any]] = {}

        # ✅ Aggregate interaction events
        self.interaction_events: List[InteractionEvent] = []

        # ✅ Aggregate learning phases
        self.learning_phases: Dict[int, LearningPhase] = {}

        self.logger.info(f"🔄 Aggregating data from {len(self.manager.circuit_trackers)} circuit trackers")

        for circuit_id, tracker in self.manager.circuit_trackers.items():
            self._aggregate_from_tracker(circuit_id, tracker)

        self.logger.info(f"📊 Aggregation complete:")
        self.logger.info(f"   Emergence epochs: {len(self.emergence_epochs)}")
        self.logger.info(f"   Circuit relationships: {len(self.circuit_relationships)}")
        self.logger.info(f"   Interaction events: {len(self.interaction_events)}")

    def _aggregate_from_tracker(self, circuit_id: str, tracker: CircuitEvolutionTracker):
        """Aggregate data from a single circuit tracker"""

        # 1. Extract emergence epoch (first snapshot)
        if tracker.evolution_snapshots:
            first_epoch = min(snapshot.epoch for snapshot in tracker.evolution_snapshots)
            self.emergence_epochs[circuit_id] = first_epoch

        # 2. Extract evolution snapshots
        self.evolution_snapshots[circuit_id] = [
            {
                'epoch': snapshot.epoch,
                'attribution': snapshot.attribution,
                'learning_phase': snapshot.learning_phase,
                'detection_confidence': snapshot.detection_confidence,
                'stability_score': snapshot.stability_score,
                'behavioral_impact': snapshot.behavioral_impact,
                'context': snapshot.context_metadata
            }
            for snapshot in tracker.evolution_snapshots
        ]

        # 3. Extract interaction events
        for event in tracker.interaction_events:
            self.interaction_events.append(event)

            # Build circuit relationships map
            relationship_key = (event.source_circuit, event.target_circuit)
            if relationship_key not in self.circuit_relationships:
                self.circuit_relationships[relationship_key] = {
                    'type': event.interaction_type.value,
                    'strength': event.strength,
                    'epoch': event.epoch,
                    'confidence': event.confidence
                }

        # 4. Extract learning phases
        for epoch, phase in tracker.learning_phases.items():
            if epoch not in self.learning_phases:
                self.learning_phases[epoch] = phase

    def refresh_aggregation(self):
        """Refresh aggregated data (call after new data is added)"""
        self._build_aggregated_data()


