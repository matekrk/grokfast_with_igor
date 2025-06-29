# analysis/core/unified_circuit_evolution_tracker.py
"""
Unified Circuit Evolution Tracker - Consolidates all circuit evolution functionality
Combines features from analyzers, core, and helpers implementations
"""

from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import numpy as np
# import matplotlib.pyplot as plt

from analysis.core.circuit_schema import Circuit, CircuitType, CircuitStability
from analysis.utils.utils import get_current_callable_info, shorten_layer_head


class UnifiedCircuitEvolutionTracker:
    """
    Unified tracker for circuit evolution, stability, and relationships

    Combines functionality from all previous CircuitEvolutionTracker implementations:
    - Circuit stability and lifetime tracking (from helpers)
    - Sophisticated relationship analysis (from analyzers)
    - Evolution event tracking (from core)
    - Visualization capabilities
    """

    def __init__(self, registry, save_dir=None, logger=None):
        """
        Initialize the unified circuit evolution tracker

        Args:
            registry: The circuit registry (EnhancedCircuitRegistry)
            save_dir: Optional directory to save analysis results
            logger: Optional logger for metrics
        """
        self.registry = registry
        self.logger = logger

        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        # ============================================================================
        # CORE TRACKING DATA STRUCTURES
        # ============================================================================

        # Temporal evolution data
        self.evolution_data = {}  # Circuit ID -> temporal data
        self.emergence_epochs = {}  # Circuit ID -> emergence epoch
        self.epoch_to_circuits = {}  # Epoch -> circuit IDs that were active

        # Stability and lifetime tracking
        self.circuit_lifetimes = {}  # circuit_id -> (birth_epoch, death_epoch)
        self.stability_scores = {}  # circuit_id -> stability_score
        self.strength_history = {}  # circuit_id -> [(epoch, strength), ...]
        self.consistency_scores = {}  # circuit_id -> consistency_score

        # Relationship tracking
        self.circuit_relationships = {}  # (Circuit ID, Circuit ID) -> relationship data
        self.co_occurrence_matrix = {}  # For tracking co-activation patterns

        # Evolution events
        self.evolution_events = []  # List of evolution events (birth, death, transformation)

        # Performance tracking
        self.analysis_history = {}  # epoch -> analysis_results

    # ============================================================================
    # MAIN TRACKING INTERFACE
    # ============================================================================

    def track_epoch(self, epoch: int, detected_circuits: List[Dict],
                    model_accuracy: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Main interface for tracking circuits in an epoch

        Args:
            epoch: Current epoch
            detected_circuits: List of detected circuit data
            model_accuracy: Current model accuracy
            **kwargs: Additional tracking data

        Returns:
            Dict with epoch tracking results
        """
        # Update all tracking systems
        active_circuit_ids = self._update_circuit_tracking(epoch, detected_circuits)
        stability_updates = self._update_stability_tracking(epoch, active_circuit_ids)
        relationship_updates = self._update_relationship_tracking(epoch, active_circuit_ids)

        # Record epoch summary
        epoch_summary = {
            'epoch': epoch,
            'active_circuits': len(active_circuit_ids),
            'new_circuits': len(stability_updates.get('birth_events', [])),
            'dead_circuits': len(stability_updates.get('death_events', [])),
            'stable_circuits': len(self.get_stable_circuits(epoch)),
            'model_accuracy': model_accuracy,
            'relationship_updates': len(relationship_updates)
        }

        self.analysis_history[epoch] = epoch_summary

        # Log significant events
        if self.logger and (epoch_summary['new_circuits'] > 0 or epoch_summary['dead_circuits'] > 0):
            self.logger.info(f"Circuit evolution @ epoch {epoch}: "
                             f"+{epoch_summary['new_circuits']} new, "
                             f"-{epoch_summary['dead_circuits']} dead, "
                             f"{epoch_summary['stable_circuits']} stable")

        return epoch_summary

    # ============================================================================
    # CIRCUIT TRACKING AND LIFETIME MANAGEMENT
    # ============================================================================

    def _update_circuit_tracking(self, epoch: int, detected_circuits: List[Dict]) -> Set[str]:
        """Update core circuit tracking data"""
        current_circuit_ids = set()

        for circuit_data in detected_circuits:
            circuit_id = self._extract_circuit_id(circuit_data)
            current_circuit_ids.add(circuit_id)

            # Initialize if new circuit
            if circuit_id not in self.evolution_data:
                self.evolution_data[circuit_id] = []
                self.strength_history[circuit_id] = []

            # Record current state
            strength = self._extract_strength(circuit_data)
            attribution = self._extract_attribution(circuit_data)

            self.evolution_data[circuit_id].append({
                'epoch': epoch,
                'attribution': attribution,
                'strength': strength,
                'elements': circuit_data.get('elements', 0),
                'connections': circuit_data.get('connections', 0)
            })

            self.strength_history[circuit_id].append((epoch, strength))

            # Track emergence
            if circuit_id not in self.emergence_epochs:
                if attribution > 0.3:  # Emergence threshold
                    self.emergence_epochs[circuit_id] = epoch
                    self._record_evolution_event('birth', circuit_id, epoch, {
                        'initial_strength': strength,
                        'initial_attribution': attribution
                    })

        # Store active circuits for this epoch
        self.epoch_to_circuits[epoch] = current_circuit_ids

        return current_circuit_ids

    def _update_stability_tracking(self, epoch: int, active_circuit_ids: Set[str]) -> Dict[str, List]:
        """Update stability scores and track circuit lifetimes"""
        birth_events = []
        death_events = []

        for circuit_id in active_circuit_ids:
            # Track birth
            if circuit_id not in self.circuit_lifetimes:
                self.circuit_lifetimes[circuit_id] = (epoch, None)
                birth_events.append(circuit_id)

        # Check for deaths (circuits not seen recently)
        for circuit_id, (birth, death) in list(self.circuit_lifetimes.items()):
            if death is None and circuit_id not in active_circuit_ids:
                if self._should_mark_as_dead(circuit_id, epoch):
                    self.circuit_lifetimes[circuit_id] = (birth, epoch)
                    death_events.append(circuit_id)
                    self._record_evolution_event('death', circuit_id, epoch, {
                        'lifetime': epoch - birth,
                        'birth_epoch': birth
                    })

        # Update stability scores for all circuits
        self._update_all_stability_scores(epoch)

        return {
            'birth_events': birth_events,
            'death_events': death_events
        }

    def _update_relationship_tracking(self, epoch: int, active_circuit_ids: Set[str]) -> int:
        """Update circuit relationship tracking"""
        relationships_updated = 0

        # Update co-occurrence relationships
        for cid1 in active_circuit_ids:
            for cid2 in active_circuit_ids:
                if cid1 != cid2:
                    rel_key = (min(cid1, cid2), max(cid1, cid2))

                    if rel_key not in self.circuit_relationships:
                        self.circuit_relationships[rel_key] = {
                            'co_occurrences': 0,
                            'first_co_occurrence': epoch,
                            'last_co_occurrence': epoch,
                            'epochs': [],
                            'correlation_history': []
                        }

                    rel_data = self.circuit_relationships[rel_key]
                    rel_data['co_occurrences'] += 1
                    rel_data['last_co_occurrence'] = epoch
                    rel_data['epochs'].append(epoch)
                    relationships_updated += 1

        return relationships_updated

    # ============================================================================
    # STABILITY AND CONSISTENCY ANALYSIS
    # ============================================================================

    def get_stability_score(self, circuit_id: str, current_epoch: int) -> float:
        """
        Calculate stability score for a circuit

        Args:
            circuit_id: Circuit identifier
            current_epoch: Current training epoch

        Returns:
            Stability score (0.0 to 1.0)
        """
        if circuit_id not in self.circuit_lifetimes:
            return 0.0

        birth_epoch, death_epoch = self.circuit_lifetimes[circuit_id]
        end_epoch = death_epoch if death_epoch is not None else current_epoch

        # Base lifetime score
        lifetime = end_epoch - birth_epoch
        lifetime_score = min(1.0, lifetime / 100.0)  # Normalize to 100 epochs

        # Consistency score (how often circuit appears when analysis is run)
        if circuit_id in self.evolution_data:
            appearances = len(self.evolution_data[circuit_id])
            possible_appearances = lifetime + 1
            consistency_score = appearances / max(1, possible_appearances)
        else:
            consistency_score = 0.0

        # Recency score (has it been seen recently?)
        recency_score = 1.0
        if death_epoch is not None:
            epochs_since_death = current_epoch - death_epoch
            recency_score = max(0.0, 1.0 - epochs_since_death / 50.0)

        # Strength stability (variance in strength over time)
        strength_stability = self._calculate_strength_stability(circuit_id)

        # Combined stability score
        stability = (
                lifetime_score * 0.3 +
                consistency_score * 0.25 +
                recency_score * 0.25 +
                strength_stability * 0.2
        )

        self.stability_scores[circuit_id] = stability
        return stability

    def get_consistency_score(self, circuit_id: str) -> float:
        """Calculate consistency score for a circuit"""
        if circuit_id not in self.evolution_data or not self.evolution_data[circuit_id]:
            return 0.0

        # Consistency based on strength variance
        strengths = [data['strength'] for data in self.evolution_data[circuit_id]]
        if len(strengths) < 2:
            return 0.5  # Neutral for single observation

        mean_strength = np.mean(strengths)
        if mean_strength == 0:
            return 0.0

        cv = np.std(strengths) / mean_strength  # Coefficient of variation
        consistency = max(0.0, 1.0 - cv)  # Lower variance = higher consistency

        self.consistency_scores[circuit_id] = consistency
        return consistency

    def get_stable_circuits(self, current_epoch: int, min_stability: float = 0.6) -> List[Dict[str, Any]]:
        """Get circuits that are considered stable"""
        stable_circuits = []

        for circuit_id in self.circuit_lifetimes:
            stability = self.get_stability_score(circuit_id, current_epoch)
            consistency = self.get_consistency_score(circuit_id)

            if stability >= min_stability:
                birth_epoch, death_epoch = self.circuit_lifetimes[circuit_id]

                stable_circuits.append({
                    'circuit_id': circuit_id,
                    'stability_score': stability,
                    'consistency_score': consistency,
                    'birth_epoch': birth_epoch,
                    'death_epoch': death_epoch,
                    'lifetime': (death_epoch or current_epoch) - birth_epoch,
                    'current_strength': self._get_current_strength(circuit_id),
                    'is_active': death_epoch is None
                })

        # Sort by stability score
        stable_circuits.sort(key=lambda x: x['stability_score'], reverse=True)
        return stable_circuits

    # ============================================================================
    # RELATIONSHIP AND EMERGENCE ANALYSIS
    # ============================================================================

    def analyze_emergence_order(self) -> Dict[str, Any]:
        """Analyze which types of circuits emerge first"""
        circuits_by_type = defaultdict(list)

        for circuit_id, emergence_epoch in self.emergence_epochs.items():
            circuit = self.registry.get_circuit(circuit_id)
            if circuit and hasattr(circuit, 'type'):
                circuit_type = circuit.type.value
                circuits_by_type[circuit_type].append((circuit_id, emergence_epoch))

        # Calculate statistics by type
        type_stats = {}
        for circuit_type, circuits in circuits_by_type.items():
            if circuits:
                epochs = [epoch for _, epoch in circuits]
                type_stats[circuit_type] = {
                    'count': len(circuits),
                    'avg_emergence': np.mean(epochs),
                    'median_emergence': np.median(epochs),
                    'earliest_emergence': min(epochs),
                    'latest_emergence': max(epochs),
                    'std_emergence': np.std(epochs)
                }

        # Determine emergence order
        emergence_order = sorted(type_stats.keys(),
                                 key=lambda t: type_stats[t]['avg_emergence'])

        return {
            'circuits_by_type': dict(circuits_by_type),
            'type_statistics': type_stats,
            'emergence_order': emergence_order,
            'total_emerged': len(self.emergence_epochs)
        }

    def analyze_circuit_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between circuits"""
        # Precedence relationships (based on emergence order)
        precedence_relationships = []
        circuit_ids = list(self.emergence_epochs.keys())

        for i, circuit_id1 in enumerate(circuit_ids):
            for circuit_id2 in circuit_ids[i + 1:]:
                epoch1 = self.emergence_epochs.get(circuit_id1)
                epoch2 = self.emergence_epochs.get(circuit_id2)

                if epoch1 is not None and epoch2 is not None and abs(epoch1 - epoch2) > 10:
                    precedes = circuit_id1 if epoch1 < epoch2 else circuit_id2
                    follows = circuit_id2 if epoch1 < epoch2 else circuit_id1

                    precedence_relationships.append({
                        'type': 'precedence',
                        'precedes': precedes,
                        'follows': follows,
                        'epoch_diff': abs(epoch1 - epoch2)
                    })

        # Co-occurrence relationships
        co_occurrence = []
        for (cid1, cid2), rel_data in self.circuit_relationships.items():
            if rel_data['co_occurrences'] >= 3:
                strength = rel_data['co_occurrences'] / (
                        rel_data['last_co_occurrence'] - rel_data['first_co_occurrence'] + 1
                )

                co_occurrence.append({
                    'circuit_pair': (cid1, cid2),
                    'co_occurrences': rel_data['co_occurrences'],
                    'strength': strength,
                    'duration': rel_data['last_co_occurrence'] - rel_data['first_co_occurrence'],
                    'first_co_occurrence': rel_data['first_co_occurrence'],
                    'last_co_occurrence': rel_data['last_co_occurrence']
                })

        co_occurrence.sort(key=lambda x: x['strength'], reverse=True)

        return {
            'precedence': precedence_relationships,
            'co_occurrence': co_occurrence,
            'total_relationships': len(self.circuit_relationships)
        }

    # ============================================================================
    # EVOLUTION SUMMARY AND REPORTING
    # ============================================================================

    def get_evolution_summary(self, current_epoch: int) -> Dict[str, Any]:
        """Get comprehensive evolution summary"""
        birth_events = [e for e in self.evolution_events if e['type'] == 'birth']
        death_events = [e for e in self.evolution_events if e['type'] == 'death']

        stable_circuits = self.get_stable_circuits(current_epoch)

        # Calculate survival statistics
        total_circuits = len(self.circuit_lifetimes)
        living_circuits = sum(1 for (_, death) in self.circuit_lifetimes.values() if death is None)

        return {
            'current_epoch': current_epoch,
            'total_circuits_discovered': total_circuits,
            'living_circuits': living_circuits,
            'dead_circuits': total_circuits - living_circuits,
            'stable_circuits': len(stable_circuits),
            'survival_rate': living_circuits / max(1, total_circuits),
            'birth_events': len(birth_events),
            'death_events': len(death_events),
            'avg_lifetime': self._calculate_average_lifetime(),
            'emergence_phases': self._analyze_emergence_phases(current_epoch),
            'relationship_count': len(self.circuit_relationships)
        }

    # ============================================================================
    # UTILITY AND HELPER METHODS
    # ============================================================================

    def _extract_circuit_id(self, circuit_data: Dict) -> str:
        """Extract circuit ID from circuit data"""
        if isinstance(circuit_data, dict):
            return circuit_data.get('id', circuit_data.get('circuit_id', f"circuit_{hash(str(circuit_data))}"))
        elif hasattr(circuit_data, 'id'):
            return circuit_data.id
        else:
            return f"circuit_{hash(str(circuit_data))}"

    def _extract_strength(self, circuit_data: Dict) -> float:
        """Extract strength from circuit data"""
        if isinstance(circuit_data, dict):
            return circuit_data.get('strength', circuit_data.get('attribution', 0.5))
        elif hasattr(circuit_data, 'attribution'):
            return circuit_data.attribution
        else:
            return 0.5

    def _extract_attribution(self, circuit_data: Dict) -> float:
        """Extract attribution from circuit data"""
        if isinstance(circuit_data, dict):
            return circuit_data.get('attribution', circuit_data.get('strength', 0.5))
        elif hasattr(circuit_data, 'attribution'):
            return circuit_data.attribution
        else:
            return 0.5

    def _should_mark_as_dead(self, circuit_id: str, current_epoch: int) -> bool:
        """Determine if circuit should be marked as dead"""
        birth_epoch, _ = self.circuit_lifetimes[circuit_id]
        epochs_since_birth = current_epoch - birth_epoch

        # Don't mark as dead too early
        if epochs_since_birth < 20:
            return False

        # Mark as dead if not seen for 30+ epochs after sufficient lifetime
        return epochs_since_birth > 50

    def _record_evolution_event(self, event_type: str, circuit_id: str, epoch: int, details: Dict):
        """Record an evolution event"""
        self.evolution_events.append({
            'type': event_type,
            'circuit_id': circuit_id,
            'epoch': epoch,
            'details': details
        })

    def _update_all_stability_scores(self, epoch: int):
        """Update stability scores for all circuits"""
        for circuit_id in self.circuit_lifetimes:
            self.get_stability_score(circuit_id, epoch)

    def _calculate_strength_stability(self, circuit_id: str) -> float:
        """Calculate how stable the strength has been over time"""
        if circuit_id not in self.strength_history or len(self.strength_history[circuit_id]) < 2:
            return 0.5

        strengths = [strength for _, strength in self.strength_history[circuit_id]]

        if len(strengths) < 2:
            return 0.5

        # Use coefficient of variation (normalized standard deviation)
        mean_strength = np.mean(strengths)
        if mean_strength == 0:
            return 0.0

        cv = np.std(strengths) / mean_strength
        return max(0.0, 1.0 - cv)

    def _get_current_strength(self, circuit_id: str) -> float:
        """Get most recent strength for circuit"""
        if circuit_id not in self.strength_history or not self.strength_history[circuit_id]:
            return 0.0
        return self.strength_history[circuit_id][-1][1]

    def _calculate_average_lifetime(self) -> float:
        """Calculate average lifetime of dead circuits"""
        dead_lifetimes = []
        for birth, death in self.circuit_lifetimes.values():
            if death is not None:
                dead_lifetimes.append(death - birth)

        return np.mean(dead_lifetimes) if dead_lifetimes else 0.0

    def _analyze_emergence_phases(self, current_epoch: int) -> Dict[str, int]:
        """Analyze emergence patterns by training phase"""
        phases = {
            'early': sum(1 for e in self.emergence_epochs.values() if e < current_epoch * 0.2),
            'middle': sum(1 for e in self.emergence_epochs.values() if current_epoch * 0.2 <= e < current_epoch * 0.7),
            'late': sum(1 for e in self.emergence_epochs.values() if e >= current_epoch * 0.7)
        }
        return phases

    def save_figure_safe(self, fig, filename, save_dir=None, **kwargs):
        """Save figure ensuring directory exists with sensible defaults"""
        if save_dir is None:
            save_dir = self.save_dir
        if save_dir is None:
            return None

        save_path = Path(save_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        default_kwargs = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        default_kwargs.update(kwargs)

        fig.savefig(save_path, **default_kwargs)
        return save_path

    # ============================================================================
    # INTEGRATION METHODS
    # ============================================================================

    def update_from_registry(self, epoch: int):
        """Update tracking from registry state"""
        if not self.registry or not hasattr(self.registry, 'circuits'):
            return

        circuits = list(self.registry.circuits.values())
        circuit_data = []

        for circuit in circuits:
            circuit_data.append({
                'id': circuit.id,
                'attribution': circuit.attribution,
                'strength': circuit.attribution,
                'elements': len(circuit.elements) if hasattr(circuit, 'elements') else 0,
                'connections': len(circuit.connections) if hasattr(circuit, 'connections') else 0
            })

        return self.track_epoch(epoch, circuit_data)

    def get_circuit_stability_for_detector(self, circuit_id: str, current_epoch: int) -> Dict[str, float]:
        """Interface for adaptive detectors to get circuit stability info"""
        return {
            'stability_score': self.get_stability_score(circuit_id, current_epoch),
            'consistency_score': self.get_consistency_score(circuit_id),
            'current_strength': self._get_current_strength(circuit_id),
            'lifetime': self._get_circuit_lifetime(circuit_id, current_epoch)
        }

    def _get_circuit_lifetime(self, circuit_id: str, current_epoch: int) -> int:
        """Get circuit lifetime"""
        if circuit_id not in self.circuit_lifetimes:
            return 0
        birth, death = self.circuit_lifetimes[circuit_id]
        return (death or current_epoch) - birth