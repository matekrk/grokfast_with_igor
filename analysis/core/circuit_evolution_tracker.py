# analysis/core/circuit_evolution_tracker.py
# fixme move to circuit_evolution_tracker after unification

from typing import Dict, List, Optional, Any

# Import existing schema
from analysis.core.circuit_schema import (
    CircuitMetadata, EmergencePhase, LearningPhase, InteractionType, InteractionEvent,
    EvolutionSnapshot, EvolutionPattern
)


class CircuitEvolutionTracker:
    """
    Extends existing CircuitMetadata with comprehensive evolution tracking
    Integrates seamlessly with existing circuit_schema.py structure
    """

    def __init__(self, circuit_metadata: CircuitMetadata):
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
