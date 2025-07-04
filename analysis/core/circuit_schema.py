# analysis/core/circuit_schema.py
"""
Fixed circuit schema with all missing components for Phase 2 transition

Addresses:
1. Missing EmergencePhase fields (MATURE, DEVELOPING)
2. Missing EvolutionPattern fields (GRADUAL_EMERGENCE, SUDDEN_EMERGENCE, PLATEAUING, DECLINING)
3. Missing EvolutionSnapshot.active_interactions field
4. Missing LearningPhaseTransition class
5. Missing analyzer classes (CircuitLevel, GrokkingPhase, TemporalMetrics, CircuitDependencies)
"""

from dataclasses import dataclass, field
from typing import Set, Tuple, Dict, List, Optional, Union, Any
import json
from enum import Enum
from pathlib import Path
from analysis.utils.utils import CircuitJSONEncoder, get_current_callable_info


# ============================================================================
# ENUMS - EXTENDED FOR PHASE 2
# ============================================================================

class CircuitType(Enum):
    TOKEN = "token"
    COMPONENT = "component"
    FUNCTIONAL = "functional"
    HYBRID = "hybrid"
    SUBSPACE = "subspace"


class CircuitInteractionType(Enum):
    """Types of circuit interactions"""
    PREREQUISITE = "prerequisite"
    ENABLES = "enables"
    COMPETES = "competes"
    COOPERATES = "cooperates"
    REPLACES = "replaces"
    COMPOSES = "composes"


class EmergencePattern(Enum):
    """Patterns of circuit emergence"""
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    CASCADING = "cascading"
    OSCILLATING = "oscillating"
    REINFORCING = "reinforcing"


class ElementType(Enum):
    TOKEN = "token"
    HEAD = "head"
    MLP = "mlp"
    SUBSPACE = "subspace"
    POSITION = "position"
    LAYER = "layer"


class ConnectionType(Enum):
    ATTENTION = "attention"
    RESIDUAL = "residual"
    MLP = "mlp"
    COMPOSITE = "composite"


class EmergencePhase(Enum):
    """Circuit emergence phases during training - FIXED"""
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"
    GROKKING = "grokking"
    POST_GROKKING = "post_grokking"
    # NEW: Missing fields for CircuitEvolutionTracker
    MATURE = "mature"
    DEVELOPING = "developing"


class CircuitStability(Enum):
    """Circuit stability classifications"""
    TRANSIENT = "transient"
    EMERGING = "emerging"
    STABLE = "stable"
    PERSISTENT = "persistent"
    DECLINING = "declining"
    DEFUNCT = "defunct"


class RelationshipType(Enum):
    """Types of relationships between circuits"""
    PREREQUISITE = "prerequisite"
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"
    COMPOSITIONAL = "compositional"
    ALTERNATIVE = "alternative"
    SUPER_ADDITIVE = "super_additive"


class LearningPhase(Enum):
    """Task-agnostic learning phases"""
    EARLY_LEARNING = "early_learning"
    MEMORIZATION = "memorization"
    TRANSITION = "transition"
    GENERALIZATION = "generalization"
    CONSOLIDATION = "consolidation"


class InteractionType(Enum):
    """Types of circuit interactions"""
    ENABLES = "enables"
    COMPETES = "competes"
    COOPERATES = "cooperates"
    REPLACES = "replaces"
    REINFORCES = "reinforces"


class EvolutionPattern(Enum):
    """Patterns of circuit evolution - FIXED"""
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    OSCILLATING = "oscillating"
    CASCADING = "cascading"
    # NEW: Missing fields for CircuitEvolutionTracker
    GRADUAL_EMERGENCE = "gradual_emergence"
    SUDDEN_EMERGENCE = "sudden_emergence"
    PLATEAUING = "plateauing"
    DECLINING = "declining"


# NEW: Missing enums for analyzer classes
class CircuitLevel(Enum):
    """Levels of circuit analysis"""
    TOKEN = "token"
    COMPONENT = "component"
    SUBSPACE = "subspace"
    LAYER = "layer"
    GLOBAL = "global"


class GrokkingPhase(Enum):
    """Phases of grokking process"""
    PRE_GROKKING = "pre_grokking"
    GROKKING_ONSET = "grokking_onset"
    GROKKING_TRANSITION = "grokking_transition"
    POST_GROKKING = "post_grokking"
    CONSOLIDATION = "consolidation"


# ============================================================================
# CORE DATACLASSES
# ============================================================================

@dataclass
class Element:
    """A single element in a circuit (token, head, etc.)"""
    id: str
    type: ElementType
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "properties": self.properties
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Element':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            type=ElementType(data["type"]),
            properties=data.get("properties", {})
        )


@dataclass
class Connection:
    """A connection between two elements in a circuit"""
    source: str
    target: str
    strength: float
    type: ConnectionType
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "source": self.source,
            "target": self.target,
            "strength": self.strength,
            "type": self.type.value,
            "properties": self.properties
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Connection':
        """Create from dictionary"""
        return cls(
            source=data["source"],
            target=data["target"],
            strength=data["strength"],
            type=ConnectionType(data["type"]),
            properties=data.get("properties", {})
        )


# ============================================================================
# TEMPORAL/EVOLUTION DATACLASSES - FIXED
# ============================================================================

@dataclass
class InteractionEvent:
    """Records specific interaction between circuits"""
    epoch: int
    source_circuit: str
    target_circuit: str
    interaction_type: InteractionType
    strength: float
    confidence: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionSnapshot:
    """Circuit state at specific epoch - FIXED"""
    epoch: int
    attribution: float
    learning_phase: LearningPhase
    detection_confidence: float = 0.5
    stability_score: float = 0.0
    behavioral_impact: float = 0.0
    # NEW: Missing field for CircuitEvolutionTracker
    active_interactions: List[str] = field(default_factory=list)
    context_metadata: Dict[str, Any] = field(default_factory=dict)


# NEW: Missing class for CircuitEvolutionAnalyzer
@dataclass
class LearningPhaseTransition:
    """Represents a transition between learning phases"""
    from_phase: LearningPhase
    to_phase: LearningPhase
    transition_epoch: int
    circuits_affected: List[str] = field(default_factory=list)
    interaction_changes: Dict[str, Any] = field(default_factory=dict)
    transition_strength: float = 0.0
    duration: int = 0  # epochs
    metadata: Dict[str, Any] = field(default_factory=dict)


# NEW: Missing analyzer classes
@dataclass
class TemporalMetrics:
    """Metrics for temporal analysis"""
    emergence_rate: float = 0.0
    stability_trend: float = 0.0
    interaction_intensity: float = 0.0
    phase_transition_frequency: float = 0.0
    circuit_lifetime: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitDependencies:
    """Circuit dependency relationships"""
    circuit_id: str
    prerequisites: List[str] = field(default_factory=list)
    dependent_circuits: List[str] = field(default_factory=list)
    dependency_strength: Dict[str, float] = field(default_factory=dict)
    dependency_type: Dict[str, str] = field(default_factory=dict)
    temporal_dependencies: List[Tuple[str, int]] = field(default_factory=list)  # (circuit_id, epoch)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CIRCUIT METADATA - ENHANCED
# ============================================================================

@dataclass
class CircuitMetadata:
    """Enhanced metadata for circuit tracking"""
    # Temporal information
    first_detected: int = 0
    last_seen: int = 0
    detection_epochs: List[int] = field(default_factory=list)
    emergence_phase: EmergencePhase = EmergencePhase.EARLY
    stability: CircuitStability = CircuitStability.TRANSIENT

    # Detection context
    detection_method: str = "unknown"
    detection_threshold: float = 0.5
    detection_confidence: float = 0.5
    content_aware: bool = False

    # Reliability metrics
    stability_score: float = 0.0
    false_positive_risk: float = 0.5
    consistency_score: float = 0.0
    behavioral_impact: float = 0.0

    # Evolution tracking
    strength_history: List[Tuple[int, float]] = field(default_factory=list)
    threshold_history: List[Tuple[int, float]] = field(default_factory=list)

    # Relationships
    prerequisite_circuits: Set[str] = field(default_factory=set)
    enables_circuits: Set[str] = field(default_factory=set)
    competes_with: Set[str] = field(default_factory=set)
    cooperates_with: Set[str] = field(default_factory=set)

    # Validation results
    manipulation_effects: Dict[str, float] = field(default_factory=dict)
    cross_method_consistency: Dict[str, float] = field(default_factory=dict)

    # Evolution recording
    evolution_snapshots: List[EvolutionSnapshot] = field(default_factory=list)
    interaction_events: List[InteractionEvent] = field(default_factory=list)
    learning_phases: Dict[int, LearningPhase] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "first_detected": self.first_detected,
            "last_seen": self.last_seen,
            "detection_epochs": self.detection_epochs,
            "emergence_phase": self.emergence_phase.value,
            "stability": self.stability.value,
            "detection_method": self.detection_method,
            "detection_threshold": self.detection_threshold,
            "detection_confidence": self.detection_confidence,
            "content_aware": self.content_aware,
            "stability_score": self.stability_score,
            "false_positive_risk": self.false_positive_risk,
            "consistency_score": self.consistency_score,
            "behavioral_impact": self.behavioral_impact,
            "strength_history": self.strength_history,
            "threshold_history": self.threshold_history,
            "prerequisite_circuits": list(self.prerequisite_circuits),
            "enables_circuits": list(self.enables_circuits),
            "competes_with": list(self.competes_with),
            "cooperates_with": list(self.cooperates_with),
            "manipulation_effects": self.manipulation_effects,
            "cross_method_consistency": self.cross_method_consistency,
            "evolution_snapshots": [
                {
                    "epoch": s.epoch,
                    "attribution": s.attribution,
                    "learning_phase": s.learning_phase.value,
                    "detection_confidence": s.detection_confidence,
                    "stability_score": s.stability_score,
                    "behavioral_impact": s.behavioral_impact,
                    "active_interactions": s.active_interactions,
                    "context_metadata": s.context_metadata
                }
                for s in self.evolution_snapshots
            ],
            "interaction_events": [
                {
                    "epoch": e.epoch,
                    "source_circuit": e.source_circuit,
                    "target_circuit": e.target_circuit,
                    "interaction_type": e.interaction_type.value,
                    "strength": e.strength,
                    "confidence": e.confidence,
                    "context": e.context
                }
                for e in self.interaction_events
            ],
            "learning_phases": {k: v.value for k, v in self.learning_phases.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CircuitMetadata':
        """Create from dictionary"""
        metadata = cls()
        metadata.first_detected = data.get("first_detected", 0)
        metadata.last_seen = data.get("last_seen", 0)
        metadata.detection_epochs = data.get("detection_epochs", [])
        metadata.emergence_phase = EmergencePhase(data.get("emergence_phase", "early"))
        metadata.stability = CircuitStability(data.get("stability", "transient"))
        metadata.detection_method = data.get("detection_method", "unknown")
        metadata.detection_threshold = data.get("detection_threshold", 0.5)
        metadata.detection_confidence = data.get("detection_confidence", 0.5)
        metadata.content_aware = data.get("content_aware", False)
        metadata.stability_score = data.get("stability_score", 0.0)
        metadata.false_positive_risk = data.get("false_positive_risk", 0.5)
        metadata.consistency_score = data.get("consistency_score", 0.0)
        metadata.behavioral_impact = data.get("behavioral_impact", 0.0)
        metadata.strength_history = data.get("strength_history", [])
        metadata.threshold_history = data.get("threshold_history", [])
        metadata.prerequisite_circuits = set(data.get("prerequisite_circuits", []))
        metadata.enables_circuits = set(data.get("enables_circuits", []))
        metadata.competes_with = set(data.get("competes_with", []))
        metadata.cooperates_with = set(data.get("cooperates_with", []))
        metadata.manipulation_effects = data.get("manipulation_effects", {})
        metadata.cross_method_consistency = data.get("cross_method_consistency", {})

        # Restore evolution snapshots
        for s_data in data.get("evolution_snapshots", []):
            snapshot = EvolutionSnapshot(
                epoch=s_data["epoch"],
                attribution=s_data["attribution"],
                learning_phase=LearningPhase(s_data["learning_phase"]),
                detection_confidence=s_data.get("detection_confidence", 0.5),
                stability_score=s_data.get("stability_score", 0.0),
                behavioral_impact=s_data.get("behavioral_impact", 0.0),
                active_interactions=s_data.get("active_interactions", []),
                context_metadata=s_data.get("context_metadata", {})
            )
            metadata.evolution_snapshots.append(snapshot)

        # Restore interaction events
        for e_data in data.get("interaction_events", []):
            event = InteractionEvent(
                epoch=e_data["epoch"],
                source_circuit=e_data["source_circuit"],
                target_circuit=e_data["target_circuit"],
                interaction_type=InteractionType(e_data["interaction_type"]),
                strength=e_data["strength"],
                confidence=e_data.get("confidence", 0.5),
                context=e_data.get("context", {})
            )
            metadata.interaction_events.append(event)

        # Restore learning phases
        metadata.learning_phases = {
            int(k): LearningPhase(v) for k, v in data.get("learning_phases", {}).items()
        }

        return metadata


# ============================================================================
# CIRCUIT CLASS - FIXED VERSION
# ============================================================================

@dataclass
class Circuit:
    """A circuit representing a functional unit in the model"""
    id: str
    type: CircuitType
    elements: List[Element] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    attribution: float = 0.0
    metadata: Union[Dict[str, Any], CircuitMetadata] = field(default_factory=dict)
    discovered_at: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        if isinstance(self.metadata, CircuitMetadata):
            metadata_dict = self.metadata.to_dict()
        else:
            metadata_dict = self.metadata

        return {
            "id": self.id,
            "type": self.type.value,
            "elements": [e.to_dict() for e in self.elements],
            "connections": [c.to_dict() for c in self.connections],
            "attribution": self.attribution,
            "metadata": metadata_dict,
            "discovered_at": self.discovered_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Circuit':
        """Create from dictionary"""
        elements = [Element.from_dict(e) for e in data.get("elements", [])]
        connections = [Connection.from_dict(c) for c in data.get("connections", [])]

        metadata = data.get("metadata", {})
        if isinstance(metadata, dict) and "emergence_phase" in metadata:
            metadata = CircuitMetadata.from_dict(metadata)

        return cls(
            id=data["id"],
            type=CircuitType(data["type"]),
            elements=elements,
            connections=connections,
            attribution=data.get("attribution", 0.0),
            metadata=metadata,
            discovered_at=data.get("discovered_at")
        )


# ============================================================================
# SAVE/LOAD FUNCTIONS (Fixed versions)
# ============================================================================

def save_circuits(circuits: List[Circuit], filepath: Union[str, Path]) -> None:
    """Save circuits to a JSON file using custom encoder"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, 'w') as f:
            json.dump([circuit.to_dict() for circuit in circuits], f,
                      cls=CircuitJSONEncoder, indent=2)
        print(f"✅ Saved {len(circuits)} circuits to {filepath}")
    except Exception as e:
        print(f"⚠️ Error saving circuits with custom encoder: {e}")
        # Fallback to basic JSON
        with open(filepath, 'w') as f:
            json.dump([circuit.to_dict() for circuit in circuits], f, indent=2)
        print(f"✅ Saved {len(circuits)} circuits to {filepath} (fallback)")


def load_circuits(filepath: Union[str, Path]) -> List[Circuit]:
    """Load circuits from a JSON file"""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"⚠️ Circuit file {filepath} does not exist")
        return []

    try:
        with open(filepath, 'r') as f:
            circuits_data = json.load(f)

        circuits = [Circuit.from_dict(circuit_data) for circuit_data in circuits_data]
        print(f"✅ Loaded {len(circuits)} circuits from {filepath}")
        return circuits
    except Exception as e:
        print(f"❌ Error loading circuits from {filepath}: {e}")
        return []


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_empty_circuit_metadata() -> CircuitMetadata:
    """Create empty CircuitMetadata with proper defaults"""
    return CircuitMetadata()


def convert_dict_to_circuit_metadata(metadata_dict: Dict[str, Any]) -> CircuitMetadata:
    """Convert dictionary to CircuitMetadata object"""
    if not metadata_dict:
        return create_empty_circuit_metadata()
    return CircuitMetadata.from_dict(metadata_dict)


def ensure_circuit_has_proper_metadata(circuit: Circuit) -> Circuit:
    """Ensure circuit has proper CircuitMetadata object"""
    if not isinstance(circuit.metadata, CircuitMetadata):
        circuit.metadata = convert_dict_to_circuit_metadata(circuit.metadata)
    return circuit


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_circuit_schema():
    """Validate that all required components are present"""
    required_enums = [
        EmergencePhase.MATURE,
        EmergencePhase.DEVELOPING,
        EvolutionPattern.GRADUAL_EMERGENCE,
        EvolutionPattern.SUDDEN_EMERGENCE,
        EvolutionPattern.PLATEAUING,
        EvolutionPattern.DECLINING
    ]

    required_classes = [
        LearningPhaseTransition,
        TemporalMetrics,
        CircuitDependencies,
        CircuitLevel,
        GrokkingPhase
    ]

    print("✅ All required enum values present")
    print("✅ All required classes defined")
    print("✅ Schema ready for Phase 2 transition")

    return True


if __name__ == "__main__":
    validate_circuit_schema()