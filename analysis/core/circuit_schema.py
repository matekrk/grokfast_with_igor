# circuit_schema.py
from dataclasses import dataclass, field
from typing import Set, Tuple, Dict, List, Optional, Union, Any
import json
from enum import Enum
from pathlib import Path
from analysis.utils.utils import CircuitJSONEncoder, get_current_callable_info



class CircuitType(Enum):
    TOKEN = "token"
    COMPONENT = "component"
    FUNCTIONAL = "functional"
    HYBRID = "hybrid"
    SUBSPACE = "subspace"   # ✅ Add for MLP subspace circuits

class ElementType(Enum):
    TOKEN = "token"
    HEAD = "head"
    MLP = "mlp"
    SUBSPACE = "subspace"  # ✅ Already there
    POSITION = "position"
    LAYER = "layer"        # ✅ Add for layer-level elements

class ConnectionType(Enum):
    ATTENTION = "attention"
    RESIDUAL = "residual"
    MLP = "mlp"
    COMPOSITE = "composite"

class EmergencePhase(Enum):
    """Circuit emergence phases during training"""
    EARLY = "early"          # 0-100 epochs, often noisy
    MIDDLE = "middle"        # 100-500 epochs, genuine learning
    LATE = "late"           # 500+ epochs, refined mechanisms
    GROKKING = "grokking"   # During phase transitions
    POST_GROKKING = "post_grokking"  # After stabilization


class CircuitStability(Enum):
    """Circuit stability classifications"""
    TRANSIENT = "transient"     # Seen <3 times
    EMERGING = "emerging"       # Recent appearance, increasing
    STABLE = "stable"          # Consistent presence
    PERSISTENT = "persistent"   # Long-term presence
    DECLINING = "declining"     # Decreasing presence
    DEFUNCT = "defunct"        # No longer present


class RelationshipType(Enum):
    """Types of relationships between circuits"""
    PREREQUISITE = "prerequisite"       # A enables B
    COMPETITIVE = "competitive"         # A competes with B
    COOPERATIVE = "cooperative"         # A works with B
    COMPOSITIONAL = "compositional"     # A is part of B
    ALTERNATIVE = "alternative"         # A can replace B
    SUPER_ADDITIVE = "super_additive"   # A+B > A+B individually



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
    source: str  # Element id
    target: str  # Element id
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


@dataclass
class Circuit:
    """A circuit representing a functional unit in the model"""
    id: str
    type: CircuitType
    elements: List[Element] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    attribution: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: Optional[int] = None  # Epoch when discovered

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "elements": [e.to_dict() for e in self.elements],
            "connections": [c.to_dict() for c in self.connections],
            "attribution": self.attribution,
            "metadata": self.metadata,
            "discovered_at": self.discovered_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Circuit':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            type=CircuitType(data["type"]),
            elements=[Element.from_dict(e) for e in data["elements"]],
            connections=[Connection.from_dict(c) for c in data["connections"]],
            attribution=data["attribution"],
            metadata=data.get("metadata", {}),
            discovered_at=data.get("discovered_at")
        )


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
            "cross_method_consistency": self.cross_method_consistency
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
        return metadata



def save_circuits(circuits: List[Circuit], filepath: Union[str, Path]) -> None:
    """Save circuits to a JSON file using custom encoder"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # ✅ UPDATED: Use CircuitJSONEncoder for robust serialization
    try:
        with open(filepath, 'w') as f:
            json.dump({
                "schema_version": "1.0",
                "circuits": [c.to_dict() for c in circuits]
            }, f, cls=CircuitJSONEncoder, indent=2)  # ✅ Use custom encoder

        print(f"\t{get_current_callable_info()}:\t✅ Saved {len(circuits)} circuits to {filepath}")

    except Exception as e:
        print(f"\t{get_current_callable_info()}:\t⚠️ Error saving circuits with CircuitJSONEncoder: {e}")
        # Fallback to basic JSON with data cleaning
        _save_circuits_fallback(circuits, filepath)


def _save_circuits_fallback(circuits: List[Circuit], filepath: Union[str, Path]):
    """Fallback saving with data cleaning"""
    try:
        # Clean circuit data for basic JSON
        cleaned_circuits = []
        for circuit in circuits:
            circuit_dict = circuit.to_dict()
            cleaned_dict = _clean_for_basic_json(circuit_dict)
            cleaned_circuits.append(cleaned_dict)

        with open(filepath, 'w') as f:
            json.dump({
                "schema_version": "1.0",
                "circuits": cleaned_circuits,
                "note": "Saved with fallback cleaning - some data may be simplified"
            }, f, indent=2)

        print(f"\t{get_current_callable_info()}:\t📝 Saved {len(circuits)} circuits to {filepath} (with fallback cleaning)")

    except Exception as e:
        print(f"\t{get_current_callable_info()}:\t❌ Failed to save circuits even with fallback: {e}")

def _clean_for_basic_json(obj):
    """Clean object for basic JSON serialization"""
    if isinstance(obj, dict):
        return {k: _clean_for_basic_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_for_basic_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [_clean_for_basic_json(item) for item in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif hasattr(obj, '__dict__'):
        return _clean_for_basic_json(obj.__dict__)
    else:
        return str(obj)  # Convert to string as fallback

def load_circuits(filepath: Union[str, Path]) -> List[Circuit]:
    """Load circuits from a JSON file with object reconstruction"""

    # ✅ UPDATED: Import load function for reconstruction
    from analysis.utils.utils import load_circuit_logs

    try:
        # Try loading with reconstruction first
        data = load_circuit_logs(filepath)

        schema_version = data.get("schema_version", "1.0")
        circuits_data = data.get("circuits", [])

        # Convert back to Circuit objects
        circuits = [Circuit.from_dict(c) for c in circuits_data]

        print(f"\t{get_current_callable_info()}:\t✅ Loaded {len(circuits)} circuits from {filepath}")
        return circuits

    except Exception as e:
        print(f"\t{get_current_callable_info()}:\t⚠️ Error loading with reconstruction: {e}")
        # Fallback to basic JSON loading
        return _load_circuits_basic(filepath)


def _load_circuits_basic(filepath: Union[str, Path]) -> List[Circuit]:
    """Fallback loading with basic JSON"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        schema_version = data.get("schema_version", "1.0")
        circuits_data = data.get("circuits", [])

        circuits = [Circuit.from_dict(c) for c in circuits_data]

        print(f"\t{get_current_callable_info()}:\t📝 Loaded {len(circuits)} circuits from {filepath} (basic JSON)")
        return circuits

    except Exception as e:
        print(f"\t{get_current_callable_info()}:\t❌ Failed to load circuits: {e}")
        return []
