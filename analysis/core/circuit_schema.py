# circuit_schema.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
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
