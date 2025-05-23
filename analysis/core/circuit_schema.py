# circuit_schema.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import json
from enum import Enum
from pathlib import Path


class CircuitType(Enum):
    TOKEN = "token"
    COMPONENT = "component"
    FUNCTIONAL = "functional"
    HYBRID = "hybrid"


class ElementType(Enum):
    TOKEN = "token"
    HEAD = "head"
    MLP = "mlp"
    SUBSPACE = "subspace"
    POSITION = "position"


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
    """Save circuits to a JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump({
            "schema_version": "1.0",
            "circuits": [c.to_dict() for c in circuits]
        }, f, indent=2)


def load_circuits(filepath: Union[str, Path]) -> List[Circuit]:
    """Load circuits from a JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    schema_version = data.get("schema_version", "1.0")
    # Version handling logic could go here

    return [Circuit.from_dict(c) for c in data["circuits"]]