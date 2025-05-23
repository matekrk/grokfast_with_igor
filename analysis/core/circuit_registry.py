# circuit_registry.py
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import json

from analysis.core.circuit_schema import Circuit, CircuitType, save_circuits, load_circuits


class CircuitRegistry:
    """Central registry for all discovered circuits"""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.circuits: Dict[str, Circuit] = {}
        self.sources: Dict[str, str] = {}  # Circuit ID to source mapping
        self.related_circuits: Dict[str, Set[str]] = {}  # Circuit ID to related circuit IDs
        self.storage_dir = storage_dir

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def register_circuit(self, circuit: Circuit, source: str = "unknown") -> None:
        """Add a circuit to the registry with its source"""
        if circuit.id in self.circuits:
            print(f"Warning: Circuit with ID {circuit.id} already exists and will be overwritten.")

        self.circuits[circuit.id] = circuit
        self.sources[circuit.id] = source

        # Initialize related circuits set if needed
        if circuit.id not in self.related_circuits:
            self.related_circuits[circuit.id] = set()

    def register_relation(self, circuit_id1: str, circuit_id2: str) -> None:
        """Register a relation between two circuits"""
        if circuit_id1 not in self.circuits or circuit_id2 not in self.circuits:
            raise ValueError(f"Both circuit IDs must be registered: {circuit_id1}, {circuit_id2}")

        # Add bidirectional relationship
        self.related_circuits.setdefault(circuit_id1, set()).add(circuit_id2)
        self.related_circuits.setdefault(circuit_id2, set()).add(circuit_id1)

    def query_circuits(self, circuit_type: Optional[CircuitType] = None,
                       source: Optional[str] = None,
                       min_attribution: Optional[float] = None) -> List[Circuit]:
        """Query circuits matching specific criteria"""
        results = []

        for circuit_id, circuit in self.circuits.items():
            # Apply filters
            if circuit_type and circuit.type != circuit_type:
                continue
            if source and self.sources.get(circuit_id) != source:
                continue
            if min_attribution is not None and circuit.attribution < min_attribution:
                continue

            results.append(circuit)

        return results

    def get_circuit(self, circuit_id: str) -> Optional[Circuit]:
        """Get a specific circuit by ID"""
        return self.circuits.get(circuit_id)

    def get_related_circuits(self, circuit_id: str) -> List[Circuit]:
        """Find circuits related to the specified one"""
        if circuit_id not in self.circuits:
            return []

        related_ids = self.related_circuits.get(circuit_id, set())
        return [self.circuits[cid] for cid in related_ids if cid in self.circuits]

    def save(self, filepath: Optional[Path] = None) -> None:
        """Save the registry to disk"""
        if filepath is None:
            if self.storage_dir is None:
                raise ValueError("No storage directory or filepath specified")
            filepath = self.storage_dir / "circuit_registry.json"

        # Save the circuits
        circuits = list(self.circuits.values())
        save_circuits(circuits, filepath)

        # Save the relationships and sources
        metadata_path = filepath.parent / (filepath.stem + "_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "sources": self.sources,
                "related_circuits": {k: list(v) for k, v in self.related_circuits.items()}
            }, f, indent=2)

    def load(self, filepath: Optional[Path] = None) -> None:
        """Load the registry from disk"""
        if filepath is None:
            if self.storage_dir is None:
                raise ValueError("No storage directory or filepath specified")
            filepath = self.storage_dir / "circuit_registry.json"

        # Load the circuits
        circuits = load_circuits(filepath)
        for circuit in circuits:
            self.circuits[circuit.id] = circuit

        # Load the relationships and sources
        metadata_path = filepath.parent / (filepath.stem + "_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.sources = metadata.get("sources", {})
                self.related_circuits = {k: set(v) for k, v in metadata.get("related_circuits", {}).items()}