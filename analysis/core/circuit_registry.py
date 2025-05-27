# circuit_registry.py
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import json

from analysis.core.circuit_logger import CircuitLogger
from analysis.core.circuit_schema import Circuit, CircuitType, save_circuits, load_circuits
from analysis.utils.utils import get_current_callable_info
from analysis.utils.utils import CircuitJSONEncoder


class CircuitRegistry:
    """Central registry for all discovered circuits"""

    def __init__(self, storage_dir: Optional[Path] = None, circuit_logger=None):
        self.circuits: Dict[str, Circuit] = {}
        self.sources: Dict[str, str] = {}  # Circuit ID to source mapping
        self.related_circuits: Dict[str, Set[str]] = {}  # Circuit ID to related circuit IDs
        self.storage_dir = storage_dir
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.circuit_logger = circuit_logger if circuit_logger else CircuitLogger(self, save_dir=self.storage_dir / "circuit_logger")

    def is_stronger_circuit(self, new_circuit, existing_circuit):
        """Determine if new circuit is stronger than existing"""

        # Multiple strength criteria
        attribution_better = new_circuit.attribution > existing_circuit.attribution

        # Consistency: circuits found in more batches are stronger
        new_consistency = new_circuit.metadata.get('consistency', 1)
        old_consistency = existing_circuit.metadata.get('consistency', 1)
        consistency_better = new_consistency > old_consistency

        # Examples: circuits found in more examples are stronger
        new_examples = new_circuit.metadata.get('examples_found', 1)
        old_examples = existing_circuit.metadata.get('examples_found', 1)
        examples_better = new_examples > old_examples

        # Weighted decision
        if consistency_better and attribution_better:
            return True
        elif consistency_better and new_examples >= old_examples:
            return True
        elif attribution_better and new_consistency >= old_consistency:
            return True

        return False

    def register_circuit(self, circuit: Circuit, source: str = "unknown") -> None:
        """Add a circuit to the registry with upgrade logic and logging"""
        if circuit.id in self.circuits:
            existing = self.circuits[circuit.id]

            if self.is_stronger_circuit(circuit, existing):
                # Log upgrade with all details preserved
                self.circuit_logger.log_circuit_event(
                    'upgraded',
                    circuit,
                    circuit.discovered_at,
                    {
                        'source': source,
                        'old_attribution': existing.attribution,
                        'new_attribution': circuit.attribution,
                        'improvement': circuit.attribution - existing.attribution
                    }
                )
                print(f"\t\t{get_current_callable_info()}: \tupgrade       \t{circuit.id}: {existing.attribution:.3f} → {circuit.attribution:.3f}")
                self._update_existing_circuit(existing, circuit)
            else:
                self.circuit_logger.log_circuit_event(
                    'rejected',
                    circuit,
                    circuit.discovered_at,
                    {'source': source, 'reason': 'weaker_than_existing'}
                )
                print(f"\t\t{get_current_callable_info()}: \tkeep stronger\t{circuit.id}")
        else:
            self.circuit_logger.log_circuit_event(
                'created',
                circuit,
                circuit.discovered_at,
                {'source': source}
            )
            print(f"\t\t{get_current_callable_info()}: \tregister new \t{circuit.id}")
            self.circuits[circuit.id] = circuit
            self.sources[circuit.id] = source

            # Initialize related circuits set if needed
            if circuit.id not in self.related_circuits:
                self.related_circuits[circuit.id] = set()

    def _update_existing_circuit(self, existing, new_circuit):
        """Update existing circuit with new information"""
        existing.attribution = max(existing.attribution, new_circuit.attribution)
        existing.metadata.update(new_circuit.metadata)
        existing.metadata['last_updated'] = new_circuit.discovered_at
        existing.metadata['update_count'] = existing.metadata.get('update_count', 0) + 1

    def generate_circuit_id(self, operation_type, component_info, epoch, **kwargs):
        """Generate unique, informative circuit IDs for any circuit type"""

        if operation_type == "copy":
            head = component_info  # For token circuits
            source_pos = kwargs.get('source_pos', -1)
            target_pos = kwargs.get('target_pos', -1)
            relative_offset = kwargs.get('relative_offset', target_pos - source_pos if source_pos >= 0 else 0)

            position_info = f"offset_{relative_offset}"
            if source_pos >= 0 and target_pos >= 0:
                position_info += f"_pos_{source_pos}_{target_pos}"

            base_id = f"{operation_type}_{head}_{position_info}"

        elif operation_type == "induction":
            head = component_info
            pattern_type = kwargs.get('pattern_type', 'basic')
            inducer_pos = kwargs.get('inducer_pos', -1)
            induced_pos = kwargs.get('induced_pos', -1)
            target_pos = kwargs.get('target_pos', -1)
            if inducer_pos >= 0 and target_pos >= 0:
                position_info = f"_pos_{inducer_pos}_{induced_pos}_{target_pos}"
            else:
                position_info = ""

            base_id = f"{operation_type}_{head}_{pattern_type}{position_info}"

        elif operation_type == "component_interaction":
            # For component-level circuits
            components = component_info  # List of component names
            base_id = f"{operation_type}_{'_'.join(sorted(components))}"

        elif operation_type == "mlp_subspace":
            # For subspace-level circuits
            layer_info = component_info
            base_id = f"{operation_type}_{layer_info}"

        else:
            # Generic fallback
            base_id = f"{operation_type}_{component_info}"

        # Add discovery context
        source = kwargs.get('source', 'individual')
        consistency = kwargs.get('consistency', 1)

        return f"{base_id}_{epoch}_{source}_c{consistency}"

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
        """Save the registry to disk using robust serialization"""
        if filepath is None:
            if self.storage_dir is None:
                raise ValueError("No storage directory or filepath specified")
            filepath = self.storage_dir / "circuit_registry.json"

        # ✅ UPDATED: save_circuits now uses CircuitJSONEncoder automatically
        circuits = list(self.circuits.values())
        save_circuits(circuits, filepath)

        # Save the relationships and sources with custom encoder
        metadata_path = filepath.parent / (filepath.stem + "_metadata.json")

        try:
            # ✅ Use CircuitJSONEncoder for metadata too
            from analysis.utils.utils import CircuitJSONEncoder

            with open(metadata_path, 'w') as f:
                json.dump({
                    "sources": self.sources,
                    "related_circuits": {k: list(v) for k, v in self.related_circuits.items()}
                }, f, cls=CircuitJSONEncoder, indent=2)

            print(f"\t{get_current_callable_info()}\t✅saved registry metadata to {metadata_path}")

        except Exception as e:
            print(f"⚠️ Error saving metadata with custom encoder: {e}")
            # Fallback to basic JSON
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