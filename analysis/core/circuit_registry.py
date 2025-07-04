# circuit_registry.py
from collections import defaultdict
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
import json

from analysis.core.circuit_logger import CircuitLogger
from analysis.core.circuit_schema import (Circuit, CircuitType, save_circuits, load_circuits, RelationshipType, \
                                          CircuitMetadata, EmergencePhase, CircuitStability)

class CircuitRegistry:
    """Central registry for all discovered circuits"""

    def __init__(self, storage_dir: Optional[Path] = None, circuit_logger=None):
        self.circuits: Dict[str, Circuit] = {}
        self.sources: Dict[str, str] = {}  # Circuit ID to source mapping
        self.related_circuits: Dict[str, Set[str]] = {}  # Circuit ID to related circuit IDs
        self.storage_dir = storage_dir
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        if circuit_logger:
            self.circuit_logger = circuit_logger
        elif self.storage_dir:
            self.circuit_logger = CircuitLogger(self, save_dir=self.storage_dir / "circuit_logger")
        else:
            self.circuit_logger = CircuitLogger(self, save_dir=None)

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
                # print(f"\t\t{get_current_callable_info()}: \tupgrade       \t{circuit.id}: {existing.attribution:.3f} → {circuit.attribution:.3f}")
                self._update_existing_circuit(existing, circuit)
            else:
                self.circuit_logger.log_circuit_event(
                    'rejected',
                    circuit,
                    circuit.discovered_at,
                    {'source': source, 'reason': 'weaker_than_existing'}
                )
                # print(f"\t\t{get_current_callable_info()}: \tkeep stronger\t{circuit.id}")
        else:
            self.circuit_logger.log_circuit_event(
                'created',
                circuit,
                circuit.discovered_at,
                {'source': source}
            )
            # print(f"\t\t{get_current_callable_info()}: \tregister new \t{circuit.id}")
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

            # print(f"\t{get_current_callable_info()}\t✅saved registry metadata to {metadata_path}")

        except Exception as e:
            # print(f"⚠️ Error saving metadata with custom encoder: {e}")
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


class EnhancedCircuitRegistry(CircuitRegistry):
    """Enhanced registry with temporal tracking and relationship management"""

    def __init__(self, storage_dir: Optional[Path] = None, circuit_logger=None):
        """Initialize enhanced registry with temporal tracking"""
        super().__init__(storage_dir, circuit_logger)

        # Enhanced tracking data structures
        self.circuit_metadata: Dict[str, CircuitMetadata] = {}
        self.relationship_graph: Dict[str, Dict[str, RelationshipType]] = defaultdict(dict)
        self.emergence_timeline: Dict[int, List[str]] = defaultdict(list)  # epoch -> circuit_ids
        self.stability_tracker: Dict[str, List[int]] = defaultdict(list)  # circuit_id -> epochs_seen

        # Method tracking for validation
        self.detection_methods: Set[str] = set()
        self.method_consistency: Dict[Tuple[str, str], float] = {}  # (method1, method2) -> agreement

        print("✅ Enhanced circuit registry initialized")

    def register_circuit_enhanced(self, circuit, source: str = "unknown",
                                  epoch: int = 0, detection_method: str = "unknown",
                                  confidence: float = 0.5, total_epochs: int = 1000,
                                  **kwargs) -> bool:
        """
        Enhanced circuit registration with temporal and reliability tracking

        Returns:
            bool: True if circuit was registered (new or updated), False if rejected
        """
        # Initialize metadata if new circuit
        if circuit.id not in self.circuit_metadata:
            self.circuit_metadata[circuit.id] = CircuitMetadata(
                first_detected=epoch,
                detection_method=detection_method,
                detection_confidence=confidence
            )

        metadata = self.circuit_metadata[circuit.id]

        # Update temporal tracking
        metadata.last_seen = epoch
        if epoch not in metadata.detection_epochs:
            metadata.detection_epochs.append(epoch)
            self.stability_tracker[circuit.id].append(epoch)

        # Update emergence timeline
        if circuit.id not in self.emergence_timeline[epoch]:
            self.emergence_timeline[epoch].append(circuit.id)

        # Determine emergence phase
        metadata.emergence_phase = self._classify_emergence_phase(epoch, total_epochs)

        # Update stability classification
        metadata.stability = self._classify_stability(circuit.id)

        # Calculate stability score
        metadata.stability_score = self._calculate_stability_score(circuit.id)

        # Update strength history
        metadata.strength_history.append((epoch, circuit.attribution))

        # Check if we should register this circuit
        should_register = self._should_register_circuit(circuit, metadata, confidence)

        if should_register:
            # Register with parent class (existing functionality)
            super().register_circuit(circuit, source)

            # Update additional metadata
            metadata.consistency_score = self._calculate_consistency_score(circuit.id)

            # Track detection method
            self.detection_methods.add(detection_method)

            return True
        else:
            return False

    def register_circuit(self, circuit, source: str = "unknown"):
        """Backward compatibility method - use basic registration"""
        return self.register_circuit_enhanced(
            circuit=circuit,
            source=source,
            epoch=0,
            detection_method="legacy",
            confidence=0.5
        )

    def add_relationship(self, circuit_id1: str, circuit_id2: str,
                         relationship: RelationshipType, strength: float = 1.0):
        """Add relationship between circuits with bidirectional tracking"""
        self.relationship_graph[circuit_id1][circuit_id2] = relationship

        # Update circuit metadata if both circuits exist
        if circuit_id1 in self.circuit_metadata and circuit_id2 in self.circuit_metadata:
            meta1 = self.circuit_metadata[circuit_id1]
            meta2 = self.circuit_metadata[circuit_id2]

            # Update relationship sets based on type
            if relationship == RelationshipType.PREREQUISITE:
                meta1.enables_circuits.add(circuit_id2)
                meta2.prerequisite_circuits.add(circuit_id1)
            elif relationship == RelationshipType.COMPETITIVE:
                meta1.competes_with.add(circuit_id2)
                meta2.competes_with.add(circuit_id1)
            elif relationship == RelationshipType.COOPERATIVE:
                meta1.cooperates_with.add(circuit_id2)
                meta2.cooperates_with.add(circuit_id1)

    def get_circuits_by_phase(self, phase: EmergencePhase) -> List:
        """Get circuits by emergence phase"""
        return [
            self.circuits[cid] for cid, metadata in self.circuit_metadata.items()
            if metadata.emergence_phase == phase and cid in self.circuits
        ]

    def get_circuits_by_stability(self, stability: CircuitStability) -> List:
        """Get circuits by stability classification"""
        return [
            self.circuits[cid] for cid, metadata in self.circuit_metadata.items()
            if metadata.stability == stability and cid in self.circuits
        ]

    def get_circuits_by_type(self, circuit_type) -> List:
        """Get circuits by type (enhanced version)"""
        return [
            circuit for circuit in self.circuits.values()
            if circuit.type == circuit_type
        ]

    def get_emergence_timeline(self) -> Dict[int, List[str]]:
        """Get timeline of circuit emergence"""
        return dict(self.emergence_timeline)

    def get_circuit_relationships(self, circuit_id: str) -> Dict[str, RelationshipType]:
        """Get all relationships for a circuit"""
        return dict(self.relationship_graph.get(circuit_id, {}))

    def _classify_emergence_phase(self, epoch: int, total_epochs: int) -> EmergencePhase:
        """Classify emergence phase based on epoch and total training"""
        if epoch < 100:
            return EmergencePhase.EARLY
        elif epoch < min(500, total_epochs * 0.5):
            return EmergencePhase.MIDDLE
        elif epoch < min(800, total_epochs * 0.8):
            return EmergencePhase.LATE
        else:
            return EmergencePhase.POST_GROKKING

    def _classify_stability(self, circuit_id: str) -> CircuitStability:
        """Classify circuit stability based on detection history"""
        epochs_seen = self.stability_tracker.get(circuit_id, [])

        if len(epochs_seen) < 2:
            return CircuitStability.TRANSIENT
        elif len(epochs_seen) < 5:
            return CircuitStability.EMERGING
        elif len(epochs_seen) >= 10:
            return CircuitStability.PERSISTENT
        else:
            return CircuitStability.STABLE

    def _calculate_stability_score(self, circuit_id: str) -> float:
        """Calculate numerical stability score (0.0 to 1.0)"""
        epochs_seen = self.stability_tracker.get(circuit_id, [])

        if len(epochs_seen) < 2:
            return 0.1  # Very unstable

        # Check for recent presence (last 20 epochs)
        if epochs_seen:
            recent_epochs = [e for e in epochs_seen if e >= max(epochs_seen) - 20]
            recency_score = len(recent_epochs) / min(20, len(epochs_seen))

            # Check for consistency over time span
            time_span = max(epochs_seen) - min(epochs_seen) + 1
            consistency_score = len(epochs_seen) / max(1, time_span)

            return min(1.0, (recency_score + consistency_score) / 2)

        return 0.1

    def _should_register_circuit(self, circuit, metadata: CircuitMetadata,
                                 confidence: float) -> bool:
        """Determine if circuit should be registered based on quality criteria"""
        # Minimum confidence threshold
        if confidence < 0.3:
            return False

        # Stability requirements - transient circuits need higher confidence
        if metadata.stability == CircuitStability.TRANSIENT and confidence < 0.7:
            return False

        # For existing circuits, check if this is an improvement
        if circuit.id in self.circuits:
            existing_metadata = self.circuit_metadata[circuit.id]
            return (confidence > existing_metadata.detection_confidence or
                    metadata.stability_score > existing_metadata.stability_score)

        return True

    def _calculate_consistency_score(self, circuit_id: str) -> float:
        """Calculate consistency score across detection methods and epochs"""
        # This is a placeholder - in full implementation would compare
        # results across different detection methods
        epochs_seen = len(self.stability_tracker.get(circuit_id, []))
        return min(1.0, epochs_seen / 10.0)  # Max score after 10 sightings

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of registry state"""
        summary = {
            "total_circuits": len(self.circuits),
            "circuits_by_type": {},
            "circuits_by_phase": {},
            "circuits_by_stability": {},
            "total_relationships": sum(len(rels) for rels in self.relationship_graph.values()),
            "detection_methods": list(self.detection_methods)
        }

        # Count by type
        for circuit in self.circuits.values():
            circuit_type = circuit.type.value
            summary["circuits_by_type"][circuit_type] = summary["circuits_by_type"].get(circuit_type, 0) + 1

        # Count by phase
        for metadata in self.circuit_metadata.values():
            phase = metadata.emergence_phase.value
            summary["circuits_by_phase"][phase] = summary["circuits_by_phase"].get(phase, 0) + 1

        # Count by stability
        for metadata in self.circuit_metadata.values():
            stability = metadata.stability.value
            summary["circuits_by_stability"][stability] = summary["circuits_by_stability"].get(stability, 0) + 1

        return summary
