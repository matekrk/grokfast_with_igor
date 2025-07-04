# analysis/core/canonical_circuit_system.py
"""
Complete Canonical Circuit System

Properly separates circuit identity (computational algorithm) from instance data (examples).
Supports evolution tracking, circuit competition analysis, and extension to new circuit types.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple

import numpy as np

from analysis.core.circuit_schema import (Circuit, CircuitType, Element, Connection)


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class CircuitInstance:
    """Single detection instance of a circuit"""
    epoch: int
    attribution: float
    strength: float
    tokens: List[str]
    positions: Dict[str, int]  # Role -> absolute position mapping
    detection_confidence: float
    detection_method: str
    example_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputationalSignature:
    """Defines the computational identity of a circuit"""
    operation_type: str
    circuit_type: str
    structural_pattern: Dict[str, Any]

    def to_string(self) -> str:
        """Create deterministic string representation"""
        signature_dict = {
            'operation_type': self.operation_type,
            'circuit_type': self.circuit_type,
            'structural_pattern': self.structural_pattern
        }
        return json.dumps(signature_dict, sort_keys=True)

    def get_hash(self) -> str:
        """Get short hash for ID generation"""
        return hashlib.md5(self.to_string().encode()).hexdigest()[:8]


@dataclass
class CanonicalCircuit:
    """Canonical representation aggregating instances of the same computational pattern"""
    canonical_id: str
    computational_signature: ComputationalSignature
    circuit_type: CircuitType

    # Instance aggregation
    instances: List[CircuitInstance] = field(default_factory=list)
    token_examples: Set[str] = field(default_factory=set)
    position_patterns: Set[str] = field(default_factory=set)

    # Evolution tracking
    first_seen: int = 0
    last_seen: int = 0
    total_detections: int = 0

    # Strength metrics
    best_attribution: float = 0.0
    current_attribution: float = 0.0
    avg_attribution: float = 0.0
    attribution_history: List[Tuple[int, float]] = field(default_factory=list)

    # Stability metrics
    stability_score: float = 0.0
    consistency_score: float = 0.0
    persistence_score: float = 0.0

    # Canonical structure (normalized)
    elements: List[Element] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# COMPUTATIONAL SIGNATURE EXTRACTORS
# ============================================================================

class SignatureExtractor(ABC):
    """Base class for extracting computational signatures from circuits"""

    @abstractmethod
    def extract_signature(self, circuit: Circuit) -> ComputationalSignature:
        """Extract computational signature from circuit"""
        pass

    @abstractmethod
    def supports_circuit_type(self, circuit: Circuit) -> bool:
        """Check if this extractor supports the circuit type"""
        pass


class TokenOperationSignatureExtractor(SignatureExtractor):
    """Extractor for token operation circuits (copy, induction)"""

    def supports_circuit_type(self, circuit: Circuit) -> bool:
        operation_type = circuit.metadata.get('operation_type', '')
        return operation_type in ['copy', 'induction']

    def extract_signature(self, circuit: Circuit) -> ComputationalSignature:
        operation_type = circuit.metadata.get('operation_type', 'unknown')

        if operation_type == 'copy':
            return self._extract_copy_signature(circuit)
        elif operation_type == 'induction':
            return self._extract_induction_signature(circuit)
        else:
            return self._extract_generic_signature(circuit)

    def _extract_copy_signature(self, circuit: Circuit) -> ComputationalSignature:
        """Extract signature for copy circuits"""
        head = circuit.metadata.get('head', 'unknown')
        relative_offset = circuit.metadata.get('relative_offset', 0)

        # Extract layer from head name if possible
        layer = self._extract_layer_from_head(head)

        structural_pattern = {
            'head': head,
            'layer': layer,
            'relative_offset': relative_offset,
            'operation_class': 'positional_copy',
            'connection_pattern': self._extract_connection_pattern(circuit)
        }

        return ComputationalSignature(
            operation_type='copy',
            circuit_type='token_operation',
            structural_pattern=structural_pattern
        )

    def _extract_induction_signature(self, circuit: Circuit) -> ComputationalSignature:
        """Extract signature for induction circuits"""
        head = circuit.metadata.get('head', 'unknown')
        pattern_distance = circuit.metadata.get('pattern_distance', 0)
        induction_span = circuit.metadata.get('induction_span', 0)

        layer = self._extract_layer_from_head(head)

        structural_pattern = {
            'head': head,
            'layer': layer,
            'pattern_distance': pattern_distance,
            'induction_span': induction_span,
            'operation_class': 'induction_head',
            'connection_pattern': self._extract_connection_pattern(circuit)
        }

        return ComputationalSignature(
            operation_type='induction',
            circuit_type='token_operation',
            structural_pattern=structural_pattern
        )

    def _extract_generic_signature(self, circuit: Circuit) -> ComputationalSignature:
        """Extract signature for unknown token operation circuits"""
        structural_pattern = {
            'connection_pattern': self._extract_connection_pattern(circuit),
            'operation_class': 'generic_token_operation'
        }

        return ComputationalSignature(
            operation_type=circuit.metadata.get('operation_type', 'unknown'),
            circuit_type='token_operation',
            structural_pattern=structural_pattern
        )

    def _extract_layer_from_head(self, head: str) -> int:
        """Extract layer number from head name"""
        # Assumes head format like "layer_0_head_1"
        if 'layer_' in head:
            try:
                parts = head.split('_')
                layer_idx = parts.index('layer')
                return int(parts[layer_idx + 1])
            except (ValueError, IndexError):
                pass
        return 0

    def _extract_connection_pattern(self, circuit: Circuit) -> List[str]:
        """Extract abstract connection pattern"""
        pattern = []
        for conn in circuit.connections:
            # Find source and target elements
            src_elem = next((e for e in circuit.elements if e.id == conn.source), None)
            tgt_elem = next((e for e in circuit.elements if e.id == conn.target), None)

            if src_elem and tgt_elem:
                # Create abstract pattern
                src_role = src_elem.properties.get('role', 'unknown') if src_elem.properties else 'unknown'
                tgt_role = tgt_elem.properties.get('role', 'unknown') if tgt_elem.properties else 'unknown'
                pattern.append(f"{src_role}->{tgt_role}:{conn.type.value}")

        return sorted(pattern)  # Sort for deterministic signatures


class ComponentInteractionSignatureExtractor(SignatureExtractor):
    """Extractor for component interaction circuits"""

    def supports_circuit_type(self, circuit: Circuit) -> bool:
        return circuit.type == CircuitType.COMPONENT

    def extract_signature(self, circuit: Circuit) -> ComputationalSignature:
        """Extract signature for component circuits"""
        interaction_type = circuit.metadata.get('interaction_type', 'unknown')
        components = circuit.metadata.get('components', [])

        structural_pattern = {
            'interaction_type': interaction_type,
            'component_count': len(components),
            'component_types': sorted(components) if isinstance(components, list) else [],
            'connection_pattern': self._extract_connection_pattern(circuit)
        }

        return ComputationalSignature(
            operation_type=interaction_type,
            circuit_type='component_interaction',
            structural_pattern=structural_pattern
        )

    def _extract_connection_pattern(self, circuit: Circuit) -> List[str]:
        """Extract component interaction pattern"""
        pattern = []
        for conn in circuit.connections:
            pattern.append(f"{conn.type.value}:{conn.strength:.2f}")
        return sorted(pattern)


class MLPSubspaceSignatureExtractor(SignatureExtractor):
    """Extractor for MLP subspace circuits"""

    def supports_circuit_type(self, circuit: Circuit) -> bool:
        return circuit.metadata.get('operation_type', '').startswith('mlp_')

    def extract_signature(self, circuit: Circuit) -> ComputationalSignature:
        """Extract signature for MLP circuits"""
        subspace_type = circuit.metadata.get('subspace_type', 'unknown')
        layer = circuit.metadata.get('layer', 0)
        dimension = circuit.metadata.get('dimension', 0)

        structural_pattern = {
            'subspace_type': subspace_type,
            'layer': layer,
            'dimension': dimension,
            'sparsity_pattern': circuit.metadata.get('sparsity_pattern', 'unknown')
        }

        return ComputationalSignature(
            operation_type=f"mlp_{subspace_type}",
            circuit_type='mlp_subspace',
            structural_pattern=structural_pattern
        )


# ============================================================================
# CANONICAL CIRCUIT REGISTRY
# ============================================================================

class CanonicalCircuitRegistry:
    """Registry that properly handles circuit identity and evolution"""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir
        if storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Core storage
        self.canonical_circuits: Dict[str, CanonicalCircuit] = {}
        self.signature_to_id: Dict[str, str] = {}  # Signature hash -> canonical ID

        # Signature extractors for different circuit types
        self.extractors: List[SignatureExtractor] = [
            TokenOperationSignatureExtractor(),
            ComponentInteractionSignatureExtractor(),
            MLPSubspaceSignatureExtractor()
        ]

        # Evolution tracking
        self.epoch_to_circuits: Dict[int, Set[str]] = defaultdict(set)
        self.circuit_relationships: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Statistics
        self.total_registrations = 0
        self.total_aggregations = 0

    def register_circuit_detection(self, circuit: Circuit, epoch: int, tokens: List[str],
                                   detection_confidence: float = 0.5,
                                   detection_method: str = "unknown",
                                   example_metadata: Optional[Dict] = None) -> str:
        """
        Main interface: Register a circuit detection

        Args:
            circuit: Detected circuit
            epoch: Training epoch
            tokens: Token sequence for this example
            detection_confidence: Confidence in detection
            detection_method: Method used for detection
            example_metadata: Additional metadata for this example

        Returns:
            Canonical circuit ID
        """
        self.total_registrations += 1

        # Extract computational signature
        signature = self._extract_computational_signature(circuit)

        # Get or create canonical ID
        canonical_id = self._get_or_create_canonical_id(signature)

        # Extract position mapping
        positions = self._extract_position_mapping(circuit)

        # Create instance
        instance = CircuitInstance(
            epoch=epoch,
            attribution=circuit.attribution,
            strength=circuit.attribution,
            tokens=tokens.copy() if tokens else [],
            positions=positions,
            detection_confidence=detection_confidence,
            detection_method=detection_method,
            example_metadata=example_metadata or {}
        )

        # Register or update canonical circuit
        if canonical_id in self.canonical_circuits:
            self._update_canonical_circuit(canonical_id, instance, circuit)
            self.total_aggregations += 1
        else:
            self._create_canonical_circuit(canonical_id, signature, instance, circuit)

        # Update epoch tracking
        self.epoch_to_circuits[epoch].add(canonical_id)

        return canonical_id

    def _extract_computational_signature(self, circuit: Circuit) -> ComputationalSignature:
        """Extract computational signature using appropriate extractor"""
        for extractor in self.extractors:
            if extractor.supports_circuit_type(circuit):
                return extractor.extract_signature(circuit)

        # Fallback: generic signature
        return ComputationalSignature(
            operation_type=circuit.metadata.get('operation_type', 'unknown'),
            circuit_type='generic',
            structural_pattern={'fallback': True}
        )

    def _get_or_create_canonical_id(self, signature: ComputationalSignature) -> str:
        """Get existing canonical ID or create new one"""
        signature_hash = signature.get_hash()

        if signature_hash in self.signature_to_id:
            return self.signature_to_id[signature_hash]

        # Create new canonical ID
        operation = signature.operation_type
        circuit_type = signature.circuit_type
        canonical_id = f"{operation}_{circuit_type}_{signature_hash}"

        # Store mapping
        self.signature_to_id[signature_hash] = canonical_id

        return canonical_id

    def _extract_position_mapping(self, circuit: Circuit) -> Dict[str, int]:
        """Extract position mapping from circuit elements"""
        positions = {}
        for element in circuit.elements:
            if element.properties:
                role = element.properties.get('role')
                pos = element.properties.get('position')
                if role and pos is not None:
                    positions[role] = pos
        return positions

    def _create_canonical_circuit(self, canonical_id: str, signature: ComputationalSignature,
                                  instance: CircuitInstance, circuit: Circuit):
        """Create new canonical circuit"""
        # Normalize elements and connections
        normalized_elements = self._normalize_elements(circuit.elements)
        normalized_connections = self._normalize_connections(circuit.connections)

        # Create canonical circuit
        canonical = CanonicalCircuit(
            canonical_id=canonical_id,
            computational_signature=signature,
            circuit_type=circuit.type,

            # Initialize with first instance
            instances=[instance],
            token_examples=set([' '.join(instance.tokens)]) if instance.tokens else set(),
            position_patterns=set([self._create_position_pattern(instance.positions)]),

            # Evolution metrics
            first_seen=instance.epoch,
            last_seen=instance.epoch,
            total_detections=1,

            # Strength metrics
            best_attribution=instance.attribution,
            current_attribution=instance.attribution,
            avg_attribution=instance.attribution,
            attribution_history=[(instance.epoch, instance.attribution)],

            # Stability (will be calculated)
            stability_score=0.1,
            consistency_score=0.5,
            persistence_score=0.1,

            # Structure
            elements=normalized_elements,
            connections=normalized_connections,
            metadata=circuit.metadata.copy()
        )

        self.canonical_circuits[canonical_id] = canonical

    def _update_canonical_circuit(self, canonical_id: str, instance: CircuitInstance, circuit: Circuit):
        """Update existing canonical circuit with new instance"""
        canonical = self.canonical_circuits[canonical_id]

        # Add instance
        canonical.instances.append(instance)

        # Update aggregated data
        if instance.tokens:
            canonical.token_examples.add(' '.join(instance.tokens))
        canonical.position_patterns.add(self._create_position_pattern(instance.positions))

        # Update evolution tracking
        canonical.last_seen = max(canonical.last_seen, instance.epoch)
        canonical.total_detections += 1

        # Update strength metrics
        canonical.best_attribution = max(canonical.best_attribution, instance.attribution)
        canonical.current_attribution = instance.attribution  # Most recent

        # Update attribution history and running average
        canonical.attribution_history.append((instance.epoch, instance.attribution))
        attributions = [attr for _, attr in canonical.attribution_history]
        canonical.avg_attribution = np.mean(attributions)

        # Recalculate stability metrics
        self._update_stability_metrics(canonical)

        # Update metadata with any new information
        canonical.metadata.update(circuit.metadata)

    def _create_position_pattern(self, positions: Dict[str, int]) -> str:
        """Create position pattern string"""
        if not positions:
            return "no_positions"

        # Create relative position pattern
        sorted_roles = sorted(positions.keys())
        if len(sorted_roles) >= 2:
            # Calculate relative offsets
            base_pos = positions[sorted_roles[0]]
            relative_positions = [positions[role] - base_pos for role in sorted_roles]
            return f"pattern_{','.join(map(str, relative_positions))}"
        else:
            return f"single_{sorted_roles[0]}"

    def _normalize_elements(self, elements: List[Element]) -> List[Element]:
        """Normalize elements for canonical representation"""
        normalized = []

        for element in elements:
            # Create normalized properties
            norm_props = {}
            if element.properties:
                # Keep structural information, remove instance-specific data
                for key, value in element.properties.items():
                    if key in ['role', 'operation', 'name']:
                        norm_props[key] = value
                    elif key == 'token':
                        norm_props['token_type'] = self._classify_token_type(str(value))
                    elif key == 'position':
                        norm_props['has_position'] = True

            # Create normalized element
            normalized_elem = Element(
                id=f"{element.type.value}_{norm_props.get('role', 'unknown')}",
                type=element.type,
                properties=norm_props
            )
            normalized.append(normalized_elem)

        return normalized

    def _normalize_connections(self, connections: List[Connection]) -> List[Connection]:
        """Normalize connections for canonical representation"""
        normalized = []

        for conn in connections:
            norm_props = conn.properties.copy() if conn.properties else {}

            # Create normalized connection
            normalized_conn = Connection(
                source=f"source_{norm_props.get('position_type', 'unknown')}",
                target=f"target_{norm_props.get('position_type', 'unknown')}",
                strength=1.0,  # Normalize strength for structural comparison
                type=conn.type,
                properties=norm_props
            )
            normalized.append(normalized_conn)

        return normalized

    def _classify_token_type(self, token: str) -> str:
        """Classify token into semantic category"""
        if token.isdigit():
            return 'NUMBER'
        elif token in ['+', '-', '*', '/', '=', '%']:
            return 'OPERATOR'
        elif token in ['(', ')', '[', ']', '{', '}']:
            return 'DELIMITER'
        elif token.isalpha():
            return 'WORD'
        else:
            return 'SYMBOL'

    def _update_stability_metrics(self, canonical: CanonicalCircuit):
        """Update stability metrics for canonical circuit"""
        # Detection frequency score
        detection_score = min(1.0, canonical.total_detections / 20.0)

        # Temporal persistence score
        temporal_span = canonical.last_seen - canonical.first_seen + 1
        persistence_score = min(1.0, temporal_span / 200.0)

        # Attribution consistency score
        if len(canonical.attribution_history) > 1:
            attributions = [attr for _, attr in canonical.attribution_history]
            attr_std = np.std(attributions)
            attr_mean = np.mean(attributions)
            consistency_score = max(0.0, 1.0 - (attr_std / max(attr_mean, 0.1)))
        else:
            consistency_score = 0.5

        # Token diversity score (indicates generality)
        diversity_score = min(1.0, len(canonical.token_examples) / 10.0)

        # Combined stability score
        stability_score = (
                detection_score * 0.3 +
                persistence_score * 0.25 +
                consistency_score * 0.25 +
                diversity_score * 0.2
        )

        # Update metrics
        canonical.stability_score = stability_score
        canonical.consistency_score = consistency_score
        canonical.persistence_score = persistence_score

    def get_canonical_circuit(self, canonical_id: str) -> Optional[CanonicalCircuit]:
        """Get canonical circuit by ID"""
        return self.canonical_circuits.get(canonical_id)

    def get_stable_circuits(self, min_stability: float = 0.6,
                            min_detections: int = 3) -> List[CanonicalCircuit]:
        """Get stable circuits above thresholds"""
        stable = []
        for circuit in self.canonical_circuits.values():
            if (circuit.stability_score >= min_stability and
                    circuit.total_detections >= min_detections):
                stable.append(circuit)

        # Sort by stability score
        return sorted(stable, key=lambda c: c.stability_score, reverse=True)

    def get_circuits_by_type(self, operation_type: str) -> List[CanonicalCircuit]:
        """Get circuits by operation type"""
        return [
            circuit for circuit in self.canonical_circuits.values()
            if circuit.computational_signature.operation_type == operation_type
        ]

    def get_circuits_active_in_epoch(self, epoch: int) -> List[str]:
        """Get canonical IDs of circuits active in specific epoch"""
        return list(self.epoch_to_circuits.get(epoch, set()))

    def analyze_circuit_evolution(self, canonical_id: str) -> Dict[str, Any]:
        """Analyze evolution of specific circuit"""
        canonical = self.get_canonical_circuit(canonical_id)
        if not canonical:
            return {}

        # Basic evolution metrics
        evolution_data = {
            'canonical_id': canonical_id,
            'computational_signature': canonical.computational_signature.to_string(),
            'total_detections': canonical.total_detections,
            'temporal_span': canonical.last_seen - canonical.first_seen + 1,
            'first_seen': canonical.first_seen,
            'last_seen': canonical.last_seen,

            # Strength evolution
            'best_attribution': canonical.best_attribution,
            'avg_attribution': canonical.avg_attribution,
            'current_attribution': canonical.current_attribution,
            'attribution_history': canonical.attribution_history,

            # Stability metrics
            'stability_score': canonical.stability_score,
            'consistency_score': canonical.consistency_score,
            'persistence_score': canonical.persistence_score,

            # Diversity metrics
            'token_examples_count': len(canonical.token_examples),
            'position_patterns_count': len(canonical.position_patterns),
            'detection_methods': list(set(inst.detection_method for inst in canonical.instances)),

            # Examples
            'token_examples': list(canonical.token_examples)[:10],  # Limit for display
            'position_patterns': list(canonical.position_patterns)
        }

        # Attribution trend analysis
        if len(canonical.attribution_history) > 2:
            epochs = [epoch for epoch, _ in canonical.attribution_history]
            attributions = [attr for _, attr in canonical.attribution_history]

            # Simple trend analysis
            early_third = attributions[:len(attributions) // 3] or [0]
            late_third = attributions[-len(attributions) // 3:] or [0]

            early_avg = np.mean(early_third)
            late_avg = np.mean(late_third)

            if late_avg > early_avg * 1.2:
                trend = 'strengthening'
            elif late_avg < early_avg * 0.8:
                trend = 'weakening'
            else:
                trend = 'stable'

            evolution_data['attribution_trend'] = {
                'trend': trend,
                'early_avg': early_avg,
                'late_avg': late_avg,
                'variance': np.var(attributions)
            }

        return evolution_data

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get comprehensive registry summary"""
        circuits_by_type = defaultdict(int)
        circuits_by_operation = defaultdict(int)

        for circuit in self.canonical_circuits.values():
            circuits_by_type[circuit.circuit_type.value] += 1
            circuits_by_operation[circuit.computational_signature.operation_type] += 1

        stable_circuits = self.get_stable_circuits()

        return {
            'total_canonical_circuits': len(self.canonical_circuits),
            'total_registrations': self.total_registrations,
            'total_aggregations': self.total_aggregations,
            'aggregation_rate': self.total_aggregations / max(1, self.total_registrations),

            'circuits_by_type': dict(circuits_by_type),
            'circuits_by_operation': dict(circuits_by_operation),

            'stable_circuits_count': len(stable_circuits),
            'stability_rate': len(stable_circuits) / max(1, len(self.canonical_circuits)),

            'avg_detections_per_circuit': np.mean(
                [c.total_detections for c in self.canonical_circuits.values()]) if self.canonical_circuits else 0,
            'max_detections': max(
                [c.total_detections for c in self.canonical_circuits.values()]) if self.canonical_circuits else 0,

            'epochs_with_activity': len(self.epoch_to_circuits),
            'signature_extractors': len(self.extractors)
        }


# ============================================================================
# INTEGRATION WITH EXISTING SYSTEM
# ============================================================================

class CanonicalRegistryAdapter:
    """Adapter to integrate canonical registry with existing EnhancedCircuitRegistry"""

    def __init__(self, enhanced_registry, canonical_registry: CanonicalCircuitRegistry):
        self.enhanced_registry = enhanced_registry
        self.canonical_registry = canonical_registry

        # Keep mapping of canonical ID -> legacy circuit for compatibility
        self.canonical_to_legacy: Dict[str, str] = {}

    def register_circuit_detection(self, circuit: Circuit, epoch: int, tokens: List[str],
                                   detection_confidence: float = 0.5,
                                   detection_method: str = "unknown",
                                   example_metadata: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Register circuit in both canonical and legacy systems

        Returns:
            Tuple of (canonical_id, legacy_id)
        """
        # Register in canonical system
        canonical_id = self.canonical_registry.register_circuit_detection(
            circuit, epoch, tokens, detection_confidence, detection_method, example_metadata
        )

        # Create or update legacy circuit for compatibility
        legacy_circuit = self._create_legacy_circuit(canonical_id, circuit, epoch)

        # Register in enhanced registry
        self.enhanced_registry.register_circuit_enhanced(
            circuit=legacy_circuit,
            source="canonical_detection",
            epoch=epoch,
            detection_method=f"canonical_{detection_method}",
            confidence=detection_confidence
        )

        # Store mapping
        self.canonical_to_legacy[canonical_id] = legacy_circuit.id

        return canonical_id, legacy_circuit.id

    def _create_legacy_circuit(self, canonical_id: str, original_circuit: Circuit, epoch: int) -> Circuit:
        """Create legacy circuit for compatibility"""
        canonical = self.canonical_registry.get_canonical_circuit(canonical_id)

        if not canonical:
            return original_circuit

        # Create enhanced metadata with canonical data
        enhanced_metadata = original_circuit.metadata.copy()
        enhanced_metadata.update({
            'canonical_id': canonical_id,
            'is_canonical': True,
            'total_detections': canonical.total_detections,
            'stability_score': canonical.stability_score,
            'consistency_score': canonical.consistency_score,
            'token_examples': list(canonical.token_examples),
            'first_seen': canonical.first_seen,
            'last_seen': canonical.last_seen,
            'attribution_history': canonical.attribution_history,
            'computational_signature': canonical.computational_signature.to_string()
        })

        # Create legacy circuit with canonical ID and aggregated data
        legacy_circuit = Circuit(
            id=canonical_id,  # Use canonical ID for consistency
            type=original_circuit.type,
            elements=original_circuit.elements,
            connections=original_circuit.connections,
            attribution=canonical.best_attribution,  # Use best attribution
            metadata=enhanced_metadata,
            discovered_at=canonical.first_seen
        )

        return legacy_circuit

    def get_canonical_evolution_data(self, canonical_id: str) -> Dict[str, Any]:
        """Get evolution data compatible with UnifiedCircuitEvolutionTracker"""
        canonical = self.canonical_registry.get_canonical_circuit(canonical_id)

        if not canonical:
            return {}

        # Convert to format expected by evolution tracker
        circuit_data = []
        for instance in canonical.instances:
            circuit_data.append({
                'epoch': instance.epoch,
                'attribution': instance.attribution,
                'strength': instance.strength,
                'elements': len(canonical.elements),
                'connections': len(canonical.connections),
                'detection_confidence': instance.detection_confidence
            })

        return {
            'canonical_id': canonical_id,
            'circuit_data': circuit_data,
            'stability_metrics': {
                'stability_score': canonical.stability_score,
                'consistency_score': canonical.consistency_score,
                'persistence_score': canonical.persistence_score
            },
            'evolution_summary': {
                'first_seen': canonical.first_seen,
                'last_seen': canonical.last_seen,
                'total_detections': canonical.total_detections,
                'temporal_span': canonical.last_seen - canonical.first_seen + 1
            }
        }

    def update_adaptive_detector(self, detector):
        """Update adaptive detector to use canonical system"""
        # Store reference to adapter
        detector.canonical_adapter = self

        # Add method to check canonical stability
        def get_canonical_stability(circuit_id_or_data, epoch):
            # Try to find canonical circuit
            if isinstance(circuit_id_or_data, str):
                canonical_id = circuit_id_or_data
            else:
                # Extract from circuit data - would need the actual circuit
                return {'stability_score': 0.5, 'consistency_score': 0.5}

            canonical = self.canonical_registry.get_canonical_circuit(canonical_id)
            if canonical:
                return {
                    'stability_score': canonical.stability_score,
                    'consistency_score': canonical.consistency_score,
                    'persistence_score': canonical.persistence_score,
                    'total_detections': canonical.total_detections,
                    'temporal_span': canonical.last_seen - canonical.first_seen + 1
                }
            return {'stability_score': 0.1, 'consistency_score': 0.1}

        detector.get_canonical_stability = get_canonical_stability

        return detector


# ============================================================================
# EXTENSION FRAMEWORK FOR NEW CIRCUIT TYPES
# ============================================================================

class FunctionalCircuitSignatureExtractor(SignatureExtractor):
    """Example: Extractor for functional composition circuits"""

    def supports_circuit_type(self, circuit: Circuit) -> bool:
        return circuit.type == CircuitType.FUNCTIONAL

    def extract_signature(self, circuit: Circuit) -> ComputationalSignature:
        composition_type = circuit.metadata.get('composition_type', 'unknown')
        token_circuit = circuit.metadata.get('token_circuit', '')
        component_circuit = circuit.metadata.get('component_circuit', '')

        structural_pattern = {
            'composition_type': composition_type,
            'token_circuit_type': self._extract_circuit_type(token_circuit),
            'component_circuit_type': self._extract_circuit_type(component_circuit),
            'interaction_strength': circuit.metadata.get('interaction_strength', 0.0)
        }

        return ComputationalSignature(
            operation_type=f"functional_{composition_type}",
            circuit_type='functional_composition',
            structural_pattern=structural_pattern
        )

    def _extract_circuit_type(self, circuit_id: str) -> str:
        # Extract circuit type from ID
        if 'copy_' in circuit_id:
            return 'copy'
        elif 'induction_' in circuit_id:
            return 'induction'
        elif 'component_' in circuit_id:
            return 'component'
        else:
            return 'unknown'


def extend_canonical_registry_for_new_types(registry: CanonicalCircuitRegistry):
    """Example of how to extend for new circuit types"""

    # Add new extractors
    registry.extractors.append(FunctionalCircuitSignatureExtractor())

    # Could add more extractors for:
    # - Sparse autoencoder features
    # - Superposition circuits
    # - Meta-learning circuits
    # - Compositional reasoning circuits
    # etc.

    return registry
