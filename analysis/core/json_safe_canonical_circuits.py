# analysis/core/json_safe_canonical_circuits.py
"""
JSON-Safe Canonical Circuit System

Integrates existing JSON cleaners to handle numpy int64 and other serialization issues,
particularly for induction circuits with position/distance data.
"""

import json

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import pandas as pd

# Import existing JSON utilities
from analysis.utils.utils import CircuitJSONEncoder, clean_for_json, is_json_serializable
from analysis.core.canonical_circuit_system import (
    CanonicalCircuitRegistry, CanonicalRegistryAdapter,
    CircuitInstance, ComputationalSignature, CanonicalCircuit
)


def clean_circuit_data_for_json(data: Any) -> Any:
    """
    Clean circuit data for JSON serialization using existing utilities

    Handles numpy int64, float64, arrays, and other non-serializable types
    commonly found in induction circuits
    """

    # First pass: Use the comprehensive CircuitJSONEncoder logic
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # Clean the key (convert numpy types to strings if needed)
            clean_key = str(key) if not isinstance(key, (str, int, float)) else key
            # Clean the value recursively
            cleaned[clean_key] = clean_circuit_data_for_json(value)
        return cleaned

    elif isinstance(data, (list, tuple)):
        return [clean_circuit_data_for_json(item) for item in data]

    elif isinstance(data, set):
        return list(clean_circuit_data_for_json(item) for item in data)

    # Handle numpy types specifically (common in induction circuits)
    elif isinstance(data, (np.integer, np.int64, np.int32)):
        return int(data)  # Convert numpy integers to Python int

    elif isinstance(data, (np.floating, np.float64, np.float32)):
        return float(data)  # Convert numpy floats to Python float

    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert arrays to lists

    elif isinstance(data, np.bool_):
        return bool(data)  # Convert numpy bool to Python bool

    # Handle other common non-serializable types
    elif hasattr(data, 'value'):  # Enums
        return data.value

    elif isinstance(data, Path):
        return str(data)

    elif callable(data):
        return f"<function:{getattr(data, '__name__', 'unknown')}>"

    # Test if it's already JSON serializable
    elif is_json_serializable(data):
        return data

    else:
        # Fallback: convert to string
        return str(data)


def safe_circuit_instance_creation(epoch: int, attribution: float, strength: float,
                                   tokens: List[str], positions: Dict[str, Any],
                                   detection_confidence: float, detection_method: str,
                                   example_metadata: Optional[Dict] = None) -> CircuitInstance:
    """Create CircuitInstance with JSON-safe data cleaning"""

    # Clean all data before creating instance
    clean_epoch = int(epoch) if isinstance(epoch, (np.integer, np.int64)) else epoch
    clean_attribution = float(attribution) if isinstance(attribution, (np.floating, np.float64)) else attribution
    clean_strength = float(strength) if isinstance(strength, (np.floating, np.float64)) else strength
    clean_tokens = [str(token) for token in tokens]  # Ensure all tokens are strings

    # Clean positions (this is where int64 issues often occur)
    clean_positions = {}
    for role, pos in positions.items():
        if isinstance(pos, (np.integer, np.int64, np.int32)):
            clean_positions[str(role)] = int(pos)
        elif isinstance(pos, (np.floating, np.float64, np.float32)):
            clean_positions[str(role)] = float(pos)
        else:
            clean_positions[str(role)] = pos

    clean_confidence = float(detection_confidence) if isinstance(detection_confidence,
                                                                 (np.floating, np.float64)) else detection_confidence
    clean_method = str(detection_method)

    # Clean example metadata
    clean_metadata = clean_circuit_data_for_json(example_metadata) if example_metadata else {}

    return CircuitInstance(
        epoch=clean_epoch,
        attribution=clean_attribution,
        strength=clean_strength,
        tokens=clean_tokens,
        positions=clean_positions,
        detection_confidence=clean_confidence,
        detection_method=clean_method,
        example_metadata=clean_metadata
    )


def safe_computational_signature_creation(operation_type: str, circuit_type: str,
                                          structural_pattern: Dict[str, Any]) -> ComputationalSignature:
    """Create ComputationalSignature with JSON-safe structural pattern"""

    # Clean structural pattern (often contains numpy types from induction analysis)
    clean_pattern = clean_circuit_data_for_json(structural_pattern)

    return ComputationalSignature(
        operation_type=str(operation_type),
        circuit_type=str(circuit_type),
        structural_pattern=clean_pattern
    )


class JSONSafeCanonicalCircuitRegistry(CanonicalCircuitRegistry):
    """Enhanced CanonicalCircuitRegistry with JSON safety built-in"""

    def register_circuit_detection(self, circuit, epoch: int, tokens: List[str],
                                   detection_confidence: float = 0.5,
                                   detection_method: str = "unknown",
                                   example_metadata: Optional[Dict] = None) -> str:
        """JSON-safe circuit registration with automatic data cleaning"""

        self.total_registrations += 1

        # Clean all input data
        clean_epoch = int(epoch) if isinstance(epoch, (np.integer, np.int64)) else epoch
        clean_tokens = [str(token) for token in tokens] if tokens else []
        clean_confidence = float(detection_confidence) if isinstance(detection_confidence, (
        np.floating, np.float64)) else detection_confidence
        clean_method = str(detection_method)
        clean_metadata = clean_circuit_data_for_json(example_metadata) if example_metadata else {}

        # Extract computational signature with cleaning
        signature = self._extract_computational_signature_safe(circuit)

        # Get or create canonical ID
        canonical_id = self._get_or_create_canonical_id(signature)

        # Extract position mapping with cleaning
        positions = self._extract_position_mapping_safe(circuit)

        # Create JSON-safe instance
        instance = safe_circuit_instance_creation(
            epoch=clean_epoch,
            attribution=circuit.attribution,
            strength=circuit.attribution,
            tokens=clean_tokens,
            positions=positions,
            detection_confidence=clean_confidence,
            detection_method=clean_method,
            example_metadata=clean_metadata
        )

        # Register or update canonical circuit
        if canonical_id in self.canonical_circuits:
            self._update_canonical_circuit_safe(canonical_id, instance, circuit)
            self.total_aggregations += 1
        else:
            self._create_canonical_circuit_safe(canonical_id, signature, instance, circuit)

        # Update epoch tracking
        self.epoch_to_circuits[clean_epoch].add(canonical_id)

        return canonical_id

    def _extract_computational_signature_safe(self, circuit) -> ComputationalSignature:
        """Extract computational signature with JSON cleaning"""
        for extractor in self.extractors:
            if extractor.supports_circuit_type(circuit):
                signature = extractor.extract_signature(circuit)
                # Clean the structural pattern
                clean_pattern = clean_circuit_data_for_json(signature.structural_pattern)
                return ComputationalSignature(
                    operation_type=str(signature.operation_type),
                    circuit_type=str(signature.circuit_type),
                    structural_pattern=clean_pattern
                )

        # Fallback with cleaning
        return ComputationalSignature(
            operation_type=str(circuit.metadata.get('operation_type', 'unknown')),
            circuit_type='generic',
            structural_pattern={'fallback': True}
        )

    def _extract_position_mapping_safe(self, circuit) -> Dict[str, int]:
        """Extract position mapping with type safety"""
        positions = {}
        for element in circuit.elements:
            if element.properties:
                role = element.properties.get('role')
                pos = element.properties.get('position')
                if role and pos is not None:
                    # Clean both role and position
                    clean_role = str(role)
                    if isinstance(pos, (np.integer, np.int64, np.int32)):
                        clean_pos = int(pos)
                    elif isinstance(pos, (np.floating, np.float64, np.float32)):
                        clean_pos = int(float(pos))  # Convert float positions to int
                    else:
                        clean_pos = int(pos) if str(pos).isdigit() else 0
                    positions[clean_role] = clean_pos
        return positions

    def _create_canonical_circuit_safe(self, canonical_id: str, signature: ComputationalSignature,
                                       instance: CircuitInstance, circuit) -> None:
        """Create canonical circuit with JSON-safe data"""

        # Normalize elements and connections with cleaning
        normalized_elements = self._normalize_elements_safe(circuit.elements)
        normalized_connections = self._normalize_connections_safe(circuit.connections)

        # Clean circuit metadata
        clean_metadata = clean_circuit_data_for_json(circuit.metadata)

        # Create canonical circuit with cleaned data
        canonical = CanonicalCircuit(
            canonical_id=canonical_id,
            computational_signature=signature,
            circuit_type=circuit.type,

            # Initialize with first instance
            instances=[instance],
            token_examples=set([' '.join(instance.tokens)]) if instance.tokens else set(),
            position_patterns=set([self._create_position_pattern_safe(instance.positions)]),

            # Evolution metrics (ensure all are Python types)
            first_seen=int(instance.epoch),
            last_seen=int(instance.epoch),
            total_detections=1,

            # Strength metrics (ensure all are Python types)
            best_attribution=float(instance.attribution),
            current_attribution=float(instance.attribution),
            avg_attribution=float(instance.attribution),
            attribution_history=[(int(instance.epoch), float(instance.attribution))],

            # Stability (Python floats)
            stability_score=0.1,
            consistency_score=0.5,
            persistence_score=0.1,

            # Structure
            elements=normalized_elements,
            connections=normalized_connections,
            metadata=clean_metadata
        )

        self.canonical_circuits[canonical_id] = canonical

    def _update_canonical_circuit_safe(self, canonical_id: str, instance: CircuitInstance, circuit) -> None:
        """Update canonical circuit with JSON-safe data"""
        canonical = self.canonical_circuits[canonical_id]

        # Add instance
        canonical.instances.append(instance)

        # Update aggregated data with type safety
        if instance.tokens:
            canonical.token_examples.add(' '.join(instance.tokens))
        canonical.position_patterns.add(self._create_position_pattern_safe(instance.positions))

        # Update evolution tracking (ensure Python types)
        canonical.last_seen = int(max(canonical.last_seen, instance.epoch))
        canonical.total_detections = int(canonical.total_detections + 1)

        # Update strength metrics (ensure Python types)
        canonical.best_attribution = float(max(canonical.best_attribution, instance.attribution))
        canonical.current_attribution = float(instance.attribution)

        # Update attribution history with type safety
        clean_entry = (int(instance.epoch), float(instance.attribution))
        canonical.attribution_history.append(clean_entry)

        # Recalculate averages
        attributions = [float(attr) for _, attr in canonical.attribution_history]
        canonical.avg_attribution = float(np.mean(attributions))

        # Recalculate stability metrics
        self._update_stability_metrics_safe(canonical)

        # Update metadata with cleaning
        clean_new_metadata = clean_circuit_data_for_json(circuit.metadata)
        canonical.metadata.update(clean_new_metadata)

    def _create_position_pattern_safe(self, positions: Dict[str, int]) -> str:
        """Create position pattern with type safety"""
        if not positions:
            return "no_positions"

        # Ensure all positions are Python ints
        clean_positions = {}
        for role, pos in positions.items():
            clean_role = str(role)
            if isinstance(pos, (np.integer, np.int64, np.int32)):
                clean_pos = int(pos)
            else:
                clean_pos = int(pos) if str(pos).isdigit() else 0
            clean_positions[clean_role] = clean_pos

        # Create relative position pattern
        sorted_roles = sorted(clean_positions.keys())
        if len(sorted_roles) >= 2:
            base_pos = clean_positions[sorted_roles[0]]
            relative_positions = [clean_positions[role] - base_pos for role in sorted_roles]
            return f"pattern_{','.join(map(str, relative_positions))}"
        else:
            return f"single_{sorted_roles[0]}"

    def _normalize_elements_safe(self, elements):
        """Normalize elements with JSON safety"""
        normalized = []

        for element in elements:
            # Clean properties
            norm_props = {}
            if element.properties:
                for key, value in element.properties.items():
                    clean_key = str(key)
                    clean_value = clean_circuit_data_for_json(value)

                    if clean_key in ['role', 'operation', 'name']:
                        norm_props[clean_key] = str(clean_value)
                    elif clean_key == 'token':
                        norm_props['token_type'] = self._classify_token_type(str(clean_value))
                    elif clean_key == 'position':
                        norm_props['has_position'] = True
                        # Store position as int if it's numeric
                        if isinstance(clean_value, (int, float, np.integer, np.floating)):
                            norm_props['position_value'] = int(float(clean_value))

            # Create normalized element
            from analysis.core.circuit_schema import Element
            normalized_elem = Element(
                id=f"{element.type.value}_{norm_props.get('role', 'unknown')}",
                type=element.type,
                properties=norm_props
            )
            normalized.append(normalized_elem)

        return normalized

    def _normalize_connections_safe(self, connections):
        """Normalize connections with JSON safety"""
        normalized = []

        for conn in connections:
            # Clean properties
            norm_props = clean_circuit_data_for_json(conn.properties) if conn.properties else {}

            # Create normalized connection
            from analysis.core.circuit_schema import Connection
            normalized_conn = Connection(
                source=f"source_{norm_props.get('position_type', 'unknown')}",
                target=f"target_{norm_props.get('position_type', 'unknown')}",
                strength=float(conn.strength),  # Ensure Python float
                type=conn.type,
                properties=norm_props
            )
            normalized.append(normalized_conn)

        return normalized

    def _update_stability_metrics_safe(self, canonical: CanonicalCircuit):
        """Update stability metrics with type safety"""
        # Ensure all calculations use Python types
        detection_score = float(min(1.0, canonical.total_detections / 20.0))

        temporal_span = int(canonical.last_seen - canonical.first_seen + 1)
        persistence_score = float(min(1.0, temporal_span / 200.0))

        # Attribution consistency with type safety
        if len(canonical.attribution_history) > 1:
            attributions = [float(attr) for _, attr in canonical.attribution_history]
            attr_std = float(np.std(attributions))
            attr_mean = float(np.mean(attributions))
            consistency_score = float(max(0.0, 1.0 - (attr_std / max(attr_mean, 0.1))))
        else:
            consistency_score = 0.5

        # Token diversity
        diversity_score = float(min(1.0, len(canonical.token_examples) / 10.0))

        # Combined stability score
        stability_score = float(
            detection_score * 0.3 +
            persistence_score * 0.25 +
            consistency_score * 0.25 +
            diversity_score * 0.2
        )

        # Update with Python types
        canonical.stability_score = stability_score
        canonical.consistency_score = consistency_score
        canonical.persistence_score = persistence_score

    def get_circuit_summary_stats(self):
        registry_summary = self.get_registry_summary()
        return {
            'total_canonical_circuits': registry_summary['total_canonical_circuits'],
            'aggregation_rate': registry_summary['aggregation_rate'],
            'stable_circuits_count': registry_summary['stable_circuits_count'],
        }

    def save_with_json_encoder(self, filepath: Optional[Path] = None) -> None:
        """Save registry using CircuitJSONEncoder for full compatibility"""
        if filepath is None:
            if self.storage_dir is None:
                raise ValueError("No storage directory or filepath specified")
            filepath = self.storage_dir / "canonical_circuit_registry.json"

        # Convert canonical circuits to serializable format
        serializable_data = {
            'canonical_circuits': {},
            'signature_to_id': dict(self.signature_to_id),
            'epoch_to_circuits': {str(k): list(v) for k, v in self.epoch_to_circuits.items()},
            'total_registrations': int(self.total_registrations),
            'total_aggregations': int(self.total_aggregations),
            'metadata': {
                'extractor_count': len(self.extractors),
                'save_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown'
            }
        }

        # Convert canonical circuits to dict format
        for cid, circuit in self.canonical_circuits.items():
            circuit_data = {
                'canonical_id': circuit.canonical_id,
                'computational_signature': circuit.computational_signature.to_string(),
                'circuit_type': circuit.circuit_type.value if hasattr(circuit.circuit_type, 'value') else str(
                    circuit.circuit_type),
                'instances': [
                    {
                        'epoch': int(instance.epoch),
                        'attribution': float(instance.attribution),
                        'strength': float(instance.strength),
                        'tokens': list(instance.tokens),
                        'positions': dict(instance.positions),
                        'detection_confidence': float(instance.detection_confidence),
                        'detection_method': str(instance.detection_method),
                        'example_metadata': clean_circuit_data_for_json(instance.example_metadata)
                    }
                    for instance in circuit.instances
                ],
                'token_examples': list(circuit.token_examples),
                'position_patterns': list(circuit.position_patterns),
                'first_seen': int(circuit.first_seen),
                'last_seen': int(circuit.last_seen),
                'total_detections': int(circuit.total_detections),
                'best_attribution': float(circuit.best_attribution),
                'current_attribution': float(circuit.current_attribution),
                'avg_attribution': float(circuit.avg_attribution),
                'attribution_history': [(int(epoch), float(attr)) for epoch, attr in circuit.attribution_history],
                'stability_score': float(circuit.stability_score),
                'consistency_score': float(circuit.consistency_score),
                'persistence_score': float(circuit.persistence_score),
                'metadata': clean_circuit_data_for_json(circuit.metadata)
            }

            serializable_data['canonical_circuits'][cid] = circuit_data

        # Save using CircuitJSONEncoder
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, cls=CircuitJSONEncoder, indent=2)
            print(f"✅ Canonical registry saved successfully to {filepath}")
        except Exception as e:
            print(f"⚠️ Failed to save with CircuitJSONEncoder: {e}")
            # Fallback: use additional cleaning
            try:
                fully_cleaned_data = clean_for_json(serializable_data)
                with open(filepath, 'w') as f:
                    json.dump(fully_cleaned_data, f, indent=2)
                print(f"✅ Canonical registry saved with fallback cleaning to {filepath}")
            except Exception as e2:
                print(f"❌ Failed to save even with fallback cleaning: {e2}")


class JSONSafeCanonicalRegistryAdapter(CanonicalRegistryAdapter):
    """JSON-safe adapter with automatic data cleaning"""

    def __init__(self, enhanced_registry, canonical_registry: JSONSafeCanonicalCircuitRegistry):
        super().__init__(enhanced_registry, canonical_registry)

    def register_circuit_detection(self, circuit, epoch: int, tokens: List[str],
                                   detection_confidence: float = 0.5,
                                   detection_method: str = "unknown",
                                   example_metadata: Optional[Dict] = None) -> Tuple[str, str]:
        """JSON-safe circuit registration with enhanced cleaning"""

        # Clean all inputs before processing
        clean_epoch = int(epoch) if isinstance(epoch, (np.integer, np.int64)) else epoch
        clean_tokens = [str(token) for token in tokens] if tokens else []
        clean_confidence = float(detection_confidence) if isinstance(detection_confidence, (
        np.floating, np.float64)) else detection_confidence
        clean_method = str(detection_method)
        clean_metadata = clean_circuit_data_for_json(example_metadata) if example_metadata else {}

        # Register in canonical system (now JSON-safe)
        canonical_id = self.canonical_registry.register_circuit_detection(
            circuit, clean_epoch, clean_tokens, clean_confidence, clean_method, clean_metadata
        )

        # Create legacy circuit for compatibility
        legacy_circuit = self._create_legacy_circuit_safe(canonical_id, circuit, clean_epoch)

        # Register in enhanced registry
        self.enhanced_registry.register_circuit_enhanced(
            circuit=legacy_circuit,
            source="canonical_detection",
            epoch=clean_epoch,
            detection_method=f"canonical_{clean_method}",
            confidence=clean_confidence
        )

        # Store mapping
        self.canonical_to_legacy[canonical_id] = legacy_circuit.id

        return canonical_id, legacy_circuit.id

    def _create_legacy_circuit_safe(self, canonical_id: str, original_circuit, epoch: int):
        """Create legacy circuit with JSON-safe metadata"""
        canonical = self.canonical_registry.get_canonical_circuit(canonical_id)

        if not canonical:
            return original_circuit

        # Create enhanced metadata with cleaned canonical data
        enhanced_metadata = clean_circuit_data_for_json(original_circuit.metadata.copy())
        enhanced_metadata.update({
            'canonical_id': canonical_id,
            'is_canonical': True,
            'total_detections': int(canonical.total_detections),
            'stability_score': float(canonical.stability_score),
            'consistency_score': float(canonical.consistency_score),
            'token_examples': list(canonical.token_examples),
            'first_seen': int(canonical.first_seen),
            'last_seen': int(canonical.last_seen),
            'attribution_history': [(int(e), float(a)) for e, a in canonical.attribution_history],
            'computational_signature': canonical.computational_signature.to_string()
        })

        # Create legacy circuit with canonical ID and cleaned data
        from analysis.core.circuit_schema import Circuit
        legacy_circuit = Circuit(
            id=canonical_id,
            type=original_circuit.type,
            elements=original_circuit.elements,
            connections=original_circuit.connections,
            attribution=float(canonical.best_attribution),
            metadata=enhanced_metadata,
            discovered_at=int(canonical.first_seen)
        )

        return legacy_circuit


# ============================================================================
# UPDATED INTEGRATION FUNCTIONS
# ============================================================================
'''
def create_json_safe_canonical_system(model, save_dir, logger, enhanced_registry=None):
    """Create JSON-safe canonical circuit system"""

    # Create JSON-safe canonical registry
    canonical_registry = JSONSafeCanonicalCircuitRegistry(save_dir / "canonical_circuits")
    # Extend for additional circuit types
    from analysis.core.canonical_circuit_system import extend_canonical_registry_for_new_types
    canonical_registry = extend_canonical_registry_for_new_types(canonical_registry)

    # Create enhanced registry if not provided
    if enhanced_registry is None:
        from analysis.core import EnhancedCircuitRegistry
        enhanced_registry = EnhancedCircuitRegistry(save_dir / "enhanced_registry")

    # Create JSON-safe adapter
    adapter = JSONSafeCanonicalRegistryAdapter(enhanced_registry, canonical_registry)

    # Initialize evolution tracker
    from analysis.core.unified_circuit_evolution_tracker import UnifiedCircuitEvolutionTracker
    evolution_tracker = UnifiedCircuitEvolutionTracker(enhanced_registry, save_dir / "evolution", logger)

    # Create canonical-aware detector with JSON safety
    from analysis.analyzers.fixed_adaptive token_operations import CanonicalAwareAdaptiveTokenOperationDetector
    canonical_detector = CanonicalAwareAdaptiveTokenOperationDetector(
        model=model,
        enhanced_registry=enhanced_registry,
        canonical_registry=canonical_registry
    )

    print("✅ JSON-safe canonical circuit system initialized")

    return {
        'canonical_registry': canonical_registry,
        'enhanced_registry': enhanced_registry,
        'adapter': adapter,
        'evolution_tracker': evolution_tracker,
        'canonical_detector': canonical_detector
    }
'''
'''
def test_json_safety(canonical_registry: JSONSafeCanonicalCircuitRegistry):
    """Test JSON safety of the canonical registry"""

    print("🧪 Testing JSON safety...")

    # Test serialization of current state
    try:
        test_data = {
            'registry_summary': canonical_registry.get_registry_summary(),
            'circuits': {}
        }

        # Test individual circuits
        for cid, circuit in list(canonical_registry.canonical_circuits.items())[:3]:  # Test first 3
            circuit_analysis = canonical_registry.analyze_circuit_evolution(cid)
            test_data['circuits'][cid] = circuit_analysis

        # Test serialization
        json_str = json.dumps(test_data, cls=CircuitJSONEncoder, indent=2)
        print(f"  ✅ JSON serialization successful: {len(json_str)} characters")

        # Test deserialization
        reloaded = json.loads(json_str)
        print(f"  ✅ JSON deserialization successful: {len(reloaded)} top-level keys")

        return True

    except Exception as e:
        print(f"  ❌ JSON safety test failed: {e}")
        print("  🔧 Applying additional cleaning...")

        # Apply additional cleaning
        cleaned_data = clean_for_json(test_data)
        try:
            json_str = json.dumps(cleaned_data, indent=2)
            print(f"  ✅ JSON serialization with cleaning successful")
            return True
        except Exception as e2:
            print(f"  ❌ Even cleaned data failed: {e2}")
            return False
'''

# ============================================================================
# USAGE EXAMPLE
# ============================================================================
'''
def example_json_safe_usage():
    """Example of using JSON-safe canonical system"""

    # Replace in your training loop:

    # OLD (JSON issues):
    # circuit_system = initialize_canonical_system(model, save_dir, logger)

    # NEW (JSON-safe):
    circuit_system = create_json_safe_canonical_system(model, save_dir, logger)
    canonical_detector = circuit_system['canonical_detector']
    canonical_registry = circuit_system['canonical_registry']

    # Use normally - all numpy int64 issues are automatically handled
    # canonical_results = run_canonical_circuit_analysis_with_sampling(...)

    # Save without JSON errors
    # canonical_registry.save_with_json_encoder()

    # Test JSON safety
    # test_json_safety(canonical_registry)

    pass
'''