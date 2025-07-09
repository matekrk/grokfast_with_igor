# analysis/helpers/circuit_evolution_helper.py (CORRECTED VERSION)
from typing import Dict, List, Set, Any
from collections import defaultdict
import numpy as np

from analysis import Circuit


# ============================================================================
# SAFE ACCESS HELPER FUNCTIONS (NEW)
# ============================================================================

def safe_get_from_analysis_results(results, result_type, key, default=None):
    """Safe helper function to get values from analysis results"""
    if (results is not None and
            isinstance(results, dict) and
            result_type in results and
            results[result_type] is not None and
            isinstance(results[result_type], dict)):
        return results[result_type].get(key, default)
    return default


def safe_get_circuits_from_results(results, result_type):
    """Safe helper to get circuits list from analysis results"""
    circuits = safe_get_from_analysis_results(results, result_type, 'circuits', [])
    return circuits if isinstance(circuits, list) else []


# ============================================================================
# CIRCUIT EVOLUTION TRACKER CLASS
# ============================================================================

class CircuitEvolutionTracker:
    """Track circuit stability and evolution patterns"""

    def __init__(self, registry):
        self.registry = registry
        self.circuit_lifetimes = {}  # circuit_id -> (birth_epoch, death_epoch)
        self.stability_scores = {}  # circuit_id -> stability_score
        self.evolution_events = []  # List of evolution events

    def track_epoch_circuits(self, epoch: int, detected_circuits: List[Dict]):
        """Track circuits detected in this epoch"""
        current_circuit_ids = set()

        for circuit_data in detected_circuits:
            circuit_id = self._generate_circuit_id(circuit_data)
            current_circuit_ids.add(circuit_id)

            # Track birth
            if circuit_id not in self.circuit_lifetimes:
                self.circuit_lifetimes[circuit_id] = (epoch, None)
                self.evolution_events.append({
                    "type": "birth",
                    "circuit_id": circuit_id,
                    "epoch": epoch,
                    "strength": circuit_data.get("attention_strength", 0)
                })

        # Track deaths (circuits not seen recently)
        for circuit_id, (birth, death) in self.circuit_lifetimes.items():
            if death is None and circuit_id not in current_circuit_ids:
                # Check if truly dead (not seen for multiple epochs)
                if self._should_mark_as_dead(circuit_id, epoch):
                    self.circuit_lifetimes[circuit_id] = (birth, epoch)
                    self.evolution_events.append({
                        "type": "death",
                        "circuit_id": circuit_id,
                        "epoch": epoch,
                        "lifetime": epoch - birth
                    })

        # Update stability scores
        self._update_stability_scores(epoch)

    def get_stable_circuits(self, current_epoch, min_lifetime: int = 10, min_stability: float = 0.7):
        """Get circuits that have proven stable over time"""
        stable_circuits = []

        for circuit_id, stability in self.stability_scores.items():
            birth, death = self.circuit_lifetimes[circuit_id]
            lifetime = (death or current_epoch) - birth

            if lifetime >= min_lifetime and stability >= min_stability:
                stable_circuits.append({
                    "circuit_id": circuit_id,
                    "stability": stability,
                    "lifetime": lifetime,
                    "birth_epoch": birth,
                    "death_epoch": death
                })

        return sorted(stable_circuits, key=lambda x: x["stability"], reverse=True)

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of circuit evolution patterns"""
        births = [e for e in self.evolution_events if e["type"] == "birth"]
        deaths = [e for e in self.evolution_events if e["type"] == "death"]

        return {
            "total_circuits_discovered": len(births),
            "total_circuits_died": len(deaths),
            "survival_rate": 1 - len(deaths) / max(1, len(births)),
            "avg_lifetime": np.mean([e["lifetime"] for e in deaths]) if deaths else 0,
            "stable_circuits": len(self.get_stable_circuits())
        }

    def _generate_circuit_id(self, circuit_data):
        """Generate circuit ID from circuit data"""
        # Implementation depends on your circuit data structure
        return circuit_data.get("id", f"circuit_{hash(str(circuit_data))}")

    def _should_mark_as_dead(self, circuit_id, current_epoch):
        """Determine if circuit should be marked as dead"""
        # Mark as dead if not seen for 20+ epochs
        birth, _ = self.circuit_lifetimes[circuit_id]
        return current_epoch - birth > 20

    def _update_stability_scores(self, epoch):
        """Update stability scores for all circuits"""
        for circuit_id in self.circuit_lifetimes:
            # Calculate stability based on lifetime and consistency
            birth, death = self.circuit_lifetimes[circuit_id]
            lifetime = (death or epoch) - birth
            self.stability_scores[circuit_id] = min(1.0, lifetime / 50.0)  # Normalize to 0-1


# ============================================================================
# FUNCTIONAL CIRCUIT ANALYSIS (CORRECTED)
# ============================================================================

def analyze_functional_circuits(registry, epoch: int, eval_loader, thresholds,
                                accuracy: float) -> Dict[str, Any]:
    """Analyze functional circuits (Token + Component combinations) - SAFE VERSION"""
    from analysis import CircuitType

    # ✅ SAFE: Check registry validity
    if not registry:
        return {"functional_circuits": [], "error": "No registry provided"}

    try:
        token_circuits = registry.get_circuits_by_type(CircuitType.TOKEN)
        component_circuits = registry.get_circuits_by_type(CircuitType.COMPONENT)
    except Exception as e:
        return {"functional_circuits": [], "error": f"Failed to get circuits from registry: {e}"}

    functional_circuits = []

    # Find token-component combinations that work together
    for token_circuit in token_circuits:
        for component_circuit in component_circuits:
            try:
                interaction_strength = calculate_circuit_interaction(token_circuit, component_circuit)
            except Exception as e:
                # Skip failed interactions
                continue

            # ✅ SAFE: Check thresholds validity
            try:
                threshold = thresholds.get_threshold("functional", epoch, 1000, accuracy) if thresholds else 0.3
            except:
                threshold = 0.3

            if interaction_strength > threshold:
                try:
                    functional_circuit = create_functional_circuit(
                        token_circuit, component_circuit, interaction_strength, epoch
                    )
                    functional_circuits.append(functional_circuit)

                    # ✅ SAFE: Register with enhanced metadata
                    if hasattr(registry, 'register_circuit_enhanced'):
                        try:
                            registry.register_circuit_enhanced(
                                functional_circuit,
                                source="functional_analysis",
                                epoch=epoch,
                                detection_method="functional_detector",
                                confidence=interaction_strength
                            )
                        except:
                            # Fallback to basic registration
                            registry.register_circuit(functional_circuit, source="functional_analysis")
                except Exception as e:
                    # Skip failed circuit creation
                    continue

    return {"functional_circuits": functional_circuits}


def calculate_circuit_interaction(token_circuit: Circuit, component_circuit: Circuit) -> float:
    """
    Calculate interaction strength between token and component circuits

    Returns:
        float: Interaction strength (0.0 to 1.0)
    """
    # ✅ SAFE: Check circuit validity
    if not token_circuit or not component_circuit:
        return 0.0

    if not hasattr(token_circuit, 'elements') or not hasattr(component_circuit, 'elements'):
        return 0.0

    try:
        # Check for shared components
        token_elements = set(e.id for e in token_circuit.elements if hasattr(e, 'id'))
        component_elements = set(e.id for e in component_circuit.elements if hasattr(e, 'id'))

        shared_elements = token_elements.intersection(component_elements)
        if not shared_elements:
            return 0.0

        # Base interaction from shared components
        base_interaction = len(shared_elements) / max(len(token_elements), len(component_elements))

        # Boost if they complement each other
        functional_boost = 0.0
        if (hasattr(token_circuit, 'metadata') and hasattr(component_circuit, 'metadata') and
                token_circuit.metadata and component_circuit.metadata):

            token_type = token_circuit.metadata.get('operation_type', '')
            comp_type = component_circuit.metadata.get('operation_type', '')

            # Copy token + attention head = strong functional relationship
            if token_type == 'copy' and 'head' in comp_type:
                functional_boost = 0.3
            # Induction token + head interaction = strong relationship
            elif token_type == 'induction' and 'interaction' in comp_type:
                functional_boost = 0.4

        return min(1.0, base_interaction + functional_boost)

    except Exception as e:
        # Return 0 for any calculation errors
        return 0.0


# ============================================================================
# CIRCUIT PRUNING (CORRECTED)
# ============================================================================

def prune_unstable_circuits(registry, epoch: int, min_stability_epochs: int = 5) -> List[str]:
    """
    Prune circuits that haven't been stable for minimum epochs

    Args:
        registry: Enhanced circuit registry
        epoch: Current epoch
        min_stability_epochs: Minimum epochs a circuit should be seen to be considered stable

    Returns:
        List of circuit IDs that were removed
    """
    from analysis.core.circuit_schema import CircuitStability

    # ✅ SAFE: Check registry validity
    if not registry or not hasattr(registry, 'circuit_metadata'):
        return []

    circuits_to_remove = []

    # Check each circuit's stability
    for circuit_id, metadata in registry.circuit_metadata.items():
        if not metadata:
            continue

        # Criteria for pruning:
        # 1. Circuit is transient or emerging
        # 2. Hasn't been seen for enough epochs
        # 3. Haven't seen it recently

        epochs_seen = len(getattr(metadata, 'detection_epochs', []))
        last_seen = getattr(metadata, 'last_seen', 0)
        epochs_since_last_seen = epoch - last_seen

        should_prune = False
        prune_reason = ""

        # ✅ SAFE: Check stability attribute exists
        stability = getattr(metadata, 'stability', CircuitStability.TRANSIENT)
        stability_score = getattr(metadata, 'stability_score', 0.0)

        # Prune if transient and not seen enough
        if (stability == CircuitStability.TRANSIENT and
                epochs_seen < min_stability_epochs and
                epochs_since_last_seen > 20):
            should_prune = True
            prune_reason = f"transient_insufficient_sightings_{epochs_seen}_last_seen_{epochs_since_last_seen}_epochs_ago"

        # Prune if emerging but hasn't been seen recently
        elif (stability == CircuitStability.EMERGING and
              epochs_since_last_seen > 50):
            should_prune = True
            prune_reason = f"emerging_not_seen_recently_{epochs_since_last_seen}_epochs_ago"

        # Prune if very low stability score
        elif stability_score < 0.1 and epochs_since_last_seen > 30:
            should_prune = True
            prune_reason = f"low_stability_score_{stability_score:.3f}"

        if should_prune:
            circuits_to_remove.append({
                "circuit_id": circuit_id,
                "reason": prune_reason,
                "epochs_seen": epochs_seen,
                "last_seen": last_seen,
                "stability": stability.value if hasattr(stability, 'value') else str(stability),
                "stability_score": stability_score
            })

    # Remove unstable circuits from registry
    removed_ids = []
    for circuit_info in circuits_to_remove:
        circuit_id = circuit_info["circuit_id"]

        try:
            if hasattr(registry, 'circuits') and circuit_id in registry.circuits:
                del registry.circuits[circuit_id]
                removed_ids.append(circuit_id)

            if hasattr(registry, 'circuit_metadata') and circuit_id in registry.circuit_metadata:
                del registry.circuit_metadata[circuit_id]

            # Also remove from relationship tracking
            if hasattr(registry, 'relationship_graph') and circuit_id in registry.relationship_graph:
                del registry.relationship_graph[circuit_id]

            # Remove references to this circuit in other relationships
            if hasattr(registry, 'relationship_graph'):
                for other_circuit, relationships in registry.relationship_graph.items():
                    if circuit_id in relationships:
                        del relationships[circuit_id]

            print(f"\t\tPruned circuit {circuit_id}: {circuit_info['reason']}")
        except Exception as e:
            print(f"\t\tFailed to prune circuit {circuit_id}: {e}")

    if removed_ids:
        print(f"\t\tTotal pruned: {len(removed_ids)} unstable circuits")

    return removed_ids


# ============================================================================
# CIRCUIT CREATION (CORRECTED)
# ============================================================================

def create_functional_circuit(token_circuit: Circuit, component_circuit: Circuit,
                              strength: float, epoch: int) -> Circuit:
    """Create a functional circuit from token + component combination"""

    # ✅ SAFE: Check circuit validity
    if not token_circuit or not component_circuit:
        raise ValueError("Both circuits must be provided")

    if not hasattr(token_circuit, 'id') or not hasattr(component_circuit, 'id'):
        raise ValueError("Circuits must have valid IDs")

    # Create unique ID
    circuit_id = f"functional_{token_circuit.id}_{component_circuit.id}_{epoch}"

    # Combine elements from both circuits
    combined_elements = []

    # ✅ SAFE: Check elements exist
    if hasattr(token_circuit, 'elements') and token_circuit.elements:
        combined_elements.extend(token_circuit.elements)

    # Add component elements that aren't already included
    if hasattr(component_circuit, 'elements') and component_circuit.elements:
        for element in component_circuit.elements:
            if not any(hasattr(e, 'id') and hasattr(element, 'id') and e.id == element.id
                       for e in combined_elements):
                combined_elements.append(element)

    # Create functional connections
    functional_connections = []

    # ✅ SAFE: Check connections exist
    if hasattr(token_circuit, 'connections') and token_circuit.connections:
        functional_connections.extend(token_circuit.connections)
    if hasattr(component_circuit, 'connections') and component_circuit.connections:
        functional_connections.extend(component_circuit.connections)

    # Add meta-connection showing functional relationship
    if (hasattr(token_circuit, 'elements') and token_circuit.elements and
            hasattr(component_circuit, 'elements') and component_circuit.elements):
        from analysis import Connection, ConnectionType

        functional_connections.append(Connection(
            source=token_circuit.elements[0].id,
            target=component_circuit.elements[0].id,
            strength=strength,
            type=ConnectionType.COMPOSITE,
            properties={"relationship": "implements"}
        ))

    # Create functional circuit
    from analysis import Circuit, CircuitType

    return Circuit(
        id=circuit_id,
        type=CircuitType.FUNCTIONAL,
        elements=combined_elements,
        connections=functional_connections,
        attribution=strength,
        metadata={
            "operation_type": "functional_composition",
            "token_circuit": token_circuit.id,
            "component_circuit": component_circuit.id,
            "composition_type": "token_plus_component",
            "interaction_strength": strength
        },
        discovered_at=epoch
    )


# ============================================================================
# CIRCUIT RELATIONSHIP ANALYSIS (CORRECTED)
# ============================================================================

def analyze_circuit_relationships_enhanced(registry, epoch: int, lineage: Dict = None) -> None:
    """Enhanced relationship analysis with all circuit types"""
    from analysis.core.circuit_schema import EmergencePhase, RelationshipType

    # ✅ SAFE: Check registry validity
    if not registry:
        return

    # Analyze prerequisite relationships
    prerequisite_relationships = []

    try:
        # Check if early circuits enable later circuits
        early_circuits = registry.get_circuits_by_phase(EmergencePhase.EARLY) if hasattr(registry,
                                                                                         'get_circuits_by_phase') else []
        later_circuits = registry.get_circuits_by_phase(EmergencePhase.MIDDLE) if hasattr(registry,
                                                                                          'get_circuits_by_phase') else []

        for early in early_circuits:
            for later in later_circuits:
                if has_prerequisite_relationship(early, later):
                    if hasattr(registry, 'add_relationship'):
                        registry.add_relationship(
                            early.id, later.id, RelationshipType.PREREQUISITE, strength=0.8
                        )
                    prerequisite_relationships.append((early.id, later.id))
    except Exception as e:
        print(f"Error analyzing prerequisite relationships: {e}")

    # Analyze competitive relationships
    competitive_relationships = []

    try:
        all_circuits = list(registry.circuits.values()) if hasattr(registry, 'circuits') else []

        for i, circuit1 in enumerate(all_circuits):
            for circuit2 in all_circuits[i + 1:]:
                if has_competitive_relationship(circuit1, circuit2):
                    if hasattr(registry, 'add_relationship'):
                        registry.add_relationship(
                            circuit1.id, circuit2.id, RelationshipType.COMPETITIVE, strength=0.6
                        )
                    competitive_relationships.append((circuit1.id, circuit2.id))
    except Exception as e:
        print(f"Error analyzing competitive relationships: {e}")

    print(f"\t\tRelationship analysis: {len(prerequisite_relationships)} prerequisites, "
          f"{len(competitive_relationships)} competitive")


def has_prerequisite_relationship(early_circuit: Circuit, later_circuit: Circuit) -> bool:
    """Check if early circuit is prerequisite for later circuit"""
    # ✅ SAFE: Check circuit validity
    if not early_circuit or not later_circuit:
        return False

    if not (hasattr(early_circuit, 'elements') and hasattr(later_circuit, 'elements')):
        return False

    try:
        early_components = set(e.id for e in early_circuit.elements if hasattr(e, 'id'))
        later_components = set(e.id for e in later_circuit.elements if hasattr(e, 'id'))

        shared = early_components.intersection(later_components)

        # If later circuit uses components from early circuit
        return len(shared) > 0 and len(shared) >= len(early_components) * 0.5
    except Exception as e:
        return False


def has_competitive_relationship(circuit1: Circuit, circuit2: Circuit) -> bool:
    """Check if circuits compete for same resources"""
    # ✅ SAFE: Check circuit validity
    if not circuit1 or not circuit2:
        return False

    if not (hasattr(circuit1, 'elements') and hasattr(circuit2, 'elements')):
        return False

    try:
        components1 = set(e.id for e in circuit1.elements if hasattr(e, 'id'))
        components2 = set(e.id for e in circuit2.elements if hasattr(e, 'id'))

        shared_components = components1.intersection(components2)

        if len(shared_components) == 0:
            return False

        # Check if functions are different
        op1 = circuit1.metadata.get('operation_type', 'unknown') if hasattr(circuit1,
                                                                            'metadata') and circuit1.metadata else 'unknown'
        op2 = circuit2.metadata.get('operation_type', 'unknown') if hasattr(circuit2,
                                                                            'metadata') and circuit2.metadata else 'unknown'

        return op1 != op2 and len(shared_components) >= 2
    except Exception as e:
        return False


# ============================================================================
# CIRCUIT STABILITY TRACKING (CORRECTED)
# ============================================================================

def track_circuit_stability_evolution(registry, epoch: int) -> Dict[str, Any]:
    """Track how circuit stability evolves over time"""
    stability_evolution = {
        'epoch': epoch,
        'stability_changes': [],
        'emerging_circuits': [],
        'declining_circuits': [],
        'stable_circuits': []
    }

    # ✅ SAFE: Check registry validity
    if not registry or not hasattr(registry, 'circuit_metadata'):
        return stability_evolution

    from analysis.core.circuit_schema import CircuitStability

    try:
        for circuit_id, metadata in registry.circuit_metadata.items():
            if not metadata:
                continue

            current_stability = getattr(metadata, 'stability', CircuitStability.TRANSIENT)

            # Track circuits by current stability
            if current_stability == CircuitStability.EMERGING:
                stability_evolution['emerging_circuits'].append(circuit_id)
            elif current_stability == CircuitStability.DECLINING:
                stability_evolution['declining_circuits'].append(circuit_id)
            elif current_stability == CircuitStability.STABLE:
                stability_evolution['stable_circuits'].append(circuit_id)
    except Exception as e:
        print(f"Error tracking stability evolution: {e}")

    return stability_evolution


def calculate_circuit_emergence_rate(registry, window_epochs: int = 50) -> Dict[str, float]:
    """Calculate rate of circuit emergence over recent epochs"""
    # ✅ SAFE: Check registry validity
    if not registry or not hasattr(registry, 'get_emergence_timeline'):
        return {}

    try:
        emergence_timeline = registry.get_emergence_timeline()

        if not emergence_timeline:
            return {}

        recent_epochs = sorted([e for e in emergence_timeline.keys()
                                if max(emergence_timeline.keys()) - e <= window_epochs])

        if len(recent_epochs) < 2:
            return {}

        # Calculate emergence rate by circuit type
        emergence_rates = {}

        from analysis import CircuitType

        for circuit_type in CircuitType:
            type_emergences = []
            for epoch in recent_epochs:
                epoch_circuits = emergence_timeline.get(epoch, [])
                type_count = sum(1 for cid in epoch_circuits
                                 if (cid in registry.circuits and
                                     hasattr(registry.circuits[cid], 'type') and
                                     registry.circuits[cid].type == circuit_type))
                type_emergences.append(type_count)

            if type_emergences:
                # Calculate average emergence rate per epoch
                total_emergences = sum(type_emergences)
                emergence_rates[circuit_type.value] = total_emergences / len(recent_epochs)

        return emergence_rates
    except Exception as e:
        print(f"Error calculating emergence rate: {e}")
        return {}