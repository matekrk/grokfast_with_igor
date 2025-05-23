# token_circuit_evolution.py
from analysis.utils.utils import get_current_callable_info, shorten_layer_head


class CircuitEvolutionTracker:
    """Tracks the evolution of circuits over training epochs"""

    def __init__(self, registry):
        self.registry = registry
        self.evolution_data = {}  # Circuit ID -> temporal data
        self.emergence_epochs = {}  # Circuit ID -> emergence epoch
        self.circuit_relationships = {}  # (Circuit ID, Circuit ID) -> relationship data

    def update_circuit_evolution(self, epoch, circuits, token_attribution):
        """Track circuit evolution for a specific epoch"""
        # For each circuit, record its state at this epoch
        for circuit in circuits:
            if circuit.id not in self.evolution_data:
                self.evolution_data[circuit.id] = []

            # Store current circuit state
            self.evolution_data[circuit.id].append({
                'epoch': epoch,
                'attribution': circuit.attribution,
                'elements': len(circuit.elements),
                'connections': len(circuit.connections),
                'strength': self._calculate_circuit_strength(circuit, token_attribution)
            })

            # Check if this is the emergence epoch
            if circuit.id not in self.emergence_epochs:
                # Define emergence as first epoch where attribution exceeds threshold
                if circuit.attribution > 0.3:  # Threshold can be adjusted
                    self.emergence_epochs[circuit.id] = epoch

    def _calculate_circuit_strength(self, circuit, token_attribution):
        """Calculate overall circuit strength based on token attribution"""
        # todo implement metrics warning metrics not implement fixme implement metric
        # Implementation depends on your specific metrics
        # Could use connection strengths, token influence, etc.
        return circuit.attribution  # Simplified version

    def analyze_emergence_order(self):
        """Analyze which types of circuits emerge first"""
        # Group circuits by type
        circuits_by_type = {}
        for circuit_id, emergence_epoch in self.emergence_epochs.items():
            circuit = self.registry.get_circuit(circuit_id)
            if circuit and hasattr(circuit, 'type'):
                circuit_type = circuit.type.value
                if circuit_type not in circuits_by_type:
                    circuits_by_type[circuit_type] = []
                circuits_by_type[circuit_type].append((circuit_id, emergence_epoch))

        # Calculate average emergence epoch by type
        avg_emergence = {}
        for circuit_type, circuits in circuits_by_type.items():
            if circuits:
                avg_epoch = sum(epoch for _, epoch in circuits) / len(circuits)
                avg_emergence[circuit_type] = avg_epoch

        return {
            'circuits_by_type': circuits_by_type,
            'avg_emergence': avg_emergence
        }

    def analyze_circuit_relationships(self):
        """Analyze relationships between circuits, including precedence"""
        relationships = []

        # Compare emergence epochs between all circuit pairs
        circuit_ids = list(self.emergence_epochs.keys())
        for i, circuit_id1 in enumerate(circuit_ids):
            for circuit_id2 in circuit_ids[i + 1:]:
                # Get emergence epochs
                epoch1 = self.emergence_epochs.get(circuit_id1)
                epoch2 = self.emergence_epochs.get(circuit_id2)

                if epoch1 is not None and epoch2 is not None:
                    # Check for precedence relationship
                    if abs(epoch1 - epoch2) > 10:  # Significant time difference
                        precedes = circuit_id1 if epoch1 < epoch2 else circuit_id2
                        follows = circuit_id2 if epoch1 < epoch2 else circuit_id1

                        relationships.append({
                            'type': 'precedence',
                            'precedes': precedes,
                            'follows': follows,
                            'epoch_diff': abs(epoch1 - epoch2)
                        })
        print(f"\t{get_current_callable_info()}: \t")

        return relationships