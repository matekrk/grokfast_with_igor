# analysis/validation/compatibility_validator.py
from typing import Dict, Any, List
from analysis.core.circuit_registry import CircuitRegistry, EnhancedCircuitRegistry


class BackwardCompatibilityValidator:
    """Ensure new infrastructure doesn't break existing functionality"""

    def __init__(self, old_registry: CircuitRegistry, new_registry: EnhancedCircuitRegistry):
        self.old_registry = old_registry
        self.new_registry = new_registry

    def validate_circuit_preservation(self) -> Dict[str, bool]:
        """Check that existing circuits are preserved in new registry"""
        results = {}

        for circuit_id in self.old_registry.circuits:
            # Check if circuit exists in new registry
            exists_in_new = circuit_id in self.new_registry.circuits

            if exists_in_new:
                # Check if circuit data is equivalent
                old_circuit = self.old_registry.circuits[circuit_id]
                new_circuit = self.new_registry.circuits[circuit_id]

                data_equivalent = self._circuits_equivalent(old_circuit, new_circuit)
                results[circuit_id] = data_equivalent
            else:
                results[circuit_id] = False

        return results

    def _circuits_equivalent(self, circuit1, circuit2) -> bool:
        """Check if two circuits are functionally equivalent"""
        return (
                circuit1.type == circuit2.type and
                abs(circuit1.attribution - circuit2.attribution) < 1e-6 and
                len(circuit1.elements) == len(circuit2.elements) and
                len(circuit1.connections) == len(circuit2.connections)
        )

    def run_compatibility_tests(self) -> Dict[str, Any]:
        """Run full compatibility test suite"""
        preservation_results = self.validate_circuit_preservation()

        return {
            'circuit_preservation': preservation_results,
            'preservation_rate': sum(preservation_results.values()) / len(
                preservation_results) if preservation_results else 1.0,
            'registry_size_comparison': {
                'old_registry_size': len(self.old_registry.circuits),
                'new_registry_size': len(self.new_registry.circuits)
            },
            'total_circuits_tested': len(preservation_results)
        }
