# analysis/validation/enhanced_circuit_validation.py
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from analysis.validation.circuit_manipulation import CircuitManipulationValidator


class EnhancedCircuitValidator(CircuitManipulationValidator):
    """Enhanced validation with content-aware testing"""

    def __init__(self, model, eval_loader, content_analyzer=None, batch_limit=3):
        super().__init__(model, eval_loader, batch_limit)
        self.content_analyzer = content_analyzer

    def validate_content_aware_circuit(self, circuit_data: Dict,
                                       validation_methods: List[str] = None) -> Dict[str, float]:
        """Validate circuit with content-aware analysis"""

        if validation_methods is None:
            validation_methods = ["ablation", "content_specific_ablation"]

        results = {}
        baseline_performance = self._get_baseline_performance()

        for method in validation_methods:
            if method == "content_specific_ablation":
                score = self._content_specific_ablation(circuit_data, baseline_performance)
            else:
                # Use parent class methods
                if hasattr(super(), f"_{method}_validation"):
                    score = getattr(super(), f"_{method}_validation")(circuit_data, baseline_performance)
                else:
                    score = 0.0

            results[method] = score

        return results

    def _content_specific_ablation(self, circuit_data: Dict, baseline: float) -> float:
        """Test circuit by ablating only on specific content types"""

        # Get circuit's semantic type
        copy_type = circuit_data.get("copy_type", "unknown")

        if copy_type == "operand_to_result":
            # Test specifically on arithmetic problems
            return self._test_on_arithmetic_problems(circuit_data, baseline)
        elif copy_type == "operand_to_operand":
            # Test on operand manipulation tasks
            return self._test_on_operand_tasks(circuit_data, baseline)
        else:
            # Fall back to general ablation
            return self._standard_ablation(circuit_data, baseline)

    def _test_on_arithmetic_problems(self, circuit_data: Dict, baseline: float) -> float:
        """Test circuit specifically on arithmetic computation"""
        # This would require selecting specific types of problems
        # For now, use standard ablation but could be enhanced
        return self._standard_ablation(circuit_data, baseline)

    def _standard_ablation(self, circuit_data: Dict, baseline: float) -> float:
        """Standard ablation testing"""
        # Convert circuit_data to Circuit object for parent class method
        from analysis.core.circuit_schema import Circuit, CircuitType, Element, ElementType

        # Create mock circuit from circuit_data
        head_element = Element(
            id=circuit_data.get("head", "unknown"),
            type=ElementType.HEAD
        )

        circuit = Circuit(
            id=f"temp_circuit_{circuit_data.get('head', 'unknown')}",
            type=CircuitType.TOKEN,
            elements=[head_element],
            attribution=circuit_data.get("attention_strength", 0.5)
        )

        return self._ablation_validation(circuit, baseline)
