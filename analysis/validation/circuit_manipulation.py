# analysis/validation/circuit_manipulation.py
import torch
import numpy as np
import time
from typing import Dict, List, Optional, Any

from analysis.core.circuit_schema import Circuit


class CircuitManipulationValidator:
    """Validate circuits through systematic manipulation (ablation, scaling, noise)"""

    def __init__(self, model, eval_loader, batch_limit: int = 3):
        """
        Initialize circuit manipulation validator

        Args:
            model: The transformer model to validate on
            eval_loader: Data loader for evaluation
            batch_limit: Maximum batches to use for validation (for speed)
        """
        self.model = model
        self.eval_loader = eval_loader
        self.batch_limit = batch_limit
        self.manipulation_history = {}

        print("✅ Circuit manipulation validator initialized")

    def validate_circuit_importance(self, circuit: Circuit,
                                    validation_methods: List[str] = None) -> Dict[str, float]:
        """
        Validate circuit importance through multiple manipulation methods

        Args:
            circuit: Circuit to validate
            validation_methods: List of methods to use

        Returns:
            Dictionary of method -> importance_score
        """
        if validation_methods is None:
            validation_methods = ["ablation", "scaling"]  # Start with basic methods

        results = {}
        baseline_performance = self._get_baseline_performance()

        print(f"🔍 Validating circuit {circuit.id} with baseline performance {baseline_performance:.3f}")

        for method in validation_methods:
            start_time = time.time()

            try:
                if method == "ablation":
                    score = self._ablation_validation(circuit, baseline_performance)
                elif method == "scaling":
                    score = self._scaling_validation(circuit, baseline_performance)
                elif method == "noise_injection":
                    score = self._noise_validation(circuit, baseline_performance)
                elif method == "directional_intervention":
                    score = self._directional_validation(circuit, baseline_performance)
                else:
                    print(f"⚠️ Unknown validation method: {method}")
                    score = 0.0

                results[method] = score
                execution_time = time.time() - start_time
                print(f"  ✅ {method}: {score:.3f} (took {execution_time:.2f}s)")

            except Exception as e:
                print(f"  ❌ {method} failed: {e}")
                results[method] = 0.0

        return results

    def _get_baseline_performance(self) -> float:
        """Get baseline model performance (accuracy)"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.eval_loader):
                if i >= self.batch_limit:
                    break

                outputs = self.model(inputs)
                predicted = outputs.argmax(dim=-1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        return correct / total if total > 0 else 0.0

    def _ablation_validation(self, circuit: Circuit, baseline: float) -> float:
        """Validate by completely removing circuit components"""
        # Store original state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        try:
            # Ablate circuit components
            self._ablate_circuit_components(circuit)

            # Measure performance without circuit
            ablated_performance = self._get_baseline_performance()

            # Calculate importance as performance drop
            importance = baseline - ablated_performance

        finally:
            # Always restore original state
            self.model.load_state_dict(original_state)

        return max(0.0, importance)  # Only positive importance scores

    def _scaling_validation(self, circuit: Circuit, baseline: float) -> float:
        """Validate by scaling circuit strength"""
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        scaling_factors = [0.5, 2.0]  # Test weaker and stronger versions
        effects = []

        try:
            for factor in scaling_factors:
                # Scale circuit components
                self._scale_circuit_components(circuit, factor)

                # Measure performance
                scaled_performance = self._get_baseline_performance()
                effect = abs(scaled_performance - baseline)
                effects.append(effect)

                # Restore for next test
                self.model.load_state_dict(original_state)

            # Return average effect magnitude
            return np.mean(effects) if effects else 0.0

        finally:
            self.model.load_state_dict(original_state)

    def _noise_validation(self, circuit: Circuit, baseline: float) -> float:
        """Validate by adding noise to circuit components"""
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        noise_levels = [0.05, 0.1]  # Light and moderate noise
        effects = []

        try:
            for noise_level in noise_levels:
                # Add noise to circuit components
                self._add_noise_to_circuit(circuit, noise_level)

                # Measure performance
                noisy_performance = self._get_baseline_performance()
                effect = abs(noisy_performance - baseline)
                effects.append(effect)

                # Restore for next test
                self.model.load_state_dict(original_state)

            return np.mean(effects) if effects else 0.0

        finally:
            self.model.load_state_dict(original_state)

    def _ablate_circuit_components(self, circuit: Circuit):
        """Remove circuit components from model"""
        for element in circuit.elements:
            if element.type.name == 'HEAD':
                self._ablate_attention_head(element.id)
            elif element.type.name == 'MLP':
                self._ablate_mlp_component(element.id)

    def _scale_circuit_components(self, circuit: Circuit, factor: float):
        """Scale circuit components by given factor"""
        for element in circuit.elements:
            if element.type.name == 'HEAD':
                self._scale_attention_head(element.id, factor)
            elif element.type.name == 'MLP':
                self._scale_mlp_component(element.id, factor)

    def _add_noise_to_circuit(self, circuit: Circuit, noise_level: float):
        """Add noise to circuit components"""
        for element in circuit.elements:
            if element.type.name == 'HEAD':
                self._add_noise_to_head(element.id, noise_level)
            elif element.type.name == 'MLP':
                self._add_noise_to_mlp(element.id, noise_level)

    def _ablate_attention_head(self, head_id: str):
        """Zero out specific attention head"""
        if '_' in head_id:
            parts = head_id.split('_')
            if len(parts) >= 4 and parts[0] == 'layer' and parts[2] == 'head':
                try:
                    layer_idx = int(parts[1])
                    head_idx = int(parts[3])

                    if layer_idx < len(self.model.layers):
                        layer = self.model.layers[layer_idx]
                        head_dim = self.model.dim // self.model.num_heads
                        start_idx = head_idx * head_dim
                        end_idx = (head_idx + 1) * head_dim

                        with torch.no_grad():
                            layer.attn.out_proj.weight[:, start_idx:end_idx] = 0
                except (ValueError, IndexError, AttributeError) as e:
                    print(f"⚠️ Could not ablate head {head_id}: {e}")

    def _scale_attention_head(self, head_id: str, factor: float):
        """Scale attention head weights"""
        if '_' in head_id:
            parts = head_id.split('_')
            if len(parts) >= 4 and parts[0] == 'layer' and parts[2] == 'head':
                try:
                    layer_idx = int(parts[1])
                    head_idx = int(parts[3])

                    if layer_idx < len(self.model.layers):
                        layer = self.model.layers[layer_idx]
                        head_dim = self.model.dim // self.model.num_heads
                        start_idx = head_idx * head_dim
                        end_idx = (head_idx + 1) * head_dim

                        with torch.no_grad():
                            layer.attn.out_proj.weight[:, start_idx:end_idx] *= factor
                except (ValueError, IndexError, AttributeError) as e:
                    print(f"⚠️ Could not scale head {head_id}: {e}")

    def _add_noise_to_head(self, head_id: str, noise_level: float):
        """Add noise to attention head"""
        if '_' in head_id:
            parts = head_id.split('_')
            if len(parts) >= 4 and parts[0] == 'layer' and parts[2] == 'head':
                try:
                    layer_idx = int(parts[1])
                    head_idx = int(parts[3])

                    if layer_idx < len(self.model.layers):
                        layer = self.model.layers[layer_idx]
                        head_dim = self.model.dim // self.model.num_heads
                        start_idx = head_idx * head_dim
                        end_idx = (head_idx + 1) * head_dim

                        with torch.no_grad():
                            weights = layer.attn.out_proj.weight[:, start_idx:end_idx]
                            noise = torch.randn_like(weights) * noise_level * torch.norm(weights)
                            layer.attn.out_proj.weight[:, start_idx:end_idx] += noise
                except (ValueError, IndexError, AttributeError) as e:
                    print(f"⚠️ Could not add noise to head {head_id}: {e}")

    def _ablate_mlp_component(self, mlp_id: str):
        """Zero out MLP component - placeholder for now"""
        # This would implement MLP ablation
        # For now, just log the attempt
        print(f"📝 MLP ablation not yet implemented for {mlp_id}")

    def _scale_mlp_component(self, mlp_id: str, factor: float):
        """Scale MLP component - placeholder for now"""
        print(f"📝 MLP scaling not yet implemented for {mlp_id}")

    def _add_noise_to_mlp(self, mlp_id: str, noise_level: float):
        """Add noise to MLP component - placeholder for now"""
        print(f"📝 MLP noise injection not yet implemented for {mlp_id}")
