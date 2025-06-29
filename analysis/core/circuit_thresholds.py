# analysis/core/circuit_thresholds.py
import numpy as np
from dataclasses import dataclass


@dataclass
class CircuitThresholds:
    """Adaptive thresholds that change during training"""
    # Base thresholds for different circuit types
    copy_attention_min: float = 0.3
    copy_attention_max: float = 0.9
    induction_attention_min: float = 0.4
    induction_attention_max: float = 0.85
    component_interaction_min: float = 0.1
    component_interaction_max: float = 0.5

    # Detection parameters
    consistency_min_examples: int = 2
    stability_required_epochs: int = 5
    warmup_epochs: int = 50
    min_accuracy_threshold: float = 0.2

    # Content analysis parameters
    content_similarity_threshold: float = 0.7
    semantic_boost_factor: float = 1.2
    positional_penalty_factor: float = 0.9

    def get_threshold(self, circuit_type: str, epoch: int, total_epochs: int = 1000,
                      model_accuracy: float = 0.0) -> float:
        """Get adaptive threshold for circuit type at current training state"""

        # Determine training progress (0.0 to 1.0)
        progress = min(1.0, epoch / (total_epochs * 0.8))

        # Select base thresholds based on circuit type
        if circuit_type.lower() == "copy":
            min_thresh, max_thresh = self.copy_attention_min, self.copy_attention_max
        elif circuit_type.lower() == "induction":
            min_thresh, max_thresh = self.induction_attention_min, self.induction_attention_max
        elif circuit_type.lower() == "component":
            min_thresh, max_thresh = self.component_interaction_min, self.component_interaction_max
        elif circuit_type.lower() == "functional":
            min_thresh, max_thresh = 0.2, 0.7  # Functional circuits need lower thresholds
        elif circuit_type.lower() == "subspace":
            min_thresh, max_thresh = 0.15, 0.6  # Subspace circuits are subtle
        else:
            min_thresh, max_thresh = 0.3, 0.8  # Default values

        # Interpolate based on training progress (start high, go lower)
        base_threshold = max_thresh - (max_thresh - min_thresh) * progress

        # Adjust based on model accuracy (higher accuracy allows lower thresholds)
        accuracy_factor = 1.0 - (model_accuracy * 0.3)  # Max 30% reduction

        final_threshold = max(min_thresh, base_threshold * accuracy_factor)

        return final_threshold

    def should_start_detection(self, epoch: int, model_accuracy: float) -> bool:
        """Determine if we should start circuit detection based on training state"""
        return (epoch >= self.warmup_epochs and
                model_accuracy >= self.min_accuracy_threshold)

    def get_consistency_requirement(self, circuit_type: str, epoch: int) -> int:
        """Get minimum consistency requirement for circuit type at epoch"""
        base_requirement = self.consistency_min_examples

        # Early training requires more consistency
        if epoch < 100:
            return base_requirement + 2
        elif epoch < 300:
            return base_requirement + 1
        else:
            return base_requirement

    def is_threshold_met(self, value: float, circuit_type: str, epoch: int,
                         total_epochs: int = 1000, model_accuracy: float = 0.0) -> bool:
        """Check if a value meets the adaptive threshold for circuit type"""
        threshold = self.get_threshold(circuit_type, epoch, total_epochs, model_accuracy)
        return value >= threshold