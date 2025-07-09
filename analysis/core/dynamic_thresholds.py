# analysis/core/dynamic_thresholds.py
"""
Dynamic threshold management for adaptive circuit detectors
"""

from dataclasses import dataclass, replace
from typing import Dict, Any, Optional
import numpy as np
from analysis.core.circuit_thresholds import CircuitThresholds


@dataclass
class ThresholdSchedule:
    """Configuration for threshold scheduling during training"""

    # Training phase definitions
    early_training_epochs: int = 200
    mid_training_epochs: int = 800
    late_training_epochs: int = 1500

    # Phase-specific threshold configs
    early_config: Optional[Dict[str, Any]] = None
    mid_config: Optional[Dict[str, Any]] = None
    late_config: Optional[Dict[str, Any]] = None

    # Transition parameters
    smooth_transitions: bool = True
    transition_window: int = 50  # Epochs over which to blend thresholds


def create_threshold_configs():
    """Create predefined threshold configurations for different training phases"""

    configs = {
        'early_training': {
            'copy_attention_min': 0.4,
            'copy_attention_max': 0.9,
            'induction_attention_min': 0.5,
            'induction_attention_max': 0.85,
            'warmup_epochs': 50,
            'min_accuracy_threshold': 0.15,
            'consistency_min_examples': 3  # Require more consistency early
        },

        'mid_training': {
            'copy_attention_min': 0.3,
            'copy_attention_max': 0.8,
            'induction_attention_min': 0.4,
            'induction_attention_max': 0.75,
            'warmup_epochs': 20,
            'min_accuracy_threshold': 0.25,
            'consistency_min_examples': 2
        },

        'late_training': {
            'copy_attention_min': 0.2,  # Very lenient for mature circuits
            'copy_attention_max': 0.7,
            'induction_attention_min': 0.3,
            'induction_attention_max': 0.65,
            'warmup_epochs': 10,
            'min_accuracy_threshold': 0.4,
            'consistency_min_examples': 1  # Mature circuits need less validation
        },

        'grokking_sensitive': {
            'copy_attention_min': 0.15,  # Extra sensitive for grokking detection
            'copy_attention_max': 0.6,
            'induction_attention_min': 0.25,
            'induction_attention_max': 0.55,
            'warmup_epochs': 5,
            'min_accuracy_threshold': 0.5,
            'consistency_min_examples': 1
        }
    }

    return configs


class DynamicThresholdManager:
    """Manages dynamic threshold updates for detector classes"""

    def __init__(self, initial_thresholds: CircuitThresholds = None,
                 schedule: ThresholdSchedule = None):
        self.current_thresholds = initial_thresholds or CircuitThresholds()
        self.schedule = schedule or ThresholdSchedule()
        self.threshold_configs = create_threshold_configs()
        self.current_phase = 'early_training'
        self.transition_progress = 0.0

        # ✅ ADD: For backward compatibility with tests
        self.threshold = self.current_thresholds  # Alias for test compatibility

    def update_thresholds_for_epoch(self, epoch: int, model_accuracy: float = None) -> CircuitThresholds:
        """Update thresholds based on current training epoch and performance"""

        # Determine current training phase
        new_phase = self._determine_training_phase(epoch, model_accuracy)

        # Check if we need to transition to a new phase
        if new_phase != self.current_phase:
            self._transition_to_phase(new_phase, epoch)

        # Apply any smooth transitions
        if self.schedule.smooth_transitions:
            self._apply_smooth_transition(epoch)

        # ✅ UPDATE: Keep alias synchronized
        self.threshold = self.current_thresholds

        return self.current_thresholds

    # ✅ ADD: Method expected by test
    def get_threshold_for_epoch(self, epoch: int, circuit_type: str = "copy",
                                model_accuracy: float = None, total_epochs: int = 1000) -> float:
        """
        Get specific threshold value for an epoch and circuit type

        This method is expected by integration tests

        Args:
            epoch: Current training epoch
            circuit_type: Type of circuit ('copy', 'induction', etc.)
            model_accuracy: Current model accuracy (optional)
            total_epochs: Total training epochs

        Returns:
            float: Threshold value for the specified circuit type
        """
        # Update thresholds for this epoch first
        current_thresholds = self.update_thresholds_for_epoch(epoch, model_accuracy)

        # Get the specific threshold using the CircuitThresholds.get_threshold method
        return current_thresholds.get_threshold(circuit_type, epoch, total_epochs, model_accuracy or 0.0)

    # ✅ ADD: Property for easy access
    @property
    def threshold(self):
        """Get current threshold object (for test compatibility)"""
        return self.current_thresholds

    @threshold.setter
    def threshold(self, value):
        """Set current threshold object (for test compatibility)"""
        self.current_thresholds = value

    # ✅ ADD: Convenience methods for common threshold access
    def get_copy_threshold(self, epoch: int, model_accuracy: float = None, total_epochs: int = 1000) -> float:
        """Get copy circuit threshold for epoch"""
        return self.get_threshold_for_epoch(epoch, "copy", model_accuracy, total_epochs)

    def get_induction_threshold(self, epoch: int, model_accuracy: float = None, total_epochs: int = 1000) -> float:
        """Get induction circuit threshold for epoch"""
        return self.get_threshold_for_epoch(epoch, "induction", model_accuracy, total_epochs)

    def get_current_thresholds_dict(self) -> dict:
        """Get current thresholds as dictionary for easy inspection"""
        return {
            'copy_attention_min': self.current_thresholds.copy_attention_min,
            'copy_attention_max': self.current_thresholds.copy_attention_max,
            'induction_attention_min': self.current_thresholds.induction_attention_min,
            'induction_attention_max': self.current_thresholds.induction_attention_max,
            'min_accuracy_threshold': self.current_thresholds.min_accuracy_threshold,
            'current_phase': self.current_phase
        }

    # ... existing methods remain unchanged ...

    def _determine_training_phase(self, epoch: int, model_accuracy: float = None) -> str:
        """Determine which training phase we're in"""

        # Phase determination logic
        if epoch < self.schedule.early_training_epochs:
            return 'early_training'
        elif epoch < self.schedule.mid_training_epochs:
            return 'mid_training'
        elif epoch < self.schedule.late_training_epochs:
            return 'late_training'
        else:
            # Check for grokking if accuracy is high enough
            if model_accuracy and model_accuracy > 0.8:
                return 'grokking_sensitive'
            return 'late_training'

    def _transition_to_phase(self, new_phase: str, epoch: int):
        """Transition to a new training phase"""
        print(f"🔄 Threshold phase transition: {self.current_phase} → {new_phase} @ epoch {epoch}")

        if new_phase in self.threshold_configs:
            new_config = self.threshold_configs[new_phase]
            self.current_thresholds = CircuitThresholds(**new_config)
            # ✅ UPDATE: Keep alias synchronized
            self.threshold = self.current_thresholds

        self.current_phase = new_phase
        self.transition_progress = 0.0

    def _apply_smooth_transition(self, epoch: int):
        """Apply smooth transitions between threshold values"""
        # Implementation for smooth blending between threshold values
        # This is where you'd implement interpolation logic if needed
        pass

    def manually_set_phase(self, phase_name: str):
        """Manually set the threshold phase"""
        if phase_name in self.threshold_configs:
            config = self.threshold_configs[phase_name]
            self.current_thresholds = CircuitThresholds(**config)
            # ✅ UPDATE: Keep alias synchronized
            self.threshold = self.current_thresholds
            self.current_phase = phase_name
            print(f"✅ Manually set threshold phase to: {phase_name}")
        else:
            print(f"⚠️ Unknown phase: {phase_name}. Available: {list(self.threshold_configs.keys())}")

    def add_custom_phase(self, phase_name: str, config: Dict[str, Any]):
        """Add a custom threshold configuration"""
        self.threshold_configs[phase_name] = config
        print(f"✅ Added custom threshold phase: {phase_name}")


