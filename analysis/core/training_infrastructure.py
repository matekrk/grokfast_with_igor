# analysis/core/training_infrastructure.py

# You can also import these components individually:
# from analysis.core.circuit_thresholds import CircuitThresholds
# from analysis.core.computational_budget import ComputationalBudget
# analysis/core/training_infrastructure.py
"""
Training infrastructure for enhanced circuit analysis
Consolidates infrastructure classes for easy import
"""

from .circuit_thresholds import CircuitThresholds
from .computational_budget import ComputationalBudget
from .circuit_registry import EnhancedCircuitRegistry

__all__ = ['CircuitThresholds', 'ComputationalBudget', 'EnhancedCircuitRegistry']