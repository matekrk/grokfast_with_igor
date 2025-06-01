# Enhanced core exports
# analysis/core/__init__.py

# Existing imports (keep as-is)
from .circuit_schema import (
    Circuit, CircuitType, Element, ElementType, Connection, ConnectionType,
    EmergencePhase, CircuitStability, RelationshipType, CircuitMetadata,  # Week 1 additions
    save_circuits, load_circuits
)

from .circuit_registry import CircuitRegistry  # Original - keep for compatibility

# Week 2 additions - carefully ordered to avoid circular imports
from .circuit_thresholds import CircuitThresholds
from .computational_budget import ComputationalBudget

# Enhanced registry - import after dependencies are established
from .circuit_registry import EnhancedCircuitRegistry

# Circuit logging
from .circuit_logger import CircuitLogger

__all__ = [
    # Schema classes
    'Circuit', 'CircuitType', 'Element', 'ElementType', 'Connection', 'ConnectionType',
    'EmergencePhase', 'CircuitStability', 'RelationshipType', 'CircuitMetadata',
    'save_circuits', 'load_circuits',

    # Registry classes
    'CircuitRegistry', 'EnhancedCircuitRegistry',

    # Infrastructure classes
    'CircuitThresholds', 'ComputationalBudget',

    # Logging
    'CircuitLogger'
]
