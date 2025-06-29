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

# info canonicalcircuit registry
from .canonical_circuit_system import CanonicalCircuit, CanonicalCircuitRegistry, CanonicalRegistryAdapter

# info Messages logging
from .logger import DataLogger

# info stability of circuits
from .circuit_stability import EnhancedRegistryLifecycleManager, CircuitStabilityAnalyzer

__all__ = [
    # Schema classes
    'Circuit', 'CircuitType', 'Element', 'ElementType', 'Connection', 'ConnectionType',
    'EmergencePhase', 'CircuitStability', 'RelationshipType', 'CircuitMetadata',
    'save_circuits', 'load_circuits',

    # Registry classes
    'CircuitRegistry', 'EnhancedCircuitRegistry',

    # Infrastructure classes
    'CircuitThresholds', 'ComputationalBudget',

    # Canonical classes
    'CanonicalCircuit', 'CanonicalCircuitRegistry', 'CanonicalRegistryAdapter',

    # Logging
    'CircuitLogger',
    
    # info messages logging
    'DataLogger',

    # info circuit stability
    'EnhancedRegistryLifecycleManager', 'CircuitStabilityAnalyzer',
]
