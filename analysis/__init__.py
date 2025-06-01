# analysis/__init__.py
"""
Transformer Circuit Analysis Framework

Main modules:
- core: Core infrastructure (schemas, registries, thresholds)
- analyzers: Circuit detection and analysis methods
- helpers: Domain-specific helper functions
- validation: Circuit validation and testing frameworks
- visualization: Plotting and visualization tools
- trainers: Enhanced training loops with circuit analysis
"""

# Core infrastructure
from .core import (
    Circuit, CircuitType, Element, ElementType, Connection, ConnectionType,
    CircuitRegistry, EnhancedCircuitRegistry,
    CircuitThresholds, ComputationalBudget,
    save_circuits, load_circuits
)

# Key analyzers (most commonly used)
from .analyzers.adaptive_token_operations import AdaptiveTokenOperationDetector
from .analyzers.integrated_token_discovery import IntegratedTokenCircuitDiscovery

# Helper functions (organized by domain)
from .helpers import (
    ContentAwareCircuitAnalyzer,
    analyze_tokens_adaptively,
    calculate_circuit_interaction,
    track_circuit_stability_evolution
)

# Validation framework
try:
    from .validation import CircuitManipulationValidator
except ImportError:
    # Validation might not be available in all setups
    pass

__version__ = "0.2.0"  # Week 2 completion

__all__ = [
    # Core classes
    'Circuit', 'CircuitType', 'Element', 'ElementType', 'Connection', 'ConnectionType',
    'CircuitRegistry', 'EnhancedCircuitRegistry',
    'CircuitThresholds', 'ComputationalBudget',
    'save_circuits', 'load_circuits',

    # Main analyzers
    'AdaptiveTokenOperationDetector',
    'IntegratedTokenCircuitDiscovery',

    # Helper functions
    'ContentAwareCircuitAnalyzer',
    'analyze_tokens_adaptively',
    'calculate_circuit_interaction',
    # 'track_circuit_stability_evolution',        # fixme warning nowhere to find

    # Validation
    'CircuitManipulationValidator',
]