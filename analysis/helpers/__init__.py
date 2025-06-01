# analysis/helpers/__init__.py
"""
Helper functions for circuit analysis organized by domain
"""

# Import from token_helpers
from .token_helpers import (
    analyze_tokens_adaptively,
    convert_mechanisms_to_circuits,
    create_copy_circuit_from_mechanism,
    create_induction_circuit_from_mechanism,
    analyze_content_vs_positional_patterns,
)

# Import from circuit_evolution
from .circuit_evolution import (
    calculate_circuit_interaction,
    create_functional_circuit,
    analyze_circuit_relationships_enhanced,
    has_prerequisite_relationship,
    has_competitive_relationship,
    calculate_circuit_emergence_rate,
    prune_unstable_circuits,
    track_circuit_stability_evolution
)

# Import from content_analysis
from .content_analysis import (
    ContentAwareCircuitAnalyzer
)

__all__ = [
    # Token helpers
    'analyze_tokens_adaptively',
    'convert_mechanisms_to_circuits',
    'create_copy_circuit_from_mechanism',
    'create_induction_circuit_from_mechanism',
    'analyze_content_vs_positional_patterns',
    'track_circuit_stability_evolution',

    # Circuit evolution helpers
    'calculate_circuit_interaction',
    'create_functional_circuit',
    'analyze_circuit_relationships_enhanced',
    'has_prerequisite_relationship',
    'has_competitive_relationship',
    'calculate_circuit_emergence_rate',

    # Content analysis
    'ContentAwareCircuitAnalyzer'
]