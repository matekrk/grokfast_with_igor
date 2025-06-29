# analysis/helpers/circuit_analysis.py
# ADD new helper functions for circuit analysis

from typing import Dict, List, Any

from analysis.core import Circuit
# from analysis.helpers.circuit_evolution import CircuitEvolutionTracker
from analysis.core.unified_circuit_evolution_tracker import UnifiedCircuitEvolutionTracker




def analyze_circuit_emergence(evolution_tracker: UnifiedCircuitEvolutionTracker, current_epoch: int) -> Dict[str, Any]:
    """Analyze circuit emergence patterns"""

    # Get evolution summary
    evolution_summary = evolution_tracker.get_evolution_summary(current_epoch=current_epoch)

    # Analyze emergence timing
    birth_events = [e for e in evolution_tracker.evolution_events if e["type"] == "birth"]
    death_events = [e for e in evolution_tracker.evolution_events if e["type"] == "death"]

    # Group births by epoch ranges
    early_births = len([e for e in birth_events if e["epoch"] < 100])
    middle_births = len([e for e in birth_events if 100 <= e["epoch"] < 500])
    late_births = len([e for e in birth_events if e["epoch"] >= 500])

    # Calculate emergence rate (births per epoch in recent period)
    recent_epochs = 20
    recent_births = len([e for e in birth_events if current_epoch - e["epoch"] <= recent_epochs])
    emergence_rate = recent_births / recent_epochs if recent_epochs > 0 else 0

    return {
        "evolution_summary": evolution_summary,
        "emergence_timing": {
            "early_births": early_births,
            "middle_births": middle_births,
            "late_births": late_births
        },
        "emergence_rate": emergence_rate,
        "total_events": len(evolution_tracker.evolution_events),
        "birth_death_ratio": len(birth_events) / max(1, len(death_events))
    }


def analyze_circuit_relationships(registry, epoch: int) -> Dict[str, Any]:
    """Analyze relationships between circuits"""

    if not hasattr(registry, 'relationship_graph'):
        return {"relationships": {}, "relationship_count": 0}

    relationship_counts = {}
    for circuit_id, relationships in registry.relationship_graph.items():
        for related_id, relationship_type in relationships.items():
            rel_type = relationship_type.value if hasattr(relationship_type, 'value') else str(relationship_type)
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1

    return {
        "relationships": relationship_counts,
        "relationship_count": sum(relationship_counts.values()),
        "total_circuits_with_relationships": len(registry.relationship_graph)
    }