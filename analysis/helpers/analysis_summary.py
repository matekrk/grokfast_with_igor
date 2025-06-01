# analysis/helpers/analysis_summary.py
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from pathlib import Path

from analysis.core.circuit_schema import CircuitType, EmergencePhase, CircuitStability


def create_final_analysis_summary(registry, analysis_results: Dict[int, Any],
                                  total_epochs: int) -> Dict[str, Any]:
    """
    Create comprehensive final analysis of circuit emergence and evolution

    Args:
        registry: Enhanced circuit registry with temporal tracking
        analysis_results: Dictionary mapping epochs to analysis results
        total_epochs: Total training epochs

    Returns:
        Comprehensive analysis summary with research insights
    """
    print(f"🔍 Creating final analysis summary ({len(analysis_results)} epochs analyzed)")

    # ================================================================
    # METADATA AND OVERVIEW
    # ================================================================

    registry_summary = registry.get_registry_summary()

    metadata = {
        "analysis_version": "week2_enhanced",
        "total_epochs": total_epochs,
        "epochs_analyzed": len(analysis_results),
        "analysis_coverage": len(analysis_results) / max(1, total_epochs),
        "total_circuits": registry_summary["total_circuits"],
        "total_relationships": registry_summary["total_relationships"],
        "detection_methods": registry_summary["detection_methods"]
    }

    # ================================================================
    # CIRCUIT EMERGENCE TIMELINE ANALYSIS
    # ================================================================

    emergence_timeline = registry.get_emergence_timeline()
    emergence_analysis = analyze_emergence_timeline(emergence_timeline, total_epochs)

    # ================================================================
    # CIRCUIT EVOLUTION PATTERNS
    # ================================================================

    evolution_patterns = analyze_evolution_patterns(registry, analysis_results)

    # ================================================================
    # CIRCUIT RELATIONSHIPS ANALYSIS
    # ================================================================

    relationship_analysis = analyze_circuit_relationships_summary(registry)

    # ================================================================
    # PERFORMANCE CORRELATION ANALYSIS
    # ================================================================

    performance_correlation = analyze_performance_correlation(analysis_results)

    # ================================================================
    # STABILITY PATTERN ANALYSIS
    # ================================================================

    stability_analysis = analyze_stability_patterns(registry)

    # ================================================================
    # RESEARCH INSIGHTS AND RECOMMENDATIONS
    # ================================================================

    research_insights = generate_research_insights(
        emergence_analysis, evolution_patterns, relationship_analysis,
        performance_correlation, stability_analysis
    )

    # ================================================================
    # ASSEMBLE FINAL SUMMARY
    # ================================================================

    final_summary = {
        "metadata": metadata,
        "circuit_emergence": emergence_analysis,
        "circuit_evolution": evolution_patterns,
        "circuit_relationships": relationship_analysis,
        "performance_correlation": performance_correlation,
        "stability_analysis": stability_analysis,
        "research_insights": research_insights,

        # Additional analysis sections
        "circuit_type_breakdown": analyze_circuit_types(registry),
        "temporal_patterns": analyze_temporal_patterns(analysis_results),
        "validation_summary": summarize_validation_results(analysis_results)
    }

    print(f"✅ Final analysis complete: {len(final_summary)} analysis sections")
    return final_summary


def analyze_emergence_timeline(emergence_timeline: Dict[int, List[str]],
                               total_epochs: int) -> Dict[str, Any]:
    """Analyze circuit emergence patterns over time"""

    if not emergence_timeline:
        return {"error": "No emergence data available"}

    epochs = sorted(emergence_timeline.keys())
    emergence_counts = [len(emergence_timeline[epoch]) for epoch in epochs]

    # Find emergence phases
    early_phase = [e for e in epochs if e < total_epochs * 0.2]
    middle_phase = [e for e in epochs if total_epochs * 0.2 <= e < total_epochs * 0.7]
    late_phase = [e for e in epochs if e >= total_epochs * 0.7]

    early_circuits = sum(len(emergence_timeline[e]) for e in early_phase)
    middle_circuits = sum(len(emergence_timeline[e]) for e in middle_phase)
    late_circuits = sum(len(emergence_timeline[e]) for e in late_phase)

    return {
        "total_emergence_events": sum(emergence_counts),
        "emergence_epochs": epochs,
        "emergence_distribution": {
            "early_phase": early_circuits,
            "middle_phase": middle_circuits,
            "late_phase": late_circuits
        },
        "peak_emergence_epoch": epochs[np.argmax(emergence_counts)] if emergence_counts else None,
        "peak_emergence_count": max(emergence_counts) if emergence_counts else 0,
        "emergence_timeline": emergence_timeline
    }


def analyze_evolution_patterns(registry, analysis_results: Dict[int, Any]) -> Dict[str, Any]:
    """Analyze how circuits evolve over training"""

    evolution_events = []
    circuit_lifespans = {}

    # Track circuit appearances across epochs
    for epoch, results in analysis_results.items():
        epoch_circuits = []

        # Collect circuits from different analysis types with null safety
        for result_type in ['adaptive_token_results', 'existing_token_results', 'component_results']:
            if result_type in results and results[result_type] is not None:
                # ✅ FIX: Add null check before calling .get()
                result_data = results[result_type]
                if isinstance(result_data, dict):
                    circuits = result_data.get('circuits', [])
                    if circuits:  # Only extend if circuits is not None/empty
                        epoch_circuits.extend([c.id for c in circuits if hasattr(c, 'id')])

        for circuit_id in epoch_circuits:
            if circuit_id not in circuit_lifespans:
                circuit_lifespans[circuit_id] = {'birth': epoch, 'appearances': []}
            circuit_lifespans[circuit_id]['appearances'].append(epoch)

    # Calculate lifespan statistics
    lifespans = []
    for circuit_id, data in circuit_lifespans.items():
        appearances = data['appearances']
        if len(appearances) > 1:
            lifespan = max(appearances) - min(appearances)
            lifespans.append(lifespan)

    return {
        "tracked_circuits": len(circuit_lifespans),
        "average_lifespan": np.mean(lifespans) if lifespans else 0,
        "max_lifespan": max(lifespans) if lifespans else 0,
        "persistent_circuits": len([l for l in lifespans if l > 100]),
        "evolution_timeline": circuit_lifespans
    }


def analyze_circuit_relationships_summary(registry) -> Dict[str, Any]:
    """Summarize circuit relationships"""

    relationship_counts = defaultdict(int)

    for circuit_id, relationships in registry.relationship_graph.items():
        for related_id, relationship_type in relationships.items():
            relationship_counts[relationship_type.value] += 1

    return {
        "total_relationships": sum(relationship_counts.values()),
        "relationship_types": dict(relationship_counts),
        "avg_relationships_per_circuit": sum(relationship_counts.values()) / max(1, len(registry.circuits)),
        "highly_connected_circuits": _find_highly_connected_circuits(registry)
    }


def _find_highly_connected_circuits(registry) -> List[Dict[str, Any]]:
    """Find circuits with many relationships"""
    circuit_connections = {}

    for circuit_id, relationships in registry.relationship_graph.items():
        circuit_connections[circuit_id] = len(relationships)

    # Sort by connection count
    sorted_circuits = sorted(circuit_connections.items(), key=lambda x: x[1], reverse=True)

    return [
        {"circuit_id": cid, "connection_count": count}
        for cid, count in sorted_circuits[:10]  # Top 10
        if count > 0
    ]


def analyze_performance_correlation(analysis_results: Dict[int, Any]) -> Dict[str, Any]:
    """Analyze correlation between circuit emergence and performance"""

    # ✅ FIX: Handle empty analysis_results
    if not analysis_results:
        return {
            "epochs_analyzed": 0,
            "total_circuits_discovered": 0,
            "peak_discovery_epoch": None,
            "circuit_discovery_trend": "no_data"
        }

    epochs_with_performance = []
    circuit_counts = []

    for epoch, results in analysis_results.items():
        # ✅ FIX: Add null safety for results
        if results is None:
            continue

        # Count total circuits discovered in this epoch
        total_circuits = 0

        for result_type in ['adaptive_token_results', 'existing_token_results', 'component_results']:
            # ✅ FIX: Check both key existence AND null value AND dict type
            if (result_type in results and
                    results[result_type] is not None and
                    isinstance(results[result_type], dict)):
                circuits = results[result_type].get('circuits', [])
                if circuits and isinstance(circuits, list):
                    total_circuits += len(circuits)

        if total_circuits > 0:
            epochs_with_performance.append(epoch)
            circuit_counts.append(total_circuits)

    # ✅ FIX: Handle case where no circuits were found
    if not circuit_counts:
        return {
            "epochs_analyzed": len(analysis_results),
            "total_circuits_discovered": 0,
            "peak_discovery_epoch": None,
            "circuit_discovery_trend": "no_circuits_detected"
        }

    return {
        "epochs_analyzed": len(epochs_with_performance),
        "total_circuits_discovered": sum(circuit_counts),
        "peak_discovery_epoch": epochs_with_performance[np.argmax(circuit_counts)] if circuit_counts else None,
        "circuit_discovery_trend": "increasing" if len(circuit_counts) > 1 and circuit_counts[-1] > circuit_counts[
            0] else "stable"
    }


def analyze_stability_patterns(registry) -> Dict[str, Any]:
    """Analyze circuit stability patterns"""

    stability_counts = defaultdict(int)
    phase_counts = defaultdict(int)

    for circuit_id, metadata in registry.circuit_metadata.items():
        stability_counts[metadata.stability.value] += 1
        phase_counts[metadata.emergence_phase.value] += 1

    return {
        "stability_distribution": dict(stability_counts),
        "emergence_phase_distribution": dict(phase_counts),
        "stable_circuit_ratio": stability_counts.get('stable', 0) / max(1, sum(stability_counts.values())),
        "persistent_circuit_ratio": stability_counts.get('persistent', 0) / max(1, sum(stability_counts.values()))
    }


def analyze_circuit_types(registry) -> Dict[str, Any]:
    """Analyze distribution of circuit types"""

    type_counts = defaultdict(int)

    for circuit in registry.circuits.values():
        type_counts[circuit.type.value] += 1

    return {
        "type_distribution": dict(type_counts),
        "most_common_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
        "type_diversity": len(type_counts)
    }


def analyze_temporal_patterns(analysis_results: Dict[int, Any]) -> Dict[str, Any]:
    """Analyze temporal patterns in circuit analysis"""

    # ✅ FIX: Handle empty analysis_results
    if not analysis_results:
        return {
            "error": "No analysis results available",
            "analysis_frequency": 0,
            "epoch_range": None,
            "average_analysis_interval": 0,
            "consistent_intervals": False
        }

    analysis_frequency = len(analysis_results)
    epochs_analyzed = sorted(analysis_results.keys())

    if len(epochs_analyzed) < 2:
        return {
            "analysis_frequency": analysis_frequency,
            "epoch_range": (min(epochs_analyzed), max(epochs_analyzed)) if epochs_analyzed else None,
            "average_analysis_interval": 0,
            "consistent_intervals": False,
            "note": "Insufficient temporal data for interval analysis"
        }

    intervals = np.diff(epochs_analyzed)

    return {
        "analysis_frequency": analysis_frequency,
        "epoch_range": (min(epochs_analyzed), max(epochs_analyzed)),
        "average_analysis_interval": np.mean(intervals),
        "consistent_intervals": np.std(intervals) < 2.0  # Whether intervals are consistent
    }


def summarize_validation_results(analysis_results: Dict[int, Any]) -> Dict[str, Any]:
    """Summarize validation results if available"""

    # ✅ FIX: Handle empty analysis_results
    if not analysis_results:
        return {"validation_performed": False, "reason": "no_analysis_results"}

    validation_epochs = []
    validation_accuracies = []

    for epoch, results in analysis_results.items():
        # ✅ FIX: Add null safety
        if results is not None and 'validation_results' in results:
            val_results = results['validation_results']
            if val_results is not None and isinstance(val_results, dict):
                validation_epochs.append(epoch)
                validation_accuracies.append(val_results.get('average_accuracy', 0.0))

    if not validation_epochs:
        return {"validation_performed": False}

    return {
        "validation_performed": True,
        "validation_epochs": validation_epochs,
        "average_validation_accuracy": np.mean(validation_accuracies),
        "validation_trend": "improving" if len(validation_accuracies) > 1 and validation_accuracies[-1] >
                                           validation_accuracies[0] else "stable"
    }


def generate_research_insights(emergence_analysis, evolution_patterns,
                               relationship_analysis, performance_correlation,
                               stability_analysis) -> Dict[str, Any]:
    """Generate research insights and recommendations"""

    insights = []
    recommendations = []

    # ✅ FIX: Add null safety for all analysis inputs
    if emergence_analysis is None:
        emergence_analysis = {}
    if evolution_patterns is None:
        evolution_patterns = {}
    if relationship_analysis is None:
        relationship_analysis = {}
    if performance_correlation is None:
        performance_correlation = {}
    if stability_analysis is None:
        stability_analysis = {}

    # Emergence insights
    if emergence_analysis.get("emergence_distribution"):
        dist = emergence_analysis["emergence_distribution"]
        if dist["early_phase"] > dist["middle_phase"] + dist["late_phase"]:
            insights.append("Circuit emergence is front-loaded - most circuits appear early in training")
            recommendations.append("Focus circuit analysis on early training phases")
        elif dist["late_phase"] > dist["early_phase"] + dist["middle_phase"]:
            insights.append("Circuit emergence is back-loaded - most circuits appear late in training")
            recommendations.append("Extend analysis to later training phases")

    # Stability insights
    stable_ratio = stability_analysis.get("stable_circuit_ratio", 0)
    if stable_ratio > 0.7:
        insights.append("High circuit stability detected - circuits are persistent and reliable")
        recommendations.append("These stable circuits are good candidates for functional validation")
    elif stable_ratio < 0.3:
        insights.append("Low circuit stability - many transient circuits detected")
        recommendations.append("Consider increasing detection thresholds or validation criteria")

    # Relationship insights
    total_relationships = relationship_analysis.get("total_relationships", 0)
    total_circuits = len(relationship_analysis.get("highly_connected_circuits", []))
    if total_relationships > total_circuits * 2:
        insights.append("High circuit interconnectivity - complex relationship network detected")
        recommendations.append("Investigate hierarchical circuit organization and dependency chains")

    # ✅ NEW: Handle case where no insights were generated
    if not insights:
        insights.append("Limited circuit activity detected - may need longer training or adjusted thresholds")
        recommendations.append("Consider extending training duration or lowering detection thresholds")

    return {
        "key_insights": insights,
        "research_recommendations": recommendations,
        "analysis_quality": "high" if len(insights) > 2 else "moderate",
        "suggested_next_steps": [
            "Extend analysis to knowledge graph environments",
            "Implement activation patching validation",
            "Study circuit composition mechanisms",
            "Investigate cross-layer circuit interactions"
        ]
    }