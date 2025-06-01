# analysis/helpers/validation_helpers.py
import numpy as np
from typing import Dict, Any  #, Optional, List
# from pathlib import Path
# import time


def run_comprehensive_validation(registry, validator, epoch: int, accuracy: float) -> Dict[str, Any]:
    """Run validation across all circuit types"""
    validation_results = {}

    # Get circuits of each type
    all_circuits = list(registry.circuits.values())

    # Sample circuits for validation (budget management)
    max_circuits_to_validate = 10
    if len(all_circuits) > max_circuits_to_validate:
        # Sample by importance/stability
        circuits_to_validate = sorted(
            all_circuits,
            key=lambda c: registry.circuit_metadata.get(c.id,
                                                        type('obj', (object,), {'stability_score': 0})).stability_score,
            reverse=True
        )[:max_circuits_to_validate]
    else:
        circuits_to_validate = all_circuits

    # Validate each circuit
    for circuit in circuits_to_validate:
        try:
            circuit_validation = validator.validate_circuit_importance(circuit)
            validation_results[circuit.id] = circuit_validation

            # Update registry with validation results if method exists
            if hasattr(registry, 'validate_circuit_with_manipulation'):
                registry.validate_circuit_with_manipulation(circuit.id, validator)
        except Exception as e:
            print(f"⚠️ Validation failed for circuit {circuit.id}: {e}")
            validation_results[circuit.id] = {"error": str(e)}

    return validation_results


def save_enhanced_checkpoint(checkpointManager, epoch: int, train_state, eval_state,
                             split_indices, train_stats, eval_stats, registry, weight_tracker):
    """Enhanced checkpoint saving with circuit analysis data"""

    # Basic checkpoint data
    extra_data = {
        'circuit_count': len(registry.circuits),
        'enhanced_registry_active': True,
    }

    # Add weight tracker data if available
    if hasattr(weight_tracker, 'detected_jumps'):
        extra_data['weight_space_jumps'] = [j.get('epoch', 0) for j in weight_tracker.detected_jumps]

    # Enhanced circuit metadata summary
    if hasattr(registry, 'circuit_metadata'):
        circuit_metadata_summary = {}
        for circuit_id, metadata in registry.circuit_metadata.items():
            circuit_metadata_summary[circuit_id] = {
                'stability': metadata.stability.value,
                'emergence_phase': metadata.emergence_phase.value,
                'detection_confidence': metadata.detection_confidence,
                'behavioral_impact': metadata.behavioral_impact
            }

        extra_data['circuit_metadata'] = circuit_metadata_summary

        # Emergence timeline
        if hasattr(registry, 'get_emergence_timeline'):
            extra_data['emergence_timeline'] = dict(registry.get_emergence_timeline())

        # Relationship count
        if hasattr(registry, 'relationship_graph'):
            extra_data['circuit_relationships'] = len(registry.relationship_graph)

        # Validation summary
        validated_circuits = len([m for m in registry.circuit_metadata.values()
                                  if m.manipulation_effects])
        extra_data['validated_circuits'] = validated_circuits

    # Top circuits by attribution
    all_circuits = list(registry.circuits.values())
    if all_circuits:
        top_circuits = sorted(all_circuits, key=lambda x: x.attribution, reverse=True)[:5]
        extra_data['top_circuits_by_attribution'] = [
            {'id': c.id, 'type': c.type.value, 'attribution': round(c.attribution, 4)}
            for c in top_circuits
        ]

        # Most stable circuits
        if hasattr(registry, 'circuit_metadata'):
            stable_circuits = sorted(
                all_circuits,
                key=lambda c: registry.circuit_metadata.get(c.id, type('obj', (object,),
                                                                       {'stability_score': 0})).stability_score,
                reverse=True
            )[:5]

            extra_data['most_stable_circuits'] = [
                {'id': c.id, 'type': c.type.value,
                 'stability_score': registry.circuit_metadata.get(c.id, type('obj', (object,),
                                                                             {'stability_score': 0})).stability_score}
                for c in stable_circuits
            ]

    # Save checkpoint with enhanced data
    checkpointManager.save_checkpoint(
        epoch=epoch,
        train_dataloader_state=train_state,
        eval_dataloader_state=eval_state,
        dataset_split_indices=split_indices,
        train_loss=train_stats['loss'] if train_stats else 1e6,
        train_accuracy=train_stats['accuracy'] if train_stats else 0.0,
        val_loss=eval_stats['loss'] if eval_stats else 1e6,
        val_accuracy=eval_stats['accuracy'] if eval_stats else 0.0,
        extra_data=extra_data,
        force_save=False
    )


def log_circuit_type_breakdown(snapshot: Dict, epoch: int) -> None:
    """Enhanced logging with detailed circuit type breakdown"""
    breakdown = snapshot.get('circuits_by_type', {})

    # Format circuit type summary
    type_summary = []
    for circuit_type, count in breakdown.items():
        if count > 0:
            type_summary.append(f"{circuit_type}: {count}")

    if type_summary:
        print(f"\t\t{' | '.join(type_summary)}")

    # Log additional metrics if available
    if 'circuits_by_phase' in snapshot:
        phase_breakdown = snapshot['circuits_by_phase']
        phase_summary = []
        for phase, count in phase_breakdown.items():
            if count > 0:
                phase_summary.append(f"{phase}: {count}")

        if phase_summary:
            print(f"\t\tPhases: {' | '.join(phase_summary)}")

    if 'circuits_by_stability' in snapshot:
        stability_breakdown = snapshot['circuits_by_stability']
        stability_summary = []
        for stability, count in stability_breakdown.items():
            if count > 0:
                stability_summary.append(f"{stability}: {count}")

        if stability_summary:
            print(f"\t\tStability: {' | '.join(stability_summary)}")


def analyze_final_circuit_dynamics(evolution_tracker, registry) -> Dict[str, Any]:
    """Comprehensive final analysis of circuit emergence and relationships"""

    final_analysis = {
        'emergence_order': {},
        'circuit_relationships': {},
        'stability_analysis': {},
        'functional_composition_analysis': {},
        'cross_level_relationships': {}
    }

    # Emergence order analysis
    if hasattr(evolution_tracker, 'analyze_emergence_order'):
        try:
            final_analysis['emergence_order'] = evolution_tracker.analyze_emergence_order()
        except Exception as e:
            print(f"⚠️ Emergence order analysis failed: {e}")

    # Circuit relationships
    if hasattr(evolution_tracker, 'analyze_circuit_relationships'):
        try:
            final_analysis['circuit_relationships'] = evolution_tracker.analyze_circuit_relationships()
        except Exception as e:
            print(f"⚠️ Circuit relationships analysis failed: {e}")

    # Stability analysis across circuit types
    from analysis.core.circuit_schema import CircuitType

    for circuit_type in CircuitType:
        try:
            type_circuits = registry.get_circuits_by_type(circuit_type)
            if type_circuits and hasattr(registry, 'circuit_metadata'):
                stabilities = []
                for circuit in type_circuits:
                    metadata = registry.circuit_metadata.get(circuit.id)
                    if metadata:
                        stabilities.append(metadata.stability_score)

                if stabilities:
                    final_analysis['stability_analysis'][circuit_type.value] = {
                        'count': len(type_circuits),
                        'avg_stability': np.mean(stabilities),
                        'most_stable': max(stabilities)
                    }
        except Exception as e:
            print(f"⚠️ Stability analysis failed for {circuit_type}: {e}")

    # Functional composition analysis
    try:
        functional_circuits = registry.get_circuits_by_type(CircuitType.FUNCTIONAL)
        if functional_circuits:
            compositions = []
            for fc in functional_circuits:
                composition_type = fc.metadata.get('composition_type', 'unknown')
                compositions.append(composition_type)

            from collections import Counter
            final_analysis['functional_composition_analysis'] = dict(Counter(compositions))
    except Exception as e:
        print(f"⚠️ Functional composition analysis failed: {e}")

    return final_analysis