# analysis/helpers/missing_functions.py
"""
Missing helper functions that are referenced in the training script
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import time


def run_circuit_validation(registry, validator, epoch: int, current_accuracy: float) -> Dict[str, Any]:
    """
    Run circuit validation (wrapper for run_comprehensive_validation)
    This function was referenced but missing
    """
    # This is just a wrapper for the comprehensive validation
    from analysis.helpers.validation_helpers import run_comprehensive_validation
    return run_comprehensive_validation(registry, validator, epoch, current_accuracy)



def save_final_analysis(analysis_results: Dict, registry, save_dir: Path) -> None:
    """
    Save final analysis results
    This function was referenced but missing
    """
    final_analysis_path = save_dir / "final_analysis.json"

    try:
        # Create comprehensive final analysis
        final_data = {
            'analysis_results': analysis_results,
            'registry_summary': registry.get_registry_summary() if hasattr(registry, 'get_registry_summary') else {},
            'total_circuits': len(registry.circuits),
            'circuit_metadata_summary': {}
        }

        # Add circuit metadata summary if available
        if hasattr(registry, 'circuit_metadata'):
            metadata_summary = {}
            for circuit_id, metadata in registry.circuit_metadata.items():
                metadata_summary[circuit_id] = {
                    'stability': metadata.stability.value,
                    'emergence_phase': metadata.emergence_phase.value,
                    'detection_confidence': metadata.detection_confidence,
                    'behavioral_impact': metadata.behavioral_impact
                }
            final_data['circuit_metadata_summary'] = metadata_summary

        # Save using JSON
        import json
        from analysis.utils.utils import CircuitJSONEncoder

        with open(final_analysis_path, 'w') as f:
            json.dump(final_data, f, cls=CircuitJSONEncoder, indent=2)

        print(f"✅ Final analysis saved to {final_analysis_path}")

    except Exception as e:
        print(f"⚠️ Failed to save final analysis: {e}")
        # Try basic save without custom encoder
        try:
            # Clean data for basic JSON
            cleaned_data = _clean_for_json(final_data)
            with open(final_analysis_path, 'w') as f:
                json.dump(cleaned_data, f, indent=2)
            print(f"📝 Final analysis saved with basic JSON to {final_analysis_path}")
        except Exception as e2:
            print(f"❌ Failed to save final analysis even with basic JSON: {e2}")


def _clean_for_json(obj):
    """Clean object for JSON serialization"""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif hasattr(obj, 'value'):  # Enum
        return obj.value
    else:
        return str(obj)


def process_weight_jumps(weight_tracker, model, eval_loader, epoch: int) -> Dict[str, Any]:
    """
    Process weight space jumps if detected
    This function might be referenced in some contexts
    """
    jump_results = {
        'jumps_processed': 0,
        'jump_epochs': [],
        'jump_analysis': {}
    }

    try:
        if hasattr(weight_tracker, 'pending_jumps') and weight_tracker.pending_jumps:
            # Process jumps using existing utility
            from analysis.trainers.utils import process_jumps

            jump_results_detailed = process_jumps(
                model=model,
                weight_tracker=weight_tracker,
                eval_loader=eval_loader,
                criterion=None,  # We'll skip criterion-dependent analysis
                optimizer=None  # We'll skip optimizer-dependent analysis
            )

            jump_results['jumps_processed'] = len(weight_tracker.pending_jumps)
            jump_results['jump_analysis'] = jump_results_detailed

    except Exception as e:
        print(f"⚠️ Weight jump processing failed: {e}")

    return jump_results


# Additional helper function to ensure all imports work
def ensure_compatibility():
    """
    Ensure all required components are available for the training script
    """
    missing_components = []

    # Check for required modules
    try:
        from analysis.core.circuit_registry import EnhancedCircuitRegistry
    except ImportError:
        missing_components.append("EnhancedCircuitRegistry")

    try:
        from analysis.core.circuit_thresholds import CircuitThresholds
    except ImportError:
        missing_components.append("CircuitThresholds")

    try:
        from analysis.core.computational_budget import ComputationalBudget
    except ImportError:
        missing_components.append("ComputationalBudget")

    try:
        from analysis.analyzers.adaptive_token_operations import AdaptiveTokenOperationDetector
    except ImportError:
        missing_components.append("AdaptiveTokenOperationDetector")

    if missing_components:
        print(f"⚠️ Missing components: {missing_components}")
        return False

    print("✅ All required components are available")
    return True