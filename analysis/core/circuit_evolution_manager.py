# analysis/core/circuit_evolution_manager.py
"""
Migration Manager and Factory for Circuit Evolution Tracking
Helps transition from multiple CircuitEvolutionTracker implementations to unified version
"""

from typing import Dict, Any, Optional
from pathlib import Path
import warnings

from analysis.core.unified_circuit_evolution_tracker import UnifiedCircuitEvolutionTracker


class CircuitEvolutionManager:
    """
    Factory and migration manager for circuit evolution tracking
    Provides backward compatibility and unified interface
    """

    def __init__(self, registry, save_dir=None, logger=None,
                 legacy_mode=False, migration_warnings=True):
        """
        Initialize the evolution manager

        Args:
            registry: Circuit registry
            save_dir: Directory for saving results
            logger: Logger instance
            legacy_mode: If True, provides legacy interface compatibility
            migration_warnings: If True, shows migration warnings
        """
        self.registry = registry
        self.save_dir = save_dir
        self.logger = logger
        self.legacy_mode = legacy_mode
        self.migration_warnings = migration_warnings

        # Create unified tracker
        self.tracker = UnifiedCircuitEvolutionTracker(
            registry=registry,
            save_dir=save_dir,
            logger=logger
        )

        # Legacy interface mappings
        self._setup_legacy_mappings()

    def _setup_legacy_mappings(self):
        """Setup mappings for legacy method names"""
        # Map old method names to new unified methods
        self.legacy_method_map = {
            # From helpers.circuit_evolution
            'track_epoch_circuits': 'track_epoch',
            'get_stable_circuits': 'get_stable_circuits',
            'get_evolution_summary': 'get_evolution_summary',

            # From analyzers.circuit_evolution_tracker
            'update_circuit_evolution': 'track_epoch',
            'analyze_emergence_order': 'analyze_emergence_order',
            'analyze_circuit_relationships': 'analyze_circuit_relationships',

            # From core.token_circuit_evolution
            '_calculate_circuit_strength': '_get_current_strength',
        }

    def create_tracker(self, tracker_type: str = 'unified') -> UnifiedCircuitEvolutionTracker:
        """
        Factory method to create appropriate tracker

        Args:
            tracker_type: Type of tracker ('unified', 'legacy_helpers', 'legacy_analyzers')

        Returns:
            UnifiedCircuitEvolutionTracker instance with appropriate interface
        """
        if tracker_type == 'unified':
            return self.tracker

        elif tracker_type in ['legacy_helpers', 'legacy_analyzers', 'legacy_core']:
            if self.migration_warnings:
                warnings.warn(
                    f"Using legacy interface '{tracker_type}'. "
                    f"Consider migrating to unified interface.",
                    DeprecationWarning,
                    stacklevel=2
                )
            return LegacyCompatibilityWrapper(self.tracker, tracker_type)

        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")

    def migrate_existing_data(self, old_tracker_data: Dict[str, Any]) -> bool:
        """
        Migrate data from old tracker implementations

        Args:
            old_tracker_data: Data from previous tracker implementations

        Returns:
            bool: Success status
        """
        try:
            # Migrate evolution data
            if 'evolution_data' in old_tracker_data:
                self.tracker.evolution_data.update(old_tracker_data['evolution_data'])

            # Migrate emergence epochs
            if 'emergence_epochs' in old_tracker_data:
                self.tracker.emergence_epochs.update(old_tracker_data['emergence_epochs'])

            # Migrate circuit lifetimes (from helpers version)
            if 'circuit_lifetimes' in old_tracker_data:
                self.tracker.circuit_lifetimes.update(old_tracker_data['circuit_lifetimes'])

            # Migrate stability scores
            if 'stability_scores' in old_tracker_data:
                self.tracker.stability_scores.update(old_tracker_data['stability_scores'])

            # Migrate relationships
            if 'circuit_relationships' in old_tracker_data:
                self.tracker.circuit_relationships.update(old_tracker_data['circuit_relationships'])

            # Migrate evolution events
            if 'evolution_events' in old_tracker_data:
                self.tracker.evolution_events.extend(old_tracker_data['evolution_events'])

            self.logger.info("Successfully migrated existing tracker data") if self.logger else None
            return True

        except Exception as e:
            error_msg = f"Failed to migrate tracker data: {e}"
            self.logger.error(error_msg) if self.logger else print(error_msg)
            return False


class LegacyCompatibilityWrapper:
    """Wrapper to provide backward compatibility with old interfaces"""

    def __init__(self, unified_tracker: UnifiedCircuitEvolutionTracker, wrapper_type: str):
        self.tracker = unified_tracker
        self.wrapper_type = wrapper_type
        self._setup_interface()

    def _setup_interface(self):
        """Setup interface based on wrapper type"""
        if self.wrapper_type == 'legacy_helpers':
            # Interface from helpers.circuit_evolution.CircuitEvolutionTracker
            self.track_epoch_circuits = self._track_epoch_circuits_helpers
            self.get_stable_circuits = self._get_stable_circuits_helpers
            self.get_evolution_summary = self.tracker.get_evolution_summary

        elif self.wrapper_type == 'legacy_analyzers':
            # Interface from analyzers.circuit_evolution_tracker.CircuitEvolutionTracker
            self.update_circuit_evolution = self._update_circuit_evolution_analyzers
            self.analyze_emergence_order = self.tracker.analyze_emergence_order
            self.analyze_circuit_relationships = self.tracker.analyze_circuit_relationships

        elif self.wrapper_type == 'legacy_core':
            # Interface from core.token_circuit_evolution.CircuitEvolutionTracker
            self.update_circuit_evolution = self._update_circuit_evolution_core
            self._calculate_circuit_strength = self._calculate_circuit_strength_core

    def _track_epoch_circuits_helpers(self, epoch: int, detected_circuits: list):
        """Legacy interface from helpers version"""
        return self.tracker.track_epoch(epoch, detected_circuits)

    def _get_stable_circuits_helpers(self, current_epoch: int, min_lifetime: int = 10,
                                     min_stability: float = 0.7):
        """Legacy interface from helpers version"""
        stable = self.tracker.get_stable_circuits(current_epoch, min_stability)
        # Filter by minimum lifetime
        return [c for c in stable if c['lifetime'] >= min_lifetime]

    def _update_circuit_evolution_analyzers(self, epoch, circuits, token_attribution):
        """Legacy interface from analyzers version"""
        # Convert Circuit objects to dict format
        circuit_data = []
        for circuit in circuits:
            circuit_data.append({
                'id': circuit.id,
                'attribution': circuit.attribution,
                'strength': circuit.attribution,
                'elements': len(circuit.elements) if hasattr(circuit, 'elements') else 0,
                'connections': len(circuit.connections) if hasattr(circuit, 'connections') else 0
            })

        result = self.tracker.track_epoch(epoch, circuit_data)

        # Return in expected format
        return {
            'active_circuits': result['active_circuits'],
            'emergence_events': [cid for cid in self.tracker.emergence_epochs
                                 if self.tracker.emergence_epochs[cid] == epoch],
            'evolution_data': self.tracker.evolution_data
        }

    def _update_circuit_evolution_core(self, epoch, circuits, token_attribution):
        """Legacy interface from core version"""
        return self._update_circuit_evolution_analyzers(epoch, circuits, token_attribution)

    def _calculate_circuit_strength_core(self, circuit, token_attribution):
        """Legacy interface from core version"""
        if hasattr(circuit, 'id'):
            return self.tracker._get_current_strength(circuit.id)
        return circuit.attribution if hasattr(circuit, 'attribution') else 0.5

    def __getattr__(self, name):
        """Delegate unknown attributes to the unified tracker"""
        return getattr(self.tracker, name)


# ============================================================================
# MIGRATION UTILITIES
# ============================================================================

def migrate_from_old_implementations(registry, save_dir=None, logger=None,
                                     old_tracker_files: Dict[str, str] = None) -> UnifiedCircuitEvolutionTracker:
    """
    Utility function to migrate from old implementations

    Args:
        registry: Circuit registry
        save_dir: Save directory
        logger: Logger
        old_tracker_files: Dict mapping tracker_type -> file_path for data migration

    Returns:
        Configured UnifiedCircuitEvolutionTracker
    """
    manager = CircuitEvolutionManager(registry, save_dir, logger)
    tracker = manager.create_tracker('unified')

    # If old data files provided, attempt migration
    if old_tracker_files:
        import pickle

        for tracker_type, file_path in old_tracker_files.items():
            try:
                if Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        old_data = pickle.load(f)
                    manager.migrate_existing_data(old_data)
                    if logger:
                        logger.info(f"Migrated data from {tracker_type}: {file_path}")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not migrate {tracker_type} data: {e}")

    return tracker


def update_imports_guide():
    """Print guide for updating imports"""
    guide = """
    # MIGRATION GUIDE: Updating CircuitEvolutionTracker imports

    # OLD IMPORTS (to be replaced):
    # from analysis.analyzers.circuit_evolution_tracker import CircuitEvolutionTracker
    # from analysis.helpers.circuit_evolution import CircuitEvolutionTracker  
    # from analysis.core.token_circuit_evolution import CircuitEvolutionTracker

    # NEW UNIFIED IMPORT:
    from analysis.core.unified_circuit_evolution_tracker import UnifiedCircuitEvolutionTracker

    # Or use the migration manager for backward compatibility:
    from analysis.core.circuit_evolution_manager import CircuitEvolutionManager

    # EXAMPLE MIGRATION:

    # OLD CODE:
    # tracker = CircuitEvolutionTracker(registry)
    # tracker.track_epoch_circuits(epoch, circuits)

    # NEW CODE:
    # manager = CircuitEvolutionManager(registry, save_dir, logger)
    # tracker = manager.create_tracker('unified')
    # tracker.track_epoch(epoch, circuit_data)

    # OR for backward compatibility:
    # tracker = manager.create_tracker('legacy_helpers')  # maintains old interface

    # KEY INTERFACE CHANGES:
    # - track_epoch_circuits() -> track_epoch()
    # - update_circuit_evolution() -> track_epoch() 
    # - More comprehensive stability and consistency tracking
    # - Unified relationship analysis
    # - Better integration with EnhancedCircuitRegistry
    """
    print(guide)


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def integrate_with_adaptive_detector(detector, registry, save_dir=None, logger=None):
    """
    Helper to integrate unified tracker with FixedAdaptiveTokenOperationDetector

    Args:
        detector: FixedAdaptiveTokenOperationDetector instance
        registry: Circuit registry
        save_dir: Save directory
        logger: Logger

    Returns:
        Configured tracker attached to detector
    """
    manager = CircuitEvolutionManager(registry, save_dir, logger)
    tracker = manager.create_tracker('unified')

    # Attach to detector
    detector.evolution_tracker = tracker

    # Add convenience method to detector
    def get_circuit_stability(circuit_id, epoch):
        return tracker.get_circuit_stability_for_detector(circuit_id, epoch)

    detector.get_circuit_stability = get_circuit_stability

    return tracker


def integrate_with_registry(registry, save_dir=None, logger=None):
    """
    Helper to integrate unified tracker with EnhancedCircuitRegistry

    Args:
        registry: EnhancedCircuitRegistry instance
        save_dir: Save directory
        logger: Logger

    Returns:
        Configured tracker attached to registry
    """
    manager = CircuitEvolutionManager(registry, save_dir, logger)
    tracker = manager.create_tracker('unified')

    # Attach to registry
    registry.evolution_tracker = tracker

    # Add convenience methods to registry
    def update_evolution_tracking(epoch):
        return tracker.update_from_registry(epoch)

    def get_stable_circuits(epoch, min_stability=0.6):
        return tracker.get_stable_circuits(epoch, min_stability)

    registry.update_evolution_tracking = update_evolution_tracking
    registry.get_stable_circuits = get_stable_circuits

    return tracker


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Example of how to use the unified tracker"""

    # Example 1: Direct usage
    from analysis.core.unified_circuit_evolution_tracker import UnifiedCircuitEvolutionTracker

    # tracker = UnifiedCircuitEvolutionTracker(registry, save_dir, logger)
    # result = tracker.track_epoch(epoch, detected_circuits, model_accuracy)
    # stable_circuits = tracker.get_stable_circuits(epoch)
    # summary = tracker.get_evolution_summary(epoch)

    # Example 2: Using migration manager
    from analysis.core.circuit_evolution_manager import CircuitEvolutionManager

    # manager = CircuitEvolutionManager(registry, save_dir, logger)
    # tracker = manager.create_tracker('unified')

    # Example 3: Legacy compatibility
    # tracker = manager.create_tracker('legacy_helpers')  # Old interface

    # Example 4: Integration with existing systems
    # integrate_with_adaptive_detector(detector, registry, save_dir, logger)
    # integrate_with_registry(registry, save_dir, logger)

    pass