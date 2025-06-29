# analysis/core/enhanced_circuit_quality_system.py
"""
Unified Circuit Quality and Lifecycle Management System

Centralizes all quality assessment around CircuitQualityAnalyzer and adds
proper lifecycle management and removal capabilities.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time

from analysis.visualization.circuit_quality_analyzer import CircuitQualityAnalyzer
from analysis.core.canonical_circuit_system import CanonicalCircuitRegistry


class CircuitLifecycleState(Enum):
    """Circuit lifecycle states"""
    NEW = "new"  # Just detected, in probation
    PROBATION = "probation"  # Being evaluated, can't be removed yet
    ESTABLISHED = "established"  # Proven stable and useful
    DECLINING = "declining"  # Quality dropping, candidate for removal
    MARKED_FOR_REMOVAL = "marked_for_removal"  # Will be removed next cycle
    PROTECTED = "protected"  # Diversity-protected, harder to remove


@dataclass
class CircuitLifecycleInfo:
    """Lifecycle information for each circuit"""
    circuit_id: str
    state: CircuitLifecycleState = CircuitLifecycleState.NEW
    first_detected: int = 0
    quality_history: List[float] = field(default_factory=list)
    assessment_count: int = 0
    removal_votes: int = 0
    protection_reason: Optional[str] = None
    last_assessment_epoch: int = 0

    def update_quality(self, quality_score: float, epoch: int):
        """Update quality history"""
        self.quality_history.append(quality_score)
        self.assessment_count += 1
        self.last_assessment_epoch = epoch

        # Keep only recent history
        if len(self.quality_history) > 20:
            self.quality_history = self.quality_history[-20:]


class EnhancedCircuitQualitySystem:
    """
    Unified system that combines quality assessment with lifecycle management
    """

    def __init__(self, model, canonical_registry: CanonicalCircuitRegistry,
                 save_dir: Optional[Path] = None, enable_real_testing: bool = True):
        self.model = model
        self.canonical_registry = canonical_registry
        self.save_dir = save_dir

        # Use CircuitQualityAnalyzer as the primary quality assessor
        self.quality_analyzer = CircuitQualityAnalyzer(model, canonical_registry, save_dir)

        # 🧪 INTEGRATE REAL FUNCTIONAL TESTING
        if enable_real_testing:
            self._integrate_real_testing()

        # Lifecycle management
        self.circuit_lifecycles: Dict[str, CircuitLifecycleInfo] = {}

        # Configuration
        self.config = {
            'probation_epochs': 100,  # Can't remove circuits younger than this
            'grace_period_epochs': 50,  # Extra time for borderline circuits
            'min_quality_threshold': 0.3,  # Below this = candidate for removal
            'removal_votes_required': 3,  # Need multiple bad assessments
            'max_circuits': 200,  # Maximum circuits to maintain
            'diversity_protection_ratio': 0.1,  # Protect 10% for diversity
            'assessment_interval': 10,  # Epochs between assessments
            'enable_functional_testing': enable_real_testing,  # Whether to use real testing
        }

        # Tracking
        self.removal_history: List[Dict] = []
        self.diversity_tracker = CircuitDiversityTracker()

    def _integrate_real_testing(self):
        """Integrate real functional testing with the quality analyzer"""
        try:
            from analysis.validation.circuit_testing import integrate_real_testing_with_quality_analyzer

            # This function replaces the mock methods with real ones
            self.quality_analyzer = integrate_real_testing_with_quality_analyzer(self.quality_analyzer)

            print("✅ Real circuit functional testing integrated")
            print("   → CircuitFunctionalTester activated for behavioral validation")
            print("   → Activation patching enabled for copy/induction testing")

        except ImportError as e:
            print(f"⚠️  Could not import real testing: {e}")
            print("   Using mock testing methods instead")
        except Exception as e:
            print(f"⚠️  Error integrating real testing: {e}")
            print("   Falling back to mock testing methods")

    def assess_and_manage_circuits(self, eval_loader, epoch: int) -> Dict[str, Any]:
        """
        Main method: assess all circuits and manage their lifecycles
        """
        # 1. Quality Assessment (using CircuitQualityAnalyzer)
        # print(f"🔍 Assessing circuit quality @ epoch {epoch}")

        quality_results = self.quality_analyzer.analyze_all_circuits(
            eval_loader, max_circuits=len(self.canonical_registry.canonical_circuits)
        )

        # 2. Update lifecycles based on quality assessment
        lifecycle_updates = self._update_circuit_lifecycles(quality_results, epoch)

        # 3. Make removal decisions
        removal_decisions = self._make_removal_decisions(epoch)

        # 4. Execute removals
        removed_circuits = self._execute_removals(removal_decisions, epoch)

        # 5. Protect diversity
        protected_circuits = self._apply_diversity_protection(epoch)

        # 6. Generate summary
        summary = self._generate_management_summary(
            quality_results, lifecycle_updates, removed_circuits, protected_circuits, epoch
        )

        return summary

    def _update_circuit_lifecycles(self, quality_results: Dict, epoch: int) -> Dict[str, Any]:
        """Update lifecycle states based on quality assessment"""
        updates = {'state_changes': [], 'new_circuits': [], 'quality_updates': []}

        for circuit_id, analysis in quality_results['circuit_analyses'].items():
            quality_score = analysis['quality_score']

            # Get or create lifecycle info
            if circuit_id not in self.circuit_lifecycles:
                first_seen = self.canonical_registry.canonical_circuits[circuit_id].first_seen
                self.circuit_lifecycles[circuit_id] = CircuitLifecycleInfo(
                    circuit_id=circuit_id,
                    first_detected=first_seen
                )
                updates['new_circuits'].append(circuit_id)

            lifecycle = self.circuit_lifecycles[circuit_id]
            old_state = lifecycle.state

            # Update quality history
            lifecycle.update_quality(quality_score, epoch)
            updates['quality_updates'].append({
                'circuit_id': circuit_id,
                'quality_score': quality_score,
                'avg_quality': np.mean(lifecycle.quality_history)
            })

            # Update lifecycle state
            new_state = self._determine_lifecycle_state(lifecycle, epoch, analysis)

            if new_state != old_state:
                lifecycle.state = new_state
                updates['state_changes'].append({
                    'circuit_id': circuit_id,
                    'old_state': old_state.value,
                    'new_state': new_state.value,
                    'reason': analysis.get('recommendation', 'unknown')
                })

        return updates

    def _determine_lifecycle_state(self, lifecycle: CircuitLifecycleInfo,
                                   epoch: int, analysis: Dict) -> CircuitLifecycleState:
        """Determine appropriate lifecycle state for a circuit"""

        circuit_age = epoch - lifecycle.first_detected
        avg_quality = np.mean(lifecycle.quality_history) if lifecycle.quality_history else 0.0
        recommendation = analysis.get('recommendation', '')

        # New circuits stay in probation for minimum period
        if circuit_age < self.config['probation_epochs']:
            return CircuitLifecycleState.PROBATION

        # Check for protection status
        if lifecycle.protection_reason:
            return CircuitLifecycleState.PROTECTED

        # State transitions based on quality and recommendations
        if 'REMOVE' in recommendation:
            lifecycle.removal_votes += 1
            if lifecycle.removal_votes >= self.config['removal_votes_required']:
                return CircuitLifecycleState.MARKED_FOR_REMOVAL
            else:
                return CircuitLifecycleState.DECLINING

        elif 'KEEP' in recommendation and avg_quality > 0.6:
            lifecycle.removal_votes = max(0, lifecycle.removal_votes - 1)  # Forgive past votes
            return CircuitLifecycleState.ESTABLISHED

        elif avg_quality < self.config['min_quality_threshold']:
            return CircuitLifecycleState.DECLINING

        else:
            return CircuitLifecycleState.ESTABLISHED

    def _make_removal_decisions(self, epoch: int) -> List[str]:
        """Decide which circuits to remove"""
        candidates_for_removal = []

        # Get circuits marked for removal
        for circuit_id, lifecycle in self.circuit_lifecycles.items():
            if lifecycle.state == CircuitLifecycleState.MARKED_FOR_REMOVAL:
                candidates_for_removal.append(circuit_id)

        # If we're over capacity, add more candidates
        current_count = len(self.canonical_registry.canonical_circuits)
        if current_count > self.config['max_circuits']:
            # Add declining circuits, starting with lowest quality
            declining_circuits = [
                (circuit_id, np.mean(lifecycle.quality_history))
                for circuit_id, lifecycle in self.circuit_lifecycles.items()
                if lifecycle.state == CircuitLifecycleState.DECLINING
            ]
            declining_circuits.sort(key=lambda x: x[1])  # Sort by quality

            needed_removals = current_count - self.config['max_circuits']
            for circuit_id, _ in declining_circuits[:needed_removals]:
                if circuit_id not in candidates_for_removal:
                    candidates_for_removal.append(circuit_id)

        return candidates_for_removal

    def _execute_removals(self, circuits_to_remove: List[str], epoch: int) -> List[Dict]:
        """Actually remove circuits from the registry"""
        removed_circuits = []

        for circuit_id in circuits_to_remove:
            if circuit_id in self.canonical_registry.canonical_circuits:
                # Get circuit info before removal
                circuit = self.canonical_registry.canonical_circuits[circuit_id]
                lifecycle = self.circuit_lifecycles[circuit_id]

                removal_info = {
                    'circuit_id': circuit_id,
                    'epoch': epoch,
                    'operation_type': circuit.computational_signature.operation_type,
                    'total_detections': circuit.total_detections,
                    'final_quality': np.mean(lifecycle.quality_history) if lifecycle.quality_history else 0.0,
                    'age': epoch - circuit.first_seen,
                    'reason': f"State: {lifecycle.state.value}, Votes: {lifecycle.removal_votes}"
                }

                # Actually remove from registry
                self._remove_circuit_from_registry(circuit_id)

                # Remove from lifecycle tracking
                del self.circuit_lifecycles[circuit_id]

                removed_circuits.append(removal_info)
                self.removal_history.append(removal_info)

        return removed_circuits

    def _remove_circuit_from_registry(self, circuit_id: str):
        """Remove circuit from all registry data structures"""
        if circuit_id in self.canonical_registry.canonical_circuits:
            circuit = self.canonical_registry.canonical_circuits[circuit_id]

            # Remove from main storage
            del self.canonical_registry.canonical_circuits[circuit_id]

            # Remove from signature mapping
            signature_hash = circuit.computational_signature.get_hash()
            if signature_hash in self.canonical_registry.signature_to_id:
                del self.canonical_registry.signature_to_id[signature_hash]

            # Clean up epoch tracking
            for epoch_set in self.canonical_registry.epoch_to_circuits.values():
                epoch_set.discard(circuit_id)

    def _apply_diversity_protection(self, epoch: int) -> List[str]:
        """Protect circuits important for diversity"""
        protected_circuits = []

        # Analyze current diversity
        diversity_analysis = self.diversity_tracker.analyze_diversity(
            self.canonical_registry.canonical_circuits
        )

        # Protect unique operation types
        for op_type, circuits in diversity_analysis['by_operation_type'].items():
            if len(circuits) <= 2:  # Protect if only 1-2 circuits of this type
                for circuit_id in circuits:
                    if circuit_id in self.circuit_lifecycles:
                        lifecycle = self.circuit_lifecycles[circuit_id]
                        if lifecycle.state != CircuitLifecycleState.PROTECTED:
                            lifecycle.state = CircuitLifecycleState.PROTECTED
                            lifecycle.protection_reason = f"Rare operation type: {op_type}"
                            protected_circuits.append(circuit_id)

        return protected_circuits

    def _generate_management_summary(self, quality_results: Dict, lifecycle_updates: Dict,
                                     removed_circuits: List, protected_circuits: List,
                                     epoch: int) -> Dict[str, Any]:
        """Generate comprehensive summary of circuit management"""

        # State distribution
        state_counts = {}
        for state in CircuitLifecycleState:
            state_counts[state.value] = sum(
                1 for lc in self.circuit_lifecycles.values() if lc.state == state
            )

        # Quality distribution
        all_qualities = []
        for circuit_id, analysis in quality_results['circuit_analyses'].items():
            all_qualities.append(analysis['quality_score'])

        return {
            'epoch': epoch,
            'circuit_management': {
                'total_circuits': len(self.canonical_registry.canonical_circuits),
                'circuits_removed': len(removed_circuits),
                'circuits_protected': len(protected_circuits),
                'new_circuits': len(lifecycle_updates['new_circuits']),
                'state_changes': len(lifecycle_updates['state_changes']),
            },
            'lifecycle_distribution': state_counts,
            'quality_statistics': {
                'mean_quality': np.mean(all_qualities) if all_qualities else 0.0,
                'median_quality': np.median(all_qualities) if all_qualities else 0.0,
                'quality_std': np.std(all_qualities) if all_qualities else 0.0,
            },
            'removed_circuits': removed_circuits,
            'protected_circuits': protected_circuits,
            'diversity_analysis': self.diversity_tracker.get_diversity_summary(
                self.canonical_registry.canonical_circuits
            ),
            'recommendations': self._generate_recommendations(quality_results, epoch)
        }

    def _generate_recommendations(self, quality_results: Dict, epoch: int) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        circuit_count = len(self.canonical_registry.canonical_circuits)
        max_circuits = self.config['max_circuits']

        if circuit_count > max_circuits * 0.9:
            recommendations.append(f"Approaching circuit limit ({circuit_count}/{max_circuits})")

        # Check for quality trends
        declining_count = sum(
            1 for lc in self.circuit_lifecycles.values()
            if lc.state == CircuitLifecycleState.DECLINING
        )

        if declining_count > circuit_count * 0.3:
            recommendations.append(f"High proportion of declining circuits ({declining_count}/{circuit_count})")

        # Check diversity
        operation_types = set(
            circuit.computational_signature.operation_type
            for circuit in self.canonical_registry.canonical_circuits.values()
        )

        if len(operation_types) < 3:
            recommendations.append("Low circuit diversity - consider protecting more circuit types")

        return recommendations

    # Public interface methods
    def get_circuit_lifecycle_info(self, circuit_id: str) -> Optional[CircuitLifecycleInfo]:
        """Get lifecycle information for a specific circuit"""
        return self.circuit_lifecycles.get(circuit_id)

    def force_protect_circuit(self, circuit_id: str, reason: str):
        """Manually protect a circuit from removal"""
        if circuit_id in self.circuit_lifecycles:
            lifecycle = self.circuit_lifecycles[circuit_id]
            lifecycle.state = CircuitLifecycleState.PROTECTED
            lifecycle.protection_reason = reason

    def get_removal_history(self) -> List[Dict]:
        """Get history of removed circuits"""
        return self.removal_history.copy()


class CircuitDiversityTracker:
    """Track and analyze circuit diversity"""

    def analyze_diversity(self, canonical_circuits: Dict) -> Dict[str, Any]:
        """Analyze diversity of current circuit population"""

        by_operation_type = {}
        by_layer = {}
        by_head = {}

        for circuit_id, circuit in canonical_circuits.items():
            # Group by operation type
            op_type = circuit.computational_signature.operation_type
            if op_type not in by_operation_type:
                by_operation_type[op_type] = []
            by_operation_type[op_type].append(circuit_id)

            # Group by layer (if available)
            if 'layer' in circuit.metadata:
                layer = circuit.metadata['layer']
                if layer not in by_layer:
                    by_layer[layer] = []
                by_layer[layer].append(circuit_id)

            # Group by head (if available)
            if 'head' in circuit.metadata:
                head = circuit.metadata['head']
                if head not in by_head:
                    by_head[head] = []
                by_head[head].append(circuit_id)

        return {
            'by_operation_type': by_operation_type,
            'by_layer': by_layer,
            'by_head': by_head
        }

    def get_diversity_summary(self, canonical_circuits: Dict) -> Dict[str, Any]:
        """Get summary statistics about diversity"""
        analysis = self.analyze_diversity(canonical_circuits)

        return {
            'operation_type_count': len(analysis['by_operation_type']),
            'layer_distribution': {k: len(v) for k, v in analysis['by_layer'].items()},
            'head_distribution': {k: len(v) for k, v in analysis['by_head'].items()},
            'diversity_score': self._calculate_diversity_score(analysis)
        }

    def _calculate_diversity_score(self, analysis: Dict) -> float:
        """Calculate overall diversity score (0-1, higher is better)"""
        # Simple diversity score based on how evenly distributed circuits are
        total_circuits = sum(len(circuits) for circuits in analysis['by_operation_type'].values())

        if total_circuits == 0:
            return 0.0

        # Calculate entropy-based diversity score
        operation_counts = [len(circuits) for circuits in analysis['by_operation_type'].values()]
        operation_probs = [count / total_circuits for count in operation_counts]

        entropy = -sum(p * np.log(p + 1e-10) for p in operation_probs)
        max_entropy = np.log(len(operation_probs) + 1e-10)

        return entropy / max_entropy if max_entropy > 0 else 0.0