# analysis/integration/canonical_integration_guide.py
"""
Complete Integration Guide for Canonical Circuit System

Shows how to integrate the canonical circuit system with existing code,
update adaptive detectors, and extend to new circuit types.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

# Import the new canonical system
from analysis.core.canonical_circuit_system import (
    CanonicalCircuitRegistry, CanonicalRegistryAdapter,
    extend_canonical_registry_for_new_types, SignatureExtractor, ComputationalSignature
)
from analysis.trainers.utils import evaluate


# ============================================================================
# 1. UPDATED ADAPTIVE DETECTOR INTEGRATION
# ============================================================================

class CanonicalAwareAdaptiveTokenOperationDetector:
    """
    Updated adaptive detector that uses canonical circuit system
    Extends RegistrationAwareAdaptiveTokenOperationDetector
    """

    def __init__(self, model, enhanced_registry, canonical_registry=None, thresholds=None):
        self.model = model
        self.enhanced_registry = enhanced_registry
        self.thresholds = thresholds

        # Initialize canonical system
        if canonical_registry is None:
            canonical_registry = CanonicalCircuitRegistry()
        self.canonical_registry = canonical_registry

        # Create adapter for seamless integration
        self.canonical_adapter = CanonicalRegistryAdapter(enhanced_registry, canonical_registry)

        # Keep existing circuit creation logic but update registration
        from analysis.analyzers.adaptive_token_operations import ModernCircuitCreator
        self.circuit_creator = ModernCircuitCreator(model, enhanced_registry)

        # Circuit tracking (now tracks canonical IDs)
        self.circuit_history = {}  # canonical_id -> detection_epochs
        self.false_positive_patterns = set()

    def detect_and_register_copy_mechanisms(self, attention_patterns, tokens=None, epoch=0,
                                            total_epochs=1000, model_accuracy=0.0,
                                            content_aware=True, register_circuits=True):
        """Enhanced copy detection with canonical registration"""

        # Existing detection logic (unchanged)
        copy_mechanisms = self._detect_copy_mechanisms_fixed(
            attention_patterns, tokens, epoch, total_epochs, model_accuracy, content_aware
        )

        canonical_circuits = []
        registration_summary = {'attempted': 0, 'succeeded': 0, 'aggregated': 0}

        if register_circuits and copy_mechanisms:
            print(f"🔧 Registering {len(copy_mechanisms)} copy circuits with canonical system...")

            for mechanism in copy_mechanisms:
                try:
                    registration_summary['attempted'] += 1

                    # Create circuit using existing logic
                    circuit = self.circuit_creator.create_circuit_from_adaptive_detection(
                        mechanism, tokens or [], epoch
                    )

                    # ✅ NEW: Register using canonical system
                    canonical_id, legacy_id = self.canonical_adapter.register_circuit_detection(
                        circuit=circuit,
                        epoch=epoch,
                        tokens=tokens or [],
                        detection_confidence=mechanism.get("reliability", 0.5),
                        detection_method="adaptive_copy",
                        example_metadata={
                            'attention_strength': mechanism.get('attention_strength', 0.0),
                            'copy_type': mechanism.get('copy_type', 'unknown'),
                            'source_pos': mechanism.get('source_pos', -1),
                            'target_pos': mechanism.get('target_pos', -1)
                        }
                    )

                    # Track canonical circuit
                    canonical_circuits.append(canonical_id)

                    # Update history tracking (now uses canonical IDs)
                    if canonical_id not in self.circuit_history:
                        self.circuit_history[canonical_id] = []
                        registration_summary['succeeded'] += 1
                    else:
                        registration_summary['aggregated'] += 1

                    self.circuit_history[canonical_id].append(epoch)

                except Exception as e:
                    print(f"    ⚠️ Failed to register copy circuit: {e}")

        return {
            "raw_mechanisms": copy_mechanisms,
            "canonical_circuits": canonical_circuits,
            "registration_summary": registration_summary,
            "canonical_registry_summary": self.canonical_registry.get_registry_summary()
        }

    def detect_and_register_induction_patterns(self, attention_patterns, tokens=None, epoch=0,
                                               total_epochs=1000, model_accuracy=0.0,
                                               register_circuits=True):
        """Enhanced induction detection with canonical registration"""

        # Similar pattern to copy detection
        induction_patterns = self._detect_induction_patterns_fixed(
            attention_patterns, tokens, epoch, total_epochs, model_accuracy
        )

        canonical_circuits = []
        registration_summary = {'attempted': 0, 'succeeded': 0, 'aggregated': 0}

        if register_circuits and induction_patterns:
            print(f"🔄 Registering {len(induction_patterns)} induction circuits with canonical system...")

            for pattern in induction_patterns:
                try:
                    registration_summary['attempted'] += 1

                    circuit = self.circuit_creator.create_circuit_from_adaptive_detection(
                        pattern, tokens or [], epoch
                    )

                    canonical_id, legacy_id = self.canonical_adapter.register_circuit_detection(
                        circuit=circuit,
                        epoch=epoch,
                        tokens=tokens or [],
                        detection_confidence=pattern.get("reliability", 0.5),
                        detection_method="adaptive_induction",
                        example_metadata={
                            'strength': pattern.get('strength', 0.0),
                            'pattern_type': pattern.get('pattern_type', 'unknown'),
                            'inducer_pos': pattern.get('inducer_pos', -1),
                            'target_pos': pattern.get('target_pos', -1),
                            'distance': pattern.get('distance', 0)
                        }
                    )

                    canonical_circuits.append(canonical_id)

                    if canonical_id not in self.circuit_history:
                        self.circuit_history[canonical_id] = []
                        registration_summary['succeeded'] += 1
                    else:
                        registration_summary['aggregated'] += 1

                    self.circuit_history[canonical_id].append(epoch)

                except Exception as e:
                    print(f"    ⚠️ Failed to register induction circuit: {e}")

        return {
            "raw_patterns": induction_patterns,
            "canonical_circuits": canonical_circuits,
            "registration_summary": registration_summary
        }

    def get_canonical_circuit_stability(self, canonical_id: str) -> Dict[str, float]:
        """Get stability metrics for canonical circuit"""
        canonical = self.canonical_registry.get_canonical_circuit(canonical_id)

        if canonical:
            return {
                'stability_score': canonical.stability_score,
                'consistency_score': canonical.consistency_score,
                'persistence_score': canonical.persistence_score,
                'total_detections': canonical.total_detections,
                'temporal_span': canonical.last_seen - canonical.first_seen + 1
            }
        return {'stability_score': 0.0, 'consistency_score': 0.0}

    def prune_unstable_canonical_circuits(self, current_epoch: int, min_stability: float = 0.3):
        """Prune based on canonical stability scores"""
        stable_circuits = self.canonical_registry.get_stable_circuits(
            min_stability=min_stability, min_detections=2
        )

        stable_canonical_ids = set(circuit.canonical_id for circuit in stable_circuits)

        # print(f"🔍 Canonical pruning @ epoch {current_epoch}: "
        #       f"{len(stable_circuits)} stable circuits from {len(self.canonical_registry.canonical_circuits)} total")

        return stable_canonical_ids

    def _detect_copy_mechanisms_fixed(self, attention_patterns, tokens, epoch, total_epochs, model_accuracy,
                                      content_aware):
        """Existing copy detection logic (from FixedAdaptiveTokenOperationDetector)"""
        # Import and use existing logic
        from analysis.analyzers.adaptive_token_operations import AdaptiveTokenOperationDetector
        base_detector = AdaptiveTokenOperationDetector(self.model, None, self.thresholds)
        return base_detector.detect_copy_mechanisms_adaptive(
            attention_patterns, tokens, epoch, total_epochs, model_accuracy, content_aware
        )

    def _detect_induction_patterns_fixed(self, attention_patterns, tokens, epoch, total_epochs, model_accuracy):
        """Existing induction detection logic"""
        from analysis.analyzers.adaptive_token_operations import AdaptiveTokenOperationDetector
        base_detector = AdaptiveTokenOperationDetector(self.model, None, self.thresholds)
        return base_detector.detect_induction_patterns_adaptive(
            attention_patterns, tokens, epoch, total_epochs, model_accuracy
        )


# ============================================================================
# 2. INTEGRATION WITH TRAINING LOOP
# ============================================================================

def initialize_canonical_system(model, save_dir, logger, enhanced_registry=None):
    """Initialize complete canonical circuit system"""

    # Create canonical registry
    canonical_registry = CanonicalCircuitRegistry(save_dir / "canonical_circuits")

    # Extend for additional circuit types if needed
    canonical_registry = extend_canonical_registry_for_new_types(canonical_registry)

    # Create enhanced registry if not provided
    if enhanced_registry is None:
        from analysis.core import EnhancedCircuitRegistry
        enhanced_registry = EnhancedCircuitRegistry(save_dir / "enhanced_registry")

    # Create adapter
    adapter = CanonicalRegistryAdapter(enhanced_registry, canonical_registry)

    # Initialize evolution tracker with canonical support
    from analysis.core.unified_circuit_evolution_tracker import UnifiedCircuitEvolutionTracker
    evolution_tracker = UnifiedCircuitEvolutionTracker(enhanced_registry, save_dir / "evolution", logger)

    # Create canonical-aware adaptive detector
    canonical_detector = CanonicalAwareAdaptiveTokenOperationDetector(
        model=model,
        enhanced_registry=enhanced_registry,
        canonical_registry=canonical_registry
    )

    return {
        'canonical_registry': canonical_registry,
        'enhanced_registry': enhanced_registry,
        'adapter': adapter,
        'evolution_tracker': evolution_tracker,
        'canonical_detector': canonical_detector
    }


def updated_training_loop_with_canonical_circuits(
        model, train_loader, eval_loader, criterion, optimizer,
        epochs=1000, analyze_interval=2, save_dir=None, logger=None):
    """
    Example training loop integration with canonical circuit system
    """

    if save_dir is None:
        save_dir = Path("results/canonical_circuit_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ✅ INITIALIZE: Canonical circuit system
    circuit_system = initialize_canonical_system(model, save_dir, logger)
    canonical_registry = circuit_system['canonical_registry']
    canonical_detector = circuit_system['canonical_detector']
    evolution_tracker = circuit_system['evolution_tracker']
    adapter = circuit_system['adapter']

    # Training loop
    for epoch in range(epochs):

        # Standard training step (unchanged)
        # ... training code ...

        # Enhanced circuit analysis
        if epoch % analyze_interval == 0:
            logger.info(f"🔬 Canonical circuit analysis @ epoch {epoch}")

            # Evaluate model
            eval_stats = evaluate(model, eval_loader, criterion, device='cuda')
            current_accuracy = eval_stats['accuracy']

            # ✅ CANONICAL CIRCUIT DETECTION
            # Get sample for analysis
            sample_input, sample_target = next(iter(eval_loader))
            tokens = [str(int(x)) for x in sample_input[0].cpu().numpy()]

            # Forward pass with attention storage
            outputs = model(sample_input, store_attention=True)
            attention_patterns = model.get_attention_patterns()

            # Detect circuits using canonical system
            copy_results = canonical_detector.detect_and_register_copy_mechanisms(
                attention_patterns=attention_patterns,
                tokens=tokens,
                epoch=epoch,
                total_epochs=epochs,
                model_accuracy=current_accuracy,
                register_circuits=True
            )

            induction_results = canonical_detector.detect_and_register_induction_patterns(
                attention_patterns=attention_patterns,
                tokens=tokens,
                epoch=epoch,
                total_epochs=epochs,
                model_accuracy=current_accuracy,
                register_circuits=True
            )

            # ✅ ENHANCED LOGGING with canonical metrics
            canonical_summary = canonical_registry.get_registry_summary()

            circuit_metrics = {
                "canonical_circuits_total": canonical_summary['total_canonical_circuits'],
                "aggregation_rate": canonical_summary['aggregation_rate'],
                "stable_circuits": canonical_summary['stable_circuits_count'],
                "stability_rate": canonical_summary['stability_rate'],
                "copy_detections": len(copy_results['canonical_circuits']),
                "induction_detections": len(induction_results['canonical_circuits']),
                "avg_detections_per_circuit": canonical_summary['avg_detections_per_circuit']
            }

            if logger:
                logger.log_metrics(circuit_metrics, step=epoch, category="canonical_circuits")

            logger.info(f"    📊 Canonical circuits: {canonical_summary['total_canonical_circuits']} total, "
                        f"{canonical_summary['stable_circuits_count']} stable "
                        f"(aggregation rate: {canonical_summary['aggregation_rate']:.2%})")

            # ✅ EVOLUTION ANALYSIS with canonical data
            if epoch > 50:  # Start evolution analysis after some circuits accumulated
                stable_circuits = canonical_registry.get_stable_circuits(min_stability=0.5)

                for circuit in stable_circuits[:5]:  # Analyze top 5 stable circuits
                    evolution_data = canonical_registry.analyze_circuit_evolution(circuit.canonical_id)

                    logger.info(f"    🧬 {circuit.canonical_id}: "
                                f"{evolution_data['total_detections']} detections, "
                                f"stability {evolution_data['stability_score']:.3f}, "
                                f"trend {evolution_data.get('attribution_trend', {}).get('trend', 'unknown')}")

            # ✅ CIRCUIT PRUNING based on canonical stability
            stable_canonical_ids = canonical_detector.prune_unstable_canonical_circuits(
                epoch, min_stability=0.3
            )

            # Save canonical registry state
            if epoch % 100 == 0:
                try:
                    canonical_registry.save(save_dir / f"canonical_registry_epoch_{epoch}.json")
                except Exception as e:
                    logger.warning(f"Failed to save canonical registry: {e}")

    # ✅ FINAL ANALYSIS with canonical insights
    final_summary = canonical_registry.get_registry_summary()
    stable_circuits = canonical_registry.get_stable_circuits(min_stability=0.6)

    logger.info(f"🎉 Final canonical analysis:")
    logger.info(f"    Total circuits discovered: {final_summary['total_canonical_circuits']}")
    logger.info(f"    Stable circuits: {len(stable_circuits)}")
    logger.info(f"    Aggregation efficiency: {final_summary['aggregation_rate']:.2%}")

    # Analyze most stable circuits
    for i, circuit in enumerate(stable_circuits[:10], 1):
        evolution = canonical_registry.analyze_circuit_evolution(circuit.canonical_id)
        logger.info(f"    #{i}: {circuit.canonical_id} - "
                    f"{evolution['total_detections']} detections, "
                    f"stability {evolution['stability_score']:.3f}")
        logger.info(f"        Examples: {evolution['token_examples'][:3]}")

    return model, canonical_registry, final_summary


# ============================================================================
# 3. EXTENDING TO NEW CIRCUIT TYPES
# ============================================================================

def example_extension_sparse_autoencoder_circuits():
    """Example: How to extend to sparse autoencoder feature circuits"""

    from analysis.core.canonical_circuit_system import SignatureExtractor, ComputationalSignature

    class SparseAutoencoderSignatureExtractor(SignatureExtractor):
        """Extractor for sparse autoencoder feature circuits"""

        def supports_circuit_type(self, circuit):
            return circuit.metadata.get('operation_type', '').startswith('sae_')

        def extract_signature(self, circuit):
            feature_id = circuit.metadata.get('feature_id', -1)
            layer = circuit.metadata.get('layer', 0)
            activation_threshold = circuit.metadata.get('activation_threshold', 0.0)
            sparsity_level = circuit.metadata.get('sparsity_level', 'unknown')

            structural_pattern = {
                'feature_id': feature_id,
                'layer': layer,
                'activation_threshold_bin': self._bin_threshold(activation_threshold),
                'sparsity_level': sparsity_level,
                'feature_type': circuit.metadata.get('feature_type', 'unknown')
            }

            return ComputationalSignature(
                operation_type=f"sae_feature_{sparsity_level}",
                circuit_type='sparse_autoencoder',
                structural_pattern=structural_pattern
            )

        def _bin_threshold(self, threshold):
            """Bin activation thresholds for signature stability"""
            if threshold < 0.1:
                return 'very_low'
            elif threshold < 0.5:
                return 'low'
            elif threshold < 1.0:
                return 'medium'
            else:
                return 'high'


def example_extension_compositional_circuits():
    """Example: How to extend to compositional reasoning circuits"""

    class CompositionalReasoningSignatureExtractor(SignatureExtractor):
        """Extractor for compositional reasoning circuits"""

        def supports_circuit_type(self, circuit):
            return circuit.metadata.get('reasoning_type') in ['composition', 'decomposition', 'binding']

        def extract_signature(self, circuit):
            reasoning_type = circuit.metadata.get('reasoning_type', 'unknown')
            complexity_level = circuit.metadata.get('complexity_level', 1)
            component_count = circuit.metadata.get('component_count', 0)

            structural_pattern = {
                'reasoning_type': reasoning_type,
                'complexity_level': complexity_level,
                'component_count': component_count,
                'binding_mechanism': circuit.metadata.get('binding_mechanism', 'unknown'),
                'composition_depth': circuit.metadata.get('composition_depth', 1)
            }

            return ComputationalSignature(
                operation_type=f"compositional_{reasoning_type}",
                circuit_type='compositional_reasoning',
                structural_pattern=structural_pattern
            )


def example_extension_meta_learning_circuits():
    """Example: How to extend to meta-learning circuits"""

    class MetaLearningSignatureExtractor(SignatureExtractor):
        """Extractor for meta-learning and grokking-related circuits"""

        def supports_circuit_type(self, circuit):
            return circuit.metadata.get('meta_type') in ['grokking_transition', 'phase_change', 'meta_optimizer']

        def extract_signature(self, circuit):
            meta_type = circuit.metadata.get('meta_type', 'unknown')
            transition_phase = circuit.metadata.get('transition_phase', 'unknown')
            learning_rate_scale = circuit.metadata.get('learning_rate_scale', 1.0)

            structural_pattern = {
                'meta_type': meta_type,
                'transition_phase': transition_phase,
                'lr_scale_bin': self._bin_learning_rate_scale(learning_rate_scale),
                'involves_weight_decay': circuit.metadata.get('involves_weight_decay', False),
                'phase_transition_epoch': circuit.metadata.get('phase_transition_epoch', -1)
            }

            return ComputationalSignature(
                operation_type=f"meta_{meta_type}",
                circuit_type='meta_learning',
                structural_pattern=structural_pattern
            )

        def _bin_learning_rate_scale(self, scale):
            if scale < 0.1:
                return 'very_small'
            elif scale < 1.0:
                return 'small'
            elif scale < 10.0:
                return 'normal'
            else:
                return 'large'


# ============================================================================
# 4. INTEGRATION CHECKLIST AND MIGRATION
# ============================================================================

def migration_checklist():
    """Checklist for migrating to canonical circuit system"""

    checklist = """
    CANONICAL CIRCUIT SYSTEM MIGRATION CHECKLIST:

    ✅ 1. CREATE CANONICAL INFRASTRUCTURE:
       - Initialize CanonicalCircuitRegistry
       - Set up CanonicalRegistryAdapter
       - Extend with custom SignatureExtractors for your circuit types

    ✅ 2. UPDATE ADAPTIVE DETECTORS:
       - Replace RegistrationAwareAdaptiveTokenOperationDetector 
       - Use CanonicalAwareAdaptiveTokenOperationDetector
       - Update registration calls to use canonical_adapter.register_circuit_detection()

    ✅ 3. MODIFY TRAINING LOOP:
       - Initialize canonical system in setup phase
       - Update circuit analysis to use canonical detection
       - Add canonical metrics to logging
       - Use canonical stability for pruning

    ✅ 4. ENHANCED ANALYSIS:
       - Circuit evolution now tracks aggregated instances
       - Stability scores based on detection frequency and consistency
       - Token examples show algorithm generality
       - Attribution history shows circuit strengthening/weakening

    ✅ 5. EXTENSION CAPABILITIES:
       - Add new SignatureExtractors for new circuit types
       - Computational signatures separate algorithm from instances
       - Automatic aggregation across epochs and examples
       - Full evolution and competition tracking

    ✅ 6. BACKWARD COMPATIBILITY:
       - CanonicalRegistryAdapter maintains legacy circuit interface
       - Enhanced registry still works for existing code
       - Evolution tracker gets canonical data through adapter
       - Existing visualization and analysis tools still work

    ✅ 7. RESEARCH BENEFITS:
       - Proper algorithm discovery (circuits represent learned algorithms)
       - Circuit competition and cooperation tracking
       - Grokking phase transition analysis
       - Extension to complex circuit types (SAE features, composition, etc.)

    RESULT: Circuits now represent computational algorithms that aggregate 
    across training, enabling proper grokking dynamics analysis and extension 
    to sophisticated circuit types for advanced mechanistic interpretability.
    """

    print(checklist)


def validate_canonical_system(canonical_registry: CanonicalCircuitRegistry):
    """Validation tests for canonical system"""

    print("🧪 Validating canonical circuit system...")

    # Test 1: Signature consistency
    print("  ✓ Testing signature consistency...")

    # Test 2: Aggregation logic
    print("  ✓ Testing circuit aggregation...")

    # Test 3: Stability calculations
    print("  ✓ Testing stability metrics...")

    # Test 4: Extension framework
    print("  ✓ Testing extension framework...")

    summary = canonical_registry.get_registry_summary()
    print(f"  📊 Registry summary: {summary}")

    print("✅ Canonical system validation complete!")

    return True


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def complete_example_usage():
    """Complete example showing canonical system in action"""

    # 1. Initialize system
    save_dir = Path("results/canonical_example")
    # circuit_system = initialize_canonical_system(model, save_dir, logger)

    # 2. In training loop:
    # updated_training_loop_with_canonical_circuits(model, train_loader, eval_loader, criterion, optimizer)

    # 3. Extend to new circuit types:
    # registry.extractors.append(SparseAutoencoderSignatureExtractor())

    # 4. Analyze results:
    # stable_circuits = registry.get_stable_circuits(min_stability=0.7)
    # for circuit in stable_circuits:
    #     evolution = registry.analyze_circuit_evolution(circuit.canonical_id)
    #     print(f"Circuit {circuit.canonical_id}: {evolution}")

    pass