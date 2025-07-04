# analysis/canonical/sampled_circuit_analysis.py
"""
Canonical Circuit Detection with ExampleSampler Integration

Uses intelligent example sampling to find robust canonical circuits across
multiple diverse examples, enabling proper algorithm discovery.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from analysis.helpers.example_sampler import ExampleSampler
from analysis.core.canonical_circuit_system import CanonicalCircuitRegistry, CanonicalRegistryAdapter
from analysis.sampling.fast_subset_sampler import FastCircuitSampler


# Create single canonical initialization function
def create_standard_canonical_system(model, save_dir, logger, eval_loader, thresholds=None):
    """Standard canonical system initialization - USE THIS EVERYWHERE"""

    # Use JSON-safe canonical registry (proven to work)
    from analysis.core.json_safe_canonical_circuits import JSONSafeCanonicalCircuitRegistry
    from analysis.core.canonical_circuit_system import extend_canonical_registry_for_new_types
    canonical_registry = JSONSafeCanonicalCircuitRegistry(save_dir / "canonical_circuits")
    canonical_registry = extend_canonical_registry_for_new_types(canonical_registry)

    # Standard enhanced registry
    from analysis import EnhancedCircuitRegistry
    enhanced_registry = EnhancedCircuitRegistry(save_dir / "enhanced_registry")

    # JSON-safe adapter (critical for data persistence)
    from analysis.core.json_safe_canonical_circuits import JSONSafeCanonicalRegistryAdapter
    adapter = JSONSafeCanonicalRegistryAdapter(enhanced_registry, canonical_registry)

    # Evolution tracker with unified interface
    from analysis.core.circuit_evolution_tracker import CircuitEvolutionTracker
    evolution_tracker = CircuitEvolutionTracker(circuit_metadata=model.circuit_metadata,)

    # Standard detector - ALWAYS use CanonicalAwareAdaptiveTokenOperationDetector
    from analysis.analyzers.adaptive_token_operations import CanonicalAwareAdaptiveTokenOperationDetector
    canonical_detector = CanonicalAwareAdaptiveTokenOperationDetector(
        model=model,
        enhanced_registry=enhanced_registry,
        canonical_registry=canonical_registry,
        enable_dynamic_thresholds=True
    )

    return {
        'canonical_registry': canonical_registry,
        'enhanced_registry': enhanced_registry,
        'adapter': adapter,
        'evolution_tracker': evolution_tracker,
        'canonical_detector': canonical_detector
    }


def run_canonical_circuit_analysis_with_sampling(
        canonical_detector, example_sampler: ExampleSampler,
        epoch: int, total_epochs: int, accuracy: float,
        sampling_strategy: str = "diverse_random",
        logger=None, detect_induction: bool = True,
        analyze_interval: int = 2,
        cross_example_threshold: float = 0.3,
        min_examples_for_robustness: int = 2) -> Dict[str, Any]:
    """
    Complete canonical circuit analysis using ExampleSampler

    Args:
        canonical_detector: CanonicalAwareAdaptiveTokenOperationDetector
        example_sampler: ExampleSampler for intelligent example selection
        epoch: Current epoch
        total_epochs: Total training epochs
        accuracy: Current model accuracy
        sampling_strategy: Strategy for example sampling
        logger: Optional logger
        detect_induction: Whether to detect induction patterns
        analyze_interval: Analysis interval for sampling budget
        cross_example_threshold: Threshold for cross-example pattern detection
        min_examples_for_robustness: Minimum examples needed for robust pattern

    Returns:
        Dict with comprehensive canonical circuit analysis results
    """

    # ---------------------------------------------------------------------------------------
    # Setup logging
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    # ============================================================================
    # 1. INTELLIGENT EXAMPLE SAMPLING
    # ============================================================================

    # Get diverse examples for analysis
    sampled_examples = example_sampler.sample_examples(
        epoch=epoch,
        total_epochs=total_epochs,
        strategy=sampling_strategy, # num_samples=num_samples, # fixme non-existant here?
        seed_offset=epoch % 100  # Vary seed based on epoch
    )

    # logger.info(f"🎲 Canonical analysis @ epoch {epoch}: "
    #             f"sampled {len(sampled_examples)} examples using '{sampling_strategy}'")

    # ============================================================================
    # 2. PER-EXAMPLE CIRCUIT DETECTION (WITHOUT REGISTRATION)
    # ============================================================================

    example_analyses = []
    all_copy_detections = []
    all_induction_detections = []

    for example_idx, (inputs, targets) in enumerate(sampled_examples):
        logger.debug(f"  🔍 Analyzing example {example_idx + 1}/{len(sampled_examples)}")

        # Extract tokens for analysis
        tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]

        # Forward pass with attention storage
        try:
            outputs = canonical_detector.model(inputs, store_attention=True)
            attention_patterns = canonical_detector.model.get_attention_patterns()
        except Exception as e:
            logger.warning(f"    ⚠️ Failed forward pass for example {example_idx}: {e}")
            continue

        # Detect copy circuits (without registration)
        try:
            copy_results = canonical_detector._detect_copy_mechanisms_fixed(
                attention_patterns=attention_patterns,
                tokens=tokens,
                epoch=epoch,
                total_epochs=total_epochs,
                model_accuracy=accuracy,
                content_aware=True
            )
        except Exception as e:
            logger.warning(f"    ⚠️ Copy detection failed for example {example_idx}: {e}")
            copy_results = []

        # Detect induction circuits (without registration)
        induction_results = []
        if detect_induction:
            try:
                induction_results = canonical_detector._detect_induction_patterns_fixed(
                    attention_patterns=attention_patterns,
                    tokens=tokens,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    model_accuracy=accuracy
                )
            except Exception as e:
                logger.warning(f"    ⚠️ Induction detection failed for example {example_idx}: {e}")

        # Store example analysis
        example_analysis = {
            'example_idx': example_idx,
            'tokens': tokens,
            'copy_detections': copy_results,
            'induction_detections': induction_results,
            'attention_patterns': attention_patterns  # Store for circuit creation
        }

        example_analyses.append(example_analysis)
        all_copy_detections.extend(copy_results)
        all_induction_detections.extend(induction_results)
        #---------------------------------------------------------------------------------------

        logger.debug(f"    📊 Example {example_idx}: "
                     f"{len(copy_results)} copy, {len(induction_results)} induction detections")

    # logger.info(f"  📊 Total detections: {len(all_copy_detections)} copy, "
    #             f"{len(all_induction_detections)} induction across {len(sampled_examples)} examples")

    # ============================================================================
    # 3. CROSS-EXAMPLE PATTERN ANALYSIS
    # ============================================================================

    # logger.info("🔗 Analyzing cross-example canonical patterns...")

    # Analyze copy patterns across examples
    robust_copy_patterns = analyze_canonical_copy_patterns(
        example_analyses,
        cross_example_threshold=cross_example_threshold,
        min_examples=min_examples_for_robustness,
        logger=logger
    )

    # Analyze induction patterns across examples
    robust_induction_patterns = analyze_canonical_induction_patterns(
        example_analyses,
        cross_example_threshold=cross_example_threshold,
        min_examples=min_examples_for_robustness,
        logger=logger
    )

    # logger.info(f"  🏆 Found {len(robust_copy_patterns)} robust copy patterns, "
    #             f"{len(robust_induction_patterns)} robust induction patterns")

    # ============================================================================
    # 4. CANONICAL CIRCUIT REGISTRATION
    # ============================================================================

    # logger.info("📝 Registering robust canonical circuits...")

    registered_canonical_circuits = []
    registration_summary = {
        'attempted_copy': 0,
        'registered_copy': 0,
        'attempted_induction': 0,
        'registered_induction': 0,
        'aggregated': 0,
        'failed': 0
    }

    # Register robust copy circuits
    for pattern_key, pattern_data in robust_copy_patterns.items():
        registration_summary['attempted_copy'] += 1

        try:
            canonical_id = register_robust_canonical_circuit(
                canonical_detector=canonical_detector,
                pattern_key=pattern_key,
                pattern_data=pattern_data,
                circuit_type='copy',
                epoch=epoch,
                logger=logger
            )

            if canonical_id:
                registered_canonical_circuits.append(canonical_id)
                registration_summary['registered_copy'] += 1

        except Exception as e:
            logger.warning(f"    ⚠️ Failed to register copy pattern {pattern_key}: {e}")
            registration_summary['failed'] += 1

    # Register robust induction circuits
    for pattern_key, pattern_data in robust_induction_patterns.items():
        registration_summary['attempted_induction'] += 1

        try:
            canonical_id = register_robust_canonical_circuit(
                canonical_detector=canonical_detector,
                pattern_key=pattern_key,
                pattern_data=pattern_data,
                circuit_type='induction',
                epoch=epoch,
                logger=logger
            )

            if canonical_id:
                registered_canonical_circuits.append(canonical_id)
                registration_summary['registered_induction'] += 1

        except Exception as e:
            logger.warning(f"    ⚠️ Failed to register induction pattern {pattern_key}: {e}")
            registration_summary['failed'] += 1

    # ======================================================================================
    # 5. CANONICAL REGISTRY ANALYSIS
    # ======================================================================================

    # logger.info("📈 Analyzing canonical registry state...")

    # Get current registry state
    registry_summary = canonical_detector.canonical_registry.get_registry_summary()

    # Analyze newly registered circuits
    newly_registered_analysis = {}
    for canonical_id in registered_canonical_circuits:
        evolution_data = canonical_detector.canonical_registry.analyze_circuit_evolution(canonical_id)
        newly_registered_analysis[canonical_id] = evolution_data

    # Get stable circuits
    stable_circuits = canonical_detector.canonical_registry.get_stable_circuits(
        min_stability=0.6, min_detections=2
    )

    # ======================================================================================
    # 6. COMPREHENSIVE RESULTS AGGREGATION
    # ======================================================================================

    analysis_results = {
        # Sampling results
        'sampling_summary': {
            'strategy': sampling_strategy,
            'examples_analyzed': len(sampled_examples),
            'total_copy_detections': len(all_copy_detections),
            'total_induction_detections': len(all_induction_detections),
            'examples_with_circuits': len([ea for ea in example_analyses
                                           if ea['copy_detections'] or ea['induction_detections']])
        },

        # Cross-example pattern analysis
        'cross_example_patterns': {
            'robust_copy_patterns': robust_copy_patterns,
            'robust_induction_patterns': robust_induction_patterns,
            'copy_pattern_diversity': len(robust_copy_patterns),
            'induction_pattern_diversity': len(robust_induction_patterns)
        },

        # Registration results
        'registration_summary': registration_summary,
        'registered_canonical_circuits': registered_canonical_circuits,
        'newly_registered_analysis': newly_registered_analysis,

        # Registry state
        'registry_summary': registry_summary,
        'stable_circuits_count': len(stable_circuits),
        'stable_circuits': [
            {
                'canonical_id': circuit.canonical_id,
                'stability_score': circuit.stability_score,
                'total_detections': circuit.total_detections,
                'operation_type': circuit.computational_signature.operation_type
            }
            for circuit in stable_circuits[:10]  # Top 10 for logging
        ],

        # Evolution insights
        'evolution_insights': analyze_canonical_evolution_insights(
            canonical_detector.canonical_registry, epoch, logger
        ),

        # Raw data for further analysis
        'example_analyses': example_analyses  # Full data for debugging
    }

    # ============================================================================
    # 7. ENHANCED LOGGING
    # ============================================================================

    log_canonical_analysis_results(analysis_results, epoch, logger)

    return analysis_results

# ============================================================================
# INTEGRATION WITH CANONICAL CIRCUIT ANALYSIS
# ============================================================================
def run_canonical_circuit_analysis_with_fast_sampling(
        canonical_detector, eval_loader,
        epoch: int, total_epochs: int, accuracy: float,
        logger=None, detect_induction: bool = True,
        cross_example_threshold: float = 0.3,
        min_examples_for_robustness: int = 2,
        num_samples: int = 8) -> Dict[str, Any]:
    """
    Canonical circuit analysis with fast sampling and robust cross-example analysis

    Args:
        canonical_detector: CanonicalAwareAdaptiveTokenOperationDetector
        eval_loader: DataLoader for sampling examples
        epoch: Current epoch
        total_epochs: Total training epochs
        accuracy: Current model accuracy
        logger: Optional logger
        detect_induction: Whether to detect induction patterns
        cross_example_threshold: Threshold for cross-example pattern detection
        min_examples_for_robustness: Minimum examples needed for robust pattern
        num_samples: Number of examples to sample

    Returns:
        Dict with comprehensive canonical circuit analysis results
    """

    # Setup logging
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    # ============================================================================
    # 1. FAST EXAMPLE SAMPLING
    # ============================================================================

    # Create simple fast sampler
    fast_sampler = FastCircuitSampler(eval_loader)
    sampled_examples = fast_sampler.sample_examples(
        epoch=epoch,
        total_epochs=total_epochs,
        # strategy="fast_random",  # num_samples=num_samples, # fixme non-existant here?
        num_samples=num_samples,
        seed_offset=epoch % 100  # Vary seed based on epoch
    )

    # logger.info(f"🎲 Canonical analysis @ epoch {epoch}: sampled {len(sampled_examples)} examples")

    # ============================================================================
    # 2. PER-EXAMPLE CIRCUIT DETECTION (WITHOUT REGISTRATION)
    # ============================================================================

    example_analyses = []
    all_copy_detections = []
    all_induction_detections = []

    for example_idx, (inputs, targets) in enumerate(sampled_examples):
        logger.debug(f"  🔍 Analyzing example {example_idx + 1}/{len(sampled_examples)}")

        # Extract tokens for analysis
        tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]

        # Forward pass with attention storage
        try:
            outputs = canonical_detector.model(inputs, store_attention=True)
            attention_patterns = canonical_detector.model.get_attention_patterns()
        except Exception as e:
            logger.warning(f"    ⚠️ Failed forward pass for example {example_idx}: {e}")
            continue

        # Detect copy circuits (without registration)
        try:
            copy_results = canonical_detector._detect_copy_mechanisms_fixed(
                attention_patterns=attention_patterns,
                tokens=tokens,
                epoch=epoch,
                total_epochs=total_epochs,
                model_accuracy=accuracy,
                content_aware=True
            )
        except Exception as e:
            logger.warning(f"    ⚠️ Copy detection failed for example {example_idx}: {e}")
            copy_results = []

        # Detect induction circuits (without registration)
        induction_results = []
        if detect_induction:
            try:
                induction_results = canonical_detector._detect_induction_patterns_fixed(
                    attention_patterns=attention_patterns,
                    tokens=tokens,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    model_accuracy=accuracy
                )
            except Exception as e:
                logger.warning(f"    ⚠️ Induction detection failed for example {example_idx}: {e}")

        # Store example analysis
        example_analysis = {
            'example_idx': example_idx,
            'tokens': tokens,
            'copy_detections': copy_results,
            'induction_detections': induction_results,
            'attention_patterns': attention_patterns  # Store for circuit creation
        }

        example_analyses.append(example_analysis)
        all_copy_detections.extend(copy_results)
        all_induction_detections.extend(induction_results)

        logger.debug(f"    📊 Example {example_idx}: "
                     f"{len(copy_results)} copy, {len(induction_results)} induction detections")

    # logger.info(f"  📊 Total detections: {len(all_copy_detections)} copy, "
    #             f"{len(all_induction_detections)} induction across {len(sampled_examples)} examples")

    # ============================================================================
    # 3. CROSS-EXAMPLE PATTERN ANALYSIS (ROBUST AGGREGATION)
    # ============================================================================

    # logger.info("🔗 Analyzing cross-example canonical patterns...")

    # Analyze copy patterns across examples
    robust_copy_patterns = analyze_canonical_copy_patterns(
        example_analyses,
        cross_example_threshold=cross_example_threshold,
        min_examples=min_examples_for_robustness,
        logger=logger
    )

    # Analyze induction patterns across examples
    robust_induction_patterns = analyze_canonical_induction_patterns(
        example_analyses,
        cross_example_threshold=cross_example_threshold,
        min_examples=min_examples_for_robustness,
        logger=logger
    )

    # logger.info(f"  🏆 Found {len(robust_copy_patterns)} robust copy patterns, "
    #             f"{len(robust_induction_patterns)} robust induction patterns")
    # logger.info(f"  🏆 {len(all_copy_detections)} / {len(robust_copy_patterns)} all / robust copy, "
    #             f"{all_induction_detections} / {len(robust_induction_patterns)} all / robust induction")
    # ============================================================================
    # 4. CANONICAL CIRCUIT REGISTRATION
    # ============================================================================

    # logger.info("📝 Registering robust canonical circuits...")

    registered_canonical_circuits = []
    registration_summary = {
        'attempted_copy': 0,
        'registered_copy': 0,
        'attempted_induction': 0,
        'registered_induction': 0,
        'aggregated': 0,
        'failed': 0
    }

    # Register robust copy circuits
    for pattern_key, pattern_data in robust_copy_patterns.items():
        registration_summary['attempted_copy'] += 1

        try:
            canonical_id = register_robust_canonical_circuit(
                canonical_detector=canonical_detector,
                pattern_key=pattern_key,
                pattern_data=pattern_data,
                circuit_type='copy',
                epoch=epoch,
                logger=logger
            )

            if canonical_id:
                registered_canonical_circuits.append(canonical_id)
                registration_summary['registered_copy'] += 1

        except Exception as e:
            logger.warning(f"    ⚠️ Failed to register copy pattern {pattern_key}: {e}")
            registration_summary['failed'] += 1

    # Register robust induction circuits
    for pattern_key, pattern_data in robust_induction_patterns.items():
        registration_summary['attempted_induction'] += 1

        try:
            canonical_id = register_robust_canonical_circuit(
                canonical_detector=canonical_detector,
                pattern_key=pattern_key,
                pattern_data=pattern_data,
                circuit_type='induction',
                epoch=epoch,
                logger=logger
            )

            if canonical_id:
                registered_canonical_circuits.append(canonical_id)
                registration_summary['registered_induction'] += 1

        except Exception as e:
            logger.warning(f"    ⚠️ Failed to register induction pattern {pattern_key}: {e}")
            registration_summary['failed'] += 1

    # ============================================================================
    # 5. CANONICAL REGISTRY ANALYSIS
    # ============================================================================

    # logger.info("📈 Analyzing canonical registry state...")

    # Get current registry state
    registry_summary = canonical_detector.canonical_registry.get_registry_summary()

    # Analyze newly registered circuits
    newly_registered_analysis = {}
    for canonical_id in registered_canonical_circuits:
        evolution_data = canonical_detector.canonical_registry.analyze_circuit_evolution(canonical_id)
        newly_registered_analysis[canonical_id] = evolution_data

    # Get stable circuits
    stable_circuits = canonical_detector.canonical_registry.get_stable_circuits(
        min_stability=0.6, min_detections=2
    )

    # ============================================================================
    # 6. COMPREHENSIVE RESULTS AGGREGATION
    # ============================================================================

    analysis_results = {
        # Sampling results
        'sampling_summary': {
            'strategy': 'fast_sampling',
            'examples_analyzed': len(sampled_examples),
            'total_copy_detections': len(all_copy_detections),
            'total_induction_detections': len(all_induction_detections),
            'examples_with_circuits': len([ea for ea in example_analyses
                                           if ea['copy_detections'] or ea['induction_detections']])
        },

        # Cross-example pattern analysis
        'cross_example_patterns': {
            'robust_copy_patterns': robust_copy_patterns,
            'robust_induction_patterns': robust_induction_patterns,
            'copy_pattern_diversity': len(robust_copy_patterns),
            'induction_pattern_diversity': len(robust_induction_patterns)
        },

        # Registration results
        'registration_summary': registration_summary,
        'registered_canonical_circuits': registered_canonical_circuits,
        'newly_registered_analysis': newly_registered_analysis,

        # Registry state
        'registry_summary': registry_summary,
        'stable_circuits_count': len(stable_circuits),
        'stable_circuits': [
            {
                'canonical_id': circuit.canonical_id,
                'stability_score': circuit.stability_score,
                'total_detections': circuit.total_detections,
                'operation_type': circuit.computational_signature.operation_type
            }
            for circuit in stable_circuits[:10]  # Top 10 for logging
        ],

        # Evolution insights
        'evolution_insights': analyze_canonical_evolution_insights(
            canonical_detector.canonical_registry, epoch, logger
        ),

        # Raw data for further analysis
        'example_analyses': example_analyses  # Full data for debugging
    }

    # ============================================================================
    # 7. ENHANCED LOGGING
    # ============================================================================

    log_canonical_analysis_results(analysis_results, epoch, logger)

    return analysis_results



def analyze_canonical_copy_patterns(example_analyses: List[Dict],
                                    cross_example_threshold: float = 0.3,
                                    min_examples: int = 2,
                                    logger=None) -> Dict[str, Dict]:
    """Analyze copy patterns across examples to find robust canonical patterns"""

    pattern_occurrences = defaultdict(list)

    # Collect all copy patterns with their computational signatures
    for analysis in example_analyses:
        for detection in analysis['copy_detections']:
            # Create computational signature
            head = detection.get('head', 'unknown')
            relative_offset = detection.get('target_pos', 0) - detection.get('source_pos', 0)
            strength = detection.get('attention_strength', detection.get('strength', 0.0))

            # Create canonical pattern key based on computational structure
            pattern_key = f"copy_{head}_offset_{relative_offset}"

            # Store occurrence with full context
            occurrence = {
                'example_idx': analysis['example_idx'],
                'tokens': analysis['tokens'],
                'detection': detection,
                'strength': strength,
                'source_pos': detection.get('source_pos', -1),
                'target_pos': detection.get('target_pos', -1),
                'relative_offset': relative_offset
            }

            pattern_occurrences[pattern_key].append(occurrence)

    # Find robust patterns (appear in multiple examples with sufficient strength)
    robust_patterns = {}

    for pattern_key, occurrences in pattern_occurrences.items():
        if len(occurrences) >= min_examples:
            # Calculate pattern statistics
            strengths = [occ['strength'] for occ in occurrences]
            avg_strength = np.mean(strengths)
            consistency = 1.0 - (np.std(strengths) / max(np.mean(strengths), 0.1))

            # Check if pattern meets robustness criteria
            if avg_strength >= cross_example_threshold and consistency >= 0.5:
                robust_patterns[pattern_key] = {
                    'occurrences': occurrences,
                    'example_count': len(occurrences),
                    'avg_strength': avg_strength,
                    'consistency': consistency,
                    'cross_example_coverage': len(set(occ['example_idx'] for occ in occurrences)) / len(
                        example_analyses),
                    'token_examples': [' '.join(occ['tokens']) for occ in occurrences[:5]],  # Sample tokens
                    'representative_detection': occurrences[np.argmax(strengths)]  # Strongest example
                }

                if logger:
                    logger.debug(f"    🔗 Robust copy pattern: {pattern_key} "
                                 f"({len(occurrences)} examples, strength {avg_strength:.3f})")

    return robust_patterns


def analyze_canonical_induction_patterns(example_analyses: List[Dict],
                                         cross_example_threshold: float = 0.3,
                                         min_examples: int = 2,
                                         logger=None) -> Dict[str, Dict]:
    """Analyze induction patterns across examples to find robust canonical patterns"""

    pattern_occurrences = defaultdict(list)

    # Collect all induction patterns
    for analysis in example_analyses:
        for detection in analysis['induction_detections']:
            # Create computational signature
            head = detection.get('head', 'unknown')
            inducer_pos = detection.get('inducer_pos', -1)
            target_pos = detection.get('target_pos', -1)
            distance = target_pos - inducer_pos if inducer_pos >= 0 and target_pos >= 0 else 0
            strength = detection.get('strength', 0.0)

            # Create canonical pattern key
            pattern_key = f"induction_{head}_dist_{distance}"

            # Store occurrence
            occurrence = {
                'example_idx': analysis['example_idx'],
                'tokens': analysis['tokens'],
                'detection': detection,
                'strength': strength,
                'distance': distance,
                'inducer_pos': inducer_pos,
                'target_pos': target_pos
            }

            pattern_occurrences[pattern_key].append(occurrence)

    # Find robust patterns
    robust_patterns = {}

    for pattern_key, occurrences in pattern_occurrences.items():
        if len(occurrences) >= min_examples:
            strengths = [occ['strength'] for occ in occurrences]
            avg_strength = np.mean(strengths)
            consistency = 1.0 - (np.std(strengths) / max(np.mean(strengths), 0.1))

            if avg_strength >= cross_example_threshold and consistency >= 0.5:
                robust_patterns[pattern_key] = {
                    'occurrences': occurrences,
                    'example_count': len(occurrences),
                    'avg_strength': avg_strength,
                    'consistency': consistency,
                    'cross_example_coverage': len(set(occ['example_idx'] for occ in occurrences)) / len(
                        example_analyses),
                    'token_examples': [' '.join(occ['tokens']) for occ in occurrences[:5]],
                    'representative_detection': occurrences[np.argmax(strengths)]
                }

                if logger:
                    logger.debug(f"    🔄 Robust induction pattern: {pattern_key} "
                                 f"({len(occurrences)} examples, strength {avg_strength:.3f})")

    return robust_patterns


def register_robust_canonical_circuit(canonical_detector, pattern_key: str, pattern_data: Dict,
                                      circuit_type: str, epoch: int, logger=None) -> Optional[str]:
    """Register a robust canonical circuit from cross-example pattern analysis"""

    # Get representative detection (strongest example)
    representative = pattern_data['representative_detection']
    detection = representative['detection']
    tokens = representative['tokens']

    try:
        # info create circuit using existing circuit creator warning it still includes epoch
        circuit = canonical_detector.circuit_creator.create_circuit_from_adaptive_detection(
            detection, tokens, epoch
        )

        # info enhanced metadata for robust patterns
        circuit.metadata.update({
            'robust_pattern': True,
            'pattern_key': pattern_key,
            'cross_example_strength': pattern_data['avg_strength'],
            'cross_example_consistency': pattern_data['consistency'],
            'cross_example_coverage': pattern_data['cross_example_coverage'],
            'example_count': pattern_data['example_count'],
            'detection_method': f'robust_cross_example_{circuit_type}'
        })

        # info register using canonical system
        canonical_id, legacy_id = canonical_detector.canonical_adapter.register_circuit_detection(
            circuit=circuit,
            epoch=epoch,
            tokens=tokens,
            detection_confidence=min(1.0, pattern_data['avg_strength'] * pattern_data['consistency']),
            detection_method=f"robust_{circuit_type}",
            example_metadata={
                'robust_pattern_data': pattern_data,
                'cross_example_evidence': True
            }
        )

        if logger:
            logger.debug(f"    ✅ Registered robust {circuit_type}: {canonical_id}")

        return canonical_id

    except Exception as e:
        if logger:
            logger.warning(f"    ⚠️ Failed to register {circuit_type} pattern {pattern_key}: {e}")
        return None


def analyze_canonical_evolution_insights(canonical_registry: CanonicalCircuitRegistry,
                                         epoch: int, logger=None) -> Dict[str, Any]:
    """Analyze evolution insights from canonical registry"""

    insights = {
        'emergence_patterns': {},
        'stability_trends': {},
        'competition_dynamics': {},
        'algorithm_diversity': {}
    }

    # Analyze emergence patterns
    circuits_by_operation = {}
    for circuit in canonical_registry.canonical_circuits.values():
        op_type = circuit.computational_signature.operation_type
        if op_type not in circuits_by_operation:
            circuits_by_operation[op_type] = []
        circuits_by_operation[op_type].append(circuit)

    for op_type, circuits in circuits_by_operation.items():
        if circuits:
            first_seen_epochs = [c.first_seen for c in circuits]
            insights['emergence_patterns'][op_type] = {
                'count': len(circuits),
                'avg_emergence': np.mean(first_seen_epochs),
                'earliest_emergence': min(first_seen_epochs),
                'latest_emergence': max(first_seen_epochs)
            }

    # Analyze stability trends
    stable_circuits = canonical_registry.get_stable_circuits(min_stability=0.6)
    if stable_circuits:
        stability_scores = [c.stability_score for c in stable_circuits]
        insights['stability_trends'] = {
            'stable_count': len(stable_circuits),
            'avg_stability': np.mean(stability_scores),
            'max_stability': max(stability_scores),
            'stability_distribution': np.histogram(stability_scores, bins=5)[0].tolist()
        }

    # Analyze algorithm diversity
    unique_signatures = set()
    token_diversity = []

    for circuit in canonical_registry.canonical_circuits.values():
        unique_signatures.add(circuit.computational_signature.to_string())
        token_diversity.append(len(circuit.token_examples))

    insights['algorithm_diversity'] = {
        'unique_algorithms': len(unique_signatures),
        'avg_token_diversity': np.mean(token_diversity) if token_diversity else 0,
        'max_token_diversity': max(token_diversity) if token_diversity else 0
    }

    return insights


def log_canonical_analysis_results(analysis_results: Dict, epoch: int, logger):
    """Enhanced logging for canonical analysis results"""

    sampling = analysis_results['sampling_summary']
    patterns = analysis_results['cross_example_patterns']
    registration = analysis_results['registration_summary']
    registry = analysis_results['registry_summary']

    # Main summary
    # logger.info(f"  📊 Sampling: {sampling['examples_analyzed']} examples → "
    #             f"{sampling['total_copy_detections']} copy + {sampling['total_induction_detections']} induction detections")
    #
    # logger.info(f"  🔗 Cross-example: {patterns['copy_pattern_diversity']} robust copy + "
    #             f"{patterns['induction_pattern_diversity']} robust induction patterns")

    # logger.info(f"  📝 Registration: {registration['registered_copy']} copy + "
    #             f"{registration['registered_induction']} induction circuits registered")

    # logger.info(f"  📈 Registry: {registry['total_canonical_circuits']} total circuits, "
    #             f"{registry['stable_circuits_count']} stable "
    #             f"(aggregation rate: {registry['aggregation_rate']:.2%})")

    # Stable circuits highlights
    '''
    stable_circuits = analysis_results['stable_circuits']
    if stable_circuits:
        logger.info(f"  🏆 Top 2 and least 2 stable circuits:")
        for i, circuit_info in enumerate(stable_circuits[:2], 1):
            logger.info(f"    #{i}: {circuit_info['canonical_id']} "
                        f"(stability: {circuit_info['stability_score']:.3f}, "
                        f"detections: {circuit_info['total_detections']})")
        for i, circuit_info in enumerate(stable_circuits[-1:], 1):
            logger.info(f"    #{i}: {circuit_info['canonical_id']} "
                        f"(stability: {circuit_info['stability_score']:.3f}, "
                        f"detections: {circuit_info['total_detections']})")
    '''
    # Evolution insights
    insights = analysis_results['evolution_insights']
    emergence = insights.get('emergence_patterns', {})
    if emergence:
        # logger.info(f"  🧬 Emergence order: {list(emergence.keys())}")
        pass


# ============================================================================
# INTEGRATION WITH TRAINING LOOP
# ============================================================================

def updated_training_loop_with_canonical_sampling(
        model, train_loader, eval_loader, criterion, optimizer,
        epochs=1000, analyze_interval=2, save_dir=None, logger=None,
        example_sampling_strategy="diverse_random"):
    """
    Updated training loop using canonical circuit analysis with ExampleSampler
    """

    from pathlib import Path
    from analysis.helpers.example_sampler import create_fixed_example_sampler

    if save_dir is None:
        save_dir = Path("results/canonical_sampling_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ✅ INITIALIZE: Canonical circuit system
    circuit_system = create_standard_canonical_system(model=model, save_dir=save_dir, logger=logger,
                                                      eval_loader=eval_loader)
    canonical_detector = circuit_system['canonical_detector']

    # ✅ INITIALIZE: Example sampler
    sampling_config = {
        "base_budget": 5,  # Sample 5 examples per analysis
        "max_cache_size": 100,
        "diversity_metrics": ["entropy", "repetition", "unique_tokens"]
    }
    example_sampler = create_fixed_example_sampler(eval_loader, sampling_config)

    logger.info(f"🎲 Initialized canonical analysis with ExampleSampler (strategy: {example_sampling_strategy})")

    # Training loop
    for epoch in range(epochs):

        # Standard training step
        # ... training code ...

        # ✅ CANONICAL CIRCUIT ANALYSIS WITH SAMPLING
        if epoch % analyze_interval == 0:
            logger.info(f"🔬 Canonical circuit analysis with sampling @ epoch {epoch}")

            # Evaluate model
            eval_stats = {'accuracy': 0.85}  # evaluate(model, eval_loader, criterion, device)
            current_accuracy = eval_stats['accuracy']

            # ✅ RUN CANONICAL ANALYSIS WITH SAMPLING
            canonical_results = run_canonical_circuit_analysis_with_sampling(
                canonical_detector=canonical_detector,
                example_sampler=example_sampler,
                epoch=epoch,
                total_epochs=epochs,
                accuracy=current_accuracy,
                sampling_strategy=example_sampling_strategy,
                logger=logger,
                detect_induction=True,
                analyze_interval=analyze_interval,
                cross_example_threshold=0.3,
                min_examples_for_robustness=2
            )

            # ✅ ENHANCED LOGGING
            if logger and hasattr(logger, 'log_metrics'):
                canonical_metrics = {
                    "examples_analyzed": canonical_results['sampling_summary']['examples_analyzed'],
                    "robust_copy_patterns": canonical_results['cross_example_patterns']['copy_pattern_diversity'],
                    "robust_induction_patterns": canonical_results['cross_example_patterns'][
                        'induction_pattern_diversity'],
                    "registered_circuits": len(canonical_results['registered_canonical_circuits']),
                    "stable_circuits": canonical_results['stable_circuits_count'],
                    "aggregation_rate": canonical_results['registry_summary']['aggregation_rate']
                }
                logger.log_metrics(canonical_metrics, step=epoch, category="canonical_sampling")

    return model, canonical_detector.canonical_registry


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
"""
def example_usage_in_main_training():
    " whatisExample of how to use in main training loop"

    # In your main training function, replace the simple circuit analysis with:

    # OLD (simple single example):
    # sample_input, _ = next(iter(eval_loader))
    # tokens = [str(int(x)) for x in sample_input[0].cpu().numpy()]
    # outputs = model(sample_input, store_attention=True)
    # copy_results = canonical_detector.detect_and_register_copy_mechanisms(...)

    # NEW (robust multi-example analysis):
    canonical_results = run_canonical_circuit_analysis_with_sampling(
        canonical_detector=canonical_detector,
        example_sampler=example_sampler,
        epoch=epoch,
        total_epochs=epochs,
        accuracy=current_accuracy,
        sampling_strategy="diverse_random",
        logger=logger
    )

    # Use results for enhanced analysis
    stable_circuits = canonical_results['stable_circuits']
    evolution_insights = canonical_results['evolution_insights']

    pass
"""