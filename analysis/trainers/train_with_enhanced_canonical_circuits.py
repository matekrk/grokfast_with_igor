import time
from collections import defaultdict
from pathlib import Path

import numpy as np
# import pandas as pd

# from analysis import EnhancedCircuitRegistry, CircuitThresholds, ComputationalBudget
# from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
# from analysis.canonical.sampled_circuit_analysis import run_canonical_circuit_analysis_with_sampling
from analysis.core.canonical_circuit_system import CanonicalCircuitRegistry, FunctionalCircuitSignatureExtractor
from analysis.core.json_safe_canonical_circuits import JSONSafeCanonicalCircuitRegistry, \
    JSONSafeCanonicalRegistryAdapter, create_json_safe_canonical_system
# from analysis.examples.enhanced_circuit_canonical_circuits_management import analyze_training_results
# from analysis.core.unified_logger import UnifiedLogger
# from analysis.helpers import example_sampler
from analysis.helpers.example_sampler import create_aggressive_example_sampler, create_fast_circuit_example_sampler
# from analysis.sampling.diversity_enhanced_sampling import analyze_circuit_capacity_limits
from analysis.trainers.utils import detect_grokking, train_epoch, evaluate
# from analysis.utils.utils import init_train_dataloader_state
from analysis.visualization.circuit_quality_analyzer import CircuitQualityAnalyzer


def get_sampling_configs(eval_loader, num_samples=12):
    # whatis ExampleSampler object configurations
    # info aggressive exploration whatis (early, higher computational cost)
    exploration_config = {
        "base_budget": 5,
        "max_cache_size": 100,
        "diversity_metrics": ["entropy", "repetition", "unique_tokens", "sequential"]
    }
    # info conservative sampling whatis (middle, low computational cost)
    conservative_config = {
        "base_budget": 2,
        "max_cache_size": 20,
        "diversity_metrics": ["entropy", "repetition"]
    }
    # info late-training focused whatis (late, minimal cost)
    focused_config = {
        "base_budget": 1,
        "max_cache_size": 10,
        "diversity_metrics": ["entropy"]
    }
    aggressive_config = {
        'base_budget': num_samples,  # Increased budget
        'max_recent_history': 500,
        'repetition_penalty': 0.925,
        'token_ranges': {'min': 0, 'max': 97},
        'force_rare_tokens': True,
        'rare_token_probability': 0.4
    }
    return {
        "conservative": conservative_config,
        "exploration": exploration_config,
        "focused": focused_config,
        'aggressive': aggressive_config,
    }


def extend_canonical_registry_for_new_types(registry: CanonicalCircuitRegistry):
    """Example of how to extend for new circuit types"""

    # Add new extractors
    registry.extractors.append(FunctionalCircuitSignatureExtractor())

    # Could add more extractors for:
    # - Sparse autoencoder features
    # - Superposition circuits
    # - Meta-learning circuits
    # - Compositional reasoning circuits
    # etc.

    return registry


def initialize_canonical_circuits_experimental_architecture(model, save_dir, logger,
                                                            eval_loader, num_samples=12,
                                                            enhanced_registry=None,
                                                            thresholds=None,):
    """Initialize complete canonical circuit system"""

    # info create canonical registry
    canonical_registry = JSONSafeCanonicalCircuitRegistry(save_dir / "canonical_circuits")
    # info extend for additional circuit types if needed
    from analysis.core.canonical_circuit_system import extend_canonical_registry_for_new_types
    canonical_registry = extend_canonical_registry_for_new_types(canonical_registry)

    # info create enhanced registry if not provided
    if enhanced_registry is None:
        from analysis.core import EnhancedCircuitRegistry
        enhanced_registry = EnhancedCircuitRegistry(save_dir / "enhanced_registry")

    # info create adapter
    adapter = JSONSafeCanonicalRegistryAdapter(enhanced_registry, canonical_registry)

    # info initialize evolution tracker with canonical support
    from analysis.core.unified_circuit_evolution_tracker import UnifiedCircuitEvolutionTracker
    evolution_tracker = UnifiedCircuitEvolutionTracker(enhanced_registry, save_dir / "evolution", logger)

    # example_sampler = create_aggressive_example_sampler(eval_loader, strategy_config)

    sampling_config = {
        'base_budget': num_samples,
        'max_recent_history': 500,
        'repetition_penalty': 0.925,
        'token_ranges': {'min': 0, 'max': 97},
        'force_rare_tokens': True,
        'rare_token_probability': 0.4
    }
    example_sampler = create_fast_circuit_example_sampler(eval_loader=eval_loader,
                                                          strategy_config=sampling_config)

    # info create canonical-aware adaptive detector with JSON safety included
    from analysis.integration.canonical_integration_guide import CanonicalAwareAdaptiveTokenOperationDetector
    canonical_detector = CanonicalAwareAdaptiveTokenOperationDetector(
        model=model,
        enhanced_registry=enhanced_registry,
        canonical_registry=canonical_registry,
        thresholds=thresholds,
    )

    print("✅ JSON-safe canonical circuit system initialized")

    circuit_quality_analyzer = CircuitQualityAnalyzer(model=model, canonical_registry=canonical_registry,
                                                      save_dir=save_dir)

    return {
        'canonical_registry': canonical_registry,
        'enhanced_registry': enhanced_registry,
        'adapter': adapter,
        'evolution_tracker': evolution_tracker,
        'canonical_detector': canonical_detector,
        'example_sampler': example_sampler,
        'quality_analyzer': circuit_quality_analyzer,
    }

'''
def _get_logger(experiment_name, model, save_dir,
                enable_wandb_logging=False, enable_file_logging=True,
                enable_screen_logging=True, log_level="INFO"):
    logger = UnifiedLogger(
        experiment_name=experiment_name,
        log_dir=save_dir / "logs",
        enable_wandb=enable_wandb_logging,
        enable_file=enable_file_logging,
        enable_screen=enable_screen_logging,
        log_level=log_level
    )
    return logger
'''

def _get_stats(train_stats, eval_stats, optimizer):
    return {
        "train_loss": train_stats.get('loss', 0.0),
        "train_accuracy": train_stats.get('accuracy', 0.0),
        "eval_loss": eval_stats.get('loss', 0.0),
        "eval_accuracy": eval_stats.get('accuracy', 0.0),
        "learning_rate": optimizer.param_groups[0]['lr'],
        # "weight decay": optimizer.param_groups[0]['weight_decay'],
    }


def get_default_circuit_config():
    """
    Get default configuration for circuit management
    """
    return {
        'probation_epochs': 100,  # Can't remove circuits younger than this
        'grace_period_epochs': 50,  # Extra time for borderline circuits
        'min_quality_threshold': 0.25,  # Below this = candidate for removal
        'removal_votes_required': 3,  # Need multiple bad assessments
        'max_circuits': 150,  # Maximum circuits to maintain
        'diversity_protection_ratio': 0.15,  # Protect 15% for diversity
        'assessment_interval': 10,  # Epochs between internal assessments
    }


def get_aggressive_circuit_config():
    """
    More aggressive circuit management for faster experimentation
    """
    return {
        'probation_epochs': 50,  # Shorter probation
        'grace_period_epochs': 25,  # Less grace time
        'min_quality_threshold': 0.4,  # Higher quality bar
        'removal_votes_required': 2,  # Faster removal
        'max_circuits': 100,  # Smaller population
        'diversity_protection_ratio': 0.1,  # Less diversity protection
        'assessment_interval': 5,  # More frequent assessment
    }


def get_conservative_circuit_config():
    """
    Conservative circuit management for important experiments
    """
    return {
        'probation_epochs': 200,  # Longer probation
        'grace_period_epochs': 100,  # More grace time
        'min_quality_threshold': 0.15,  # Lower quality bar
        'removal_votes_required': 5,  # Require more votes
        'max_circuits': 300,  # Larger population
        'diversity_protection_ratio': 0.25,  # Strong diversity protection
        'assessment_interval': 20,  # Less frequent assessment
    }


def perform_robust_circuit_detection(canonical_detector, eval_loader, epoch, total_epochs,
                                     accuracy, logger, sampler=None, num_samples=12):
    """
    🔬 ROBUST multi-example circuit detection using intelligent sampling

    Replaces the old single-example approach with cross-example validation
    """

    # Create intelligent example sampler
    sampling_config = {
        'base_budget': num_samples,
        'max_recent_history': 500,
        'repetition_penalty': 0.925,
        'token_ranges': {'min': 0, 'max': 97},
        'force_rare_tokens': True,
        'rare_token_probability': 0.4
    }
    example_sampler = sampler if sampler else create_aggressive_example_sampler(eval_loader, sampling_config)

    # Sample diverse examples for analysis fixme change to fast_subset_sampler
    sampled_examples = example_sampler.sample_examples(
        epoch=epoch, total_epochs=total_epochs,
        strategy="diverse_random", seed_offset=epoch % 100
    )

    # logger.debug(f"🎲 Sampled {len(sampled_examples)} examples for robust circuit detection")

    # Analyze each example WITHOUT registering circuits yet
    all_copy_detections = []
    all_induction_detections = []
    example_analyses = []

    for example_idx, (inputs, targets) in enumerate(sampled_examples):
        tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]

        # Forward pass with attention storage
        try:
            outputs = canonical_detector.model(inputs, store_attention=True)
            attention_patterns = canonical_detector.model.get_attention_patterns()
        except Exception as e:
            logger.warning(f"Forward pass failed for example {example_idx}: {e}")
            continue

        # Detect circuits without registration (to find patterns first)
        copy_results = canonical_detector._detect_copy_mechanisms_fixed(
            attention_patterns=attention_patterns,
            tokens=tokens, epoch=epoch, total_epochs=total_epochs,
            model_accuracy=accuracy, content_aware=True
        )

        induction_results = canonical_detector._detect_induction_patterns_fixed(
            attention_patterns=attention_patterns,
            tokens=tokens, epoch=epoch, total_epochs=total_epochs,
            model_accuracy=accuracy
        )

        # Store for cross-example analysis
        example_analyses.append({
            'example_idx': example_idx,
            'tokens': tokens,
            'copy_detections': copy_results,
            'induction_detections': induction_results,
            'attention_patterns': attention_patterns
        })

        all_copy_detections.extend(copy_results)
        all_induction_detections.extend(induction_results)

    # 🔗 CROSS-EXAMPLE PATTERN ANALYSIS
    robust_patterns = analyze_cross_example_patterns(
        example_analyses,
        cross_example_threshold=0.3,
        min_examples_for_robustness=max(2, len(sampled_examples) // 3),
        logger=logger
    )

    # 📝 REGISTER ONLY ROBUST CIRCUITS
    registered_circuits = []
    registration_summary = {'attempted': 0, 'succeeded': 0, 'aggregated': 0}

    # Register robust copy circuits
    for pattern_key, pattern_data in robust_patterns['robust_copy_patterns'].items():
        try:
            registration_summary['attempted'] += 1

            # Get representative example
            representative = pattern_data['representative_detection']
            circuit = canonical_detector.circuit_creator.create_circuit_from_adaptive_detection(
                representative['detection'], representative['tokens'], epoch
            )

            # Enhanced metadata for robust circuits
            circuit.metadata.update({
                'robust_pattern': True,
                'pattern_key': pattern_key,
                'cross_example_strength': pattern_data['avg_strength'],
                'cross_example_consistency': pattern_data['consistency'],
                'cross_example_coverage': pattern_data['cross_example_coverage'],
                'example_count': pattern_data['example_count'],
                'detection_method': 'robust_cross_example_copy'
            })

            # Register with canonical system
            aggregated_up_to_now = canonical_detector.canonical_registry.total_aggregations
            canonical_id, legacy_id = canonical_detector.canonical_adapter.register_circuit_detection(
                circuit=circuit, epoch=epoch, tokens=representative['tokens'],
                detection_confidence=min(1.0, pattern_data['avg_strength'] * pattern_data['consistency']),
                detection_method="robust_copy",
                example_metadata={'robust_pattern_data': pattern_data}
            )

            registered_circuits.append(canonical_id)
            registration_summary['aggregated'] += (canonical_detector.canonical_registry.total_aggregations
                                                   - aggregated_up_to_now)
            registration_summary['succeeded'] += 1

        except Exception as e:
            logger.warning(f"Failed to register robust copy pattern {pattern_key}: {e}")

    # Register robust induction circuits
    for pattern_key, pattern_data in robust_patterns['robust_induction_patterns'].items():
        try:
            registration_summary['attempted'] += 1

            representative = pattern_data['representative_detection']
            circuit = canonical_detector.circuit_creator.create_circuit_from_adaptive_detection(
                representative['detection'], representative['tokens'], epoch
            )

            circuit.metadata.update({
                'robust_pattern': True,
                'pattern_key': pattern_key,
                'cross_example_strength': pattern_data['avg_strength'],
                'cross_example_consistency': pattern_data['consistency'],
                'detection_method': 'robust_cross_example_induction'
            })

            aggregated_up_to_now = canonical_detector.canonical_registry.total_aggregations
            canonical_id, legacy_id = canonical_detector.canonical_adapter.register_circuit_detection(
                circuit=circuit, epoch=epoch, tokens=representative['tokens'],
                detection_confidence=min(1.0, pattern_data['avg_strength'] * pattern_data['consistency']),
                detection_method="robust_induction",
                example_metadata={'robust_pattern_data': pattern_data}
            )

            registered_circuits.append(canonical_id)
            registration_summary['aggregated'] += (canonical_detector.canonical_registry.total_aggregations
                                                   - aggregated_up_to_now)
            registration_summary['succeeded'] += 1

        except Exception as e:
            logger.warning(f"Failed to register robust induction pattern {pattern_key}: {e}")

    return {
        'sampling_summary': {
            'strategy': 'diverse_random',
            'examples_analyzed': len(sampled_examples),
            'total_copy_detections': len(all_copy_detections),
            'total_induction_detections': len(all_induction_detections),
        },
        'robust_patterns': robust_patterns,
        'registered_circuits': registered_circuits,
        'registration_summary': registration_summary,
        'cross_example_analysis': {
            'robust_copy_patterns': len(robust_patterns['robust_copy_patterns']),
            'robust_induction_patterns': len(robust_patterns['robust_induction_patterns']),
            'pattern_diversity': len(robust_patterns['robust_copy_patterns']) + len(
                robust_patterns['robust_induction_patterns'])
        }
    }


def analyze_cross_example_patterns(example_analyses, cross_example_threshold=0.3,
                                   min_examples_for_robustness=2, logger=None):
    """
    🔗 Find patterns that appear consistently across multiple examples
    """

    # Collect patterns by computational signature
    copy_pattern_occurrences = defaultdict(list)
    induction_pattern_occurrences = defaultdict(list)

    for analysis in example_analyses:
        # Analyze copy patterns
        for detection in analysis['copy_detections']:
            head = detection.get('head', 'unknown')
            relative_offset = detection.get('target_pos', 0) - detection.get('source_pos', 0)
            strength = detection.get('attention_strength', detection.get('strength', 0.0))

            pattern_key = f"copy_{head}_offset_{relative_offset}"
            occurrence = {
                'example_idx': analysis['example_idx'],
                'tokens': analysis['tokens'],
                'detection': detection,
                'strength': strength,
                'relative_offset': relative_offset
            }
            copy_pattern_occurrences[pattern_key].append(occurrence)

        # Analyze induction patterns
        for detection in analysis['induction_detections']:
            head = detection.get('head', 'unknown')
            distance = detection.get('target_pos', -1) - detection.get('inducer_pos', -1)
            strength = detection.get('strength', 0.0)

            pattern_key = f"induction_{head}_dist_{distance}"
            occurrence = {
                'example_idx': analysis['example_idx'],
                'tokens': analysis['tokens'],
                'detection': detection,
                'strength': strength,
                'distance': distance
            }
            induction_pattern_occurrences[pattern_key].append(occurrence)

    # Find robust patterns (appear in multiple examples with sufficient strength)
    def find_robust_patterns(pattern_occurrences, pattern_type):
        robust = {}
        for pattern_key, occurrences in pattern_occurrences.items():
            if len(occurrences) >= min_examples_for_robustness:
                strengths = [occ['strength'] for occ in occurrences]
                avg_strength = np.mean(strengths)
                consistency = 1.0 - (np.std(strengths) / max(np.mean(strengths), 0.1))

                if avg_strength >= cross_example_threshold and consistency >= 0.5:
                    robust[pattern_key] = {
                        'occurrences': occurrences,
                        'example_count': len(occurrences),
                        'avg_strength': avg_strength,
                        'consistency': consistency,
                        'cross_example_coverage': len(set(occ['example_idx'] for occ in occurrences)) / len(
                            example_analyses),
                        'representative_detection': occurrences[np.argmax(strengths)]
                    }

                    if logger:
                        logger.debug(f"🔗 Robust {pattern_type}: {pattern_key} "
                                     f"({len(occurrences)} examples, strength {avg_strength:.3f})")

        return robust

    robust_copy = find_robust_patterns(copy_pattern_occurrences, "copy")
    robust_induction = find_robust_patterns(induction_pattern_occurrences, "induction")

    return {
        'robust_copy_patterns': robust_copy,
        'robust_induction_patterns': robust_induction,
        'pattern_diversity': {
            'copy_patterns_total': len(copy_pattern_occurrences),
            'induction_patterns_total': len(induction_pattern_occurrences),
            'copy_patterns_robust': len(robust_copy),
            'induction_patterns_robust': len(robust_induction)
        }
    }


def perform_circuit_detection(canonical_detector, eval_loader, epoch, total_epochs,
                              accuracy, logger):
    """
    Perform circuit detection using existing logic
    """
    # Get sample for analysis
    sample_input, sample_target = next(iter(eval_loader))
    tokens = [str(int(x)) for x in sample_input[0].cpu().numpy()]

    # Forward pass with attention storage
    outputs = canonical_detector.model(sample_input, store_attention=True)
    attention_patterns = canonical_detector.model.get_attention_patterns()

    # Detect circuits
    copy_results = canonical_detector.detect_and_register_copy_mechanisms(
        attention_patterns=attention_patterns,
        tokens=tokens,
        epoch=epoch,
        total_epochs=total_epochs,
        model_accuracy=accuracy,
        register_circuits=True
    )

    induction_results = canonical_detector.detect_and_register_induction_patterns(
        attention_patterns=attention_patterns,
        tokens=tokens,
        epoch=epoch,
        total_epochs=total_epochs,
        model_accuracy=accuracy,
        register_circuits=True
    )

    return {
        'copy_results': copy_results,
        'induction_results': induction_results
    }




def train_with_enhanced_circuit_management(
        model, train_loader, eval_loader,
        criterion, optimizer,
        scheduler=None, device='cuda', checkpointManager=None,
        epochs=10000, log_interval=4,
        # Enhanced circuit management parameters
        circuit_assessment_interval=32,
        circuit_management_config=None,
        enable_circuit_removal=True,
        enable_diversity_protection=True,
        enable_real_testing=True,
        enable_robust_detection=True,  # 🆕 NEW: Enable robust multi-example detection
        robust_detection_samples=12,  # 🆕 NEW: Number of samples for robust detection
        # Logging parameters
        enable_wandb_logging=False,
        enable_file_logging=True,
        enable_screen_logging=True,
        log_level="INFO"
):
    """
    🔬 ENHANCED training with FIXED bugs and robust multi-example circuit detection

    Key improvements:
    1. ✅ Fixed eval_loader=None bug in DataFrame logging
    2. 🔬 Replaced single-example detection with robust multi-example approach
    3. 🔗 Cross-example pattern validation for circuit robustness
    4. 📊 Enhanced circuit quality metrics
    """

    # Setup (same as before)
    if checkpointManager:
        save_dir = Path(checkpointManager.experiment_dir)
    else:
        save_dir = Path("results/enhanced_circuit_management_fixed")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging
    from analysis.core.unified_logger import UnifiedLogger
    experiment_name = f"enhanced_circuits_fixed_{model.get_id() if hasattr(model, 'get_id') else 'transformer'}"
    logger = UnifiedLogger(
        experiment_name=experiment_name,
        log_dir=save_dir / "logs",
        enable_wandb=enable_wandb_logging,
        enable_file=enable_file_logging,
        enable_screen=enable_screen_logging,
        log_level=log_level
    )

    # Initialize JSON-safe canonical circuit system
    circuit_system = create_json_safe_canonical_system(model, save_dir, logger)
    canonical_registry = circuit_system['canonical_registry']
    canonical_detector = circuit_system['canonical_detector']

    # Initialize enhanced circuit quality system
    circuit_config = circuit_management_config or get_default_circuit_config()

    from analysis.core.enhanced_circuit_quality_system import EnhancedCircuitQualitySystem
    quality_system = EnhancedCircuitQualitySystem(
        model=model,
        canonical_registry=canonical_registry,
        save_dir=save_dir / "circuit_quality",
        enable_real_testing=enable_real_testing
    )
    quality_system.config.update(circuit_config)

    logger.info(f"🔧 Enhanced circuit management (FIXED VERSION) initialized:")
    logger.info(f"   Assessment interval: {circuit_assessment_interval} epochs")
    logger.info(f"   Max circuits: {quality_system.config['max_circuits']}")
    logger.info(f"   Removal enabled: {enable_circuit_removal}")
    logger.info(f"   🔬 Robust detection: {enable_robust_detection} ({robust_detection_samples} samples)")
    logger.info(f"   🧪 Real functional testing: {enable_real_testing}")

    # Training metrics tracking
    training_metrics = {'epoch': [], 'loss': [], 'accuracy': []}
    circuit_management_history = []

    # Enhanced DataFrame logging for circuit analysis
    # from analysis.trainers.train_with_enhanced_canonical_circuits import initialize_circuit_dataframe
    df_circuit = initialize_circuit_dataframe()
    circuit_snapshots = []
    eval_stats = None
    access_circuits_threshold = 0.225

    num_samples = 12
    sampling_config = {
        'base_budget': num_samples,
        'max_recent_history': 500,
        'repetition_penalty': 0.925,
        'token_ranges': {'min': 0, 'max': 97},
        'force_rare_tokens': True,
        'rare_token_probability': 0.4
    }
    example_sampler = create_fast_circuit_example_sampler(eval_loader=eval_loader,
                                                          strategy_config=sampling_config)

    # Main training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Standard training step
        train_stats = train_epoch(model, train_loader, criterion, optimizer, epoch, device)

        if scheduler:
            scheduler.step()

        # Evaluation
        pre_current_accuracy = eval_stats['accuracy'] if eval_stats else 0.0
        should_evaluate = epoch % log_interval == 0 or epoch == epochs - 1
        should_assess_circuits = (epoch % circuit_assessment_interval == 0 and epoch > 50 and
                                  pre_current_accuracy > access_circuits_threshold)

        if should_evaluate or should_assess_circuits:
            eval_stats = evaluate(model, eval_loader, criterion, device)
            current_accuracy = eval_stats['accuracy']
            should_assess_circuits = (epoch % circuit_assessment_interval == 0 and epoch > 50 and
                                      current_accuracy > access_circuits_threshold)
            # Update training metrics history
            training_metrics['epoch'].append(epoch)
            training_metrics['loss'].append(eval_stats['loss'])
            training_metrics['accuracy'].append(current_accuracy)

            # Log basic training metrics
            if should_evaluate:
                training_metrics_log = {
                    "train_loss": train_stats.get('loss', 0.0),
                    "train_accuracy": train_stats.get('accuracy', 0.0),
                    "eval_loss": eval_stats['loss'],
                    "eval_accuracy": eval_stats['accuracy'],
                    "learning_rate": optimizer.param_groups[0]['lr'],
                }
                if epoch % (2 * log_interval) == 0 or should_assess_circuits:
                    logger.log_metrics(training_metrics_log, step=epoch, category="training")

                # info detect grokking  todo fixme Is it needed here?
                detect_grokking(model, epoch, train_stats, eval_stats)

        # 🔬 ENHANCED CIRCUIT DETECTION AND MANAGEMENT
        if should_assess_circuits:
            # logger.info(f"🔬 Enhanced circuit management @ epoch {epoch}")

            # 🆕 Step 1: ROBUST MULTI-EXAMPLE CIRCUIT DETECTION
            if enable_robust_detection:
                robust_detection_results = perform_robust_circuit_detection(
                    canonical_detector, eval_loader, epoch, epochs,
                    current_accuracy, logger, sampler=example_sampler,
                    num_samples=robust_detection_samples
                )

                # Log robust detection metrics
                robust_metrics = {
                    "examples_analyzed": robust_detection_results['sampling_summary']['examples_analyzed'],
                    "robust_copy_patterns": robust_detection_results['cross_example_analysis']['robust_copy_patterns'],
                    "robust_induction_patterns": robust_detection_results['cross_example_analysis'][
                        'robust_induction_patterns'],
                    "registered_robust_circuits": len(robust_detection_results['registered_circuits']),
                    "pattern_diversity": robust_detection_results['cross_example_analysis']['pattern_diversity']
                }
                logger.log_metrics(robust_metrics, step=epoch, category="robust_detection")
                registered = robust_metrics['registered_robust_circuits']
                aggregated = robust_detection_results["registration_summary"]["aggregated"]
                total_circuits = len(canonical_registry.canonical_circuits)
                logger.info(f"   🔗 Found {robust_metrics['robust_copy_patterns']} copy + "
                            f"{robust_metrics['robust_induction_patterns']} induction robust patterns"
                            f" |  📝 Registered {registered} |  📝 Aggregated {aggregated} |  📝 New {registered - aggregated} |  📝 Total {total_circuits} circuits")

            # Step 2: COMPREHENSIVE QUALITY ASSESSMENT AND MANAGEMENT
            if enable_circuit_removal:
                management_results = quality_system.assess_and_manage_circuits(
                    eval_loader, epoch
                )
                circuit_management_history.append(management_results)

                # Log circuit management metrics
                circuit_metrics = {
                    "total_circuits": management_results['circuit_management']['total_circuits'],
                    "circuits_removed": management_results['circuit_management']['circuits_removed'],
                    "circuits_protected": management_results['circuit_management']['circuits_protected'],
                    "mean_quality": management_results['quality_statistics']['mean_quality'],
                    "established_circuits": management_results['lifecycle_distribution'].get('established', 0),
                    "declining_circuits": management_results['lifecycle_distribution'].get('declining', 0),
                    "diversity_score": management_results['diversity_analysis']['diversity_score'],
                }
                logger.log_metrics(circuit_metrics, step=epoch, category="circuit_management")

                # Enhanced logging
                logger.info(f"   📊 Circuit population: {circuit_metrics['total_circuits']} total"
                            f" | 📈 Quality: μ={circuit_metrics['mean_quality']:.3f}, "
                            f"diversity={circuit_metrics['diversity_score']:.3f}")
                removed_protected = ""
                if circuit_metrics['circuits_removed'] > 0:
                    removed_protected = removed_protected + f"   🗑️  Removed: {circuit_metrics['circuits_removed']} circuits"
                if circuit_metrics['circuits_protected'] > 0:
                    removed_protected = removed_protected + f"   🛡️  Protected: {circuit_metrics['circuits_protected']} circuits for diversity"
                if len(removed_protected) > 0:
                    logger.info(removed_protected)

                # 📊 FIXED DATAFRAME LOGGING - now passes eval_loader
                df_circuit, circuit_snapshot = log_circuits_to_dataframe(
                    quality_system, management_results, epoch, df_circuit, eval_loader
                )
                circuit_snapshots.append(circuit_snapshot)

        # Periodic saves (same as before)
        if epoch % 500 == 0 and epoch > 0:
            try:
                canonical_registry.save_with_json_encoder(save_dir / f"canonical_registry_epoch_{epoch}.json")

                import json
                with open(save_dir / f"circuit_management_history_epoch_{epoch}.json", 'w') as f:
                    json.dump(circuit_management_history, f, indent=2, default=str)

                if not df_circuit.empty:
                    df_circuit.to_csv(save_dir / f"circuit_dataframe_epoch_{epoch}.csv", index=False)
                    # df_circuit.to_pickle(save_dir / f"circuit_dataframe_epoch_{epoch}.pkl")

                logger.info(f"💾 Saved circuit data @ epoch {epoch}")
            except Exception as e:
                logger.warning(f"Failed to save circuit data: {e}")
        if epoch % 400 == 0 and epoch > 0:
            # info perform df_circuit analysis
            analyze_training_results(
                results={
                    'model': model,
                    'eval_loader': eval_loader,
                    'canonical_registry': canonical_registry,
                    'quality_system': quality_system,
                    'final_stats': canonical_registry.get_registry_summary(),
                    'circuit_management_history': circuit_management_history,
                    'training_metrics': training_metrics,
                    'circuit_dataframe': df_circuit,
                    'circuit_snapshots': circuit_snapshots,
                })

    # Final analysis (same as before, but with robust detection insights)
    logger.info("🎉 Training completed. Generating final circuit analysis...")

    final_management_results = quality_system.assess_and_manage_circuits(eval_loader, epochs)
    final_registry_stats = canonical_registry.get_registry_summary()

    logger.info(f"📊 Final circuit statistics:")
    logger.info(f"   Total circuits discovered: {final_registry_stats['total_canonical_circuits']}")
    logger.info(f"   Final circuit population: {final_management_results['circuit_management']['total_circuits']}")
    logger.info(f"   Aggregation efficiency: {final_registry_stats['aggregation_rate']:.2%}")
    logger.info(f"   Mean circuit quality: {final_management_results['quality_statistics']['mean_quality']:.3f}")

    # 📊 FINAL DATAFRAME SAVE
    try:
        if not df_circuit.empty:
            final_df_path = save_dir / "final_circuit_dataframe.csv"
            df_circuit.to_csv(final_df_path, index=False)
            # df_circuit.to_pickle(save_dir / "final_circuit_dataframe.pkl")

            logger.info(f"💾 Final DataFrame saved: {len(df_circuit)} circuit records")
            logger.info(f"   Columns: {list(df_circuit.columns)}")
            logger.info(f"   Epochs covered: {df_circuit['epoch'].min()}-{df_circuit['epoch'].max()}")

    except Exception as e:
        logger.warning(f"Failed to save final DataFrame: {e}")

    return {
        'model': model,
        'canonical_registry': canonical_registry,
        'quality_system': quality_system,
        'final_stats': final_registry_stats,
        'circuit_management_history': circuit_management_history,
        'training_metrics': training_metrics,
        'circuit_dataframe': df_circuit,
        'circuit_snapshots': circuit_snapshots,
    }


# ============================================================================
# 📊 ENHANCED DATAFRAME LOGGING FUNCTIONS
# ============================================================================

def initialize_circuit_dataframe():
    """
    Initialize DataFrame with comprehensive circuit tracking columns
    """
    import pandas as pd

    return pd.DataFrame({
        # Basic identification
        'epoch': pd.Series(dtype='int'),
        'circuit_id': pd.Series(dtype='str'),
        'operation_type': pd.Series(dtype='str'),

        # Lifecycle information
        'lifecycle_state': pd.Series(dtype='str'),
        'circuit_age': pd.Series(dtype='int'),
        'first_seen': pd.Series(dtype='int'),
        'last_seen': pd.Series(dtype='int'),
        'total_detections': pd.Series(dtype='int'),

        # Quality metrics (comprehensive)
        'overall_quality': pd.Series(dtype='float'),
        'temporal_quality': pd.Series(dtype='float'),
        'functional_quality': pd.Series(dtype='float'),
        'attention_quality': pd.Series(dtype='float'),
        'stability_score': pd.Series(dtype='float'),

        # Management status
        'recommendation': pd.Series(dtype='str'),
        'removal_votes': pd.Series(dtype='int'),
        'is_protected': pd.Series(dtype='bool'),
        'protection_reason': pd.Series(dtype='str'),

        # Trend analysis
        'quality_trend': pd.Series(dtype='float'),  # Recent trend in quality
        'strengthening': pd.Series(dtype='bool'),  # Is circuit getting stronger?

        # Diversity and uniqueness
        'operation_rarity': pd.Series(dtype='float'),  # How rare is this operation type?
        'uniqueness_score': pd.Series(dtype='float'),  # How unique is this circuit?

        # Behavioral metrics
        'attention_entropy': pd.Series(dtype='float'),
        'pattern_consistency': pd.Series(dtype='float'),
        'cross_example_robustness': pd.Series(dtype='float'),

        # Performance impact
        'attribution_strength': pd.Series(dtype='float'),
        'behavioral_impact': pd.Series(dtype='float'),

        # Meta information
        'assessment_count': pd.Series(dtype='int'),
        'last_assessment_epoch': pd.Series(dtype='int'),
    })


def log_circuits_to_dataframe(quality_system, management_results, epoch,
                                    df_circuit, eval_loader):
    """
    🔧 FIXED: Pass eval_loader instead of None
    """
    import pandas as pd
    import numpy as np

    # ✅ FIXED: Pass eval_loader instead of None
    quality_results = quality_system.quality_analyzer.analyze_all_circuits(
        eval_loader, max_circuits=1000  # ✅ Now passes the actual eval_loader
    )

    # Get diversity analysis
    diversity_analysis = quality_system.diversity_tracker.analyze_diversity(
        quality_system.canonical_registry.canonical_circuits
    )

    # Prepare batch data (same as before)
    batch_dict = {col: [] for col in df_circuit.columns}

    # Process each circuit
    for circuit_id, analysis in quality_results['circuit_analyses'].items():
        # Basic identification
        batch_dict['epoch'].append(epoch)
        batch_dict['circuit_id'].append(circuit_id)
        batch_dict['operation_type'].append(analysis['operation_type'])

        # Get lifecycle info
        lifecycle_info = quality_system.circuit_lifecycles.get(circuit_id)
        canonical_circuit = quality_system.canonical_registry.canonical_circuits.get(circuit_id)

        if lifecycle_info:
            batch_dict['lifecycle_state'].append(lifecycle_info.state.value)
            batch_dict['circuit_age'].append(epoch - lifecycle_info.first_detected)
            batch_dict['removal_votes'].append(lifecycle_info.removal_votes)
            batch_dict['is_protected'].append(lifecycle_info.state.value == 'protected')
            batch_dict['protection_reason'].append(lifecycle_info.protection_reason or '')
            batch_dict['assessment_count'].append(lifecycle_info.assessment_count)
            batch_dict['last_assessment_epoch'].append(lifecycle_info.last_assessment_epoch)

            # Calculate quality trend
            if len(lifecycle_info.quality_history) >= 3:
                recent = np.mean(lifecycle_info.quality_history[-3:])
                earlier = np.mean(lifecycle_info.quality_history[:3])
                quality_trend = recent - earlier
            else:
                quality_trend = 0.0
            batch_dict['quality_trend'].append(quality_trend)
        else:
            # Default values for missing lifecycle info
            batch_dict['lifecycle_state'].append('unknown')
            batch_dict['circuit_age'].append(0)
            batch_dict['removal_votes'].append(0)
            batch_dict['is_protected'].append(False)
            batch_dict['protection_reason'].append('')
            batch_dict['assessment_count'].append(0)
            batch_dict['last_assessment_epoch'].append(epoch)
            batch_dict['quality_trend'].append(0.0)

        if canonical_circuit:
            batch_dict['first_seen'].append(canonical_circuit.first_seen)
            batch_dict['last_seen'].append(canonical_circuit.last_seen)
            batch_dict['total_detections'].append(canonical_circuit.total_detections)
            batch_dict['attribution_strength'].append(canonical_circuit.best_attribution)
        else:
            batch_dict['first_seen'].append(epoch)
            batch_dict['last_seen'].append(epoch)
            batch_dict['total_detections'].append(1)
            batch_dict['attribution_strength'].append(0.0)

        # Quality metrics (comprehensive breakdown)
        batch_dict['overall_quality'].append(analysis['quality_score'])
        batch_dict['stability_score'].append(analysis['stability_score'])

        # Extract component scores safely
        temporal_analysis = analysis.get('temporal_analysis', {})
        functional_analysis = analysis.get('functional_analysis', {})
        attention_analysis = analysis.get('attention_analysis', {})

        batch_dict['temporal_quality'].append(temporal_analysis.get('temporal_consistency', 0.0))
        batch_dict['functional_quality'].append(functional_analysis.get('functional_quality', 0.0))
        batch_dict['attention_quality'].append(attention_analysis.get('attention_quality', 0.0))

        # Behavioral metrics
        batch_dict['attention_entropy'].append(attention_analysis.get('avg_attention_entropy', 0.0))
        batch_dict['pattern_consistency'].append(attention_analysis.get('pattern_consistency', 0.0))
        batch_dict['behavioral_impact'].append(functional_analysis.get('avg_consistency', 0.0))
        batch_dict['cross_example_robustness'].append(functional_analysis.get('behavior_variance', 1.0))

        # Management status
        batch_dict['recommendation'].append(analysis.get('recommendation', 'UNKNOWN').split(' ')[0])

        # Calculate operation rarity and uniqueness
        op_type = analysis['operation_type']
        total_circuits = len(quality_results['circuit_analyses'])
        same_type_count = len(diversity_analysis['by_operation_type'].get(op_type, []))
        operation_rarity = 1.0 - (same_type_count / total_circuits) if total_circuits > 0 else 0.0
        batch_dict['operation_rarity'].append(operation_rarity)

        # Uniqueness score (combination of rarity and quality)
        uniqueness_score = operation_rarity * analysis['quality_score']
        batch_dict['uniqueness_score'].append(uniqueness_score)

        # Strengthening trend
        temporal_trend = temporal_analysis.get('strengthening_trend', False)
        batch_dict['strengthening'].append(temporal_trend)

    # Add batch to DataFrame
    if batch_dict['epoch']:  # Only if we have data
        new_batch = pd.DataFrame(batch_dict)
        df_circuit = pd.concat([df_circuit, new_batch], ignore_index=True)

    # Create summary snapshot for this epoch
    snapshot = {
        'epoch': epoch,
        'total_circuits': len(batch_dict['epoch']),
        'mean_quality': np.mean(batch_dict['overall_quality']) if batch_dict['overall_quality'] else 0.0,
        'circuits_removed': management_results['circuit_management']['circuits_removed'],
        'circuits_protected': management_results['circuit_management']['circuits_protected'],
        'diversity_score': management_results['diversity_analysis']['diversity_score'],
        'lifecycle_distribution': management_results['lifecycle_distribution'],
    }

    return df_circuit, snapshot


def log_circuits_to_dataframe_assessment_only(quality_system, quality_results, epoch, df_circuit):
    """
    Simplified DataFrame logging when circuit removal is disabled
    """
    import pandas as pd
    import numpy as np

    # Simplified logging without lifecycle management
    batch_dict = {col: [] for col in df_circuit.columns}

    for circuit_id, analysis in quality_results['circuit_analyses'].items():
        # Basic info
        batch_dict['epoch'].append(epoch)
        batch_dict['circuit_id'].append(circuit_id)
        batch_dict['operation_type'].append(analysis['operation_type'])

        # Quality info
        batch_dict['overall_quality'].append(analysis['quality_score'])
        batch_dict['stability_score'].append(analysis['stability_score'])
        batch_dict['recommendation'].append(analysis.get('recommendation', 'UNKNOWN').split(' ')[0])

        # Registry info
        canonical_circuit = quality_system.canonical_registry.canonical_circuits.get(circuit_id)
        if canonical_circuit:
            batch_dict['first_seen'].append(canonical_circuit.first_seen)
            batch_dict['last_seen'].append(canonical_circuit.last_seen)
            batch_dict['total_detections'].append(canonical_circuit.total_detections)
            batch_dict['circuit_age'].append(epoch - canonical_circuit.first_seen)
        else:
            # Defaults
            batch_dict['first_seen'].append(epoch)
            batch_dict['last_seen'].append(epoch)
            batch_dict['total_detections'].append(1)
            batch_dict['circuit_age'].append(0)

        # Fill other columns with defaults
        for col in df_circuit.columns:
            if col not in batch_dict or len(batch_dict[col]) < len(batch_dict['epoch']):
                if df_circuit[col].dtype == 'float64':
                    batch_dict.setdefault(col, []).append(0.0)
                elif df_circuit[col].dtype == 'int64':
                    batch_dict.setdefault(col, []).append(0)
                elif df_circuit[col].dtype == 'bool':
                    batch_dict.setdefault(col, []).append(False)
                else:
                    batch_dict.setdefault(col, []).append('')

    # Add to DataFrame
    if batch_dict['epoch']:
        new_batch = pd.DataFrame(batch_dict)
        df_circuit = pd.concat([df_circuit, new_batch], ignore_index=True)

    # Simple snapshot
    snapshot = {
        'epoch': epoch,
        'total_circuits': len(batch_dict['epoch']),
        'mean_quality': np.mean(batch_dict['overall_quality']) if batch_dict['overall_quality'] else 0.0,
        'assessment_only': True,
    }

    return snapshot


#################
## info test some analysis
##################
def analyze_training_results(results):
    """
    Analyze the results of enhanced circuit training
    """
    model = results['model']
    eval_loader = results['eval_loader']
    canonical_registry = results['canonical_registry']
    quality_system = results['quality_system']
    circuit_history = results['circuit_management_history']
    df_circuit = results['circuit_dataframe']  # 📊 NEW: Access to comprehensive DataFrame
    circuit_snapshots = results['circuit_snapshots']  # 📊 NEW: Detailed snapshots

    print("\n🔍 Training Results Analysis:")
    print("=" * 50)

    # 1. Final circuit population
    final_stats = results['final_stats']
    print(f"📊 Circuit Population:")
    print(f"   Total circuits discovered: {final_stats['total_canonical_circuits']}")
    print(f"   Aggregation rate: {final_stats['aggregation_rate']:.2%}")
    print(f"   Stable circuits: {final_stats['stable_circuits_count']}")

    # 📊 2. ENHANCED DATAFRAME ANALYSIS
    if not df_circuit.empty:
        print(f"\n📈 DataFrame Analysis ({len(df_circuit)} records):")
        print(f"   Epochs covered: {df_circuit['epoch'].min()}-{df_circuit['epoch'].max()}")
        print(f"   Unique circuits tracked: {df_circuit['circuit_id'].nunique()}")

        # Quality evolution analysis
        quality_evolution = df_circuit.groupby('epoch')['overall_quality'].agg(['mean', 'std', 'count'])
        print(
            f"   Quality evolution: {quality_evolution.iloc[-1]['mean']:.3f} ± {quality_evolution.iloc[-1]['std']:.3f}")

        # Lifecycle distribution
        if 'lifecycle_state' in df_circuit.columns:
            lifecycle_dist = df_circuit['lifecycle_state'].value_counts()
            print(f"   Final lifecycle distribution: {dict(lifecycle_dist)}")

        # Operation type diversity
        op_type_dist = df_circuit['operation_type'].value_counts()
        print(f"   Operation types: {dict(op_type_dist)}")

        # Circuit longevity analysis
        circuit_lifespans = df_circuit.groupby('circuit_id')['circuit_age'].max()
        print(f"   Average circuit lifespan: {circuit_lifespans.mean():.1f} epochs")
        print(f"   Longest surviving circuit: {circuit_lifespans.max()} epochs")

        # Quality trend analysis
        if 'quality_trend' in df_circuit.columns:
            improving_circuits = (df_circuit['quality_trend'] > 0.1).sum()
            declining_circuits = (df_circuit['quality_trend'] < -0.1).sum()
            print(f"   Circuits improving: {improving_circuits}, declining: {declining_circuits}")

    # 3. Circuit lifecycle analysis
    if circuit_history:
        removal_count = sum(len(h['removed_circuits']) for h in circuit_history)
        protection_count = sum(len(h['protected_circuits']) for h in circuit_history)

        print(f"\n🔄 Circuit Lifecycle:")
        print(f"   Total removals: {removal_count}")
        print(f"   Total protections: {protection_count}")

        # Analyze removal reasons
        removal_reasons = {}
        for history in circuit_history:
            for removal in history['removed_circuits']:
                reason = removal['reason']
                removal_reasons[reason] = removal_reasons.get(reason, 0) + 1

        print(f"   Removal reasons: {dict(removal_reasons)}")

    # 4. Quality distribution analysis
    quality_assessment = quality_system.quality_analyzer.analyze_all_circuits(
        eval_loader=eval_loader,  # You'd pass your eval_loader here
        max_circuits=1000
    )

    quality_stats = quality_assessment['quality_report']['quality_statistics']
    print(f"\n📈 Quality Distribution:")
    print(f"   Mean quality: {quality_stats['mean_quality']:.3f}")
    print(f"   Quality std: {quality_stats['std_quality']:.3f}")

    # 5. Diversity analysis
    diversity = quality_system.diversity_tracker.get_diversity_summary(
        canonical_registry.canonical_circuits
    )
    print(f"\n🌈 Circuit Diversity:")
    print(f"   Operation types: {diversity['operation_type_count']}")
    print(f"   Diversity score: {diversity['diversity_score']:.3f}")
    print(f"   Distribution: {diversity['layer_distribution']}")

    # 6. Circuit removal history analysis
    removal_history = quality_system.get_removal_history()
    if removal_history:
        print(f"\n🗑️  Removal History:")

        # Group by operation type
        removed_by_type = {}
        for removal in removal_history:
            op_type = removal['operation_type']
            removed_by_type[op_type] = removed_by_type.get(op_type, 0) + 1

        print(f"   Removed by type: {dict(removed_by_type)}")

        # Average age at removal
        avg_age = sum(r['age'] for r in removal_history) / len(removal_history)
        print(f"   Average age at removal: {avg_age:.1f} epochs")

    # 📊 7. DATAFRAME-SPECIFIC INSIGHTS
    analyze_dataframe_insights(df_circuit)


def analyze_dataframe_insights(df_circuit):
    """
    Advanced analysis using the comprehensive DataFrame
    """
    if df_circuit.empty:
        return

    import pandas as pd
    import numpy as np

    print(f"\n🔬 Advanced DataFrame Insights:")
    print("=" * 35)

    # Circuit survival analysis
    if 'lifecycle_state' in df_circuit.columns:
        # Find circuits that were removed
        final_epoch = df_circuit['epoch'].max()
        final_circuits = df_circuit[df_circuit['epoch'] == final_epoch]['circuit_id'].unique()
        all_circuits = df_circuit['circuit_id'].unique()
        removed_circuits = set(all_circuits) - set(final_circuits)

        survival_rate = len(final_circuits) / len(all_circuits) if len(all_circuits) > 0 else 0
        print(f"   Circuit survival rate: {survival_rate:.2%} ({len(final_circuits)}/{len(all_circuits)})")

    # Quality predictors analysis
    if 'overall_quality' in df_circuit.columns and 'circuit_age' in df_circuit.columns:
        # Correlation between age and quality
        age_quality_corr = df_circuit['circuit_age'].corr(df_circuit['overall_quality'])
        print(f"   Age-Quality correlation: {age_quality_corr:.3f}")

        # Quality by operation type
        quality_by_type = df_circuit.groupby('operation_type')['overall_quality'].mean().sort_values(ascending=False)
        print(f"   Best operation types: {dict(quality_by_type.head(3))}")

    # Protection effectiveness
    if 'is_protected' in df_circuit.columns:
        protected_circuits = df_circuit[df_circuit['is_protected'] == True]['circuit_id'].nunique()
        total_unique_circuits = df_circuit['circuit_id'].nunique()
        protection_rate = protected_circuits / total_unique_circuits if total_unique_circuits > 0 else 0
        print(f"   Circuits ever protected: {protection_rate:.2%}")

    # Quality evolution patterns
    if len(df_circuit) > 100:  # Only if we have enough data
        # Find circuits that improved significantly
        circuit_quality_trends = df_circuit.groupby('circuit_id').agg({
            'overall_quality': ['first', 'last', 'count'],
            'epoch': ['first', 'last']
        }).round(3)

        # Flatten column names
        circuit_quality_trends.columns = ['_'.join(col).strip() for col in circuit_quality_trends.columns.values]

        # Calculate improvement
        circuit_quality_trends['quality_improvement'] = (
                circuit_quality_trends['overall_quality_last'] - circuit_quality_trends['overall_quality_first']
        )

        # Find most improved circuits
        most_improved = circuit_quality_trends.nlargest(3, 'quality_improvement')
        print(f"   Most improved circuits (Δ quality):")
        for circuit_id, row in most_improved.iterrows():
            print(f"     {circuit_id[:30]}...: +{row['quality_improvement']:.3f}")

    # Diversity trends over time
    if 'operation_rarity' in df_circuit.columns:
        final_diversity = df_circuit[df_circuit['epoch'] == df_circuit['epoch'].max()]
        rare_circuits = (final_diversity['operation_rarity'] > 0.7).sum()
        common_circuits = (final_diversity['operation_rarity'] < 0.3).sum()
        print(f"   Final diversity: {rare_circuits} rare, {common_circuits} common circuits")
