# analysis/trainers/train_with_enhanced_circuit_analysis.py (UPDATED with UnifiedLogger)
from collections import defaultdict

# import torch
import time
from pathlib import Path
from typing import Dict, List, Any  #, Optional

from analysis.analyzers.circuit_evolution_tracker import CircuitEvolutionTracker
# from analysis import AdaptiveTokenOperationDetector
# Enhanced infrastructure imports
from analysis.core import EnhancedCircuitRegistry, CircuitThresholds, ComputationalBudget
from analysis.core.unified_logger import UnifiedLogger  # ✅ ADD: UnifiedLogger import
from analysis.helpers.analysis_summary import create_final_analysis_summary
from analysis.helpers.example_sampler import ExampleSampler, \
    create_fixed_example_sampler  # , create_example_sampler, create_fixed_example_sampler
from analysis.helpers.validation_helpers import save_enhanced_checkpoint
from analysis.helpers.missing_functions import run_circuit_validation
from analysis.helpers.circuit_analysis import analyze_circuit_emergence, analyze_circuit_relationships
from analysis.utils.utils import init_train_dataloader_state
from analysis.validation import CircuitManipulationValidator

# Existing imports (unchanged)
# from analysis.analyzers.integrated_token_discovery import IntegratedTokenCircuitDiscovery
from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
from analysis.analyzers.continuous_circuit_tracker import ContinuousCircuitTracker
from analysis.trainers.utils import evaluate, train_epoch, detect_grokking  #, log_metrics


def train_with_enhanced_circuit_analysis(
        model, train_loader, eval_loader,
        # dataset_split_indices,
        criterion, optimizer,
        scheduler=None, device='cuda', checkpointManager=None,
        epochs=10000, log_interval=4, analyze_interval=2,
        circuit_sampling_freq=20, checkpoint_interval=200,
        # New parameters for Week 2
        enable_adaptive_detection=True,
        enable_circuit_validation=False,
        validation_interval=100,
        computational_budget_per_epoch=30.0,
        adaptive_threshold_config=None,
        # ✅ NEW: UnifiedLogger parameters
        enable_wandb_logging=True,
        enable_file_logging=True,
        enable_screen_logging=True,
        log_level="INFO",
        example_sampling_strategy = "diverse_random",
        example_sampling_config = None
):
    """Enhanced training with adaptive circuit detection, temporal tracking, and unified logging"""

    # Initialize dataloader states
    train_dataloader_state = init_train_dataloader_state(dataloader=train_loader)
    eval_dataloader_state = init_train_dataloader_state(dataloader=eval_loader)

    # Setup directories
    if checkpointManager:
        save_dir = Path(checkpointManager.experiment_dir)
    else:
        save_dir = Path("results/enhanced_circuit_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # ✅ UNIFIED LOGGER INITIALIZATION
    # ============================================================================

    experiment_name = f"circuit_analysis_{model.get_id() if hasattr(model, 'get_id') else 'transformer'}"

    logger = UnifiedLogger(
        experiment_name=experiment_name,
        log_dir=save_dir / "logs",
        enable_wandb=enable_wandb_logging,
        enable_file=enable_file_logging,
        enable_screen=enable_screen_logging,
        log_level=log_level
    )

    logger.info(f"🔬 Enhanced Circuit Analysis Experiment: {experiment_name}")
    logger.info(f"📊 Training for {epochs} epochs with adaptive detection: {enable_adaptive_detection}")

    # ============================================================================
    # ENHANCED INFRASTRUCTURE INITIALIZATION (Week 2)
    # ============================================================================

    # Enhanced registry with temporal tracking
    registry = EnhancedCircuitRegistry(save_dir / "enhanced_registry")

    # Adaptive threshold system
    if adaptive_threshold_config:
        thresholds = CircuitThresholds(**adaptive_threshold_config)
    else:
        thresholds = CircuitThresholds(
            copy_attention_min=0.3, copy_attention_max=0.9,
            induction_attention_min=0.4, induction_attention_max=0.85,
            warmup_epochs=50, min_accuracy_threshold=0.2
        )

    # Computational budget manager
    budget = ComputationalBudget(max_time_per_epoch=computational_budget_per_epoch)

    # Circuit validation (optional)
    validator = None
    if enable_circuit_validation:
        validator = CircuitManipulationValidator(model, eval_loader, batch_limit=2)
        logger.info("🧪 Circuit validation enabled")

    # ✅ LOG: Initial configuration
    config_metrics = {
        "total_epochs": epochs,
        "analyze_interval": analyze_interval,
        "adaptive_detection": enable_adaptive_detection,
        "validation_enabled": enable_circuit_validation,
        "budget_per_epoch": computational_budget_per_epoch
    }
    logger.log_metrics(config_metrics, step=0, category="configuration")

    # ============================================================================
    # ANALYZER INITIALIZATION (Enhanced + Existing)
    # ============================================================================

    # Existing analyzers (keep unchanged for now)
    shared_logger = model.logger if hasattr(model, 'logger') else None

    weight_tracker = EnhancedWeightSpaceTracker(
        model=model, save_dir=save_dir / "weight_tracking",
        logger=shared_logger, registry=registry,
        jump_detection_window=100, snapshot_freq=analyze_interval // 2
    )

    circuit_tracker = ContinuousCircuitTracker(
        model=model, save_dir=save_dir / "circuit_tracking",
        logger=shared_logger, registry=registry,
        sampling_freq=circuit_sampling_freq
    )

    circuit_evolution_tracker = CircuitEvolutionTracker(
        registry=registry, save_dir=save_dir / "circuit_evolution_tracking",
        logger=shared_logger
    )
    """
    token_discovery = IntegratedTokenCircuitDiscovery(
        model=model, save_dir=save_dir / "token_circuits",
        logger=shared_logger, circuit_registry=registry,
        circuit_tracker=circuit_tracker, weight_tracker=weight_tracker
    )
    """

    # NEW: Adaptive token detector
    adaptive_detector = None
    if enable_adaptive_detection:
        from analysis.analyzers.fixed_adaptive_token_operations import RegistrationAwareAdaptiveTokenOperationDetector # FixedAdaptiveTokenOperationDetector
        adaptive_detector = RegistrationAwareAdaptiveTokenOperationDetector(
            model=model, registry=registry, thresholds=thresholds
        )
        logger.info("🔍 Adaptive token detection enabled")

    logger.info("📋 All analyzers initialized successfully")

    # Configure example sampling
    # info conservative sampling whatis (middle, low computational cost)
    conservative_config = {
        "base_budget": 2,
        "max_cache_size": 20,
        "diversity_metrics": ["entropy", "repetition"]
    }
    # info aggressive exploration whatis (early, higher computational cost)
    exploration_config = {
        "base_budget": 5,
        "max_cache_size": 100,
        "diversity_metrics": ["entropy", "repetition", "unique_tokens", "sequential"]
    }
    # info late-training focused whatis (late, minimal cost)
    focused_config = {
        "base_budget": 1,
        "max_cache_size": 10,
        "diversity_metrics": ["entropy"]
    }
    # sampling_config = example_sampling_config or {
    #     "base_budget": 3,
    #     "max_cache_size": 50,
    #     "diversity_metrics": ["entropy", "repetition", "unique_tokens", "sequential"]
    # }
    sampling_config = exploration_config
    # example_sampler = create_example_sampler(eval_loader, sampling_config)
    example_sampler = create_fixed_example_sampler(eval_loader, sampling_config)
    logger.info(f"🎲 Example sampler initialized with strategy: {example_sampling_strategy}")

    # ============================================================================
    # TRAINING LOOP WITH ENHANCED ANALYSIS AND UNIFIED LOGGING
    # ============================================================================

    analysis_results = {}
    training_start_time = time.time()

    # Initial setup
    weight_tracker.take_snapshot(epoch=0, force=True)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        budget.start_epoch()

        # 1. Standard training step
        train_stats = train_epoch(model, train_loader, criterion, optimizer, epoch, device)

        if scheduler:
            scheduler.step()

        # 2. Evaluation
        should_evaluate = epoch % log_interval == 0 or epoch == epochs - 1
        eval_stats = None
        current_accuracy = 0.0

        if should_evaluate and eval_loader:
            eval_stats = evaluate(model, eval_loader, criterion, device)
            current_accuracy = eval_stats['accuracy']

            # ✅ LOG: Training metrics to unified logger
            training_metrics = {
                "train_loss": train_stats.get('loss', 0.0),
                "train_accuracy": train_stats.get('accuracy', 0.0),
                "eval_loss": eval_stats.get('loss', 0.0),
                "eval_accuracy": current_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr'] ,
                "weight decay" : optimizer.param_groups[0]['weight_decay'],
            }
            logger.log_metrics(training_metrics, step=epoch, category="training")

            # log_metrics(model, epoch, train_stats, eval_stats)
            detect_grokking(model, epoch, train_stats, eval_stats)

        # 3. Weight tracking
        weight_tracker.take_snapshot(epoch=epoch)

        # 4. ENHANCED CIRCUIT ANALYSIS WITH LOGGING
        should_analyze = epoch % analyze_interval == 0 or epoch == epochs - 1
        should_detect = thresholds.should_start_detection(epoch, current_accuracy)

        if should_analyze and eval_loader and should_detect:
            logger.info(f"📊 Enhanced circuit analysis @ epoch {epoch}")

            if eval_stats is None:
                eval_stats = evaluate(model, eval_loader, criterion, device)
                current_accuracy = eval_stats['accuracy']

            analysis_start_time = time.time()
            epoch_results = {}

            # =================================================================
            # ADAPTIVE TOKEN DETECTION (New in Week 2) WITH LOGGING
            # =================================================================

            if (enable_adaptive_detection and adaptive_detector and
                    budget.can_run_method("adaptive_token_detection")):
                token_start = time.time()
                logger.debug("🔍 Running adaptive token detection with intelligent sampling...")
                # logger.debug("🔍 Running adaptive token detection...")

                # info Get sample batch for analysis
                # sample_batch = next(iter(eval_loader))
                # inputs, targets = sample_batch
                # info and run adaptive detection for one example from that batch
                # adaptive_results = run_adaptive_token_analysis(
                #     adaptive_detector, inputs, targets, epoch, epochs, current_accuracy
                # )
                # info end of single example from single batch

                sampling_strategy = "diverse_random"
                # whatis ✅ NEW: Use intelligent example sampling instead of fixed example
                # adaptive_results = run_adaptive_token_analysis_enhanced(
                #     adaptive_detector, example_sampler, epoch, epochs, current_accuracy,
                #     sampling_strategy=sampling_strategy,
                #     logger=logger
                # )
                adaptive_results = run_complete_adaptive_token_analysis_with_registration(
                    detector=adaptive_detector, example_sampler=example_sampler,
                    epoch=epoch, total_epochs=epochs,
                    accuracy=current_accuracy,
                    sampling_strategy=sampling_strategy,
                    logger=logger,
                    detect_induction=True,
                    analyze_interval=analyze_interval,
                    register_robust_only=True
                )
                # whatis end of intelligent sampling

                epoch_results["adaptive_token_results"] = adaptive_results

                token_time = time.time() - token_start
                budget.record_execution_time("adaptive_token_detection", token_time)

                # ✅ LOG: Adaptive detection results
                adaptive_metrics = {
                    "stable_circuits": len(adaptive_results.get('stable_circuits', [])),
                    "total_detected": adaptive_results.get('detection_summary', {}).get('total_detected', 0),
                    "stability_rate": adaptive_results.get('detection_summary', {}).get('stability_rate', 0.0),
                    "detection_time": token_time
                }
                logger.log_metrics(adaptive_metrics, step=epoch, category="adaptive_detection")

                logger.info(f"    ✅ Found {adaptive_metrics['total_detected']} circuits "
                    f" | {adaptive_metrics['stable_circuits']} stable (rate: {adaptive_metrics['stability_rate']:.2%})")
                if adaptive_metrics["stability_rate"] > 0.01:
                    pass

            # =================================================================
            # EXISTING ANALYSIS (Enhanced with new registry) WITH LOGGING
            # =================================================================
            """
            # info this is no longer needed - we now have the adaptive methods
            # info the token_discover == ntegratedTokenCircuit~~~Discovery actually 
            # info discovers huge amounts of circits, registering all of them, and
            # info actually polluting the registry
            # Existing token discovery (now uses enhanced registry)
            if budget.can_run_method("token_discovery"):
                existing_start = time.time()

                existing_results = token_discovery.analyze_epoch(
                    epoch=epoch, eval_loader=eval_loader, baseline_acc=current_accuracy
                )

                epoch_results["existing_token_results"] = existing_results

                existing_time = time.time() - existing_start
                budget.record_execution_time("token_discovery", existing_time)

                # ✅ LOG: Existing token analysis results
                existing_metrics = {
                    "token_circuits": len(existing_results.get('circuits', [])),
                    "copy_mechanisms": len(existing_results.get('copy_mechanisms', [])),
                    "induction_patterns": len(existing_results.get('induction_patterns', [])),
                    "analysis_time": existing_time
                }
                logger.log_metrics(existing_metrics, step=epoch, category="token_discovery")
            """
            # Existing component analysis
            if budget.can_run_method("component_analysis"):
                """
                component_start = time.time()

                component_results = circuit_tracker.sample_circuits(
                    epoch=epoch, eval_loader=eval_loader, baseline_acc=current_accuracy
                )
                component_time = time.time() - component_start
                # ✅ LOG: Component analysis results
                component_metrics = {
                    "component_circuits": len(component_results.get('circuits', [])) if component_results else 0,
                    "component_analysis_time": component_time
                }
                logger.log_metrics(component_metrics, step=epoch, category="component_analysis")
                """

                component_start_enhanced = time.time()
                component_results_enhanced = circuit_tracker.sample_circuits_enhanced(
                    epoch=epoch, eval_loader=eval_loader, baseline_acc=current_accuracy,
                    adaptive_thresholds=thresholds,
                    current_accuracy=eval_stats['accuracy'],
                    logger=logger
                )
                component_time_enhanced = time.time() - component_start_enhanced
                component_metrics_enhanced = {
                    "total_interactions": len(component_results_enhanced.get('component_interactions', [])),
                    "stable_interactions": len(component_results_enhanced.get('stable_interactions', [])),
                    "component_circuits": len(component_results_enhanced.get('circuits', [])),
                    "stability_rate": component_results_enhanced.get('stability_rate', 0.0),
                    "component_analysis_time": component_time_enhanced
                }
                epoch_results["component_results"] = component_results_enhanced
                logger.log_metrics(component_metrics_enhanced, step=epoch,
                                   category="component_analysis_enhanced")
                budget.record_execution_time("component_analysis_enhanced", component_time_enhanced)


            # =================================================================
            # CIRCUIT VALIDATION (Optional) WITH LOGGING
            # =================================================================

            should_validate = (enable_circuit_validation and validator and
                               epoch % validation_interval == 0 and
                               budget.can_run_method("validation"))

            if should_validate:
                validation_start = time.time()
                logger.info("🧪 Running circuit validation...")

                validation_results = run_circuit_validation(registry, validator, epoch, eval_stats['accuracy'])
                epoch_results["validation_results"] = validation_results

                validation_time = time.time() - validation_start
                budget.record_execution_time("validation", validation_time)

                # ✅ LOG: Validation results
                logger.log_validation_results(validation_results, epoch)

            # =================================================================
            # ENHANCED REGISTRY ANALYSIS WITH LOGGING
            # =================================================================

            # Circuit emergence analysis
            if len(registry.emergence_timeline) > 1:
                emergence_analysis = analyze_circuit_emergence(circuit_evolution_tracker, epoch)
                epoch_results["emergence_analysis"] = emergence_analysis

                # ✅ LOG: Emergence analysis
                emergence_metrics = {
                    "total_emerged": len(registry.emergence_timeline.get(epoch, [])),
                    "emergence_rate": len(registry.emergence_timeline.get(epoch, [])) / max(1, epoch)
                }
                logger.log_metrics(emergence_metrics, step=epoch, category="circuit_emergence")

            # Relationship discovery
            if len(registry.circuits) > 1:
                relationship_analysis = analyze_circuit_relationships(registry, epoch)
                epoch_results["relationship_analysis"] = relationship_analysis

                # ✅ LOG: Circuit evolution
                circuit_evolution_info = {
                    "total_circuits": len(registry.circuits),
                    "stable_circuits": len(registry.get_circuits_by_stability("stable")),
                    "emerging_circuits": len(registry.get_circuits_by_stability("emerging")),
                    "total_relationships": sum(len(rels) for rels in registry.relationship_graph.values())
                }
                logger.log_circuit_evolution(circuit_evolution_info, epoch)

            analysis_results[epoch] = epoch_results
            total_analysis_time = time.time() - analysis_start_time

            # ✅ LOG: Analysis performance metrics
            analysis_performance = {
                "total_analysis_time": total_analysis_time,
                "budget_utilization": budget.get_usage_summary()["budget_utilization"]
            }
            logger.log_metrics(analysis_performance, step=epoch, category="analysis_performance")

            logger.info(f"    ⏱️  Total analysis time: {total_analysis_time:.2f}s")

            # Budget summary
            usage = budget.get_usage_summary()
            if usage["budget_utilization"] > 0.8:
                logger.warning(f"⚠️  High budget usage: {usage['budget_utilization']:.1%}")

        # 5. Enhanced registry logging
        if epoch % 50 == 0:
            summary = registry.get_registry_summary()

            # ✅ LOG: Registry summary
            registry_metrics = {
                "total_circuits": summary['total_circuits'],
                "total_relationships": summary['total_relationships'],
                "detection_methods": len(summary['detection_methods'])
            }
            logger.log_metrics(registry_metrics, step=epoch, category="registry_status")

            logger.info(f"📈 Registry @ epoch {epoch}: {summary['total_circuits']} circuits, "
                        f"{summary['total_relationships']} relationships")

        # 6. Checkpointing (enhanced)
        if checkpointManager and (epoch % checkpoint_interval == 0 or epoch == epochs - 1):
            save_enhanced_checkpoint(checkpointManager=checkpointManager,
                                     epoch=epoch,
                                     train_state=train_dataloader_state,
                                     eval_state=eval_dataloader_state,
                                     split_indices=None,
                                     train_stats=train_stats,
                                     eval_stats=eval_stats,
                                     registry=registry,
                                     weight_tracker=weight_tracker,
                                     )

        # Save registry periodically
        if epoch % 50 == 0:
            registry.save()

        # ✅ LOG: Epoch timing
        # info epoch timing temporarily switched off
        # epoch_time = time.time() - epoch_start_time
        # logger.log_metrics({"epoch_time": epoch_time}, step=epoch, category="timing")

    # ============================================================================
    # ✅ FINAL ANALYSIS AND LOGGING
    # ============================================================================

    logger.info("🔍 Creating final analysis summary...")

    final_summary = create_final_analysis_summary(registry, analysis_results, epochs)

    # ✅ LOG: Final summary metrics
    summary_metrics = {
        "total_circuits_final": final_summary['metadata']['total_circuits'],
        "analysis_coverage": final_summary['metadata']['analysis_coverage'],
        "stable_circuit_ratio": final_summary['stability_analysis']['stable_circuit_ratio'],
        "total_training_time": time.time() - training_start_time
    }
    logger.log_metrics(summary_metrics, step=epochs, category="final_summary")

    # ✅ LOG: Research insights
    insights = final_summary['research_insights']
    logger.info(f"🔬 Research Insights Generated: {len(insights['key_insights'])} insights")
    for i, insight in enumerate(insights['key_insights'], 1):
        logger.info(f"  {i}. {insight}")

    logger.info(f"📋 Recommendations: {len(insights['research_recommendations'])} recommendations")
    for i, rec in enumerate(insights['research_recommendations'], 1):
        logger.info(f"  {i}. {rec}")

    # Save final summary
    final_summary_path = save_dir / "final_analysis_summary.json"
    with open(final_summary_path, 'w') as f:
        import json
        json.dump(final_summary, f, indent=2, default=str)

    logger.info(f"💾 Final analysis saved to: {final_summary_path}")

    # ✅ FINALIZE: Close unified logger
    logger.finalize_experiment()

    print("🎉 Enhanced circuit analysis training complete with unified logging!")
    return model, analysis_results, registry, final_summary




def analyze_cross_example_patterns(example_results, logger=None):
    """
    Enhanced cross-example analysis for both copy and induction patterns

    Args:
        example_results: List of example analysis results, each containing:
                        - example_idx: Index of the example
                        - tokens: List of token strings
                        - copy_mechanisms: List of copy mechanism dictionaries
                        - induction_patterns: List of induction pattern dictionaries
        logger: Optional logger for detailed analysis logging

    Returns:
        dict: Cross-example pattern analysis with robust and consistent patterns
    """

    # Initialize collections for both pattern types
    copy_pattern_counts = defaultdict(list)
    induction_pattern_counts = defaultdict(list)

    # Process each example's results
    for result in example_results:
        example_idx = result["example_idx"]

        # Process copy mechanisms
        for mechanism in result["copy_mechanisms"]:
            head = mechanism.get("head", "unknown")
            src_pos = mechanism.get("source_pos", -1)
            tgt_pos = mechanism.get("target_pos", -1)
            relative_offset = tgt_pos - src_pos if src_pos >= 0 and tgt_pos >= 0 else 0

            # Create pattern signature based on head and relative position
            pattern_key = f"{head}_copy_offset_{relative_offset}"
            copy_pattern_counts[pattern_key].append({
                "example_idx": example_idx,
                "mechanism": mechanism
            })

        # Process induction patterns
        for pattern in result["induction_patterns"]:
            head = pattern.get("head", "unknown")
            inducer_pos = pattern.get("inducer_pos", -1)
            target_pos = pattern.get("target_pos", -1)
            distance = target_pos - inducer_pos if inducer_pos >= 0 and target_pos >= 0 else 0

            # Create pattern signature based on head and distance
            pattern_key = f"{head}_induction_dist_{distance}"
            induction_pattern_counts[pattern_key].append({
                "example_idx": example_idx,
                "pattern": pattern
            })

    # Helper function to find robust patterns
    def find_robust_patterns(pattern_counts, min_examples=2):
        """
        Find patterns that appear in multiple examples

        Args:
            pattern_counts: Dictionary of pattern_key -> list of occurrences
            min_examples: Minimum number of examples pattern must appear in

        Returns:
            tuple: (robust_patterns, consistent_patterns)
        """
        robust = {}
        consistent = {}

        for pattern_key, occurrences in pattern_counts.items():
            occurrence_count = len(occurrences)
            example_coverage = len(set(occ["example_idx"] for occ in occurrences))
            consistency_ratio = example_coverage / len(example_results)

            if example_coverage >= min_examples:
                robust[pattern_key] = {
                    "occurrence_count": occurrence_count,
                    "example_coverage": example_coverage,
                    "consistency_ratio": consistency_ratio,
                    "occurrences": occurrences
                }

                # Highly consistent patterns (appear in 70%+ of examples)
                if consistency_ratio >= 0.7:
                    consistent[pattern_key] = robust[pattern_key]

        return robust, consistent

    # Find robust patterns for both types
    robust_copy, consistent_copy = find_robust_patterns(copy_pattern_counts)
    robust_induction, consistent_induction = find_robust_patterns(induction_pattern_counts)

    # Log findings if logger available
    if logger and hasattr(logger, 'debug'):
        logger.debug(f"  📊 Copy patterns: {len(copy_pattern_counts)} unique, {len(robust_copy)} robust")
        logger.debug(f"  📊 Induction patterns: {len(induction_pattern_counts)} unique, {len(robust_induction)} robust")

        # Log top robust patterns
        for pattern_name, pattern_data in list(robust_copy.items())[:3]:  # Top 3 copy
            consistency = pattern_data["consistency_ratio"]
            logger.debug(f"    🔗 Robust copy: {pattern_name} ({consistency:.1%} consistency)")

        for pattern_name, pattern_data in list(robust_induction.items())[:3]:  # Top 3 induction
            consistency = pattern_data["consistency_ratio"]
            logger.debug(f"    🔄 Robust induction: {pattern_name} ({consistency:.1%} consistency)")

    # Calculate additional statistics
    total_examples = len(example_results)
    total_copy_occurrences = sum(len(occurrences) for occurrences in copy_pattern_counts.values())
    total_induction_occurrences = sum(len(occurrences) for occurrences in induction_pattern_counts.values())

    # Return comprehensive analysis
    return {
        # Robust patterns (appear in 2+ examples)
        "robust_copy_patterns": robust_copy,
        "robust_induction_patterns": robust_induction,

        # Consistent patterns (appear in 70%+ examples)
        "consistent_copy_patterns": consistent_copy,
        "consistent_induction_patterns": consistent_induction,

        # Diversity metrics
        "copy_pattern_diversity": len(copy_pattern_counts),
        "induction_pattern_diversity": len(induction_pattern_counts),

        # Coverage metrics
        "cross_example_coverage": {
            "copy": {pattern: data["consistency_ratio"] for pattern, data in robust_copy.items()},
            "induction": {pattern: data["consistency_ratio"] for pattern, data in robust_induction.items()}
        },

        # Summary statistics
        "summary_stats": {
            "total_examples": total_examples,
            "total_copy_occurrences": total_copy_occurrences,
            "total_induction_occurrences": total_induction_occurrences,
            "avg_copy_per_example": total_copy_occurrences / max(1, total_examples),
            "avg_induction_per_example": total_induction_occurrences / max(1, total_examples),
            "robust_copy_ratio": len(robust_copy) / max(1, len(copy_pattern_counts)),
            "robust_induction_ratio": len(robust_induction) / max(1, len(induction_pattern_counts))
        }
    }



# ============================================================================
# 🔧 SOLUTION 2: Enhanced run_complete_adaptive_token_analysis with Registration
# ============================================================================


def run_complete_adaptive_token_analysis_with_registration(
        detector, example_sampler, epoch, total_epochs, accuracy,
        sampling_strategy="diverse_random", logger=None, detect_induction=True,
        analyze_interval=2, register_robust_only=True):
    """
    Complete analysis with proper circuit registration strategy - FIXED VERSION

    Args:
        detector: FixedRegistrationAwareAdaptiveDetector instance
        example_sampler: ExampleSampler instance for intelligent sampling
        epoch: Current epoch
        total_epochs: Total training epochs
        accuracy: Current model accuracy
        sampling_strategy: Sampling strategy to use
        logger: Optional logger for comprehensive logging
        detect_induction: Whether to detect induction patterns
        analyze_interval: How often analysis is performed
        register_robust_only: If True, only register robust cross-example circuits

    Returns:
        dict: Enhanced analysis results with registered circuits
    """

    # Logger setup
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    # Get diverse examples
    sampled_examples = example_sampler.sample_examples(
        epoch=epoch, total_epochs=total_epochs, strategy=sampling_strategy, seed_offset=42
    )

    logger.info(f"🎲 Sampled {len(sampled_examples)} examples using '{sampling_strategy}' strategy @ epoch {epoch}")

    # Analyze each example WITHOUT registering circuits yet
    all_copy_results = []
    all_induction_results = []
    example_results = []

    for example_idx, (inputs, targets) in enumerate(sampled_examples):
        logger.debug(f"  🔍 Analyzing example {example_idx + 1}/{len(sampled_examples)}")

        tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]

        # Forward pass
        outputs = detector.model(inputs, store_attention=True)
        attention_patterns = detector.model.get_attention_patterns()

        # ✅ DETECT WITHOUT REGISTERING (register_circuits=False)
        copy_results = detector.detect_and_register_copy_mechanisms(
            attention_patterns=attention_patterns,
            tokens=tokens,
            epoch=epoch,
            total_epochs=total_epochs,
            model_accuracy=accuracy,
            content_aware=True,
            register_circuits=False  # ✅ Don't register individual examples yet
        )

        induction_results = None
        if detect_induction:
            induction_results = detector.detect_and_register_induction_patterns(
                attention_patterns=attention_patterns,
                tokens=tokens,
                epoch=epoch,
                total_epochs=total_epochs,
                model_accuracy=accuracy,
                register_circuits=False  # ✅ Don't register individual examples yet
            )

        # Store results for cross-example analysis
        all_copy_results.append(copy_results)
        all_induction_results.append(induction_results)

        example_results.append({
            "example_idx": example_idx,
            "tokens": tokens,
            "copy_mechanisms": copy_results["raw_mechanisms"],
            "induction_patterns": induction_results["raw_patterns"] if induction_results else [],
        })

    # ✅ CROSS-EXAMPLE ANALYSIS: Find robust patterns
    logger.info("🔗 Finding robust patterns across examples...")
    cross_example_patterns = analyze_cross_example_patterns(example_results, logger=logger)

    # ✅ REGISTRATION STRATEGY: Register robust circuits with high confidence
    registered_circuits = []

    if register_robust_only:
        logger.info("📝 Registering only robust cross-example circuits...")

        # Register robust copy patterns
        robust_copy = cross_example_patterns.get("robust_copy_patterns", {})
        for pattern_key, pattern_data in robust_copy.items():
            try:
                # Create representative circuit from robust pattern
                representative_occurrence = pattern_data["occurrences"][0]
                mechanism = representative_occurrence["mechanism"]

                # Get tokens from first example that showed this pattern
                example_idx = representative_occurrence["example_idx"]
                tokens = example_results[example_idx]["tokens"]

                # ✅ FIX: Use correct method name from ModernCircuitCreator
                circuit = detector.circuit_creator.create_circuit_from_adaptive_detection(
                    mechanism, tokens, epoch
                )

                # Enhanced metadata for robust patterns
                circuit.metadata.update({
                    "detection_method": "robust_cross_example_copy",
                    "detection_confidence": 0.8,  # High confidence for robust patterns
                    "cross_example_consistency": pattern_data["consistency_ratio"],
                    "examples_found": pattern_data["example_coverage"],
                    "occurrence_count": pattern_data["occurrence_count"],
                    "pattern_type": "robust_copy"
                })

                # Register with high confidence
                detector.registry.register_circuit_enhanced(
                    circuit=circuit,
                    source="robust_cross_example",
                    epoch=epoch,
                    detection_method="robust_copy",
                    confidence=0.8,
                    total_epochs=total_epochs
                )

                registered_circuits.append(circuit)
                logger.debug(f"  ✅ Registered robust copy: {pattern_key}")

            except Exception as e:
                logger.warning(f"  ⚠️ Failed to register robust copy {pattern_key}: {e}")
                import traceback
                traceback.print_exc()

        # Register robust induction patterns
        robust_induction = cross_example_patterns.get("robust_induction_patterns", {})
        for pattern_key, pattern_data in robust_induction.items():
            try:
                representative_occurrence = pattern_data["occurrences"][0]
                pattern = representative_occurrence["pattern"]

                example_idx = representative_occurrence["example_idx"]
                tokens = example_results[example_idx]["tokens"]

                # ✅ FIX: Use correct method name from ModernCircuitCreator
                circuit = detector.circuit_creator.create_circuit_from_adaptive_detection(
                    pattern, tokens, epoch
                )

                circuit.metadata.update({
                    "detection_method": "robust_cross_example_induction",
                    "detection_confidence": 0.8,
                    "cross_example_consistency": pattern_data["consistency_ratio"],
                    "examples_found": pattern_data["example_coverage"],
                    "pattern_type": "robust_induction"
                })

                detector.registry.register_circuit_enhanced(
                    circuit=circuit,
                    source="robust_cross_example",
                    epoch=epoch,
                    detection_method="robust_induction",
                    confidence=0.8,
                    total_epochs=total_epochs
                )

                registered_circuits.append(circuit)
                logger.debug(f"  ✅ Registered robust induction: {pattern_key}")

            except Exception as e:
                logger.warning(f"  ⚠️ Failed to register robust induction {pattern_key}: {e}")
                import traceback
                traceback.print_exc()

    else:
        # Alternative strategy: Register all stable circuits
        logger.info("📝 Registering all stable circuits...")

        for copy_result in all_copy_results:
            for circuit in copy_result["created_circuits"]:
                registered_circuits.append(circuit)

        for induction_result in all_induction_results:
            if induction_result:
                for circuit in induction_result["created_circuits"]:
                    registered_circuits.append(circuit)

    # Aggregate results
    aggregated_results = {
        "copy_mechanisms": [m for er in example_results for m in er["copy_mechanisms"]],
        "induction_patterns": [p for er in example_results for p in er["induction_patterns"]],
        "registered_circuits": registered_circuits,
        "detection_summary": {
            "examples_analyzed": len(sampled_examples),
            "registered_circuits": len(registered_circuits),
            "robust_copy_patterns": len(cross_example_patterns.get("robust_copy_patterns", {})),
            "robust_induction_patterns": len(cross_example_patterns.get("robust_induction_patterns", {})),
            "total_detected": len([m for er in example_results for m in er["copy_mechanisms"]]) +
                              len([p for er in example_results for p in er["induction_patterns"]])
        },
        "cross_example_analysis": cross_example_patterns,
        "example_results": example_results
    }

    logger.info(f"  ✅ Registered {len(registered_circuits)} circuits to registry")

    return aggregated_results


###############################
#### previous obsolete versions
###############################
def run_complete_adaptive_token_analysis_with_registration_previous(
        detector, example_sampler, epoch, total_epochs, accuracy,
        sampling_strategy="diverse_random", logger=None, detect_induction=True,
        analyze_interval=2, register_robust_only=True):
    """
    Complete analysis with proper circuit registration strategy

    Args:
        register_robust_only: If True, only register robust cross-example circuits
                             If False, register all stable circuits
    """

    # Logger setup
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    # Get diverse examples
    sampled_examples = example_sampler.sample_examples(
        epoch=epoch, total_epochs=total_epochs, strategy=sampling_strategy, seed_offset=42
    )

    logger.info(f"🎲 Sampled {len(sampled_examples)} examples using '{sampling_strategy}' strategy @ epoch {epoch}")

    # Analyze each example WITHOUT registering circuits yet
    all_copy_results = []
    all_induction_results = []
    example_results = []

    for example_idx, (inputs, targets) in enumerate(sampled_examples):
        logger.debug(f"  🔍 Analyzing example {example_idx + 1}/{len(sampled_examples)}")

        tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]

        # Forward pass
        outputs = detector.model(inputs, store_attention=True)
        attention_patterns = detector.model.get_attention_patterns()

        # ✅ DETECT WITHOUT REGISTERING (register_circuits=False)
        copy_results = detector.detect_and_register_copy_mechanisms(
            attention_patterns=attention_patterns,
            tokens=tokens,
            epoch=epoch,
            total_epochs=total_epochs,
            model_accuracy=accuracy,
            content_aware=True,
            register_circuits=False  # ✅ Don't register individual examples yet
        )

        induction_results = None
        if detect_induction:
            induction_results = detector.detect_and_register_induction_patterns(
                attention_patterns=attention_patterns,
                tokens=tokens,
                epoch=epoch,
                total_epochs=total_epochs,
                model_accuracy=accuracy,
                register_circuits=False  # ✅ Don't register individual examples yet
            )

        # Store results for cross-example analysis
        all_copy_results.append(copy_results)
        all_induction_results.append(induction_results)

        example_results.append({
            "example_idx": example_idx,
            "tokens": tokens,
            "copy_mechanisms": copy_results["raw_mechanisms"],
            "induction_patterns": induction_results["raw_patterns"] if induction_results else [],
        })

    # ✅ CROSS-EXAMPLE ANALYSIS: Find robust patterns
    logger.info("🔗 Finding robust patterns across examples...")
    cross_example_patterns = analyze_cross_example_patterns(example_results, logger=logger)

    # ✅ REGISTRATION STRATEGY: Register robust circuits with high confidence
    registered_circuits = []

    if register_robust_only:
        logger.info("📝 Registering only robust cross-example circuits...")

        # Register robust copy patterns
        robust_copy = cross_example_patterns.get("robust_copy_patterns", {})
        for pattern_key, pattern_data in robust_copy.items():
            try:
                # Create representative circuit from robust pattern
                representative_occurrence = pattern_data["occurrences"][0]
                mechanism = representative_occurrence["mechanism"]

                # Get tokens from first example that showed this pattern
                example_idx = representative_occurrence["example_idx"]
                tokens = example_results[example_idx]["tokens"]

                # Create circuit
                circuit = detector.circuit_creator.create_token_operation_circuit(
                    mechanism, tokens, epoch
                )

                # Enhanced metadata for robust patterns
                circuit.metadata.update({
                    "detection_method": "robust_cross_example_copy",
                    "detection_confidence": 0.8,  # High confidence for robust patterns
                    "cross_example_consistency": pattern_data["consistency_ratio"],
                    "examples_found": pattern_data["example_coverage"],
                    "occurrence_count": pattern_data["occurrence_count"],
                    "pattern_type": "robust_copy"
                })

                # Register with high confidence
                detector.registry.register_circuit_enhanced(
                    circuit=circuit,
                    source="robust_cross_example",
                    epoch=epoch,
                    detection_method="robust_copy",
                    confidence=0.8,
                    total_epochs=total_epochs
                )

                registered_circuits.append(circuit)
                logger.debug(f"  ✅ Registered robust copy: {pattern_key}")

            except Exception as e:
                logger.warning(f"  ⚠️ Failed to register robust copy {pattern_key}: {e}")

        # Register robust induction patterns
        robust_induction = cross_example_patterns.get("robust_induction_patterns", {})
        for pattern_key, pattern_data in robust_induction.items():
            try:
                representative_occurrence = pattern_data["occurrences"][0]
                pattern = representative_occurrence["pattern"]

                example_idx = representative_occurrence["example_idx"]
                tokens = example_results[example_idx]["tokens"]

                circuit = detector.circuit_creator.create_token_operation_circuit(
                    pattern, tokens, epoch
                )

                circuit.metadata.update({
                    "detection_method": "robust_cross_example_induction",
                    "detection_confidence": 0.8,
                    "cross_example_consistency": pattern_data["consistency_ratio"],
                    "examples_found": pattern_data["example_coverage"],
                    "pattern_type": "robust_induction"
                })

                detector.registry.register_circuit_enhanced(
                    circuit=circuit,
                    source="robust_cross_example",
                    epoch=epoch,
                    detection_method="robust_induction",
                    confidence=0.8,
                    total_epochs=total_epochs
                )

                registered_circuits.append(circuit)
                logger.debug(f"  ✅ Registered robust induction: {pattern_key}")

            except Exception as e:
                logger.warning(f"  ⚠️ Failed to register robust induction {pattern_key}: {e}")

    else:
        # Register all stable circuits (alternative strategy)
        logger.info("📝 Registering all stable circuits...")

        for copy_result in all_copy_results:
            for circuit in copy_result["created_circuits"]:
                registered_circuits.append(circuit)

        for induction_result in all_induction_results:
            if induction_result:
                for circuit in induction_result["created_circuits"]:
                    registered_circuits.append(circuit)

    # Aggregate results
    aggregated_results = {
        "copy_mechanisms": [m for er in example_results for m in er["copy_mechanisms"]],
        "induction_patterns": [p for er in example_results for p in er["induction_patterns"]],
        "registered_circuits": registered_circuits,
        "detection_summary": {
            "examples_analyzed": len(sampled_examples),
            "registered_circuits": len(registered_circuits),
            "robust_copy_patterns": len(cross_example_patterns.get("robust_copy_patterns", {})),
            "robust_induction_patterns": len(cross_example_patterns.get("robust_induction_patterns", {}))
        },
        "cross_example_analysis": cross_example_patterns,
        "example_results": example_results
    }

    logger.info(f"  ✅ Registered {len(registered_circuits)} circuits to registry")

    return aggregated_results

def analyze_cross_example_patterns_previous(example_results, logger=None):
    """Enhanced cross-example analysis for both copy and induction patterns"""

    # Separate copy and induction patterns
    copy_pattern_counts = defaultdict(list)
    induction_pattern_counts = defaultdict(list)

    for result in example_results:
        # Process copy mechanisms
        for mechanism in result["copy_mechanisms"]:
            head = mechanism.get("head", "unknown")
            src_pos = mechanism.get("source_pos", -1)
            tgt_pos = mechanism.get("target_pos", -1)
            relative_offset = tgt_pos - src_pos if src_pos >= 0 and tgt_pos >= 0 else 0

            pattern_key = f"{head}_copy_offset_{relative_offset}"
            copy_pattern_counts[pattern_key].append({
                "example_idx": result["example_idx"],
                "mechanism": mechanism
            })

        # Process induction patterns
        for pattern in result["induction_patterns"]:
            head = pattern.get("head", "unknown")
            inducer_pos = pattern.get("inducer_pos", -1)
            target_pos = pattern.get("target_pos", -1)
            distance = target_pos - inducer_pos if inducer_pos >= 0 and target_pos >= 0 else 0

            pattern_key = f"{head}_induction_dist_{distance}"
            induction_pattern_counts[pattern_key].append({
                "example_idx": result["example_idx"],
                "pattern": pattern
            })

    # Find robust patterns for both types
    def find_robust_patterns(pattern_counts, min_examples=2):
        robust = {}
        consistent = {}

        for pattern_key, occurrences in pattern_counts.items():
            occurrence_count = len(occurrences)
            example_coverage = len(set(occ["example_idx"] for occ in occurrences))
            consistency_ratio = example_coverage / len(example_results)

            if example_coverage >= min_examples:
                robust[pattern_key] = {
                    "occurrence_count": occurrence_count,
                    "example_coverage": example_coverage,
                    "consistency_ratio": consistency_ratio,
                    "occurrences": occurrences
                }

                if consistency_ratio >= 0.7:  # 70%+ of examples
                    consistent[pattern_key] = robust[pattern_key]

        return robust, consistent

    robust_copy, consistent_copy = find_robust_patterns(copy_pattern_counts)
    robust_induction, consistent_induction = find_robust_patterns(induction_pattern_counts)

    # Log findings
    if logger and hasattr(logger, 'debug'):
        logger.debug(f"  📊 Copy patterns: {len(copy_pattern_counts)} unique, {len(robust_copy)} robust")
        logger.debug(f"  📊 Induction patterns: {len(induction_pattern_counts)} unique, {len(robust_induction)} robust")

    return {
        "robust_copy_patterns": robust_copy,
        "robust_induction_patterns": robust_induction,
        "consistent_copy_patterns": consistent_copy,
        "consistent_induction_patterns": consistent_induction,
        "copy_pattern_diversity": len(copy_pattern_counts),
        "induction_pattern_diversity": len(induction_pattern_counts)
    }

