# analysis/trainers/train_with_enhanced_circuit_analysis.py (UPDATED with UnifiedLogger)
from collections import defaultdict

# import torch
import time
from pathlib import Path

from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
from analysis.analyzers.continuous_circuit_tracker import ContinuousCircuitTracker

from analysis.core import EnhancedCircuitRegistry, CircuitThresholds, ComputationalBudget
from analysis.core.circuit_stability import EnhancedRegistryLifecycleManager
from analysis.core.unified_logger import UnifiedLogger  # ✅ ADD: UnifiedLogger import
from analysis.core.circuit_evolution_manager import integrate_with_registry

from analysis.utils.analysis_summary import create_final_analysis_summary
from analysis.utils.example_sampler import create_fixed_example_sampler  # , create_example_sampler, create_fixed_example_sampler
from analysis.utils.validation_helpers import save_enhanced_checkpoint
from analysis.utils.missing_functions import run_circuit_validation
from analysis.utils.circuit_analysis import analyze_circuit_emergence, analyze_circuit_relationships

from analysis.utils.utils import init_train_dataloader_state

from analysis.validation import CircuitManipulationValidator


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
        enable_wandb_logging=False,
        enable_file_logging=False,
        enable_screen_logging=True,
        log_level="INFO",
        example_sampling_strategy = "diverse_random",
        example_sampling_config = None
):
    """Enhanced training with adaptive circuit detection, temporal tracking, and unified logging"""


    # whatis ####################################################################
    # info initialization of several objects
    # whatis ####################################################################
    # info dataloader states
    train_dataloader_state = init_train_dataloader_state(dataloader=train_loader)
    eval_dataloader_state = init_train_dataloader_state(dataloader=eval_loader)

    # info setup directories
    if checkpointManager:
        save_dir = Path(checkpointManager.experiment_dir)
    else:
        save_dir = Path("results/enhanced_circuit_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)


    # info generate experiment name # todo get a better one
    experiment_name = f"circuit_analysis_{model.get_id() if hasattr(model, 'get_id') else 'transformer'}"

    # info unified messages logger
    logger = UnifiedLogger(
        experiment_name=experiment_name,
        log_dir=save_dir / "logs",
        enable_wandb=enable_wandb_logging,
        enable_file=enable_file_logging,
        enable_screen=enable_screen_logging,
        log_level=log_level
    )
    shared_logger = model.logger if hasattr(model, 'logger') else None
    logger.info(f"🔬 Enhanced Circuit Analysis Experiment: {experiment_name}")
    logger.info(f"📊 Training for {epochs} epochs with adaptive detection: {enable_adaptive_detection}")

    # whatis configurations for several objects
    # info initial configuration
    config_metrics = {
        "total_epochs": epochs,
        "analyze_interval": analyze_interval,
        "adaptive_detection": enable_adaptive_detection,
        "validation_enabled": enable_circuit_validation,
        "budget_per_epoch": computational_budget_per_epoch
    }
    logger.log_metrics(config_metrics, step=0, category="configuration")

    # whatis ExampleSampler object configurations
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

    # whatis end of objects' configurations

    # info =======================================================================
    # whatis experiment infrastructure; objects
    # info =======================================================================

    # info enhanced registry with temporal tracking
    registry = EnhancedCircuitRegistry(save_dir / "enhanced_registry")
    # info get circuit evolution tracker object
    evolution_tracker = integrate_with_registry(registry=registry, save_dir=save_dir / "evolution_tracking")
    # info alternative explicit initialization:
    # evolution_manager = CircuitEvolutionManager(registry, save_dir / "evolution_tracking", logger)
    # evolution_tracker = evolution_manager.create_tracker('unified')

    # info initialize lifecycle management
    # stability_analyzer = CircuitStabilityAnalyzer(registry, stability_window=50)
    # lifecycle_manager = EnhancedRegistryLifecycleManager(registry, stability_analyzer)
    lifecycle_manager = EnhancedRegistryLifecycleManager(registry)

    # info adaptive threshold system
    if adaptive_threshold_config:
        thresholds = CircuitThresholds(**adaptive_threshold_config)
    else:
        thresholds = CircuitThresholds(
            copy_attention_min=0.3, copy_attention_max=0.9,
            induction_attention_min=0.4, induction_attention_max=0.85,
            warmup_epochs=50, min_accuracy_threshold=0.2
        )

    # info computational budget manager
    budget = ComputationalBudget(max_time_per_epoch=computational_budget_per_epoch)

    # info circuit validation (optional)
    validator = None
    if enable_circuit_validation:
        validator = CircuitManipulationValidator(model, eval_loader, batch_limit=2)
        logger.info("🧪 Circuit validation enabled")

    # info analyzers (already existing, warning shall be changed later)

    # info weight space tracker
    weight_tracker = EnhancedWeightSpaceTracker(
        model=model, save_dir=save_dir / "weight_tracking",
        logger=shared_logger, registry=registry,
        jump_detection_window=100, snapshot_freq=analyze_interval // 2
    )
    # info continuous circuit tracker fixme to be changed?
    circuit_tracker = ContinuousCircuitTracker(
        model=model, save_dir=save_dir / "circuit_tracking",
        logger=shared_logger, registry=registry,
        sampling_freq=circuit_sampling_freq
    )

    # info adaptive token detector
    adaptive_detector = None
    if enable_adaptive_detection:
        from analysis.analyzers.adaptive_token_operations import RegistrationAwareAdaptiveTokenOperationDetector
        adaptive_detector = RegistrationAwareAdaptiveTokenOperationDetector(
            model=model, registry=registry, thresholds=thresholds
        )
        adaptive_detector.evolution_tracker = evolution_tracker
        logger.info("🔍 Adaptive token detection enabled")
    logger.info("📋 All analyzers initialized successfully") # fixme is that true?

    sampling_config = exploration_config
    # example_sampler = create_example_sampler(eval_loader, sampling_config)
    example_sampler = create_fixed_example_sampler(eval_loader, sampling_config)
    logger.info(f"🎲 Example sampler initialized with strategy: {example_sampling_strategy}")


    # info =====================================================================
    # whatis end of experiment infrastructure; objects
    # info =====================================================================

    # whatis ============================================================================
    # info training loop with adaptive token circuit detection, enhanced analysis
    # whatis ============================================================================

    analysis_results = {}
    training_start_time = time.time()

    # info initial setup
    weight_tracker.take_snapshot(epoch=0, force=True)

    for epoch in range(epochs):
        # info train and evaluate; save results in tain_stats and eval_stats
        epoch_start_time = time.time()
        budget.start_epoch()

        train_stats = train_epoch(model, train_loader, criterion, optimizer, epoch, device)

        if scheduler:
            scheduler.step()

        should_evaluate = epoch % log_interval == 0 or epoch == epochs - 1
        eval_stats = None
        current_accuracy = 0.0

        # info evaluate if eval_stats is not anymore current; whatis check for grokking
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


        # info tracking weights by taking the model's current state
        weight_tracker.take_snapshot(epoch=epoch)

        # info check if token detection should be done now; register results
        should_analyze = epoch % analyze_interval == 0 or epoch == epochs - 1
        should_detect = thresholds.should_start_detection(epoch, current_accuracy)

        if should_analyze and eval_loader and should_detect:
            # logger.info(f"📊 Enhanced circuit analysis @ epoch {epoch}")

            if eval_stats is None:
                eval_stats = evaluate(model, eval_loader, criterion, device)
                current_accuracy = eval_stats['accuracy']

            analysis_start_time = time.time()
            epoch_results = {}

            # =================================================================
            # info perform adaptive token detection; register detected token circuits

            # whatis check for available budget to run info adaptive_token_detection
            if (enable_adaptive_detection and adaptive_detector and
                    budget.can_run_method("adaptive_token_detection")):
                token_start = time.time()
                logger.debug("🔍 Running adaptive token detection with intelligent sampling...")

                # info change the sampling config in later stages of learning;
                #  the configurations are defined in the top part
                sampling_strategy = "diverse_random"    # fixme change and try other strategies
                # info run adaptive token-based circuit detection
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

                epoch_results["adaptive_token_results"] = adaptive_results

                token_time = time.time() - token_start
                budget.record_execution_time("adaptive_token_detection", token_time)

                # info log detection results
                adaptive_metrics = {
                    "stable_circuits": len(adaptive_results.get('stable_circuits', [])),
                    "total_detected": adaptive_results.get('detection_summary', {}).get('total_detected', 0),
                    "stability_rate": adaptive_results.get('detection_summary', {}).get('stability_rate', 0.0),
                    "detection_time": token_time
                }
                logger.log_metrics(adaptive_metrics, step=epoch, category="adaptive_detection")
                logger.info(f"    ✅ Found {adaptive_metrics['total_detected']} circuits "
                            f" | {adaptive_metrics['stable_circuits']} stable (rate: {adaptive_metrics['stability_rate']:.2%})")

                # info update evolution statistics
                if len(registry.circuits) > 1:
                    # info update evolution tracking from registry state
                    evolution_summary = evolution_tracker.analyze_emergence_order()
                    epoch_results["evolution_summary"] = evolution_summary

                    # info get comprehensive evolution analysis
                    emergence_analysis = evolution_tracker.analyze_emergence_order()
                    relationship_analysis = evolution_tracker.analyze_circuit_relationships()
                    stability_summary = evolution_tracker.get_evolution_summary(epoch)

                    epoch_results["emergence_analysis"] = emergence_analysis
                    epoch_results["relationship_analysis"] = relationship_analysis
                    epoch_results["stability_summary"] = stability_summary

                    # info enhanced logging with unified tracker data
                    stable_circuits = evolution_tracker.get_stable_circuits(epoch)
                    evolution_metrics = {
                        "total_circuits": len(registry.circuits),
                        "stable_circuits": len(stable_circuits),
                        "circuit_birth_rate": stability_summary.get('birth_events', 0),
                        "circuit_death_rate": stability_summary.get('death_events', 0),
                        "survival_rate": stability_summary.get('survival_rate', 0.0),
                        "avg_circuit_lifetime": stability_summary.get('avg_lifetime', 0.0)
                    }
                    logger.log_metrics(evolution_metrics, step=epoch, category="circuit_evolution_unified")

                    # info log some detailed
                    if emergence_analysis.get("emergence_order", False):
                        logger.info(f"    📈 Circuit emergence order: {emergence_analysis['emergence_order']}")
                    if len(stable_circuits) > 0:
                        top_stable = stable_circuits[:3]
                        for i, circuit_info in enumerate(top_stable, 1):
                            logger.info(f"    🏆 Stable #{i}: {circuit_info['circuit_id']} "
                                        f"(stability: {circuit_info['stability_score']:.3f}, "
                                        f"lifetime: {circuit_info['lifetime']} epochs)")
            # whatis ======================================================================
            # info component analysis
            # whatis ======================================================================
            if budget.can_run_method("component_analysis"):
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


            # whatis =================================================================
            # info validate circuits
            # whatis =================================================================

            should_validate = (enable_circuit_validation and validator and
                               epoch % validation_interval == 0 and
                               budget.can_run_method("validation"))

            if should_validate:
                validation_start = time.time()
                # logger.info("🧪 Running circuit validation...")

                validation_results = run_circuit_validation(registry, validator, epoch, eval_stats['accuracy'])
                epoch_results["validation_results"] = validation_results

                validation_time = time.time() - validation_start
                budget.record_execution_time("validation", validation_time)

                # ✅ LOG: Validation results
                logger.log_validation_results(validation_results, epoch)

            # whatis =================================================================
            # info ENHANCED REGISTRY ANALYSIS WITH LOGGING
            # whatis =================================================================

            # info circuit emergence analysis
            if len(registry.emergence_timeline) > 1:
                emergence_analysis = analyze_circuit_emergence(evolution_tracker, epoch)
                epoch_results["emergence_analysis"] = emergence_analysis

                emergence_metrics = {
                    "total_emerged": len(registry.emergence_timeline.get(epoch, [])),
                    "emergence_rate": len(registry.emergence_timeline.get(epoch, [])) / max(1, epoch)
                }
                logger.log_metrics(emergence_metrics, step=epoch, category="circuit_emergence")

            # info relationship discovery
            if len(registry.circuits) > 1:
                relationship_analysis = analyze_circuit_relationships(registry, epoch)
                epoch_results["relationship_analysis"] = relationship_analysis

                circuit_evolution_info = {
                    "total_circuits": len(registry.circuits),
                    "stable_circuits": len(registry.get_circuits_by_stability("stable")),
                    "emerging_circuits": len(registry.get_circuits_by_stability("emerging")),
                    "total_relationships": sum(len(rels) for rels in registry.relationship_graph.values())
                }
                logger.log_circuit_evolution(circuit_evolution_info, epoch)

            analysis_results[epoch] = epoch_results
            total_analysis_time = time.time() - analysis_start_time

            analysis_performance = {
                "total_analysis_time": total_analysis_time,
                "budget_utilization": budget.get_usage_summary()["budget_utilization"]
            }
            logger.log_metrics(analysis_performance, step=epoch, category="analysis_performance")

            # logger.info(f"    ⏱️  Total analysis time: {total_analysis_time:.2f}s")

            # Budget summary
            usage = budget.get_usage_summary()
            if usage["budget_utilization"] > 0.8:
                logger.warning(f"⚠️  High budget usage: {usage['budget_utilization']:.1%}")

        # info enhanced registry logging
        if epoch % 50 == 0 and epoch > 0:
            summary = registry.get_registry_summary()

            # info registry summary
            registry_metrics = {
                "total_circuits": summary['total_circuits'],
                "total_relationships": summary['total_relationships'],
                "detection_methods": len(summary['detection_methods'])
            }
            logger.log_metrics(registry_metrics, step=epoch, category="registry_status")

            logger.info(f"📈 Registry @ epoch {epoch}: {summary['total_circuits']} circuits, "
                        f"{summary['total_relationships']} relationships")

            maintenance_summary = lifecycle_manager.perform_lifecycle_maintenance(epoch, logger)
            lifecycle_metrics = {
                "circuits_removed": maintenance_summary["removed"],
                "circuits_downgraded": maintenance_summary["downgraded"],
                "circuits_active": maintenance_summary["kept"],
                "total_circuits": len(registry.circuits)
            }
            logger.info(f"    ✅ Lifecycle metrics {lifecycle_metrics['total_circuits']} total circuits "
                        f" | {lifecycle_metrics['circuits_active']} active "
                        f" | {lifecycle_metrics['circuits_downgraded']} downgraded | {lifecycle_metrics['circuits_removed']} removed ")

            summary = registry.get_registry_summary()
            evolution_summary = evolution_tracker.get_evolution_summary(epoch)

            # Combined registry and evolution metrics
            combined_metrics = {
                "total_circuits": summary['total_circuits'],
                "total_relationships": summary['total_relationships'],
                "living_circuits": evolution_summary['living_circuits'],
                "stable_circuits": evolution_summary['stable_circuits'],
                "survival_rate": evolution_summary['survival_rate']
            }
            logger.log_metrics(combined_metrics, step=epoch, category="registry_evolution_status")

            logger.info(f"📈 Registry + Evolution @ epoch {epoch}: "
                        f"{summary['total_circuits']} total circuits, "
                        f"{evolution_summary['living_circuits']} living, "
                        f"{evolution_summary['stable_circuits']} stable "
                        f"(survival rate: {evolution_summary['survival_rate']:.2%})")

            registry.save()

        # info save checkpoints
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

    logger.info("🔍 Creating final analysis summary with unified evolution data...")

    # ✅ UPDATED: Enhanced final summary with evolution data
    final_evolution_summary = evolution_tracker.get_evolution_summary(epochs)
    final_summary = create_final_analysis_summary(registry, analysis_results, epochs)

    # Add evolution data to final summary
    final_summary['evolution_analysis'] = final_evolution_summary
    final_summary['emergence_patterns'] = evolution_tracker.analyze_emergence_order()
    final_summary['circuit_relationships'] = evolution_tracker.analyze_circuit_relationships()

    # ✅ NEW: Enhanced final metrics with evolution data
    summary_metrics = {
        "total_circuits_final": final_summary['metadata']['total_circuits'],
        "stable_circuits_final": final_evolution_summary['stable_circuits'],
        "circuit_survival_rate": final_evolution_summary['survival_rate'],
        "avg_circuit_lifetime": final_evolution_summary['avg_lifetime'],
        "total_birth_events": final_evolution_summary['birth_events'],
        "total_death_events": final_evolution_summary['death_events'],
        "analysis_coverage": final_summary['metadata']['analysis_coverage'],
        "total_training_time": time.time() - training_start_time
    }
    logger.log_metrics(summary_metrics, step=epochs, category="final_summary_with_evolution")

    # ✅ NEW: Evolution-specific insights
    evolution_insights = []

    emergence_order = final_summary['emergence_patterns'].get('emergence_order', [])
    if emergence_order:
        evolution_insights.append(f"Circuit emergence order: {' → '.join(emergence_order)}")

    if final_evolution_summary['survival_rate'] > 0.7:
        evolution_insights.append("High circuit survival rate indicates stable learning dynamics")
    elif final_evolution_summary['survival_rate'] < 0.3:
        evolution_insights.append("Low circuit survival rate suggests rapid circuit turnover")

    if final_evolution_summary['avg_lifetime'] > 100:
        evolution_insights.append("Long average circuit lifetime indicates persistent computational patterns")

    stable_circuit_ratio = final_evolution_summary['stable_circuits'] / max(1, final_evolution_summary[
        'total_circuits_discovered'])
    if stable_circuit_ratio > 0.5:
        evolution_insights.append("High stable circuit ratio suggests robust circuit formation")

    # Add evolution insights to final summary
    final_summary['evolution_insights'] = evolution_insights

    logger.info(f"🧬 Evolution Insights:")
    for i, insight in enumerate(evolution_insights, 1):
        logger.info(f"  {i}. {insight}")

    # Save enhanced final summary
    final_summary_path = save_dir / "final_analysis_summary_with_evolution.json"
    with open(final_summary_path, 'w') as f:
        import json
        json.dump(final_summary, f, indent=2, default=str)

    logger.info(f"💾 Enhanced final analysis saved to: {final_summary_path}")

    # ✅ FINALIZE: Close unified logger
    logger.finalize_experiment()

    print("🎉 Enhanced circuit analysis training complete with unified evolution tracking!")
    return model, analysis_results, registry, final_summary, evolution_tracker





def analyze_cross_example_patterns(example_results, logger=None):
    """
    whatis Enhanced cross-example analysis for both copy and induction patterns

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

    # info initialize collections for both pattern types
    copy_pattern_counts = defaultdict(list)
    induction_pattern_counts = defaultdict(list)

    # info process each example's results
    for result in example_results:
        example_idx = result["example_idx"]

        # info process copy mechanisms
        for mechanism in result["copy_mechanisms"]:
            head = mechanism.get("head", "unknown")
            src_pos = mechanism.get("source_pos", -1)
            tgt_pos = mechanism.get("target_pos", -1)
            relative_offset = tgt_pos - src_pos if src_pos >= 0 and tgt_pos >= 0 else 0

            # info create pattern signature based on head and relative position
            pattern_key = f"{head}_copy_offset_{relative_offset}"
            copy_pattern_counts[pattern_key].append({
                "example_idx": example_idx,
                "mechanism": mechanism
            })

        # info process induction patterns
        for pattern in result["induction_patterns"]:
            head = pattern.get("head", "unknown")
            inducer_pos = pattern.get("inducer_pos", -1)
            target_pos = pattern.get("target_pos", -1)
            distance = target_pos - inducer_pos if inducer_pos >= 0 and target_pos >= 0 else 0

            # info create pattern signature based on head and distance
            pattern_key = f"{head}_induction_dist_{distance}"
            induction_pattern_counts[pattern_key].append({
                "example_idx": example_idx,
                "pattern": pattern
            })

    # Helper function to find robust patterns
    def find_robust_patterns(pattern_counts, min_examples=2):
        """
        whatis find patterns that appear in multiple examples

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

                # info highly consistent patterns (appear in 70%+ of examples)
                if consistency_ratio >= 0.7:
                    consistent[pattern_key] = robust[pattern_key]

        return robust, consistent

    # info find robust patterns for both types
    robust_copy, consistent_copy = find_robust_patterns(copy_pattern_counts)
    robust_induction, consistent_induction = find_robust_patterns(induction_pattern_counts)

    if logger and hasattr(logger, 'debug'):
        logger.debug(f"  📊 Copy patterns: {len(copy_pattern_counts)} unique, {len(robust_copy)} robust")
        logger.debug(f"  📊 Induction patterns: {len(induction_pattern_counts)} unique, {len(robust_induction)} robust")

        # info log top robust patterns
        for pattern_name, pattern_data in list(robust_copy.items())[:3]:  # Top 3 copy
            consistency = pattern_data["consistency_ratio"]
            logger.debug(f"    🔗 Robust copy: {pattern_name} ({consistency:.1%} consistency)")

        for pattern_name, pattern_data in list(robust_induction.items())[:3]:  # Top 3 induction
            consistency = pattern_data["consistency_ratio"]
            logger.debug(f"    🔄 Robust induction: {pattern_name} ({consistency:.1%} consistency)")

    # info calculate additional statistics
    total_examples = len(example_results)
    total_copy_occurrences = sum(len(occurrences) for occurrences in copy_pattern_counts.values())
    total_induction_occurrences = sum(len(occurrences) for occurrences in induction_pattern_counts.values())

    # Return comprehensive analysis
    return {
        # info robust patterns (appear in 2+ examples)
        "robust_copy_patterns": robust_copy,
        "robust_induction_patterns": robust_induction,

        # info consistent patterns (appear in 70%+ examples)
        "consistent_copy_patterns": consistent_copy,
        "consistent_induction_patterns": consistent_induction,

        # info diversity metrics
        "copy_pattern_diversity": len(copy_pattern_counts),
        "induction_pattern_diversity": len(induction_pattern_counts),

        # info coverage metrics
        "cross_example_coverage": {
            "copy": {pattern: data["consistency_ratio"] for pattern, data in robust_copy.items()},
            "induction": {pattern: data["consistency_ratio"] for pattern, data in robust_induction.items()}
        },

        # info summary statistics
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




def run_complete_adaptive_token_analysis_with_registration(
        detector, example_sampler, epoch, total_epochs, accuracy,
        sampling_strategy="diverse_random", logger=None, detect_induction=True,
        analyze_interval=2, register_robust_only=True):
    """
    whatis complete analysis with proper circuit registration strategy - FIXED VERSION

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

    # info logger setup
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    # info get diverse examples
    sampled_examples = example_sampler.sample_examples(
        epoch=epoch, total_epochs=total_epochs, strategy=sampling_strategy, seed_offset=42
    )

    # logger.info(f"🎲 Sampled {len(sampled_examples)} examples using '{sampling_strategy}' strategy @ epoch {epoch}")

    # info analyze each example WITHOUT registering circuits yet
    all_copy_results = []
    all_induction_results = []
    example_results = []

    for example_idx, (inputs, targets) in enumerate(sampled_examples):
        logger.debug(f"  🔍 Analyzing example {example_idx + 1}/{len(sampled_examples)}")

        tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]

        # info forward pass
        outputs = detector.model(inputs, store_attention=True)
        attention_patterns = detector.model.get_attention_patterns()

        # info detect without registering (register_circuits=False)
        copy_results = detector.detect_and_register_copy_mechanisms(
            attention_patterns=attention_patterns,
            tokens=tokens,
            epoch=epoch,
            total_epochs=total_epochs,
            model_accuracy=accuracy,
            content_aware=True,
            register_circuits=False  # info don't register individual examples yet
        )

        induction_results = None
        if detect_induction:
            induction_results = detector.detect_and_register_induction_patterns(
                attention_patterns=attention_patterns,
                tokens=tokens,
                epoch=epoch,
                total_epochs=total_epochs,
                model_accuracy=accuracy,
                register_circuits=False  # info don't register individual examples yet
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

    # info cross example analysis: find robust patterns
    # logger.info("🔗 Finding robust patterns across examples...")
    cross_example_patterns = analyze_cross_example_patterns(example_results, logger=logger)

    # info registration strategy:: register robust circuits with high confidence
    registered_circuits = []

    if register_robust_only:
        # logger.info("📝 Registering only robust cross-example circuits...")

        # info register robust copy patterns
        robust_copy = cross_example_patterns.get("robust_copy_patterns", {})
        for pattern_key, pattern_data in robust_copy.items():
            try:
                # info create representative circuit from robust pattern
                representative_occurrence = pattern_data["occurrences"][0]
                mechanism = representative_occurrence["mechanism"]

                # info get tokens from first example that showed this pattern
                example_idx = representative_occurrence["example_idx"]
                tokens = example_results[example_idx]["tokens"]

                # info use correct method name from ModernCircuitCreator
                circuit = detector.circuit_creator.create_circuit_from_adaptive_detection(
                    mechanism, tokens, epoch
                )

                # info enhanced metadata for robust patterns
                circuit.metadata.update({
                    "detection_method": "robust_cross_example_copy",
                    "detection_confidence": 0.8,  # High confidence for robust patterns
                    "cross_example_consistency": pattern_data["consistency_ratio"],
                    "examples_found": pattern_data["example_coverage"],
                    "occurrence_count": pattern_data["occurrence_count"],
                    "pattern_type": "robust_copy"
                })

                # info register with high confidence
                detector.registry.register_circuit_enhanced(
                    circuit=circuit,
                    source="robust_cross_example",
                    epoch=epoch,
                    detection_method="robust_copy",
                    confidence=0.8,
                    total_epochs=total_epochs
                )

                registered_circuits.append(circuit)
                # logger.debug(f"  ✅ Registered robust copy: {pattern_key}")

            except Exception as e:
                logger.warning(f"  ⚠️ Failed to register robust copy {pattern_key}: {e}")
                import traceback
                traceback.print_exc()

        # info register robust induction patterns
        robust_induction = cross_example_patterns.get("robust_induction_patterns", {})
        for pattern_key, pattern_data in robust_induction.items():
            try:
                representative_occurrence = pattern_data["occurrences"][0]
                pattern = representative_occurrence["pattern"]

                example_idx = representative_occurrence["example_idx"]
                tokens = example_results[example_idx]["tokens"]

                # info use correct method name from ModernCircuitCreator
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
                # logger.debug(f"  ✅ Registered robust induction: {pattern_key}")

            except Exception as e:
                logger.warning(f"  ⚠️ Failed to register robust induction {pattern_key}: {e}")
                import traceback
                traceback.print_exc()

    else:
        # info alternative strategy: register all stable circuits
        logger.info("📝 Registering all stable circuits...")

        for copy_result in all_copy_results:
            for circuit in copy_result["created_circuits"]:
                registered_circuits.append(circuit)

        for induction_result in all_induction_results:
            if induction_result:
                for circuit in induction_result["created_circuits"]:
                    registered_circuits.append(circuit)

    # info aggregate results
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
