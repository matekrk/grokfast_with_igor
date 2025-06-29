import time
from pathlib import Path

import pandas as pd

from analysis import EnhancedCircuitRegistry, CircuitThresholds, ComputationalBudget
from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
from analysis.canonical.sampled_circuit_analysis import run_canonical_circuit_analysis_with_sampling
from analysis.core.canonical_circuit_system import CanonicalCircuitRegistry, FunctionalCircuitSignatureExtractor
from analysis.core.json_safe_canonical_circuits import JSONSafeCanonicalCircuitRegistry, \
    JSONSafeCanonicalRegistryAdapter
from analysis.core.unified_logger import UnifiedLogger
from analysis.helpers import example_sampler
from analysis.helpers.example_sampler import create_aggressive_example_sampler
from analysis.sampling.diversity_enhanced_sampling import analyze_circuit_capacity_limits
from analysis.trainers.utils import detect_grokking, train_epoch, evaluate
from analysis.utils.utils import init_train_dataloader_state
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
                                enhanced_registry=None, thresholds=None):
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


def _get_logger(experiment_name, save_dir,
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


def _get_stats(train_stats, eval_stats, optimizer):
    return {
        "train_loss": train_stats.get('loss', 0.0),
        "train_accuracy": train_stats.get('accuracy', 0.0),
        "eval_loss": eval_stats.get('loss', 0.0),
        "eval_accuracy": eval_stats.get('accuracy', 0.0),
        "learning_rate": optimizer.param_groups[0]['lr'],
        # "weight decay": optimizer.param_groups[0]['weight_decay'],
    }


def train_with_default_canonical_circuits(
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
        example_sampling_strategy="diverse_random",
        example_sampling_config=None
):
    """Enhanced training with adaptive canonical circuit detection, temporal tracking, and unified logging"""

    train_dataloader_state = init_train_dataloader_state(dataloader=train_loader)
    eval_dataloader_state = init_train_dataloader_state(dataloader=eval_loader)
    if checkpointManager:
        save_dir = Path(checkpointManager.experiment_dir)
    else:
        save_dir = Path("results/enhanced_circuit_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)

    # info generate experiment name # todo get a better one
    experiment_name = f"canonical_circuit_{model.get_id() if hasattr(model, 'get_id') else 'transformer'}"
    # fixme change the name?
    logger = _get_logger(experiment_name=experiment_name, save_dir=save_dir,
                         enable_screen_logging=enable_screen_logging, log_level=log_level)
    shared_logger = model.logger if hasattr(model, 'logger') else None

    # info enhanced registry with temporal tracking
    registry = EnhancedCircuitRegistry(save_dir / "enhanced_registry")
    # info adaptive threshold system
    aggressive_thresholds = True
    if adaptive_threshold_config:
        thresholds = CircuitThresholds(**adaptive_threshold_config)
    else:
        thresholds = CircuitThresholds(
            copy_attention_min=0.5, copy_attention_max=0.95,
            induction_attention_min=0.6, induction_attention_max=0.9,
            warmup_epochs=150, min_accuracy_threshold=0.25,
        ) if aggressive_thresholds else (
            CircuitThresholds(
                copy_attention_min=0.3, copy_attention_max=0.9,
                induction_attention_min=0.4, induction_attention_max=0.85,
                warmup_epochs=50, min_accuracy_threshold=0.2
            ))
    circuit_system = initialize_canonical_circuits_experimental_architecture(
        model, save_dir, logger, enhanced_registry=registry, thresholds=thresholds)
    canonical_registry = circuit_system['canonical_registry']
    canonical_detector = circuit_system['canonical_detector']
    evolution_tracker = circuit_system['evolution_tracker']
    example_sampler = circuit_system['example_sampler']
    adapter = circuit_system['adapter']
    circuits_quality_analyzer = circuit_system['quality_analyzer']

    # fixme ##############################################################################
    # todo write a new true circuit testing for each problem
    # fixme ##############################################################################
    from analysis.validation.circuit_testing import integrate_real_testing_with_quality_analyzer
    circuits_quality_analyzer = integrate_real_testing_with_quality_analyzer(circuits_quality_analyzer)


    # whatis other analyzers initializations fixme move to some local function
    weight_tracker = EnhancedWeightSpaceTracker(
        model=model, save_dir=save_dir / "weight_tracking",
        logger=shared_logger, registry=registry,
        jump_detection_window=100, snapshot_freq=analyze_interval // 2
    )

    # info computational budget manager
    budget = ComputationalBudget(max_time_per_epoch=10 * computational_budget_per_epoch)
    # info sampling_configs
    all_sampling_configs = get_sampling_configs(eval_loader=eval_loader, num_samples=14)
    # info create enhanced sampler
    # sampling_config = all_sampling_configs["exploration"]
    # example_sampler = create_fixed_example_sampler(eval_loader, sampling_config)

    # info create aggressive sampler
    aggressive_config = all_sampling_configs["aggressive"]
    example_sampler = create_aggressive_example_sampler(eval_loader, aggressive_config)
    logger.info(f"🎲 Example sampler initialized with strategy: {example_sampling_strategy}")

    # info set the log interval multiplier fixme if epoch % (log_interval_multiplier * log_interval) == 0: ...
    log_interval_mult = 2

    df_circuit = pd.DataFrame({
        'epoch': pd.Series(dtype='int'),
        'circuit_id': pd.Series(dtype='str'),
        'type': pd.Series(dtype='str'),
        'detections': pd.Series(dtype='int'),
        'stability': pd.Series(dtype='float'),
        'quality': pd.Series(dtype='float'),
        'recommendation': pd.Series(dtype='str'),
        'stable': pd.Series(dtype='bool'),
        'first_seen': pd.Series(dtype='int'),
        'last_seen': pd.Series(dtype='int'),
    })

    # warning ######################################################################################
    # fixme   ######################################################################################
    # whatis  ######################################################################################
    # whatis  start main loop start  fixme main loop  warning start main loop
    # whatis  ######################################################################################
    # fixme   ######################################################################################
    # warning ######################################################################################
    analysis_results = {}
    eval_stats = None
    current_accuracy = 0.0
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

        should_analyze = epoch % analyze_interval == 0 or epoch == epochs - 1
        should_evaluate = epoch % log_interval == 0 or epoch == epochs - 1

        # info evaluate if eval_stats obsolete; whatis check for grokking
        if (should_evaluate or should_analyze) and eval_loader:
            eval_stats = evaluate(model, eval_loader, criterion, device)
            current_accuracy = eval_stats['accuracy']

            training_metrics = _get_stats(train_stats, eval_stats, optimizer)
            if epoch % (log_interval_mult * log_interval) == 0:
                logger.log_metrics(training_metrics, step=epoch, category="training")

            # log_metrics(model, epoch, train_stats, eval_stats)
            detect_grokking(model, epoch, train_stats, eval_stats)

        # info tracking weights by taking the model's current state
        weight_tracker.take_snapshot(epoch=epoch)

        # info check if token detection should be done now; register results
        should_detect = thresholds.should_start_detection(epoch, current_accuracy)
        # whatis start analysis
        if should_analyze and eval_loader and should_detect:
            # logger.info(f"📊 canonical circuit analysis @ epoch {epoch}")

            if eval_stats is None:
                eval_stats = evaluate(model, eval_loader, criterion, device)
                current_accuracy = eval_stats['accuracy']

            analysis_start_time = time.time()
            epoch_results = {}
            # info check for budget
            canonical_circuit_analysis = 'fast'
            # canonical_circuit_analysis = 'diversity_sampling'
            if canonical_detector and budget.can_run_method("canonical_detector"):
                if canonical_circuit_analysis == 'fast':
                    from analysis.canonical.sampled_circuit_analysis import run_canonical_circuit_analysis_with_fast_sampling
                    canonical_results=run_canonical_circuit_analysis_with_fast_sampling(
                        canonical_detector=canonical_detector,
                        eval_loader=eval_loader,
                        epoch=epoch, total_epochs=epochs,
                        accuracy=current_accuracy,
                        logger=logger, detect_induction=True,
                        cross_example_threshold=0.5,
                        min_examples_for_robustness=6,
                        num_samples=12)

                elif canonical_circuit_analysis == 'diversity_sampling':
                    # fixme change the strategy according to the learning stage
                    sampling_strategy = "diverse_random"
                    canonical_results = run_canonical_circuit_analysis_with_sampling(
                        canonical_detector=canonical_detector,
                        example_sampler=example_sampler,
                        epoch=epoch, total_epochs=epochs, accuracy=current_accuracy,
                        sampling_strategy=sampling_strategy,
                        logger=logger,
                        detect_induction=True,
                        analyze_interval=analyze_interval,
                        cross_example_threshold=0.5,  # 0.3,
                        min_examples_for_robustness=6,  # 2
                    )
                else:
                    # warning no method evailable
                    canonical_results = None
                # whatis get the theoretical transformer circuit capacity limits
                # circuit_limits = analyze_circuit_capacity_limits(
                #     model_architecture={
                #         'num_layers': model.num_layers,
                #         'num_heads': model.num_heads,
                #         'embedding_dim': model.dim,
                #     },
                #     discovered_circuits=canonical_results,
                # )
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
                    # logger.log_metrics(canonical_metrics, step=epoch, category="canonical_sampling")
                    # logger.info(f"    ✅ {canonical_metrics['registered_circuits']} circuits registered"
                    #             f" | {canonical_metrics['stable_circuits']} stable (aggregation_rate: {canonical_metrics['aggregation_rate']:.2%})")

                # if epoch > 50:  # Start evolution analysis after some circuits accumulated
                #     stable_circuits = canonical_registry.get_stable_circuits(min_stability=0.5)
                #
                #     for circuit in stable_circuits[:5]:  # Analyze top 5 stable circuits
                #         evolution_data = canonical_registry.analyze_circuit_evolution(circuit.canonical_id)
                #
                #         logger.info(f"    🧬 {circuit.canonical_id}: "
                #                     f"{evolution_data['total_detections']} detections, "
                #                     f"stability {evolution_data['stability_score']:.3f}, "
                #                     f"trend {evolution_data.get('attribution_trend', {}).get('trend', 'unknown')}")

                stable_canonical_ids, most_stable, least_stable = canonical_detector.prune_unstable_canonical_circuits(
                    epoch, min_stability=0.3
                )
                # todo fixme ######################################################################################
                # fixme ###########################################################################################
                # todo warning prune circuits which are not true anymore warning warning fixme or are unimportant
                # fixme ###########################################################################################
                # todo ############################################################################################
                # info log the most and least stable circuits # warning only stable!
                if (len(stable_canonical_ids) > 1 and most_stable is not None and
                        least_stable is not None and epoch % (log_interval_mult * log_interval) == 0):
                    logger.info(f"    🧬 [{len(stable_canonical_ids)} stable]:"
                        f"  (most) {"::".join([most_stable.canonical_id.split("_")[0], most_stable.canonical_id.split("_")[-1]])} "
                        f" (stabil. {most_stable.stability_score:.3f}), "
                        f" (persist. {most_stable.persistence_score:.3f}), "
                        f" [{most_stable.first_seen}-{most_stable.last_seen}]  | "
                        f"  (least) {"::".join([least_stable.canonical_id.split("_")[0], least_stable.canonical_id.split("_")[-1]])} "
                        f" (stabil. {least_stable.stability_score:.3f}), "
                        f" [{most_stable.first_seen}-{most_stable.last_seen}])"
                        f" (persist. {least_stable.persistence_score:.3f})")

                # info check the circuits quality
                # info do it, probably, on some epoch multiplicity basis
                if epoch >= 200 and epoch % (log_interval_mult * log_interval * 10) == 0:
                    quality_results = circuits_quality_analyzer.analyze_all_circuits(eval_loader=eval_loader,
                                                                                      max_circuits=100)
                    quality_report = quality_results["quality_report"]
                    if quality_report is not None:
                        pass
                    circuits_quality_analyzer.visualize_circuit_quality(quality_results['circuit_analyses'],)

                    print(f"\n📊 Circuit Quality Report:")
                    print(f"  Total circuits: {quality_report['total_circuits']}")
                    print(f"  Recommended to KEEP: {quality_report['keep_circuits']}")
                    print(f"  Recommended to REMOVE: {quality_report['remove_circuits']}")
                    print(f"  Mean quality score: {quality_report['quality_statistics']['mean_quality']:.3f}")

                    batch_dict = {col: [] for col in df_circuit.columns}
                    # info save results to dataframe
                    for k, v in quality_results['circuit_analyses'].items():
                        batch_dict['epoch'].append(epoch)
                        batch_dict['circuit_id'].append(k)
                        batch_dict['type'].append(v['operation_type'])
                        batch_dict['detections'].append(v['total_detections'])
                        batch_dict['stability'].append(v['stability_score'])
                        batch_dict['quality'].append(v['quality_score'])
                        batch_dict['recommendation'].append(v['recommendation'].split(' ')[0])
                        if k in stable_canonical_ids:
                            batch_dict['stable'].append(True)
                        else:
                            batch_dict['stable'].append(False)
                        if k in canonical_registry.canonical_circuits.keys():
                            batch_dict['first_seen'].append(canonical_registry.canonical_circuits[k].first_seen)
                            batch_dict['last_seen'].append(canonical_registry.canonical_circuits[k].last_seen)
                        else:
                            batch_dict['first_seen'].append(0)      # warning perhaps None?
                            batch_dict['last_seen'].append(0)       # warning perhaps None?
                    new_batch = pd.DataFrame(batch_dict)
                    df_circuit = pd.concat([df_circuit, new_batch], ignore_index=True)
                    pass

                continue
                # info update evolution statistics
                if len(canonical_registry.circuits) > 1:
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
        if epoch % 100 == 0 and epoch > 0:
            summary = registry.get_registry_summary()
            try:
                canonical_registry.save(save_dir / f"canonical_registry_epoch_{epoch}.json")
            except Exception as e:
                logger.warning(f"Failed to save canonical registry: {e}")
    # whatis final
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

    return model, canonical_registry, final_summary, None
