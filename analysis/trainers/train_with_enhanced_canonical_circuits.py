import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.canonical.sampled_circuit_analysis import create_standard_canonical_system
# from analysis.core.circuit_evolution_analyzer import create_evolution_analyzer_from_existing_tracker
from analysis.helpers.example_sampler import create_aggressive_example_sampler, create_fast_circuit_example_sampler
from analysis.temporal.temporal_circuit_emergence_analyzer import create_temporal_emergence_analyzer
from analysis.trainers.utils import detect_grokking, train_epoch, evaluate, get_default_circuit_config, \
    create_standard_thresholds, validate_canonical_system_health, initialize_circuit_dataframe, \
    log_circuits_to_dataframe, analyze_training_results, get_sampling_configs, _get_stats, _should_log_metrics, \
    _should_assess_circuits


def perform_robust_circuit_detection(canonical_detector, eval_loader, epoch, total_epochs,
                                     accuracy, logger, sampler=None, num_samples=12):
    """
    🔬 ROBUST multi-example circuit detection using intelligent sampling
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


def train_with_enhanced_circuit_management(
        model, train_loader, eval_loader,
        criterion, optimizer,
        scheduler=None, device='cuda', checkpointManager=None,
        epochs=6000, log_interval=4,
        # Enhanced circuit management parameters
        circuit_assessment_interval=32,
        circuit_management_config=None,
        enable_circuit_removal=True,
        enable_diversity_protection=True,
        enable_real_testing=True,
        enable_robust_detection=True,
        robust_detection_samples=12,
        # Logging parameters
        enable_wandb_logging=False,
        enable_file_logging=True,
        enable_screen_logging=True,
        log_level="INFO"
):

    # info setup
    if checkpointManager:
        save_dir = Path(checkpointManager.experiment_dir)
    else:
        save_dir = Path("results/enhanced_circuit_management_fixed")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging
    from analysis.core.unified_logger import UnifiedLogger
    experiment_name = f"enhanced_circuits_fixed_{model.get_id() if hasattr(model, 'get_id') else 'transformer'}"
    logger = UnifiedLogger(experiment_name=experiment_name, log_dir=save_dir / "logs",
                           enable_wandb=enable_wandb_logging, enable_file=enable_file_logging,
                           enable_screen=enable_screen_logging, log_level=log_level
                           )

    thresholds = create_standard_thresholds("early_training")
    # info init JSON-safe canonical circuit system
    circuit_system = create_standard_canonical_system(model=model, save_dir=save_dir,
                                                      logger=logger, eval_loader=eval_loader,
                                                      thresholds=thresholds)
    canonical_registry = circuit_system['canonical_registry']
    canonical_detector = circuit_system['canonical_detector']
    circuit_train_param = {
        'num_samples': 12,
        'min_circuits_threshold': 0.200,
        'current_accuracy':  0.0,
        'pre_post_width': 3,      #   info logs pronted: pre_post_width before/after circuit assess
    }
    current_accuracy =  circuit_train_param['current_accuracy']
    pre_post_width = circuit_train_param['pre_post_width']      #   info logs pronted: pre_post_width before/after circuit assess
    num_samples = circuit_train_param['num_samples']
    min_circuits_threshold = circuit_train_param['min_circuits_threshold']

    # info check canonical system fixme check other additions too?
    issues = validate_canonical_system_health(circuit_system)

    # info initialize enhanced circuit quality system
    circuit_config = circuit_management_config or get_default_circuit_config()

    from analysis.core.enhanced_circuit_quality_system import EnhancedCircuitQualitySystem
    quality_system = EnhancedCircuitQualitySystem(
        model=model,
        canonical_registry=canonical_registry,
        save_dir=save_dir / "circuit_quality",
        enable_real_testing=enable_real_testing
    )
    quality_system.config.update(circuit_config)

    logger.info(f"🔧 Enhanced circuit management initialized:")
    logger.info(f"   🔬 Assessment interval: {circuit_assessment_interval} epochs"
                f"  |  Max circuits: {quality_system.config['max_circuits']}")
    logger.info(f"   🔬 Removal enabled: {enable_circuit_removal}"
                f"  | 🧪 Real functional testing: {enable_real_testing}")
    logger.info(f"   🔬 Robust detection: {enable_robust_detection} ({robust_detection_samples} samples)")

    # Training metrics tracking
    # fixme use trainers.utils._get_stats warning and later refer to 'eval_loss', 'eval_accuracy' etc, instead of loss
    training_metrics = {'epoch': [], 'eval_loss': [], 'eval_accuracy': []}
    circuit_management_history = []

    # Enhanced DataFrame logging for circuit analysis
    # from analysis.trainers.train_with_enhanced_canonical_circuits import initialize_circuit_dataframe
    df_circuit = initialize_circuit_dataframe()
    circuit_snapshots = []
    eval_stats = None
    sampling_config = get_sampling_configs(strategy="aggressive", num_samples=num_samples)
    example_sampler = create_fast_circuit_example_sampler(eval_loader=eval_loader,
                                                          strategy_config=sampling_config)

    from analysis.temporal import create_temporal_analysis_system
    circuit_evolution_tracker = circuit_system["evolution_tracker"]
    # fixme change for new analyzer integrated with CircuitEmergenceAnalyzer

    circuit_evolution_analyzer = create_temporal_emergence_analyzer(
        evolution_tracker=circuit_system["evolution_tracker"],
        enhanced_registry=circuit_system["enhanced_registry"],
        storage_dir=save_dir / "temporal_analysis"
    )

    # fixme no references as yet
    # temporal_analysis_system = create_temporal_analysis_system(save_dir / "temporal_evolution")

    # Main training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_stats = train_epoch(model, train_loader, criterion, optimizer, epoch, device, scheduler)

        # info evaluation   #   fixme Always? replace with utility function with width as parameter
        should_log_metrics = _should_log_metrics(epoch, circuit_assessment_interval, pre_post_width)
        should_assess_circuits = _should_assess_circuits(epoch, current_accuracy,
                                                         min_circuits_threshold, circuit_assessment_interval)
        should_evaluate = (epoch % log_interval) == 0
        if should_log_metrics or should_assess_circuits or should_evaluate:
            eval_stats = evaluate(model, eval_loader, criterion, device)
            training_metrics_log = _get_stats(epoch, train_stats, eval_stats, optimizer)
            current_accuracy = training_metrics_log['eval_accuracy']
            should_assess_circuits = _should_assess_circuits(epoch, current_accuracy,
                                                             min_circuits_threshold, circuit_assessment_interval)
            # Update training metrics history
            training_metrics['epoch'].append(training_metrics_log['epoch'])
            training_metrics['eval_loss'].append(training_metrics_log['eval_loss'])
            training_metrics['eval_accuracy'].append(training_metrics_log['eval_accuracy'])

            # Log basic training metrics
            if should_log_metrics:
                logger.log_metrics(training_metrics_log, step=epoch, category="training") if should_log_metrics else None

                # info detect grokking  todo fixme Is it needed here? warning detect_grokking is very simple as yet
                grokking_detected = detect_grokking(model, epoch, train_stats, eval_stats)

        # 🔬 ENHANCED CIRCUIT DETECTION AND MANAGEMENT
        if should_assess_circuits:
            # logger.info(f"🔬 Enhanced circuit management @ epoch {epoch}")
            canonical_detector.update_thresholds_for_epoch(epoch=epoch, model_accuracy=current_accuracy)

            # 🆕 info step 1: ROBUST MULTI-EXAMPLE CIRCUIT DETECTION
            if enable_robust_detection:     #   info detect component based patterns fixme or more?
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
                logger.info(f"   🔗  {robust_metrics['robust_copy_patterns']} copy"
                            f" |  📝 {robust_metrics['robust_induction_patterns']} induction"
                            f" |  📝 Registered {registered} |  📝 Aggregated {aggregated}"
                            f" |  📝 New {registered - aggregated} |  📝 Total {total_circuits} circuits")

            # info step 2: COMPREHENSIVE QUALITY ASSESSMENT AND MANAGEMENT
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

                # info enhanced logging
                logger.info(f"   📊  Circuit population: {circuit_metrics['total_circuits']} total"
                            f" | 📈  Quality: μ={circuit_metrics['mean_quality']:.3f}, "
                            f"diversity={circuit_metrics['diversity_score']:.3f}")
                removed_protected = ""
                if circuit_metrics['circuits_removed'] > 0:
                    removed_protected = removed_protected + f"   🗑️  Removed: {circuit_metrics['circuits_removed']} circuits"
                if circuit_metrics['circuits_protected'] > 0:
                    removed_protected = removed_protected + f"   🛡️  Protected: {circuit_metrics['circuits_protected']} circuits for diversity"
                if len(removed_protected) > 0:
                    logger.info(removed_protected)

                # 📊 info save in a dataframe
                df_circuit, circuit_snapshot = log_circuits_to_dataframe(
                    quality_system, management_results, epoch, df_circuit, eval_loader
                )
                circuit_snapshots.append(circuit_snapshot)
                # info perform analysis of circuits found
                #  fixme need to implement some analyze_<<type>>_circuits
                #   warning see examples.temporal_emergence_analysis_example.py

                # info analyze circuits with the temporal circuit evolution analyzer
                if epoch > 600:
                    circuit_evolution_analyzer._analyze_circuit_lifetimes()
                    circuit_evolution_analyzer.analyze_dependency_chains()
                    circuit_evolution_analyzer._detect_grokking_epochs(training_metrics['eval_accuracy'])
                    circuit_evolution_analyzer._find_dependency_chains()

                    # info do the complete evolution analysis
                    depend_chain = circuit_evolution_analyzer.analyze_dependency_chains()
                    grok_trans = circuit_evolution_analyzer.analyze_grokking_transitions(training_metrics['eval_accuracy'])
                    emerg_casc = circuit_evolution_analyzer.detect_emergence_cascades()
                    temp_patt = circuit_evolution_analyzer.detect_temporal_patterns()
                    # temp_emerg_report = circuit_evolution_analyzer.generate_temporal_emergence_report()
                    emerg_order = circuit_evolution_analyzer.track_emergence_order()

                    circ_depend = circuit_evolution_analyzer.analyze_circuit_dependencies()
                    depend_order = circuit_evolution_analyzer.track_emergence_order()
                    emerg_patt = circuit_evolution_analyzer.analyze_emergence_patterns()
                    interact_patt = circuit_evolution_analyzer.analyze_interaction_patterns()
                    learn_phase_trans = circuit_evolution_analyzer.analyze_learning_phase_transitions()
                    # compreh_report = circuit_evolution_analyzer.generate_comprehensive_report()
                    # research_rep = circuit_evolution_analyzer.generate_research_report()








        # info periodic saves
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

    # info final analysis
    logger.info("🎉 Training completed. Generating final circuit analysis...")

    final_management_results = quality_system.assess_and_manage_circuits(eval_loader, epochs)
    final_registry_stats = canonical_registry.get_registry_summary()

    logger.info(f"📊 Final circuit statistics:")
    logger.info(f"   Total circuits discovered: {final_registry_stats['total_canonical_circuits']}")
    logger.info(f"   Final circuit population: {final_management_results['circuit_management']['total_circuits']}")
    logger.info(f"   Aggregation efficiency: {final_registry_stats['aggregation_rate']:.2%}")
    logger.info(f"   Mean circuit quality: {final_management_results['quality_statistics']['mean_quality']:.3f}")

    # 📊 info FINAL DATAFRAME SAVE
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

    # report = circuit_evolution_analyzer.generate_research_report()

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
