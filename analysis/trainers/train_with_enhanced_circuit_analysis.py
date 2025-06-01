# analysis/trainers/train_with_enhanced_circuit_analysis.py (UPDATED with UnifiedLogger)
import torch
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Enhanced infrastructure imports
from analysis.core import EnhancedCircuitRegistry, CircuitThresholds, ComputationalBudget
from analysis.core.unified_logger import UnifiedLogger  # ✅ ADD: UnifiedLogger import
from analysis.helpers.analysis_summary import create_final_analysis_summary
from analysis.helpers.validation_helpers import save_enhanced_checkpoint
from analysis.helpers.missing_functions import run_circuit_validation
from analysis.helpers.circuit_analysis import analyze_circuit_emergence, analyze_circuit_relationships
from analysis.validation import CircuitManipulationValidator
from analysis.analyzers.adaptive_token_operations import AdaptiveTokenOperationDetector

# Existing imports (unchanged)
from analysis.analyzers.integrated_token_discovery import IntegratedTokenCircuitDiscovery
from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
from analysis.analyzers.continuous_circuit_tracker import ContinuousCircuitTracker
from analysis.trainers.utils import evaluate, log_metrics, train_epoch, detect_grokking


def train_with_enhanced_circuit_analysis(
        model, train_loader, eval_loader, dataset_split_indices, criterion, optimizer,
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
        log_level="INFO"
):
    """Enhanced training with adaptive circuit detection, temporal tracking, and unified logging"""

    print(f"🚀 Starting enhanced circuit analysis training (Week 2 + UnifiedLogger)")

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

    token_discovery = IntegratedTokenCircuitDiscovery(
        model=model, save_dir=save_dir / "token_circuits",
        logger=shared_logger, circuit_registry=registry,
        circuit_tracker=circuit_tracker, weight_tracker=weight_tracker
    )

    # NEW: Adaptive token detector
    adaptive_detector = None
    if enable_adaptive_detection:
        adaptive_detector = AdaptiveTokenOperationDetector(
            model=model, registry=registry, thresholds=thresholds
        )
        logger.info("🔍 Adaptive token detection enabled")

    logger.info("📋 All analyzers initialized successfully")

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
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            logger.log_metrics(training_metrics, step=epoch, category="training")

            log_metrics(model, epoch, train_stats, eval_stats)
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
                logger.debug("🔍 Running adaptive token detection...")

                # Get sample batch for analysis
                sample_batch = next(iter(eval_loader))
                inputs, targets = sample_batch

                # Run adaptive detection
                adaptive_results = run_adaptive_token_analysis(
                    adaptive_detector, inputs, targets, epoch, epochs, current_accuracy
                )

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

                logger.info(
                    f"    ✅ Found {adaptive_metrics['stable_circuits']} stable circuits (rate: {adaptive_metrics['stability_rate']:.2%})")

            # =================================================================
            # EXISTING ANALYSIS (Enhanced with new registry) WITH LOGGING
            # =================================================================

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

            # Existing component analysis
            if budget.can_run_method("component_analysis"):
                component_start = time.time()

                component_results = circuit_tracker.sample_circuits(
                    epoch=epoch, eval_loader=eval_loader, baseline_acc=current_accuracy
                )

                epoch_results["component_results"] = component_results

                component_time = time.time() - component_start
                budget.record_execution_time("component_analysis", component_time)

                # ✅ LOG: Component analysis results
                component_metrics = {
                    "component_circuits": len(component_results.get('circuits', [])) if component_results else 0,
                    "component_analysis_time": component_time
                }
                logger.log_metrics(component_metrics, step=epoch, category="component_analysis")

            # =================================================================
            # CIRCUIT VALIDATION (Optional) WITH LOGGING
            # =================================================================

            should_validate = (enable_circuit_validation and validator and
                               epoch % validation_interval == 0 and
                               budget.can_run_method("validation"))

            if should_validate:
                validation_start = time.time()
                logger.info("🧪 Running circuit validation...")

                validation_results = run_circuit_validation(registry, validator, epoch)
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
                emergence_analysis = analyze_circuit_emergence(registry, epoch)
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
            save_enhanced_checkpoint(checkpointManager, epoch, registry, weight_tracker,
                                     train_stats, eval_stats)

        # Save registry periodically
        if epoch % 50 == 0:
            registry.save()

        # ✅ LOG: Epoch timing
        epoch_time = time.time() - epoch_start_time
        logger.log_metrics({"epoch_time": epoch_time}, step=epoch, category="timing")

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


# Helper functions for enhanced training (unchanged from Week 2)
def run_adaptive_token_analysis(detector, inputs, targets, epoch, total_epochs, accuracy):
    """Run adaptive token detection analysis"""
    # Convert inputs to tokens (simplified for modular arithmetic)
    tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]

    # Forward pass with attention storage
    outputs = detector.model(inputs, store_attention=True)
    attention_patterns = detector.model.get_attention_patterns()

    # Adaptive copy detection
    copy_mechanisms = detector.detect_copy_mechanisms_adaptive(
        attention_patterns=attention_patterns,
        tokens=tokens,
        epoch=epoch,
        total_epochs=total_epochs,
        model_accuracy=accuracy,
        content_aware=True
    )

    # Prune unstable circuits
    stable_circuits = detector.prune_unstable_circuits(copy_mechanisms, epoch)

    return {
        "copy_mechanisms": copy_mechanisms,
        "stable_circuits": stable_circuits,
        "detection_summary": {
            "total_detected": len(copy_mechanisms),
            "stable_count": len(stable_circuits),
            "stability_rate": len(stable_circuits) / max(1, len(copy_mechanisms))
        }
    }