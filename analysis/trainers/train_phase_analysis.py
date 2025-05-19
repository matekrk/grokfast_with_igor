from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from analysis.analyzers.continuous_circuit_tracker import ContinuousCircuitTracker
from analysis.analyzers.enhanced_phase_analyzer import EnhancedPhaseAnalyzer
from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
from analysis.analyzers.phase_transition_analyzer import PhaseTransitionAnalyzer
from analysis.trainers.utils import evaluate, log_metrics, train_epoch, detect_grokking, process_jumps
from analysis.utils.utils import init_train_dataloader_state, debug_grokking_step
from analysis.utils.memory_manager import MemoryManager



def train_with_phase_analysis(model, train_loader, eval_loader,
                              dataset_split_indices, criterion, optimizer,
                              scheduler=None, epochs=10000, device='cuda',
                              checkpointManager: object = None, log_interval=10,
                              analyze_interval=50, checkpoint_interval=200,
                              jump_detection_threshold=1.5):
    """
    Extended training function that incorporates phase transition analysis.
    This function builds on train_with_enhanced_analysis to add detailed
    phase transition detection and characterization.

    Args:
        model: The model to train
        train_loader: Training data loader
        eval_loader: Evaluation data loader
        dataset_split_indices: Dataset split indices for reproducibility
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        epochs: Number of training epochs
        device: Device to train on
        checkpointManager: Optional checkpoint manager
        log_interval: How often to log metrics
        analyze_interval: How often to run analysis
        checkpoint_interval: How often to save checkpoints
        jump_detection_threshold: Threshold for jump detection

    Returns:
        tuple: (model, weight_tracker, phase_analyzer)
    """
    print(f"\tStarting training with phase transition analysis for {epochs} epochs...")
    # info set up save directories
    save_dir = Path(checkpointManager.experiment_dir) if checkpointManager else Path("results/phase_analysis")
    save_dir.mkdir(exist_ok=True, parents=True)
    train_dataloader_state = init_train_dataloader_state(dataloader=train_loader)
    eval_dataloader_state = init_train_dataloader_state(dataloader=eval_loader)

    # info set frequency with which to track circuits
    circuit_tracker_freq = 20

    # info initialize analysis trackers  # fixme move to parent method?
    sliding_window_size = 20
    weight_tracker = EnhancedWeightSpaceTracker(
        model=model,
        save_dir=save_dir / "weight_tracking",
        logger=model.logger if hasattr(model, 'logger') else None,
        jump_detection_window=100,
        snapshot_freq=analyze_interval // 2,  # More frequent snapshots for better resolution
        sliding_window_size=sliding_window_size,
        dense_sampling=True,
        jump_threshold=jump_detection_threshold
    )

    # info initialize circuit tracker (sampling every circuit_tracker_freq epochs)
    # info set history length to have all possible grokking/transition points
    min_epoch_for_detection = 100
    history_length = (epochs - min_epoch_for_detection) // circuit_tracker_freq
    circuit_tracker = ContinuousCircuitTracker(
        model=model,
        save_dir=save_dir / "circuit_tracking",
        logger=model.logger if hasattr(model, 'logger') else None,
        sampling_freq=circuit_tracker_freq,
        history_length=history_length
    )

    # info initialize phase transition analyzer
    phase_analyzer = PhaseTransitionAnalyzer(
        model=model,
        save_dir=save_dir / "phase_analysis",
        logger=model.logger if hasattr(model, 'logger') else None,
        circuit_tracker=circuit_tracker,
        weight_tracker=weight_tracker
    )

    # info take initial snapshot and sample
    weight_tracker.take_snapshot(epoch=0, force=True)

    # info sample initial circuit state if eval_loader is provided
    if eval_loader is not None:
        initial_eval_stats = evaluate(
            model=model,
            eval_loader=eval_loader,
            criterion=criterion,
            device=device
        )
        baseline_acc = initial_eval_stats['accuracy']

        # Log initial metrics
        log_metrics(model, 0, {'loss': 0, 'accuracy': 0}, initial_eval_stats)

        # Only for epoch 0, we handle the initial circuit sampling here
        # to avoid duplicate sampling later
        circuit_tracker.sample_circuits(
            epoch=0,
            eval_loader=eval_loader,
            baseline_acc=baseline_acc
        )

    # info training loop
    for epoch in range(epochs):
        # 1. Train for one epoch
        train_stats = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device
        )

        # info 2. Step scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # info 3. Evaluate periodically
        should_evaluate = epoch % log_interval == 0 or epoch == epochs - 1
        eval_stats = None

        if should_evaluate and eval_loader is not None:
            eval_stats = evaluate(
                model=model,
                eval_loader=eval_loader,
                criterion=criterion,
                device=device
            )

            # info log metrics
            log_metrics(model, epoch, train_stats, eval_stats)

            # info calculate baseline accuracy for analysis
            baseline_acc = eval_stats['accuracy']

            # info detect grokking if performance improves significantly
            detect_grokking(model, epoch, train_stats, eval_stats)

        # info 4. Run weight space tracking
        took_snapshot = weight_tracker.take_snapshot(epoch=epoch)

        # info 5. Run detailed analysis periodically
        should_analyze = epoch % analyze_interval == 0 or epoch == epochs - 1

        if should_analyze and eval_loader is not None:
            # info ensure we have eval_stats
            if eval_stats is None:
                eval_stats = evaluate(
                    model=model,
                    eval_loader=eval_loader,
                    criterion=criterion,
                    device=device
                )
                baseline_acc = eval_stats['accuracy']
            else:
                baseline_acc = eval_stats['accuracy']

            # info run phase transition analysis - this will internally sample circuits when needed
            #  let the phase_analyzer handle sampling to avoid duplication
            phase_analyzer.analyze(
                epoch=epoch,
                eval_loader=eval_loader,
                baseline_acc=baseline_acc
            )

            # info process any pending jumps
            if weight_tracker.pending_jumps:
                jump_results = process_jumps(
                    model=model,
                    weight_tracker=weight_tracker,
                    eval_loader=eval_loader,
                    criterion=criterion,
                    optimizer=optimizer
                )

                # info correlate jumps with phase transitions
                phase_analyzer.analyze_with_weight_space_jumps(
                    jump_results=jump_results,
                    eval_loader=eval_loader
                )
            # info save logs on regular frequency basis
            checkpointManager.save_logger_data(epoch=epoch, logger=model.logger)

        # info 6. Save checkpoint periodically
        if checkpointManager and (epoch % checkpoint_interval == 0 or epoch == epochs - 1):
            # info always save stats with checkpoint
            if eval_stats is None and eval_loader is not None:
                eval_stats = evaluate(
                    model=model,
                    eval_loader=eval_loader,
                    criterion=criterion,
                    device=device
                )

            # info add phase analysis data to checkpoint
            extra_data = {
                'phase_transitions': phase_analyzer.detected_transitions,
                'weight_space_jumps': [j['epoch'] for j in weight_tracker.detected_jumps]
            }

            checkpointManager.save_checkpoint(
                epoch=epoch,
                train_dataloader_state=train_dataloader_state,
                eval_dataloader_state=eval_dataloader_state,
                dataset_split_indices=dataset_split_indices,
                train_loss=train_stats['loss'] if train_stats else 1.e6,
                train_accuracy=train_stats['accuracy'] if train_stats else 0.0,
                val_loss=eval_stats['loss'] if eval_stats else 1.e6,
                val_accuracy=eval_stats['accuracy'] if eval_stats else 0.0,
                extra_data=extra_data,
                force_save=False,
            )

    # info final phase summary
    phase_summary = phase_analyzer.get_learning_phase_summary()
    print("\tLearning Phase Summary:")
    for insight in phase_summary.get('insights', []):
        print(f"\t - {insight}")

    # info perform PCA/SVD analysis across learning phases
    print("\tPerforming PCA and SVD analysis across learning phases...")
    phase_weight_analysis = weight_tracker.analyze_phase_weight_spaces(
        phase_analyzer=phase_analyzer,
        eval_loader=eval_loader
    )

    if phase_weight_analysis:
        print("\tPCA/SVD analysis complete. Key findings:")

        # info report on transitions
        if phase_weight_analysis['transitions']:
            print(f"\t- Analyzed {len(phase_weight_analysis['transitions'])} phase transitions")

            # info find transition with highest SVD condition number
            highest_cond = 0
            highest_cond_epoch = None
            highest_layer = None

            for epoch, data in phase_weight_analysis['transitions'].items():
                for layer_name, svd_results in data['svd_analysis'].items():
                    cond = svd_results.get('condition_number', 0)
                    if cond > highest_cond:
                        highest_cond = cond
                        highest_cond_epoch = epoch
                        highest_layer = layer_name

            if highest_cond_epoch:
                print(f"\tHighest weight matrix condition number ({highest_cond:.1f}) at transition "
                      f"epoch {highest_cond_epoch} in layer {highest_layer}")

        # info report on grokking correlation with weight space changes
        if phase_weight_analysis['grokking_points']:
            print(f"\tAnalyzed {len(phase_weight_analysis['grokking_points'])} grokking points")

            # info check for functional changes at grokking
            if phase_weight_analysis['functional_analysis']:
                for grok_id, grok_data in phase_weight_analysis['grokking_points'].items():
                    grok_epoch = grok_data['epoch']
                    if grok_epoch in phase_weight_analysis['functional_analysis']:
                        func_data = phase_weight_analysis['functional_analysis'][grok_epoch]
                        print(f"\tGrokking at epoch {grok_epoch}: "
                              f"Accuracy={func_data['accuracy']:.4f}, "
                              f"Attn Entropy={func_data['attention_entropy_avg']:.4f}")

        # info report on phases
        if phase_weight_analysis['phases']:
            print(f"\tAnalyzed {len(phase_weight_analysis['phases'])} distinct learning phases")

            # info find phase with most dramatic weight norm changes
            phase_norms = {}
            for phase_id, data in phase_weight_analysis['phases'].items():
                total_norm = sum(layer['total_norm'] for layer in data['layer_weight_norms'].values())
                phase_norms[phase_id] = total_norm

            if len(phase_norms) >= 2:
                phase_ids = sorted(phase_norms.keys())
                max_change_ratio = 0
                max_change_phases = (None, None)

                for i in range(len(phase_ids) - 1):
                    curr_phase = phase_ids[i]
                    next_phase = phase_ids[i + 1]

                    curr_norm = phase_norms[curr_phase]
                    next_norm = phase_norms[next_phase]

                    if curr_norm > 0:
                        change_ratio = abs(next_norm - curr_norm) / curr_norm
                        if change_ratio > max_change_ratio:
                            max_change_ratio = change_ratio
                            max_change_phases = (curr_phase, next_phase)

                if max_change_phases[0]:
                    print(f"\tLargest weight norm change ({max_change_ratio * 100:.1f}%) between "
                          f"{max_change_phases[0]} and {max_change_phases[1]}")

    # info save the analysis to a file for later reference
    import json

    # info sonvert non-serializable parts to strings
    simplified_analysis = {}
    for key_type, data in phase_weight_analysis.items():
        simplified_analysis[key_type] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                # info keep most of the structure but simplify complex parts
                cleaned_v = {}
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float, str, list, bool)) or v2 is None:
                        cleaned_v[k2] = v2
                    else:
                        # info convert non-serializable values to string representation
                        cleaned_v[k2] = str(v2)
                simplified_analysis[key_type][k] = cleaned_v
            else:
                simplified_analysis[key_type][k] = str(v)

    with open(save_dir / "phase_weight_analysis.json", "w") as f:
        json.dump(simplified_analysis, f, indent=2)

    print(f"\tPCA/SVD analysis saved to {save_dir / 'phase_weight_analysis.json'}")
    print("\tVisualizations saved to the weight tracking directory")

    # info additional final timeline visualization
    weight_tracker.visualize_jumps_timeline()

    # info return the original objects plus the weight analysis
    return model, weight_tracker, phase_analyzer, phase_weight_analysis


def train_with_enhanced_phase_analysis(model, train_loader, eval_loader,
                                       dataset_split_indices, criterion, optimizer,
                                       scheduler=None, epochs=10000, device='cuda',
                                       checkpointManager=None, log_interval=10,
                                       analyze_interval=50, checkpoint_interval=200,
                                       jump_detection_threshold=1.5,
                                       circuit_sampling_freq=20,
                                       sparsity_threshold=0.1):
    """
    Extended training function that incorporates enhanced phase transition analysis.
    This function builds on train_with_phase_analysis to add detailed circuit-class
    relationships, MLP sparsity patterns, and attention-MLP interactions.

    Args:
        model: The model to train
        train_loader: Training data loader
        eval_loader: Evaluation data loader
        dataset_split_indices: Dataset split indices for reproducibility
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        epochs: Number of training epochs
        device: Device to train on
        checkpointManager: Optional checkpoint manager
        log_interval: How often to log metrics
        analyze_interval: How often to run analysis
        checkpoint_interval: How often to save checkpoints
        jump_detection_threshold: Threshold for jump detection
        circuit_sampling_freq: How often to sample circuits
        sparsity_threshold: Threshold for neuron activation

    Returns:
        tuple: (model, weight_tracker, enhanced_phase_analyzer, phase_weight_analysis)
    """

    # info add adaptive analysis frequency based on memory pressure
    def get_memory_adaptive_interval(base_interval, epoch):
        """Dynamically adjust analysis interval based on memory usage and epoch"""
        if torch.cuda.is_available():
            # info get available GPU memory
            available_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            memory_pressure = 1 - (available_mem / torch.cuda.get_device_properties(0).total_memory)

            # info increase interval under high memory pressure
            if memory_pressure > 0.8:
                return base_interval * 3
            elif memory_pressure > 0.6:
                return base_interval * 2
            else:
                return base_interval
        else:
            # info for CPU, increase interval at later epochs when we likely have more history
            return base_interval * (1 + epoch // 1000)

    print(f"\tStarting training with enhanced phase transition analysis for {epochs} epochs...")

    # Set up save directories
    save_dir = Path(checkpointManager.experiment_dir) if checkpointManager else Path("results/enhanced_phase_analysis")
    save_dir.mkdir(exist_ok=True, parents=True)

    train_dataloader_state = init_train_dataloader_state(dataloader=train_loader)
    eval_dataloader_state = init_train_dataloader_state(dataloader=eval_loader)

    # Initialize analysis trackers
    sliding_window_size = 20
    weight_tracker = EnhancedWeightSpaceTracker(
        model=model,
        save_dir=save_dir / "weight_tracking",
        logger=model.logger if hasattr(model, 'logger') else None,
        jump_detection_window=100,
        snapshot_freq=analyze_interval // 2,  # More frequent snapshots for better resolution
        sliding_window_size=sliding_window_size,
        dense_sampling=True,
        jump_threshold=jump_detection_threshold
    )

    # Initialize circuit tracker with specified sampling frequency
    min_epoch_for_detection = 100
    history_length = (epochs - min_epoch_for_detection) // circuit_sampling_freq
    circuit_tracker = ContinuousCircuitTracker(
        model=model,
        save_dir=save_dir / "circuit_tracking",
        logger=model.logger if hasattr(model, 'logger') else None,
        sampling_freq=circuit_sampling_freq,
        history_length=history_length
    )

    # Initialize the enhanced phase analyzer
    enhanced_analyzer = EnhancedPhaseAnalyzer(
        model=model,
        save_dir=save_dir / "enhanced_phase_analysis",
        logger=model.logger if hasattr(model, 'logger') else None,
        circuit_tracker=circuit_tracker,
        weight_tracker=weight_tracker
    )

    # Configure sparsity threshold and analysis intervals
    enhanced_analyzer.mlp_sparsity_tracker.activation_threshold = sparsity_threshold
    enhanced_analyzer.analyze_interval = analyze_interval

    # info init a memory manager #
    #  todo implement a check_memory for non-cuda (?)
    memory_manager = MemoryManager(model=model)
    memory_manager.register_analyzer(enhanced_analyzer)
    memory_manager.register_analyzer(weight_tracker)
    memory_manager.register_analyzer(circuit_tracker)

    # Take initial snapshot and sample
    weight_tracker.take_snapshot(epoch=0, force=True)

    # Sample initial circuit state if eval_loader is provided
    if eval_loader is not None:
        initial_eval_stats = evaluate(
            model=model,
            eval_loader=eval_loader,
            criterion=criterion,
            device=device
        )
        baseline_acc = initial_eval_stats['accuracy']

        # Log initial metrics
        log_metrics(model, 0, {'loss': 0, 'accuracy': 0}, initial_eval_stats)

        # Only for epoch 0, handle the initial circuit sampling here
        circuit_tracker.sample_circuits(
            epoch=0,
            eval_loader=eval_loader,
            baseline_acc=baseline_acc
        )

    adaptive_interval_memory = False
    adaptive_interval = analyze_interval
    # Training loop
    for epoch in range(epochs):
        # 1. Train for one epoch
        train_stats = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device
        )

        # 2. Step scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # 3. Evaluate periodically
        should_evaluate = epoch % log_interval == 0 or epoch == epochs - 1
        eval_stats = None

        # if should_evaluate:
        #     print("\t<------ memory_manager.memory_check()")
        #     memory_manager.check_memory(epoch=epoch)

        if should_evaluate and eval_loader is not None:
            eval_stats = evaluate(
                model=model,
                eval_loader=eval_loader,
                criterion=criterion,
                device=device
            )

            # Log metrics
            log_metrics(model, epoch, train_stats, eval_stats)

            # Calculate baseline accuracy for analysis
            baseline_acc = eval_stats['accuracy']

            # Detect grokking if performance improves significantly
            detect_grokking(model, epoch, train_stats, eval_stats)

        # 4. Run weight space tracking
        took_snapshot = weight_tracker.take_snapshot(epoch=epoch)

        # 5. Run enhanced analysis periodically
        if adaptive_interval_memory:
            adaptive_interval = get_memory_adaptive_interval(adaptive_interval, epoch)
            if adaptive_interval != analyze_interval:
                analyze_interval = adaptive_interval
        should_analyze = epoch % analyze_interval == 0 or epoch == epochs - 1

        if epoch % 10 == 0:
            memory_manager.check_memory_adaptive(epoch=epoch)

        if should_analyze and eval_loader is not None:
            # Ensure we have eval_stats
            if eval_stats is None:
                eval_stats = evaluate(
                    model=model,
                    eval_loader=eval_loader,
                    criterion=criterion,
                    device=device
                )
                baseline_acc = eval_stats['accuracy']
            else:
                baseline_acc = eval_stats['accuracy']

            # Run enhanced phase transition analysis
            analysis_results = enhanced_analyzer.analyze(
                epoch=epoch,
                eval_loader=eval_loader,
                baseline_acc=baseline_acc
            )

            # Process any pending jumps
            if weight_tracker.pending_jumps:
                # info there might be some major analysis - perform memory_check
                memory_manager.check_memory_adaptive(epoch=epoch)

                jump_results = process_jumps(
                    model=model,
                    weight_tracker=weight_tracker,
                    eval_loader=eval_loader,
                    criterion=criterion,
                    optimizer=optimizer
                )

                # Correlate jumps with phase transitions
                enhanced_analyzer.analyze_with_weight_space_jumps(
                    jump_results=jump_results,
                    eval_loader=eval_loader
                )

                # Log key insights from jump analysis
                if 'enhanced_analysis' in analysis_results and 'insights' in analysis_results['enhanced_analysis']:
                    insights = analysis_results['enhanced_analysis']['insights']
                    for insight_key, insight_data in insights.items():
                        if 'interpretation' in insight_data:
                            print(f"\tInsight: {insight_key} - {insight_data['interpretation']}")

            # info save logs on regular frequency basis
            checkpointManager.save_logger_data(epoch=epoch, logger=model.logger)



        # 6. Save checkpoint periodically
        if checkpointManager and (epoch % checkpoint_interval == 0 or epoch == epochs - 1):
            # Always save stats with checkpoint
            if eval_stats is None and eval_loader is not None:
                eval_stats = evaluate(
                    model=model,
                    eval_loader=eval_loader,
                    criterion=criterion,
                    device=device
                )

            # Add enhanced analysis data to checkpoint
            extra_data = {
                'phase_transitions': enhanced_analyzer.detected_transitions,
                'weight_space_jumps': [j['epoch'] for j in weight_tracker.detected_jumps],
                'enhanced_analysis_epochs': list(enhanced_analyzer.enhanced_analysis_history.keys())
            }

            checkpointManager.save_checkpoint(
                epoch=epoch,
                train_dataloader_state=train_dataloader_state,
                eval_dataloader_state=eval_dataloader_state,
                dataset_split_indices=dataset_split_indices,
                train_loss=train_stats['loss'] if train_stats else 1.e6,
                train_accuracy=train_stats['accuracy'] if train_stats else 0.0,
                val_loss=eval_stats['loss'] if eval_stats else 1.e6,
                val_accuracy=eval_stats['accuracy'] if eval_stats else 0.0,
                extra_data=extra_data,
                force_save=False,
            )

        #7. info do some visualizations and analysis periodically
            min_pca_svd_epoch = 2500
            if epoch >= min_pca_svd_epoch and epoch % 500 == 0:
                # info perform PCA/SVD analysis across learning phases
                print(f"\tPerforming PCA and SVD analysis across learning phases at epoch {epoch}")
                phase_weight_analysis = weight_tracker.analyze_phase_weight_spaces(
                    phase_analyzer=enhanced_analyzer,
                    eval_loader=eval_loader
                )

            # memory cleanup
            enhanced_analyzer.cleanup()
            weight_tracker.cleanup()
            circuit_tracker.cleanup()

    # info final enhanced summary
    enhanced_summary = enhanced_analyzer.get_enhanced_learning_phase_summary()
    print("\tEnhanced Learning Phase Summary:")

    # info standard insights
    for insight in enhanced_summary.get('insights', []):
        print(f"\t\t{insight}")

    # info enhanced insights
    if 'enhanced_insights' in enhanced_summary:
        print("\tEnhanced Analysis Insights:")
        for insight in enhanced_summary['enhanced_insights']:
            print(f"\t\t{insight}")

    # info perform PCA/SVD analysis across learning phases
    print("\tPerforming PCA and SVD analysis across learning phases...")
    phase_weight_analysis = weight_tracker.analyze_phase_weight_spaces(
        phase_analyzer=enhanced_analyzer,
        eval_loader=eval_loader
    )

    if phase_weight_analysis:
        print("\tPCA/SVD analysis complete. Key findings:")

        # info report on transitions
        if phase_weight_analysis['transitions']:
            print(f"\t\tAnalyzed {len(phase_weight_analysis['transitions'])} phase transitions")

        # info report on grokking correlation
        if phase_weight_analysis['grokking_points']:
            print(f"\t\tAnalyzed {len(phase_weight_analysis['grokking_points'])} grokking points")

        # info report on phases
        if phase_weight_analysis['phases']:
            print(f"\t\tAnalyzed {len(phase_weight_analysis['phases'])} distinct learning phases")

    # info additional visualization of sparsity patterns
    if hasattr(enhanced_analyzer, 'mlp_sparsity_tracker') and enhanced_analyzer.mlp_sparsity_tracker.sparsity_history:
        print("\tGenerating sparsity evolution visualization...")
        sparsity_epochs = sorted(enhanced_analyzer.mlp_sparsity_tracker.sparsity_history.keys())

        if len(sparsity_epochs) >= 2:
            # info create visualization comparing sparsity across phases
            fig, ax = plt.subplots(figsize=(12, 6))

            # info extract layer names
            first_epoch = sparsity_epochs[0]
            layer_names = list(
                enhanced_analyzer.mlp_sparsity_tracker.sparsity_history[first_epoch]['avg_sparsity'].keys())

            # info plot sparsity evolution for each layer
            for layer in layer_names:
                sparsity_values = [
                    enhanced_analyzer.mlp_sparsity_tracker.sparsity_history[e]['avg_sparsity'].get(layer, 0)
                    for e in sparsity_epochs
                ]
                ax.plot(sparsity_epochs, sparsity_values, 'o-',
                        label=layer.replace('layer_', '').replace('_mlp_expanded', ''))

            # info mark phase transitions
            for transition in enhanced_analyzer.detected_transitions:
                ax.axvline(x=transition['epoch'], color='r', linestyle='--', alpha=0.5)
                ax.text(transition['epoch'], 0.95,
                        f"T:{transition['epoch']}",
                        transform=ax.get_xaxis_transform(),
                        rotation=90, va='top')

            # info mark grokking points
            if hasattr(model, 'logger') and 'grokking_phases' in model.logger.logs:
                grokking_step = model.logger.logs['grokking_phases'].get('grokking_step')
                debug_grokking_step(grokking_step, "train_with_enhanced_phase_analysis", model.logger)

                if grokking_step:
                    if isinstance(grokking_step, list):
                        for step in grokking_step:
                            ax.axvline(x=step, color='g', linestyle='-', alpha=0.5)
                            ax.text(step, 0.95,
                                    f"G:{step}",
                                    transform=ax.get_xaxis_transform(),
                                    rotation=90, va='top', color='green')
                    else:
                        ax.axvline(x=grokking_step, color='g', linestyle='-', alpha=0.5)
                        ax.text(grokking_step, 0.95,
                                f"G:{grokking_step}",
                                transform=ax.get_xaxis_transform(),
                                rotation=90, va='top', color='green')

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Sparsity (% inactive neurons)')
            ax.set_title('MLP Neuron Sparsity Evolution')
            ax.legend()

            plt.tight_layout()
            plt.suptitle(f"{model.plot_prefix}")
            plt.savefig(save_dir / "sparsity_evolution_with_phases.png")
            plt.close(fig)

            print(f"\tSparsity visualization saved to {save_dir / 'sparsity_evolution_with_phases.png'}")

    # info additional visualization of attention-MLP interactions
    if hasattr(enhanced_analyzer,
               'interaction_analyzer') and enhanced_analyzer.interaction_analyzer.interaction_history:
        print("\tGenerating attention-MLP interaction visualization...")
        interaction_epochs = sorted(enhanced_analyzer.interaction_analyzer.interaction_history.keys())

        if len(interaction_epochs) >= 2:
            # info create visualization of correlation trends
            fig, ax = plt.subplots(figsize=(12, 6))

            # info extract average correlations over time
            avg_correlations = []
            for epoch in interaction_epochs:
                layer_correlations = enhanced_analyzer.interaction_analyzer.interaction_history[epoch][
                    'layer_correlations']
                avg_correlations.append(np.mean(list(layer_correlations.values())))

            ax.plot(interaction_epochs, avg_correlations, 'o-', color='purple', label='Avg Attention-MLP Correlation')

            # info mark phase transitions
            for transition in enhanced_analyzer.detected_transitions:
                ax.axvline(x=transition['epoch'], color='r', linestyle='--', alpha=0.5)
                ax.text(transition['epoch'], 0.95,
                        f"T:{transition['epoch']}",
                        transform=ax.get_xaxis_transform(),
                        rotation=90, va='top')

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Correlation')
            ax.set_title('Attention-MLP Coordination Evolution')
            ax.legend()

            plt.tight_layout()
            plt.suptitle(f"{model.plot_prefix}")
            plt.savefig(save_dir / "attn_mlp_correlation_with_phases.png")
            plt.close(fig)

            print(f"\tAttention-MLP correlation visualization saved to {save_dir / 'attn_mlp_correlation_with_phases.png'}")

    # info clean up resources
    print("\tCleaning up resources...")
    if hasattr(enhanced_analyzer, 'cleanup'):
        enhanced_analyzer.cleanup()

    # info return the model and analysis components
    return model, weight_tracker, enhanced_analyzer, phase_weight_analysis
