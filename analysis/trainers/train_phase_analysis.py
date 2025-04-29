from pathlib import Path

import torch

from analysis.analyzers.continuous_circuit_tracker import ContinuousCircuitTracker
from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
from analysis.analyzers.phase_transition_analyzer import PhaseTransitionAnalyzer
from analysis.utils.utils import init_train_dataloader_state


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
    print(f"Starting training with phase transition analysis for {epochs} epochs...")
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
    print("\nLearning Phase Summary:")
    for insight in phase_summary.get('insights', []):
        print(f" - {insight}")

    # info perform PCA/SVD analysis across learning phases
    print("\nPerforming PCA and SVD analysis across learning phases...")
    phase_weight_analysis = weight_tracker.analyze_phase_weight_spaces(
        phase_analyzer=phase_analyzer,
        eval_loader=eval_loader
    )

    if phase_weight_analysis:
        print("PCA/SVD analysis complete. Key findings:")

        # info report on transitions
        if phase_weight_analysis['transitions']:
            print(f"- Analyzed {len(phase_weight_analysis['transitions'])} phase transitions")

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
                print(f"  Highest weight matrix condition number ({highest_cond:.1f}) at transition "
                      f"epoch {highest_cond_epoch} in layer {highest_layer}")

        # info report on grokking correlation with weight space changes
        if phase_weight_analysis['grokking_points']:
            print(f"- Analyzed {len(phase_weight_analysis['grokking_points'])} grokking points")

            # info check for functional changes at grokking
            if phase_weight_analysis['functional_analysis']:
                for grok_id, grok_data in phase_weight_analysis['grokking_points'].items():
                    grok_epoch = grok_data['epoch']
                    if grok_epoch in phase_weight_analysis['functional_analysis']:
                        func_data = phase_weight_analysis['functional_analysis'][grok_epoch]
                        print(f"  Grokking at epoch {grok_epoch}: "
                              f"Accuracy={func_data['accuracy']:.4f}, "
                              f"Attn Entropy={func_data['attention_entropy_avg']:.4f}")

        # info report on phases
        if phase_weight_analysis['phases']:
            print(f"- Analyzed {len(phase_weight_analysis['phases'])} distinct learning phases")

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
                    print(f"  Largest weight norm change ({max_change_ratio * 100:.1f}%) between "
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

    print(f"PCA/SVD analysis saved to {save_dir / 'phase_weight_analysis.json'}")
    print("Visualizations saved to the weight tracking directory")

    # info additional final timeline visualization
    weight_tracker.visualize_jumps_timeline()

    # info return the original objects plus the weight analysis
    return model, weight_tracker, phase_analyzer, phase_weight_analysis


def train_epoch(model, train_loader, criterion, optimizer, epoch, device):
    """Run a single training epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss.item() * targets.size(0)
        predicted = outputs.argmax(dim=-1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    # Calculate epoch metrics
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def evaluate(model, eval_loader, criterion, device):
    """Evaluate the model on the provided data"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in eval_loader:
            # Move to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update statistics
            total_loss += loss.item() * targets.size(0)
            predicted = outputs.argmax(dim=-1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Calculate metrics
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def log_metrics(model, epoch, train_stats, eval_stats):
    """Log training and evaluation metrics"""
    # Use model logger if available
    if hasattr(model, 'logger'):
        logger = model.logger

        # Log training metrics
        if train_stats:
            logger.log_data('training', 'epoch', epoch)
            logger.log_data('training', 'loss', train_stats['loss'])
            logger.log_data('training', 'accuracy', train_stats['accuracy'])

        # Log evaluation metrics
        if eval_stats:
            logger.log_data('evaluation', 'epoch', epoch)
            logger.log_data('evaluation', 'loss', eval_stats['loss'])
            logger.log_data('evaluation', 'accuracy', eval_stats['accuracy'])

    # Print metrics
    print(f"Epoch  {epoch:5d}: "
          f"\tTrain Loss={train_stats['loss']:.4g}, "
          f"\tAcc={train_stats['accuracy']:.3f}, "
          f"\t\tVal Loss={eval_stats['loss']:.4g}, "
          f"\tAcc={eval_stats['accuracy']:.3f}")


def detect_grokking(model, epoch, train_stats, eval_stats):
    """Detect if grokking is occurring at this epoch"""
    # Skip if model has no logger or if stats are missing
    if not hasattr(model, 'logger') or not train_stats or not eval_stats:
        return False

    logger = model.logger

    # Check for grokking conditions:
    # 1. Training accuracy is high (memorization)
    # 2. Evaluation accuracy is rapidly improving

    # Check if we have enough history
    if logger.get_length('evaluation', 'accuracy') >= 5:
        # Get recent history
        recent_eval_accs = logger.logs['evaluation']['accuracy'][-5:]

        # Check if training accuracy is high
        train_high = train_stats['accuracy'] > 0.9

        # Check if evaluation accuracy is improving rapidly
        prev_eval_accs = recent_eval_accs[:-1]  # All but the latest
        prev_avg = sum(prev_eval_accs) / len(prev_eval_accs) if prev_eval_accs else 0

        significant_improvement = (eval_stats['accuracy'] > prev_avg * 1.2)

        # Detect potential grokking
        if train_high and significant_improvement:
            print(f"\nPotential grokking detected at epoch {epoch}")

            # Log the grokking point
            logger.log_data('grokking_phases', 'grokking_step', epoch)
            return True

    return False


def process_jumps(model, weight_tracker, eval_loader, criterion, optimizer):
    """Process pending jumps detected by the weight tracker"""
    # Get a batch of data for analysis
    sample_inputs, sample_targets = next(iter(eval_loader))
    sample_inputs = sample_inputs.to(next(model.parameters()).device)
    sample_targets = sample_targets.to(next(model.parameters()).device)

    jump_results = weight_tracker.analyze_pending_jumps(
        inputs=sample_inputs,
        targets=sample_targets,
        criterion=criterion,
        optimizer=optimizer,
        jump_analyzer=None,  # We'll handle this separately
        eval_loader=eval_loader,
        mini_train_steps=weight_tracker.sliding_window_size - 1,
    )
    # Process jumps

    # Print summary of processed jumps
    if jump_results:
        print(f"\nProcessed {len(jump_results)} weight space jumps:")
        for result in jump_results:
            jump_epoch = result['jump_epoch']
            jump_char = result['characterization']

            print(f" - Jump at epoch {jump_epoch}: "
                  f"Magnitude={jump_char['total_magnitude']['pre_to_jump']:.4f}, "
                  f"Top layers: {', '.join(jump_char['top_layers'][:2])}, "
                  f"Top heads: {', '.join(jump_char['top_heads'][:2])}")

        # Visualize jump timeline
        weight_tracker.visualize_jumps_timeline()

    return jump_results


'''
# Example usage:
if __name__ == "__main__":
    # This would be called in your main training script
    import torch
    import torch.nn as nn
    from torch import optim

    # Create model, loaders, etc.
    model = YourTransformerModel()
    train_loader, eval_loader, dataset_split_indices = create_data_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None

    # Create checkpoint manager
    checkpointManager = YourCheckpointManager()

    # Train with phase analysis
    model, weight_tracker, phase_analyzer = train_with_phase_analysis(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        dataset_split_indices=dataset_split_indices,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=10000,
        device='cuda',
        checkpointManager=checkpointManager,
        log_interval=10,
        analyze_interval=50,
        checkpoint_interval=200
    )

    # Get learning phase summary
    summary = phase_analyzer.get_learning_phase_summary()

    # Print insights
    print("\nLearning Phase Insights:")
    for insight in summary['insights']:
        print(f" - {insight}")
'''
