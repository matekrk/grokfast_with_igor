from pathlib import Path

import numpy as np
import torch

from analysis.analyzers.attention_mlp_analyzer import AttentionMLPAnalyzer
# Import the enhanced weight tracker and jump analysis tools
from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
from analysis.analyzers.grokking_detection import analyze_grokking_transitions
from analysis.analyzers.jump_analysis_manager import JumpAnalysisManager
from analysis.analyzers.jump_analysis_tools import JumpAnalysisTools
from analysis.utils.utils import init_train_dataloader_state, FittingScore, debug_grokking_step


def get_force_snapshot(method, epoch, analyze_interval, min_epoch_for_detection, weight_tracker=None):
    force_snapshot, velocity_force_snapshot = False, False
    if epoch < min_epoch_for_detection:
        return force_snapshot, velocity_force_snapshot
    if method == "simple":
        force_snapshot = False
    elif method == "simple_neighbors":
        force_snapshot = epoch > min_epoch_for_detection and (
                ((epoch - 1) % analyze_interval == 0) or
                ((epoch + 1) % analyze_interval == 0)
        )
    elif method == "enhanced_simple_neighbors" and hasattr(weight_tracker, "velocities"):
        force_snapshot = (
                epoch > min_epoch_for_detection and (
                epoch % analyze_interval == 0 or  # Regular interval
                (epoch - 1) % analyze_interval == 0 or  # Just before
                (epoch + 1) % analyze_interval == 0 or  # Just after
                (epoch - 2) % analyze_interval == 0 or  # Two before
                (epoch + 2) % analyze_interval == 0  # Two after
        ))
    elif method == "velocities" and weight_tracker is not None:
        # info increase sampling rate when we notice larger weight changes
        recent_velocities = [vn[2] for vn in weight_tracker.velocities[-5:]]
        avg_velocity = sum(recent_velocities) / len(recent_velocities)
        current_velocity = weight_tracker.velocities[-1][2]
        # info if current velocity is significantly higher than recent average
        threshold_multiplier = 1.15
        force_snapshot = (force_snapshot or
                          (current_velocity > threshold_multiplier * avg_velocity))
        if force_snapshot:
            velocity_force_snapshot = True
    return force_snapshot, velocity_force_snapshot


def perform_additional_analyses(jump_results, jump_analyzer, weight_tracker, eval_loader, criterion, device):
    # info perform additional analyses for each jump
    for result in jump_results:
        jump_epoch = result['jump_epoch']
        print(f"  Additional analysis for jump at epoch {jump_epoch}...")

        attribution_results = jump_analyzer.analyze_head_attribution_around_jump(
            jump_epoch=jump_epoch,
            eval_loader=eval_loader,
            weight_tracker=weight_tracker
        )

        # info analyze attention patterns around the jump
        attention_results = jump_analyzer.analyze_attention_patterns_around_jump(
            jump_epoch=jump_epoch,
            eval_loader=eval_loader,
            weight_tracker=weight_tracker
        )
        # info analyze loss landscape
        sample_inputs, sample_targets = next(iter(eval_loader))
        sample_inputs, sample_targets = sample_inputs.to(device), sample_targets.to(device)

        landscape_results = jump_analyzer.analyze_loss_curvature(
            inputs=sample_inputs,
            targets=sample_targets,
            criterion=criterion
        )


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    train_correct = train_total = 0
    train_loss = 0.0
    for _, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        train_last_token_preds = logits.argmax(dim=-1)
        train_correct += (train_last_token_preds == targets).sum().item()
        train_total += targets.size(0)
        train_loss += loss.item() * targets.size(0)
    model.eval()
    train_accuracy = train_correct / train_total if train_total > 0 else 0.0
    train_loss = train_loss / train_total if train_total > 0 else 0.0

    return train_accuracy, train_loss


def get_sample_inputs_targets(eval_loader, device):
    sample_inputs, sample_targets = next(iter(eval_loader))
    sample_inputs, sample_targets = sample_inputs.to(device), sample_targets.to(device)
    return sample_inputs, sample_targets


def print_jump_analysis_results(jump_epoch, jump_char, significant_jumps):
    print(f"Jump analysis for epoch {jump_epoch}:")
    print(f"\tTotal magnitude:     \t{jump_char['total_magnitude']['pre_to_jump']:.4f}")
    print(f"\tSymmetry ratio:      \t{jump_char['total_magnitude']['symmetry_ratio']:.4f}")
    print(f"\tTop changing layers: \t{', '.join(jump_char['top_layers'])}")
    print(f"\tTop changing heads:  \t{', '.join(jump_char['top_heads'])}")
    print(f"\tsignificant jumps:   \t{significant_jumps}")


def train_with_enhanced_analysis(model, train_loader, eval_loader,
                                 dataset_split_indices,
                                 criterion, optimizer, scheduler,
                                 epochs, device,
                                 checkpointManager,
                                 log_interval=5,
                                 analyze_interval=50,
                                 jump_detection_threshold=2.0,
                                 checkpoint_interval=200):
    """
    Train the model with enhanced weight space tracking and jump analysis

    Parameters:
    -----------
    model : Decoder The transformer model with analysis capabilities
    train_loader : DataLoader Training data loader
    eval_loader : DataLoader Evaluation data loader
    dataset_split_indices : dict Dataset split indices for reproducibility
    criterion : loss function Loss function for training
    optimizer : optimizer Optimizer for training
    scheduler : scheduler Learning rate scheduler
    epochs : int Number of training epochs
    device : torch.device Device to train on
    checkpointManager : CheckpointManager Checkpoint manager instance
    log_interval : int How often to log the basic metrics to the screen (and logger?) (steps)
    analyze_interval : int How often to perform detailed analysis (steps)
    jump_detection_threshold : float Threshold for jump detection (in standard deviations)
    checkpoint_interval : int How often to save checkpoints (steps)
    """
    # info set analyze_interval to 4 * log_interval
    # analyze_interval = 4 * log_interval    # fixme this is somehow wrong, correct

    train_dataloader_state = init_train_dataloader_state(dataloader=train_loader)
    eval_dataloader_state = init_train_dataloader_state(dataloader=eval_loader)
    min_epoch_for_detection = 100
    epochs_to_highlight = []
    grokking_analysis = None
    last_timeline_pca_visualized = -1
    velocity_force_snapshot = False

    # fixme either log_interval = k * snapshot_frequency, or snapshot_frequency = k * log_interval;
    #  which is better? probably snapshot_frequency = k * log_interval, so that snapshots are taken
    #  at regular inetrvals, but not too frequently warning NOOOO! snap_freq = analyze_interv // 2
    snapshot_frequency = 2 * log_interval
    sliding_window_size = 20
    # info initialize the enhanced weight space tracker with sliding window
    weight_tracker = EnhancedWeightSpaceTracker(
        model=model,
        save_dir=checkpointManager.checkpoint_dir,
        logger=model.logger,
        jump_detection_window=100,
        snapshot_freq=snapshot_frequency,
        sliding_window_size=sliding_window_size,  # info keep track recent states for better pre-jump analysis
        dense_sampling=True,  # info sample more frequently for the sliding window fixme ???
        jump_threshold=jump_detection_threshold  # fixme ???
    )

    # info fitting score computation
    fitscore = FittingScore()

    jump_analyzer = JumpAnalysisManager(
        model=model,
        save_dir=checkpointManager.checkpoint_dir,
        logger=model.logger
    )

    force_snapshot = False
    weight_tracker.take_snapshot(epoch=0, force=True)

    for epoch in range(epochs):
        train_dataloader_state['epoch'] = epoch
        train_accuracy, train_loss = train_one_epoch(model=model, train_loader=train_loader,
                                                     optimizer=optimizer, scheduler=scheduler,
                                                     criterion=criterion, device=device)
        model.eval()

        velocity_force_snapshot_last_epoch = velocity_force_snapshot
        (force_snapshot,
         velocity_force_snapshot) = get_force_snapshot(method="velocities", epoch=epoch,
                                                       analyze_interval=analyze_interval,
                                                       min_epoch_for_detection=min_epoch_for_detection,
                                                       weight_tracker=weight_tracker)
        if velocity_force_snapshot and not velocity_force_snapshot_last_epoch:
            # info highlight only if it was not highlighted in the last epoch
            #  since the PCA plot becomes unreadable in that case
            velocity_force_snapshot_last_epoch = False
            epochs_to_highlight.append(epoch)
        # info return true if snapshot was taken for this epoch
        took_snapshot = weight_tracker.take_snapshot(epoch=epoch, force=force_snapshot)

        # info check if we need to log and analyze
        if epoch % log_interval == 0 or took_snapshot:
            # info log training stats
            model.log_stats('training', {'accuracy': train_accuracy, 'loss': train_loss, 'epoch': epoch})

            # info evaluate loss and accuracy on validation set
            model.eval()
            eval_dataloader_state['epoch'] = epoch
            eval_accuracy, eval_loss = model.evaluate(eval_loader)
            model.log_stats('evaluation', {'accuracy': eval_accuracy, 'loss': eval_loss, 'epoch': epoch})
            model.train()

            # info calculate fitting score for monitoring
            fitting_score = fitscore.fitting_score(train_loss=train_loss, train_accu=train_accuracy,
                                                   eval_loss=eval_loss, eval_accu=eval_accuracy)
            took_snps_str = " <<<" if took_snapshot else ""
            print(f"Epoch {epoch:5d}: Train Loss={train_loss:.5g}, Train Acc={train_accuracy:.3f}, "
                  f"Val Loss={eval_loss:.5g}, \tVal Acc={eval_accuracy:.3f}, \tFitting_score={fitting_score:.3f}  \t{took_snps_str}")
            if train_accuracy < 0.9 and epoch >= min_epoch_for_detection:
                print(f"<<<<<<<<<<<<<<< Train Acc {train_accuracy} !!!!!!!")

            # info track metrics for grokking detection  fixme remove this call, at least for some time
            # track_metrics_for_grokking(epoch=epoch, model=model, train_loader=train_loader, eval_loader=eval_loader)

        # info check for pending jumps that need analysis
        if weight_tracker.pending_jumps:
            sample_inputs, sample_targets = get_sample_inputs_targets(eval_loader=eval_loader, device=device)

            # info analyze pending jumps with balanced before/after snapshots
            jump_results = weight_tracker.analyze_pending_jumps(
                inputs=sample_inputs, targets=sample_targets,
                criterion=criterion, optimizer=optimizer,
                jump_analyzer=jump_analyzer,
                eval_loader=eval_loader,
                mini_train_steps=sliding_window_size - 1
            )
            # info deeper attention-2-MLP analysis
            attn_mlp_analyzer = AttentionMLPAnalyzer(
                model=model, eval_loader=eval_loader, device=device,
                save_dir=checkpointManager.checkpoint_dir / "attn_mlp_analysis",
            )
            jump_attn_mlp_analysis = attn_mlp_analyzer.analyze_across_jumps(
                weight_tracker=weight_tracker,
                jump_results=jump_results,
                num_batches=3  # fixme is this a correct number?
            )

            # info process results fixme is it possible to have m,ore than one result in jump_results?
            # info list of significant jumps
            significant_jumps = []
            for result in jump_results:  # fixme perhaps ifany(result in jump_results)???
                jump_epoch, jump_char = result['jump_epoch'], result['characterization']

                # info find circuits if the jump is significant
                if jump_char['total_magnitude']['pre_to_jump'] > 1.0:
                    significant_jumps.append(result)
                    model.log_stats('significant_jumps', {'epoch': jump_epoch,
                                                          'char': jump_char['total_magnitude']['pre_to_jump']
                                                          })
                print_jump_analysis_results(jump_epoch=jump_epoch, jump_char=jump_char,
                                            significant_jumps=significant_jumps)

                # info create jump characterization visualizations
                viz_dir = weight_tracker.visualize_jump_characterization(jump_char)
                print(f"  Visualizations saved to {viz_dir}")

                # info visualize jumps on a timeline and a PCA trajectory
                weight_tracker.visualize_jumps_timeline()
                weight_tracker.visualize_trajectory(
                    selected_dims=[0, 1], highlight_epochs={
                        # 'jumps': [j['epoch'] for j in weight_tracker.detected_jumps],
                        'forced': list(set(epochs_to_highlight))}
                )
                last_timeline_pca_visualized = epoch

            if significant_jumps:
                # info circuit analysis
                circuit_analysis = attn_mlp_analyzer.discover_circuits(
                    weight_tracker=weight_tracker, jump_results=significant_jumps,
                    eval_loader=eval_loader, criterion=criterion,
                    circuit_threshold=0.05
                )
                # info log circuits results
                if circuit_analysis:
                    for jump_epoch, analysis in circuit_analysis.items():
                        if 'sorted_analysis' in circuit_analysis and analysis['sorted_components']:
                            # info log top circuit components
                            top_component = analysis['sorted_analysis'][:3]
                            for i, (component, attributes) in enumerate(top_component):
                                model.log_stats('circuit_analysis',
                                                {f'jump_{jump_epoch}_top{i + 1}_component': component,
                                                 f'jump_{jump_epoch}_top{i + 1}_attribution': attributes
                                                 })
                            # info log circuit interaactions
                            if 'pairwise_interactions' in analysis:
                                circuit_pairs = [(pair, data) for pair, data in
                                                 analysis['pairwise_interactions'].items()
                                                 if data['is_circuit']]
                                if circuit_pairs:
                                    # info sort by strength
                                    circuit_pairs.sort(key=lambda pair: pair[1]['interaction_count'], reverse=True)
                                    # info log
                                    for i, (pair, data) in enumerate(circuit_pairs[:3]):
                                        model.log_stats('circuit_analysis', {
                                            f'jump_{jump_epoch}_circuit{i + 1}_pair': pair,
                                            f'jump_{jump_epoch}_circuit{i + 1}_interaction': data['interaction']
                                        })

                # info should we do additional jump analysis?
                #  warning it is possible that this analysis is a duplicate of the above
                do_additional_jump_analysis = False
                if do_additional_jump_analysis:
                    perform_additional_analyses(jump_results=jump_results, jump_analyzer=jump_analyzer,
                                                weight_tracker=weight_tracker, eval_loader=eval_loader,
                                                criterion=criterion, device=device)

            # info look for correlation with grokking  # fixme  move up to significant jumps search area
            if 'grokking_phases' in model.logger.logs and 'grokking_step' in model.logger.logs['grokking_phases']:
                grokking_step = model.logger.logs['grokking_phases']['grokking_step']
                if grokking_step:
                    if isinstance(grokking_step, list):
                        dist_l = []
                        for grokking_epoch in grokking_step:
                            dist_l.append(abs(jump_epoch - grokking_epoch))
                        distance = min(dist_l)
                    else:
                        distance = abs(jump_epoch - grokking_step)
                    print(f"  Distance to grokking point: {distance} epochs")
                    # info if this jump is very close to grokking, highlight it
                    if distance < 50:
                        print(f"  *** THIS JUMP MAY BE RELATED TO GROKKING! ***")

        # info perform detailed analysis at regular intervals
        do_grokking_check = False
        if do_grokking_check and epoch % (analyze_interval) == 0 and min_epoch_for_detection > 0:  # fixme 16 * ...
            print(f"Performing periodic detailed analysis at epoch {epoch}...")
            grokking_analysis = perform_grokking_analysis(
                model=model, weight_tracker=weight_tracker,
                train_loader=train_loader, eval_loader=eval_loader
            )

        if epoch % 1000 == 0:  # and (epoch - last_timeline_pca_visualized) > 50:
            # info visualize timeline and PCA traajectories
            weight_tracker.visualize_jumps_timeline()
            weight_tracker.visualize_trajectory(
                selected_dims=[0, 1], highlight_epochs={
                    # 'jumps': [j['epoch'] for j in weight_tracker.detected_jumps],
                    'forced': list(set(epochs_to_highlight))
                }
            )
            last_timeline_pca_visualized = epoch

        # info save checkpoint
        if epoch % checkpoint_interval == 0 or epoch == epochs - 1:
            checkpointManager.save_checkpoint(
                epoch=epoch + 1,
                train_dataloader_state=train_dataloader_state,
                eval_dataloader_state=eval_dataloader_state,
                dataset_split_indices=dataset_split_indices,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=eval_loss,
                val_accuracy=eval_accuracy
            )

    # info final analysis at the end of training
    print("Training complete. Generating final analysis...")

    # info visualize the overall jump timeline
    weight_tracker.visualize_jumps_timeline()
    # info analyze the trajectory in weight space
    weight_tracker.visualize_trajectory(
        selected_dims=[0, 1],
        highlight_epochs={
            # 'jumps': [j['epoch'] for j in weight_tracker.detected_jumps],
            'weak': list(set(epochs_to_highlight))
        }
    )

    # info log jump summary to the model's logger
    jump_summary = weight_tracker.get_jump_summary()
    if jump_summary is not None and model.logger:
        model.logger.log_data('weight_space_jumps', 'summary', jump_summary.to_dict('records'))

    # info check if there's correlation between jumps and grokking
    if 'grokking_phases' in model.logger.logs and model.logger.logs['grokking_phases'].get('grokking_step'):
        grokking_step = model.logger.logs['grokking_phases']['grokking_step']
        jumps = [j['epoch'] for j in weight_tracker.detected_jumps]

        if jumps and grokking_step:
            # info find closest jump to grokking point
            # fixme only in some close distance?
            if isinstance(grokking_step, list):
                dist_l = []
                for grokking_epoch in grokking_step:
                    dist_l.append(abs(jump_epoch - grokking_epoch))
                distance = min(dist_l)
            else:
                distance = abs(jump_epoch - grokking_step)
            print(f"\nGrokking detected at epoch {grokking_step}")
            # print(f"Closest weight space jump was at epoch {closest_jump} (distance: {distance} epochs)")

            # info log correlations
            if model.logger:
                model.logger.log_data('grokking_jump_correlation', 'grokking_step', grokking_step)
                # model.logger.log_data('grokking_jump_correlation', 'closest_jump', closest_jump)
                model.logger.log_data('grokking_jump_correlation', 'distance', distance)

    return model, weight_tracker, jump_analyzer


# info perform some grokking analysis and save (?)
def perform_grokking_analysis(model, weight_tracker, train_loader, eval_loader):
    # todo move all to some class

    grokking_analysis = analyze_grokking_transitions(model=model, weight_tracker=weight_tracker,
                                                     train_loader=train_loader, eval_loader=eval_loader)
    if grokking_analysis:
        if 'primary_grokking_step' in grokking_analysis and grokking_analysis['primary_grokking_step']:
            debug_grokking_step(
                grokking_analysis['primary_grokking_step'],
                "perform_grokking_analysis",
                model.logger if hasattr(model, "logger") else None
            )
            # Handle both scalar and list cases for primary_grokking_step
            primary_step = grokking_analysis['primary_grokking_step']

            if isinstance(primary_step, list):
                print(f"  Multiple grokking points detected: {primary_step}")
                grokking_steps = primary_step
            else:
                print(f"  Grokking detected at epoch {primary_step}")
                grokking_steps = [primary_step]

            # Check if any grokking steps coincide with detected jumps
            jumps = [j['epoch'] for j in weight_tracker.detected_jumps]

            if jumps:
                for step in grokking_steps:
                    try:
                        closest_jump = min(jumps, key=lambda x: abs(x - step))
                        distance = abs(closest_jump - step)
                        print(
                            f"  Closest jump to grokking point at epoch {step} is at epoch {closest_jump} (distance: {distance} epochs)")
                    except (TypeError, ValueError) as e:
                        print(f"  Error finding closest jump for grokking step {step}: {e}")
                        print(f"  grokking_step type: {type(step)}, value: {step}")
                        print(f"  jumps: {jumps[:5]}{'...' if len(jumps) > 5 else ''}")

    return grokking_analysis

def get_loss_landscape_slice(model, inputs, targets, criterion, direction1, direction2,
                             step_size=0.1, n_steps=10):
    """
    Sample the loss landscape along two directions for visualization

    Parameters:
    -----------
    model : torch.nn.Module
        The model to analyze
    inputs, targets : torch.Tensor
        Batch of data
    criterion : loss function
        Loss function
    direction1, direction2 : list of torch.Tensor
        Two random directions in weight space
    step_size : float
        Size of steps in each direction
    n_steps : int
        Number of steps to take in each direction

    Returns:
    --------
    numpy.ndarray
        2D array of loss values
    numpy.ndarray
        1D array of step values
    """
    # info store original parameters
    original_params = [p.detach().clone() for p in model.parameters()]

    # info generate grid
    steps = np.linspace(-step_size * n_steps, step_size * n_steps, 2 * n_steps + 1)
    landscape = np.zeros((len(steps), len(steps)))

    # info sample the landscape
    for i, alpha in enumerate(steps):
        for j, beta in enumerate(steps):
            # info update model parameters
            with torch.no_grad():
                for p, p0, d1, d2 in zip(model.parameters(), original_params, direction1, direction2):
                    p.copy_(p0 + alpha * d1 + beta * d2)

            # info calculate loss
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets).item()
                landscape[i, j] = loss

    # info restore original parameters
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), original_params):
            p.copy_(p0)

    return landscape, steps


def analyze_model_at_jump(model, checkpoint_path, jump_epoch, eval_loader, criterion, device):
    """
    Load a model checkpoint and analyze it at a specific jump point

    Parameters:
    -----------
    model : torch.nn.Module
        The model to analyze
    checkpoint_path : str or Path
        Path to the checkpoint directory
    jump_epoch : int
        The epoch of the jump to analyze
    eval_loader : DataLoader
        Evaluation data loader
    criterion : loss function
        Loss function
    device : torch.device
        Device to run analysis on

    Returns:
    --------
    dict
        Analysis results
    """
    # info load the checkpoint
    checkpoint_path = Path(checkpoint_path)
    checkpoint_file = checkpoint_path / f"checkpoint_step_{jump_epoch}.pt"

    if not checkpoint_file.exists():
        print(f"Checkpoint file not found: {checkpoint_file}")
        return None

    # info load checkpoint
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # info initialize analysis tools
    jump_analyzer = JumpAnalysisTools(
        model=model,
        save_dir=checkpoint_path / "jump_analysis",
        logger=model.logger if hasattr(model, "logger") else None
    )

    # info get a batch of data
    sample_inputs, sample_targets = next(iter(eval_loader))
    sample_inputs, sample_targets = sample_inputs.to(device), sample_targets.to(device)

    # info analyze loss landscape
    landscape_analysis = jump_analyzer.analyze_loss_curvature(
        inputs=sample_inputs,
        targets=sample_targets,
        criterion=criterion
    )

    # info analyze head attribution
    attribution_analysis = model.analyze_head_attribution(eval_loader)

    # info analyze attention patterns
    attention_entropy = model.compute_attention_entropy(eval_loader)

    # info get attention patterns
    _ = model(sample_inputs, store_attention=True)
    attention_patterns = model.get_attention_patterns()

    results = {
        'epoch': jump_epoch,
        'landscape_analysis': landscape_analysis,
        'attribution_analysis': attribution_analysis,
        'attention_entropy': attention_entropy,
        'attention_patterns': {k: v.cpu().numpy() for k, v in attention_patterns.items()} if attention_patterns else {}
    }

    return results
