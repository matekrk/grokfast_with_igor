import torch
import numpy as np

# Import the enhanced weight tracker and jump analysis tools
from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
from grokking_detection import track_metrics_for_grokking, analyze_grokking_transitions
from jump_analysis_tools import JumpAnalysisTools
from analysis.utils.utils import init_train_dataloader_state, FittingScore


def train_with_enhanced_jumps_analysis(model, train_loader, eval_loader,
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
    model : Decoder
        The transformer model with analysis capabilities
    train_loader : DataLoader
        Training data loader
    eval_loader : DataLoader
        Evaluation data loader
    dataset_split_indices : dict
        Dataset split indices for reproducibility
    criterion : loss function
        Loss function for training
    optimizer : optimizer
        Optimizer for training
    scheduler : scheduler
        Learning rate scheduler
    epochs : int
        Number of training epochs
    device : torch.device
        Device to train on
    checkpointManager : CheckpointManager
        Checkpoint manager instance
    log_interval : int
        How often to log basic metrics (steps)
    analyze_interval : int
        How often to perform detailed analysis (steps)
    jump_detection_threshold : float
        Threshold for jump detection (in standard deviations)
    checkpoint_interval : int
        How often to save checkpoints (steps)
    """
    # info set analyze_interval to 4 * log_interval
    analyze_interval = 4 * log_interval

    train_dataloader_state = init_train_dataloader_state(dataloader=train_loader)
    eval_dataloader_state = init_train_dataloader_state(dataloader=eval_loader)

    # Initialize the enhanced weight space tracker with sliding window
    weight_tracker = EnhancedWeightSpaceTracker(
        model=model,
        save_dir=checkpointManager.checkpoint_dir,
        logger=model.logger,
        jump_detection_window=100,
        snapshot_freq=analyze_interval,
        sliding_window_size=10,  # Keep track of 10 recent states for better pre-jump analysis
        dense_sampling=True,  # Sample more frequently for the sliding window
        jump_threshold=jump_detection_threshold
    )

    # info fitting score computation
    fitscore = FittingScore()

    # info initialize the jump analysis tools
    jump_analyzer = JumpAnalysisTools(
        model=model,
        save_dir=checkpointManager.checkpoint_dir,
        logger=model.logger
    )

    force_snapshot = False
    # info get the initial snapshot
    weight_tracker.take_snapshot(epoch=0, force=True)

    # info main training loop
    for epoch in range(epochs):
        train_dataloader_state['epoch'] = epoch
        train_correct = train_total = 0
        train_loss = 0.0
        model.train()

        # info training loop
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # info track accuracy and loss
            train_last_token_preds = logits.argmax(dim=-1)
            train_correct += (train_last_token_preds == targets).sum().item()
            train_total += targets.size(0)
            train_loss += loss.item() * targets.size(0)

        # info compute the final training statistics
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        train_loss = train_loss / train_total if train_total > 0 else 0.0

        # info take weight space snapshot with enhanced jump detection
        min_epoch_for_detection = 100
        # info force snapshot around analyze intervals for better coverage
        # force_snapshot = epoch > min_epoch_for_detection and (
        #     ((epoch - 1) % analyze_interval == 0) or
        #     ((epoch + 1) % analyze_interval == 0)
        # )
        # info more aggressive sampling around potential transition points
        force_snapshot = (
                epoch > min_epoch_for_detection and (
                epoch % analyze_interval == 0 or  # Regular interval
                (epoch - 1) % analyze_interval == 0 or  # Just before
                (epoch + 1) % analyze_interval == 0 or  # Just after
                (epoch - 2) % analyze_interval == 0 or  # Two before
                (epoch + 2) % analyze_interval == 0  # Two after
        ))
        # info sampling based on actual velocity values
        # info increase sampling rate when we notice larger weight changes
        # if epoch >= 2 and 'weight_velocity' in model.logger.logs:
        #     recent_velocities = model.logger.logs['weight_velocity'][-5:]
        #     avg_velocity = sum(recent_velocities) / len(recent_velocities)
        #     current_velocity = model.logger.logs['weight_velocity'][-1]
        #
        # info if current velocity is significantly higher than recent average
        #     force_snapshot = force_snapshot or (current_velocity > 1.5 * avg_velocity)

        # info return true if snapshot was taken for this epoch
        took_snapshot = weight_tracker.take_snapshot(epoch=epoch, force=force_snapshot)

        # info check if we need to log and analyze
        # fixme should condition on log_interval be here or should it be run unconditionally?
        if epoch % log_interval == 0 or took_snapshot:
            # info log training stats
            train_stats = {'accuracy': train_accuracy, 'loss': train_loss, 'epoch': epoch}
            model.log_stats('training', train_stats)

            # info evaluate loss and accuracy on validation set
            model.eval()
            eval_dataloader_state['epoch'] = epoch
            eval_accuracy, eval_loss = model.evaluate(eval_loader)
            eval_stats = {'accuracy': eval_accuracy, 'loss': eval_loss, 'epoch': epoch}
            model.log_stats('evaluation', eval_stats)
            model.train()

            # info calculate fitting score for monitoring
            fitting_score = fitscore.fitting_score(train_loss=train_loss, train_accu=train_accuracy,
                                                   eval_loss=eval_loss, eval_accu=eval_accuracy)
            print(f"Epoch {epoch:5d}: Train Loss={train_loss:.3g}, Train Acc={train_accuracy:.4f}, "
                  f"Val Loss={eval_loss:.5g}, Val Acc={eval_accuracy:.4f}, Fitting_score={fitting_score:.4f}")

        all_grokking_tests = False
        if all_grokking_tests:
            # info track metrics for grokking detection
            track_metrics_for_grokking(epoch=epoch, model=model, train_loader=train_loader, eval_loader=eval_loader)

        # info check for pending jumps that need analysis
        if weight_tracker.pending_jumps:
            # fixme in my runs always (?) a single jump is detected nad analyzed
            print(f"Analyzing {len(weight_tracker.pending_jumps)} detected jump(s)...")

            # Get a batch of data for analysis
            sample_inputs, sample_targets = next(iter(eval_loader))
            sample_inputs, sample_targets = sample_inputs.to(device), sample_targets.to(device)

            # info analyze pending jumps with balanced before/after snapshots
            jump_results = weight_tracker.analyze_pending_jumps(
                inputs=sample_inputs,
                targets=sample_targets,
                criterion=criterion,
                jump_analyzer=jump_analyzer
            )
            # info process results
            for result in jump_results:
                jump_epoch = result['jump_epoch']
                jump_char = result['characterization']

                print(f"Jump analysis for epoch {jump_epoch}:")
                print(f"  Total magnitude: {jump_char['total_magnitude']['pre_to_jump']:.4f}")
                print(f"  Symmetry ratio: {jump_char['total_magnitude']['symmetry_ratio']:.4f}")
                print(f"  Top changing layers: {', '.join(jump_char['top_layers'])}")
                print(f"  Top changing heads: {', '.join(jump_char['top_heads'])}")

                # info create jump characterization visualizations
                viz_dir = weight_tracker.visualize_jump_characterization(jump_char)
                print(f"  Visualizations saved to {viz_dir}")

                # info look for correlation with grokking
                # warning is information on "grokking_phases" saved in weight_tracker
                #  or in analyze_grokking_transitions?
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

                            '''
                            # info perform additional analysis here, e.g.for example, 
                            #  analyze which specific heads are changing most,
                            #  connect to attention-attribution analysis

                            # info AttentionMLPAnalyzer for additional analysis
                            if 'top_heads' in jump_char:
                                print(f"  Analyzing top changing heads during grokking transition...")
                                # Additional specialized analysis for grokking-related jumps
                            '''

            ###################################################################
            # info perform additional analyses for each jump
            workout_attention = False
            workout_attribution = False
            for result in jump_results:
                jump_epoch = result['jump_epoch']
                print(f"  Additional analysis for jump at epoch {jump_epoch}...")

                # info Analyze head attribution around the jump
                if workout_attribution:
                    attribution_results = jump_analyzer.analyze_head_attribution_around_jump(
                        jump_epoch=jump_epoch,
                        eval_loader=eval_loader,
                        weight_tracker=weight_tracker
                    )

                # info analyze attention patterns around the jump
                if workout_attention:
                    attention_results = jump_analyzer.analyze_attention_patterns_around_jump(
                        jump_epoch=jump_epoch,
                        eval_loader=eval_loader,
                        weight_tracker=weight_tracker
                    )

                # info save visualization of jumps timeline after each analysis
                weight_tracker.visualize_jumps_timeline()

                # info save visualization of weight trajectory with jump highlighted
                weight_tracker.visualize_trajectory(
                    selected_dims=[0, 1],
                    highlight_epochs={
                        'jumps': [j['epoch'] for j in weight_tracker.detected_jumps]
                    }
                )

        # info perform detailed analysis at regular intervals
        if epoch % (16 * analyze_interval) == 0 and epoch > 0:
            print(f"Performing periodic detailed analysis at epoch {epoch}...")

            # Analyze loss landscape
            sample_inputs, sample_targets = next(iter(eval_loader))
            sample_inputs, sample_targets = sample_inputs.to(device), sample_targets.to(device)

            landscape_results = jump_analyzer.analyze_loss_curvature(
                inputs=sample_inputs,
                targets=sample_targets,
                criterion=criterion
            )

            # info look for grokking transitions
            # warning this may be a redundant repetition of the analysis in weight tracker
            grokking_analysis = analyze_grokking_transitions(
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader
            )

            # info check if a grokking point occured at or very near a jump
            if grokking_analysis and 'primary_grokking_step' in grokking_analysis and grokking_analysis['primary_grokking_step']:
                print(f"  Grokking detected at epoch {grokking_analysis['primary_grokking_step']}")

                # info check if this coincides with any detected jumps
                # todo do some better statistics
                jumps = [j['epoch'] for j in weight_tracker.detected_jumps]
                closest_jump = min(jumps, key=lambda x: abs(x - grokking_analysis['primary_grokking_step'])) if jumps else None

                if closest_jump:
                    # info avoid repetitions todo (a dictionary?)
                    distance = abs(closest_jump - grokking_analysis['primary_grokking_step'])
                    print(f"  Closest jump to grokking point is at epoch {closest_jump} (distance: {distance} epochs)")

        # Save checkpoint
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

    # info Final analysis at the end of training
    print("Training complete. Generating final analysis...")

    # Visualize the overall jump timeline
    weight_tracker.visualize_jumps_timeline()

    # Analyze the trajectory in weight space
    weight_tracker.visualize_trajectory(
        selected_dims=[0, 1],
        highlight_epochs={
            'jumps': [j['epoch'] for j in weight_tracker.detected_jumps]
        }
    )

    # Log jump summary to the model's logger
    jump_summary = weight_tracker.get_jump_summary()
    if jump_summary is not None and model.logger:
        model.logger.log_data('weight_space_jumps', 'summary', jump_summary.to_dict('records'))

    # Check if there's correlation between jumps and grokking
    # warning an error in last (probably?) call and operation ... lambda x: abs(x - grokking_step)...
    #  TypeError: unsupported operand type(s) for -: 'int' and 'list'
    if 'grokking_phases' in model.logger.logs and model.logger.logs['grokking_phases'].get('grokking_step'):
        grokking_step = model.logger.logs['grokking_phases']['grokking_step']
        jumps = [j['epoch'] for j in weight_tracker.detected_jumps]

        if jumps and grokking_step:
            # Find closest jump to grokking point
            if isinstance(grokking_step, list):
                dist_l = []
                for grokking_epoch in grokking_step:
                    dist_l.append(abs(jump_epoch - grokking_epoch))
                distance = min(dist_l)
            else:
                distance = abs(jump_epoch - grokking_step)
            print(f"\nGrokking detected at epoch {grokking_step}")
            print(f"Closest weight space jump was at epoch {closest_jump} (distance: {distance} epochs)")

            # Log this correlation
            if model.logger:
                model.logger.log_data('grokking_jump_correlation', 'grokking_step', grokking_step)
                model.logger.log_data('grokking_jump_correlation', 'closest_jump', closest_jump)
                model.logger.log_data('grokking_jump_correlation', 'distance', distance)

    return model, weight_tracker, jump_analyzer


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
    # Store original parameters
    original_params = [p.detach().clone() for p in model.parameters()]

    # Generate grid
    steps = np.linspace(-step_size * n_steps, step_size * n_steps, 2 * n_steps + 1)
    landscape = np.zeros((len(steps), len(steps)))

    # Sample the landscape
    for i, alpha in enumerate(steps):
        for j, beta in enumerate(steps):
            # Update model parameters
            with torch.no_grad():
                for p, p0, d1, d2 in zip(model.parameters(), original_params, direction1, direction2):
                    p.copy_(p0 + alpha * d1 + beta * d2)

            # Calculate loss
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets).item()
                landscape[i, j] = loss

    # Restore original parameters
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), original_params):
            p.copy_(p0)

    return landscape, steps


