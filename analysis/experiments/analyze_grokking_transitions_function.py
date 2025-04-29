def analyze_grokking_transitions(model, train_loader, eval_loader, window_size=10):
    """
    Enhanced detection and analysis of grokking transitions.
    This function identifies different phases of the grokking phenomenon
    with greater precision by analyzing multiple signals.

    Args:
        model: The model to analyze
        train_loader: Training data loader
        eval_loader: Evaluation data loader
        window_size: Size of the moving window for trend detection

    Returns:
        dict: Grokking transition analysis results
    """
    # Get accuracy history if available
    if not hasattr(model, 'logger'):
        return {"status": "Model logger not available"}

    logger = model.logger

    # Check if we have enough history
    if (logger.get_length('training', 'accuracy') < window_size or
            logger.get_length('evaluation', 'accuracy') < window_size):
        return {"status": "Not enough training history"}

    # Extract training and evaluation metrics
    trn_epochs = np.array(logger.logs['training']['epoch'])
    trn_accs = np.array(logger.logs['training']['accuracy'])
    trn_losses = np.array(logger.logs['training']['loss'])

    eval_epochs = np.array(logger.logs['evaluation']['epoch'])
    eval_accs = np.array(logger.logs['evaluation']['accuracy'])
    eval_losses = np.array(logger.logs['evaluation']['loss'])

    # Analyze training-evaluation gap (memorization vs. generalization)
    gaps = []
    common_epochs = set(trn_epochs).intersection(set(eval_epochs))

    for epoch in sorted(common_epochs):
        trn_idx = np.where(trn_epochs == epoch)[0][0]
        eval_idx = np.where(eval_epochs == epoch)[0][0]

        gap = trn_accs[trn_idx] - eval_accs[eval_idx]
        gaps.append((epoch, gap))

    gap_epochs, gap_values = zip(*gaps)

    # Calculate moving derivatives to identify phase changes
    def calculate_derivatives(epochs, values, window=window_size):
        derivatives = []
        for i in range(window, len(epochs)):
            # Use linear regression to get slope over the window
            x = epochs[i - window:i]
            y = values[i - window:i]

            # Ensure we have enough points
            if len(x) < 2:
                derivatives.append(0)
                continue

            # Calculate slope using linear regression
            A = np.vstack([x, np.ones(len(x))]).T
            try:
                slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                derivatives.append(slope)
            except:
                derivatives.append(0)

        return np.array(derivatives)

    # Calculate derivatives
    eval_acc_derivatives = calculate_derivatives(eval_epochs, eval_accs)
    trn_loss_derivatives = calculate_derivatives(trn_epochs, trn_losses)
    gap_derivatives = calculate_derivatives(gap_epochs, gap_values)

    # Identify key phase transition points

    # 1. Memorization phase: Training accuracy high, eval accuracy low
    memorization_phase = []
    for i, epoch in enumerate(trn_epochs):
        if i >= window_size and trn_accs[i] > 0.9 and eval_accs[i] < 0.5:
            memorization_phase.append(epoch)

    # 2. Transition onset: First significant increase in eval accuracy derivative
    transition_start = None
    for i, deriv in enumerate(eval_acc_derivatives):
        if deriv > 0.005:  # Significant positive slope in eval accuracy
            transition_start = eval_epochs[i + window_size]
            break

    # 3. Primary grokking step: Maximum acceleration in eval accuracy
    primary_grokking_step = None
    max_deriv = 0
    for i, deriv in enumerate(eval_acc_derivatives):
        if deriv > max_deriv:
            max_deriv = deriv
            primary_grokking_step = eval_epochs[i + window_size]

    # 4. Transition completion: Eval accuracy approaches train accuracy
    transition_complete = None
    for i, (epoch, gap) in enumerate(gaps):
        if i > 0 and gap < 0.1 and gap_values[i - 1] > 0.1:
            transition_complete = epoch
            break

    # 5. Efficiency phase: Both accuracies high, continuing improvement in loss
    efficiency_phase = []
    for i, epoch in enumerate(eval_epochs):
        if i >= window_size and trn_accs[i] > 0.9 and eval_accs[i] > 0.9:
            efficiency_phase.append(epoch)

    # 6. Saturation: Performance plateaus
    saturation_point = None
    if len(eval_acc_derivatives) > window_size:
        for i in range(len(eval_acc_derivatives) - window_size, 0, -1):
            if abs(eval_acc_derivatives[i]) < 0.0001 and eval_accs[i + window_size] > 0.9:
                saturation_point = eval_epochs[i + window_size]
                break

    # Calculate alignment with circuit/weight space transitions
    circuit_alignment = {}
    weight_space_alignment = {}

    # Check for circuit transitions if data is available
    if 'circuit_tracking' in logger.logs:
        circuit_data = logger.logs['circuit_tracking']
        circuit_epochs = circuit_data.get('epoch', [])

        for key in circuit_data.keys():
            if 'circuit_change' in key or 'connectivity_change' in key:
                changes = circuit_data[key]
                for i, val in enumerate(changes):
                    if i < len(circuit_epochs) and val > 0.2:  # Significant change
                        epoch = circuit_epochs[i]
                        distance_to_grokking = abs(epoch - primary_grokking_step) if primary_grokking_step else None

                        if distance_to_grokking is not None and distance_to_grokking < 50:
                            circuit_alignment[epoch] = {
                                'distance': distance_to_grokking,
                                'metric': key,
                                'value': val
                            }

    # Check for weight space jumps if data is available
    if 'weight_space_jumps' in logger.logs:
        jump_epochs = logger.logs['weight_space_jumps'].get('jump_epochs', [])

        for i, epoch in enumerate(jump_epochs):
            distance_to_grokking = abs(epoch - primary_grokking_step) if primary_grokking_step else None

            if distance_to_grokking is not None and distance_to_grokking < 50:
                jump_z_score = logger.logs['weight_space_jumps'].get('jump_z_scores', [])[i] if i < len(
                    logger.logs['weight_space_jumps'].get('jump_z_scores', [])) else None

                weight_space_alignment[epoch] = {
                    'distance': distance_to_grokking,
                    'z_score': jump_z_score
                }

    # Compile the results
    results = {
        'memorization_phase': memorization_phase,
        'transition_start': transition_start,
        'primary_grokking_step': primary_grokking_step,
        'transition_complete': transition_complete,
        'efficiency_phase': efficiency_phase,
        'saturation_point': saturation_point,
        'max_eval_acc_derivative': max_deriv,
        'circuit_alignment': circuit_alignment,
        'weight_space_alignment': weight_space_alignment
    }

    # Visualize the grokking phases
    visualize_grokking_phases(
        trn_epochs, trn_accs, eval_epochs, eval_accs,
        gap_epochs, gap_values, results,
        model=model
    )

    return results


def visualize_grokking_phases(trn_epochs, trn_accs, eval_epochs, eval_accs,
                              gap_epochs, gap_values, results, model=None):
    """
    Visualize the different phases of grokking with enhanced markers for transitions.

    Args:
        trn_epochs, trn_accs: Training epochs and accuracies
        eval_epochs, eval_accs: Evaluation epochs and accuracies
        gap_epochs, gap_values: Epochs and values for train-eval gap
        results: Results from analyze_grokking_transitions
        model: The model (for saving the visualization)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create figure with two subplots (accuracy and gap)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot accuracy data
    ax1.plot(trn_epochs, trn_accs, 'b-', label='Training Accuracy')
    ax1.plot(eval_epochs, eval_accs, 'g-', label='Evaluation Accuracy')

    # Highlight memorization phase
    if results['memorization_phase']:
        memo_start = min(results['memorization_phase'])
        memo_end = max(results['memorization_phase'])
        ax1.axvspan(memo_start, memo_end, alpha=0.15, color='red', label='Memorization Phase')

    # Highlight transition start
    if results['transition_start']:
        ax1.axvline(x=results['transition_start'], color='orange', linestyle='--',
                    label=f'Transition Start: Epoch {results["transition_start"]}')

    # Highlight primary grokking step
    if results['primary_grokking_step']:
        ax1.axvline(x=results['primary_grokking_step'], color='red', linestyle='-', linewidth=2,
                    label=f'Primary Grokking: Epoch {results["primary_grokking_step"]}')

    # Highlight transition completion
    if results['transition_complete']:
        ax1.axvline(x=results['transition_complete'], color='green', linestyle='--',
                    label=f'Transition Complete: Epoch {results["transition_complete"]}')

    # Highlight efficiency phase
    if results['efficiency_phase']:
        eff_start = min(results['efficiency_phase'])
        eff_end = max(results['efficiency_phase'])
        ax1.axvspan(eff_start, eff_end, alpha=0.15, color='green', label='Efficiency Phase')

    # Highlight saturation point
    if results['saturation_point']:
        ax1.axvline(x=results['saturation_point'], color='purple', linestyle='-.',
                    label=f'Saturation: Epoch {results["saturation_point"]}')

    # Add circuit and weight space alignments as markers
    for epoch, data in results['circuit_alignment'].items():
        ax1.plot(epoch, 0.5, 'ro', markersize=10, alpha=0.7)
        ax1.annotate(f"Circuit Δ", xy=(epoch, 0.5), xytext=(5, 5),
                     textcoords="offset points", fontsize=8)

    for epoch, data in results['weight_space_alignment'].items():
        ax1.plot(epoch, 0.4, 'bs', markersize=10, alpha=0.7)
        ax1.annotate(f"Weight Jump", xy=(epoch, 0.4), xytext=(5, 5),
                     textcoords="offset points", fontsize=8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Grokking Phase Analysis')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right')

    # Plot train-eval gap in second subplot
    ax2.plot(gap_epochs, gap_values, 'r-', label='Train-Eval Gap')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gap')
    ax2.set_title('Train-Eval Accuracy Gap')

    # Add vertical lines for transition points on gap plot
    if results['transition_start']:
        ax2.axvline(x=results['transition_start'], color='orange', linestyle='--')

    if results['primary_grokking_step']:
        ax2.axvline(x=results['primary_grokking_step'], color='red', linestyle='-', linewidth=2)

    if results['transition_complete']:
        ax2.axvline(x=results['transition_complete'], color='green', linestyle='--')

    plt.tight_layout()

    # Save if model is provided
    if model and hasattr(model, 'plot_prefix'):
        save_path = f'results/{model.plot_prefix}_enhanced_grokking_phases.png'
        plt.savefig(save_path)
        return save_path

    plt.show()
    return None