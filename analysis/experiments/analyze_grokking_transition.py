#!/usr/bin/env python
"""
Script for analyzing a transformer model around the grokking transition point
using fine-grained resumption and the Attention-MLP Analyzer framework.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

# Import our analyzer class (assuming it's in a file called attention_mlp_analyzer.py)
from analysis.analyzers.attention_mlp_analyzer import AttentionMLPAnalyzer, SparseAutoencoder
from analysis.analyzers.grokking_detection import detect_grokking_multi_metric
# Import your existing modules
from analysis.models.analysis_transformer import Decoder
from analysis.models.modular_data import create_modular_dataloaders
# Import the resume training function we created earlier
from resume_analysis import resume_training_around_grokking


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


def identify_grokking_checkpoint(model_dir):
    """
    Find the grokking point by analyzing the model's training history.

    Args:
        model_dir: Directory containing model checkpoints and logs

    Returns:
        tuple: (pre_grokking_epoch, grokking_epoch, post_grokking_epoch)
    """
    # First, load the model to access its logger with training history
    checkpoint_dir = Path(model_dir) / "checkpoints"
    stats_dir = Path(model_dir) / "stats"

    # Find the latest checkpoint
    checkpoint_files = sorted(list(checkpoint_dir.glob("checkpoint_step_*.pt")))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading latest checkpoint: {latest_checkpoint}")

    # Load the checkpoint to get model configuration
    checkpoint = torch.load(latest_checkpoint)

    # Extract model config
    model_config = {
        "dim": checkpoint.get("dim", 128),
        "num_layers": checkpoint.get("num_layers", 2),
        "num_heads": checkpoint.get("num_heads", 4),
        "num_tokens": checkpoint.get("num_tokens", 97),
        "seq_len": checkpoint.get("seq_len", 5)
    }

    # info If model info is missing, try to infer from the model state dictionary
    model_state = checkpoint["model_state_dict"]
    if "dim" not in checkpoint and "token_embeddings.weight" in model_state:
        model_config["dim"] = model_state["token_embeddings.weight"].size(1)
    if "num_tokens" not in checkpoint and "token_embeddings.weight" in model_state:
        model_config["num_tokens"] = model_state["token_embeddings.weight"].size(0)
    if "num_layers" not in checkpoint:
        # Try to infer from largest layer index
        layer_indices = [int(k.split('.')[1]) for k in model_state.keys()
                         if k.startswith("layers.") and k.split('.')[1].isdigit()]
        if layer_indices:
            model_config["num_layers"] = max(layer_indices) + 1

    # Create a temporary model to access the logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_model = Decoder(
        dim=model_config["dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        num_tokens=model_config["num_tokens"],
        seq_len=model_config["seq_len"],
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        id=checkpoint.get("id", "temp_model"),
        save_dir=Path(model_dir),
        checkpoint_dir=checkpoint_dir,
    )

    # Load state dict to get logger data
    temp_model.load_state_dict(model_state)

    # Try to detect grokking using our existing function
    detection_result = detect_grokking_multi_metric(temp_model)

    if detection_result is None or 'primary_grokking_step' not in detection_result:
        print("No clear grokking point detected automatically. Using manual fallback.")
        # Fallback: analyze accuracy curve manually
        eval_data = pd.DataFrame(temp_model.logger.get_logs("evaluation"))

        # Look for the point where accuracy starts consistently improving
        window_size = 10
        eval_data['smooth_acc'] = eval_data['accuracy'].rolling(window=window_size, center=True).mean()
        eval_data['acc_change'] = eval_data['smooth_acc'].diff()

        # Find the point with highest acceleration in accuracy
        peak_idx = eval_data['acc_change'].argmax()
        if peak_idx < window_size or peak_idx >= len(eval_data) - window_size:
            # If peak is at the edge, choose a point in the middle
            peak_idx = len(eval_data) // 2

        grokking_epoch = eval_data.iloc[peak_idx]['epoch']
    else:
        grokking_epoch = detection_result['primary_grokking_step']

    print(f"Identified grokking transition at epoch: {grokking_epoch}")

    # Choose points before and after grokking
    all_epochs = sorted([int(f.stem.split('_')[-1]) for f in checkpoint_files])

    # Find nearest checkpoints before and after grokking
    pre_grokking_epoch = max([e for e in all_epochs if e < grokking_epoch], default=all_epochs[0])
    post_grokking_epoch = min([e for e in all_epochs if e > grokking_epoch], default=all_epochs[-1])

    print(
        f"Using checkpoints at epochs: {pre_grokking_epoch} (pre), {grokking_epoch} (transition), {post_grokking_epoch} (post)")

    return pre_grokking_epoch, grokking_epoch, post_grokking_epoch


# info understand model's attention - mlp structure
def get_debug_hook(name, layer_idx):
    def hook(module, input, output):
        print(f"Debug {name} layer {layer_idx}:")
        print(f"  module: {type(module).__name__}")
        print(f"  input type: {type(input)}")
        if isinstance(input, tuple):
            print(f"  input tuple length: {len(input)}")
            for i, inp in enumerate(input):
                print(f"    input[{i}] type: {type(inp)}")
                if torch.is_tensor(inp):
                    print(f"    input[{i}] shape: {inp.shape}")
        print(f"  output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"  output tuple length: {len(output)}")
            for i, out in enumerate(output):
                print(f"    output[{i}] type: {type(out)}")
                if torch.is_tensor(out):
                    print(f"    output[{i}] shape: {out.shape}")
        elif torch.is_tensor(output):
            print(f"  output shape: {output.shape}")
        print()

    return hook


def run_fine_grained_analysis(model_dir, pre_epoch, grok_epoch, post_epoch,
                              window_size=20, epoch_step=1, device=None):
    """
    Run fine-grained analysis around the grokking transition.

    Args:
        model_dir: Directory containing model checkpoints
        pre_epoch: Epoch before grokking
        grok_epoch: Epoch of grokking transition
        post_epoch: Epoch after grokking
        window_size: Size of window around grokking point for fine analysis
        epoch_step: Steps between epochs for fine analysis
        device: Device to run on

    Returns:
        dict: Analysis results
    """
    # Set up paths
    model_dir = Path(model_dir)
    checkpoint_dir = model_dir / "checkpoints"

    # Choose a checkpoint right before the transition to resume from
    start_epoch = max(pre_epoch, grok_epoch - window_size // 2)

    # Calculate number of epochs to train
    num_epochs = window_size + 2  # Add buffer for safety

    # Path to the checkpoint to resume from
    resume_checkpoint = checkpoint_dir / f"checkpoint_step_{start_epoch}.pt"

    if not resume_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint {resume_checkpoint} not found")

    print(f"Resuming training from epoch {start_epoch} for {num_epochs} epochs with step {epoch_step}")

    # Resume training with fine-grained logging
    model = resume_training_around_grokking(
        checkpoint_path=str(resume_checkpoint),
        start_epoch=start_epoch,
        num_epochs=num_epochs,
        log_interval=epoch_step,  # Log at every epoch
        analyze_interval=epoch_step,  # Analyze at every epoch
        save_visualizations=True
    )

    # Get paths for pre, during, and post grokking checkpoints
    pre_ckpt = checkpoint_dir / f"checkpoint_step_{pre_epoch}.pt"
    grok_ckpt = checkpoint_dir / f"checkpoint_step_{grok_epoch}.pt"
    post_ckpt = checkpoint_dir / f"checkpoint_step_{post_epoch}.pt"

    # Reload dataloaders for analysis
    checkpoint = torch.load(str(pre_ckpt))

    # Get dataloader parameters
    modulus = model.num_tokens - 2  # p + 2 tokens
    operation = checkpoint.get("operation", "multiply")
    batch_size = checkpoint.get("batch_size", 256)
    train_ratio = checkpoint.get("train_ratio", 0.5)
    seed = checkpoint.get("seed", 42)

    # Recreate dataloaders
    train_loader, eval_loader, _, _ = create_modular_dataloaders(
        modulus=modulus,
        op=operation,
        train_ratio=train_ratio,
        batch_size=batch_size,
        sequence_format=True,
        seed=seed
    )

    # Run in-depth analysis using our analyzer
    analysis_dir = model_dir / "fine_analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Create analyzer
    analyzer = AttentionMLPAnalyzer(
        model=model,
        eval_loader=eval_loader,
        device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_dir=analysis_dir
    )

    # Define phase checkpoints for analysis
    phase_checkpoints = [
        ("pre_grokking", str(pre_ckpt)),
        ("during_transition", str(grok_ckpt)),
        ("post_grokking", str(post_ckpt))
    ]

    # Analyze across phases
    print("Running cross-phase analysis...")
    phase_analysis = analyzer.analyze_across_grokking_phases(
        save_dir=analysis_dir / "phase_comparison",
        phase_checkpoints=phase_checkpoints
    )

    # Run sparse autoencoder analysis
    autoencoder_results = {}

    for phase_name, ckpt_path in phase_checkpoints:
        print(f"Analyzing {phase_name} with sparse autoencoder...")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        # info  Add before your main analysis loop to understand the model structure
        # warning remove afterwards
        debug_hooks = []
        for i, layer in enumerate(model.layers):
            debug_hooks.append(layer.attn.register_forward_hook(get_debug_hook("attention", i)))
            debug_hooks.append(layer.mlp.register_forward_hook(get_debug_hook("mlp", i)))

        # Run forward pass
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(eval_loader):
                if batch_idx >= 1:  # Just one batch for debugging
                    break
                inputs = inputs.to(device)
                _ = model(inputs)

        # Remove debug hooks
        for hook in debug_hooks:
            hook.remove()

        # For each phase, analyze both attention and MLP components
        for component in ['mlp', 'attention']:
            # Create a fresh autoencoder
            autoencoder = SparseAutoencoder(
                input_dim=model.dim,
                code_dim=model.dim * 2,  # Double the size for overcomplete representation
                l1_coef=0.001
            )

            # Analyze each layer
            for layer_idx in range(model.num_layers):
                print(f"  Analyzing {component} in layer {layer_idx}...")

                # Collect and analyze activations
                results = analyzer.analyze_with_sparse_autoencoder(
                    autoencoder=autoencoder,
                    component=component,
                    layer_idx=layer_idx,
                    train_epochs=100
                )

                # Visualize results
                vis_paths = analyzer.visualize_autoencoder_results(
                    results=results,
                    component=component,
                    layer_idx=layer_idx,
                    epoch=phase_name
                )

                # Store results
                key = f"{phase_name}_{component}_layer{layer_idx}"
                autoencoder_results[key] = {
                    'analysis': results,
                    'visualizations': vis_paths
                }

    return {
        'model': model,
        'phase_analysis': phase_analysis,
        'autoencoder_results': autoencoder_results
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze transformer model around grokking transition")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing model checkpoints")
    parser.add_argument("--window-size", type=int, default=20, help="Window size around grokking point")
    parser.add_argument("--epoch-step", type=int, default=1, help="Step size between epochs")
    parser.add_argument("--pre-epoch", type=int, help="Manually specify pre-grokking epoch")
    parser.add_argument("--grok-epoch", type=int, help="Manually specify grokking epoch")
    parser.add_argument("--post-epoch", type=int, help="Manually specify post-grokking epoch")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use")

    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Either use provided epochs or detect automatically
    if args.pre_epoch is not None and args.grok_epoch is not None and args.post_epoch is not None:
        pre_epoch = args.pre_epoch
        grok_epoch = args.grok_epoch
        post_epoch = args.post_epoch
    else:
        pre_epoch, grok_epoch, post_epoch = identify_grokking_checkpoint(args.model_dir)

    # Run analysis
    results = run_fine_grained_analysis(
        model_dir=args.model_dir,
        pre_epoch=pre_epoch,
        grok_epoch=grok_epoch,
        post_epoch=post_epoch,
        window_size=args.window_size,
        epoch_step=args.epoch_step,
        device=device
    )

    print("Analysis complete! Results saved to:")
    print(f"  {Path(args.model_dir) / 'fine_analysis'}")


if __name__ == "__main__":
    main()
