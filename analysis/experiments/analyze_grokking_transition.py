#!/usr/bin/env python
"""
Script for analyzing a transformer model around the grokking transition point
using fine-grained resumption and the Attention-MLP Analyzer framework.
"""

import argparse
import torch
from pathlib import Path
import pandas as pd

# Import your existing modules
from analysis.models.analysis_transformer import Decoder
from grokking_detection import detect_grokking_multi_metric
from analysis.models.modular_data import create_modular_dataloaders

# Import the resume training function we created earlier
from resume_analysis import resume_training_around_grokking

# Import our analyzer class (assuming it's in a file called attention_mlp_analyzer.py)
from analysis.analyzers.attention_mlp_analyzer import AttentionMLPAnalyzer, SparseAutoencoder


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