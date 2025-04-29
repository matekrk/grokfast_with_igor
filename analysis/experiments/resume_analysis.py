import argparse
import os

import torch
from pathlib import Path

from analysis.models.analysis_transformer import Decoder
from analysis.utils.checkpoint_manager import CheckpointManager
from analysis.analyzers.grokking_detection import track_metrics_for_grokking, analyze_grokking_transitions
from analysis.models.modular_data import create_modular_dataloaders
from analysis.visualization import visualize_model_analysis


def resume_training_around_grokking(
        checkpoint_path,
        start_epoch=None,
        num_epochs=20,
        log_interval=1,  # Set to 1 for epoch-by-epoch analysis
        analyze_interval=5,
        save_visualizations=True
):
    """
    Resume training from a checkpoint with fine-grained analysis around the grokking point.

    Args:
        checkpoint_path: Path to the checkpoint file
        start_epoch: Override the epoch to start from (if None, use checkpoint's epoch)
        num_epochs: Number of epochs to continue training
        log_interval: How often to log metrics (in epochs)
        analyze_interval: How often to perform detailed analysis (in epochs)
        save_visualizations: Whether to save visualization plots
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Extract model and training state
    model_state = checkpoint["model_state_dict"]
    optimizer_state = checkpoint["optimizer_state_dict"]
    scheduler_state = checkpoint.get("scheduler_state_dict")
    stats = checkpoint.get("stats", {})

    # Determine the epoch to resume from
    resume_epoch = start_epoch if start_epoch is not None else checkpoint.get("epoch", 0)

    # Load associated state info for dataloader states
    state_path = str(checkpoint_path).replace(".pt", "_state.json")
    if Path(state_path).exists():
        import json
        with open(state_path, "r") as f:
            state_info = json.load(f)

        train_dataloader_state = state_info.get("train_dataloader_state")
        eval_dataloader_state = state_info.get("eval_dataloader_state")
        dataset_split_indices = None

        # Load dataset split indices if available
        split_indices_path = state_info.get("dataset_split_indices_file")
        if split_indices_path and Path(split_indices_path).exists():
            dataset_split_indices = torch.load(split_indices_path)
    else:
        train_dataloader_state = None
        eval_dataloader_state = None
        dataset_split_indices = None

    # Recreate the model
    # Assuming we can extract model config from the checkpoint
    model_config = {
        "dim": checkpoint.get("dim", 128),
        "num_layers": checkpoint.get("num_layers", 2),
        "num_heads": checkpoint.get("num_heads", 4),
        "num_tokens": checkpoint.get("num_tokens", 97),
        "seq_len": checkpoint.get("seq_len", 5)
    }

    # If model info is missing from checkpoint, try to infer from the model state dictionary
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

    # Extract important information from checkpoint for recreating dataloaders
    modulus = model_config["num_tokens"] - 2  # Assuming p + 2 tokens as in analysis.py

    # Determine operation from checkpoint or use multiply as default
    operation = checkpoint.get("operation", "multiply")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Recreate model with same configuration
    model = Decoder(
        dim=model_config["dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        num_tokens=model_config["num_tokens"],
        seq_len=model_config["seq_len"],
        criterion=criterion,
        device=device,
        id=checkpoint.get("id", "resumed_model"),
        save_dir=Path(checkpoint_path).parent.parent,
        checkpoint_dir=Path(checkpoint_path).parent,
    )

    # Load the state into model
    model.load_state_dict(model_state)
    model.to(device)

    # Recreate dataloaders
    batch_size = train_dataloader_state.get("batch_size", 256) if train_dataloader_state else 256
    train_ratio = checkpoint.get("train_ratio", 0.5)

    train_loader, eval_loader, vocab_size, new_dataset_split_indices = create_modular_dataloaders(
        modulus=modulus,
        op=operation,
        train_ratio=train_ratio,
        batch_size=batch_size,
        sequence_format=True,
        seed=checkpoint.get("seed", 42),
        dataset_split_indices=dataset_split_indices  # Use saved indices if available
    )

    # Recreate optimizer
    optimizer_type = checkpoint.get("optimizer_type", "AdamW")
    lr = checkpoint.get("lr", 1e-3)

    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Load optimizer state
    optimizer.load_state_dict(optimizer_state)

    # Recreate scheduler  #todo do we need to recreate only if it was used?
    #  todo How to check? perhaps using the scheduler_state variable read above?
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )

    # Load scheduler state if available
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    # Create checkpoint manager
    checkpointManager = CheckpointManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_name=model.get_id(),
        base_dir="../../results",
        save_dir=model.save_dir,
        checkpoint_dir=model.checkpoint_dir,
        stats_dir="../../stats",
        save_freq=1,  # Save every epoch for fine-grained analysis
        max_to_keep=25
    )

    # Continue training with fine-grained logging
    continue_training_with_analysis(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        dataset_split_indices=new_dataset_split_indices,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=resume_epoch,
        num_epochs=num_epochs,
        device=device,
        checkpointManager=checkpointManager,
        log_interval=log_interval,
        analyze_interval=analyze_interval,
        save_visualizations=save_visualizations
    )

    return model


def continue_training_with_analysis(
        model, train_loader, eval_loader, dataset_split_indices,
        criterion, optimizer, scheduler,
        start_epoch, num_epochs, device,
        checkpointManager, log_interval=1, analyze_interval=5,
        save_visualizations=True
):
    """
    Continue training with fine-grained analysis.

    This is a modified version of train_with_analysis_epochs from analysis.py,
    adapted for resuming training with epoch-by-epoch granularity.
    """
    from tqdm import tqdm

    end_epoch = start_epoch + num_epochs

    # Initialize dataloader states
    train_dataloader_state = {
        'batch_idx': 0,
        'epoch': start_epoch,
        'sampler_iter_state': None,
        'batch_size': train_loader.batch_size,
        'shuffle': True,
        'num_workers': train_loader.num_workers,
        'pin_memory': train_loader.pin_memory,
        'drop_last': train_loader.drop_last
    }

    eval_dataloader_state = {
        'batch_idx': 0,
        'batch_size': eval_loader.batch_size,
        'shuffle': False,
        'num_workers': eval_loader.num_workers,
        'pin_memory': eval_loader.pin_memory,
        'drop_last': eval_loader.drop_last
    }

    # Training loop
    for epoch in tqdm(range(start_epoch, end_epoch)):
        train_dataloader_state['epoch'] = epoch
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        model.train()

        for inputs, targets in train_loader:
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

        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        train_loss = train_loss / train_total if train_total > 0 else 0.0

        # Log metrics at every log_interval epoch
        if epoch % log_interval == 0:
            train_stats = {'accuracy': train_accuracy, 'loss': train_loss, 'epoch': epoch}
            model.log_stats('training', train_stats)

            # Evaluate
            model.eval()
            eval_dataloader_state['epoch'] = epoch
            eval_accuracy, eval_loss = model.evaluate(eval_loader)
            eval_stats = {'accuracy': eval_accuracy, 'loss': eval_loss, 'epoch': epoch}
            model.log_stats('evaluation', eval_stats)
            model.train()

            # Track metrics for grokking detection
            track_metrics_for_grokking(
                epoch=epoch,
                model=model,
                train_loader=train_loader,
                eval_loader=eval_loader
            )

            # Perform detailed analysis at analyze_interval
            if epoch % analyze_interval == 0 and save_visualizations:
                grokking_analysis = analyze_grokking_transitions(model=model, train_loader=train_loader,
                                                                 eval_loader=eval_loader,
                                                                 save_path_prefix="fine_")
                visualize_model_analysis(
                    model=model,
                    epoch=epoch,
                    eval_loader=eval_loader,
                    include_metrics=['attention', 'attribution', 'cross_attribution', 'entropy',
                                     'weight_norms', 'accuracy', 'loss'],
                    save_path=f"{model.save_dir}/visualization_fine_epoch_{epoch}.png"
                )

        # Always save checkpoint (since we're doing fine-grained analysis)
        checkpointManager.save_checkpoint(
            epoch=epoch + 1,
            train_dataloader_state=train_dataloader_state,
            eval_dataloader_state=eval_dataloader_state,
            dataset_split_indices=dataset_split_indices,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=eval_loss if 'eval_loss' in locals() else None,
            val_accuracy=eval_accuracy if 'eval_accuracy' in locals() else None,
            extra_data=None,
            force_save=True,
        )

        # Print status
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch}: train_acc={train_accuracy:.4f}, val_acc={eval_accuracy:.4f}")

    # Final analysis
    if save_visualizations:
        grokking_analysis = analyze_grokking_transitions(model=model, train_loader=train_loader,
                                                         eval_loader=eval_loader,
                                                         save_path_prefix="fine_")
        visualize_model_analysis(
            model=model,
            epoch=end_epoch - 1,
            eval_loader=eval_loader,
            include_metrics=['attention', 'attribution', 'cross_attribution', 'entropy',
                             'weight_norms', 'accuracy', 'loss', 'grokking_phases'],
            save_path=f"{model.save_dir}/visualization_fine_final.png",
            logx=True
        )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume training around grokking point")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--start_epoch", type=int, help="Override start epoch")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--log_interval", type=int, default=1, help="How often to log metrics")
    parser.add_argument("--analyze_interval", type=int, default=5, help="How often to do detailed analysis")
    parser.add_argument("--no_visualize", action="store_false", dest="save_visualizations",
                        help="Disable saving visualizations")

    args = parser.parse_args()

    cwd = os.getcwd()
    resume_training_around_grokking(
        checkpoint_path=args.checkpoint,
        start_epoch=args.start_epoch,
        num_epochs=args.epochs,
        log_interval=args.log_interval,
        analyze_interval=args.analyze_interval,
        save_visualizations=args.save_visualizations
    )