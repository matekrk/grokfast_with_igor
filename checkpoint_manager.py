import os
import json
import torch
import numpy as np
import random
import time
from datetime import datetime
from pathlib import Path


class CheckpointManager:
    """
    Manages saving and loading of experiment checkpoints with complete state
    for resumable training.
    """

    def __init__(self,
                 model,
                 optimizer,
                 scheduler=None,
                 experiment_name=None,
                 base_dir="results",
                 save_dir = "results",
                 checkpoint_dir = "checkpoints",
                 stats_dir = "stats",
                 save_freq=1000,
                 max_to_keep=5):
        """
        Initialize the checkpoint manager.

        Args:
            model: The PyTorch model to checkpoint
            optimizer: The optimizer
            scheduler: Optional learning rate scheduler
            experiment_name: Name for the experiment (auto-generated if None)
            base_dir: Base directory for all experiments
            save_freq: How often to save checkpoints (in steps)
            max_to_keep: Maximum number of checkpoints to retain
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_freq = save_freq
        self.max_to_keep = max_to_keep

        # Generate experiment name if not provided
        self.experiment_name = experiment_name

        # Create experiment directory
        self.experiment_dir = save_dir
        self.checkpoint_dir = checkpoint_dir
        self.stats_dir = stats_dir

        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        # self.checkpoint_dir.mkdir(exist_ok=True)
        self.stats_dir.mkdir(exist_ok=True)

        # Track saved checkpoints
        self.checkpoint_files = []

        # Create experiment metadata file
        self.create_experiment_metadata()

        # Initialize statistics tracking
        self.stats = {
            "steps": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
            "epoch": [],
            "best_val_accuracy": 0.0,
            "best_step": 0,
            "experiment_time": 0
        }

        # For tracking training time
        self.start_time = time.time()

    def create_experiment_metadata(self):
        """Create a metadata file with experiment configuration"""
        metadata = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_config": {
                "num_layers": self.model.num_layers,
                "num_heads": self.model.num_heads,
                "dim": self.model.dim,
                "num_tokens": self.model.num_tokens,
                "seq_len": self.model.seq_len
            },
            "optimizer": self.optimizer.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__ if self.scheduler else None,
            "pytorch_version": torch.__version__,
            "save_frequency": self.save_freq,
            "max_checkpoints_kept": self.max_to_keep
        }

        # Save metadata
        with open(self.experiment_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def save_checkpoint(self, epoch,
                        train_dataloader_state=None,
                        eval_dataloader_state=None,
                        dataset_split_indices=None,
                        train_loss=None, train_accuracy=None,
                        val_loss=None, val_accuracy=None,
                        extra_data=None, force_save=False):
        """
        Save a complete checkpoint with all training state.

        Args:
            step: Current training step
            epoch: Current epoch
            dataloader_state: Optional state dict for the dataloader
            train_loss: Current training loss
            train_accuracy: Current training accuracy
            val_loss: Current validation loss
            val_accuracy: Current validation accuracy
            extra_data: Any additional data to save
            force_save: Force saving even if not at regular interval
        """
        # Only save at specified frequency unless forced
        if epoch % self.save_freq != 0 and not force_save:
            return

        # Update experiment stats
        self.update_stats(epoch, train_loss, train_accuracy,
                          val_loss, val_accuracy)

        # Generate checkpoint paths
        checkpoint_name = f"checkpoint_step_{epoch}"
        model_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        state_path = self.checkpoint_dir / f"{checkpoint_name}_state.json"

        # Create the checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": self.stats,
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                # Corrected to get_rng_state_all()
            }
        }

        # Add scheduler state if it exists
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save additional state information as JSON
        state_info = {
            "epoch": epoch,
            "train_loss": train_loss if train_loss is not None else "N/A",
            "train_accuracy": train_accuracy if train_accuracy is not None else "N/A",
            "val_loss": val_loss if val_loss is not None else "N/A",
            "val_accuracy": val_accuracy if val_accuracy is not None else "N/A",
            "checkpoint_file": str(model_path),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add any extra data
        if extra_data is not None:
            state_info["extra_data"] = extra_data

        # Save the model checkpoint
        torch.save(checkpoint, model_path)

        # Save the state info as JSON
        with open(state_path, "w") as f:
            json.dump(state_info, f, indent=4)

        # Update checkpoint file list
        self.checkpoint_files.append((model_path, state_path))

        # Manage maximum checkpoints to keep
        self._manage_checkpoint_retention()


        # Add dataloader states if provided
        if train_dataloader_state is not None:
            state_info["train_dataloader_state"] = train_dataloader_state

        if eval_dataloader_state is not None:
            state_info["eval_dataloader_state"] = eval_dataloader_state

        # Save dataset split indices if provided
        if dataset_split_indices is not None:
            split_indices_path = self.checkpoint_dir / f"{checkpoint_name}_split_indices.pt"
            torch.save(dataset_split_indices, split_indices_path)
            state_info["dataset_split_indices_file"] = str(split_indices_path)


        print(f"\nCheckpoint saved at epoch {epoch}")

    def update_stats(self, epoch, train_loss=None, train_accuracy=None,
                     val_loss=None, val_accuracy=None):
        """Update the training statistics"""
        self.stats["epoch"].append(epoch)

        # Update current learning rate
        if self.scheduler is not None:
            lr = self.scheduler.get_last_lr()[0]
        else:
            lr = self.optimizer.param_groups[0]['lr']
        self.stats["learning_rate"].append(lr)

        # Update loss and accuracy metrics if provided
        if train_loss is not None:
            self.stats["train_loss"].append(train_loss)

        if train_accuracy is not None:
            self.stats["train_accuracy"].append(train_accuracy)

        if val_loss is not None:
            self.stats["val_loss"].append(val_loss)

        if val_accuracy is not None:
            self.stats["val_accuracy"].append(val_accuracy)

            # Update best validation accuracy if improved
            if val_accuracy > self.stats["best_val_accuracy"]:
                self.stats["best_val_accuracy"] = val_accuracy
                self.stats["best_epoch"] = epoch

        # Update experiment time
        self.stats["experiment_time"] = time.time() - self.start_time

        # Save the updated stats to a JSON file
        stats_path = self.stats_dir / "training_stats.json"
        with open(stats_path, "w") as f:
            # Create a copy of stats with numpy arrays converted to lists
            serializable_stats = self.stats.copy()
            for key, value in serializable_stats.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        serializable_stats[key] = [v.tolist() for v in value]

            json.dump(serializable_stats, f, indent=4)

    def _manage_checkpoint_retention(self):
        """Manage the number of checkpoints to keep"""
        if len(self.checkpoint_files) > self.max_to_keep:
            # Remove the oldest checkpoints
            num_to_remove = len(self.checkpoint_files) - self.max_to_keep
            for i in range(num_to_remove):
                model_path, state_path = self.checkpoint_files.pop(0)

                # Remove the files if they exist
                if os.path.exists(model_path):
                    os.remove(model_path)
                if os.path.exists(state_path):
                    os.remove(state_path)

    def load_checkpoint(self, checkpoint_path=None, step=None):
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            step: Alternatively, specify the step to load

        Returns:
            dict: The loaded state info
        """
        # If step is provided, find the corresponding checkpoint
        if checkpoint_path is None and step is not None:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"

        # If neither is provided, find the latest checkpoint
        if checkpoint_path is None:
            checkpoint_files = sorted(list(self.checkpoint_dir.glob("checkpoint_step_*.pt")))
            if not checkpoint_files:
                raise FileNotFoundError("No checkpoints found in the experiment directory")
            checkpoint_path = checkpoint_files[-1]

        # Ensure checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load the checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # Restore model and optimizer state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore RNG states
        if "rng_states" in checkpoint:
            rng_states = checkpoint["rng_states"]
            random.setstate(rng_states["python"])
            np.random.set_state(rng_states["numpy"])
            torch.set_rng_state(rng_states["torch"])
            if torch.cuda.is_available() and rng_states["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(rng_states["torch_cuda"])  # Corrected if it's a list of states

        # Restore stats
        if "stats" in checkpoint:
            self.stats = checkpoint["stats"]

        # Load associated state info
        state_path = str(checkpoint_path).replace(".pt", "_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state_info = json.load(f)
        else:
            state_info = {
                "step": checkpoint["step"],
                "epoch": checkpoint["epoch"]
            }

        # Print resumption information
        print(f"Resumed training from epoch {checkpoint['epoch']}")
        if "best_val_accuracy" in self.stats:
            print(
                f"Best validation accuracy so far: {self.stats['best_val_accuracy']:.4f} at epoch {self.stats['best_epoch']}")

        return state_info

    def save_head_attribution(self, attribution_scores, file_prefix="head_attribution"):
        """
        Save head attribution analysis results

        Args:
            step: Current training step
            attribution_scores: Dictionary of attribution scores
            file_prefix: Prefix for the filename
        """
        attribution_dir = self.stats_dir / "attributions"
        attribution_dir.mkdir(exist_ok=True)

        # Save as JSON
        file_path = attribution_dir / f"{file_prefix}_step_{epoch}.json"

        # Convert to serializable format if needed
        serializable_scores = {}
        for head, score in attribution_scores.items():
            if isinstance(score, (torch.Tensor, np.ndarray)):
                serializable_scores[head] = float(score)
            else:
                serializable_scores[head] = score

        with open(file_path, "w") as f:
            json.dump(serializable_scores, f, indent=4)

        return file_path

    def save_attention_patterns(self, epoch, attention_patterns, example_ids=None):
        """
        Save attention patterns for analysis

        Args:
            epoch: Current training epoch
            attention_patterns: Dictionary of attention patterns
            example_ids: Optional IDs for the examples
        """
        attention_dir = self.stats_dir / "attention_patterns"
        attention_dir.mkdir(exist_ok=True)

        # Save in NPZ format for efficient storage of matrices
        file_path = attention_dir / f"attention_patterns_step_{epoch}.npz"

        # Convert torch tensors to numpy arrays
        np_patterns = {}
        for key, pattern in attention_patterns.items():
            if isinstance(pattern, torch.Tensor):
                np_patterns[key] = pattern.detach().cpu().numpy()
            else:
                np_patterns[key] = pattern

        # Add example IDs if provided
        if example_ids is not None:
            np_patterns["example_ids"] = example_ids

        np.savez_compressed(file_path, **np_patterns)

        return file_path

    def save_circuit_analysis(self, epoch, individual_results, pairwise_results):
        """
        Save circuit analysis results

        Args:
            step: Current training step
            individual_results: Results from individual head ablation
            pairwise_results: Results from pairwise head ablation
        """
        circuit_dir = self.stats_dir / "circuits"
        circuit_dir.mkdir(exist_ok=True)

        # Save individual results
        ind_file_path = circuit_dir / f"individual_ablation_step_{epoch}.json"
        serializable_ind = {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                            for k, v in individual_results.items()}

        with open(ind_file_path, "w") as f:
            json.dump(serializable_ind, f, indent=4)

        # Save pairwise results
        pair_file_path = circuit_dir / f"pairwise_ablation_step_{epoch}.json"

        # Convert to serializable format
        serializable_pair = {}
        for (head1, head2), results in pairwise_results.items():
            key = f"{head1}_{head2}"
            serializable_pair[key] = {
                k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                for k, v in results.items()
            }

        with open(pair_file_path, "w") as f:
            json.dump(serializable_pair, f, indent=4)

        return ind_file_path, pair_file_path


# Example usage
def train_with_checkpointing(model, train_loader, test_loader, criterion, optimizer,
                             scheduler=None, epochs=10, device="cuda",
                             checkpoint_freq=1000, experiment_name=None):
    """
    Training loop with comprehensive checkpointing

    Args:
        model: The model to train
        train_loader: Training data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        epochs: Number of training epochs
        device: Device to train on
        checkpoint_freq: How often to save checkpoints (in steps)
        experiment_name: Name for the experiment
    """
    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_name=experiment_name,
        save_freq=checkpoint_freq
    )

    # Check if we can resume from a checkpoint
    start_epoch = 0
    start_step = 0
    try:
        state_info = ckpt_manager.load_checkpoint()
        start_epoch = state_info["epoch"]
        start_step = state_info["step"]
    except FileNotFoundError:
        print("No checkpoint found, starting training from scratch")

    # Keep track of dataloader state for reproducibility
    # Note: This is a simplified approach and might not work for all dataloaders
    dataloader_state = {
        "train_batch_idx": 0,
        "epoch": start_epoch
    }

    # Move model to device
    model.to(device)

    # Training loop
    step = start_step
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Reset batch idx at the start of epoch
        dataloader_state["train_batch_idx"] = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Update dataloader state
            dataloader_state["train_batch_idx"] = batch_idx

            # Skip batches if resuming within an epoch
            if epoch == start_epoch and batch_idx < dataloader_state["train_batch_idx"]:
                continue

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Update learning rate if using scheduler
            if scheduler is not None:
                scheduler.step()

            # Update metrics
            epoch_loss += loss.item()
            predicted = outputs.argmax(dim=-1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Log info
            if step % 100 == 0:
                train_loss = epoch_loss / (batch_idx + 1)
                train_acc = correct / total

                # Evaluate on test set
                val_loss, val_acc = evaluate(model, test_loader, criterion, device)

                print(f"Step {step}, Epoch {epoch}, Batch {batch_idx}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # Save checkpoint and stats
                ckpt_manager.save_checkpoint(
                    step=step,
                    epoch=epoch,
                    dataloader_state=dataloader_state,
                    train_loss=train_loss,
                    train_accuracy=train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc
                )

                # Perform head attribution analysis every 1000 steps
                if step % 1000 == 0:
                    print("Performing head attribution analysis...")
                    attribution_scores = model.analyze_head_attribution(test_loader)
                    ckpt_manager.save_head_attribution(step, attribution_scores)

                    # Save attention patterns for a few examples
                    sample_inputs, _ = next(iter(test_loader))
                    sample_inputs = sample_inputs[:3].to(device)  # Take first 3 examples

                    # Forward pass with attention capture
                    _ = model(sample_inputs, store_attention=True)
                    attention_patterns = model.get_attention_patterns()

                    # Save patterns
                    ckpt_manager.save_attention_patterns(step, attention_patterns,
                                                         example_ids=list(range(3)))

                # Perform circuit analysis every 5000 steps
                if step % 5000 == 0:
                    print("Performing circuit analysis...")
                    individual_results, pairwise_results, _ = identify_circuits(
                        model, test_loader, baseline_acc=val_acc)
                    ckpt_manager.save_circuit_analysis(step, individual_results, pairwise_results)

            step += 1

        # End of epoch
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"End of Epoch {epoch}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save checkpoint at the end of each epoch
        ckpt_manager.save_checkpoint(
            step=step,
            epoch=epoch + 1,  # Save as the next epoch since we're at the end
            dataloader_state={"train_batch_idx": 0, "epoch": epoch + 1},
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            force_save=True
        )

    # Final checkpoint
    ckpt_manager.save_checkpoint(
        step=step,
        epoch=epochs,
        train_loss=train_loss,
        train_accuracy=train_acc,
        val_loss=val_loss,
        val_accuracy=val_acc,
        extra_data={"status": "completed"},
        force_save=True
    )

    print("Training completed!")
    return model


def evaluate(model, data_loader, criterion, device):
    """Evaluate the model on the provided data loader"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            predicted = outputs.argmax(dim=-1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return running_loss / len(data_loader), correct / total


def identify_circuits(model, eval_loader, baseline_acc=None):
    """Identify potential circuits through head ablation"""
    model.eval()

    # Get baseline accuracy if not provided
    if baseline_acc is None:
        baseline_acc = model.evaluate(eval_loader)

    # All possible head combinations for a 2-layer, 2-head model
    all_heads = ['layer_0_head_0', 'layer_0_head_1', 'layer_1_head_0', 'layer_1_head_1']

    # Individual ablation results
    individual_results = {}
    for head in all_heads:
        layer_idx = int(head.split('_')[1])
        head_idx = int(head.split('_')[3])

        # Store original weights
        original_weights = model.layers[layer_idx].attn.out_proj.weight.clone()

        # Mask this head
        head_dim = model.dim // model.num_heads
        start_idx = head_idx * head_dim
        end_idx = (head_idx + 1) * head_dim

        with torch.no_grad():
            model.layers[layer_idx].attn.out_proj.weight[:, start_idx:end_idx] = 0

        # Evaluate
        ablated_acc = model.evaluate(eval_loader)
        individual_results[head] = baseline_acc - ablated_acc

        # Restore weights
        with torch.no_grad():
            model.layers[layer_idx].attn.out_proj.weight.copy_(original_weights)

    # Pairwise ablation
    pairwise_results = {}
    for i, head1 in enumerate(all_heads):
        for head2_idx in range(i + 1, len(all_heads)):
            head2 = all_heads[head2_idx]

            # Get layer and head indices
            layer_idx1 = int(head1.split('_')[1])
            head_idx1 = int(head1.split('_')[3])
            layer_idx2 = int(head2.split('_')[1])
            head_idx2 = int(head2.split('_')[3])

            # Store original weights
            original_weights1 = model.layers[layer_idx1].attn.out_proj.weight.clone()
            original_weights2 = model.layers[layer_idx2].attn.out_proj.weight.clone()

            # Mask both heads
            head_dim = model.dim // model.num_heads
            start_idx1 = head_idx1 * head_dim
            end_idx1 = (head_idx1 + 1) * head_dim
            start_idx2 = head_idx2 * head_dim
            end_idx2 = (head_idx2 + 1) * head_dim

            with torch.no_grad():
                model.layers[layer_idx1].attn.out_proj.weight[:, start_idx1:end_idx1] = 0
                model.layers[layer_idx2].attn.out_proj.weight[:, start_idx2:end_idx2] = 0

            # Evaluate
            ablated_acc = model.evaluate(eval_loader)
            actual_drop = baseline_acc - ablated_acc
            expected_drop = individual_results[head1] + individual_results[head2]

            # Calculate circuit strength (super-additivity)
            circuit_strength = actual_drop - expected_drop

            pairwise_results[(head1, head2)] = {
                'expected_drop': expected_drop,
                'actual_drop': actual_drop,
                'circuit_strength': circuit_strength
            }

            # Restore weights
            with torch.no_grad():
                model.layers[layer_idx1].attn.out_proj.weight.copy_(original_weights1)
                model.layers[layer_idx2].attn.out_proj.weight.copy_(original_weights2)

    # Identify strongest circuits
    sorted_circuits = sorted(pairwise_results.items(),
                             key=lambda x: x[1]['circuit_strength'],
                             reverse=True)

    return individual_results, pairwise_results, sorted_circuits