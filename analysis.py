import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from IPython.core.pylabtools import figsize
from jupyter_server.services.contents import checkpoints
from torch import optim
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from analysis_transformer import Decoder
from checkpoint_manager import CheckpointManager
from grokking_detection import track_metrics_for_grokking, analyze_grokking_transitions
from modular_data import create_modular_dataloaders
from visualization import plot_attributions, plot_cross_attributions, plot_entropy, visualize_model_analysis


def get_sampler_iter_state(dataloader):
    """
    Try to capture the state of a dataloader's sampler iterator.
    This is inherently difficult in PyTorch.
    """
    # Check if we have access to the sampler
    if not hasattr(dataloader, 'sampler'):
        return None

    sampler = dataloader.sampler

    # Check if sampler is a RandomSampler (most common with shuffle=True)
    if hasattr(sampler, 'generator'):
        # If it has a generator, we can save its state
        return {
            'generator_state': sampler.generator.get_state() if sampler.generator else None
        }

    # For SequentialSampler, we don't need state
    return None

# Training loop with analysis
# todo get train/eval dataloader states dicts
def train_with_analysis_epochs(model: object, train_loader: object, eval_loader: object,
                               dataset_split_indices,
                               criterion: object, optimizer: object, scheduler: object,
                               epochs: object, device: object,
                               checkpointManager,
                               log_interval: int = 5,
                               analyze_interval: int = 50,
                               checkpoint_interval: int = 200):
    """
    Train the model with periodic analysis and logging

    Parameters:
    -----------
    model : Decoder
        The transformer model with analysis capabilities
    train_loader : DataLoader
        Training data loader
    eval_loader : DataLoader
        Evaluation data loader
    criterion : loss function
        Loss function for training
    optimizer : optimizer
        Optimizer for training
    epochs : int
        Number of training epochs
    device : torch.device
        Device to train on
    log_interval : int
        How often to log basic metrics (steps)
    analyze_interval : int
        How often to perform detailed analysis (steps)
    """
    # model.to(device)
    phases = {}
    train_dataloader_state = {
        # Current position information
        'batch_idx': 0,  # Which batch we're on
        'epoch': 0,  # Current epoch
        # Sampler state (critical for reproducibility)
        'sampler_iter_state': get_sampler_iter_state(train_loader),
        # DataLoader configuration (for recreation)
        'batch_size': train_loader.batch_size,
        'shuffle': True,  # or whatever your setting is
        'num_workers': train_loader.num_workers,
        'pin_memory': train_loader.pin_memory,
        'drop_last': train_loader.drop_last
    }
    eval_dataloader_state = {
        # Similar structure but typically simpler since evaluation is often sequential
        'batch_idx': 0,  # Usually reset for each evaluation
        # DataLoader configuration
        'batch_size': eval_loader.batch_size,
        'shuffle': False,  # Typically False for eval
        'num_workers': eval_loader.num_workers,
        'pin_memory': eval_loader.pin_memory,
        'drop_last': eval_loader.drop_last
    }
    min_epochs_for_detection = 100
    grokking_analysis = None
    # info the logging / analyzing / saving integrals must be
    track_metrics_interval = 1

    for epoch in tqdm(range(epochs)):
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
            scheduler.step()
            train_last_token_preds = logits.argmax(dim=-1)
            train_correct += (train_last_token_preds == targets).sum().item()
            train_total += targets.size(0)
            train_loss += loss.item() * targets.size(0)

        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        train_loss = train_loss / train_total if train_total > 0 else 0.0
        train_stats = {'accuracy': train_accuracy, 'loss': train_loss, 'epoch': epoch}
        model.log_stats('training', train_stats)

        if epoch % log_interval == 0:
            # info evaluate each epoch
            model.eval()
            eval_dataloader_state['epoch'] = epoch
            eval_accuracy, eval_loss = model.evaluate(eval_loader)
            eval_stats = {'accuracy': eval_accuracy, 'loss': eval_loss, 'epoch': epoch}
            model.log_stats('evaluation', eval_stats)
            model.train()
            # At the end of each epoch, try to identify grokking phases warning every epoch?
            track_metrics_for_grokking(epoch=epoch, model=model, train_loader=train_loader, eval_loader=eval_loader,)
        """
        # Detailed analysis at less frequent intervals
        # if epoch % analyze_interval == 0:
        #     Store a sample input for visualization
            # sample_input = next(iter(eval_loader))[0].to(device)

            # Calculate head attribution if we're past the early training stage
            # if epoch > 0 and epoch % (analyze_interval * 5) == 0:
                # Visualize attention patterns
                # model.visualize_attention(sample_input, title=f"Attention Patterns at epoch {epoch}")
                # print("Analyzing head attribution...")
                # attribution = model.analyze_head_attribution(eval_loader)
                # plot_attributions(attribution=attribution, epoch=epoch, title=f"Head Attribution at Step {epoch}")
                # cross_attribution = model.analyze_head_cross_attribution(eval_loader)
                # plot_cross_attributions(attribution=cross_attribution, epoch=epoch,
                #                         title=f"Cross-Attention Attribution score (accuracy decrease when both heads masked) at epoch {epoch}")

                # Also calculate attention entropy
                # entropies = model.compute_attention_entropy(eval_loader)
                # plot_entropy(entropies=entropies, epoch=epoch, title=f"Attention Entropy at Step {epoch}")
        """
        if epoch > min_epochs_for_detection and epoch % (25 * log_interval) == 0:
            # info analyze_grokking_transistions() should BEFORE visualize_model_transitions()
            grokking_analysis = analyze_grokking_transitions(model=model, train_loader=train_loader,
                                                             eval_loader=eval_loader)
            visualize_model_analysis(model=model, epoch=epoch,
                                     eval_loader=eval_loader,
                                     include_metrics=['attention', 'attribution', 'cross_attribution', 'entropy',
                                                      'weight_norms', 'accuracy', 'loss'],   # 'grokking_phases'],
                                     save_path=f"{model.save_dir}/comprehensive_visualization_{epoch}.png",
                                     logx=True
                                     )

        if (epoch + 1) % 10 == 0:
            comm = f"\t val_acc: {model.logger.get_last_value('evaluation', 'accuracy'):.5f}\t trn_acc: {model.logger.get_last_value('training', 'accuracy'):.5f}"
            # comm = f"\t val_acc: {model.train_history['accuracy'][-1]:.5f}\t trn_acc: {train_accuracy:.5f} "
            # if phases and 'grokking_step' in phases:
            #     comm = comm + f"\t |\t Grokking at epoch {phases['grokking_step']}"
            if (grokking_analysis is not None and 'primary_grokking_step' in grokking_analysis and
                    grokking_analysis["primary_grokking_step"] is not None):
                primary_grokking_step = grokking_analysis['primary_grokking_step']
                comm = f"{comm}\t Primary grokking detected at epoch {primary_grokking_step}"
            tqdm.write(comm)

        checkpointManager.save_checkpoint(epoch=epoch + 1,
                                          train_dataloader_state=train_dataloader_state,
                                          eval_dataloader_state=eval_dataloader_state,
                                          dataset_split_indices=dataset_split_indices,
                                          train_loss=train_stats['loss'],
                                          train_accuracy=train_stats['accuracy'],
                                          val_loss=eval_stats['loss'],
                                          val_accuracy=eval_stats['accuracy'],
                                          extra_data=None,
                                          force_save=False,
                                          )

    # Final detailed analysis
    print("\nFinal Analysis:")
    model.plot_training_dynamics()
    phases = model.identify_grokking_phase()

    # Compare attention patterns before and after grokking
    if phases:
        # Get sample input for visualization
        sample_input = next(iter(eval_loader))[0].to(device)

        if np.any(phases['pre_grokking_mask']):
            # Find a representative pre-grokking step
            pre_step = phases['pre_grokking_steps'][-1]
            print(f"Analyzing pre-grokking attention (step {pre_step})...")

            # We can't go back in time, so this is just illustrative
            model.visualize_attention(sample_input, title=f"Pre-Grokking Attention (Step {pre_step})")

        print(f"Analyzing post-grokking attention (latest step)...")
        model.visualize_attention(sample_input, title="Post-Grokking Attention (Latest)")

    return model

def create_model(embedding, num_layers, heads_per_layer, num_tokens, seq_len,
                 criterion, device, base_dir="results", init_xavier=False):
    ff = datetime.now().strftime("%f")
    id = f"l{num_layers}_h{heads_per_layer}_e{embedding}_{ff}"

    # info save_dir
    save_dir = Path(base_dir) / f"{id}"
    save_dir.mkdir(exist_ok=True)

    # info checkpoint saves
    checkpoint_dir = Path(save_dir) / f"checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # info stats directory
    stats_dir = Path(save_dir) / "stats"
    stats_dir.mkdir(exist_ok=True)

    model = Decoder(
        dim=embedding,
        num_layers=num_layers,
        num_heads=heads_per_layer,
        num_tokens=num_tokens,
        seq_len=seq_len,
        criterion=criterion,
        device=device,
        id=id,
        save_dir=save_dir,
        checkpoint_dir=checkpoint_dir,
    )
    model.to(device)
    if init_xavier:
        model.apply_xavier_init()

    return model, save_dir, checkpoint_dir, stats_dir


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    model, save_dir, checkpoint_dir, stats_dir = create_model(
        embedding=args.embedding,
        num_layers=args.num_layers,
        heads_per_layer=args.num_heads,
        num_tokens=args.p + 2,
        seq_len=5,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        base_dir="results"
    )

    modulus = args.p

    if args.operation.startswith('mult'):
        operation = "multiply"
    elif args.operation.startswith('divide'):
        operation = "div"
    elif args.operation.startswith('power'):
        operation = "power"
    elif args.operation.startswith('add'):
        operation = "add"
    elif args.operation.startswith('subtract'):
        operation = "subtract"
    else:
        operation = "multiply"

    train_loader, test_loader, vocab_size, dataset_split_indices = create_modular_dataloaders(
        modulus=modulus,
        op=operation,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        sequence_format=True, seed=args.seed
    )

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )
    checkpointManager = CheckpointManager(model=model, optimizer=optimizer, scheduler=scheduler,
                                          experiment_name=model.get_id(),
                                          save_dir=save_dir,
                                          checkpoint_dir=checkpoint_dir,
                                          stats_dir=stats_dir,
                                          save_freq=args.checkpoint_interval)
    model = train_with_analysis_epochs(model=model,
                                       train_loader=train_loader, eval_loader=test_loader,
                                       dataset_split_indices=dataset_split_indices,
                                       criterion=nn.CrossEntropyLoss(),
                                       optimizer=optimizer, scheduler=scheduler,
                                       epochs=args.epochs,
                                       device=device,
                                       checkpointManager=checkpointManager,
                                       log_interval=args.log_interval,
                                       analyze_interval=args.analyze_interval,
                                       checkpoint_interval=args.checkpoint_interval, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # architecture parameters
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)

    # run params
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=3e5)
    parser.add_argument("--batch_size", type=int, default=256)   #512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--train_ratio", type=float, default=0.5)

    # Grokfast
    parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir"], default="none")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)

    # analysis intervals
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--analyze_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=100)

    parser.add_argument("--operation", type=str, default='multiply')

    # Ablation studies
    parser.add_argument("--two_stage", action='store_true')
    parser.add_argument("--save_weights", action='store_true')
    args = parser.parse_args()

    filter_str = ('_' if args.label != '' else '') + args.filter
    window_size_str = f'_w{args.window_size}'
    alpha_str = f'_a{args.alpha:.3f}'.replace('.', '')
    lamb_str = f'_l{int(args.lamb)}'

    if args.filter == 'none':
        filter_suffix = ''
    elif args.filter == 'ma':
        filter_suffix = window_size_str + lamb_str
    elif args.filter == 'ema':
        filter_suffix = alpha_str + lamb_str
    else:
        raise ValueError(f"Unrecognized filter type {args.filter}")

    optim_suffix = ''
    if args.weight_decay != 0:
        optim_suffix = optim_suffix + f'_wd{args.weight_decay:.1e}'.replace('.', '')
    if args.lr != 1e-3:
        optim_suffix = optim_suffix + f'_lrx{int(args.lr / 1e-3)}'

    args.label = args.label + filter_str + filter_suffix + optim_suffix
    print(f'Experiment results saved under name: {args.label}')

    main(args)