import argparse

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from IPython.core.pylabtools import figsize
from torch import optim
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from analysis_transformer import Decoder
from modular_data import create_modular_dataloaders


def plot_attributions(attribution, epoch, title):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    heads = list(attribution.keys())
    scores = [attribution[h] for h in heads]
    plt.bar(heads, scores)
    plt.xlabel('Attention Head')
    plt.ylabel('Attribution Score (decrease in accuracy when masked)')
    plt.title(f'Head Attribution at Step {epoch}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_cross_attributions(attribution, epoch, title):
    # numpy array to Dataframe
    df = pd.DataFrame(attribution)
    df.columns = ['layer_f', 'head_f', 'layer_s', 'head_s', 'cross_attn_score']
    df["first"] = df["layer_f"].astype(int).astype(str) + ':' + df["head_f"].astype(int).astype(str)
    df["second"] = df["layer_s"].astype(int).astype(str) + ':' + df["head_s"].astype(int).astype(str)
    df = df[["first", "second", "cross_attn_score"]]
    pivot_df = pd.pivot_table(df, values='cross_attn_score', index=['first'], columns=['second'])
    mean_val = df["cross_attn_score"].mean()
    pivot_df = pivot_df.fillna(mean_val)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", square=True, cmap="mako", ax=ax)
    plt.title(f'{title}')
    plt.tight_layout()
    plt.show()


def plot_entropy(entropies, epoch, title):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    heads = list(entropies.keys())
    entropy_vals = [entropies[h] for h in heads]
    plt.bar(heads, entropy_vals, color='skyblue')
    plt.xlabel('Attention Head')
    plt.ylabel('Entropy')
    plt.title(f'{title} at Step {epoch} (lower = more specialized)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Training loop with analysis
def train_with_analysis(model, train_loader, eval_loader, criterion, optimizer,
                        epochs, device,
                        log_interval=20, analyze_interval=100, analyze_plot_interval=200):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )
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
    model.to(device)
    step = 0

    for epoch in range(epochs):
        model.train()
        # metric = MulticlassAccuracy(num_classes=args.p + 1)
        # loss_sum = torch.zeros(1, device=device)
        # all_inputs = torch.zeros(1, device=device)
        # total_acc = torch.zeros(1, device=device)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            # Basic logging at regular intervals
            if step % log_interval == 0:
                # Evaluate
                model.eval()
                with torch.no_grad():
                    eval_accuracy, eval_loss = model.evaluate(eval_loader)
                model.train()

                # Log stats
                model.log_training_stats(loss=loss.item(), accuracy=eval_accuracy)

                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, Accuracy: {eval_accuracy:.4f}")

            # Detailed analysis at less frequent intervals
            if step % analyze_interval == 0:
                # Store a sample input for visualization
                sample_input = next(iter(eval_loader))[0].to(device)

                # Visualize attention patterns
                model.visualize_attention(sample_input, title=f"Attention Patterns at Step {step}")

                # Calculate head attribution if we're past the early training stage
                if step > 0 and step % (analyze_interval * 5) == 0:
                    print("Analyzing head attribution...")
                    attribution = model.analyze_head_attribution(eval_loader)

                    # Plot attribution scores
                    plt.figure(figsize=(10, 6))
                    heads = list(attribution.keys())
                    scores = [attribution[h] for h in heads]
                    plt.bar(heads, scores)
                    plt.xlabel('Attention Head')
                    plt.ylabel('Attribution Score (decrease in accuracy when masked)')
                    plt.title(f'Head Attribution at Step {step}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()

                    # Also calculate attention entropy
                    entropies = model.compute_attention_entropy(eval_loader)

                    # Plot entropy
                    plt.figure(figsize=(10, 6))
                    heads = list(entropies.keys())
                    entropy_vals = [entropies[h] for h in heads]
                    plt.bar(heads, entropy_vals)
                    plt.xlabel('Attention Head')
                    plt.ylabel('Attention Entropy')
                    plt.title(f'Attention Entropy at Step {step} (lower = more specialized)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()

            step += 1

        # At the end of each epoch, try to identify grokking phases
        if epoch > 0 and len(model.train_history['accuracy']) > 20:
            print(f"\nAnalyzing training dynamics after epoch {epoch}...")
            model.plot_training_dynamics()
            phases = model.identify_grokking_phase()

            if phases and 'grokking_step' in phases:
                print(f"Potential grokking detected at step {phases['grokking_step']}")

        print(f"Completed epoch {epoch}")

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


# Training loop with analysis
def train_with_analysis_epochs(model, train_loader, eval_loader, criterion, optimizer,
                               epochs, device,
                               log_interval=20, analyze_interval=50):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )
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
    model.to(device)
    phases = {}

    for epoch in tqdm(range(epochs)):
        model.train()
        # metric = MulticlassAccuracy(num_classes=args.p + 1)
        # loss_sum = torch.zeros(1, device=device)
        # all_inputs = torch.zeros(1, device=device)
        # total_acc = torch.zeros(1, device=device)
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
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

        # info evaluate each epoch
        # Evaluate
        model.eval()
        with torch.no_grad():
            eval_accuracy, eval_loss = model.evaluate(eval_loader)
        model.train()
        eval_stats = {'accuracy': eval_accuracy, 'loss': eval_loss, 'epoch': epoch}
        model.log_stats('evaluation', eval_stats)

        # Log stats
        # model.log_training_stats(epoch, loss=loss.item(), accuracy=accuracy)

        # Detailed analysis at less frequent intervals
        if epoch % analyze_interval == 0:
            # Store a sample input for visualization
            sample_input = next(iter(eval_loader))[0].to(device)

            # Calculate head attribution if we're past the early training stage
            if epoch > 0 and epoch % (analyze_interval * 5) == 0:
                # Visualize attention patterns
                model.visualize_attention(sample_input, title=f"Attention Patterns at epoch {epoch}")
                # print("Analyzing head attribution...")
                attribution = model.analyze_head_attribution(eval_loader)
                plot_attributions(attribution=attribution, epoch=epoch, title=f"Head Attribution at Step {epoch}")
                cross_attribution = model.analyze_head_cross_attribution(eval_loader)
                plot_cross_attributions(attribution=cross_attribution, epoch=epoch,
                                        title=f"Cross-Attention Attribution score (accuracy decrease when both heads masked) at epoch {epoch}")

                # Also calculate attention entropy
                entropies = model.compute_attention_entropy(eval_loader)
                plot_entropy(entropies=entropies, epoch=epoch, title=f"Attention Entropy at Step {epoch}")

        # At the end of each epoch, try to identify grokking phases
        if (epoch > 0 and model.logger.get_length('training', 'accuracy') > 20 and
                epoch % (5 * analyze_interval) == 0):
            # print(f"\nAnalyzing training dynamics after epoch {epoch}...")
            # model.plot_training_dynamics()
            phases = model.identify_grokking_phase()

            # if phases and 'grokking_step' in phases:
            #     print(f"Potential grokking detected at epoch {phases['grokking_step']}")

        if epoch % 10 == 0:
            comm = f"\t val_acc: {model.logger.get_last_value('evaluation', 'accuracy'):.5f}\t trn_acc: {model.logger.get_last_value('training', 'accuracy'):.5f}"
            # comm = f"\t val_acc: {model.train_history['accuracy'][-1]:.5f}\t trn_acc: {train_accuracy:.5f} "
            if phases and 'grokking_step' in phases:
                comm = comm + f"\t |\t Grokking at epoch {phases['grokking_step']}"
            tqdm.write(comm)

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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Decoder(
        dim=args.embedding,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_tokens=args.p + 2,
        seq_len=5,
        criterion=nn.CrossEntropyLoss()
    ).to(device)

    modulus = args.p

    # op = 'add'
    # op = 'subtract'
    op = 'multiply'
    # op = 'power'
    # op = 'divide'

    train_loader, test_loader, vocab_size = create_modular_dataloaders(
        modulus=modulus,
        op=op,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        sequence_format=True, seed=42
    )

    model = train_with_analysis_epochs(model=model,
                                       train_loader=train_loader, eval_loader=test_loader,
                                       criterion=nn.CrossEntropyLoss(),
                                       optimizer=optim.Adam(model.parameters(), lr=args.lr),
                                       epochs=args.epochs,
                                       device=device,
                                       log_interval=args.log_interval,
                                       analyze_interval=args.analyze_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # architecture parameters
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)

    # run params
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=3e5)
    parser.add_argument("--batch_size", type=int, default=512)
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
    parser.add_argument("--log_interval", type=int, default=20)

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

    args = parser.parse_args()

    main(args)