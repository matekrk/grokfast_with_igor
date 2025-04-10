import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from analysis import train_with_analysis
from analysis_transformer import Decoder
from checkpoint_manager import GrokAwareCheckpointManager
from enhanced_analysis import train_with_enhanced_analysis
from grokking_detection import track_metrics_for_grokking, analyze_grokking_transitions
from modular_data import create_modular_dataloaders
from track_gradients import WeightSpaceTracker, CyclicBehaviorDetector, analyze_gradient_flow, analyze_loss_curvature
from utils import init_train_dataloader_state, init_val_dataloader_state
from visualization import visualize_model_analysis


def create_model(embedding, num_layers, heads_per_layer, batch_size,
                 operation, num_tokens, seq_len, ratio,
                 criterion, device,
                 optimizer_name, scheduler_name,
                 learning_rate, weight_decay,
                 base_dir="results", init_xavier=False,
                 ):
    ff = datetime.now().strftime("%f")
    xavier = '_xavier' if init_xavier else ''
    sched = '_sched' if init_xavier else '_nosched'
    optim = f'_{optimizer_name}' if optimizer_name else ''
    sched = f'_{scheduler_name}' if scheduler_name else ''
    lr = f'_lr{learning_rate}' if learning_rate > 0 else ''
    wd = f'_wd{weight_decay:.1g}' if weight_decay > 0 else ''
    id = f"l{num_layers}_h{heads_per_layer}_e{embedding}_b{batch_size}_{operation[:4]}{xavier}{optim}{lr}{wd}{sched}_r{ratio}_{ff}"

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


def main_with_analysis(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # todo use label smoothing to fight overconfidence, eg. LabelSmoothing loss above
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingLoss(args.classes, args.smoothing)

    torch.manual_seed(args.seed)
    model, save_dir, checkpoint_dir, stats_dir = create_model(
        embedding=args.embedding,
        num_layers=args.num_layers,
        heads_per_layer=args.num_heads,
        batch_size=args.batch_size,
        operation=args.operation,
        num_tokens=args.p + 2,
        seq_len=5,
        ratio=args.train_ratio,
        criterion=criterion,
        device=device,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        base_dir="results",
        init_xavier=False, )
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        ValueError("Optimizer '{}' not recognized".format(args.optimizer))

    if args.scheduler is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda update: 1 if update > 10 else update / 10
        )
    else:
        scheduler = None

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

    checkpointManager = GrokAwareCheckpointManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_name=model.get_id(),
        save_dir=save_dir,
        checkpoint_dir=checkpoint_dir,
        stats_dir=stats_dir,
        save_freq=args.checkpoint_interval,
        grokking_window=120,  # Save checkpoints 50 epochs before/after grokking
        max_to_keep=80
    )

    model = train_with_analysis(model=model,
                                train_loader=train_loader, eval_loader=test_loader,
                                dataset_split_indices=dataset_split_indices,
                                criterion=criterion,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epochs=args.epochs,
                                device=device,
                                checkpointManager=checkpointManager,
                                log_interval=args.log_interval,
                                analyze_interval=args.analyze_interval,
                                checkpoint_interval=args.checkpoint_interval, )


# Example of how to modify your existing code to use the enhanced tracking
# This is a skeleton example that you would integrate into your analysis.py
def main_with_enhanced_tracking(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(args.seed)
    model, save_dir, checkpoint_dir, stats_dir = create_model(
        embedding=args.embedding,
        num_layers=args.num_layers,
        heads_per_layer=args.num_heads,
        batch_size=args.batch_size,
        operation=args.operation,
        num_tokens=args.p + 2,
        seq_len=5,
        ratio=args.train_ratio,
        criterion=criterion,
        device=device,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        base_dir="results",
        init_xavier=False,
    )

    # Choose optimizer
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        ValueError("Optimizer '{}' not recognized".format(args.optimizer))

    # Setup scheduler if specified
    if args.scheduler is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda update: 1 if update > 10 else update / 10
        )
    else:
        scheduler = None

    # Create dataloaders
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
        sequence_format=True,
        seed=args.seed
    )

    # Initialize checkpoint manager
    checkpointManager = GrokAwareCheckpointManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_name=model.get_id(),
        save_dir=save_dir,
        checkpoint_dir=checkpoint_dir,
        stats_dir=stats_dir,
        save_freq=args.checkpoint_interval,
        grokking_window=120,  # Save checkpoints 50 epochs before/after grokking
        max_to_keep=80
    )

    # Use the enhanced training function
    model, weight_tracker, jump_analyzer = train_with_enhanced_analysis(
        model=model,
        train_loader=train_loader,
        eval_loader=test_loader,
        dataset_split_indices=dataset_split_indices,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        checkpointManager=checkpointManager,
        log_interval=args.log_interval,
        analyze_interval=args.analyze_interval,
        jump_detection_threshold=2.0,  # Detect jumps that are 2 std devs above normal
        checkpoint_interval=args.checkpoint_interval,
    )

    # Print summary of detected jumps
    jump_summary = weight_tracker.get_jump_summary()
    if jump_summary is not None:
        print("\nDetected weight space jumps:")
        print(jump_summary)

        # Correlate jumps with grokking phases if available
        if 'grokking_phases' in model.logger.logs:
            grokking_step = model.logger.logs['grokking_phases'].get('grokking_step')
            if grokking_step:
                print(f"\nGrokking detected at epoch {grokking_step}")
                closest_jump = find_closest_jump(jump_summary['epoch'].values, grokking_step)
                print(f"Closest jump to grokking point is at epoch {closest_jump}")

    return model, weight_tracker, jump_analyzer

def find_closest_jump(jump_epochs, target_epoch):
    """Find the jump epoch closest to the target epoch"""
    return jump_epochs[np.argmin(np.abs(np.array(jump_epochs) - target_epoch))]




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
    parser.add_argument("--batch_size", type=int, default=256)  # 512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.5)

    parser.add_argument("--enhanced", action="store_true", help="Run enhanced analysis")

    # Grokfast
    # parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir"], default="none")
    # parser.add_argument("--alpha", type=float, default=0.99)
    # parser.add_argument("--window_size", type=int, default=100)
    # parser.add_argument("--lamb", type=float, default=5.0)

    # analysis intervals
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--analyze_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=100)

    parser.add_argument("--operation", type=str, default='multiply')

    # Ablation studies
    parser.add_argument("--two_stage", action='store_true')
    parser.add_argument("--save_weights", action='store_true')
    args = parser.parse_args()

    if args.enhanced:
        main_with_enhanced_tracking(args)
    else:
        main_with_analysis(args)
