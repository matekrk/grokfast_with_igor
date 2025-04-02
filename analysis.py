import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

from torch import optim
from tqdm import tqdm

from analysis_transformer import Decoder
from checkpoint_manager import CheckpointManager, GrokAwareCheckpointManager
from grokking_detection import track_metrics_for_grokking, analyze_grokking_transitions
from modular_data import create_modular_dataloaders
from track_gradients import WeightSpaceTracker, CyclicBehaviorDetector, analyze_gradient_flow, analyze_loss_curvature
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

def init_train_dataloader_state(dataloader):
    return {
        # Current position information
        'batch_idx': 0,  # Which batch we're on
        'epoch': 0,  # Current epoch
        # Sampler state (critical for reproducibility)
        'sampler_iter_state': get_sampler_iter_state(dataloader),
        # DataLoader configuration (for recreation)
        'batch_size': dataloader.batch_size,
        'shuffle': True,  # or whatever your setting is
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
        'drop_last': dataloader.drop_last
    }

def init_val_dataloader_state(dataloader):
    return {
        # Similar structure but typically simpler since evaluation is often sequential
        'batch_idx': 0,  # Usually reset for each evaluation
        # DataLoader configuration
        'batch_size': dataloader.batch_size,
        'shuffle': False,  # Typically False for eval
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
        'drop_last': dataloader.drop_last
    }

def train_with_analysis(model: object, train_loader: object, eval_loader: object,
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
    train_dataloader_state = init_train_dataloader_state(dataloader=train_loader)
    eval_dataloader_state = init_val_dataloader_state(dataloader=eval_loader)
    min_epochs_for_detection = 100
    grokking_analysis = None
    track_gradient = True
    gradient_tracker = []
    weight_tracker = WeightSpaceTracker(model=model, save_dir=checkpointManager.checkpoint_dir)
    cyclic_detector = CyclicBehaviorDetector(window_size=100, min_cycles=2)

    for epoch in tqdm(range(epochs)):
        train_dataloader_state['epoch'] = epoch
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        model.train()
        # warning the examples are organized with EXAMPLES AS ROWS where each row is a sequence,
        #  with row[:-1] is the input, while [-1] is the target
        #  therefore in the forward model it needs to be transposed!
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

        if epoch % log_interval == 0:
            # info log train and eval statistics at the same epochs,
            #  to have the same number of logs; it might fail otherwise todo repair analyze_grokking_transitions()
            # todo check if it really is true -- seems to be working with different (?)
            train_stats = {'accuracy': train_accuracy, 'loss': train_loss, 'epoch': epoch}
            model.log_stats('training', train_stats)
            model.eval()
            eval_dataloader_state['epoch'] = epoch
            eval_accuracy, eval_loss = model.evaluate(eval_loader)
            eval_stats = {'accuracy': eval_accuracy, 'loss': eval_loss, 'epoch': epoch}
            model.log_stats('evaluation', eval_stats)
            model.train()
            # info now try to identify grokking phases
            track_metrics_for_grokking(epoch=epoch, model=model, train_loader=train_loader, eval_loader=eval_loader,)

            # info weight gradient tracking
            if track_gradient:
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    # info now analyze gradients
                    grad_stats = analyze_gradient_flow(model=model, loss=loss, optimizer=optimizer)
                    gradient_tracker.append({'epoch': epoch, 'stats': grad_stats})
                    # info track metrics for cycle detection
                    cyclic_detector.add_metric('total_grad_norm', grad_stats['total_grad_norm'], epoch=epoch)
                    for layer_idx, norm in enumerate(grad_stats['layer_grad_norms']):
                        cyclic_detector.add_metric(f'layer_{layer_idx}_grad_norm', norm, epoch=epoch)
                    break  # analyze only one batch
                # info weight space snapshot
                weight_tracker.take_snapshot(epoch=epoch)
                # cycle tracking detection
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        cyclic_detector.add_metric(f'{name}_norm', param.norm().item(), epoch=epoch)
                # loss landscape analysis
                if epoch % 10 * log_interval == 0:
                    for inputs, targets in train_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        curvature = analyze_loss_curvature(model=model, inputs=inputs, targets=targets, criterion=criterion)
                        # track eigenvalues
                        cyclic_detector.add_metric('max_curvature', curvature['max_curvature'], epoch=epoch)
                        break


        if epoch > min_epochs_for_detection and epoch % (40 * log_interval) == 0:
            # info analyze_grokking_transistions() should BEFORE visualize_model_transitions()
            grokking_analysis = analyze_grokking_transitions(model=model, train_loader=train_loader,
                                                             eval_loader=eval_loader)
            # If grokking was detected, update the checkpoint manager
            if grokking_analysis is not None and 'primary_grokking_step' in grokking_analysis:
                primary_grokking_step = grokking_analysis['primary_grokking_step']
                if primary_grokking_step is not None:
                    checkpointManager.update_grokking_points(primary_grokking_step)
                    if track_gradient and cyclic_detector is not None:
                        if cyclic_detector.has_enough_data():
                            cycle_results = cyclic_detector.detect_cycles()
                            cyclic_detector.visualize_cycles(save_dir=checkpointManager.checkpoint_dir)
                            highlight_epochs = {
                                # info grokking steps  # todo add all (how to add the primary one to mark differently?)
                                'grok': [primary_grokking_step],
                                # info add rapid change steps detected with cyclic_detector
                                'peaks': cycle_results['layers.0.attn.in_proj_weight_norm']['peak_epochs'],
                            }
                            weight_tracker.visualize_trajectory(
                                highlight_epochs=highlight_epochs,
                            )

            visualize_model_analysis(model=model, epoch=epoch,
                                     eval_loader=eval_loader,
                                     include_metrics=['attention', 'attribution', 'cross_attribution', 'entropy',
                                                      'weight_norms', 'accuracy', 'loss', 'grokking_phases'],
                                     save_path=f"{model.save_dir}/comprehensive_visualization_{epoch}.png",
                                     logx=False
                                     )

        if (epoch + 1) % 10 == 0:
            # todo add some mean value counterpart to the logger.get_last_value() method
            comm = f"\tval_acc: {model.logger.get_last_value('evaluation', 'accuracy'):.4f}\t trn_acc: {model.logger.get_last_value('training', 'accuracy'):.4f}"
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
    visualize_model_analysis(model=model, epoch=epoch,
                             eval_loader=eval_loader,
                             include_metrics=['attention', 'attribution', 'cross_attribution', 'entropy',
                                              'weight_norms', 'accuracy', 'loss'],  # 'grokking_phases'],
                             save_path=f"{model.save_dir}/comprehensive_visualization_final.png",
                             logx=True
                             )

    return model

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


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

    def __repr__(self):
        return "LabelSmoothingLoss(classes={}, smoothing={})".format(self.classes, self.smoothing)

def main(args):
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
        init_xavier=False,)
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
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.5)

    # Grokfast
    # parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir"], default="none")
    # parser.add_argument("--alpha", type=float, default=0.99)
    # parser.add_argument("--window_size", type=int, default=100)
    # parser.add_argument("--lamb", type=float, default=5.0)

    # analysis intervals
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--analyze_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=100)

    parser.add_argument("--operation", type=str, default='multiply')

    # Ablation studies
    parser.add_argument("--two_stage", action='store_true')
    parser.add_argument("--save_weights", action='store_true')
    args = parser.parse_args()

    main(args)