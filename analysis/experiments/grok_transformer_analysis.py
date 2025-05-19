import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim

# warning is this import OK?
from analysis.models.analysis_transformer import Decoder
from analysis.models.modular_data import create_modular_dataloaders
from analysis.staged_experiment import staged_experiment_framework
from analysis.utils.checkpoint_manager import GrokAwareCheckpointManager
from analysis.utils.utils import create_model, create_optimizer, create_scheduler, find_closest_jump
from analysis.visualization.visualize_phases_anthropic_style import create_phase_visualizations

from analysis.trainers.analysis import train_with_analysis
from analysis.trainers.train_enhanced_weight_analysis import train_with_enhanced_analysis
from analysis.trainers.train_phase_analysis import train_with_phase_analysis, train_with_enhanced_phase_analysis


def main(args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        base_dir="../../results",
        init_xavier=False,
    )

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

    config = {
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'operation': operation,
        'modulus': args.p,
        'train_ratio': args.train_ratio,
        'batch_size': args.batch_size,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'heads_per_layer': args.num_heads,
        'seed': args.seed,
    }
    # Choose optimizer
    optimizer = create_optimizer(model=model, config=config)
    config['optimizer'] = optimizer
    scheduler = create_scheduler(config=config)

    train_loader, eval_loader, vocab_size, dataset_split_indices = create_modular_dataloaders(
        modulus=config['modulus'],
        op=config['operation'],
        train_ratio=config['train_ratio'],
        batch_size=config['batch_size'],
        sequence_format=True,
        seed=config['seed'],
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
        grokking_window=40,  # Save checkpoints 50 epochs before/after grokking
        max_to_keep=5,  # maximum number of checkpoint files (groups) to keep
    )

    if args.mode == 'enhanced' and args.type == 'weight':
        model, weight_tracker, jump_analyzer = main_with_enhanced_weight_tracking(args, model=model,
                                                                                  train_loader=train_loader,
                                                                                  eval_loader=eval_loader,
                                                                                  criterion=criterion,
                                                                                  optimizer=optimizer,
                                                                                  scheduler=scheduler, device=device,
                                                                                  checkpointManager=checkpointManager,
                                                                                  dataset_split_indices=dataset_split_indices)
    elif args.mode == 'default' and args.mode == 'phase':
        model, weight_tracker, phase_analyzer, phase_weight_analysis = main_with_phase_tracking(
            args,
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpointManager=checkpointManager,
            dataset_split_indices=dataset_split_indices,
        )
    elif args.mode == 'enhanced' and args.type == 'phase':
        model, weight_tracker, phase_analyzer, phase_weight_analysis = main_with_enhanced_phase_tracking(
            args,
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpointManager=checkpointManager,
            dataset_split_indices=dataset_split_indices,
            # jump_analyzer=jump_analyzer,
            # phase_weight_analysis=phase_weight_analysis,
        )
    else:
        model = main_with_analysis(
            args=args,
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            dataset_split_indices=dataset_split_indices,
            checkpointManager=checkpointManager,
        )
    return


def main_with_analysis(args, model, train_loader, eval_loader, criterion,
                       optimizer, scheduler, device,
                       dataset_split_indices, checkpointManager):
    model = train_with_analysis(model=model,
                                train_loader=train_loader,
                                eval_loader=eval_loader,
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
    return model


def main_with_phase_tracking(args, model, train_loader, eval_loader, criterion,
                             optimizer, scheduler, device,
                             dataset_split_indices, checkpointManager):
    model, weight_tracker, phase_analyzer, phase_weight_analysis = train_with_phase_analysis(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        dataset_split_indices=dataset_split_indices,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        checkpointManager=checkpointManager,
        log_interval=args.log_interval,
        analyze_interval=args.analyze_interval,
        checkpoint_interval=args.checkpoint_interval
    )
    # info get learning phase summary
    summary = phase_analyzer.get_learning_phase_summary()

    # info print insights
    print("\nLearning Phase Insights:")
    for insight in summary['insights']:
        print(f" - {insight}")

    # info create the complete set of visualizations
    visualization_results = create_phase_visualizations(
        model=model,
        phase_analyzer=phase_analyzer,
        weight_tracker=weight_tracker,
        save_dir=model.save_dir / "phase_visualizations"
    )

    return model, weight_tracker, phase_analyzer, phase_weight_analysis


def main_with_enhanced_phase_tracking(args, model, train_loader, eval_loader, criterion,
                                      optimizer, scheduler, device,
                                      dataset_split_indices, checkpointManager):
    model, weight_tracker, enhanced_phase_analyzer, phase_weight_analysis = train_with_enhanced_phase_analysis(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        dataset_split_indices=dataset_split_indices,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpointManager=checkpointManager,
        epochs=args.epochs,
        log_interval=args.log_interval,
        analyze_interval=args.analyze_interval,
        checkpoint_interval=args.checkpoint_interval,
        # jump_detection_threshold=args.jump_detection_threshold,
        # circuit_sampling_freq=args.circuit_sampling_freq,
        # sparsity_threshold=args.sparsity_threshold,
    )
    # info get learning phase summary
    summary = enhanced_phase_analyzer.get_learning_phase_summary()

    # info print insights
    print("\t\tLearning Phase Insights:")
    for insight in summary['insights']:
        print(f"\t\t{insight}")

    # info create the complete set of visualizations
    visualization_results = create_phase_visualizations(
        model=model,
        phase_analyzer=enhanced_phase_analyzer,
        weight_tracker=weight_tracker,
        save_dir=model.save_dir / "phase_visualizations"
    )

    return model, weight_tracker, enhanced_phase_analyzer, phase_weight_analysis


# def main_with_enhanced_tracking(args):
def main_with_enhanced_weight_tracking(args, model, train_loader, eval_loader,
                                       criterion, optimizer, scheduler,  # checkpoint_dir, stats_dir,
                                       device,  # checkpoint_interval,
                                       checkpointManager,
                                       # vocab_size,
                                       dataset_split_indices):
    # Use the enhanced training function
    model, weight_tracker, jump_analyzer = train_with_enhanced_analysis(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        dataset_split_indices=dataset_split_indices,  # fixme is it needed?
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


def main_with_staged_analysis(args):
    """Run experiment with staged analysis"""
    # Base configuration
    base_config = {
        'embedding': args.embedding,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'operation': args.operation,
        'p': args.p,
        'batch_size': args.batch_size,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'train_ratio': args.train_ratio,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'save_dir': f"../../results/staged_{args.operation}_{args.p}",
        'checkpoint_epoch': 1000,  # Epoch to save initial checkpoint
        'max_epochs': args.epochs,
        'log_interval': args.log_interval,
        'analyze_interval': args.analyze_interval,
        'checkpoint_interval': args.checkpoint_interval
    }

    # Analysis variants
    analysis_variants = {
        'sparsity_focus': {
            'type': 'phase',
            'save_dir': f"../../results/staged_{args.operation}_{args.p}/sparsity_focus",
            'circuit_sampling_freq': 20,
            'snapshot_freq': 25,
            'sliding_window_size': 20,
            'jump_threshold': 1.5
        },
        'circuit_class_focus': {
            'type': 'phase',
            'save_dir': f"../../results/staged_{args.operation}_{args.p}/circuit_focus",
            'circuit_sampling_freq': 10,  # More frequent circuit sampling
            'snapshot_freq': 25,
            'sliding_window_size': 20,
            'jump_threshold': 1.5
        }
    }

    # Run staged experiment
    results = staged_experiment_framework(
        base_config=base_config,
        analysis_variants=analysis_variants
    )

    # Print summary of results
    for variant_name, variant_results in results.items():
        print(f"\n======= Results for {variant_name} =======")

        if 'summary' in variant_results:
            summary = variant_results['summary']

            print("\nLearning Phase Insights:")
            for insight in summary.get('insights', []):
                print(f" - {insight}")

            # Print enhanced insights if available
            if 'enhanced_insights' in summary:
                print("\nEnhanced Insights:")
                for insight in summary['enhanced_insights']:
                    print(f" - {insight}")

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # architecture parameters
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)

    # run params
    parser.add_argument("--label", default="")  # fixme ?
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p", type=int, default=97)  # fixme ?
    parser.add_argument("--budget", type=int, default=3e5)  # fixme ?
    parser.add_argument("--batch_size", type=int, default=256)  # 512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)  # fixme ?
    parser.add_argument("--beta2", type=float, default=0.98)  # fixme ?
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.5)

    # parser.add_argument("--enhanced", action="store_true", help="Run enhanced analysis")
    # parser.add_argument("--phase", action="store_true", help="Run phase transition analysis")
    parser.add_argument('--mode',
                        choices=['enhanced', 'default'],  # The three possible values
                        default='enhanced',  # Default value
                        help='Set the analysis type: enhanced [default] or standard')
    parser.add_argument('--type',
                        choices=['phase', 'weight'],  # The three possible values
                        default='phase',  # Default value
                        help='Set the analysis mode: phase [default] or weight')

    # analysis intervals
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--analyze_interval", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=4)
    parser.add_argument("--checkpoint_interval", type=int, default=200)

    parser.add_argument("--operation", type=str, default='multiply')

    # Ablation studies
    parser.add_argument("--two_stage", action='store_true')  # fixme ?
    parser.add_argument("--save_weights", action='store_true')  # fixme ?
    args = parser.parse_args()


    main(args)
