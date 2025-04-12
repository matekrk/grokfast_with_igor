import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from analysis_transformer import Decoder
from checkpoint_manager import GrokAwareCheckpointManager
from enhanced_analysis import train_with_enhanced_analysis
from grokking_detection import track_metrics_for_grokking, analyze_grokking_transitions
from modular_data import create_modular_dataloaders
from track_gradients import WeightSpaceTracker, CyclicBehaviorDetector, analyze_gradient_flow, analyze_loss_curvature
from utils import init_train_dataloader_state, init_val_dataloader_state
from visualization import visualize_model_analysis


# Import the enhanced weight tracker and jump analysis tools

# info run analysis for several different points
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
            track_metrics_for_grokking(epoch=epoch, model=model, train_loader=train_loader, eval_loader=eval_loader, )

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
                        curvature = analyze_loss_curvature(model=model, inputs=inputs, targets=targets,
                                                           criterion=criterion)
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

    # info final detailed analysis
    print("\nFinal Analysis:")
    visualize_model_analysis(model=model, epoch=epoch,
                             eval_loader=eval_loader,
                             include_metrics=['attention', 'attribution', 'cross_attribution', 'entropy',
                                              'weight_norms', 'accuracy', 'loss'],  # 'grokking_phases'],
                             save_path=f"{model.save_dir}/comprehensive_visualization_final.png",
                             logx=True
                             )

    return model

