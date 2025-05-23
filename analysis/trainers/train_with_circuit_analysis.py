# train_with_circuit_analysis.py
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from analysis.core.circuit_registry import CircuitRegistry
from analysis.analyzers.integrated_token_discovery import IntegratedTokenCircuitDiscovery
from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
from analysis.analyzers.continuous_circuit_tracker import ContinuousCircuitTracker
from analysis.analyzers.attention_pattern_analyzer import AttentionAnalyzer
from analysis.trainers.utils import (
    evaluate, log_metrics, train_epoch, detect_grokking, process_jumps
)
from analysis.utils.utils import init_train_dataloader_state, get_current_callable_info, shorten_layer_head


def train_with_circuit_analysis(
        model, train_loader, eval_loader,
        dataset_split_indices,
        criterion,
        optimizer, scheduler=None,
        device='cuda', checkpointManager=None,
        epochs=10000,
        log_interval=4,
        analyze_interval=2,
        circuit_sampling_freq=20,
        checkpoint_interval=200):
    """
    Train a transformer model while analyzing circuit formation and evolution

    Args:
        model: The transformer model
        train_loader: Training data loader
        eval_loader: Evaluation data loader
        dataset_split_indices: Dataset split indices
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epochs: Number of training epochs
        device: Device to train on
        checkpointManager: Checkpoint manager
        log_interval: How often to log metrics
        analyze_interval: How often to perform circuit analysis
        checkpoint_interval: How often to save checkpoints

    Returns:
        tuple: (model, analysis_results)
    """
    # print(f"Starting training with circuit analysis for {epochs} epochs...")
    print(f"\t{get_current_callable_info()}: \tstart")

    # Set up save directories
    if checkpointManager:
        save_dir = Path(checkpointManager.experiment_dir)
    else:
        save_dir = Path("results/circuit_analysis")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataloader states
    train_dataloader_state = init_train_dataloader_state(dataloader=train_loader)
    eval_dataloader_state = init_train_dataloader_state(dataloader=eval_loader)

    # Initialize circuit registry
    registry = CircuitRegistry(save_dir / "circuit_registry")

    # Initialize weight tracker
    weight_tracker = EnhancedWeightSpaceTracker(
        model=model,
        save_dir=save_dir / "weight_tracking",
        logger=model.logger if hasattr(model, 'logger') else None,
        jump_detection_window=100,
        snapshot_freq=analyze_interval // 2,
        sliding_window_size=20,
        dense_sampling=True,
        jump_threshold=1.5
    )

    # Initialize attention analyzer
    attention_analyzer = AttentionAnalyzer(
        model=model,
        save_dir=save_dir / "attention_analysis",
        logger=model.logger if hasattr(model, 'logger') else None
    )

    # Initialize component circuit tracker
    circuit_tracker = ContinuousCircuitTracker(
        model=model,
        save_dir=save_dir / "circuit_tracking",
        logger=model.logger if hasattr(model, 'logger') else None,
        sampling_freq=circuit_sampling_freq,
        history_length=min(200, epochs // analyze_interval)
    )

    # Initialize token discovery
    token_discovery = IntegratedTokenCircuitDiscovery(
        model=model,
        save_dir=save_dir / "token_circuits",
        circuit_registry=registry,
        attention_analyzer=attention_analyzer,
        circuit_tracker=circuit_tracker,
        weight_tracker=weight_tracker
    )

    # Storage for analysis results
    analysis_results = {}

    # Initial weight snapshot
    weight_tracker.take_snapshot(epoch=0, force=True)

    # Initial circuit analysis
    if eval_loader:
        # Get baseline performance
        eval_stats = evaluate(model, eval_loader, criterion, device)
        baseline_acc = eval_stats['accuracy']

        # Log initial metrics
        log_metrics(model, 0, {'loss': 0, 'accuracy': 0}, eval_stats)

        # Run initial circuit analysis
        token_discovery.analyze_epoch(
            epoch=0,
            eval_loader=eval_loader,
            baseline_acc=baseline_acc
        )

    # Training loop
    for epoch in range(epochs):
        # 1. Train for one epoch
        train_stats = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device
        )

        # 2. Step scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # 3. Evaluate periodically
        should_evaluate = epoch % log_interval == 0 or epoch == epochs - 1
        eval_stats = None

        if should_evaluate and eval_loader is not None:
            eval_stats = evaluate(
                model=model,
                eval_loader=eval_loader,
                criterion=criterion,
                device=device
            )

            # Log metrics
            log_metrics(model, epoch, train_stats, eval_stats)

            # Calculate baseline accuracy for analysis
            baseline_acc = eval_stats['accuracy']

            # Detect grokking if performance improves significantly
            detect_grokking(model, epoch, train_stats, eval_stats)

        # 4. Run weight space tracking
        took_snapshot = weight_tracker.take_snapshot(epoch=epoch)

        # 5. Run detailed circuit analysis at regular intervals
        should_analyze = epoch % analyze_interval == 0 or epoch == epochs - 1

        if should_analyze and eval_loader is not None:
            print(f"Running circuit analysis at epoch {epoch}")

            # Ensure we have eval_stats
            if eval_stats is None:
                eval_stats = evaluate(
                    model=model,
                    eval_loader=eval_loader,
                    criterion=criterion,
                    device=device
                )
                baseline_acc = eval_stats['accuracy']

            # Run integrated token circuit analysis
            epoch_results = token_discovery.analyze_epoch(
                epoch=epoch,
                eval_loader=eval_loader,
                baseline_acc=baseline_acc
            )

            # info analyze circuit relationships and evolution
            if len(token_discovery.evolution_tracker.epoch_to_circuits) >= 2:
                lineage = token_discovery.evolution_tracker.track_circuit_lineage(epoch)

                if lineage:
                    print("\nCircuit Evolution:")
                    print(f"\t{get_current_callable_info()} @ {epoch}: \tcircuit evolution")
                    print(f"\t\tnew circuits: {len(lineage.get('new_circuits', []))}")
                    print(f"\t\tevolved circuits: {len(lineage.get('evolved_circuits', {}))}")
                    print(f"\t\tdefunct circuits: {len(lineage.get('defunct_circuits', []))}")

                    # Print transformations
                    transformations = lineage.get('transformations', {})

                    merges = transformations.get('merges', {})
                    if merges:
                        print(f"\t\tcircuit merges: {len(merges)}")

                    splits = transformations.get('splits', {})
                    if splits:
                        print(f"\t\tcircuit splits: {len(splits)}")

                    refinements = transformations.get('refinements', {})
                    if refinements:
                        print(f"\t\tcircuit refinements: {len(refinements)}")

                # Analyze circuit cooperation
                cooperation = token_discovery.analyze_circuit_cooperation(epoch, eval_loader)

                if cooperation and 'cooperating_groups' in cooperation:
                    cooperating_groups = cooperation['cooperating_groups']
                    if cooperating_groups:
                        print("\nCircuit Cooperation:")
                        print(f"\t\tfound {len(cooperating_groups)} cooperating groups")
                        for i, group in enumerate(cooperating_groups):
                            print(f"\t\tgroup {i + 1}: {len(group['circuit_ids'])} circuits")

                # Visualize circuit lineage
                if epoch >= 2 * analyze_interval:
                    # Generate visualization for recent history
                    start_epoch = max(0, epoch - 3 * analyze_interval)
                    token_discovery.evolution_tracker.visualize_circuit_lineage(
                        start_epoch=start_epoch,
                        end_epoch=epoch,
                        save_path=save_dir / f"visualizations/circuit_lineage_{epoch}.png"
                    )

            # whatis ###########################################################
            # fixme todo perhaps it should also be within that if lineage... ???
            # Analyze both cooperation and competition
            cooperation = token_discovery.analyze_circuit_cooperation(epoch, eval_loader)
            competition = token_discovery.analyze_circuit_competition(epoch, eval_loader)

            # Log insights
            if cooperation and 'cooperating_groups' in cooperation:
                cooperating_groups = cooperation['cooperating_groups']
                if cooperating_groups:
                    print("\nCircuit Cooperation:")
                    print(f"  Found {len(cooperating_groups)} cooperating groups")
                    for i, group in enumerate(cooperating_groups):
                        print(f"  Group {i + 1}: {len(group['circuit_ids'])} circuits")

            if competition:
                print("\nCircuit Competition:")
                competing = competition.get('competing_pairs', [])
                if competing:
                    print(f"  Found {len(competing)} competing circuit pairs")

                contention = competition.get('resource_contention', [])
                if contention:
                    print(f"  Found {len(contention)} resource contention cases")

                interference = competition.get('interference_relationships', [])
                if interference:
                    print(f"  Found {len(interference)} interference relationships")

            # Visualize circuit interactions
            token_discovery.evolution_tracker.visualize_circuit_interactions(
                epoch=epoch,
                competition_data=competition,
                cooperation_data=cooperation,
                save_path=save_dir / f"visualizations/circuit_interactions_{epoch}.png"
            )
            # whatis #################################################################

            # info statistical circuit discovery
            # Add to train_with_circuit_analysis.py
            # Modify the analyze_epoch section:

            if should_analyze and eval_loader is not None:
                print(f"Running circuit analysis at epoch {epoch}")

                # Analyze multiple batches for better statistics
                circuit_results_list = []
                num_batches_to_analyze = 3  # Analyze 3 batches instead of 1

                eval_iter = iter(eval_loader)
                for batch_num in range(num_batches_to_analyze):
                    try:
                        batch = next(eval_iter)
                        inputs, targets = batch

                        # Run token circuit analysis on this batch
                        batch_results = token_discovery.analyze_token_relationships(
                            inputs=inputs,
                            targets=targets,
                            epoch=epoch,
                            analyze_multiple_examples=True,
                            max_examples=5  # Analyze 5 examples per batch
                        )

                        circuit_results_list.append(batch_results)

                    except StopIteration:
                        break  # No more batches

                # Combine results across batches
                epoch_results = token_discovery.combine_batch_results(
                    circuit_results_list, epoch)

            # info end of statistical circuit analysis

            # Store analysis results
            analysis_results[epoch] = epoch_results

            # Process any pending jumps
            if weight_tracker.pending_jumps:
                jump_results = process_jumps(
                    model=model,
                    weight_tracker=weight_tracker,
                    eval_loader=eval_loader,
                    criterion=criterion,
                    optimizer=optimizer
                )

                # Analyze the relationship between jumps and circuits
                token_discovery.analyze_jumps_and_circuits(
                    jump_results=jump_results,
                    epoch=epoch,
                    eval_loader=eval_loader
                )

            # Analyze circuit evolution dynamics if we have enough data
            if token_discovery.evolution_tracker and epoch > analyze_interval * 2:
                # Calculate emergence order
                emergence_analysis = token_discovery.evolution_tracker.analyze_emergence_order()

                # Calculate relationships
                relationship_analysis = token_discovery.evolution_tracker.analyze_circuit_relationships()

                # Visualize evolution
                token_discovery.evolution_tracker.visualize_circuit_evolution(
                    save_path=save_dir / f"visualizations/circuit_evolution_{epoch}.png"
                )

                token_discovery.evolution_tracker.visualize_emergence_order(
                    save_path=save_dir / f"visualizations/emergence_order_{epoch}.png"
                )

                # Store dynamics analysis
                analysis_results[f"dynamics_{epoch}"] = {
                    'emergence': emergence_analysis,
                    'relationships': relationship_analysis
                }

                # Log key insights
                print(f"\nCircuit Dynamics at Epoch {epoch}:")

                if 'emergence_order' in emergence_analysis:
                    print("Circuit Emergence Order:")
                    for i, circuit_type in enumerate(emergence_analysis['emergence_order']):
                        avg_epoch = emergence_analysis['avg_emergence'][circuit_type]
                        count = emergence_analysis['count_by_type'][circuit_type]
                        print(f"  {i + 1}. {circuit_type}: {avg_epoch:.1f} avg epoch ({count} circuits)")

                if 'co_occurrence' in relationship_analysis and relationship_analysis['co_occurrence']:
                    print("\nTop Circuit Co-occurrences:")
                    for i, rel in enumerate(relationship_analysis['co_occurrence'][:3]):
                        cid1, cid2 = rel['circuit_pair']
                        print(f"  {i + 1}. {cid1} + {cid2}: {rel['co_occurrences']} co-occurrences")

        # 6. Save checkpoint
        if checkpointManager and (epoch % checkpoint_interval == 0 or epoch == epochs - 1):
            # Ensure we have eval_stats
            if eval_stats is None and eval_loader is not None:
                eval_stats = evaluate(
                    model=model,
                    eval_loader=eval_loader,
                    criterion=criterion,
                    device=device
                )

            # Add circuit analysis data to checkpoint
            extra_data = {
                'circuit_count': len(registry.circuits),
                'weight_space_jumps': [j['epoch'] for j in weight_tracker.detected_jumps]
            }

            checkpointManager.save_checkpoint(
                epoch=epoch,
                train_dataloader_state=train_dataloader_state,
                eval_dataloader_state=eval_dataloader_state,
                dataset_split_indices=dataset_split_indices,
                train_loss=train_stats['loss'] if train_stats else 1.e6,
                train_accuracy=train_stats['accuracy'] if train_stats else 0.0,
                val_loss=eval_stats['loss'] if eval_stats else 1.e6,
                val_accuracy=eval_stats['accuracy'] if eval_stats else 0.0,
                extra_data=extra_data,
                force_save=False,
            )

    # Final analysis of learning dynamics
    if token_discovery.evolution_tracker:
        final_dynamics = {
            'emergence_order': token_discovery.evolution_tracker.analyze_emergence_order(),
            'circuit_relationships': token_discovery.evolution_tracker.analyze_circuit_relationships()
        }

        # Create final visualizations
        token_discovery.evolution_tracker.visualize_circuit_evolution(
            save_path=save_dir / "visualizations/final_circuit_evolution.png"
        )

        token_discovery.evolution_tracker.visualize_emergence_order(
            save_path=save_dir / "visualizations/final_emergence_order.png"
        )

        # Store final dynamics
        analysis_results['final_dynamics'] = final_dynamics

    # Save the registry
    registry.save()

    print("Training and circuit analysis complete!")

    return model, analysis_results