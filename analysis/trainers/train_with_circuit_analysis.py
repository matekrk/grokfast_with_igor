# train_with_circuit_analysis.py
# import torch
# import numpy as np
from pathlib import Path
# from typing import Dict, List, Optional, Any, Union

from analysis.core.circuit_registry import CircuitRegistry
from analysis.analyzers.integrated_token_discovery import IntegratedTokenCircuitDiscovery
from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
from analysis.analyzers.continuous_circuit_tracker import ContinuousCircuitTracker
from analysis.analyzers.attention_pattern_analyzer import AttentionAnalyzer
from analysis.core.circuit_schema import save_circuits, load_circuits, ElementType, CircuitType
from analysis.trainers.utils import (
    evaluate, log_metrics, train_epoch, detect_grokking, process_jumps
)
from analysis.utils.utils import init_train_dataloader_state, get_current_callable_info, shorten_layer_head
from gists.statistical_circuit_discovery import total_batches


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
        checkpoint_interval=200,
        randomize_circuit_analysis=True,
):
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

    # info init  a shared circuit registry
    registry = CircuitRegistry(save_dir / "circuit_registry")
    # info set a shared logger
    shared_logger = model.logger if hasattr(model, 'logger') else None

    # Initialize weight tracker
    weight_tracker = EnhancedWeightSpaceTracker(
        model=model,
        save_dir=save_dir / "weight_tracking",
        logger=shared_logger,
        registry=registry,
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
        logger=shared_logger,
        registry=registry
    )

    # Initialize component circuit tracker
    circuit_tracker = ContinuousCircuitTracker(
        model=model,
        save_dir=save_dir / "circuit_tracking",
        logger=shared_logger,
        registry=registry,
        sampling_freq=circuit_sampling_freq,
        history_length=min(200, epochs // analyze_interval)
    )

    # Initialize token discovery
    token_discovery = IntegratedTokenCircuitDiscovery(
        model=model,
        save_dir=save_dir / "token_circuits",
        logger=shared_logger,
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
            # print(f"\t{get_current_callable_info()} @ {epoch}:\t")

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
            # info now we can get the token-level circuits with
            if "token_results" in epoch_results and "circuits" in epoch_results["token_results"]:
                token_circuits = epoch_results['token_results']["circuits"]
            else:
                token_circuits = []
            token_circuits_by_query = registry.query_circuits(circuit_type=CircuitType.TOKEN)

            # info analyze circuit relationships and evolution
            if len(token_discovery.evolution_tracker.epoch_to_circuits) >= 2:
                lineage = token_discovery.evolution_tracker.track_circuit_lineage(epoch)

                if lineage:
                    print(f"\t{get_current_callable_info()} @ {epoch}: \tcircuit evolution:")
                    print(f"\t\tnew:\t{len(lineage.get('new_circuits', []))}\tevolved: {len(lineage.get('evolved_circuits', {}))}\tdefunct: {len(lineage.get('defunct_circuits', []))}")
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
                        # print("\nCircuit Cooperation:")
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
                        save_path=f"visualizations/circuit_lineage_{epoch}.png"
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
                    print(f"\t{get_current_callable_info()} @ {epoch}: \t\t{len(cooperating_groups)} cooperating groups")
                    for i, group in enumerate(cooperating_groups):
                        print(f"\t\tgroup {i + 1}: {len(group['circuit_ids'])} circuits")

            if competition:
                competing = competition.get('competing_pairs', [])
                contention = competition.get('resource_contention', [])
                interference = competition.get('interference_relationships', [])
                if competing or contention or interference:
                    print(f"\t{get_current_callable_info()} @ {epoch}:\tcircuit competition found")
                    if competing:
                        print(f"\t\t{len(competing)} competing circuit pairs")
                    if contention:
                        print(f"\t\t{len(contention)} resource contention cases")
                    if interference:
                        print(f"\t\t{len(interference)} interference relationships")

            # Visualize circuit interactions
            token_discovery.evolution_tracker.visualize_circuit_interactions(
                epoch=epoch,
                competition_data=competition,
                cooperation_data=cooperation,
                save_path=f"visualizations/circuit_interactions_{epoch}.png"
            )
            # whatis #################################################################

            # info statistical circuit discovery
            # Add to train_with_circuit_analysis.py
            # Modify the analyze_epoch section:

            if should_analyze and eval_loader is not None:
                print(f"\t{get_current_callable_info()} @ {epoch}:\tstatistical token circuit analysis block")

                import random

                total_batches = len(eval_loader)
                min_batches_to_analyze = 3
                num_batches_to_analyze = min (min_batches_to_analyze, total_batches)  # Analyze 3 batches instead of 1
                selected_batch_indices = set(random.sample(range(total_batches, num_batches_to_analyze)))
                # Analyze multiple batches for better statistics
                circuit_results_list = []

                # fixme add random selection of batches
                # eval_iter = list(eval_loader)
                for batch_idx, (inputs, targets) in enumerate(eval_loader):
                    if batch_idx in selected_batch_indices:
                        # Run token circuit analysis on this batch
                        batch_results = token_discovery.analyze_token_relationships(
                            inputs=inputs,
                            targets=targets,
                            epoch=epoch,
                            analyze_multiple_examples=True,
                            max_examples=5,  # Analyze 5 examples per batch
                            random_sampling=True,
                            random_seed=epoch * 1000 + batch_idx
                        )

                        circuit_results_list.append(batch_results)
                        selected_batch_indices.remove(batch_idx)
                    if not selected_batch_indices:
                        break

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
                    save_path=f"visualizations/circuit_evolution_{epoch}.png"
                )

                token_discovery.evolution_tracker.visualize_emergence_order(
                    save_path=f"visualizations/emergence_order_{epoch}.png"
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


        if registry.circuit_logger.should_take_snapshot(epoch=epoch, total_epochs=epochs):
            snapshot = registry.circuit_logger.snapshot_registry_state(epoch=epoch)
            # info log major discoveries
            if snapshot['total_circuits'] > 0:
                print(f"\t{get_current_callable_info()} @ {epoch}: \t{snapshot['total_circuits']} total circuits discovered")
                # info log circuit type distribution
                type_summary = "\t".join([f"{t}-level: {c}" for t, c in snapshot['circuits_by_type'].items()])
                print(f"\t\t{type_summary}")

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
                'weight_space_jumps': [j['epoch'] for j in weight_tracker.detected_jumps],
                'top_circuits': [
                    {
                        'id': c.id,
                        'type': c.type.value,  # Convert enum to value
                        'attribution': round(c.attribution, 4)
                    }
                    for c in sorted(registry.circuits.values(), key=lambda x: x.attribution, reverse=True)[:5]
                ]
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

        # info save circuits registry
        circuit_interval = 50
        if epoch % circuit_interval == 0:
            registry.save()
            registry.circuit_logger.save_logs(epoch=epoch)
            # info perform a circuitry cross-analysis
            token_circuits = registry.query_circuits(circuit_type=CircuitType.TOKEN)
            component_circuits = registry.query_circuits(circuit_type=CircuitType.COMPONENT)
            subspace_circuits = registry.query_circuits(circuit_type=CircuitType.SUBSPACE)
            functional_circuits = registry.query_circuits(circuit_type=CircuitType.FUNCTIONAL)
            circuit_relationships = analyze_cross_level_relationships(
                token_circuits=token_circuits,
                component_circuits=component_circuits,
                functional_circuits=functional_circuits
            )



    # Final analysis of learning dynamics
    if token_discovery.evolution_tracker:
        final_dynamics = {
            'emergence_order': token_discovery.evolution_tracker.analyze_emergence_order(),
            'circuit_relationships': token_discovery.evolution_tracker.analyze_circuit_relationships()
        }

        # Create final visualizations
        token_discovery.evolution_tracker.visualize_circuit_evolution(
            save_path="visualizations/final_circuit_evolution.png"
        )

        token_discovery.evolution_tracker.visualize_emergence_order(
            save_path="visualizations/final_emergence_order.png"
        )

        # Store final dynamics
        analysis_results['final_dynamics'] = final_dynamics

    # Save the registry
    registry.save()

    print("Training and circuit analysis complete!")

    return model, analysis_results


def analyze_cross_level_relationships(token_circuits, component_circuits, functional_circuits):
    """Analyze relationships between different circuit levels"""
    relationships = []

    # Find token circuits implemented by component circuits
    for token_circuit in token_circuits:
        # Extract heads mentioned in token circuit
        token_heads = [e.id for e in token_circuit.elements if e.type == ElementType.HEAD]

        # Find component circuits that use the same heads
        implementing_components = []
        for comp_circuit in component_circuits:
            comp_heads = [e.id for e in comp_circuit.elements if e.type == ElementType.HEAD]

            # Check for overlap
            if set(token_heads).intersection(set(comp_heads)):
                implementing_components.append(comp_circuit.id)

        if implementing_components:
            relationships.append({
                'token_circuit': token_circuit.id,
                'implementing_components': implementing_components,
                'relationship_type': 'implementation'
            })

    return relationships


def test_circuit_serialization(registry, save_dir):
    """Test that circuit serialization works correctly"""
    try:
        test_path = save_dir / "test_circuits.json"

        # Save current circuits
        circuits = list(registry.circuits.values())
        if circuits:
            save_circuits(circuits, test_path)

            # Try to load them back
            loaded_circuits = load_circuits(test_path)

            print(f"✅ Circuit serialization test passed: {len(loaded_circuits)} circuits")

            # Clean up test file
            test_path.unlink()
        else:
            print("📝 No circuits to test serialization")

    except Exception as e:
        print(f"⚠️ Circuit serialization test failed: {e}")
