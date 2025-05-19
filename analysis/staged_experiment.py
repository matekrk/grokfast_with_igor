# staged_experiment.py
import torch
from pathlib import Path

from analysis.analyzers.continuous_circuit_tracker import ContinuousCircuitTracker
from analysis.analyzers.enhanced_phase_analyzer import EnhancedPhaseAnalyzer
from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
from analysis.analyzers.phase_transition_analyzer import PhaseTransitionAnalyzer
from analysis.models.modular_data import create_modular_dataloaders
from analysis.trainers.analysis import train_with_analysis
from analysis.trainers.train_phase_analysis import train_with_phase_analysis
from analysis.utils.checkpoint_manager import GrokAwareCheckpointManager
from analysis.utils.utils import create_optimizer, create_scheduler, create_model


def staged_experiment_framework(base_config, analysis_variants, checkpoint_path=None):
    """
    Run experiments in stages, with ability to restart from checkpoints

    Args:
        base_config: Basic training configuration
        analysis_variants: List of analysis configurations to apply
        checkpoint_path: Optional path to restart from existing checkpoint
    """
    # Stage 1: Train model to checkpoint if not provided
    if checkpoint_path is None:
        model, checkpoint_path = train_to_checkpoint(
            config=base_config,
            max_epochs=base_config['checkpoint_epoch'],
            save_dir=base_config['save_dir']
        )

    # Stage 2: Apply different analysis variants
    results = {}
    for variant_name, variant_config in analysis_variants.items():
        print(f"Running analysis variant: {variant_name}")

        # Load model from checkpoint
        model = load_from_checkpoint(checkpoint_path)

        # Configure analysis based on variant
        analyzer = configure_analyzer(model, variant_config)

        # Continue training with specified analysis
        results[variant_name] = continue_training_with_analysis(
            model=model,
            config=base_config,
            analyzer=analyzer,
            start_epoch=base_config['checkpoint_epoch'],
            max_epochs=base_config['max_epochs']
        )

    return results


def train_to_checkpoint(config, max_epochs, save_dir):
    """Train model to a checkpoint before detailed analysis"""
    # Create model
    model = create_model(config)

    # Create data loaders
    train_loader, eval_loader, vocab_size, dataset_split_indices = create_modular_dataloaders(config)
    train_loader, eval_loader, vocab_size, dataset_split_indices = create_modular_dataloaders(
        modulus=config['modulus'],
        op=config['operation'],
        train_ratio=config['train_ratio'],
        batch_size=config['batch_size'],
        sequence_format=True,
        seed=config['seed'],
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(config) if config.get('use_scheduler') else None

    # Create checkpoint manager
    checkpoint_dir = Path(save_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_manager = GrokAwareCheckpointManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_name=model.get_id(),
        save_dir=save_dir,
        checkpoint_dir=checkpoint_dir,
        stats_dir=Path(save_dir) / "stats",
        save_freq=config.get('checkpoint_interval', 200),
        grokking_window=120,
        max_to_keep=5,
    )

    # Train model with basic analysis
    trained_model = train_with_analysis(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        dataset_split_indices=dataset_split_indices,
        criterion=config['criterion'],
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=max_epochs,
        device=config['device'],
        checkpointManager=checkpoint_manager,
        log_interval=config.get('log_interval', 10),
        analyze_interval=config.get('analyze_interval', 50),
        checkpoint_interval=config.get('checkpoint_interval', 200)
    )

    # Return trained model and checkpoint path
    return trained_model, checkpoint_dir / f"checkpoint_epoch_{max_epochs}.pt"


def load_from_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)

    # Create model
    model_config = checkpoint.get('model_config', {})
    model = create_model(model_config)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def configure_analyzer(model, variant_config):
    """Configure appropriate analyzer based on variant config"""
    save_dir = Path(variant_config['save_dir'])

    if variant_config['type'] == 'phase':
        # Create circuit and weight trackers
        circuit_tracker = ContinuousCircuitTracker(
            model=model,
            save_dir=save_dir / "circuit_tracking",
            logger=model.logger,
            sampling_freq=variant_config.get('circuit_sampling_freq', 20),
            history_length=variant_config.get('history_length', 100)
        )

        weight_tracker = EnhancedWeightSpaceTracker(
            model=model,
            save_dir=save_dir / "weight_tracking",
            logger=model.logger,
            jump_detection_window=variant_config.get('jump_detection_window', 100),
            snapshot_freq=variant_config.get('snapshot_freq', 25),
            sliding_window_size=variant_config.get('sliding_window_size', 20),
            dense_sampling=True,
            jump_threshold=variant_config.get('jump_threshold', 1.5)
        )

        # Create enhanced analyzer
        return EnhancedPhaseAnalyzer(
            model=model,
            save_dir=save_dir,
            logger=model.logger,
            circuit_tracker=circuit_tracker,
            weight_tracker=weight_tracker
        )
    else:
        # Default to standard phase analyzer
        return PhaseTransitionAnalyzer(
            model=model,
            save_dir=save_dir,
            logger=model.logger
        )


def continue_training_with_analysis(model, config, analyzer, start_epoch, max_epochs):
    """Continue training with the specified analyzer"""
    # Create data loaders
    train_loader, eval_loader, dataset_split_indices = create_dataloaders(config)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config) if config.get('use_scheduler') else None

    # Create checkpoint manager
    checkpoint_dir = Path(config['save_dir']) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_manager = GrokAwareCheckpointManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_name=model.get_id(),
        save_dir=config['save_dir'],
        checkpoint_dir=checkpoint_dir,
        stats_dir=Path(config['save_dir']) / "stats",
        save_freq=config.get('checkpoint_interval', 200),
        grokking_window=120,
        max_to_keep=5,
    )

    # Select appropriate training function based on analyzer type
    if isinstance(analyzer, EnhancedPhaseAnalyzer):
        trained_model, weight_tracker, phase_analyzer, phase_weight_analysis = train_with_enhanced_phase_analysis(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            dataset_split_indices=dataset_split_indices,
            criterion=config['criterion'],
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=max_epochs,
            device=config['device'],
            checkpointManager=checkpoint_manager,
            log_interval=config.get('log_interval', 10),
            analyze_interval=config.get('analyze_interval', 50),
            checkpoint_interval=config.get('checkpoint_interval', 200),
            jump_detection_threshold=config.get('jump_threshold', 1.5)
        )

        # Get phase summary with enhanced insights
        summary = phase_analyzer.get_enhanced_learning_phase_summary()

        return {
            'model': trained_model,
            'phase_analyzer': phase_analyzer,
            'weight_tracker': weight_tracker,
            'summary': summary,
            'phase_weight_analysis': phase_weight_analysis
        }
    else:
        # Standard phase analysis
        trained_model, weight_tracker, phase_analyzer, phase_weight_analysis = train_with_phase_analysis(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            dataset_split_indices=dataset_split_indices,
            criterion=config['criterion'],
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=max_epochs,
            device=config['device'],
            checkpointManager=checkpoint_manager,
            log_interval=config.get('log_interval', 10),
            analyze_interval=config.get('analyze_interval', 50),
            checkpoint_interval=config.get('checkpoint_interval', 200)
        )

        summary = phase_analyzer.get_learning_phase_summary()

        return {
            'model': trained_model,
            'phase_analyzer': phase_analyzer,
            'weight_tracker': weight_tracker,
            'summary': summary,
            'phase_weight_analysis': phase_weight_analysis
        }
