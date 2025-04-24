# Enhanced analysis helper
def train_with_enhanced_analysis(model, train_loader, eval_loader,
                                 dataset_split_indices, criterion, optimizer, scheduler,
                                 epochs, device, checkpointManager,
                                 log_interval=5, analyze_interval=50,
                                 jump_detection_threshold=2.0, checkpoint_interval=200):
    """Wrapper function for the EnhancedTrainer"""
    from analysis.trainers.enhanced_trainer import EnhancedTrainer

    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_manager=checkpointManager,
        logger=model.logger if hasattr(model, 'logger') else None,
        log_interval=log_interval,
        analyze_interval=analyze_interval,
        jump_detection_threshold=jump_detection_threshold
    )

    return trainer.train(
        epochs=epochs,
        dataset_split_indices=dataset_split_indices,
        checkpoint_interval=checkpoint_interval
    )
