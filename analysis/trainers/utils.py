import torch


def train_epoch(model, train_loader, criterion, optimizer, epoch, device):
    """Run a single training epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update statistics
        total_loss += loss.item() * targets.size(0)
        predicted = outputs.argmax(dim=-1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    # Calculate epoch metrics
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def evaluate(model, eval_loader, criterion, device):
    """Evaluate the model on the provided data"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in eval_loader:
            # Move to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update statistics
            total_loss += loss.item() * targets.size(0)
            predicted = outputs.argmax(dim=-1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Calculate metrics
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def log_metrics(model, epoch, train_stats, eval_stats):
    """Log training and evaluation metrics"""
    # Use model logger if available
    if hasattr(model, 'logger'):
        logger = model.logger

        # Log training metrics
        if train_stats:
            logger.log_data('training', 'epoch', epoch)
            logger.log_data('training', 'loss', train_stats['loss'])
            logger.log_data('training', 'accuracy', train_stats['accuracy'])

        # Log evaluation metrics
        if eval_stats:
            logger.log_data('evaluation', 'epoch', epoch)
            logger.log_data('evaluation', 'loss', eval_stats['loss'])
            logger.log_data('evaluation', 'accuracy', eval_stats['accuracy'])

    # Print metrics
    print(f"Epoch  {epoch:5d}: "
          f"\tTrain Loss={train_stats['loss']:.4g}, "
          f"\tAcc={train_stats['accuracy']:5.3f}, "
          f"\t\tVal Loss={eval_stats['loss']:6.4g}, "
          f"\tAcc={eval_stats['accuracy']:5.3f}")


def detect_grokking(model, epoch, train_stats, eval_stats):
    """Detect if grokking is occurring at this epoch"""
    # Skip if model has no logger or if stats are missing
    if not hasattr(model, 'logger') or not train_stats or not eval_stats:
        return False

    logger = model.logger

    # Check for grokking conditions:
    # 1. Training accuracy is high (memorization)
    # 2. Evaluation accuracy is rapidly improving

    # Check if we have enough history
    if logger.get_length('evaluation', 'accuracy') >= 5:
        # Get recent history
        recent_eval_accs = logger.logs['evaluation']['accuracy'][-5:]

        # Check if training accuracy is high
        train_high = train_stats['accuracy'] > 0.9

        # Check if evaluation accuracy is improving rapidly
        prev_eval_accs = recent_eval_accs[:-1]  # All but the latest
        prev_avg = sum(prev_eval_accs) / len(prev_eval_accs) if prev_eval_accs else 0

        significant_improvement = (eval_stats['accuracy'] > prev_avg * 1.2)

        # Detect potential grokking
        if train_high and significant_improvement:
            print(f"\tdetect_grokking()\tPotential grokking detected at epoch {epoch}")

            # Log the grokking point
            logger.log_data('grokking_phases', 'grokking_step', epoch)
            return True

    return False


def process_jumps(model, weight_tracker, eval_loader, criterion, optimizer):
    """Process pending jumps detected by the weight tracker"""
    # Get a batch of data for analysis
    sample_inputs, sample_targets = next(iter(eval_loader))
    sample_inputs = sample_inputs.to(next(model.parameters()).device)
    sample_targets = sample_targets.to(next(model.parameters()).device)

    jump_results = weight_tracker.analyze_pending_jumps(
        inputs=sample_inputs,
        targets=sample_targets,
        criterion=criterion,
        optimizer=optimizer,
        jump_analyzer=None,  # We'll handle this separately
        eval_loader=eval_loader,
        mini_train_steps=weight_tracker.sliding_window_size - 1,
    )
    # Process jumps

    # Print summary of processed jumps
    if jump_results:
        print(f"\tprocess_jumps()\tProcessed {len(jump_results)} weight space jumps:")
        for result in jump_results:
            jump_epoch = result['jump_epoch']
            jump_char = result['characterization']

            print(f"\t\tJump at epoch {jump_epoch}: "
                  f"Magnitude={jump_char['total_magnitude']['pre_to_jump']:.4f}, "
                  f"Top layers: {', '.join(jump_char['top_layers'][:2])}, "
                  f"Top heads: {', '.join(jump_char['top_heads'][:2])}")

        # Visualize jump timeline
        weight_tracker.visualize_jumps_timeline()

    return jump_results
