# In base_trainer.py
class BaseTrainer:
    """Base class for training transformer models"""

    def __init__(self, model, train_loader, eval_loader,
                 criterion, optimizer, scheduler, device,
                 checkpoint_manager, logger=None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger or (model.logger if hasattr(model, 'logger') else None)

    def train_epoch(self, epoch):
        """Run a single training epoch"""
        self.model.train()
        train_stats = self._run_training_loop(epoch)
        return train_stats

    def _run_training_loop(self, epoch):
        """Core training loop implementation"""
        train_correct = train_total = 0
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Track accuracy and loss
            train_last_token_preds = logits.argmax(dim=-1)
            train_correct += (train_last_token_preds == targets).sum().item()
            train_total += targets.size(0)
            train_loss += loss.item() * targets.size(0)

        # Calculate final statistics
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        train_loss = train_loss / train_total if train_total > 0 else 0.0

        return {
            'accuracy': train_accuracy,
            'loss': train_loss,
            'epoch': epoch
        }

    def evaluate(self, epoch):
        """Evaluate the model"""
        self.model.eval()
        eval_accuracy, eval_loss = self.model.evaluate(self.eval_loader)
        return {'accuracy': eval_accuracy, 'loss': eval_loss, 'epoch': epoch}

    def log_metrics(self, train_stats, eval_stats):
        """Log metrics to the logger"""
        if self.logger:
            self.logger.log_stats('training', train_stats)
            self.logger.log_stats('evaluation', eval_stats)

    def save_checkpoint(self, epoch, train_stats, eval_stats, dataset_split_indices):
        """Save a checkpoint"""
        self.checkpoint_manager.save_checkpoint(
            epoch=epoch,
            train_dataloader_state=self._get_train_dataloader_state(epoch),
            eval_dataloader_state=self._get_eval_dataloader_state(),
            dataset_split_indices=dataset_split_indices,
            train_loss=train_stats['loss'],
            train_accuracy=train_stats['accuracy'],
            val_loss=eval_stats['loss'],
            val_accuracy=eval_stats['accuracy']
        )

    def _get_train_dataloader_state(self, epoch):
        """Get the current state of the training dataloader"""
        from analysis.utils.utils import init_train_dataloader_state
        state = init_train_dataloader_state(self.train_loader)
        state['epoch'] = epoch
        return state

    def _get_eval_dataloader_state(self):
        """Get the current state of the evaluation dataloader"""
        from analysis.utils.utils import init_val_dataloader_state
        return init_val_dataloader_state(self.eval_loader)