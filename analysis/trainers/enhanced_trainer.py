# In enhanced_trainer.py
from analysis.core.base_trainer import BaseTrainer


class EnhancedTrainer(BaseTrainer):
    """Enhanced trainer with weight space jump detection and analysis"""

    def __init__(self, model, train_loader, eval_loader,
                 criterion, optimizer, scheduler, device,
                 checkpoint_manager, logger=None,
                 weight_tracker=None, jump_analyzer=None,
                 log_interval=5, analyze_interval=50,
                 jump_detection_threshold=2.0):
        super().__init__(model, train_loader, eval_loader,
                         criterion, optimizer, scheduler,
                         device, checkpoint_manager, logger)

        self.log_interval = log_interval
        self.analyze_interval = analyze_interval
        self.jump_detection_threshold = jump_detection_threshold

        # Initialize trackers and analyzers if not provided
        if weight_tracker is None:
            from analysis.analyzers.enhanced_weight_space_tracker import EnhancedWeightSpaceTracker
            self.weight_tracker = EnhancedWeightSpaceTracker(
                model=model,
                save_dir=checkpoint_manager.checkpoint_dir,
                logger=self.logger,
                jump_detection_window=100,
                snapshot_freq=analyze_interval,
                sliding_window_size=10,
                dense_sampling=True,
                jump_threshold=jump_detection_threshold
            )
        else:
            self.weight_tracker = weight_tracker

        if jump_analyzer is None:
            from jump_analysis_tools import JumpAnalysisTools
            self.jump_analyzer = JumpAnalysisTools(
                model=model,
                save_dir=checkpoint_manager.checkpoint_dir,
                logger=self.logger
            )
        else:
            self.jump_analyzer = jump_analyzer

        # Get initial snapshot
        self.weight_tracker.take_snapshot(epoch=0, force=True)

    def train(self, epochs, dataset_split_indices, checkpoint_interval=200):
        """Run the full training process with enhanced analysis"""
        from analysis.utils.utils import FittingScore
        fitscore = FittingScore()

        for epoch in range(epochs):
            # 1. Run training epoch
            train_stats = self.train_epoch(epoch)

            # 2. Take weight snapshot and detect jumps
            took_snapshot = self._take_weight_snapshot(epoch)

            # 3. Periodically evaluate and log metrics
            should_log = epoch % self.log_interval == 0 or took_snapshot
            if should_log:
                eval_stats = self.evaluate(epoch)
                self.log_metrics(train_stats, eval_stats)

                # Calculate fitting score
                fitting_score = fitscore(
                    train_loss=train_stats['loss'],
                    train_accu=train_stats['accuracy'],
                    eval_loss=eval_stats['loss'],
                    eval_accu=eval_stats['accuracy']
                )

                print(f"Epoch {epoch:5d}: Train Loss={train_stats['loss']:.3g}, "
                      f"Train Acc={train_stats['accuracy']:.4f}, "
                      f"Val Loss={eval_stats['loss']:.5g}, "
                      f"Val Acc={eval_stats['accuracy']:.4f}, "
                      f"Fitting_score={fitting_score:.4f}")

                # Track grokking metrics
                self._track_grokking_metrics(epoch)

            # 4. Analyze pending jumps
            if self.weight_tracker.pending_jumps:
                self._analyze_pending_jumps()

            # 5. Periodic detailed analysis
            if epoch % (16 * self.analyze_interval) == 0 and epoch > 0:
                self._perform_detailed_analysis(epoch)

            # 6. Save checkpoint
            if epoch % checkpoint_interval == 0 or epoch == epochs - 1:
                self.save_checkpoint(
                    epoch=epoch + 1,
                    train_stats=train_stats,
                    eval_stats=eval_stats if 'eval_stats' in locals() else None,
                    dataset_split_indices=dataset_split_indices
                )

        # Final analysis
        self._perform_final_analysis()

        return self.model, self.weight_tracker, self.jump_analyzer

    def _take_weight_snapshot(self, epoch):
        """Take a weight space snapshot with jump detection logic"""
        min_epoch_for_detection = 100
        force_snapshot = (
                epoch > min_epoch_for_detection and (
                epoch % self.analyze_interval == 0 or  # Regular interval
                (epoch - 1) % self.analyze_interval == 0 or  # Just before
                (epoch + 1) % self.analyze_interval == 0 or  # Just after
                (epoch - 2) % self.analyze_interval == 0 or  # Two before
                (epoch + 2) % self.analyze_interval == 0  # Two after
        )
        )

        # Take snapshot and return whether it was taken
        return self.weight_tracker.take_snapshot(epoch=epoch, force=force_snapshot)

    def _track_grokking_metrics(self, epoch):
        """Track metrics for grokking detection"""
        from grokking_detection import track_metrics_for_grokking
        track_metrics_for_grokking(
            epoch=epoch,
            model=self.model,
            train_loader=self.train_loader,
            eval_loader=self.eval_loader
        )

    def _analyze_pending_jumps(self):
        """Analyze pending jumps detected by the weight tracker"""
        # Get a batch of data for analysis
        sample_inputs, sample_targets = next(iter(self.eval_loader))
        sample_inputs = sample_inputs.to(self.device)
        sample_targets = sample_targets.to(self.device)

        # Process jumps
        jump_results = self.weight_tracker.analyze_pending_jumps(
            inputs=sample_inputs,
            targets=sample_targets,
            criterion=self.criterion,
            jump_analyzer=self.jump_analyzer
        )

        # Process and log results
        for result in jump_results:
            self._process_jump_result(result)

        # Visualize results
        self.weight_tracker.visualize_jumps_timeline()
        self.weight_tracker.visualize_trajectory(
            selected_dims=[0, 1],
            highlight_epochs={
                'jumps': [j['epoch'] for j in self.weight_tracker.detected_jumps]
            }
        )

    def _process_jump_result(self, result):
        """Process and log a single jump result"""
        jump_epoch = result['jump_epoch']
        jump_char = result['characterization']

        print(f"Jump analysis for epoch {jump_epoch}:")
        print(f"  Total magnitude: {jump_char['total_magnitude']['pre_to_jump']:.4f}")
        print(f"  Symmetry ratio: {jump_char['total_magnitude']['symmetry_ratio']:.4f}")
        print(f"  Top changing layers: {', '.join(jump_char['top_layers'])}")
        print(f"  Top changing heads: {', '.join(jump_char['top_heads'])}")

        # Create visualizations
        viz_dir = self.weight_tracker.visualize_jump_characterization(jump_char)
        print(f"  Visualizations saved to {viz_dir}")

        # Check correlation with grokking
        if 'grokking_phases' in self.model.logger.logs and 'grokking_step' in self.model.logger.logs['grokking_phases']:
            grokking_step = self.model.logger.logs['grokking_phases']['grokking_step']
            if grokking_step:
                if isinstance(grokking_step, list):
                    dist_l = []
                    for grokking_epoch in grokking_step:
                        dist_l.append(abs(jump_epoch - grokking_epoch))
                    distance = min(dist_l)
                else:
                    distance = abs(jump_epoch - grokking_step)
                print(f"  Distance to grokking point: {distance} epochs")

                # Highlight if jump is close to grokking
                if distance < 50:
                    print(f"  *** THIS JUMP MAY BE RELATED TO GROKKING! ***")

    def _perform_detailed_analysis(self, epoch):
        """Perform periodic detailed analysis"""
        print(f"Performing periodic detailed analysis at epoch {epoch}...")

        # Analyze loss landscape
        sample_inputs, sample_targets = next(iter(self.eval_loader))
        sample_inputs = sample_inputs.to(self.device)
        sample_targets = sample_targets.to(self.device)

        self.jump_analyzer.analyze_loss_curvature(
            inputs=sample_inputs,
            targets=sample_targets,
            criterion=self.criterion
        )

        # Check for grokking transitions
        from grokking_detection import analyze_grokking_transitions
        grokking_analysis = analyze_grokking_transitions(
            model=self.model,
            train_loader=self.train_loader,
            eval_loader=self.eval_loader
        )

        # Log correlations between jumps and grokking
        if grokking_analysis and 'primary_grokking_step' in grokking_analysis and grokking_analysis[
            'primary_grokking_step']:
            self._log_grokking_jump_correlation(grokking_analysis['primary_grokking_step'])

    def _log_grokking_jump_correlation(self, grokking_step):
        """Log correlation between grokking and jumps"""
        print(f"  Grokking detected at epoch {grokking_step}")

        # Find closest jump
        jumps = [j['epoch'] for j in self.weight_tracker.detected_jumps]
        if jumps:
            closest_jump = min(jumps, key=lambda x: abs(x - grokking_step))
            distance = abs(closest_jump - grokking_step)
            print(f"  Closest jump to grokking point is at epoch {closest_jump} (distance: {distance} epochs)")

    def _perform_final_analysis(self):
        """Perform final analysis at the end of training"""
        print("Training complete. Generating final analysis...")

        # Visualize the overall jump timeline
        self.weight_tracker.visualize_jumps_timeline()

        # Analyze the trajectory in weight space
        self.weight_tracker.visualize_trajectory(
            selected_dims=[0, 1],
            highlight_epochs={
                'jumps': [j['epoch'] for j in self.weight_tracker.detected_jumps]
            }
        )

        # Log jump summary
        jump_summary = self.weight_tracker.get_jump_summary()
        if jump_summary is not None and self.logger:
            self.logger.log_data('weight_space_jumps', 'summary', jump_summary.to_dict('records'))