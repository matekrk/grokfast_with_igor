import torch

from analysis.utils.utils import get_class_name


class MemoryManager:
    def __init__(self, model, target_gpu_usage=0.7, target_cpu_usage=0.8, max_message_length=160):
        self.model = model
        self.target_gpu_usage = target_gpu_usage
        self.target_cpu_usage = target_cpu_usage
        self.last_cleanup = 0
        self.analyzers = []
        self.message = ""
        self.max_message_length = max_message_length

    def register_analyzer(self, analyzer):
        self.analyzers.append(analyzer)

    def add_msg(self, msg):
        cur_msg_len = len(self.message) // self.max_message_length
        if (len (msg) + len(self.message)) // self.max_message_length > cur_msg_len:
            self.message += "\n\t\t"
        self.message += msg

    def check_memory(self, epoch):
        """Check memory usage and perform cleanup if needed"""
        needs_cleanup = False

        self.message = f"\tMemoryManager.check_memory() @ {epoch}:"
        # Check GPU memory if available
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            gpu_usage = allocated / total

            if gpu_usage > self.target_gpu_usage and epoch - self.last_cleanup > 10:
                # print(f"\tMemoryManager.check_memory()\tGPU memory usage high ({gpu_usage:.1%}), triggering cleanup at epoch {epoch}")
                self.message += f" GPU ({gpu_usage:.1%})"
                needs_cleanup = True

        # Check CPU memory pressure
        import psutil
        cpu_usage = psutil.virtual_memory().percent / 100.0
        if cpu_usage > self.target_cpu_usage and epoch - self.last_cleanup > 10:
            # print(f"\tMemoryManager.check_memory()\tCPU memory usage high ({cpu_usage:.1%}), triggering cleanup at epoch {epoch}")
            self.message += f" CPU ({cpu_usage:.1%})"
            needs_cleanup = True

        return needs_cleanup  # Just return whether cleanup is needed

    def check_memory_adaptive(self, epoch):
        """Check memory and apply adaptive pruning based on training phase"""
        needs_cleanup = self.check_memory(epoch)

        # Determine training phase
        early_upper = 100
        middle_upper = 500
        early_phase = epoch < early_upper
        middle_phase = early_upper <= epoch < middle_upper
        late_phase = epoch >= middle_upper

        # Adjust pruning strategy based on phase
        if early_phase:
            # Early training: keep most data for early trend analysis
            if needs_cleanup:
                self.light_cleanup()  # Only essential cleanup
        elif middle_phase:
            # Middle training: moderate pruning
            if epoch % 500 == 0 or needs_cleanup:
                self.standard_cleanup()
        else:
            # Late training: aggressive pruning of early epochs
            if epoch % 200 == 0 or needs_cleanup:
                self.aggressive_cleanup()
        if needs_cleanup:
            print(self.message)
        self.message = ""

        return needs_cleanup


    # Different levels of cleanup
    def light_cleanup(self):
        """Perform light memory cleanup - just clear caches"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.add_msg(":: light_cleanup()")

    def standard_cleanup(self):
        """Perform standard cleanup - clear caches and recent histories"""
        # Call cleanup on all analyzers
        for analyzer in self.analyzers:
            if hasattr(analyzer, 'cleanup'):
                analyzer.cleanup()
                self.add_msg(f":: {get_class_name(analyzer)}.cleanup()")

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.add_msg(f":: light_cleanup()")


    def aggressive_cleanup(self):
        """Perform aggressive cleanup - prune histories and clear all caches"""
        # Call cleanup on all analyzers
        for analyzer in self.analyzers:
            if hasattr(analyzer, 'cleanup'):
                analyzer.cleanup()
            self.add_msg(f":: {get_class_name(analyzer)}.cleanup()")

        # Prune history data
        for analyzer in self.analyzers:
            self._prune_history_data(analyzer)
            self.add_msg(f":: {get_class_name(analyzer)}.cleanup()")

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.add_msg(f":: light_cleanup()")

    def _prune_history_data(self, analyzer):
        """
        Prune history data with awareness of different storage patterns
        """
        # Get list of attributes that might be history data
        for attr_name in dir(analyzer):
            attr = getattr(analyzer, attr_name)

            # Skip non-dict attributes and empty dicts
            if not isinstance(attr, dict) or not attr:
                continue

            # Check what type of history structure we have
            if attr_name.endswith('_history'):
                # Case 1: Direct epoch keys (e.g., enhanced_analysis_history)
                if all(isinstance(k, (int, float)) for k in attr.keys()):
                    self._prune_epoch_keyed_history(analyzer, attr_name)

                # Case 2: Component keys with (epoch, value) tuples (e.g., group_velocities_history)
                elif any(isinstance(v, list) and len(v) > 0 and
                         isinstance(v[0], tuple) and len(v[0]) == 2 and
                         isinstance(v[0][0], (int, float)) for v in attr.values()):
                    self._prune_component_keyed_history(analyzer, attr_name)

            # Special case for circuit_history structure
            elif attr_name == 'circuit_history':
                self._prune_circuit_history(analyzer)

    def _prune_epoch_keyed_history(self, analyzer, attr_name):
        """Prune history with epoch keys"""
        history = getattr(analyzer, attr_name)
        if len(history) < 10:
            return  # Skip if not enough entries to prune

        # Strategy: Keep first few, last few, and sparse middle
        epochs = sorted(history.keys())

        # Keep essential epochs to maintain integrity
        # 1. Always keep the first and last epochs
        to_keep = {epochs[0], epochs[-1]}

        # 2. Keep a few more at the beginning and end
        if len(epochs) >= 4:
            to_keep.update(epochs[1:3])  # Keep 2nd and 3rd
            to_keep.update(epochs[-3:-1])  # Keep 3rd and 2nd to last

        # 3. Keep any phase transition or grokking points
        if hasattr(analyzer, 'detected_transitions'):
            for transition in analyzer.detected_transitions:
                transition_epoch = transition.get('epoch')
                if transition_epoch in history:
                    to_keep.add(transition_epoch)

        # 4. Keep regularly spaced samples
        spacing = max(1, len(epochs) // 20)  # Keep ~20 samples across range
        to_keep.update(epochs[::spacing])

        # Create new history
        new_history = {epoch: history[epoch] for epoch in epochs if epoch in to_keep}
        setattr(analyzer, attr_name, new_history)
        self.add_msg(f":: _prune_epoch_keyed_history({attr_name}): {len(history)} → {len(new_history)} entries")
        # print(f"\t\t_prune_epoch_keyed_history()\tPruned\t{attr_name}: {len(history)} → {len(new_history)} entries")


    def _prune_component_keyed_history(self, analyzer, attr_name):
        """Prune history with component keys containing (epoch, value) tuples"""
        history = getattr(analyzer, attr_name)

        # For each component, prune its epoch-value list
        for component, epoch_value_list in history.items():
            if len(epoch_value_list) < 10:
                continue

            # Sort by epoch
            epoch_value_list.sort(key=lambda x: x[0])

            # Keep first, last and regularly spaced samples
            to_keep = [0, len(epoch_value_list) - 1]  # First and last indices
            spacing = max(1, len(epoch_value_list) // 10)  # Keep ~10 samples
            to_keep.extend(range(1, len(epoch_value_list) - 1, spacing))

            # Create pruned list
            pruned_list = [epoch_value_list[i] for i in sorted(set(to_keep))]
            history[component] = pruned_list
        self.add_msg(f":: _prune_component_keyed_history({attr_name})")
        # print(f"\t\t_prune_component_keyed_history()\tPruned\t{attr_name} component histories")


    def _prune_circuit_history(self, analyzer):
        """Special handling for circuit_history structure"""
        if not hasattr(analyzer, 'circuit_history'):
            return

        circuit_history = analyzer.circuit_history

        # Check if we have enough epochs to prune
        if 'epochs' not in circuit_history or len(circuit_history['epochs']) < 10:
            return

        # Get indices to keep
        epochs = circuit_history['epochs']
        num_epochs = len(epochs)

        # Keep first, last, and regularly spaced samples
        indices_to_keep = [0, num_epochs - 1]  # First and last
        spacing = max(1, num_epochs // 10)
        indices_to_keep.extend(range(1, num_epochs - 1, spacing))
        indices_to_keep = sorted(set(indices_to_keep))

        # Update all lists in circuit_history
        for key, value in circuit_history.items():
            if isinstance(value, list) and len(value) == num_epochs:
                circuit_history[key] = [value[i] for i in indices_to_keep]
        self.add_msg(f":: _prune_circuit_history(): {num_epochs} → {len(indices_to_keep)} epochs")
        # print(f"\t\t_prune_circuit_history()\tPruned circuit_history\t{num_epochs} → {len(indices_to_keep)} epochs")

