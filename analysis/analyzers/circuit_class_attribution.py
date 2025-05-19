# circuit_class_attribution.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import seaborn as sns


class CircuitClassAttribution:
    """Analyze how circuits contribute to class-specific predictions"""

    def __init__(self, model, save_dir, circuit_tracker=None, logger=None):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.circuit_tracker = circuit_tracker
        self.logger = logger if logger else (model.logger if hasattr(model, 'logger') else None)

        # Storage for analysis results
        self.circuit_class_map = {}
        self.class_circuit_map = {}
        self.circuit_activation_history = {}

    def analyze_epoch(self, epoch, eval_loader, batch_limit=10):
        """Analyze circuit-class relationships at the current epoch"""
        # Ensure circuit tracker is provided
        if self.circuit_tracker is None:
            # print("No circuit tracker provided. Circuit-class analysis requires a circuit tracker.")
            return None

        # Get active circuits from the circuit tracker
        active_circuits = self._get_active_circuits()
        if not active_circuits:
            # print("No active circuits detected. Skipping circuit-class analysis.")
            return None
        # print("CircuitClassAttribution.analyze_epoch(): probable active circuits --> circuit-class analysis.")

        # Measure circuit activation for different inputs
        circuit_activations = self._measure_circuit_activations(
            active_circuits, eval_loader, batch_limit)

        # Identify class-specific circuits
        class_specific_circuits = self._identify_class_specific_circuits(
            circuit_activations)

        # Update history
        self.circuit_class_map[epoch] = class_specific_circuits

        # Visualize results
        self._visualize_circuit_class_relationship(epoch, circuit_activations, class_specific_circuits)

        # Log results if logger is available
        if self.logger:
            for class_id, circuits in class_specific_circuits['class_to_circuits'].items():
                if circuits:
                    self.logger.log_data('circuit_class_attribution',
                                         f'class_{class_id}_circuit_count',
                                         len(circuits))

            # Log overall circuit specialization
            self.logger.log_data('circuit_class_attribution',
                                 'specialized_circuit_percentage',
                                 class_specific_circuits['specialized_percentage'])

        return {
            'class_specific_circuits': class_specific_circuits,
            'circuit_activations': circuit_activations
        }

    def _get_active_circuits(self):
        """Get active circuits from the circuit tracker"""
        # This method should be adapted based on your circuit_tracker's API
        if hasattr(self.circuit_tracker, 'circuit_history'):
            # Extract active circuits from the most recent epoch
            if self.circuit_tracker.circuit_history['active_circuits']:
                return self.circuit_tracker.circuit_history['active_circuits'][-1]

        return []

    def _measure_circuit_activations(self, active_circuits, eval_loader, batch_limit=10):
        """Measure how circuits activate for different inputs"""
        self.model.eval()

        # Storage for results
        circuit_activations = defaultdict(list)
        input_classes = []

        # Process batches
        batch_count = 0
        with torch.no_grad():
            for inputs, targets in eval_loader:
                if batch_count >= batch_limit:
                    break

                # Store class labels
                input_classes.extend(targets.cpu().numpy())

                # For each circuit, measure activation strength
                for circuit in active_circuits:
                    # Reset model to original state
                    original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                    # First get normal predictions
                    outputs_normal = self.model(inputs)
                    predictions_normal = outputs_normal.argmax(dim=-1)

                    # Then zero out this circuit and get predictions
                    self._zero_out_circuit(circuit)
                    outputs_ablated = self.model(inputs)
                    predictions_ablated = outputs_ablated.argmax(dim=-1)

                    # Calculate impact on predictions
                    impact = (predictions_normal == targets).float() - (predictions_ablated == targets).float()

                    # Store activations for each input
                    for i in range(len(inputs)):
                        circuit_key = self._format_circuit_key(circuit)
                        input_key = f"batch_{batch_count}_input_{i}"
                        class_id = targets[i].item()

                        circuit_activations[circuit_key].append({
                            'input_key': input_key,
                            'class_id': class_id,
                            'impact': impact[i].item(),
                            'confidence_change': (outputs_normal[i, targets[i]] -
                                                  outputs_ablated[i, targets[i]]).item()
                        })

                    # Restore original model state
                    self.model.load_state_dict(original_state)

                batch_count += 1

        return {
            'circuit_activations': dict(circuit_activations),
            'input_classes': input_classes
        }

    def _zero_out_circuit(self, circuit):
        """Zero out a circuit by disabling its components"""
        # Parse the circuit components
        if '+' in circuit:
            components = circuit.split('+')
        else:
            components = [circuit]  # Single component circuit

        # Zero out each component
        for component in components:
            if 'layer_' in component and 'head_' in component:
                # Extract layer and head indices
                layer_str, head_str = component.split('_head_')
                layer_idx = int(layer_str.replace('layer_', ''))
                head_idx = int(head_str)

                # Zero out this attention head
                if layer_idx < len(self.model.layers):
                    head_dim = self.model.dim // self.model.num_heads
                    start_idx = head_idx * head_dim
                    end_idx = (head_idx + 1) * head_dim

                    # Zero out the output projection weights for this head
                    with torch.no_grad():
                        self.model.layers[layer_idx].attn.out_proj.weight[:, start_idx:end_idx] = 0

    def _format_circuit_key(self, circuit):
        """Format a circuit identifier consistently"""
        if '+' in circuit:
            components = circuit.split('+')
            # Sort components for consistency
            components.sort()
            return '+'.join(components)
        return circuit

    def _identify_class_specific_circuits(self, circuit_activations):
        """Identify circuits that are selective for specific classes"""
        # Extract activation data
        activations = circuit_activations['circuit_activations']

        # Map from circuits to their class preferences
        circuit_to_class = {}
        class_to_circuits = defaultdict(list)

        for circuit_key, activation_data in activations.items():
            # Group by class
            class_impacts = defaultdict(list)

            for entry in activation_data:
                class_id = entry['class_id']
                impact = entry['impact']
                class_impacts[class_id].append(impact)

            # Calculate average impact per class
            avg_impacts = {cls: np.mean(impacts) for cls, impacts in class_impacts.items()}

            # Find preferred class (highest positive impact)
            if avg_impacts:
                max_class = max(avg_impacts, key=avg_impacts.get)
                max_impact = avg_impacts[max_class]

                # Check if impact is significant enough to consider specialized
                if max_impact > 0.1:  # This threshold can be adjusted
                    circuit_to_class[circuit_key] = {
                        'preferred_class': max_class,
                        'impact': max_impact,
                        'selectivity': self._calculate_selectivity(avg_impacts, max_class)
                    }
                    class_to_circuits[max_class].append(circuit_key)

        # Calculate circuit specialization percentage
        total_circuits = len(activations)
        specialized_circuits = len(circuit_to_class)
        specialized_percentage = (specialized_circuits / total_circuits) * 100 if total_circuits > 0 else 0

        return {
            'circuit_to_class': circuit_to_class,
            'class_to_circuits': dict(class_to_circuits),
            'specialized_percentage': specialized_percentage
        }

    def _calculate_selectivity(self, class_impacts, preferred_class):
        """Calculate how selective a circuit is for its preferred class"""
        preferred_impact = class_impacts[preferred_class]
        other_impacts = [impact for cls, impact in class_impacts.items() if cls != preferred_class]

        if not other_impacts:
            return 1.0  # Perfectly selective (only one class tested)

        avg_other_impact = np.mean(other_impacts)
        max_other_impact = max(other_impacts) if other_impacts else 0

        # Selectivity based on difference between preferred and maximum other impact
        return (preferred_impact - max_other_impact) / (preferred_impact + 1e-6)

    def _visualize_circuit_class_relationship(self, epoch, circuit_activations, class_specific_circuits):
        """Visualize the relationship between circuits and classes"""
        # Create directory for visualizations
        viz_dir = self.save_dir / f"epoch_{epoch}"
        viz_dir.mkdir(exist_ok=True, parents=True)

        # 1. Circuit-Class Impact Heatmap
        activations = circuit_activations['circuit_activations']
        if not activations:
            return

        # Prepare data for heatmap
        unique_classes = sorted(set(circuit_activations['input_classes']))
        circuit_keys = list(activations.keys())

        # Calculate average impact for each circuit-class pair
        impact_matrix = np.zeros((len(circuit_keys), len(unique_classes)))

        for i, circuit_key in enumerate(circuit_keys):
            circuit_data = activations[circuit_key]

            # Group by class
            for j, class_id in enumerate(unique_classes):
                class_impacts = [entry['impact'] for entry in circuit_data
                                 if entry['class_id'] == class_id]

                if class_impacts:
                    impact_matrix[i, j] = np.mean(class_impacts)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        # Limit to top circuits for readability if many circuits
        if len(circuit_keys) > 20:
            # Select circuits with highest impact
            max_impacts = np.max(impact_matrix, axis=1)
            top_indices = np.argsort(max_impacts)[-20:]

            impact_matrix = impact_matrix[top_indices]
            circuit_keys = [circuit_keys[i] for i in top_indices]

        # Plot heatmap
        sns.heatmap(impact_matrix, cmap='coolwarm', center=0,
                    xticklabels=unique_classes,
                    yticklabels=[key[:15] + '...' if len(key) > 15 else key for key in circuit_keys],
                    ax=ax)

        ax.set_title(f'Circuit-Class Impact at Epoch {epoch}')
        ax.set_xlabel('Class')
        ax.set_ylabel('Circuit')

        plt.suptitle(f"{self.model.plot_prefix}")
        plt.tight_layout()
        plt.savefig(viz_dir / "circuit_class_impact.png")
        plt.close(fig)

        # 2. Circuit Specialization Pie Chart
        specialized = len(class_specific_circuits['circuit_to_class'])
        non_specialized = len(activations) - specialized

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie([specialized, non_specialized],
               labels=['Class-Specific', 'General Purpose'],
               autopct='%1.1f%%',
               colors=['#ff9999', '#66b3ff'])

        ax.set_title(f'Circuit Specialization at Epoch {epoch}')
        plt.suptitle(f"{self.model.plot_prefix}")
        plt.savefig(viz_dir / "circuit_specialization.png")
        plt.close(fig)

        # 3. Class-to-Circuit Distribution
        if class_specific_circuits['class_to_circuits']:
            class_counts = {cls: len(circuits) for cls, circuits in
                            class_specific_circuits['class_to_circuits'].items()}

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.bar(range(len(class_counts)), list(class_counts.values()))
            ax.set_xticks(range(len(class_counts)))
            ax.set_xticklabels(list(class_counts.keys()))

            ax.set_title(f'Specialized Circuits per Class at Epoch {epoch}')
            ax.set_xlabel('Class')
            ax.set_ylabel('Number of Specialized Circuits')
            plt.suptitle(f"{self.model.plot_prefix}")
            plt.savefig(viz_dir / "circuits_per_class.png")
            plt.close(fig)

    def cleanup(self):
        """Release memory held by various analyzers"""
        # Clear cached activations
        self.layer_activations = {}

        # Clear large stored tensors
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, dict) and any(isinstance(v, (torch.Tensor, np.ndarray))
                                              for v in attr.values() if hasattr(attr, 'values')):
                setattr(self, attr_name, {})

        # Call torch.cuda.empty_cache() if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Call cleanup on child analyzers
        for analyzer_name in ['mlp_sparsity_tracker', 'circuit_class_analyzer', 'interaction_analyzer']:
            if hasattr(self, analyzer_name):
                analyzer = getattr(self, analyzer_name)
                if hasattr(analyzer, 'cleanup'):
                    analyzer.cleanup()