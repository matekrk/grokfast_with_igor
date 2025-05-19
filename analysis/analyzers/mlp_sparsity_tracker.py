# mlp_sparsity_tracker.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class MLPSparsityTracker:
    """Track the development of sparse representations in MLP layers"""

    def __init__(self, model, save_dir, logger=None, activation_threshold=0.1):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger if logger else (model.logger if hasattr(model, 'logger') else None)
        self.activation_threshold = activation_threshold

        # info storage for tracking sparsity evolution
        self.sparsity_history = {}
        self.neuron_class_selectivity = {}
        self.activation_patterns = {}

        # info register hooks for capturing MLP activations
        self.hooks = []
        self.layer_activations = {}
        self._register_activation_hooks()

    def _register_activation_hooks(self):
        """Register forward hooks to capture MLP activations"""
        for layer_idx, layer in enumerate(self.model.layers):
            # info hook for MLP intermediate activations (after first linear layer and activation)
            def get_hook(layer_idx, component):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self.layer_activations[f"layer_{layer_idx}_{component}"] = output[0].detach()
                    else:
                        self.layer_activations[f"layer_{layer_idx}_{component}"] = output.detach()

                return hook

            # info hook after first MLP linear layer + activation
            #  (captures the expanded representation space)
            mlp_expanded_hook = get_hook(layer_idx, "mlp_expanded")
            handle = layer.mlp[1].register_forward_hook(mlp_expanded_hook)  # After GELU
            self.hooks.append(handle)

    def analyze_neuron_activity(self, eval_loader, class_labels=None, batch_limit=10):
        """Analyze activation patterns of MLP neurons for input batches"""
        self.model.eval()

        # info storage for batch analysis
        batch_activations = {}
        batch_sparsity = {}
        batch_classes = []

        # info process batches
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                if batch_idx >= batch_limit:
                    break

                # info store class labels if provided
                if class_labels is not None:
                    batch_classes.extend([class_labels[t.item()] for t in targets])
                else:
                    batch_classes.extend([t.item() for t in targets])

                # info forward pass to trigger hooks
                _ = self.model(inputs)

                # info analyze activations from hooks
                for layer_name, activations in self.layer_activations.items():
                    if "mlp_expanded" in layer_name:
                        # info calculate sparsity (% of inactive neurons)
                        is_active = (activations > self.activation_threshold).float()
                        sparsity = 1.0 - (torch.sum(is_active) / is_active.numel()).item()

                        if layer_name not in batch_sparsity:
                            batch_sparsity[layer_name] = []
                        batch_sparsity[layer_name].append(sparsity)

                        # info store activation patterns by input
                        for idx, input_tensor in enumerate(inputs):
                            input_key = f"batch_{batch_idx}_input_{idx}"
                            if input_key not in batch_activations:
                                batch_activations[input_key] = {}
                            batch_activations[input_key][layer_name] = is_active[:, idx].cpu().numpy()

        # info calculate average sparsity and store results
        avg_sparsity = {layer: np.mean(values) for layer, values in batch_sparsity.items()}

        # info analyze neuron selectivity for classes
        class_selectivity = self._calculate_class_selectivity(batch_activations, batch_classes)

        return {
            'avg_sparsity': avg_sparsity,
            'class_selectivity': class_selectivity,
            'sample_activations': batch_activations
        }

    def _calculate_class_selectivity(self, batch_activations, batch_classes):
        """Calculate how selective neurons are for specific classes"""
        # info group inputs by class
        class_groups = {}
        for idx, class_label in enumerate(batch_classes):
            input_key = f"batch_{idx // len(self.model.layers)}_input_{idx % len(self.model.layers)}"
            if class_label not in class_groups:
                class_groups[class_label] = []
            class_groups[class_label].append(input_key)

        # info calculate neuron selectivity for each class
        selectivity = {}
        for layer_name in next(iter(batch_activations.values())).keys():
            if "mlp_expanded" in layer_name:
                layer_selectivity = {}

                # info get neuron count from first activation
                first_key = list(batch_activations.keys())[0]
                n_neurons = len(batch_activations[first_key][layer_name])

                # info for each neuron, calculate selectivity for each class
                for neuron_idx in range(n_neurons):
                    neuron_selectivity = {}

                    for class_label, input_keys in class_groups.items():
                        # info count activations for this class
                        activations = [
                            batch_activations[key][layer_name][neuron_idx]
                            for key in input_keys
                            if key in batch_activations
                        ]

                        activation_rate = np.mean(activations) if activations else 0
                        neuron_selectivity[class_label] = activation_rate

                    # info calculate selectivity score (max activation - mean of others)
                    if neuron_selectivity:
                        max_class = max(neuron_selectivity, key=neuron_selectivity.get)
                        max_value = neuron_selectivity[max_class]
                        other_values = [v for c, v in neuron_selectivity.items() if c != max_class]
                        mean_others = np.mean(other_values) if other_values else 0

                        selectivity_score = max_value - mean_others
                        preferred_class = max_class if selectivity_score > 0.2 else None

                        layer_selectivity[f"neuron_{neuron_idx}"] = {
                            'score': selectivity_score,
                            'preferred_class': preferred_class,
                            'class_activations': neuron_selectivity
                        }

                selectivity[layer_name] = layer_selectivity

        return selectivity

    def track_sparsity_evolution(self, epoch, eval_loader, class_labels=None):
        """Track how sparsity patterns evolve during training"""
        # info analyze current sparsity patterns
        analysis_results = self.analyze_neuron_activity(eval_loader, class_labels)

        # info store results in history
        self.sparsity_history[epoch] = {
            'avg_sparsity': analysis_results['avg_sparsity'],
            'selectivity_summary': self._summarize_selectivity(analysis_results['class_selectivity'])
        }

        # info log metrics if logger is available
        if self.logger:
            for layer_name, sparsity in analysis_results['avg_sparsity'].items():
                self.logger.log_data('mlp_sparsity', f'{layer_name}_sparsity', sparsity)

                # info log selectivity metrics
                if layer_name in analysis_results['class_selectivity']:
                    selective_neurons = self._count_selective_neurons(
                        analysis_results['class_selectivity'][layer_name]
                    )
                    self.logger.log_data('mlp_selectivity',
                                         f'{layer_name}_selective_neurons',
                                         selective_neurons)

        # info visualize current sparsity state if this is a significant epoch
        #  (e.g., after a detected phase transition)
        self._visualize_sparsity_patterns(epoch, analysis_results)

        return analysis_results

    def _summarize_selectivity(self, selectivity_data):
        """Summarize neuron selectivity data"""
        summary = {}

        for layer_name, layer_data in selectivity_data.items():
            layer_summary = {
                'total_neurons': len(layer_data),
                'selective_neurons': self._count_selective_neurons(layer_data),
                'class_distribution': {}
            }

            # info count neurons selective for each class
            class_counts = {}
            for neuron_data in layer_data.values():
                preferred_class = neuron_data.get('preferred_class')
                if preferred_class is not None:
                    if preferred_class not in class_counts:
                        class_counts[preferred_class] = 0
                    class_counts[preferred_class] += 1

            layer_summary['class_distribution'] = class_counts
            summary[layer_name] = layer_summary

        return summary

    def _count_selective_neurons(self, layer_selectivity):
        """Count neurons with clear class selectivity"""
        return sum(1 for data in layer_selectivity.values()
                   if data.get('preferred_class') is not None)

    def _visualize_sparsity_patterns(self, epoch, analysis_results):
        """Generate visualizations of sparsity patterns"""
        # info create directory for visualizations
        viz_dir = self.save_dir / f"epoch_{epoch}"
        viz_dir.mkdir(exist_ok=True, parents=True)

        # info 1. plot sparsity by layer
        fig, ax = plt.subplots(figsize=(10, 6))
        layers = list(analysis_results['avg_sparsity'].keys())
        sparsity_values = [analysis_results['avg_sparsity'][layer] for layer in layers]

        ax.bar(range(len(layers)), sparsity_values)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Sparsity (% inactive neurons)')
        ax.set_title(f'MLP Sparsity by Layer at Epoch {epoch}')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([layer.replace('layer_', '').replace('_mlp_expanded', '')
                            for layer in layers], rotation=45)

        plt.tight_layout()
        plt.suptitle(f"{self.model.plot_prefix}")
        plt.savefig(viz_dir / "sparsity_by_layer.png")
        plt.close(fig)

        # info 2. plot neuron selectivity
        if analysis_results['class_selectivity']:
            for layer_name, layer_data in analysis_results['class_selectivity'].items():
                # info skip if empty
                if not layer_data:
                    continue

                # info extract selectivity scores
                neuron_ids = list(layer_data.keys())
                selectivity_scores = [data['score'] for data in layer_data.values()]

                # info sort by selectivity
                sorted_indices = np.argsort(selectivity_scores)[::-1]
                sorted_neurons = [neuron_ids[i] for i in sorted_indices]
                sorted_scores = [selectivity_scores[i] for i in sorted_indices]

                # info plot top neurons by selectivity
                top_n = min(50, len(sorted_neurons))
                fig, ax = plt.subplots(figsize=(12, 6))

                ax.bar(range(top_n), sorted_scores[:top_n])
                ax.set_xlabel('Neuron Index')
                ax.set_ylabel('Class Selectivity Score')
                ax.set_title(f'Top {top_n} Selective Neurons in {layer_name} at Epoch {epoch}')

                plt.tight_layout()
                plt.suptitle(f"{self.model.plot_prefix}")
                plt.savefig(viz_dir / f"{layer_name}_selectivity.png")
                plt.close(fig)

        # info 3. plot sparsity evolution if we have history
        if len(self.sparsity_history) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))

            # info extract epochs and sparsity values
            epochs = sorted(self.sparsity_history.keys())

            # info plot for each layer
            for layer in layers:
                sparsity_trend = [self.sparsity_history[e]['avg_sparsity'].get(layer, 0)
                                  for e in epochs]
                ax.plot(epochs, sparsity_trend, 'o-', label=layer)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Sparsity (% inactive neurons)')
            ax.set_title('MLP Sparsity Evolution Over Time')
            ax.legend()

            plt.tight_layout()
            plt.suptitle(f"{self.model.plot_prefix}")
            plt.savefig(viz_dir / "sparsity_evolution.png")
            plt.close(fig)

    def _store_selective_activations(self, layer_name, activations, epoch):
        """Store only necessary activation data to reduce memory usage"""
        # Convert to numpy to reduce memory overhead
        if isinstance(activations, torch.Tensor):
            # Store only statistics rather than full activations
            activation_mean = activations.mean().item()
            activation_sparsity = (activations <= self.activation_threshold).float().mean().item()
            activation_std = activations.std().item()

            # Store compressed representation
            self.activation_stats.setdefault(layer_name, {})[epoch] = {
                'mean': activation_mean,
                'sparsity': activation_sparsity,
                'std': activation_std
            }

            # For selective neurons, store indices only
            active_neuron_indices = torch.where(activations.mean(dim=1) > self.activation_threshold)[0].tolist()
            self.active_neurons.setdefault(layer_name, {})[epoch] = active_neuron_indices

            # Return memory to system
            del activations
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return None  # Don't store the full tensor

    def _cleanup(self):
        """Remove hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

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
