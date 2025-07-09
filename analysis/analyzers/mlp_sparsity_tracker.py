# mlp_sparsity_tracker.py
from typing import Dict, List, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from analysis.core import CanonicalRegistryAdapter
# from analysis.core.circuit_registry import EnhancedCircuitRegistry as CircuitRegistry
from analysis.core.circuit_schema import Element, ElementType, Circuit, CircuitType
from analysis.core.json_safe_analyzer import JSONSafeAnalyzer
from analysis.core.unified_logger import UnifiedLogger


class MLPSparsitySignatureExtractor:
    """Extracts MLP sparsity-specific signatures for canonical circuit registration"""

    def __init__(self):
        self.signature_type = "mlp_sparsity"

    def extract_circuit_signature(self, circuit: Circuit, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive signature for MLP sparsity circuit"""

        signature = {
            'signature_type': self.signature_type,
            'circuit_type': circuit.type.value if hasattr(circuit.type, 'value') else str(circuit.type),
            'layer_info': circuit.metadata.get('layer', 'unknown'),
            'sparsity_analysis': self._extract_sparsity_signature(circuit),
            'neuron_analysis': self._extract_neuron_signature(circuit),
            'activation_patterns': self._extract_activation_signature(circuit),
            'functional_analysis': self._analyze_sparsity_function(circuit),
            'stability_metrics': self._extract_stability_metrics(circuit, context),
            'detection_context': {
                'epoch': context.get('epoch', 0),
                'method': context.get('detection_method', 'unknown'),
                'confidence': context.get('detection_confidence', 0.5)
            }
        }

        return signature

    def _extract_sparsity_signature(self, circuit: Circuit) -> Dict[str, Any]:
        """Extract sparsity-specific signature components"""

        sparsity_sig = {
            'sparsity_level': circuit.metadata.get('sparsity', 0.0),
            'sparsity_type': circuit.metadata.get('sparsity_type', 'unknown'),
            'activation_threshold': circuit.metadata.get('activation_threshold', 0.1),
            'density_profile': circuit.metadata.get('density_profile', {}),
            'sparse_pattern_type': self._classify_sparsity_pattern(circuit)
        }

        return sparsity_sig

    def _extract_neuron_signature(self, circuit: Circuit) -> Dict[str, Any]:
        """Extract neuron-specific signature components"""

        neuron_sig = {
            'neuron_indices': circuit.metadata.get('neuron_indices', []),
            'neuron_count': circuit.metadata.get('neuron_count', 0),
            'layer_distribution': circuit.metadata.get('layer_distribution', {}),
            'activation_strength': circuit.metadata.get('avg_activation', 0.0),
            'selectivity_score': circuit.metadata.get('selectivity', 0.0)
        }

        # Extract from circuit elements
        if hasattr(circuit, 'elements'):
            for element in circuit.elements:
                if hasattr(element, 'component_data') and element.component_data:
                    comp_data = element.component_data

                    if 'neuron_idx' in comp_data:
                        if comp_data['neuron_idx'] not in neuron_sig['neuron_indices']:
                            neuron_sig['neuron_indices'].append(comp_data['neuron_idx'])

                    if 'layer_idx' in comp_data:
                        layer_key = f"layer_{comp_data['layer_idx']}"
                        neuron_sig['layer_distribution'][layer_key] = neuron_sig['layer_distribution'].get(layer_key,
                                                                                                           0) + 1

        return neuron_sig

    def _extract_activation_signature(self, circuit: Circuit) -> Dict[str, Any]:
        """Extract activation pattern signature"""

        activation_sig = {
            'activation_pattern': circuit.metadata.get('activation_pattern', 'unknown'),
            'response_selectivity': circuit.metadata.get('response_selectivity', {}),
            'class_preferences': circuit.metadata.get('class_preferences', {}),
            'temporal_dynamics': circuit.metadata.get('temporal_dynamics', {})
        }

        return activation_sig

    def _classify_sparsity_pattern(self, circuit: Circuit) -> str:
        """Classify the type of sparsity pattern"""

        # Check metadata for explicit pattern type
        if 'sparsity_pattern_type' in circuit.metadata:
            return circuit.metadata['sparsity_pattern_type']

        # Infer from sparsity level and other characteristics
        sparsity_level = circuit.metadata.get('sparsity', 0.0)

        if sparsity_level > 0.8:
            return 'highly_sparse'
        elif sparsity_level > 0.5:
            return 'moderately_sparse'
        elif sparsity_level > 0.2:
            return 'low_sparse'
        else:
            return 'dense'

    def _analyze_sparsity_function(self, circuit: Circuit) -> str:
        """Infer functional role of sparse circuit"""

        # Check for explicit functional role
        if 'functional_role' in circuit.metadata:
            return circuit.metadata['functional_role']

        # Infer from pattern characteristics
        selectivity = circuit.metadata.get('selectivity', 0.0)
        class_specific = bool(circuit.metadata.get('class_preferences', {}))

        if selectivity > 0.8:
            return 'highly_selective_feature'
        elif class_specific:
            return 'class_discriminative_feature'
        elif circuit.metadata.get('sparsity', 0.0) > 0.7:
            return 'sparse_feature_detector'
        else:
            return 'general_feature_processor'

    def _extract_stability_metrics(self, circuit: Circuit, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract stability-related metrics for sparsity circuits"""

        stability = {
            'detection_confidence': circuit.attribution if hasattr(circuit, 'attribution') else 0.5,
            'sparsity_stability': context.get('sparsity_stability', 0.0),
            'activation_consistency': context.get('activation_consistency', 0.0),
            'temporal_persistence': context.get('temporal_persistence', 0.0),
            'cross_example_stability': context.get('cross_example_stability', 0.0)
        }

        return stability

class JSONSafeMLPSparsityTracker(JSONSafeAnalyzer):
    """Track the development of sparse representations in MLP layers"""

    def __init__(self, model, save_dir, logger=None, canonical_registry=None, activation_threshold=0.1):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.activation_threshold = activation_threshold

        # info use the canonical registry
        self.canonical_registry = canonical_registry
        if self.canonical_registry is None:
            print("⚠️ JSONSafeMLPSparsityTracker: No canonical registry provided!")

        # info storage for tracking sparsity evolution
        self.sparsity_history = {}
        self.neuron_class_selectivity = {}
        self.activation_patterns = {}

        self.activation_stats = {}      # info compressed activation statistics
        self.active_neurons = {}        # info indices of active neurons in layers

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

        # info store selective activations
        for layer_name, activations in self.layer_activations.items():
            if "mlp_expanded" in layer_name:
                self._store_selective_activations(layer_name, activations, epoch=None)

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

        # info Create circuits from sparsity patterns
        sparsity_circuits = self._create_sparsity_circuits(analysis_results, epoch)

        # info register circuits if registry is available
        if self.canonical_registry:
            for circuit in sparsity_circuits:
                self.canonical_registry.register_circuit(circuit, source="sparsity_analysis")

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

        return {
            **analysis_results,
            'sparsity_circuits': sparsity_circuits,
        }

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

    def _create_sparsity_circuits(self, analysis_results, epoch):
        """Create circuits from identified sparsity patterns"""
        circuits = []

        # 1. Create circuits for sparse subspaces
        sparse_subspace_circuits = self._create_sparse_subspace_circuits(
            analysis_results['avg_sparsity'], epoch)
        circuits.extend(sparse_subspace_circuits)

        # 2. Create circuits for class-selective neuron groups
        selective_circuits = self._create_selective_neuron_circuits(
            analysis_results['class_selectivity'], epoch)
        circuits.extend(selective_circuits)

        # 3. Create circuits for co-active neuron clusters
        cluster_circuits = self._create_coactive_cluster_circuits(
            analysis_results['sample_activations'], epoch)
        circuits.extend(cluster_circuits)

        return circuits

    def _create_sparse_subspace_circuits(self, avg_sparsity, epoch):
        """Create circuits for layers with high sparsity (sparse subspaces)"""
        circuits = []
        sparsity_threshold = 0.7  # Only create circuits for highly sparse layers

        for layer_name, sparsity in avg_sparsity.items():
            if sparsity >= sparsity_threshold:
                # Extract layer index
                layer_idx = self._extract_layer_index(layer_name)

                # Generate circuit ID
                circuit_id = self.canonical_registry.generate_circuit_id(
                    operation_type="sparse_subspace",
                    component_info=f"layer_{layer_idx}",
                    epoch=epoch,
                    sparsity_level=sparsity,
                    source="sparsity_analysis"
                ) if self.canonical_registry else f"sparse_subspace_layer_{layer_idx}_{epoch}"

                # Create subspace element
                subspace_element = Element(
                    id=f"layer_{layer_idx}_sparse_subspace",
                    type=ElementType.SUBSPACE,
                    properties={
                        "layer": layer_idx,
                        "sparsity_level": sparsity,
                        "activation_threshold": self.activation_threshold
                    }
                )

                circuit = Circuit(
                    id=circuit_id,
                    type=CircuitType.FUNCTIONAL,
                    elements=[subspace_element],
                    connections=[],
                    attribution=sparsity,  # Higher sparsity = higher attribution for sparse circuits
                    metadata={
                        "operation_type": "sparse_subspace",
                        "layer": layer_idx,
                        "sparsity_level": sparsity,
                        "circuit_class": "computational_efficiency"
                    },
                    discovered_at=epoch
                )

                circuits.append(circuit)

        return circuits

    def _create_selective_neuron_circuits(self, class_selectivity, epoch):
        """Create circuits for groups of class-selective neurons"""
        circuits = []

        for layer_name, layer_selectivity in class_selectivity.items():
            if not layer_selectivity:
                continue

            layer_idx = self._extract_layer_index(layer_name)

            # Group neurons by their preferred class
            neurons_by_class = {}
            for neuron_id, neuron_data in layer_selectivity.items():
                preferred_class = neuron_data.get('preferred_class')
                selectivity_score = neuron_data.get('score', 0)

                if preferred_class is not None and selectivity_score > 0.3:  # High selectivity threshold
                    if preferred_class not in neurons_by_class:
                        neurons_by_class[preferred_class] = []

                    neurons_by_class[preferred_class].append({
                        'neuron_id': neuron_id,
                        'selectivity_score': selectivity_score
                    })

            # Create circuit for each class with sufficient selective neurons
            for class_label, neurons in neurons_by_class.items():
                if len(neurons) >= 3:  # Need at least 3 selective neurons
                    avg_selectivity = np.mean([n['selectivity_score'] for n in neurons])
                    neuron_indices = [int(n['neuron_id'].replace('neuron_', '')) for n in neurons]

                    # Generate circuit ID
                    circuit_id = self.canonical_registry.generate_circuit_id(
                        operation_type="class_selective_neurons",
                        component_info=f"layer_{layer_idx}_class_{class_label}",
                        epoch=epoch,
                        selectivity_score=avg_selectivity,
                        neuron_count=len(neurons),
                        source="selectivity_analysis"
                    ) if self.canonical_registry else f"selective_neurons_layer_{layer_idx}_class_{class_label}_{epoch}"

                    # Create selective subspace element
                    selective_element = Element(
                        id=f"layer_{layer_idx}_class_{class_label}_selective",
                        type=ElementType.SUBSPACE,
                        properties={
                            "layer": layer_idx,
                            "preferred_class": class_label,
                            "selective_neurons": neuron_indices,
                            "avg_selectivity": avg_selectivity
                        }
                    )

                    circuit = Circuit(
                        id=circuit_id,
                        type=CircuitType.FUNCTIONAL,
                        elements=[selective_element],
                        connections=[],
                        attribution=avg_selectivity,
                        metadata={
                            "operation_type": "class_selective_neurons",
                            "layer": layer_idx,
                            "preferred_class": class_label,
                            "neuron_count": len(neurons),
                            "avg_selectivity": avg_selectivity,
                            "circuit_class": "class_discrimination"
                        },
                        discovered_at=epoch
                    )

                    circuits.append(circuit)

        return circuits

    def _create_coactive_cluster_circuits(self, sample_activations, epoch, min_cluster_size=5):
        """Create circuits for clusters of neurons that consistently activate together"""
        circuits = []

        if not sample_activations:
            return circuits

        # Group activations by layer
        layer_activations = {}
        for input_key, activation_data in sample_activations.items():
            for layer_name, activations in activation_data.items():
                if layer_name not in layer_activations:
                    layer_activations[layer_name] = []
                layer_activations[layer_name].append(activations)

        # Find co-active clusters for each layer
        for layer_name, all_activations in layer_activations.items():
            if len(all_activations) < 3:  # Need sufficient samples
                continue

            layer_idx = self._extract_layer_index(layer_name)

            # Stack activations: [n_samples, n_neurons]
            activation_matrix = np.array(all_activations)

            # Find clusters of co-active neurons using correlation
            neuron_correlations = np.corrcoef(activation_matrix.T)  # [n_neurons, n_neurons]

            # Find strongly correlated neuron groups
            clusters = self._find_correlation_clusters(neuron_correlations, correlation_threshold=0.7)

            for cluster_idx, neuron_indices in enumerate(clusters):
                if len(neuron_indices) >= min_cluster_size:
                    # Calculate cluster strength (average within-cluster correlation)
                    cluster_correlations = neuron_correlations[np.ix_(neuron_indices, neuron_indices)]
                    avg_correlation = np.mean(cluster_correlations[np.triu_indices_from(cluster_correlations, k=1)])

                    # Generate circuit ID
                    circuit_id = self.canonical_registry.generate_circuit_id(
                        operation_type="coactive_cluster",
                        component_info=f"layer_{layer_idx}_cluster_{cluster_idx}",
                        epoch=epoch,
                        correlation_strength=avg_correlation,
                        cluster_size=len(neuron_indices),
                        source="coactivation_analysis"
                    ) if self.canonical_registry else f"coactive_cluster_layer_{layer_idx}_c{cluster_idx}_{epoch}"

                    # Create cluster element
                    cluster_element = Element(
                        id=f"layer_{layer_idx}_cluster_{cluster_idx}",
                        type=ElementType.SUBSPACE,
                        properties={
                            "layer": layer_idx,
                            "cluster_neurons": neuron_indices,
                            "avg_correlation": avg_correlation,
                            "cluster_size": len(neuron_indices)
                        }
                    )

                    circuit = Circuit(
                        id=circuit_id,
                        type=CircuitType.COMPONENT,  # Co-active clusters are component-level
                        elements=[cluster_element],
                        connections=[],
                        attribution=avg_correlation,
                        metadata={
                            "operation_type": "coactive_cluster",
                            "layer": layer_idx,
                            "cluster_size": len(neuron_indices),
                            "avg_correlation": avg_correlation,
                            "circuit_class": "neural_coordination"
                        },
                        discovered_at=epoch
                    )

                    circuits.append(circuit)

        return circuits

    def _find_correlation_clusters(self, correlation_matrix, correlation_threshold=0.7):
        """Find clusters of highly correlated neurons"""
        n_neurons = correlation_matrix.shape[0]
        visited = np.zeros(n_neurons, dtype=bool)
        clusters = []

        for i in range(n_neurons):
            if visited[i]:
                continue

            # Find all neurons correlated with neuron i
            cluster = [i]
            visited[i] = True

            # Expand cluster by finding correlated neurons
            for j in range(i + 1, n_neurons):
                if not visited[j] and correlation_matrix[i, j] >= correlation_threshold:
                    # Check if j is also correlated with existing cluster members
                    correlated_with_cluster = all(
                        correlation_matrix[j, k] >= correlation_threshold * 0.8  # Slightly lower threshold
                        for k in cluster
                    )

                    if correlated_with_cluster:
                        cluster.append(j)
                        visited[j] = True

            if len(cluster) > 1:  # Only keep multi-neuron clusters
                clusters.append(cluster)

        return clusters

    def _extract_layer_index(self, layer_name):
        """Extract layer index from layer name like 'layer_0_mlp_expanded'"""
        parts = layer_name.split('_')
        for i, part in enumerate(parts):
            if part == 'layer' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return 0  # Default fallback




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


class CanonicalAwareMLP_SparsityTracker:
    """
    Enhanced canonical-aware wrapper for JSONSafeMLPSparsityTracker

    Improvements over original:
    - Robust method discovery (no more uncertain if/elif chains)
    - UnifiedLogger integration
    - MLP sparsity-specific signature extraction
    - Comprehensive error handling
    - Registration statistics tracking
    - Detailed result reporting
    """

    def __init__(self, model, save_dir, logger: UnifiedLogger = None,
                 canonical_adapter: CanonicalRegistryAdapter = None,
                 activation_threshold: float = 0.1):

        # Core components
        self.model = model
        self.canonical_adapter = canonical_adapter
        self.logger = logger  # UnifiedLogger as requested
        self.activation_threshold = activation_threshold  # ✅ Fix: Initialize before use

        # Signature extraction
        self.signature_extractor = MLPSparsitySignatureExtractor()

        # ✅ Preserve excellent design: delegate to existing tracker
        self.mlp_tracker = JSONSafeMLPSparsityTracker(
            model=model,
            save_dir=save_dir,
            logger=logger,  # Pass UnifiedLogger to underlying tracker
            canonical_registry=canonical_adapter.canonical_registry if canonical_adapter else None,
            activation_threshold=activation_threshold
        )

        # Enhanced capabilities
        self._detection_methods = self._discover_available_methods()
        self._method_fallback_chain = self._build_fallback_chain()

        # Statistics tracking
        from collections import defaultdict
        self.registration_stats = {
            'total_attempts': 0,
            'successful_new': 0,
            'successful_aggregated': 0,
            'failed_registrations': 0,
            'methods_used': defaultdict(int),
            'error_types': defaultdict(int)
        }

        # Stability tracking for temporal consistency
        self.circuit_stability_tracker = {}

        if self.logger:
            self.logger.info(
                f"Enhanced MLP Sparsity Tracker initialized with methods: {list(self._detection_methods.keys())}")

    def _discover_available_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        Robust discovery of available detection methods in underlying tracker
        No more uncertain if/elif chains!
        """

        methods = {}

        # Check for sparsity analysis methods
        if hasattr(self.mlp_tracker, 'analyze_neuron_activity'):
            methods['neuron_activity'] = {
                'method_name': 'analyze_neuron_activity',
                'requires_eval_loader': True,
                'requires_class_labels': True,
                'priority': 1,
                'description': 'Neuron activity analysis with sparsity detection'
            }

        if hasattr(self.mlp_tracker, 'detect_sparse_features'):
            methods['sparse_features'] = {
                'method_name': 'detect_sparse_features',
                'requires_eval_loader': False,
                'requires_class_labels': False,
                'priority': 2,
                'description': 'Direct sparse feature detection'
            }

        if hasattr(self.mlp_tracker, 'analyze_sparsity_patterns'):
            methods['sparsity_patterns'] = {
                'method_name': 'analyze_sparsity_patterns',
                'requires_eval_loader': False,
                'requires_class_labels': False,
                'priority': 3,
                'description': 'Sparsity pattern analysis'
            }

        if hasattr(self.mlp_tracker, 'track_sparsity_evolution'):
            methods['sparsity_evolution'] = {
                'method_name': 'track_sparsity_evolution',
                'requires_eval_loader': True,
                'requires_class_labels': False,
                'priority': 4,
                'description': 'Sparsity evolution tracking'
            }

        # Check for circuit creation methods
        if hasattr(self.mlp_tracker, '_create_sparsity_circuits'):
            methods['create_circuits'] = {
                'method_name': '_create_sparsity_circuits',
                'requires_eval_loader': False,
                'requires_class_labels': False,
                'priority': 5,
                'description': 'Create circuits from sparsity analysis'
            }

        if self.logger:
            for method_key, method_info in methods.items():
                self.logger.debug(f"Discovered method '{method_key}': {method_info['description']}")

        return methods

    def _build_fallback_chain(self) -> List[str]:
        """Build ordered fallback chain based on method priorities"""

        # Sort methods by priority
        sorted_methods = sorted(
            self._detection_methods.items(),
            key=lambda x: x[1]['priority']
        )

        fallback_chain = [method_key for method_key, _ in sorted_methods]

        if self.logger:
            self.logger.debug(f"Method fallback chain: {' -> '.join(fallback_chain)}")

        return fallback_chain

    def detect_mlp_sparsity_circuits(self, epoch: int, mlp_activations: Dict = None,
                                     tokens: List[str] = None, eval_loader=None,
                                     class_labels=None, **context) -> Dict[str, Any]:
        """
        Enhanced MLP sparsity circuit detection with comprehensive error handling and reporting

        Maintains backward compatibility while providing rich analysis results
        """

        if self.logger:
            self.logger.info(f"Starting MLP sparsity circuit detection at epoch {epoch}")
            self.logger.debug(f"Context: eval_loader={eval_loader is not None}, "
                              f"class_labels={class_labels is not None}, tokens={len(tokens or [])}")

        # Initialize comprehensive results
        results = {
            'canonical_circuits': [],  # Backward compatible return value
            'raw_circuits': [],
            'registration_summary': {
                'attempted': 0,
                'succeeded_new': 0,
                'succeeded_aggregated': 0,
                'failed': 0
            },
            'analysis_summary': {
                'methods_attempted': [],
                'methods_succeeded': [],
                'methods_failed': [],
                'total_detections': 0,
                'avg_sparsity_level': 0.0,
                'detection_confidence_avg': 0.0
            },
            'performance_metrics': {
                'detection_time': 0.0,
                'registration_time': 0.0,
                'method_performance': {}
            }
        }

        # Enhanced context with all available information
        enhanced_context = {
            'epoch': epoch,
            'mlp_activations': mlp_activations,
            'eval_loader': eval_loader,
            'class_labels': class_labels,
            'tokens': tokens or [],
            **context
        }

        detected_circuits = []

        # Try methods in fallback chain order
        import time
        detection_start = time.time()

        for method_key in self._method_fallback_chain:
            method_info = self._detection_methods[method_key]
            method_start = time.time()

            try:
                if self.logger:
                    self.logger.debug(f"Attempting detection with method: {method_key}")

                results['analysis_summary']['methods_attempted'].append(method_key)

                # Check if method requirements are met
                if not self._check_method_requirements(method_info, enhanced_context):
                    if self.logger:
                        self.logger.debug(f"Method {method_key} requirements not met, skipping")
                    continue

                # Execute detection method
                method_circuits = self._execute_detection_method(method_key, method_info, enhanced_context)

                if method_circuits:
                    detected_circuits.extend(method_circuits)
                    results['analysis_summary']['methods_succeeded'].append(method_key)
                    self.registration_stats['methods_used'][method_key] += 1

                    method_time = time.time() - method_start
                    results['performance_metrics']['method_performance'][method_key] = {
                        'execution_time': method_time,
                        'circuits_detected': len(method_circuits),
                        'success': True
                    }

                    if self.logger:
                        self.logger.debug(
                            f"Method {method_key} detected {len(method_circuits)} circuits in {method_time:.3f}s")

                else:
                    results['performance_metrics']['method_performance'][method_key] = {
                        'execution_time': time.time() - method_start,
                        'circuits_detected': 0,
                        'success': True
                    }

            except Exception as e:
                method_time = time.time() - method_start
                results['analysis_summary']['methods_failed'].append(method_key)
                self.registration_stats['error_types'][f"{method_key}_error"] += 1

                results['performance_metrics']['method_performance'][method_key] = {
                    'execution_time': method_time,
                    'circuits_detected': 0,
                    'success': False,
                    'error': str(e)
                }

                if self.logger:
                    self.logger.warning(f"Method {method_key} failed: {e}")

                continue

        # Fallback if no methods succeeded
        if not detected_circuits:
            if self.logger:
                self.logger.info("All primary methods failed, attempting basic fallback")

            try:
                fallback_circuits = self._basic_fallback_detection(enhanced_context)
                if fallback_circuits:
                    detected_circuits.extend(fallback_circuits)
                    results['analysis_summary']['methods_succeeded'].append('basic_fallback')
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Even fallback detection failed: {e}")

        results['performance_metrics']['detection_time'] = time.time() - detection_start
        results['raw_circuits'] = detected_circuits
        results['analysis_summary']['total_detections'] = len(detected_circuits)

        # Calculate analysis metrics
        if detected_circuits:
            sparsity_levels = [
                circuit.metadata.get('sparsity', 0.0) for circuit in detected_circuits
            ]
            results['analysis_summary']['avg_sparsity_level'] = sum(sparsity_levels) / len(sparsity_levels)

            confidences = [
                getattr(circuit, 'attribution', 0.5) for circuit in detected_circuits
            ]
            results['analysis_summary']['detection_confidence_avg'] = sum(confidences) / len(confidences)

        # Register circuits canonically
        if detected_circuits:
            registration_start = time.time()
            canonical_ids = self._register_circuits_with_comprehensive_tracking(
                circuits=detected_circuits,
                epoch=epoch,
                tokens=tokens or [],
                context=enhanced_context,
                registration_summary=results['registration_summary']
            )
            results['canonical_circuits'] = canonical_ids
            results['performance_metrics']['registration_time'] = time.time() - registration_start

        # Update overall statistics
        self._update_overall_statistics(results)

        if self.logger:
            summary = results['analysis_summary']
            reg_summary = results['registration_summary']
            self.logger.info(f"MLP sparsity detection complete: {summary['total_detections']} circuits detected, "
                             f"avg sparsity: {summary['avg_sparsity_level']:.3f}, "
                             f"{reg_summary['succeeded_new']} new registrations, "
                             f"{reg_summary['succeeded_aggregated']} aggregations")

        return results

    def _check_method_requirements(self, method_info: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if method requirements are met"""

        if method_info.get('requires_eval_loader', False) and context.get('eval_loader') is None:
            return False

        if method_info.get('requires_class_labels', False) and context.get('class_labels') is None:
            return False

        return True

    def _execute_detection_method(self, method_key: str, method_info: Dict[str, Any],
                                  context: Dict[str, Any]) -> List[Circuit]:
        """Execute a specific detection method with proper parameter handling"""

        circuits = []
        method_name = method_info['method_name']
        method = getattr(self.mlp_tracker, method_name)

        if method_key == 'neuron_activity':
            # Special handling for analyze_neuron_activity
            circuits = self._execute_neuron_activity_method(method, context)

        elif method_key == 'create_circuits':
            # Special handling for _create_sparsity_circuits
            circuits = self._execute_circuit_creation_method(method, context)

        else:
            # Direct method execution
            circuits = self._execute_direct_mlp_method(method, method_info, context)

        return circuits

    def _execute_neuron_activity_method(self, method, context: Dict[str, Any]) -> List[Circuit]:
        """Execute the analyze_neuron_activity method"""

        circuits = []

        try:
            # Get required parameters
            eval_loader = context.get('eval_loader')
            class_labels = context.get('class_labels')

            if eval_loader is None:
                if self.logger:
                    self.logger.warning("analyze_neuron_activity requires eval_loader")
                return circuits

            # Execute analysis
            analysis_results = method(eval_loader, class_labels)

            # Create circuits from analysis results
            if hasattr(self.mlp_tracker, '_create_sparsity_circuits'):
                circuits = self.mlp_tracker._create_sparsity_circuits(
                    analysis_results=analysis_results,
                    epoch=context['epoch']
                )

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Neuron activity analysis failed: {e}")

        return circuits

    def _execute_circuit_creation_method(self, method, context: Dict[str, Any]) -> List[Circuit]:
        """Execute circuit creation methods"""

        circuits = []

        try:
            # This method typically requires analysis results
            # For now, return empty - would need prior analysis step
            pass

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Circuit creation method failed: {e}")

        return circuits

    def _execute_direct_mlp_method(self, method, method_info: Dict[str, Any],
                                   context: Dict[str, Any]) -> List[Circuit]:
        """Execute methods that can be called directly"""

        circuits = []

        try:
            # Prepare arguments based on method requirements
            method_args = {}

            # Add epoch if method expects it
            if 'epoch' in method.__code__.co_varnames:
                method_args['epoch'] = context['epoch']

            # Add activations if method expects them
            if 'activations' in method.__code__.co_varnames or 'mlp_activations' in method.__code__.co_varnames:
                method_args['activations'] = context.get('mlp_activations')

            # Add eval_loader if method expects it
            if method_info.get('requires_eval_loader', False):
                method_args['eval_loader'] = context.get('eval_loader')

            # Execute method
            result = method(**method_args)

            # Handle different return types
            if isinstance(result, list):
                circuits = result
            elif isinstance(result, dict):
                if 'circuits' in result:
                    circuits = result['circuits']
                elif 'detected_circuits' in result:
                    circuits = result['detected_circuits']

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Direct method execution failed: {e}")

        return circuits

    def _basic_fallback_detection(self, context: Dict[str, Any]) -> List[Circuit]:
        """Basic fallback detection when all other methods fail"""

        circuits = []

        try:
            # Create simple circuits based on MLP layer analysis
            for name, module in self.model.named_modules():
                if ('mlp' in name.lower() or 'ffn' in name.lower()) and hasattr(module, 'weight'):

                    # Calculate basic sparsity statistics
                    with torch.no_grad():
                        weight_data = module.weight.data.detach().cpu()

                        # Calculate sparsity (percentage of near-zero weights)
                        sparsity = (torch.abs(weight_data) < 1e-6).float().mean().item()

                        # Create circuit if significant sparsity detected
                        if 0.3 < sparsity < 0.95:  # Interesting sparsity range

                            circuit = Circuit(
                                id=f"mlp_fallback_{name}_{context['epoch']}",
                                type=CircuitType.MLP,
                                elements=[Element(
                                    id=f"mlp_fallback_element_{name}",
                                    type=ElementType.MLP_NEURON,
                                    properties={
                                        'module_name': name,
                                        'sparsity': sparsity,
                                        'weight_shape': list(weight_data.shape)
                                    }
                                )],
                                connections=[],
                                attribution=1.0 - sparsity,  # Dense regions as attribution
                                metadata={
                                    'module_name': name,
                                    'sparsity': sparsity,
                                    'detection_method': 'mlp_fallback',
                                    'layer': self._extract_layer_info(name),
                                    'sparsity_type': 'weight_based'
                                },
                                discovered_at=context['epoch']
                            )

                            circuits.append(circuit)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Basic MLP fallback detection failed: {e}")

        return circuits

    def _extract_layer_info(self, module_name: str) -> str:
        """Extract layer information from module name"""

        parts = module_name.split('.')
        for i, part in enumerate(parts):
            if any(keyword in part.lower() for keyword in ['layer', 'block', 'transformer']):
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    return f"layer_{parts[i + 1]}"
        return 'unknown'

    def _register_circuits_with_comprehensive_tracking(self, circuits: List[Circuit], epoch: int,
                                                       tokens: List[str], context: Dict[str, Any],
                                                       registration_summary: Dict[str, int]) -> List[str]:
        """Register circuits with comprehensive tracking and signature extraction"""

        canonical_ids = []

        for circuit in circuits:
            try:
                registration_summary['attempted'] += 1
                self.registration_stats['total_attempts'] += 1

                # Update stability tracking
                stability_score = self._update_circuit_stability_tracking(circuit.id, epoch)

                # Create signature context
                signature_context = {
                    'epoch': epoch,
                    'detection_method': 'mlp_sparsity_analysis',
                    'detection_confidence': getattr(circuit, 'attribution', 0.5),
                    'sparsity_stability': stability_score,
                    'activation_consistency': self._calculate_activation_consistency(circuit),
                    'temporal_persistence': stability_score
                }

                # Extract comprehensive signature
                circuit_signature = self.signature_extractor.extract_circuit_signature(circuit, signature_context)

                # Register with canonical system
                canonical_id, legacy_id = self.canonical_adapter.register_circuit_detection(
                    circuit=circuit,
                    epoch=epoch,
                    tokens=tokens,
                    detection_confidence=signature_context['detection_confidence'],
                    detection_method="enhanced_mlp_sparsity_analysis",
                    example_metadata={
                        'analysis_type': 'enhanced_mlp_sparsity',
                        'signature': circuit_signature,
                        'sparsity_metrics': {
                            'sparsity_level': circuit.metadata.get('sparsity', 0.0),
                            'activation_threshold': self.activation_threshold,
                            'detection_method': circuit.metadata.get('detection_method', 'unknown')
                        },
                        'stability_tracking': {
                            'sparsity_stability': stability_score,
                            'detection_history': self.circuit_stability_tracker.get(circuit.id, {})
                        },
                        'context': context
                    }
                )

                canonical_ids.append(canonical_id)

                # Track registration success type
                canonical_circuit = self.canonical_adapter.canonical_registry.canonical_circuits.get(canonical_id)
                if canonical_circuit:
                    if canonical_circuit.total_detections == 1:
                        registration_summary['succeeded_new'] += 1
                        self.registration_stats['successful_new'] += 1
                    else:
                        registration_summary['succeeded_aggregated'] += 1
                        self.registration_stats['successful_aggregated'] += 1

            except Exception as e:
                registration_summary['failed'] += 1
                self.registration_stats['failed_registrations'] += 1

                if self.logger:
                    self.logger.warning(f"Failed to register circuit {circuit.id}: {e}")

        return canonical_ids

    def _update_circuit_stability_tracking(self, circuit_id: str, epoch: int) -> float:
        """Update stability tracking for circuit and return stability score"""

        if circuit_id not in self.circuit_stability_tracker:
            self.circuit_stability_tracker[circuit_id] = {
                'first_seen': epoch,
                'last_seen': epoch,
                'detection_epochs': [epoch],
                'stability_score': 0.0
            }
        else:
            tracker = self.circuit_stability_tracker[circuit_id]
            tracker['last_seen'] = epoch
            tracker['detection_epochs'].append(epoch)

            # Calculate stability score based on detection frequency
            detection_span = epoch - tracker['first_seen'] + 1
            detection_frequency = len(tracker['detection_epochs']) / detection_span
            tracker['stability_score'] = min(1.0, detection_frequency * 2.0)

        return self.circuit_stability_tracker[circuit_id]['stability_score']

    def _calculate_activation_consistency(self, circuit: Circuit) -> float:
        """Calculate activation consistency score for circuit"""

        # For now, return a default value based on circuit metadata
        # Could be enhanced with actual activation tracking
        sparsity = circuit.metadata.get('sparsity', 0.5)
        selectivity = circuit.metadata.get('selectivity', 0.5)

        # Higher consistency for more selective and stable sparsity patterns
        consistency = (selectivity + (1.0 - abs(sparsity - 0.5) * 2)) / 2.0
        return min(1.0, max(0.0, consistency))

    def _update_overall_statistics(self, results: Dict[str, Any]):
        """Update overall statistics from batch results"""

        # Update method success rates
        for method in results['analysis_summary']['methods_succeeded']:
            self.registration_stats['methods_used'][method] += 1

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about detection and registration performance"""

        stats = {
            'registration_stats': self.registration_stats.copy(),
            'method_performance': dict(self.registration_stats['methods_used']),
            'error_analysis': dict(self.registration_stats['error_types']),
            'stability_tracking': {
                'total_tracked_circuits': len(self.circuit_stability_tracker),
                'stable_circuits': sum(1 for tracker in self.circuit_stability_tracker.values()
                                       if tracker['stability_score'] > 0.7),
                'average_stability': (sum(tracker['stability_score']
                                          for tracker in self.circuit_stability_tracker.values()) /
                                      len(self.circuit_stability_tracker)
                                      if self.circuit_stability_tracker else 0.0)
            },
            'detection_capabilities': {
                'available_methods': list(self._detection_methods.keys()),
                'fallback_chain': self._method_fallback_chain,
                'signature_extraction_enabled': True,
                'stability_tracking_enabled': True
            }
        }

        # Calculate success rates
        total_attempts = stats['registration_stats']['total_attempts']
        if total_attempts > 0:
            stats['registration_stats']['success_rate'] = (
                    (stats['registration_stats']['successful_new'] +
                     stats['registration_stats']['successful_aggregated']) / total_attempts
            )
            stats['registration_stats']['new_circuit_rate'] = (
                    stats['registration_stats']['successful_new'] / total_attempts
            )

        return stats

    def get_detection_method_info(self) -> Dict[str, Any]:
        """Get detailed information about available detection methods"""

        return {
            'discovered_methods': self._detection_methods.copy(),
            'fallback_chain': self._method_fallback_chain,
            'method_capabilities': {
                method_key: {
                    'available': True,
                    'description': method_info['description'],
                    'priority': method_info['priority'],
                    'requirements': {
                        'eval_loader': method_info.get('requires_eval_loader', False),
                        'class_labels': method_info.get('requires_class_labels', False)
                    }
                }
                for method_key, method_info in self._detection_methods.items()
            }
        }

    def __getattr__(self, name):
        """Delegate any other method calls to the existing tracker"""
        return getattr(self.mlp_tracker, name)

