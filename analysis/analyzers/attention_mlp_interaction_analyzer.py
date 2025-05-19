# attention_mlp_interaction_analyzer.py
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
from collections import defaultdict
import seaborn as sns


class AttentionMLPInteractionAnalyzer:
    """Analyze interactions between attention and MLP components"""

    def __init__(self, model, save_dir, logger=None):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger if logger else (model.logger if hasattr(model, 'logger') else None)

        # Storage for analysis results
        self.interaction_history = {}
        self.layer_activations = {}

        # Register hooks for capturing intermediate activations
        self.hooks = []
        self._register_activation_hooks()

    def _register_activation_hooks(self):
        """Register hooks to capture activations at key points in the network"""
        for layer_idx, layer in enumerate(self.model.layers):
            # Hook functions for different components
            def get_hook(layer_idx, component):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self.layer_activations[f"layer_{layer_idx}_{component}"] = output[0].detach()
                    else:
                        self.layer_activations[f"layer_{layer_idx}_{component}"] = output.detach()

                return hook

            # info hook for normalized input to attention
            norm1_hook = get_hook(layer_idx, "norm_1")
            # info hooks for attention outputs and MLP inputs
            attn_hook = get_hook(layer_idx, "attn_out")
            mlp_in_hook = get_hook(layer_idx, "mlp_in")
            mlp_out_hook = get_hook(layer_idx, "mlp_out")

            # Register hooks at appropriate points
            handle0 = layer.ln_1.register_forward_hook(norm1_hook)
            handle1 = layer.attn.register_forward_hook(attn_hook)
            handle2 = layer.ln_2.register_forward_hook(mlp_in_hook)  # After layer norm, before MLP
            handle3 = layer.mlp.register_forward_hook(mlp_out_hook)

            self.hooks.extend([handle0, handle1, handle2, handle3])

    def analyze_interactions(self, epoch, eval_loader, batch_limit=10):
        """Analyze interactions between attention and MLP components"""
        self.model.eval()

        # Storage for batch analysis
        batch_correlations = defaultdict(list)
        information_flow = defaultdict(list)

        # Process batches
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                if batch_idx >= batch_limit:
                    break

                # Forward pass to trigger hooks
                outputs = self.model(inputs)
                predictions = outputs.argmax(dim=-1)

                # Analyze interactions from activations
                for layer_idx in range(len(self.model.layers)):
                    # Get activations for this layer
                    attn_key = f"layer_{layer_idx}_attn_out"
                    mlp_in_key = f"layer_{layer_idx}_mlp_in"

                    if attn_key in self.layer_activations and mlp_in_key in self.layer_activations:
                        attn_out = self.layer_activations[attn_key]
                        mlp_in = self.layer_activations[mlp_in_key]

                        # Handle tuple outputs
                        if isinstance(attn_out, tuple):
                            attn_out = attn_out[0]
                        if isinstance(mlp_in, tuple):
                            mlp_in = mlp_in[0]

                        # Compute correlation between attention outputs and MLP inputs
                        correlation = self._compute_representation_correlation(attn_out, mlp_in)
                        batch_correlations[f"layer_{layer_idx}"].append(correlation)

                        # For correctly predicted samples, analyze information flow
                        for i in range(len(inputs)):
                            if predictions[i] == targets[i]:
                                # Get head-specific activations
                                head_activations = self._extract_head_activations(layer_idx, i)

                                # Get MLP input for this sample
                                sample_mlp_in = mlp_in[:, i] if mlp_in.dim() > 1 else mlp_in

                                # Fix: Calculate contribution of each head to the MLP input
                                for head_idx, head_act in enumerate(head_activations):
                                    # For each head, calculate how much it contributes to the MLP input
                                    # We need to project the head activation to the full model dimension

                                    # Instead of correlation, calculate contribution measure
                                    # Option 1: Project head activation to full dimension first
                                    try:
                                        # Get projection weights for this head (simplified approach)
                                        head_dim = self.model.dim // self.model.num_heads
                                        head_start = head_idx * head_dim
                                        head_end = (head_idx + 1) * head_dim

                                        # Get output projection weights for this head
                                        layer = self.model.layers[layer_idx]
                                        head_out_weights = layer.attn.out_proj.weight[:, head_start:head_end]
                                        # Instead of directly trying to correlate the vectors with mismatched shapes
                                        if isinstance(head_act, torch.Tensor) and head_act.numel() == head_dim:
                                            projected_head = torch.matmul(head_out_weights, head_act)

                                            # fixme warning sample_mlp_in may be2D, while the projected_head is 1D,
                                            #  we need to handle this dimensionality issue in a correct way

                                            # info if sample_mlp_in is 2D (as here [deq_len, hidden_dim]),
                                            #  there is a need to select the appropriate position, ie. the output,
                                            #  usually the last one, as used in classification
                                            if sample_mlp_in.dim() > 1:
                                                # info get the last token position
                                                last_pos = sample_mlp_in.dim() - 1
                                                sample_mlp_vector = sample_mlp_in[last_pos]
                                            correlation = self._compute_vector_correlation(
                                                projected_head.flatten(), sample_mlp_vector.flatten())
                                        else:
                                            # Fallback if head activation isn't as expected
                                            correlation = 0.0
                                    except Exception as e:
                                        # Fallback if projection fails
                                        print(f"Warning: Failed to project head activation: {e}")
                                        correlation = 0.0

                                    information_flow[f"layer_{layer_idx}_head_{head_idx}"].append({
                                        'correlation': correlation,
                                        'class_id': targets[i].item()
                                    })

        # Compute average correlations
        avg_correlations = {layer: np.mean(correlations)
                            for layer, correlations in batch_correlations.items()}

        # Analyze head-MLP flow statistics
        flow_stats = self._analyze_information_flow(information_flow)

        # Build interaction graph
        interaction_graph = self._build_interaction_graph(flow_stats)

        # Store results in history
        self.interaction_history[epoch] = {
            'layer_correlations': avg_correlations,
            'flow_stats': flow_stats,
            'interaction_graph': interaction_graph
        }

        # Log results
        if self.logger:
            for layer, correlation in avg_correlations.items():
                self.logger.log_data('attn_mlp_interaction',
                                     f'{layer}_correlation',
                                     correlation)

            # Log key flow statistics
            for component, stats in flow_stats.items():
                if stats['is_significant']:
                    self.logger.log_data('attn_mlp_interaction',
                                         f'{component}_flow_strength',
                                         stats['avg_correlation'])

        # Visualize results
        self._visualize_interactions(epoch, avg_correlations, flow_stats, interaction_graph)

        return {
            'layer_correlations': avg_correlations,
            'flow_stats': flow_stats,
            'interaction_graph': interaction_graph
        }

    def _compute_representation_correlation(self, tensor1, tensor2):
        """Compute correlation between two representation tensors"""
        # Flatten last dimension (representation dimension)
        t1_flat = tensor1.reshape(tensor1.shape[0], -1)
        t2_flat = tensor2.reshape(tensor2.shape[0], -1)

        # Compute correlation for each sample, then average
        sample_correlations = []
        for i in range(t1_flat.shape[1]):
            corr = torch.corrcoef(torch.stack([t1_flat[:, i], t2_flat[:, i]]))[0, 1].item()
            if not np.isnan(corr):
                sample_correlations.append(corr)

        return np.mean(sample_correlations) if sample_correlations else 0

    def _compute_vector_correlation(self, v1, v2):
        """Compute correlation between two vectors"""
        corr = torch.corrcoef(torch.stack([v1, v2]))[0, 1].item()
        return corr if not np.isnan(corr) else 0

    """
    def _extract_head_activations(self, layer_idx, sample_idx):
        ""Extract per-head activations from the attention layer""
        # This method needs to be adapted based on your model architecture
        # For most transformer implementations, we'd need to:
        # 1. Access the attention weights
        # 2. Apply them to the value projections
        # 3. Extract per-head outputs

        # Simplified implementation - assume we can get attention patterns
        attn_key = f"layer_{layer_idx}_attn_out"
        attn_out = self.layer_activations[attn_key][:, sample_idx]

        # Split by heads
        head_dim = self.model.dim // self.model.num_heads
        head_activations = []

        for head_idx in range(self.model.num_heads):
            start_idx = head_idx * head_dim
            end_idx = (head_idx + 1) * head_dim
            head_act = attn_out[start_idx:end_idx]
            head_activations.append(head_act)

        return head_activations
    """

    def _previous_extract_head_activations(self, layer_idx, sample_idx):
        """Extract per-head activations from the attention layer"""
        # For PyTorch's nn.MultiheadAttention, we need to:
        # 1. Get the attention weights previously stored
        # 2. Get the value projections (we'll need to compute these)
        # 3. Apply weights to values to get per-head activations

        # Check if attention weights are stored
        if not hasattr(self.model.layers[layer_idx], 'attention_weights') or \
                self.model.layers[layer_idx].attention_weights is None:
            # If not, we need a fallback
            attn_key = f"layer_{layer_idx}_attn_out"
            if attn_key not in self.layer_activations:
                # No way to extract head-specific activations
                return [torch.zeros(self.model.dim // self.model.num_heads)
                        for _ in range(self.model.num_heads)]

            # Get combined attention output and split by heads
            # (This is approximate since heads are already combined)
            attn_out = self.layer_activations[attn_key]
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]  # Handle tuple output

            # Extract for specific sample
            if attn_out.dim() > 1 and sample_idx < attn_out.size(1):
                attn_out = attn_out[:, sample_idx]

            # Split by heads
            head_dim = self.model.dim // self.model.num_heads
            head_activations = []

            for head_idx in range(self.model.num_heads):
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim

                if start_idx < attn_out.size(0):
                    head_act = attn_out[start_idx:end_idx]
                    head_activations.append(head_act)
                else:
                    # Handle case where dimensions don't match
                    head_activations.append(torch.zeros(head_dim, device=attn_out.device))

            return head_activations

        # If we have attention weights, we can do a more accurate extraction
        # Get attention weights for this sample - shape [num_heads, seq_len, seq_len]
        attn_weights = self.model.layers[layer_idx].attention_weights[sample_idx]

        # Get normalized input to attention (x_norm in your Block.forward)
        # We need to register an additional hook to capture this
        # For now, let's assume we have it in layer_activations
        norm_key = f"layer_{layer_idx}_norm_1"
        if norm_key not in self.layer_activations:
            # Fall back to approximate method
            return self._approximate_head_activations(layer_idx, sample_idx)

        x_norm = self.layer_activations[norm_key]
        if isinstance(x_norm, tuple):
            x_norm = x_norm[0]

        # Compute value projections (this assumes in_proj_weight structure from PyTorch)
        layer = self.model.layers[layer_idx]
        in_proj_weight = layer.attn.in_proj_weight
        in_proj_bias = layer.attn.in_proj_bias if hasattr(layer.attn, 'in_proj_bias') else None

        # Extract value projection weights
        dim = self.model.dim
        v_weight = in_proj_weight[2 * dim:]
        v_bias = in_proj_bias[2 * dim:] if in_proj_bias is not None else None

        # Compute value projections
        values = F.linear(x_norm, v_weight, v_bias)

        # Reshape for multi-head attention
        seq_len = x_norm.size(0)
        batch_size = x_norm.size(1) if x_norm.dim() > 1 else 1
        head_dim = dim // self.model.num_heads

        values = values.view(seq_len, batch_size, self.model.num_heads, head_dim)
        values = values.permute(1, 2, 0, 3)  # [batch, heads, seq, dim]

        # Apply attention weights to values
        head_activations = []
        for head_idx in range(self.model.num_heads):
            # Get weights for this head
            head_weights = attn_weights[head_idx]  # [seq_len, seq_len]

            # Get values for this head
            head_values = values[0, head_idx]  # [seq_len, head_dim]

            # Apply attention weights to values
            head_output = torch.matmul(head_weights, head_values)  # [seq_len, head_dim]

            # Take the last token's representation (for classification tasks)
            head_activations.append(head_output[-1])

        return head_activations

    def _extract_head_activations(self, layer_idx, sample_idx):
        """Memory-optimized version that remains compatible with enhanced Block"""
        # First check if the Block has already computed and stored head outputs
        if (hasattr(self.model.layers[layer_idx], 'store_head_outputs') and
                self.model.layers[layer_idx].store_head_outputs and
                hasattr(self.model.layers[layer_idx], 'head_outputs') and
                self.model.layers[layer_idx].head_outputs is not None):

            # Most efficient path: directly use pre-computed head outputs
            try:
                head_outputs = self.model.layers[layer_idx].head_outputs
                # Get the last position outputs for each head (for classification)
                head_activations = [
                    head_outputs[0, h_idx, -1].cpu() for h_idx in range(self.model.num_heads)
                ]
                return head_activations
            except (IndexError, AttributeError) as e:
                print(f"Error accessing pre-computed head outputs: {e}")
                # Continue to fallback methods

        # Second attempt: use attention weights if available
        if (hasattr(self.model.layers[layer_idx], 'attention_weights') and
                self.model.layers[layer_idx].attention_weights is not None):

            # Get attention weights for this sample
            try:
                attn_weights = self.model.layers[layer_idx].attention_weights[sample_idx]

                # Get normalized input (needed for value projection)
                norm_key = f"layer_{layer_idx}_norm_1"
                if norm_key in self.layer_activations:
                    x_norm = self.layer_activations[norm_key]
                    if isinstance(x_norm, tuple):
                        x_norm = x_norm[0]

                    # Memory optimization: move to CPU early if on GPU
                    if x_norm.device.type == 'cuda':
                        x_norm = x_norm.cpu()

                    # Compute value projections with memory optimization
                    layer = self.model.layers[layer_idx]
                    dim = self.model.dim
                    head_dim = dim // self.model.num_heads

                    # Extract value projection weights (reuse existing code but with memory optimization)
                    in_proj_weight = layer.attn.in_proj_weight.cpu()  # Move to CPU
                    in_proj_bias = layer.attn.in_proj_bias.cpu() if hasattr(layer.attn, 'in_proj_bias') else None

                    # Extract value projection weights
                    v_weight = in_proj_weight[2 * dim:].cpu()
                    v_bias = in_proj_bias[2 * dim:].cpu() if in_proj_bias is not None else None

                    # Compute value projections with reduced precision if possible
                    with torch.no_grad():
                        values = F.linear(x_norm, v_weight, v_bias)

                    # Process in smaller chunks to reduce memory
                    seq_len = x_norm.size(0)
                    batch_size = x_norm.size(1) if x_norm.dim() > 1 else 1

                    values = values.view(seq_len, batch_size, self.model.num_heads, head_dim)
                    values = values.permute(1, 2, 0, 3)  # [batch, heads, seq, dim]

                    # Process one head at a time to save memory
                    head_activations = []
                    for head_idx in range(self.model.num_heads):
                        # Get weights and values for this head
                        head_weights = attn_weights[head_idx].cpu()  # [seq_len, seq_len]
                        head_values = values[0, head_idx]  # [seq_len, head_dim]

                        # Apply attention weights to values
                        head_output = torch.matmul(head_weights, head_values)  # [seq_len, head_dim]

                        # Take the last token's representation
                        head_activations.append(head_output[-1])

                        # Force cleanup
                        del head_weights, head_values

                    return head_activations

            except Exception as e:
                print(f"Error in attention-based head extraction: {e}")
                # Continue to fallback method

        # Fallback method: Split the combined attention output
        # This is the memory-optimized version of the fallback
        attn_key = f"layer_{layer_idx}_attn_out"
        if attn_key not in self.layer_activations:
            # Return minimal empty tensors
            head_dim = self.model.dim // self.model.num_heads
            return [torch.zeros(head_dim, device='cpu') for _ in range(self.model.num_heads)]

        attn_out = self.layer_activations[attn_key]
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]

        # Memory optimization: move to CPU
        if attn_out.device.type == 'cuda':
            attn_out = attn_out.cpu()

        # Extract for specific sample
        if attn_out.dim() > 1 and sample_idx < attn_out.size(1):
            attn_out = attn_out[:, sample_idx]

        # Split by heads with memory optimization
        head_dim = self.model.dim // self.model.num_heads
        head_activations = []

        for head_idx in range(self.model.num_heads):
            start_idx = head_idx * head_dim
            end_idx = (head_idx + 1) * head_dim

            if start_idx < attn_out.size(0):
                # Use narrow operation to avoid allocation
                head_act = attn_out.narrow(0, start_idx, min(head_dim, attn_out.size(0) - start_idx))
                head_activations.append(head_act)
            else:
                head_activations.append(torch.zeros(head_dim, device='cpu'))

        return head_activations

    def _approximate_head_activations(self, layer_idx, sample_idx):
        """Fallback method for when we don't have all the necessary data"""
        attn_key = f"layer_{layer_idx}_attn_out"
        if attn_key not in self.layer_activations:
            return [torch.zeros(self.model.dim // self.model.num_heads)
                    for _ in range(self.model.num_heads)]

        attn_out = self.layer_activations[attn_key]
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]

        if attn_out.dim() > 1 and sample_idx < attn_out.size(1):
            attn_out = attn_out[:, sample_idx]

        head_dim = self.model.dim // self.model.num_heads
        head_activations = []

        for head_idx in range(self.model.num_heads):
            start_idx = head_idx * head_dim
            end_idx = (head_idx + 1) * head_dim

            if start_idx < attn_out.size(0):
                head_act = attn_out[start_idx:end_idx]
                head_activations.append(head_act)
            else:
                head_activations.append(torch.zeros(head_dim, device=attn_out.device))

        return head_activations

    def _analyze_information_flow(self, flow_data):
        """Analyze statistics of information flow between components"""
        flow_stats = {}

        for component, entries in flow_data.items():
            correlations = [entry['correlation'] for entry in entries]

            # Skip if no data
            if not correlations:
                continue

            avg_correlation = np.mean(correlations) if correlations else 0.0
            std_correlation = np.std(correlations)

            # Group by class
            class_correlations = defaultdict(list)
            for entry in entries:
                class_correlations[entry['class_id']].append(entry['correlation'])

            # Calculate class-specific statistics
            class_stats = {}
            for class_id, class_corrs in class_correlations.items():
                class_avg = np.mean(class_corrs)
                class_stats[class_id] = {
                    'avg_correlation': class_avg,
                    'sample_count': len(class_corrs)
                }

            # Determine if this component has significant flow
            is_significant = avg_correlation > 0.3  # Threshold can be adjusted

            flow_stats[component] = {
                'avg_correlation': avg_correlation,
                'std_correlation': std_correlation,
                'class_stats': class_stats,
                'is_significant': is_significant
            }

        return flow_stats

    def _build_interaction_graph(self, flow_stats):
        """Build a graph representing interactions between components"""
        G = nx.DiGraph()

        # Add nodes for attention heads
        for component, stats in flow_stats.items():
            if 'head' in component and stats['is_significant']:
                G.add_node(component,
                           type='attention_head',
                           correlation=stats['avg_correlation'])

        # Add nodes for MLP components
        for layer_idx in range(len(self.model.layers)):
            mlp_node = f"layer_{layer_idx}_mlp"
            G.add_node(mlp_node, type='mlp')

            # Add edges from significant attention heads to MLP
            for component, stats in flow_stats.items():
                if f"layer_{layer_idx}_head" in component and stats['is_significant']:
                    G.add_edge(component, mlp_node,
                               weight=stats['avg_correlation'],
                               type='attn_to_mlp')

        # Add cross-layer edges (attention head to next layer head)
        for layer_idx in range(len(self.model.layers) - 1):
            for head_idx in range(self.model.num_heads):
                src_node = f"layer_{layer_idx}_head_{head_idx}"

                # Find correlated heads in next layer
                for next_head_idx in range(self.model.num_heads):
                    dst_node = f"layer_{layer_idx + 1}_head_{next_head_idx}"

                    if src_node in G and dst_node in G:
                        # Simplified: add edge with moderate weight
                        # In a full implementation, we'd compute correlation between these heads
                        G.add_edge(src_node, dst_node,
                                   weight=0.2,  # Placeholder weight
                                   type='head_to_head')

        return G

    def _visualize_interactions(self, epoch, correlations, flow_stats, interaction_graph):
        """Create visualizations of component interactions"""
        # Create directory for visualizations
        viz_dir = self.save_dir / f"epoch_{epoch}"
        viz_dir.mkdir(exist_ok=True, parents=True)

        # 1. Layer-wise correlation plot
        fig, ax = plt.subplots(figsize=(10, 6))

        layers = list(correlations.keys())
        correlation_values = [correlations[layer] for layer in layers]

        ax.bar(range(len(layers)), correlation_values)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Attention-MLP Correlation')
        ax.set_title(f'Attention-MLP Correlation by Layer at Epoch {epoch}')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([layer.replace('layer_', '') for layer in layers])

        plt.tight_layout()
        plt.savefig(viz_dir / "layer_correlations.png")
        plt.close(fig)

        # 2. Head-MLP flow heatmap
        head_correlations = {comp: stats['avg_correlation']
                             for comp, stats in flow_stats.items()
                             if 'head' in comp}

        if head_correlations:
            # Organize by layer and head
            max_layer = max([int(comp.split('_')[1]) for comp in head_correlations.keys()])
            max_head = max([int(comp.split('_')[3]) for comp in head_correlations.keys()])

            correlation_matrix = np.zeros((max_layer + 1, max_head + 1))

            for comp, corr in head_correlations.items():
                layer_idx = int(comp.split('_')[1])
                head_idx = int(comp.split('_')[3])
                correlation_matrix[layer_idx, head_idx] = corr

            fig, ax = plt.subplots(figsize=(12, 8))

            sns.heatmap(correlation_matrix, cmap='viridis',
                        xticklabels=[f'Head {i}' for i in range(max_head + 1)],
                        yticklabels=[f'Layer {i}' for i in range(max_layer + 1)],
                        ax=ax)

            ax.set_title(f'Head-MLP Information Flow at Epoch {epoch}')
            ax.set_xlabel('Attention Head')
            ax.set_ylabel('Layer')

            plt.tight_layout()
            plt.savefig(viz_dir / "head_mlp_flow.png")
            plt.close(fig)

        # 3. Interaction network visualization
        if interaction_graph and len(interaction_graph.nodes()) > 0:
            fig, ax = plt.subplots(figsize=(12, 10))

            # Define node colors by type
            node_colors = []
            for node in interaction_graph.nodes():
                if interaction_graph.nodes[node]['type'] == 'attention_head':
                    node_colors.append('skyblue')
                else:  # MLP
                    node_colors.append('salmon')

            # Define node sizes based on importance
            node_sizes = []
            for node in interaction_graph.nodes():
                if 'correlation' in interaction_graph.nodes[node]:
                    # Size proportional to correlation
                    correlation = interaction_graph.nodes[node]['correlation']
                    node_sizes.append(300 * correlation + 100)
                else:
                    node_sizes.append(200)  # Default size

            # Define edge widths based on weight
            edge_widths = [
                interaction_graph[u][v]['weight'] * 3 + 1
                for u, v in interaction_graph.edges()
            ]

            # Create layout
            pos = nx.spring_layout(interaction_graph, seed=42)

            # Draw the graph
            nx.draw_networkx_nodes(interaction_graph, pos,
                                   node_color=node_colors,
                                   node_size=node_sizes,
                                   alpha=0.8, ax=ax)

            nx.draw_networkx_edges(interaction_graph, pos,
                                   width=edge_widths,
                                   alpha=0.6, edge_color='gray',
                                   arrows=True, ax=ax)

            nx.draw_networkx_labels(interaction_graph, pos, font_size=8, ax=ax)

            ax.set_title(f'Component Interaction Network at Epoch {epoch}')
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(viz_dir / "interaction_network.png")
            plt.close(fig)

            # Save network in GraphML format for further analysis
            nx.write_graphml(interaction_graph, viz_dir / "interaction_network.graphml")

    def identify_cross_component_circuits(self, threshold=0.4):
        """Identify circuits spanning attention and MLP components"""
        if not self.interaction_history:
            print("No interaction history available. Run analyze_interactions first.")
            return None

        # Use the most recent epoch's interaction graph
        latest_epoch = max(self.interaction_history.keys())
        graph = self.interaction_history[latest_epoch]['interaction_graph']

        # Identify strong connections (potential circuit components)
        strong_edges = [(u, v) for u, v in graph.edges()
                        if graph[u][v]['weight'] > threshold]

        # Create subgraph of strong connections
        strong_subgraph = graph.edge_subgraph(strong_edges).copy()

        # Find connected components (potential circuits)
        connected_components = list(nx.connected_components(strong_subgraph.to_undirected()))

        # Extract circuits that span attention and MLP components
        cross_component_circuits = []

        for component in connected_components:
            component_types = set(strong_subgraph.nodes[node]['type'] for node in component)

            # Check if circuit spans attention and MLP
            if 'attention_head' in component_types and 'mlp' in component_types:
                # Calculate circuit strength (average edge weight)
                edges = [(u, v) for u, v in strong_edges
                         if u in component and v in component]

                avg_strength = np.mean([graph[u][v]['weight'] for u, v in edges])

                cross_component_circuits.append({
                    'components': list(component),
                    'strength': avg_strength,
                    'edge_count': len(edges)
                })

        # Sort circuits by strength
        cross_component_circuits.sort(key=lambda x: x['strength'], reverse=True)

        return cross_component_circuits

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

        for hook in self.hooks:
            hook.remove()
        self.hooks = []


