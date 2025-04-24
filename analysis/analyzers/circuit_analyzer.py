# analysis/analyzers/circuit_analyzer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from pathlib import Path

from analysis.analyzers.base_analyzer import BaseAnalyzer


class CircuitAnalyzer(BaseAnalyzer):
    """
    Analyzer for discovering and characterizing computational circuits in transformers

    This class implements methods to identify circuits by analyzing the impact
    of ablating individual attention heads and head combinations, as well as
    analyzing connections between heads and MLPs.
    """

    def _get_analysis_dir_name(self):
        return "circuit_analysis"

    def identify_circuits(self, eval_loader, baseline_acc=None, threshold=0.01):
        """
        Identify potential circuits through head ablation

        Args:
            eval_loader: Evaluation data loader
            baseline_acc: Optional baseline accuracy (will be computed if None)
            threshold: Threshold for circuit identification (attribution score)

        Returns:
            dict: Circuit analysis results
        """
        self.model.eval()

        # Get baseline accuracy if not provided
        if baseline_acc is None:
            baseline_acc, _ = self.model.evaluate(eval_loader)

        # Identify all attention heads
        heads = []
        for layer_idx in range(self.model.num_layers):
            for head_idx in range(self.model.num_heads):
                heads.append((layer_idx, head_idx))

        # Individual ablation results
        individual_results = {}

        # Analyze each head individually
        for layer_idx, head_idx in heads:
            head_key = f'layer_{layer_idx}_head_{head_idx}'

            # Store original weights
            original_weights = self.model.layers[layer_idx].attn.out_proj.weight.clone()

            # Mask this head
            head_dim = self.model.dim // self.model.num_heads
            start_idx = head_idx * head_dim
            end_idx = (head_idx + 1) * head_dim

            with torch.no_grad():
                self.model.layers[layer_idx].attn.out_proj.weight[:, start_idx:end_idx] = 0

            # Evaluate
            ablated_acc, _ = self.model.evaluate(eval_loader)
            individual_results[head_key] = baseline_acc - ablated_acc

            # Restore weights
            with torch.no_grad():
                self.model.layers[layer_idx].attn.out_proj.weight.copy_(original_weights)

        # Filter for significant attribution scores
        significant_heads = {k: v for k, v in individual_results.items() if v >= threshold}

        # Sort by attribution
        sorted_heads = sorted(significant_heads.items(), key=lambda x: x[1], reverse=True)

        # If we have enough significant heads, analyze pairwise interactions
        pairwise_results = {}
        if len(significant_heads) >= 2:
            # Analyze all significant head pairs
            head_keys = list(significant_heads.keys())

            for i, head1 in enumerate(head_keys):
                for j in range(i + 1, len(head_keys)):
                    head2 = head_keys[j]

                    # Parse head keys
                    layer_idx1 = int(head1.split('_')[1])
                    head_idx1 = int(head1.split('_')[3])
                    layer_idx2 = int(head2.split('_')[1])
                    head_idx2 = int(head2.split('_')[3])

                    # Store original weights
                    original_weights1 = self.model.layers[layer_idx1].attn.out_proj.weight.clone()
                    original_weights2 = self.model.layers[layer_idx2].attn.out_proj.weight.clone()

                    # Mask both heads
                    head_dim = self.model.dim // self.model.num_heads
                    start_idx1 = head_idx1 * head_dim
                    end_idx1 = (head_idx1 + 1) * head_dim
                    start_idx2 = head_idx2 * head_dim
                    end_idx2 = (head_idx2 + 1) * head_dim

                    with torch.no_grad():
                        self.model.layers[layer_idx1].attn.out_proj.weight[:, start_idx1:end_idx1] = 0
                        self.model.layers[layer_idx2].attn.out_proj.weight[:, start_idx2:end_idx2] = 0

                    # Evaluate
                    ablated_acc, _ = self.model.evaluate(eval_loader)
                    actual_drop = baseline_acc - ablated_acc
                    expected_drop = significant_heads[head1] + significant_heads[head2]

                    # Calculate circuit strength (super-additivity)
                    circuit_strength = actual_drop - expected_drop

                    # Store results
                    pair_key = f"{head1}+{head2}"
                    pairwise_results[pair_key] = {
                        'expected_drop': expected_drop,
                        'actual_drop': actual_drop,
                        'circuit_strength': circuit_strength,
                        'is_circuit': circuit_strength > 0.01  # Positive interaction indicates circuit
                    }

                    # Restore weights
                    with torch.no_grad():
                        self.model.layers[layer_idx1].attn.out_proj.weight.copy_(original_weights1)
                        self.model.layers[layer_idx2].attn.out_proj.weight.copy_(original_weights2)

        # Identify circuits (pairs with positive interaction)
        circuits = {k: v for k, v in pairwise_results.items() if v['is_circuit']}

        # Sort circuits by strength
        sorted_circuits = sorted(circuits.items(), key=lambda x: x[1]['circuit_strength'], reverse=True)

        # Prepare results
        results = {
            'individual_attribution': individual_results,
            'significant_heads': significant_heads,
            'sorted_heads': sorted_heads,
            'pairwise_interactions': pairwise_results,
            'circuits': circuits,
            'sorted_circuits': sorted_circuits
        }

        # Generate visualizations
        self._visualize_circuits(results)

        # Log key metrics
        if self.logger:
            # Log top head attributions
            for head, score in sorted_heads[:5]:  # Top 5 heads
                self.logger.log_data('circuit_analysis', f'top_head_{head}', score)

            # Log top circuits
            for i, (pair, data) in enumerate(sorted_circuits[:3]):  # Top 3 circuits
                self.logger.log_data('circuit_analysis', f'circuit_{i}_pair', pair)
                self.logger.log_data('circuit_analysis', f'circuit_{i}_strength', data['circuit_strength'])

        return results

    def _visualize_circuits(self, results):
        """
        Create visualizations of identified circuits

        Args:
            results: Results from identify_circuits
        """
        if not results.get('significant_heads'):
            return

        # 1. Head attribution bar chart
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        heads = []
        scores = []
        for head, score in results['sorted_heads']:
            heads.append(head)
            scores.append(score)

        # Create bar chart
        ax1.bar(heads, scores)
        ax1.axhline(y=0.01, color='r', linestyle='--', label='Threshold')
        ax1.set_xlabel('Head')
        ax1.set_ylabel('Attribution (Accuracy Decrease When Ablated)')
        ax1.set_title('Head Attribution Analysis')
        ax1.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / "head_attribution.png")
        plt.close(fig1)

        # 2. Circuit interaction network
        if results.get('circuits'):
            plt.figure(figsize=(10, 8))

            # Create graph
            G = nx.Graph()

            # Add nodes for significant heads
            for head, score in results['significant_heads'].items():
                G.add_node(head, weight=score)

            # Add edges for circuit interactions
            for pair, data in results['circuits'].items():
                head1, head2 = pair.split('+')
                G.add_edge(head1, head2, weight=data['circuit_strength'])

            # Set node sizes based on attribution
            node_sizes = [G.nodes[node]['weight'] * 5000 for node in G.nodes]

            # Set edge widths based on interaction strength
            edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges]

            # Position nodes using spring layout
            pos = nx.spring_layout(G, seed=42)

            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8,
                                   node_color='lightblue')
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5,
                                   edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_size=10)

            plt.title('Circuit Interaction Network')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "circuit_network.png")
            plt.close()

        # 3. Heatmap of pairwise interactions
        if results.get('pairwise_interactions'):
            # Create a matrix representation of interactions
            heads_list = list(results['significant_heads'].keys())
            n_heads = len(heads_list)

            if n_heads > 1:
                interaction_matrix = np.zeros((n_heads, n_heads))

                # Fill the matrix
                for pair, data in results['pairwise_interactions'].items():
                    h1, h2 = pair.split('+')
                    if h1 in heads_list and h2 in heads_list:
                        i = heads_list.index(h1)
                        j = heads_list.index(h2)
                        interaction_matrix[i, j] = data['circuit_strength']
                        interaction_matrix[j, i] = data['circuit_strength']  # Symmetric

                # Create heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(interaction_matrix, annot=True, fmt=".3f",
                            xticklabels=heads_list, yticklabels=heads_list,
                            cmap='coolwarm', center=0)

                plt.title('Pairwise Head Interactions')
                plt.tight_layout()
                plt.savefig(self.analysis_dir / "interaction_heatmap.png")
                plt.close()

    def analyze_mlp_head_interactions(self, eval_loader, threshold=0.01):
        """
        Analyze interactions between attention heads and MLP layers

        Args:
            eval_loader: Evaluation data loader
            threshold: Threshold for significance

        Returns:
            dict: Analysis results
        """
        self.model.eval()

        # Get baseline accuracy
        baseline_acc, _ = self.model.evaluate(eval_loader)

        # Identify all attention heads and MLP layers
        heads = []
        mlps = []

        for layer_idx in range(self.model.num_layers):
            # Add heads from this layer
            for head_idx in range(self.model.num_heads):
                heads.append((layer_idx, head_idx))

            # Add MLP from this layer
            mlps.append(layer_idx)

        # Individual ablation results
        head_results = {}
        mlp_results = {}

        # Analyze each head individually
        for layer_idx, head_idx in heads:
            head_key = f'layer_{layer_idx}_head_{head_idx}'

            # Store original weights
            original_weights = self.model.layers[layer_idx].attn.out_proj.weight.clone()

            # Mask this head
            head_dim = self.model.dim // self.model.num_heads
            start_idx = head_idx * head_dim
            end_idx = (head_idx + 1) * head_dim

            with torch.no_grad():
                self.model.layers[layer_idx].attn.out_proj.weight[:, start_idx:end_idx] = 0

            # Evaluate
            ablated_acc, _ = self.model.evaluate(eval_loader)
            head_results[head_key] = baseline_acc - ablated_acc

            # Restore weights
            with torch.no_grad():
                self.model.layers[layer_idx].attn.out_proj.weight.copy_(original_weights)

        # Analyze each MLP individually
        for layer_idx in mlps:
            mlp_key = f'layer_{layer_idx}_mlp'

            # Store original weights
            original_up_weights = self.model.layers[layer_idx].mlp[0].weight.clone()
            original_down_weights = self.model.layers[layer_idx].mlp[2].weight.clone()

            # Mask MLP
            with torch.no_grad():
                self.model.layers[layer_idx].mlp[0].weight.fill_(0)
                self.model.layers[layer_idx].mlp[2].weight.fill_(0)

            # Evaluate
            ablated_acc, _ = self.model.evaluate(eval_loader)
            mlp_results[mlp_key] = baseline_acc - ablated_acc

            # Restore weights
            with torch.no_grad():
                self.model.layers[layer_idx].mlp[0].weight.copy_(original_up_weights)
                self.model.layers[layer_idx].mlp[2].weight.copy_(original_down_weights)

        # Filter for significant components
        significant_heads = {k: v for k, v in head_results.items() if v >= threshold}
        significant_mlps = {k: v for k, v in mlp_results.items() if v >= threshold}

        # Analyze head-MLP interactions
        head_mlp_interactions = {}

        for head_key, head_score in significant_heads.items():
            layer_idx_h = int(head_key.split('_')[1])
            head_idx = int(head_key.split('_')[3])

            for mlp_key, mlp_score in significant_mlps.items():
                layer_idx_m = int(mlp_key.split('_')[1])

                # Only consider heads and MLPs from the same or adjacent layers
                if abs(layer_idx_h - layer_idx_m) <= 1:
                    # Store original weights
                    original_head_weights = self.model.layers[layer_idx_h].attn.out_proj.weight.clone()
                    original_up_weights = self.model.layers[layer_idx_m].mlp[0].weight.clone()
                    original_down_weights = self.model.layers[layer_idx_m].mlp[2].weight.clone()

                    # Mask both components
                    head_dim = self.model.dim // self.model.num_heads
                    start_idx = head_idx * head_dim
                    end_idx = (head_idx + 1) * head_dim

                    with torch.no_grad():
                        self.model.layers[layer_idx_h].attn.out_proj.weight[:, start_idx:end_idx] = 0
                        self.model.layers[layer_idx_m].mlp[0].weight.fill_(0)
                        self.model.layers[layer_idx_m].mlp[2].weight.fill_(0)

                    # Evaluate
                    ablated_acc, _ = self.model.evaluate(eval_loader)
                    actual_drop = baseline_acc - ablated_acc
                    expected_drop = head_score + mlp_score

                    # Calculate interaction strength
                    interaction_strength = actual_drop - expected_drop

                    # Store results
                    interaction_key = f"{head_key}+{mlp_key}"
                    head_mlp_interactions[interaction_key] = {
                        'expected_drop': expected_drop,
                        'actual_drop': actual_drop,
                        'interaction_strength': interaction_strength,
                        'is_interaction': interaction_strength > 0.01
                    }

                    # Restore weights
                    with torch.no_grad():
                        self.model.layers[layer_idx_h].attn.out_proj.weight.copy_(original_head_weights)
                        self.model.layers[layer_idx_m].mlp[0].weight.copy_(original_up_weights)
                        self.model.layers[layer_idx_m].mlp[2].weight.copy_(original_down_weights)

        # Identify strong interactions
        strong_interactions = {k: v for k, v in head_mlp_interactions.items() if v['is_interaction']}

        # Sort by interaction strength
        sorted_interactions = sorted(strong_interactions.items(),
                                     key=lambda x: x[1]['interaction_strength'],
                                     reverse=True)

        # Prepare results
        results = {
            'head_attribution': head_results,
            'mlp_attribution': mlp_results,
            'significant_heads': significant_heads,
            'significant_mlps': significant_mlps,
            'head_mlp_interactions': head_mlp_interactions,
            'strong_interactions': strong_interactions,
            'sorted_interactions': sorted_interactions
        }

        # Generate visualizations
        self._visualize_head_mlp_interactions(results)

        # Log key metrics
        if self.logger:
            # Log top interactions
            for i, (pair, data) in enumerate(sorted_interactions[:3]):  # Top 3 interactions
                self.logger.log_data('circuit_analysis', f'head_mlp_interaction_{i}_pair', pair)
                self.logger.log_data('circuit_analysis', f'head_mlp_interaction_{i}_strength',
                                     data['interaction_strength'])

        return results

    def _visualize_head_mlp_interactions(self, results):
        """
        Create visualizations of head-MLP interactions

        Args:
            results: Results from analyze_mlp_head_interactions
        """
        if not results.get('strong_interactions'):
            return

        # 1. Component attribution bar chart
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Head attribution
        heads = []
        head_scores = []
        for head, score in sorted(results['head_attribution'].items(),
                                  key=lambda x: x[1], reverse=True)[:10]:  # Top 10 heads
            heads.append(head)
            head_scores.append(score)

        ax1.bar(heads, head_scores)
        ax1.axhline(y=0.01, color='r', linestyle='--', label='Threshold')
        ax1.set_xlabel('Head')
        ax1.set_ylabel('Attribution')
        ax1.set_title('Head Attribution')
        ax1.tick_params(axis='x', rotation=45)

        # MLP attribution
        mlps = []
        mlp_scores = []
        for mlp, score in sorted(results['mlp_attribution'].items(),
                                 key=lambda x: x[1], reverse=True):
            mlps.append(mlp)
            mlp_scores.append(score)

        ax2.bar(mlps, mlp_scores)
        ax2.axhline(y=0.01, color='r', linestyle='--', label='Threshold')
        ax2.set_xlabel('MLP')
        ax2.set_ylabel('Attribution')
        ax2.set_title('MLP Attribution')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / "component_attribution.png")
        plt.close(fig1)

        # 2. Interaction bar chart
        fig2, ax3 = plt.subplots(figsize=(12, 6))

        pairs = []
        interaction_scores = []
        for pair, data in results['sorted_interactions'][:10]:  # Top 10 interactions
            pairs.append(pair)
            interaction_scores.append(data['interaction_strength'])

        ax3.bar(pairs, interaction_scores)
        ax3.axhline(y=0.01, color='r', linestyle='--', label='Threshold')
        ax3.set_xlabel('Head-MLP Pair')
        ax3.set_ylabel('Interaction Strength')
        ax3.set_title('Head-MLP Interactions')
        ax3.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.analysis_dir / "head_mlp_interactions.png")
        plt.close(fig2)

        # 3. Interaction network
        plt.figure(figsize=(12, 8))

        # Create graph
        G = nx.Graph()

        # Add nodes for heads and MLPs
        for head, score in results['significant_heads'].items():
            G.add_node(head, weight=score, type='head')

        for mlp, score in results['significant_mlps'].items():
            G.add_node(mlp, weight=score, type='mlp')

        # Add edges for interactions
        for pair, data in results['strong_interactions'].items():
            head, mlp = pair.split('+')
            G.add_edge(head, mlp, weight=data['interaction_strength'])

        # Set node sizes and colors based on type and attribution
        node_sizes = []
        node_colors = []
        for node in G.nodes:
            if 'head' in node and '_head_' in node:
                node_sizes.append(G.nodes[node]['weight'] * 3000)
                node_colors.append('lightblue')
            else:  # MLP
                node_sizes.append(G.nodes[node]['weight'] * 3000)
                node_colors.append('lightgreen')

        # Set edge widths based on interaction strength
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges]

        # Position nodes using spring layout
        pos = nx.spring_layout(G, seed=42)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8,
                               node_color=node_colors)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5,
                               edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title('Head-MLP Interaction Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.analysis_dir / "head_mlp_network.png")
        plt.close()
