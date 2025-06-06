# circuit_evolution_tracker.py
from typing import Dict, List, Set, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from analysis.core.circuit_schema import Circuit, CircuitType
from analysis.utils.utils import get_current_callable_info, shorten_layer_head


class CircuitEvolutionTracker:
    """Tracks the evolution of circuits over training epochs"""

    def __init__(self, registry, save_dir=None, logger=None):
        """
        Initialize the circuit evolution tracker

        Args:
            registry: The circuit registry
            save_dir: Optional directory to save analysis results
            logger: Optional logger for metrics
        """
        self.registry = registry
        self.logger = logger

        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        # Storage for evolution data
        self.evolution_data = {}  # Circuit ID -> temporal data
        self.emergence_epochs = {}  # Circuit ID -> emergence epoch
        self.circuit_relationships = {}  # (Circuit ID, Circuit ID) -> relationship data
        self.epoch_to_circuits = {}  # Epoch -> circuit IDs that were active

    def save_figure_safe(self, fig,  filename, save_dir=None, **kwargs):
        """
        Save figure ensuring directory exists with sensible defaults.

        Args:
            fig: matplotlib figure
            save_dir: base directory path
            filename: relative filename (can include subdirs)
            **kwargs: arguments passed to savefig (overrides defaults)
        """
        if save_dir is None:
            save_dir = self.save_dir
        save_path = Path(save_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Set good defaults for ML plots
        default_kwargs = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none',
            'format': None  # Auto-detect from extension
        }

        # Override defaults with user-provided kwargs
        default_kwargs.update(kwargs)

        fig.savefig(save_path, **default_kwargs)
        return save_path

    def update_circuit_evolution(self, epoch, circuits, token_attribution):
        """
        Track circuit evolution for a specific epoch

        Args:
            epoch: Current training epoch
            circuits: List of circuits discovered in this epoch
            token_attribution: Token attribution matrix

        Returns:
            Dict with updated evolution metrics
        """
        # Track which circuits are active in this epoch
        active_circuit_ids = set()
        # For each circuit, record its state at this epoch
        for circuit in circuits:
            active_circuit_ids.add(circuit.id)

            if circuit.id not in self.evolution_data:
                self.evolution_data[circuit.id] = []

            # Store current circuit state
            self.evolution_data[circuit.id].append({
                'epoch': epoch,
                'attribution': circuit.attribution,
                'elements': len(circuit.elements),
                'connections': len(circuit.connections),
                'strength': self._calculate_circuit_strength(circuit, token_attribution)
            })

            # Check if this is the emergence epoch
            if circuit.id not in self.emergence_epochs:
                # Define emergence as first epoch where attribution exceeds threshold
                if circuit.attribution > 0.3:  # Threshold can be adjusted
                    self.emergence_epochs[circuit.id] = epoch

                    # Log emergence if logger is available
                    if self.logger:
                        circuit_type = circuit.type.value
                        self.logger.log_data(
                            'circuit_evolution',
                            f'emergence_{circuit_type}_{circuit.id}',
                            epoch
                        )

        # Store circuits active in this epoch
        self.epoch_to_circuits[epoch] = active_circuit_ids

        # Update circuit relationships based on co-occurrence
        self._update_circuit_relationships(epoch, active_circuit_ids)

        if len(active_circuit_ids) > 0:
            # print(f"\t{get_current_callable_info()}: \t{len(active_circuit_ids)} detected token circuits")
            pass

        return {
            'active_circuits': len(active_circuit_ids),
            'emergence_events': [cid for cid, e in self.emergence_epochs.items() if e == epoch],
            'evolution_data': self.evolution_data
        }

    def _calculate_circuit_strength(self, circuit, token_attribution):
        """Calculate overall circuit strength based on token attribution"""
        # A more sophisticated calculation would consider the specific tokens in the circuit
        # For now, use a simplified approach based on connection strengths

        if not circuit.connections:
            return circuit.attribution

        # Calculate average connection strength
        avg_strength = sum(conn.strength for conn in circuit.connections) / len(circuit.connections)

        # Combine with overall attribution
        return (circuit.attribution + avg_strength) / 2

    def _update_circuit_relationships(self, epoch, active_circuit_ids):
        """
        Update relationships between circuits based on co-activation

        Args:
            epoch: Current epoch
            active_circuit_ids: Set of circuit IDs active in this epoch
        """
        # Update co-occurrence relationships
        for cid1 in active_circuit_ids:
            for cid2 in active_circuit_ids:
                if cid1 != cid2:
                    rel_key = (min(cid1, cid2), max(cid1, cid2))

                    if rel_key not in self.circuit_relationships:
                        self.circuit_relationships[rel_key] = {
                            'co_occurrences': 0,
                            'first_co_occurrence': epoch,
                            'last_co_occurrence': epoch,
                            'epochs': []
                        }

                    rel_data = self.circuit_relationships[rel_key]
                    rel_data['co_occurrences'] += 1
                    rel_data['last_co_occurrence'] = epoch
                    rel_data['epochs'].append(epoch)

    def analyze_emergence_order(self):
        """
        Analyze which types of circuits emerge first

        Returns:
            Dict with emergence order analysis
        """
        # Group circuits by type
        circuits_by_type = {}
        for circuit_id, emergence_epoch in self.emergence_epochs.items():
            circuit = self.registry.get_circuit(circuit_id)
            if circuit and hasattr(circuit, 'type'):
                circuit_type = circuit.type.value
                if circuit_type not in circuits_by_type:
                    circuits_by_type[circuit_type] = []
                circuits_by_type[circuit_type].append((circuit_id, emergence_epoch))

        # Calculate average emergence epoch by type
        avg_emergence = {}
        median_emergence = {}
        earliest_emergence = {}
        count_by_type = {}

        for circuit_type, circuits in circuits_by_type.items():
            if circuits:
                epochs = [epoch for _, epoch in circuits]
                avg_emergence[circuit_type] = sum(epochs) / len(epochs)
                median_emergence[circuit_type] = np.median(epochs)
                earliest_emergence[circuit_type] = min(epochs)
                count_by_type[circuit_type] = len(circuits)

        # Determine emergence order
        ordered_types = sorted(
            avg_emergence.keys(),
            key=lambda t: avg_emergence[t]
        )
        # print(f"\t{get_current_callable_info()}:\t{get_current_callable_info()}: ")

        return {
            'circuits_by_type': circuits_by_type,
            'avg_emergence': avg_emergence,
            'median_emergence': median_emergence,
            'earliest_emergence': earliest_emergence,
            'count_by_type': count_by_type,
            'emergence_order': ordered_types
        }

    def analyze_circuit_relationships(self):
        """
        Analyze relationships between circuits, including precedence and co-occurrence

        Returns:
            Dict with relationship analysis
        """
        relationships = []

        # Compare emergence epochs between all circuit pairs
        circuit_ids = list(self.emergence_epochs.keys())
        for i, circuit_id1 in enumerate(circuit_ids):
            for circuit_id2 in circuit_ids[i + 1:]:
                # Get emergence epochs
                epoch1 = self.emergence_epochs.get(circuit_id1)
                epoch2 = self.emergence_epochs.get(circuit_id2)

                if epoch1 is not None and epoch2 is not None:
                    # Check for precedence relationship
                    if abs(epoch1 - epoch2) > 10:  # Significant time difference
                        precedes = circuit_id1 if epoch1 < epoch2 else circuit_id2
                        follows = circuit_id2 if epoch1 < epoch2 else circuit_id1

                        relationships.append({
                            'type': 'precedence',
                            'precedes': precedes,
                            'follows': follows,
                            'epoch_diff': abs(epoch1 - epoch2)
                        })

        # Analyze co-occurrence patterns
        co_occurrence = []

        for (cid1, cid2), rel_data in self.circuit_relationships.items():
            if rel_data['co_occurrences'] >= 3:  # Require multiple co-occurrences
                co_occurrence.append({
                    'circuit_pair': (cid1, cid2),
                    'co_occurrences': rel_data['co_occurrences'],
                    'first_co_occurrence': rel_data['first_co_occurrence'],
                    'last_co_occurrence': rel_data['last_co_occurrence'],
                    'strength': rel_data['co_occurrences'] /
                                (rel_data['last_co_occurrence'] - rel_data['first_co_occurrence'] + 1)
                })

        # Sort by co-occurrence strength
        co_occurrence.sort(key=lambda x: x['strength'], reverse=True)

        # print(f"\t{get_current_callable_info()}: \t{shorten_layer_head(co_occurrence)}")
        return {
            'precedence': relationships,
            'co_occurrence': co_occurrence
        }

    # Add to circuit_evolution_tracker.py
    def compute_circuit_similarity_matrix(self, epoch=None):
        """Compute similarity between all circuits"""
        # Get active circuits
        if epoch is not None:
            circuit_ids = list(self.epoch_to_circuits.get(epoch, set()))
        else:
            circuit_ids = list(self.evolution_data.keys())

        n_circuits = len(circuit_ids)
        similarity_matrix = np.zeros((n_circuits, n_circuits))

        # Calculate similarity between all circuit pairs
        for i in range(n_circuits):
            for j in range(i, n_circuits):
                cid1, cid2 = circuit_ids[i], circuit_ids[j]

                # Get circuits
                circuit1 = self.registry.get_circuit(cid1)
                circuit2 = self.registry.get_circuit(cid2)

                if circuit1 and circuit2:
                    # Component overlap score
                    component_overlap = self._calculate_component_overlap(circuit1, circuit2)

                    # Functional similarity score
                    functional_similarity = self._calculate_functional_similarity(circuit1, circuit2)

                    # Combined similarity (weighted average)
                    similarity = 0.6 * component_overlap + 0.4 * functional_similarity

                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Symmetric

        return similarity_matrix, circuit_ids

    def _calculate_component_overlap(self, circuit1, circuit2):
        """Calculate component overlap between two circuits"""
        # Extract element IDs from both circuits
        elements1 = set(e.id for e in circuit1.elements)
        elements2 = set(e.id for e in circuit2.elements)

        # Jaccard similarity = intersection / union
        intersection = len(elements1.intersection(elements2))
        union = len(elements1.union(elements2))

        return intersection / union if union > 0 else 0

    def _calculate_functional_similarity(self, circuit1, circuit2):
        """Calculate functional similarity based on metadata and connections"""
        # Compare operation types
        if 'operation_type' in circuit1.metadata and 'operation_type' in circuit2.metadata:
            if circuit1.metadata['operation_type'] == circuit2.metadata['operation_type']:
                return 0.8  # Same operation type is a strong indicator

        # Compare connection patterns
        if circuit1.connections and circuit2.connections:
            # Create connection signature (source_type -> target_type patterns)
            conn_sig1 = self._get_connection_signature(circuit1)
            conn_sig2 = self._get_connection_signature(circuit2)

            # Calculate signature similarity
            common_patterns = len(set(conn_sig1).intersection(set(conn_sig2)))
            total_patterns = len(set(conn_sig1).union(set(conn_sig2)))

            return common_patterns / total_patterns if total_patterns > 0 else 0

        return 0.0

    def _get_connection_signature(self, circuit):
        """Create a connection signature for functional comparison"""
        signature = []

        # Get elements by ID
        element_dict = {e.id: e for e in circuit.elements}

        # Create abstract connection patterns
        for conn in circuit.connections:
            src_elem = element_dict.get(conn.source)
            tgt_elem = element_dict.get(conn.target)

            if src_elem and tgt_elem:
                # Create abstract connection pattern
                signature.append(f"{src_elem.type.name}->{tgt_elem.type.name}:{conn.type.name}")

        return signature

    # Add to circuit_evolution_tracker.py
    def track_circuit_lineage(self, current_epoch, similarity_threshold=0.7):
        """
        Track circuit lineage - identify when circuits evolve from previous ones

        Args:
            current_epoch: Current training epoch
            similarity_threshold: Threshold for considering circuits related

        Returns:
            Dict with lineage analysis
        """
        if current_epoch <= min(self.epoch_to_circuits.keys()):
            return {}  # Not enough history

        # Get circuits from current and previous epochs
        current_circuits = list(self.epoch_to_circuits.get(current_epoch, set()))

        # Find the closest previous epoch with circuits
        prev_epochs = [e for e in sorted(self.epoch_to_circuits.keys()) if e < current_epoch]
        if not prev_epochs:
            return {}

        prev_epoch = max(prev_epochs)
        previous_circuits = list(self.epoch_to_circuits.get(prev_epoch, set()))

        # No previous circuits
        if not previous_circuits:
            return {'new_circuits': current_circuits, 'evolved_circuits': {}}

        # Compare current circuits to previous ones
        evolved_circuits = {}
        similarity_scores = {}

        for current_id in current_circuits:
            current_circuit = self.registry.get_circuit(current_id)
            if not current_circuit:
                continue

            # Find similar previous circuits
            best_match = None
            best_score = 0

            for prev_id in previous_circuits:
                prev_circuit = self.registry.get_circuit(prev_id)
                if not prev_circuit:
                    continue

                # Skip if different types
                if current_circuit.type != prev_circuit.type:
                    continue

                # Calculate similarity
                component_overlap = self._calculate_component_overlap(current_circuit, prev_circuit)
                functional_similarity = self._calculate_functional_similarity(current_circuit, prev_circuit)

                # Combined similarity (weighted average)
                similarity = 0.6 * component_overlap + 0.4 * functional_similarity

                similarity_scores[(prev_id, current_id)] = similarity

                if similarity > best_score:
                    best_score = similarity
                    best_match = prev_id

            # Record lineage if above threshold
            if best_match and best_score >= similarity_threshold:
                evolved_circuits[current_id] = {
                    'parent': best_match,
                    'similarity': best_score
                }

        # Identify circuit transformations
        transformations = self._identify_transformations(
            current_circuits, previous_circuits, similarity_scores)

        # Identify new circuits (not evolved from previous ones)
        new_circuits = [cid for cid in current_circuits if cid not in evolved_circuits]

        # Identify defunct circuits (disappeared from previous epoch)
        defunct_circuits = [
            cid for cid in previous_circuits
            if not any(evolved.get('parent') == cid for evolved in evolved_circuits.values())
        ]

        return {
            'evolved_circuits': evolved_circuits,
            'new_circuits': new_circuits,
            'defunct_circuits': defunct_circuits,
            'transformations': transformations,
            'previous_epoch': prev_epoch,
            'current_epoch': current_epoch
        }

    def _identify_transformations(self, current_circuits, previous_circuits, similarity_scores):
        """Identify circuit transformations like splitting and merging"""
        # Identify potential merges (multiple previous -> one current)
        merges = {}
        for curr_id in current_circuits:
            # Find all previous circuits with significant similarity
            similar_prevs = [
                prev_id for prev_id in previous_circuits
                if (prev_id, curr_id) in similarity_scores
                   and similarity_scores[(prev_id, curr_id)] >= 0.4
            ]

            if len(similar_prevs) >= 2:
                merges[curr_id] = {
                    'parents': similar_prevs,
                    'similarities': [similarity_scores[(p, curr_id)] for p in similar_prevs]
                }

        # Identify potential splits (one previous -> multiple current)
        splits = {}
        for prev_id in previous_circuits:
            # Find all current circuits with significant similarity
            similar_currs = [
                curr_id for curr_id in current_circuits
                if (prev_id, curr_id) in similarity_scores
                   and similarity_scores[(prev_id, curr_id)] >= 0.4
            ]

            if len(similar_currs) >= 2:
                splits[prev_id] = {
                    'children': similar_currs,
                    'similarities': [similarity_scores[(prev_id, c)] for c in similar_currs]
                }

        # Identify refinements (same function but improved implementation)
        refinements = {}
        for curr_id in current_circuits:
            for prev_id in previous_circuits:
                if (prev_id, curr_id) in similarity_scores and similarity_scores[(prev_id, curr_id)] >= 0.8:
                    # High similarity but not identical
                    curr_circuit = self.registry.get_circuit(curr_id)
                    prev_circuit = self.registry.get_circuit(prev_id)

                    if curr_circuit and prev_circuit:
                        # Consider it a refinement if attribution increased
                        if curr_circuit.attribution > prev_circuit.attribution * 1.2:
                            refinements[curr_id] = {
                                'parent': prev_id,
                                'similarity': similarity_scores[(prev_id, curr_id)],
                                'attribution_change': curr_circuit.attribution / prev_circuit.attribution
                            }

        return {
            'merges': merges,
            'splits': splits,
            'refinements': refinements
        }

    # Add to circuit_evolution_tracker.py
    def visualize_circuit_interactions(self, epoch, competition_data=None, cooperation_data=None, save_path=None):
        """
        Visualize circuit cooperation and competition

        Args:
            epoch: Current epoch
            competition_data: Optional competition analysis data
            cooperation_data: Optional cooperation analysis data
            save_path: Path to save visualization

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        # Get active circuits
        active_circuit_ids = self.epoch_to_circuits.get(epoch, set())

        if not active_circuit_ids:
            return None

        # Create interaction graph
        G = nx.Graph()

        # Add nodes for each circuit
        for cid in active_circuit_ids:
            circuit = self.registry.get_circuit(cid)
            if circuit:
                G.add_node(cid,
                           type=circuit.type.value,
                           attribution=circuit.attribution)

        # Add cooperation edges
        if cooperation_data and 'cooperation_matrix' in cooperation_data:
            matrix = cooperation_data['cooperation_matrix']
            circuit_ids = cooperation_data['circuit_ids']

            for i in range(len(circuit_ids)):
                for j in range(i + 1, len(circuit_ids)):
                    correlation = matrix[i, j]

                    # Add edge for strong positive correlation
                    if correlation > 0.5:
                        G.add_edge(circuit_ids[i], circuit_ids[j],
                                   weight=correlation,
                                   type='cooperation',
                                   style='solid')

        # Add competition edges
        if competition_data:
            # Add negative correlation edges
            for pair in competition_data.get('competing_pairs', []):
                G.add_edge(pair['circuit1'], pair['circuit2'],
                           weight=abs(pair['negative_correlation']),
                           type='competition',
                           style='dashed')

            # Add resource contention edges
            for contention in competition_data.get('resource_contention', []):
                G.add_edge(contention['circuit1'], contention['circuit2'],
                           weight=contention['component_overlap'],
                           type='contention',
                           style='dotted')

            # Add interference edges
            for interference in competition_data.get('interference_relationships', []):
                G.add_edge(interference['circuit1'], interference['circuit2'],
                           weight=abs(interference['interference']),
                           type='interference',
                           style='dashdot')

        # Create visualization
        fig = plt.figure(figsize=(14, 12))

        # Node properties
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            circuit = self.registry.get_circuit(node)
            if circuit:
                if circuit.type == CircuitType.TOKEN:
                    node_colors.append('skyblue')
                elif circuit.type == CircuitType.COMPONENT:
                    node_colors.append('salmon')
                else:
                    node_colors.append('lightgreen')

                # Size by attribution
                node_sizes.append(100 + circuit.attribution * 500)
            else:
                node_colors.append('gray')
                node_sizes.append(100)

        # Edge properties
        edge_colors = []
        edge_styles = []
        edge_widths = []

        for u, v, data in G.edges(data=True):
            if data.get('type') == 'cooperation':
                edge_colors.append('green')
            elif data.get('type') == 'competition':
                edge_colors.append('red')
            elif data.get('type') == 'contention':
                edge_colors.append('orange')
            elif data.get('type') == 'interference':
                edge_colors.append('purple')
            else:
                edge_colors.append('gray')

            # Style based on relationship type
            edge_styles.append(data.get('style', 'solid'))

            # Width by relationship strength
            edge_widths.append(1 + data.get('weight', 0.5) * 3)

        # Layout that separates competitive circuits
        pos = nx.spring_layout(G, k=0.4, seed=42)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

        # Draw edges by style
        for i, (u, v, data) in enumerate(G.edges(data=True)):
            style = data.get('style', 'solid')
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                   width=edge_widths[i],
                                   edge_color=[edge_colors[i]],
                                   style=style,
                                   alpha=0.7)

        # Add small labels
        nx.draw_networkx_labels(G, pos, font_size=8)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue',
                   markersize=10, label='Token Circuit'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon',
                   markersize=10, label='Component Circuit'),
            Line2D([0], [0], color='green', lw=2, label='Cooperation'),
            Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Competition'),
            Line2D([0], [0], color='orange', lw=2, linestyle='dotted', label='Resource Contention'),
            Line2D([0], [0], color='purple', lw=2, linestyle='dashdot', label='Interference')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        plt.title(f"Circuit Cooperation and Competition at Epoch {epoch}")
        plt.axis('off')
        plt.tight_layout()
        # save_path = self.save_dir / save_path
        if save_path:
            self.save_figure_safe(fig=fig, filename=save_path, save_dir=self.save_dir)
            # plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return plt.gcf()

    # Add to circuit_evolution_tracker.py
    def visualize_circuit_lineage(self, start_epoch, end_epoch, save_path=None):
        """
        Visualize circuit lineage evolution over a range of epochs

        Args:
            start_epoch: Start epoch for visualization
            end_epoch: End epoch for visualization
            save_path: Path to save visualization

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        # Get all epochs in range
        epochs = [e for e in sorted(self.epoch_to_circuits.keys())
                  if start_epoch <= e <= end_epoch]

        if not epochs:
            return None

        # Create graph of circuit evolution
        G = nx.DiGraph()

        # Track all transformations for these epochs
        lineage_data = {}

        for i, epoch in enumerate(epochs[1:], 1):  # Start from second epoch
            prev_epoch = epochs[i - 1]
            lineage = self.track_circuit_lineage(epoch)
            lineage_data[epoch] = lineage

            # Add nodes for circuits
            current_circuits = self.epoch_to_circuits.get(epoch, set())
            for cid in current_circuits:
                circuit = self.registry.get_circuit(cid)
                if circuit:
                    node_id = f"{cid}_{epoch}"
                    G.add_node(node_id,
                               epoch=epoch,
                               circuit_id=cid,
                               circuit_type=circuit.type.value,
                               attribution=circuit.attribution)

            # Add lineage edges
            for child_id, info in lineage.get('evolved_circuits', {}).items():
                parent_id = info['parent']
                child_node = f"{child_id}_{epoch}"
                parent_node = f"{parent_id}_{prev_epoch}"

                if parent_node in G and child_node in G:
                    G.add_edge(parent_node, child_node,
                               similarity=info['similarity'],
                               type='evolution')

            # Add merge edges
            for merged_id, merge_info in lineage.get('transformations', {}).get('merges', {}).items():
                merged_node = f"{merged_id}_{epoch}"

                for i, parent_id in enumerate(merge_info['parents']):
                    parent_node = f"{parent_id}_{prev_epoch}"
                    if parent_node in G and merged_node in G:
                        G.add_edge(parent_node, merged_node,
                                   similarity=merge_info['similarities'][i],
                                   type='merge')

            # Add split edges
            for parent_id, split_info in lineage.get('transformations', {}).get('splits', {}).items():
                parent_node = f"{parent_id}_{prev_epoch}"

                for i, child_id in enumerate(split_info['children']):
                    child_node = f"{child_id}_{epoch}"
                    if parent_node in G and child_node in G:
                        G.add_edge(parent_node, child_node,
                                   similarity=split_info['similarities'][i],
                                   type='split')

        # Create visualization
        fig = plt.figure(figsize=(16, 12))

        # Layout that emphasizes temporal flow
        pos = {}
        max_circuits_per_epoch = max(len(self.epoch_to_circuits.get(e, set())) for e in epochs)

        for node, data in G.nodes(data=True):
            epoch = data['epoch']
            epoch_circuits = sorted(list(n for n, d in G.nodes(data=True) if d['epoch'] == epoch))
            idx = epoch_circuits.index(node)

            # Position nodes in columns by epoch, evenly spaced vertically
            x = (epoch - start_epoch) / max(1, end_epoch - start_epoch)
            y = idx / max(1, len(epoch_circuits) - 1)

            pos[node] = (x, y)

        # Node properties
        node_colors = []
        node_sizes = []
        for node, data in G.nodes(data=True):
            if data['circuit_type'] == 'TOKEN':
                node_colors.append('skyblue')
            elif data['circuit_type'] == 'COMPONENT':
                node_colors.append('salmon')
            else:
                node_colors.append('lightgreen')

            # Size by attribution
            node_sizes.append(100 + data.get('attribution', 0.5) * 500)

        # Edge properties
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            if data.get('type') == 'merge':
                edge_colors.append('red')
            elif data.get('type') == 'split':
                edge_colors.append('green')
            else:  # evolution
                edge_colors.append('gray')

            # Width by similarity
            edge_widths.append(1 + data.get('similarity', 0.5) * 4)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                               alpha=0.7, arrowsize=15, connectionstyle='arc3,rad=0.1')

        # Add small labels
        nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['circuit_id'].split('_')[0]
                                                for n in G.nodes}, font_size=8)

        # Add epoch markers
        for epoch in epochs:
            plt.axvline(x=(epoch - start_epoch) / max(1, end_epoch - start_epoch),
                        color='gray', linestyle='--', alpha=0.3)
            plt.text((epoch - start_epoch) / max(1, end_epoch - start_epoch),
                     -0.05, f"Epoch {epoch}", rotation=90, va='top')

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue',
                   markersize=10, label='Token Circuit'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon',
                   markersize=10, label='Component Circuit'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                   markersize=10, label='Functional Circuit'),
            Line2D([0], [0], color='gray', lw=2, label='Evolution'),
            Line2D([0], [0], color='red', lw=2, label='Merge'),
            Line2D([0], [0], color='green', lw=2, label='Split')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        plt.title(f"Circuit Evolution from Epoch {start_epoch} to {end_epoch}")
        plt.axis('off')
        plt.tight_layout()
        # save_path = self.save_dir / save_path
        if save_path:
            self.save_figure_safe(fig=fig, filename=save_path, save_dir=self.save_dir)
            # plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return plt.gcf()





    def visualize_circuit_evolution(self, save_path=None):
        """
        Create visualizations of circuit evolution over time

        Args:
            save_path: Optional path to save the visualization

        Returns:
            matplotlib Figure
        """
        # Create figure for circuit evolution
        fig = plt.figure(figsize=(12, 8))

        # Count circuits per epoch
        epochs = sorted(self.epoch_to_circuits.keys())
        circuit_counts = [len(self.epoch_to_circuits[e]) for e in epochs]

        # Plot circuit count evolution
        plt.plot(epochs, circuit_counts, 'b-', marker='o', label='Active Circuits')

        # Separate by circuit type
        circuit_types = {}

        for epoch in epochs:
            for circuit_id in self.epoch_to_circuits[epoch]:
                circuit = self.registry.get_circuit(circuit_id)
                if circuit and hasattr(circuit, 'type'):
                    circuit_type = circuit.type.value

                    if circuit_type not in circuit_types:
                        circuit_types[circuit_type] = {}

                    if epoch not in circuit_types[circuit_type]:
                        circuit_types[circuit_type][epoch] = 0

                    circuit_types[circuit_type][epoch] += 1

        # Plot counts by type
        for circuit_type, epoch_counts in circuit_types.items():
            type_epochs = sorted(epoch_counts.keys())
            type_counts = [epoch_counts[e] for e in type_epochs]

            plt.plot(type_epochs, type_counts, 'o-',
                     label=f'{circuit_type} Circuits')

        # Mark emergence events
        for circuit_id, emergence_epoch in self.emergence_epochs.items():
            circuit = self.registry.get_circuit(circuit_id)
            if circuit and hasattr(circuit, 'type'):
                circuit_type = circuit.type.value

                # Only mark first few emergencies to avoid clutter
                if epoch_counts.get(emergence_epoch, 0) <= 3:
                    plt.axvline(x=emergence_epoch, color='r', linestyle='--', alpha=0.3)
                    plt.text(
                        emergence_epoch,
                        max(circuit_counts) * 0.8,
                        f"{circuit_type} circuit",
                        rotation=90,
                        alpha=0.7
                    )

        plt.xlabel('Epoch')
        plt.ylabel('Number of Circuits')
        plt.title('Circuit Evolution Over Training')
        plt.legend()
        plt.grid(alpha=0.3)
        # save_path = self.save_dir / save_path
        if save_path:
            self.save_figure_safe(fig=fig, filename=save_path, save_dir=self.save_dir)
            # plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # print(f"\t{get_current_callable_info()}:\t")

        return plt.gcf()

    def visualize_emergence_order(self, save_path=None):
        """
        Visualize the order in which different circuit types emerge

        Args:
            save_path: Optional path to save the visualization

        Returns:
            matplotlib Figure
        """
        # Get emergence analysis
        emergence = self.analyze_emergence_order()

        if not emergence['avg_emergence']:
            # Not enough data
            return None

        fig = plt.figure(figsize=(10, 6))

        # Create boxplot data
        boxplot_data = []
        labels = []

        for circuit_type in emergence['emergence_order']:
            circuits = emergence['circuits_by_type'][circuit_type]
            epochs = [epoch for _, epoch in circuits]
            boxplot_data.append(epochs)
            labels.append(f"{circuit_type} ({len(epochs)})")

        # Create boxplot
        plt.boxplot(boxplot_data, labels=labels)

        # Add individual points
        for i, (circuit_type, values) in enumerate(zip(emergence['emergence_order'], boxplot_data)):
            x = [i + 1] * len(values)
            plt.scatter(x, values, alpha=0.5)

        plt.xlabel('Circuit Type')
        plt.ylabel('Emergence Epoch')
        plt.title('Circuit Emergence Order')
        plt.grid(axis='y', alpha=0.3)
        # save_path = self.save_dir / save_path
        if save_path:
            self.save_figure_safe(fig=fig, filename=save_path, save_dir=self.save_dir)
            # plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return plt.gcf()