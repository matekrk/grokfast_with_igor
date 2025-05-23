# visualization/token_visualizer.py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from analysis.core.circuit_schema import Element, ElementType, Circuit

class TokenCircuitVisualizer:
    """Visualization tools for token-level circuits"""

    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the token circuit visualizer

        Args:
            save_dir: Directory to save visualizations
        """
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

    def visualize_token_attribution(self, attribution_map: np.ndarray,
                                    tokens: List[str],
                                    title: str = "Token Attribution Map",
                                    show: bool = True,
                                    save_path: Optional[Union[str, Path]] = None):
        """
        Visualize token-to-token attribution as a heatmap

        Args:
            attribution_map: 2D array of token influence (from_token, to_token)
            tokens: List of token strings
            title: Plot title
            show: Whether to display the plot
            save_path: Path to save the visualization

        Returns:
            matplotlib Figure
        """
        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(attribution_map, annot=False, cmap="viridis",
                    xticklabels=tokens, yticklabels=tokens)

        plt.title(title)
        plt.xlabel("Target Token")
        plt.ylabel("Source Token")

        # Adjust layout for readability
        plt.tight_layout()

        # Save if path provided
        if save_path:
            if self.save_dir and not isinstance(save_path, Path):
                save_path = self.save_dir / save_path
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()

        return plt.gcf()

    def visualize_token_circuit(self, circuit: Circuit,
                                title: Optional[str] = None,
                                show: bool = True,
                                save_path: Optional[Union[str, Path]] = None):
        """
        Visualize a token-level circuit as a graph

        Args:
            circuit: The circuit to visualize
            title: Plot title (defaults to circuit ID)
            show: Whether to display the plot
            save_path: Path to save the visualization

        Returns:
            matplotlib Figure
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes with their types
        for element in circuit.elements:
            node_type = element.type.value
            properties = element.properties

            # Get node labels
            if element.type == ElementType.TOKEN:
                label = f"{properties.get('token', '')} ({properties.get('position', '?')})"
            elif element.type == ElementType.HEAD:
                label = properties.get('name', element.id)
            else:
                label = element.id

            G.add_node(element.id, type=node_type, properties=properties, label=label)

        # Add edges with weights
        for connection in circuit.connections:
            G.add_edge(connection.source, connection.target,
                       weight=connection.strength,
                       type=connection.type.value,
                       properties=connection.properties)

        # Create plot
        plt.figure(figsize=(10, 8))

        # Define node colors by type
        node_colors = {
            "token": "skyblue",
            "head": "salmon",
            "mlp": "lightgreen",
            "position": "yellow",
            "subspace": "purple"
        }

        # Get node colors and sizes
        colors = [node_colors.get(G.nodes[n]['type'], 'gray') for n in G.nodes]
        sizes = [100 + G.nodes[n]['properties'].get('importance', 1) * 50 for n in G.nodes]

        # Get edge widths based on weights
        edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges]

        # Use a hierarchical layout for clarity
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot') if nx.nx_agraph.graphviz_layout else nx.spring_layout(G)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray',
                               connectionstyle='arc3,rad=0.1', alpha=0.7)
        nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes})

        # Add title
        if title is None:
            title = f"Circuit: {circuit.id}"
        plt.title(title)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=10, label=node_type)
            for node_type, color in node_colors.items()
            if any(G.nodes[n]['type'] == node_type for n in G.nodes)
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.axis('off')
        plt.tight_layout()

        # Save if path provided
        if save_path:
            if self.save_dir and not isinstance(save_path, Path):
                save_path = self.save_dir / save_path
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()

        return plt.gcf()

    def visualize_token_flow(self, circuit: Circuit,
                             tokens: List[str],
                             title: Optional[str] = None,
                             show: bool = True,
                             save_path: Optional[Union[str, Path]] = None):
        """
        Visualize token information flow as a sequence diagram

        Args:
            circuit: The circuit to visualize
            tokens: List of token strings
            title: Plot title
            show: Whether to display the plot
            save_path: Path to save the visualization

        Returns:
            matplotlib Figure
        """
        # Extract token elements
        token_elements = [e for e in circuit.elements if e.type == ElementType.TOKEN]
        token_elements.sort(key=lambda e: e.properties.get('position', 0))

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Draw tokens as sequence of boxes
        box_width = 0.8
        token_y = 1.0
        token_boxes = []

        for i, token_element in enumerate(token_elements):
            pos = token_element.properties.get('position', i)
            token_text = token_element.properties.get('token', f"Token {pos}")

            # Draw token box
            box = plt.Rectangle((pos, token_y - box_width / 2), box_width, box_width,
                                fill=True, color='skyblue', alpha=0.8)
            ax.add_patch(box)
            token_boxes.append(box)

            # Add token text
            ax.text(pos + box_width / 2, token_y, token_text,
                    ha='center', va='center', fontsize=10)

        # Draw operations as arrows
        for conn in circuit.connections:
            # Find source and target elements
            source_elem = next((e for e in circuit.elements if e.id == conn.source), None)
            target_elem = next((e for e in circuit.elements if e.id == conn.target), None)

            if source_elem and target_elem:
                if source_elem.type == ElementType.TOKEN and target_elem.type == ElementType.TOKEN:
                    # Direct token-to-token flow
                    src_pos = source_elem.properties.get('position', 0)
                    tgt_pos = target_elem.properties.get('position', 0)

                    # Draw curved arrow
                    ax.annotate("",
                                xy=(tgt_pos + box_width / 2, token_y - box_width / 2 - 0.1),
                                xytext=(src_pos + box_width / 2, token_y - box_width / 2 - 0.1),
                                arrowprops=dict(arrowstyle="->", color="red", lw=conn.strength * 2,
                                                connectionstyle=f"arc3,rad=0.3"))

                    # Add operation label
                    mid_x = (src_pos + tgt_pos) / 2
                    mid_y = token_y - box_width / 2 - 0.3
                    op_type = conn.properties.get('operation', conn.type.value)
                    ax.text(mid_x, mid_y, op_type,
                            ha='center', va='center', fontsize=8, style='italic')

                # Other connection types could be visualized differently

        # Set plot limits
        ax.set_xlim(-0.5, len(token_elements) + 0.5)
        ax.set_ylim(0, 2)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add title
        if title is None:
            title = f"Token Flow: {circuit.id}"
        ax.set_title(title)

        # Save if path provided
        if save_path:
            if self.save_dir and not isinstance(save_path, Path):
                save_path = self.save_dir / save_path
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()

        return fig