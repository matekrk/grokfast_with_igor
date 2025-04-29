import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.path as mpath
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
import os


class AnthropicStyleTransformerViz:
    """
    Creates Anthropic-style visualizations of transformer models at different training states.
    Captures attention patterns, MLP activities, and circuit formation.
    """

    def __init__(self, num_layers=2, num_heads=4, head_dim=32, mlp_dim=128):
        """
        Initialize the visualizer.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head
            mlp_dim: Dimension of MLP hidden layers
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim

        # Visual style configuration
        self.figsize = (12, 6 * num_layers + 3)  # Scale with model size
        self.layer_height = 4
        self.layer_spacing = 3.5
        self.box_width = 2.0
        self.box_height = 1.0

        # Color scheme - Anthropic style
        self.colors = {
            'attention': '#5470C6',  # Blue
            'mlp': '#91CC75',  # Green
            'residual': '#FAC858',  # Yellow/Gold
            'circuit': '#EE6666',  # Red
            'text': '#303133',  # Dark gray
            'background': '#F9F9F9',  # Light gray
            'embedding': '#9A60B4',  # Purple
            'layernorm': '#73C0DE',  # Light blue
        }

        # Gradient colormap for attention weights
        self.attn_cmap = LinearSegmentedColormap.from_list(
            'attn_cmap',
            [(0.0, '#FFFFFF00'), (0.3, '#5470C640'), (1.0, '#5470C6')]
        )

        # Store positions for drawing connections
        self.component_positions = {}

    def _create_figure(self):
        """Create the figure and axes objects."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Set background color
        fig.patch.set_facecolor(self.colors['background'])
        ax.set_facecolor(self.colors['background'])

        # Configure axes
        ax.set_xlim(-1, 13)
        ax.set_ylim(-2, self.num_layers * self.layer_spacing + 3)
        ax.axis('off')

        return fig, ax

    def _draw_block(self, ax, x, y, width, height, color, alpha=1.0,
                    label=None, label_pos='center', label_color=None,
                    component_id=None, zorder=2):
        """Draw a generic block with optional label."""
        if label_color is None:
            label_color = self.colors['text']

        rect = patches.Rectangle(
            (x - width / 2, y - height / 2),
            width, height,
            facecolor=color,
            alpha=alpha,
            edgecolor='black',
            linewidth=1,
            zorder=zorder,
            clip_on=False
        )
        ax.add_patch(rect)

        if label:
            if label_pos == 'center':
                ax.text(x, y, label, color=label_color, ha='center', va='center',
                        fontsize=10, fontweight='medium', zorder=zorder + 1)
            elif label_pos == 'top':
                ax.text(x, y + height / 2 + 0.2, label, color=label_color, ha='center',
                        va='bottom', fontsize=10, fontweight='medium', zorder=zorder + 1)

        # Store position for connections
        if component_id:
            self.component_positions[component_id] = (x, y)

    def _draw_arrow(self, ax, x1, y1, x2, y2, color='black', width=1.0,
                    style='-', alpha=1.0, zorder=1):
        """Draw an arrow between two points."""
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle="arc3,rad=0.0",
            arrowstyle="-|>",
            mutation_scale=15,
            color=color,
            linewidth=width,
            linestyle=style,
            alpha=alpha,
            zorder=zorder,
            clip_on=False
        )
        ax.add_patch(arrow)

    def _draw_curved_connection(self, ax, start, end, control_points=None,
                                color='black', width=1.0, style='-',
                                alpha=1.0, arrow=True, zorder=1):
        """Draw a curved connection between two points."""
        if control_points is None:
            # Default bezier curve
            dx = end[0] - start[0]
            control_points = [
                (start[0] + dx * 0.3, start[1]),
                (start[0] + dx * 0.7, end[1])
            ]

        # Create path
        verts = [start, control_points[0], control_points[1], end]
        codes = [
            mpath.Path.MOVETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4
        ]

        path = mpath.Path(verts, codes)
        patch = patches.PathPatch(
            path,
            facecolor='none',
            edgecolor=color,
            linewidth=width,
            linestyle=style,
            alpha=alpha,
            zorder=zorder,
            clip_on=False
        )
        ax.add_patch(patch)

        # Add arrow if requested
        if arrow:
            # Calculate arrow position and orientation
            # This is at 90% of the path
            t = 0.9
            point = path.interpolated(101).vertices[int(100 * t)]
            # Approximate tangent direction
            t_prev = t - 0.01
            prev_point = path.interpolated(101).vertices[int(100 * t_prev)]
            direction = point - prev_point
            direction = direction / np.linalg.norm(direction)

            # Draw arrow head
            arrow_size = 0.15
            arrow = patches.FancyArrowPatch(
                point - arrow_size * direction,
                point,
                arrowstyle="-|>",
                mutation_scale=15,
                color=color,
                linewidth=width,
                alpha=alpha,
                zorder=zorder,
                clip_on=False
            )
            ax.add_patch(arrow)

    def _draw_attention_head(self, ax, x, y, head_idx, attribution=0.5, scale=1.0):
        """Draw an attention head with signal strength visualization."""
        # Calculate size based on attribution and scale
        size = 0.6 * scale
        label_size = 9 * scale

        intensity = max(0.3, attribution)
        color = self.colors['attention']

        # Draw head
        head_id = f"head_{head_idx}"
        self._draw_block(
            ax, x, y, size, size, color,
            alpha=intensity,
            label=f"{head_idx}",
            label_color='white',
            component_id=head_id,
            zorder=3
        )

        return head_id

    def _draw_attention_block(self, ax, x, y, layer_idx, head_attributions=None,
                              attention_patterns=None, width=2.5, height=2.0):
        """Draw a multi-head attention block with all heads."""
        if head_attributions is None:
            head_attributions = [0.5] * self.num_heads

        block_id = f"attn_block_{layer_idx}"

        # Draw attention block outline
        self._draw_block(
            ax, x, y, width, height, 'white', alpha=0.8,
            label=f"Multi-Head Attention", label_pos='top',
            component_id=block_id
        )

        # Draw individual heads
        head_positions = []
        head_spacing = width / (self.num_heads + 1)
        for h in range(self.num_heads):
            head_x = x - width / 2 + head_spacing * (h + 1)
            head_y = y

            attr = head_attributions[h]
            head_id = self._draw_attention_head(ax, head_x, head_y, h, attr)
            head_positions.append((head_id, head_x, head_y))

            # Store position with layer context
            full_head_id = f"L{layer_idx}_H{h}"
            self.component_positions[full_head_id] = (head_x, head_y)

        return block_id, head_positions

    def _draw_mlp_block(self, ax, x, y, layer_idx, activation=0.5,
                        width=2.5, height=1.5):
        """Draw an MLP block with activation indication."""
        block_id = f"mlp_block_{layer_idx}"
        intensity = max(0.3, activation)

        # Draw MLP block
        self._draw_block(
            ax, x, y, width, height, self.colors['mlp'],
            alpha=intensity,
            label=f"MLP",
            component_id=block_id
        )

        return block_id

    def _draw_layernorm(self, ax, x, y, width=0.8, height=0.3, label="LN"):
        """Draw a layer normalization block."""
        ln_id = f"ln_{x}_{y}"
        self._draw_block(
            ax, x, y, width, height, self.colors['layernorm'],
            label=label, component_id=ln_id
        )
        return ln_id

    def _draw_add(self, ax, x, y, size=0.3):
        """Draw an addition operation."""
        add_id = f"add_{x}_{y}"
        circle = patches.Circle(
            (x, y),
            size,
            facecolor='white',
            edgecolor='black',
            linewidth=1,
            zorder=3,
            clip_on=False
        )
        ax.add_patch(circle)
        ax.text(x, y, "+", color='black', ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=4)

        self.component_positions[add_id] = (x, y)
        return add_id

    def _draw_embedding(self, ax, x, y, width=2.5, height=1.0):
        """Draw the embedding layer."""
        embed_id = "embedding"
        self._draw_block(
            ax, x, y, width, height, self.colors['embedding'],
            label="Token + Position\nEmbedding",
            component_id=embed_id
        )
        return embed_id

    def _draw_output(self, ax, x, y, width=2.5, height=1.0):
        """Draw the output layer."""
        output_id = "output"
        self._draw_block(
            ax, x, y, width, height, self.colors['text'],
            label="Output",
            label_color='white',
            component_id=output_id
        )
        return output_id

    def _draw_circuit(self, ax, src, tgt, strength=0.5, offset=0.0):
        """Draw a circuit connection between components."""
        if src not in self.component_positions or tgt not in self.component_positions:
            return

        src_pos = self.component_positions[src]
        tgt_pos = self.component_positions[tgt]

        # Define control points for a nice curve
        mid_y = (src_pos[1] + tgt_pos[1]) / 2

        if offset != 0:
            # If we need to offset (for multiple circuits)
            ctrl1 = (src_pos[0] + 0.5, src_pos[1] + offset)
            ctrl2 = (tgt_pos[0] - 0.5, tgt_pos[1] + offset)
        else:
            # Normal curve
            ctrl1 = (src_pos[0] + 0.5, mid_y)
            ctrl2 = (tgt_pos[0] - 0.5, mid_y)

        # Draw the circuit connection
        width = 1.0 + strength * 2.0  # Vary width by strength
        alpha = 0.4 + strength * 0.5  # Vary opacity by strength

        self._draw_curved_connection(
            ax,
            src_pos, tgt_pos,
            control_points=[ctrl1, ctrl2],
            color=self.colors['circuit'],
            width=width,
            style='-',
            alpha=alpha,
            arrow=True,
            zorder=1
        )

    def _draw_residual(self, ax, start, end, y_offset=0.7):
        """Draw a residual connection."""
        if start not in self.component_positions or end not in self.component_positions:
            return

        start_pos = self.component_positions[start]
        end_pos = self.component_positions[end]

        # Offset the residual connection
        start_pos_offset = (start_pos[0], start_pos[1] + y_offset)
        end_pos_offset = (end_pos[0], end_pos[1] + y_offset)

        # Draw the residual connection
        self._draw_curved_connection(
            ax,
            start_pos_offset, end_pos_offset,
            color=self.colors['residual'],
            width=1.5,
            style='--',
            alpha=0.7,
            zorder=0
        )

    def _draw_transformer_layer(self, ax, x, y, layer_idx,
                                head_attributions=None, mlp_activation=0.5):
        """Draw a complete transformer layer with attention and MLP."""
        if head_attributions is None:
            head_attributions = [0.5] * self.num_heads

        # Spacing parameters
        h_spacing = 5.0
        v_spacing = 1.2

        # Draw layer components
        layer_id = f"layer_{layer_idx}"

        # Layer label
        ax.text(x, y + 1.5, f"Layer {layer_idx}",
                fontsize=12, fontweight='bold', ha='center')

        # Layer normalization 1
        ln1_id = self._draw_layernorm(ax, x, y)

        # Multi-head attention
        attn_x = x + h_spacing / 2
        attn_y = y
        attn_id, head_positions = self._draw_attention_block(
            ax, attn_x, attn_y, layer_idx, head_attributions
        )

        # Layer normalization 2
        ln2_id = self._draw_layernorm(ax, x + h_spacing, y)

        # MLP block
        mlp_x = x + h_spacing * 1.5
        mlp_y = y
        mlp_id = self._draw_mlp_block(ax, mlp_x, mlp_y, layer_idx, mlp_activation)

        # Addition nodes
        add1_x = x + h_spacing * 0.75
        add1_y = y - v_spacing
        add1_id = self._draw_add(ax, add1_x, add1_y)

        add2_x = x + h_spacing * 1.75
        add2_y = y - v_spacing
        add2_id = self._draw_add(ax, add2_x, add2_y)

        # Main flow connections
        # Input to LN1
        input_id = f"layer_{layer_idx - 1}_output" if layer_idx > 0 else "embedding"
        if input_id in self.component_positions:
            input_pos = self.component_positions[input_id]
            self._draw_arrow(ax, input_pos[0], input_pos[1], x, y)

        # LN1 to Attention
        self._draw_arrow(ax, x, y, attn_x, attn_y)

        # Attention to Add1
        self._draw_arrow(ax, attn_x, attn_y, add1_x, add1_y)

        # Add1 to LN2
        self._draw_arrow(ax, add1_x, add1_y, x + h_spacing, y)

        # LN2 to MLP
        self._draw_arrow(ax, x + h_spacing, y, mlp_x, mlp_y)

        # MLP to Add2
        self._draw_arrow(ax, mlp_x, mlp_y, add2_x, add2_y)

        # Residual connections
        if input_id in self.component_positions:
            self._draw_residual(ax, input_id, add1_id)
        self._draw_residual(ax, add1_id, add2_id)

        # Store output position
        self.component_positions[f"layer_{layer_idx}_output"] = (add2_x, add2_y)

        # Return layer components for reference
        return {
            'layer_id': layer_id,
            'ln1_id': ln1_id,
            'attn_id': attn_id,
            'ln2_id': ln2_id,
            'mlp_id': mlp_id,
            'add1_id': add1_id,
            'add2_id': add2_id,
            'head_positions': head_positions
        }

    def visualize(self, head_attributions=None, mlp_activations=None, circuits=None,
                  title="Transformer Model Visualization", save_path=None):
        """
        Create a complete transformer visualization.

        Args:
            head_attributions: List of lists with attribution values [[layer0_heads], [layer1_heads]]
            mlp_activations: List of MLP activation values per layer
            circuits: List of circuits [("L0_H1", "L1_H2", strength), ...]
            title: Title for the visualization
            save_path: Path to save the visualization (if None, the plot is displayed)

        Returns:
            matplotlib figure object
        """
        # Create default values if not provided
        if head_attributions is None:
            head_attributions = [[0.5] * self.num_heads for _ in range(self.num_layers)]

        if mlp_activations is None:
            mlp_activations = [0.5] * self.num_layers

        if circuits is None:
            circuits = []

        # Create figure
        fig, ax = self._create_figure()

        # Draw embedding layer
        embed_x = 2
        base_y = self.num_layers * self.layer_spacing
        embed_id = self._draw_embedding(ax, embed_x, base_y)

        # Draw each transformer layer
        layers = []
        for i in range(self.num_layers):
            layer_y = base_y - i * self.layer_spacing
            layer_components = self._draw_transformer_layer(
                ax, embed_x, layer_y, i,
                head_attributions[i],
                mlp_activations[i]
            )
            layers.append(layer_components)

        # Draw output layer
        output_x = embed_x + 8.75  # Aligned with the last add node
        output_y = base_y - (self.num_layers - 1) * self.layer_spacing - 1.2
        output_id = self._draw_output(ax, output_x, output_y)

        # Connect last layer to output
        last_output_id = f"layer_{self.num_layers - 1}_output"
        if last_output_id in self.component_positions:
            last_output_pos = self.component_positions[last_output_id]
            self._draw_arrow(ax, last_output_pos[0], last_output_pos[1], output_x, output_y)

        # Draw circuits between attention heads
        circuit_offsets = {}  # Track offsets for multiple circuits between same components
        for circuit in circuits:
            src, tgt, strength = circuit

            # Generate offset for multiple connections between same components
            key = f"{src}_{tgt}"
            if key in circuit_offsets:
                circuit_offsets[key] += 0.1
            else:
                circuit_offsets[key] = 0.0

            self._draw_circuit(ax, src, tgt, strength, circuit_offsets[key])

        # Add title
        ax.text(embed_x + 4.5, base_y + 2, title, fontsize=16, fontweight='bold', ha='center')

        # Add legend
        legend_x = 11
        legend_y = base_y - 1
        legend_items = [
            (self.colors['attention'], "Attention Head"),
            (self.colors['mlp'], "MLP Block"),
            (self.colors['residual'], "Residual Connection"),
            (self.colors['circuit'], "Circuit Connection"),
            (self.colors['layernorm'], "Layer Normalization"),
            (self.colors['embedding'], "Embedding Layer")
        ]

        for i, (color, label) in enumerate(legend_items):
            y_offset = -i * 0.5
            rect = patches.Rectangle(
                (legend_x, legend_y + y_offset),
                0.3, 0.3,
                facecolor=color,
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
            ax.text(legend_x + 0.5, legend_y + y_offset + 0.15, label, fontsize=9, ha='left', va='center')

        # Save or show plot
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        return fig

    def visualize_training_phases(self, phases=None, save_dir=None):
        """
        Create visualizations for different training phases.

        Args:
            phases: Dictionary of phase data (early, middle, late)
            save_dir: Directory to save visualizations

        Returns:
            List of figure objects
        """
        if phases is None:
            # Default phase data
            phases = {
                'early': {
                    'head_attributions': [[0.1, 0.3, 0.2, 0.1], [0.2, 0.1, 0.3, 0.1]],
                    'mlp_activations': [0.2, 0.3],
                    'circuits': [("L0_H1", "L1_H2", 0.2)]
                },
                'middle': {
                    'head_attributions': [[0.3, 0.7, 0.4, 0.2], [0.4, 0.3, 0.6, 0.2]],
                    'mlp_activations': [0.5, 0.6],
                    'circuits': [
                        ("L0_H1", "L1_H2", 0.5),
                        ("L0_H2", "L1_H0", 0.3)
                    ]
                },
                'late': {
                    'head_attributions': [[0.4, 0.9, 0.6, 0.4], [0.5, 0.6, 0.9, 0.7]],
                    'mlp_activations': [0.8, 0.9],
                    'circuits': [
                        ("L0_H1", "L1_H2", 0.9),
                        ("L0_H2", "L1_H0", 0.7),
                        ("L0_H0", "L1_H3", 0.5)
                    ]
                }
            }

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        figures = []

        # Create visualization for each phase
        for phase_name, phase_data in phases.items():
            title = f"Transformer Model - {phase_name.capitalize()} Training Phase"

            if save_dir:
                save_path = os.path.join(save_dir, f"transformer_{phase_name}_phase.svg")
            else:
                save_path = None

            fig = self.visualize(
                head_attributions=phase_data['head_attributions'],
                mlp_activations=phase_data['mlp_activations'],
                circuits=phase_data['circuits'],
                title=title,
                save_path=save_path
            )

            figures.append(fig)

        return figures
