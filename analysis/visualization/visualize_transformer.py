import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.path as mpath
import numpy as np
from matplotlib.colors import to_rgba
import os


class TransformerVisualizer:
    """
    Visualizes a Transformer model architecture with attention patterns,
    MLP activations, and head attributions.
    """

    def __init__(self, num_layers=2, num_heads=4, figsize=(10, 8)):
        """
        Initialize the visualizer.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            figsize: Figure size for the visualization
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.figsize = figsize

        # Layout parameters
        self.layer_spacing = 2.5
        self.head_spacing = 0.8
        self.layer_height = 4

        # Colors
        self.head_color = "#4285F4"  # Blue
        self.mlp_color = "#34A853"  # Green
        self.circuit_color = "#EA4335"  # Red
        self.residual_color = "#AAAAAA"  # Gray
        self.embedding_color = "#DDDDDD"  # Light gray
        self.output_color = "#DDDDDD"  # Light gray

        # Store head positions for circuit connections
        self.head_positions = {}

    def _get_intensity_color(self, base_color, value):
        """Convert color with an alpha value based on intensity."""
        rgba = to_rgba(base_color)
        return (rgba[0], rgba[1], rgba[2], max(0.2, min(0.9, value)))

    def _draw_head(self, ax, x, y, layer_idx, head_idx, attribution=0.5):
        """Draw an attention head with size based on attribution."""
        size = max(0.3, attribution * 0.6)
        color = self._get_intensity_color(self.head_color, attribution)

        head = plt.Circle((x, y), size, color=color, ec='black', lw=1, zorder=10)
        ax.add_artist(head)

        # Add head index label
        ax.text(x, y, str(head_idx), color='white', ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=11)

        # Store position for circuit connections
        self.head_positions[f"{layer_idx}-{head_idx}"] = (x, y)

    def _draw_mlp(self, ax, x, y, width, height, activation=0.5):
        """Draw an MLP block with color intensity based on activation."""
        color = self._get_intensity_color(self.mlp_color, activation)
        mlp = patches.Rectangle((x - width / 2, y - height / 2), width, height,
                                facecolor=color, edgecolor='black', lw=1,
                                alpha=max(0.5, activation), zorder=5,
                                corner_radius=0.1)
        ax.add_patch(mlp)
        ax.text(x, y, "MLP", ha='center', va='center', fontsize=9, zorder=6)

    def _draw_embedding(self, ax, x, y, width, height):
        """Draw the input embedding block."""
        embedding = patches.Rectangle((x - width / 2, y - height / 2), width, height,
                                      facecolor=self.embedding_color, edgecolor='black',
                                      lw=1, zorder=5, corner_radius=0.1)
        ax.add_patch(embedding)
        ax.text(x, y, "Input Embedding", ha='center', va='center', fontsize=9, zorder=6)

    def _draw_output(self, ax, x, y, width, height):
        """Draw the output block."""
        output = patches.Rectangle((x - width / 2, y - height / 2), width, height,
                                   facecolor=self.output_color, edgecolor='black',
                                   lw=1, zorder=5, corner_radius=0.1)
        ax.add_patch(output)
        ax.text(x, y, "Output", ha='center', va='center', fontsize=9, zorder=6)

    def _draw_data_flow(self, ax, x1, y1, x2, y2, color='black', arrow=True):
        """Draw an arrow representing data flow."""
        line = mlines.Line2D([x1, x2], [y1, y2], color=color, lw=1.5, zorder=4)
        ax.add_line(line)

        if arrow:
            # Add arrowhead
            arrow_size = 0.1
            angle = np.arctan2(y2 - y1, x2 - x1)
            ax.add_patch(
                patches.Polygon(
                    [
                        (x2, y2),
                        (x2 - arrow_size * np.cos(angle) - arrow_size / 2 * np.sin(angle),
                         y2 - arrow_size * np.sin(angle) + arrow_size / 2 * np.cos(angle)),
                        (x2 - arrow_size * np.cos(angle) + arrow_size / 2 * np.sin(angle),
                         y2 - arrow_size * np.sin(angle) - arrow_size / 2 * np.cos(angle))
                    ],
                    closed=True,
                    color=color,
                    zorder=5
                )
            )

    def _draw_residual(self, ax, x1, y1, x2, y2):
        """Draw a residual connection."""
        # Offset from the main path
        offset = 0.5

        # Control points for curved line
        mid_x = (x1 + x2) / 2

        # Create curved path
        verts = [
            (x1, y1 + offset),  # Start at layer input with offset
            (mid_x, y1 + offset),  # First curve control point
            (mid_x, y2 + offset),  # Second curve control point
            (x2, y2 + offset)  # End point
        ]

        codes = [
            mpath.Path.MOVETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4
        ]

        path = mpath.Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', edgecolor=self.residual_color,
                                  linewidth=1.5, linestyle='--', zorder=2)
        ax.add_patch(patch)

        # Add arrowhead
        arrow_size = 0.1
        ax.add_patch(
            patches.Polygon(
                [
                    (x2, y2 + offset),
                    (x2 - arrow_size, y2 + offset + arrow_size / 2),
                    (x2 - arrow_size, y2 + offset - arrow_size / 2)
                ],
                closed=True,
                color=self.residual_color,
                zorder=3
            )
        )

    def _draw_circuit(self, ax, src_pos, tgt_pos, strength=0.5):
        """Draw a circuit connection between attention heads."""
        x1, y1 = src_pos
        x2, y2 = tgt_pos

        # Create curve control points
        ctrl_x1 = x1 + (x2 - x1) * 0.4
        ctrl_y1 = y1
        ctrl_x2 = x2 - (x2 - x1) * 0.4
        ctrl_y2 = y2

        # Create curved path
        verts = [
            (x1, y1),  # Start point
            (ctrl_x1, ctrl_y1),  # First control point
            (ctrl_x2, ctrl_y2),  # Second control point
            (x2, y2)  # End point
        ]

        codes = [
            mpath.Path.MOVETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4
        ]

        path = mpath.Path(verts, codes)
        line_width = max(0.5, strength * 3)
        color = self._get_intensity_color(self.circuit_color, strength)

        patch = patches.PathPatch(path, facecolor='none', edgecolor=color,
                                  linewidth=line_width, linestyle='--', zorder=1)
        ax.add_patch(patch)

    def visualize(self, head_attributions=None, mlp_activations=None, circuits=None,
                  save_path=None, title=None):
        """
        Create a visualization of the transformer model.

        Args:
            head_attributions: List of lists with attribution values [[layer0_heads], [layer1_heads]]
            mlp_activations: List of MLP activation values per layer
            circuits: List of circuits [(src_layer, src_head, tgt_layer, tgt_head, strength)]
            save_path: Path to save the visualization (if None, the plot is displayed)
            title: Title for the visualization

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

        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate dimensions
        total_width = self.layer_spacing * (self.num_layers + 1)
        total_height = self.layer_height + 1

        # Set axis limits with padding
        ax.set_xlim(-0.5, total_width + 0.5)
        ax.set_ylim(-0.5, total_height + 0.5)

        # Remove axis ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Draw model components
        layer_center_y = total_height / 2

        # Draw input embedding
        input_x = 0.5
        self._draw_embedding(ax, input_x, layer_center_y, 1, 0.8)

        # Draw each layer
        for layer_idx in range(self.num_layers):
            layer_x = 1.5 + layer_idx * self.layer_spacing

            # Layer label
            ax.text(layer_x, total_height - 0.3, f"Layer {layer_idx}",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Attention heads label
            ax.text(layer_x, total_height - 0.7, "Attention Heads",
                    ha='center', va='bottom', fontsize=9)

            # Draw attention heads
            for head_idx in range(self.num_heads):
                head_y = total_height - 1.2 - head_idx * self.head_spacing
                attr_value = head_attributions[layer_idx][head_idx]
                self._draw_head(ax, layer_x, head_y, layer_idx, head_idx, attr_value)

            # Draw MLP
            mlp_y = layer_center_y - 1
            act_value = mlp_activations[layer_idx]
            self._draw_mlp(ax, layer_x, mlp_y, 0.8, 0.5, act_value)

            # Data flow into layer
            if layer_idx == 0:
                self._draw_data_flow(ax, input_x + 0.6, layer_center_y, layer_x - 0.6, layer_center_y)

            # Data flow out of layer
            if layer_idx < self.num_layers - 1:
                next_x = 1.5 + (layer_idx + 1) * self.layer_spacing
                self._draw_data_flow(ax, layer_x + 0.6, layer_center_y, next_x - 0.6, layer_center_y)

                # Residual connection
                self._draw_residual(ax, layer_x - 0.6, layer_center_y, next_x - 0.6, layer_center_y)

        # Draw output
        output_x = 1.5 + self.num_layers * self.layer_spacing
        self._draw_output(ax, output_x, layer_center_y, 1, 0.8)

        # Draw circuits
        for circuit in circuits:
            src_layer, src_head, tgt_layer, tgt_head, strength = circuit

            src_key = f"{src_layer}-{src_head}"
            tgt_key = f"{tgt_layer}-{tgt_head}"

            if src_key in self.head_positions and tgt_key in self.head_positions:
                self._draw_circuit(
                    ax,
                    self.head_positions[src_key],
                    self.head_positions[tgt_key],
                    strength
                )

        # Add title
        if title:
            plt.title(title, fontsize=12, pad=20)

        # Add legend
        head_patch = patches.Patch(color=self.head_color, label='Attention Head')
        mlp_patch = patches.Patch(color=self.mlp_color, label='MLP')
        circuit_line = mlines.Line2D([], [], color=self.circuit_color, linestyle='--',
                                     lw=2, label='Circuit Connection')
        residual_line = mlines.Line2D([], [], color=self.residual_color, linestyle='--',
                                      lw=1.5, label='Residual Connection')

        plt.legend(handles=[head_patch, mlp_patch, circuit_line, residual_line],
                   loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

        plt.tight_layout()

        # Save or show the figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to {save_path}")

        return fig


def sample_visualization():
    """Create a sample visualization to demonstrate the usage."""
    # Create visualizer
    viz = TransformerVisualizer(num_layers=2, num_heads=4)

    # Early training phase
    head_attr_early = [
        [0.2, 0.4, 0.3, 0.2],  # Layer 0 heads
        [0.1, 0.2, 0.3, 0.1]  # Layer 1 heads
    ]
    mlp_act_early = [0.2, 0.3]
    circuits_early = [
        [0, 1, 1, 2, 0.3]  # Simple circuit from L0H1 to L1H2
    ]

    # Middle training phase
    head_attr_middle = [
        [0.3, 0.7, 0.4, 0.3],  # Layer 0 heads
        [0.2, 0.4, 0.7, 0.3]  # Layer 1 heads
    ]
    mlp_act_middle = [0.5, 0.6]
    circuits_middle = [
        [0, 1, 1, 2, 0.6],
        [0, 2, 1, 0, 0.4]
    ]

    # Late training phase
    head_attr_late = [
        [0.4, 0.9, 0.6, 0.4],  # Layer 0 heads
        [0.3, 0.5, 0.9, 0.6]  # Layer 1 heads
    ]
    mlp_act_late = [0.8, 0.9]
    circuits_late = [
        [0, 1, 1, 2, 0.9],
        [0, 2, 1, 0, 0.7],
        [0, 0, 1, 3, 0.5]
    ]

    # Create visualizations
    viz.visualize(
        head_attributions=head_attr_early,
        mlp_activations=mlp_act_early,
        circuits=circuits_early,
        save_path="transformer_early_phase.svg",
        title="Transformer - Early Training Phase"
    )

    viz.visualize(
        head_attributions=head_attr_middle,
        mlp_activations=mlp_act_middle,
        circuits=circuits_middle,
        save_path="transformer_middle_phase.svg",
        title="Transformer - Middle Training Phase"
    )

    viz.visualize(
        head_attributions=head_attr_late,
        mlp_activations=mlp_act_late,
        circuits=circuits_late,
        save_path="transformer_late_phase.svg",
        title="Transformer - Late Training Phase"
    )


def visualize_from_analyzer_data(model, epoch, jump_analyzer, weight_tracker, eval_loader):
    """
    Create a visualization using real data from your analyzer classes.

    Args:
        model: Your transformer model
        epoch: Current training epoch
        jump_analyzer: JumpAnalysisManager instance
        weight_tracker: EnhancedWeightSpaceTracker instance
        eval_loader: Evaluation data loader

    Returns:
        matplotlib figure object
    """
    # 1. Get model configuration
    num_layers = model.num_layers
    num_heads = model.num_heads

    # 2. Determine training phase
    if hasattr(model.logger, 'logs') and 'grokking_phases' in model.logger.logs:
        grokking_step = model.logger.logs['grokking_phases'].get('grokking_step')
        if grokking_step:
            if epoch < grokking_step - 100:
                phase = "Early"
            elif epoch > grokking_step + 100:
                phase = "Late"
            else:
                phase = "Middle (Grokking Transition)"
        else:
            # Fallback if grokking point not identified
            phase = "Current Training"
    else:
        phase = "Current Training"

    # 3. Get head attributions
    head_attributions = []
    attribution_results = model.analyze_head_attribution(eval_loader)

    for layer_idx in range(num_layers):
        layer_heads = []
        for head_idx in range(num_heads):
            head_key = f'layer_{layer_idx}_head_{head_idx}'
            attribution = attribution_results.get(head_key, 0.5)
            # Normalize if needed (typically attribution scores are small differences)
            normalized_attr = min(1.0, max(0.1, attribution * 10))
            layer_heads.append(normalized_attr)
        head_attributions.append(layer_heads)

    # 4. Get MLP activations
    mlp_activations = []
    for layer_idx in range(num_layers):
        # Check if MLP activations are tracked in weight_tracker
        mlp_key = f'layer_{layer_idx}_mlp_combined_norm'
        if hasattr(weight_tracker, 'train_history') and mlp_key in weight_tracker.train_history:
            # Get the most recent activation
            activation = weight_tracker.train_history[mlp_key][-1]
            # Normalize
            normalized_act = min(1.0, max(0.1, activation / 5.0))
        else:
            # Fallback value
            normalized_act = 0.5
        mlp_activations.append(normalized_act)

    # 5. Get circuits
    circuits = []
    # Try to get circuits from jump_analyzer if available
    if hasattr(jump_analyzer, 'circuit_analyzer'):
        circuit_results = jump_analyzer.circuit_analyzer.identify_circuits(eval_loader)
        if 'circuits' in circuit_results:
            for circuit_key, circuit_data in circuit_results['circuits'].items():
                # Parse the circuit key (format could be "layer_0_head_1+layer_1_head_2")
                parts = circuit_key.split('+')
                if len(parts) == 2:
                    src = parts[0].split('_')
                    tgt = parts[1].split('_')

                    if len(src) >= 4 and len(tgt) >= 4:
                        src_layer = int(src[1])
                        src_head = int(src[3])
                        tgt_layer = int(tgt[1])
                        tgt_head = int(tgt[3])

                        strength = circuit_data.get('circuit_strength', 0.5)
                        # Normalize strength
                        normalized_strength = min(1.0, max(0.1, strength * 5))

                        circuits.append([src_layer, src_head, tgt_layer, tgt_head, normalized_strength])

    # Create visualizer and generate visualization
    viz = TransformerVisualizer(num_layers=num_layers, num_heads=num_heads)

    title = f"Transformer Model Analysis - {phase} Phase (Epoch {epoch})"
    save_path = f"transformer_visualization_epoch_{epoch}.svg"

    fig = viz.visualize(
        head_attributions=head_attributions,
        mlp_activations=mlp_activations,
        circuits=circuits,
        save_path=save_path,
        title=title
    )

    return fig, save_path


if __name__ == "__main__":
    # Run the sample visualization
    sample_visualization()