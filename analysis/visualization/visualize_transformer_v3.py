import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

class AnthropicTransformerViz:
    """
    Creates clean, bottom-up Transformer visualizations in the style of
    Anthropic's circuit blog posts.
    """

    def __init__(self, num_layers=2, num_heads=4, figsize=None):
        """
        Initialize the visualizer.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            figsize: Custom figure size (optional)
        """
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Figure sizing
        self.figsize = figsize or (10, 6 + 2 * num_layers)
        self.scale = 1.0  # Scale factor for all elements

        # Element sizing
        self.box_width = 1.5 * self.scale
        self.box_height = 0.6 * self.scale
        self.head_width = 0.7 * self.scale
        self.head_height = 0.7 * self.scale
        self.add_circle_radius = 0.25 * self.scale
        self.vertical_spacing = 1.2 * self.scale
        self.head_spacing = 0.9 * self.scale

        # Visual style
        self.colors = {
            'token': '#E8E8E8',        # Light gray
            'embed': '#E6CCAF',        # Light brown
            'attention_head': '#E6CCAF', # Light brown
            'mlp': '#E6CCAF',          # Light brown
            'unembed': '#E6CCAF',      # Light brown
            'logits': '#E8E8E8',       # Light gray
            'add': '#D4D4D4',          # Light gray for add circles
            'arrow': '#888888',        # Dark gray
            'text': '#333333',         # Dark gray text
            'background': '#FFFFFF'    # White background
        }

        # For head intensity variations based on attribution/activity
        self.head_alphas = {i: 1.0 for i in range(num_heads)}
        self.mlp_alpha = 1.0

        # For circuit connections
        self.circuit_connections = []  # (from_idx, to_idx, strength)

        # Store component positions for connections
        self.component_positions = {}

    def _create_figure(self):
        """Create the base figure."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_facecolor(self.colors['background'])
        ax.axis('off')

        # Set reasonable bounds
        max_width = max(8, self.num_heads * self.head_spacing)
        ax.set_xlim(-1, max_width + 1)
        ax.set_ylim(-1, self.vertical_spacing * (5 + self.num_layers) + 1)

        return fig, ax

    def _draw_box(self, ax, x, y, width, height, color, alpha=1.0,
                 label=None, edge_color='#000000', linewidth=1.0, zorder=5):
        """Draw a rectangular box with optional label."""
        rect = patches.Rectangle(
            (x - width/2, y - height/2),
            width, height,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder
        )
        ax.add_patch(rect)

        if label:
            ax.text(x, y, label, color=self.colors['text'],
                   ha='center', va='center', fontsize=10 * self.scale,
                   fontweight='normal', zorder=zorder+1)

        return (x, y)

    def _draw_add_circle(self, ax, x, y, radius=None, zorder=5):
        """Draw an addition circle with a plus sign."""
        if radius is None:
            radius = self.add_circle_radius

        circle = patches.Circle(
            (x, y),
            radius,
            facecolor=self.colors['add'],
            edgecolor='#000000',
            linewidth=1.0,
            zorder=zorder
        )
        ax.add_patch(circle)

        # Add the plus sign
        ax.text(x, y, '+', color=self.colors['text'],
               ha='center', va='center', fontsize=12 * self.scale,
               fontweight='bold', zorder=zorder+1)

        return (x, y)

    def _draw_arrow(self, ax, x1, y1, x2, y2, zorder=3):
        """Draw a vertical or horizontal arrow."""
        ax.arrow(
            x1, y1, x2-x1, y2-y1,
            head_width=0.1 * self.scale,
            head_length=0.15 * self.scale,
            fc=self.colors['arrow'],
            ec=self.colors['arrow'],
            length_includes_head=True,
            linewidth=1.0,
            zorder=zorder
        )

    def _draw_dashed_line(self, ax, x1, y1, x2, y2, zorder=2):
        """Draw a dashed line to represent skip connections."""
        ax.plot(
            [x1, x2], [y1, y2],
            color=self.colors['arrow'],
            linestyle='--',
            linewidth=1.0,
            zorder=zorder
        )

    def _draw_residual_stream(self, ax, x, y1, y2, label=None, zorder=1):
        """Draw a vertical line representing the residual stream."""
        # Dotted vertical line
        ax.plot(
            [x, x], [y1, y2],
            color=self.colors['arrow'],
            linestyle=':',
            linewidth=1.0,
            zorder=zorder
        )

        # Optional label
        if label:
            midpoint_y = (y1 + y2) / 2
            ax.text(x + 0.2, midpoint_y, label,
                   color=self.colors['text'],
                   ha='left', va='center',
                   fontsize=8 * self.scale,
                   rotation=90)

    def _draw_circuit_connection(self, ax, from_pos, to_pos, strength=1.0, color='red', zorder=10):
        """Draw a circuit connection between components."""
        # Adjust line width based on strength
        linewidth = 1.0 + strength * 2.0
        alpha = min(0.8, 0.3 + strength * 0.5)

        # Draw curved line
        ax.plot(
            [from_pos[0], to_pos[0]],
            [from_pos[1], to_pos[1]],
            color=color,
            linestyle='-',
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder
        )

    def _label_residual_state(self, ax, x, y, text, zorder=6):
        """Add a label for the residual stream state."""
        ax.text(
            x + 0.3, y,
            text,
            color=self.colors['text'],
            fontsize=9 * self.scale,
            ha='left',
            va='center',
            fontweight='normal',
            fontstyle='italic',
            zorder=zorder
        )

    def set_head_activations(self, activations):
        """
        Set activation levels for attention heads.

        Args:
            activations: List of activation values for heads
        """
        for i, act in enumerate(activations):
            if i < len(self.head_alphas):
                # Map activation to alpha range 0.3-1.0
                self.head_alphas[i] = 0.3 + min(0.7, act * 0.7)

    def set_mlp_activation(self, activation):
        """
        Set activation level for MLP block.

        Args:
            activation: Activation value for MLP
        """
        # Map activation to alpha range 0.3-1.0
        self.mlp_alpha = 0.3 + min(0.7, activation * 0.7)

    def add_circuit_connection(self, from_head_idx, to_component, strength=1.0):
        """
        Add a circuit connection to be visualized.

        Args:
            from_head_idx: Index of the source attention head
            to_component: Target component ('mlp' or head index)
            strength: Connection strength (0.0-1.0)
        """
        self.circuit_connections.append((from_head_idx, to_component, strength))

    def draw_one_layer_transformer(self, ax, base_y=0, highlight_circuits=True):
        """
        Draw a one-layer transformer in Anthropic blog style.

        Args:
            ax: Matplotlib axis
            base_y: Base Y position for drawing
            highlight_circuits: Whether to draw circuit connections

        Returns:
            Dictionary of component positions
        """
        # Starting position
        center_x = 4 if self.num_heads <= 4 else self.num_heads * 0.7
        token_y = base_y + 1 * self.vertical_spacing
        embed_y = base_y + 2 * self.vertical_spacing

        # Calculate positions for the residual stream stages
        x0_y = base_y + 3 * self.vertical_spacing  # After embedding
        xi_y = base_y + 4 * self.vertical_spacing  # Before attention
        xi1_y = base_y + 5 * self.vertical_spacing  # After attention, before MLP
        xi2_y = base_y + 6 * self.vertical_spacing  # After MLP
        x_final_y = base_y + 7 * self.vertical_spacing  # Final output
        unembed_y = base_y + 8 * self.vertical_spacing
        logits_y = base_y + 9 * self.vertical_spacing

        # Draw tokens box
        token_pos = self._draw_box(
            ax, center_x, token_y,
            self.box_width, self.box_height,
            self.colors['token'], label="tokens"
        )
        self.component_positions['tokens'] = token_pos

        # Draw embedding box
        self._draw_arrow(ax, center_x, token_y + self.box_height/2,
                        center_x, embed_y - self.box_height/2)

        embed_pos = self._draw_box(
            ax, center_x, embed_y,
            self.box_width, self.box_height,
            self.colors['embed'], label="embed"
        )
        self.component_positions['embed'] = embed_pos

        # Draw residual stream line and states
        stream_x = center_x
        self._draw_residual_stream(ax, stream_x, x0_y, x_final_y)

        # Draw x0 (after embedding)
        self._draw_arrow(ax, center_x, embed_y + self.box_height/2,
                        center_x, x0_y)
        self._label_residual_state(ax, center_x, x0_y, "x₀")

        # Calculate attention heads position
        head_block_width = self.num_heads * self.head_spacing
        head_block_center = center_x
        head_start_x = head_block_center - (head_block_width - self.head_spacing) / 2

        # Draw attention heads
        heads_pos = []
        for i in range(self.num_heads):
            head_x = head_start_x + i * self.head_spacing
            head_y = xi_y

            # Connect from residual stream to head
            self._draw_dashed_line(ax, center_x, xi_y, head_x, xi_y)

            # Draw the head
            head_pos = self._draw_box(
                ax, head_x, head_y,
                self.head_width, self.head_height,
                self.colors['attention_head'],
                alpha=self.head_alphas.get(i, 1.0),
                label=f"h{i}"
            )
            heads_pos.append(head_pos)
            self.component_positions[f'head_{i}'] = head_pos

        # Draw xi state (before attention)
        self._draw_arrow(ax, center_x, x0_y, center_x, xi_y)
        self._label_residual_state(ax, center_x, xi_y, "xᵢ")

        # Draw addition circle after attention
        add1_pos = self._draw_add_circle(ax, center_x, xi1_y)
        self.component_positions['add1'] = add1_pos

        # Connect heads to addition circle
        for i, head_pos in enumerate(heads_pos):
            # Curved line from head to add circle
            head_x, head_y = head_pos
            add_x, add_y = add1_pos

            # Simple direct line
            ax.plot(
                [head_x, add_x], [head_y, add_y],
                color=self.colors['arrow'],
                linestyle='-',
                linewidth=1.0,
                zorder=4
            )

        # Connect first residual point to addition
        self._draw_arrow(ax, center_x, xi_y, center_x, xi1_y - self.add_circle_radius)

        # Label xi+1 state (after attention, before MLP)
        self._label_residual_state(ax, center_x, xi1_y, "xᵢ₊₁")

        # Draw MLP
        mlp_y = (xi1_y + xi2_y) / 2

        # Draw arrow from add to MLP
        self._draw_arrow(ax, center_x, xi1_y + self.add_circle_radius,
                        center_x - self.box_width/2, mlp_y)

        # Draw MLP box
        mlp_pos = self._draw_box(
            ax, center_x, mlp_y,
            self.box_width, self.box_height,
            self.colors['mlp'],
            alpha=self.mlp_alpha,
            label="MLP m"
        )
        self.component_positions['mlp'] = mlp_pos

        # Draw second addition circle
        add2_pos = self._draw_add_circle(ax, center_x, xi2_y)
        self.component_positions['add2'] = add2_pos

        # Connect MLP to second add
        self._draw_arrow(ax, center_x + self.box_width/2, mlp_y,
                        center_x, xi2_y - self.add_circle_radius)

        # Connect first add to second add (skip connection)
        self._draw_dashed_line(ax, center_x, xi1_y + self.add_circle_radius,
                              center_x, xi2_y - self.add_circle_radius)

        # Label xi+2 state (after MLP)
        self._label_residual_state(ax, center_x, xi2_y, "xᵢ₊₂")

        # Draw final state (x_{-1})
        self._draw_arrow(ax, center_x, xi2_y + self.add_circle_radius,
                        center_x, x_final_y)
        self._label_residual_state(ax, center_x, x_final_y, "x₋₁")

        # Draw unembedding
        self._draw_arrow(ax, center_x, x_final_y,
                        center_x, unembed_y - self.box_height/2)

        unembed_pos = self._draw_box(
            ax, center_x, unembed_y,
            self.box_width, self.box_height,
            self.colors['unembed'], label="unembed"
        )
        self.component_positions['unembed'] = unembed_pos

        # Draw logits
        self._draw_arrow(ax, center_x, unembed_y + self.box_height/2,
                        center_x, logits_y - self.box_height/2)

        logits_pos = self._draw_box(
            ax, center_x, logits_y,
            self.box_width, self.box_height,
            self.colors['logits'], label="logits"
        )
        self.component_positions['logits'] = logits_pos

        # Draw circuit connections if requested
        if highlight_circuits and self.circuit_connections:
            for from_idx, to_component, strength in self.circuit_connections:
                if f'head_{from_idx}' in self.component_positions:
                    from_pos = self.component_positions[f'head_{from_idx}']

                    # Determine target position
                    if to_component == 'mlp':
                        to_pos = self.component_positions['mlp']
                    elif isinstance(to_component, int) and f'head_{to_component}' in self.component_positions:
                        to_pos = self.component_positions[f'head_{to_component}']
                    else:
                        continue

                    # Draw the connection
                    self._draw_circuit_connection(ax, from_pos, to_pos, strength)

        # Add residual block annotation
        block_height = xi2_y - xi_y
        ax.text(
            center_x + 3.5, (xi_y + xi2_y) / 2,
            "One\nresidual\nblock",
            color=self.colors['text'],
            fontsize=10 * self.scale,
            ha='left',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5',
                     edgecolor=self.colors['arrow'])
        )

        return self.component_positions

    def draw_multi_layer_transformer(self, highlight_circuits=True, save_path=None, title=None):
        """
        Draw a multi-layer transformer in Anthropic blog style.

        Args:
            highlight_circuits: Whether to draw circuit connections
            save_path: Path to save the visualization
            title: Optional title for the visualization

        Returns:
            Matplotlib figure
        """
        fig, ax = self._create_figure()

        if title:
            ax.text(
                0.5, 0.98,
                title,
                transform=ax.transAxes,
                fontsize=14 * self.scale,
                fontweight='bold',
                ha='center',
                va='top'
            )

        # For single layer, use the specialized function
        if self.num_layers == 1:
            self.draw_one_layer_transformer(ax, base_y=0, highlight_circuits=highlight_circuits)
        else:
            # TODO: Implement multi-layer visualization
            # This would extend the one-layer approach to stack multiple layers
            pass

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        return fig

def sample_usage():
    """Demonstrate usage of the Anthropic-style visualization."""
    # Create visualizer
    visualizer = AnthropicTransformerViz(num_layers=1, num_heads=3)

    # Set head activations (0.0-1.0 range)
    visualizer.set_head_activations([0.3, 0.9, 0.5, 0.2])

    # Set MLP activation
    visualizer.set_mlp_activation(0.7)

    # Add some circuit connections
    visualizer.add_circuit_connection(from_head_idx=1, to_component='mlp', strength=0.8)
    visualizer.add_circuit_connection(from_head_idx=2, to_component=3, strength=0.5)

    # Draw and save
    visualizer.draw_multi_layer_transformer(
        highlight_circuits=True,
        save_path="anthropic_style_transformer.png",
        title="Transformer Architecture with Circuit Connections"
    )

if __name__ == "__main__":
    sample_usage()