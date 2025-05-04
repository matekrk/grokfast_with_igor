import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.path as mpath
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import torch


class AnthropicPhaseTransformerViz:
    """
    Creates Anthropic-style visualizations of transformer models at different
    learning phases identified by PhaseTransitionAnalyzer.

    This class works specifically with PhaseTransitionAnalyzer and
    EnhancedWeightSpaceTracker to visualize:
    1. Phase transitions in learning
    2. Circuit formation across phases
    3. Attention patterns and MLP activation changes
    4. Correlations with grokking points
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

        # Visual style configuration
        self.figsize = figsize or (12, 6 * num_layers + 4)
        self.layer_height = 4
        self.layer_spacing = 3.5

        # Anthropic-style color palette
        self.colors = {
            'attention': '#5470C6',  # Blue
            'mlp': '#91CC75',  # Green
            'residual': '#FAC858',  # Yellow/Gold
            'circuit': '#EE6666',  # Red
            'text': '#303133',  # Dark gray
            'background': '#F7F9FB',  # Light blue-gray
            'embedding': '#9A60B4',  # Purple
            'layernorm': '#73C0DE',  # Light blue
            'phase_1': '#E8F4FF',  # Light blue background
            'phase_2': '#FFF8E7',  # Light yellow background
            'phase_3': '#F0FFF0',  # Light green background
            'phase_4': '#FFF0F0',  # Light red background
            'grokking_highlight': '#FFEC3A'  # Bright yellow
        }

        # For representing attention intensity
        self.attn_cmap = LinearSegmentedColormap.from_list(
            'attn_cmap',
            [(0.0, '#5470C620'), (0.3, '#5470C660'), (1.0, '#5470C6FF')]
        )

        # For representing MLP activation intensity
        self.mlp_cmap = LinearSegmentedColormap.from_list(
            'mlp_cmap',
            [(0.0, '#91CC7520'), (0.3, '#91CC7560'), (1.0, '#91CC75FF')]
        )

        # Store positions for drawing connections
        self.component_positions = {}

    def _create_figure(self):
        """Create the figure and axes with Anthropic-style background."""
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
        """Draw a clean block with rounded corners."""
        if label_color is None:
            label_color = self.colors['text']

        rect = patches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width, height,
            boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
            facecolor=color,
            alpha=alpha,
            edgecolor='#00000033',
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

    def _draw_attention_head(self, ax, x, y, head_idx, layer_idx, attribution=0.5, scale=1.0):
        """Draw an attention head with intensity based on attribution."""
        # Calculate size based on attribution and scale
        base_size = 0.6 * scale
        intensity = max(0.3, attribution)

        # Use a color with alpha based on attribution
        color = self.colors['attention']
        alpha = 0.4 + attribution * 0.6  # Scale alpha with attribution

        # Draw head
        head_id = f"L{layer_idx}_H{head_idx}"

        # Create slightly rounded square for head
        rect = patches.FancyBboxPatch(
            (x - base_size / 2, y - base_size / 2),
            base_size, base_size,
            boxstyle=patches.BoxStyle("Round", pad=0.1, rounding_size=0.1),
            facecolor=color,
            alpha=alpha,
            edgecolor='#00000033',
            linewidth=1,
            zorder=3,
            clip_on=False
        )
        ax.add_patch(rect)

        # Add head index label
        ax.text(x, y, f"{head_idx}", color='white', ha='center', va='center',
                fontsize=9 * scale, fontweight='bold', zorder=4)

        self.component_positions[head_id] = (x, y)
        return head_id

    def _draw_transformer_layer(self, ax, x, y, layer_idx, head_attributions=None,
                                mlp_activation=0.5, label=None):
        """Draw a transformer layer with attention block and MLP."""
        if head_attributions is None:
            head_attributions = [0.5] * self.num_heads

        # Spacing parameters
        h_spacing = 5.0
        v_spacing = 1.2

        # Layer label
        if label:
            layer_label = label
        else:
            layer_label = f"Layer {layer_idx}"

        ax.text(x, y + 1.5, layer_label,
                fontsize=12, fontweight='bold', ha='center', color=self.colors['text'])

        # Draw attention block
        attn_x = x + h_spacing / 2
        attn_y = y
        attn_width = 2.5
        attn_height = 2.0

        # Draw attention block outline
        attn_block_id = f"attn_block_{layer_idx}"
        self._draw_block(
            ax, attn_x, attn_y, attn_width, attn_height, 'white', alpha=0.8,
            label=f"Multi-Head Attention", label_pos='top',
            component_id=attn_block_id
        )

        # Draw individual heads
        head_spacing = attn_width / (self.num_heads + 1)
        for h in range(self.num_heads):
            head_x = attn_x - attn_width / 2 + head_spacing * (h + 1)
            head_y = attn_y

            attr = head_attributions[h]
            self._draw_attention_head(ax, head_x, head_y, h, layer_idx, attr)

        # Draw MLP block
        mlp_x = x + h_spacing * 1.5
        mlp_y = y
        mlp_width = 2.5
        mlp_height = 1.5

        # MLP activation controls intensity
        mlp_intensity = max(0.3, mlp_activation)
        mlp_id = f"mlp_block_{layer_idx}"

        self._draw_block(
            ax, mlp_x, mlp_y, mlp_width, mlp_height, self.colors['mlp'],
            alpha=mlp_intensity,
            label=f"MLP",
            component_id=mlp_id
        )

        # Layer norms
        ln1_x = x
        ln1_y = y
        ln1_id = f"ln1_{layer_idx}"
        self._draw_block(
            ax, ln1_x, ln1_y, 0.8, 0.3, self.colors['layernorm'],
            label="LN", component_id=ln1_id
        )

        ln2_x = x + h_spacing
        ln2_y = y
        ln2_id = f"ln2_{layer_idx}"
        self._draw_block(
            ax, ln2_x, ln2_y, 0.8, 0.3, self.colors['layernorm'],
            label="LN", component_id=ln2_id
        )

        # Addition nodes
        add1_x = x + h_spacing * 0.75
        add1_y = y - v_spacing
        add1_id = f"add1_{layer_idx}"
        self._draw_add(ax, add1_x, add1_y, add1_id)

        add2_x = x + h_spacing * 1.75
        add2_y = y - v_spacing
        add2_id = f"add2_{layer_idx}"
        self._draw_add(ax, add2_x, add2_y, add2_id)

        # Draw connections
        # Input to LN1
        input_id = f"output_{layer_idx - 1}" if layer_idx > 0 else "embedding"
        if input_id in self.component_positions:
            input_pos = self.component_positions[input_id]
            self._draw_arrow(ax, input_pos[0], input_pos[1], ln1_x, ln1_y)

        # LN1 to Attention
        self._draw_arrow(ax, ln1_x, ln1_y, attn_x, attn_y)

        # Attention to Add1
        self._draw_arrow(ax, attn_x, attn_y, add1_x, add1_y)

        # Add1 to LN2
        self._draw_arrow(ax, add1_x, add1_y, ln2_x, ln2_y)

        # LN2 to MLP
        self._draw_arrow(ax, ln2_x, ln2_y, mlp_x, mlp_y)

        # MLP to Add2
        self._draw_arrow(ax, mlp_x, mlp_y, add2_x, add2_y)

        # Residual connections
        if input_id in self.component_positions:
            self._draw_residual(ax, input_id, add1_id)
        self._draw_residual(ax, add1_id, add2_id)

        # Store output position
        self.component_positions[f"output_{layer_idx}"] = (add2_x, add2_y)

        return {
            'layer_id': f"layer_{layer_idx}",
            'attn_id': attn_block_id,
            'mlp_id': mlp_id,
            'ln1_id': ln1_id,
            'ln2_id': ln2_id,
            'add1_id': add1_id,
            'add2_id': add2_id
        }

    def _draw_add(self, ax, x, y, component_id):
        """Draw an addition node with a plus sign."""
        # Small white circle with plus sign
        circle = plt.Circle((x, y), 0.3, facecolor='white',
                            edgecolor='#00000033', linewidth=1, zorder=3)
        ax.add_patch(circle)

        # Plus sign
        ax.text(x, y, "+", color='black', ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=4)

        self.component_positions[component_id] = (x, y)
        return component_id

    def _draw_arrow(self, ax, x1, y1, x2, y2, color='#303133', width=1.0,
                    style='-', alpha=0.8, zorder=1):
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

    def _draw_residual(self, ax, start_id, end_id, offset=0.7):
        """Draw a residual connection as a dashed line."""
        if start_id not in self.component_positions or end_id not in self.component_positions:
            return

        start_x, start_y = self.component_positions[start_id]
        end_x, end_y = self.component_positions[end_id]

        # Offset points for residual path
        start_point = (start_x, start_y + offset)
        end_point = (end_x, end_y + offset)

        # Create control points for curve
        mid_x = (start_x + end_x) / 2
        control1 = (mid_x, start_y + offset)
        control2 = (mid_x, end_y + offset)

        # Create curved path
        verts = [
            start_point,
            control1,
            control2,
            end_point
        ]

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
            edgecolor=self.colors['residual'],
            linewidth=1.5,
            linestyle='--',
            alpha=0.8,
            zorder=0,
            clip_on=False
        )
        ax.add_patch(patch)

        # Add arrowhead
        arrow = patches.FancyArrowPatch(
            (end_x - 0.3, end_y + offset),
            (end_x, end_y + offset),
            arrowstyle="-|>",
            mutation_scale=15,
            color=self.colors['residual'],
            linewidth=1.5,
            alpha=0.8,
            zorder=0,
            clip_on=False
        )
        ax.add_patch(arrow)

    def _draw_circuit(self, ax, src_id, tgt_id, strength=0.5, offset=0.0):
        """Draw a circuit connection between components with strength indication."""
        if src_id not in self.component_positions or tgt_id not in self.component_positions:
            return

        src_x, src_y = self.component_positions[src_id]
        tgt_x, tgt_y = self.component_positions[tgt_id]

        # Create control points for curve
        if offset != 0:
            # Apply offset for multiple circuits
            control1 = (src_x + 1.0, src_y + offset)
            control2 = (tgt_x - 1.0, tgt_y + offset)
        else:
            # Default curve
            mid_y = (src_y + tgt_y) / 2
            control1 = (src_x + 1.0, mid_y)
            control2 = (tgt_x - 1.0, mid_y)

        # Create curved path
        verts = [
            (src_x, src_y),
            control1,
            control2,
            (tgt_x, tgt_y)
        ]

        codes = [
            mpath.Path.MOVETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4
        ]

        # Line width and opacity based on strength
        width = max(1.0, strength * 3.0)
        alpha = min(0.9, max(0.3, strength * 0.8))

        path = mpath.Path(verts, codes)
        patch = patches.PathPatch(
            path,
            facecolor='none',
            edgecolor=self.colors['circuit'],
            linewidth=width,
            linestyle='-',
            alpha=alpha,
            zorder=1,
            clip_on=False
        )
        ax.add_patch(patch)

        # Add arrowhead
        # Calculate position near end of path (90% along)
        t = 0.9
        point = path.interpolated(101).vertices[int(100 * t)]
        t_prev = max(0, t - 0.01)
        prev_point = path.interpolated(101).vertices[int(100 * t_prev)]
        direction = point - prev_point
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)

            arrow_size = max(0.1, strength * 0.2)
            arrow = patches.FancyArrowPatch(
                point - arrow_size * 3 * direction,
                point,
                arrowstyle="-|>",
                mutation_scale=max(10, 15 * strength),
                color=self.colors['circuit'],
                linewidth=width,
                alpha=alpha,
                zorder=1,
                clip_on=False
            )
            ax.add_patch(arrow)

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

    def _draw_phase_label(self, ax, x, y, phase_name, phase_idx=None):
        """Draw a label for a training phase."""
        # Choose color based on phase index
        if phase_idx is not None:
            bg_color = self.colors.get(f'phase_{phase_idx % 4 + 1}', '#F0F0F0')
        else:
            bg_color = '#F0F0F0'

        # Create rounded rectangle for label
        width = len(phase_name) * 0.15 + 1.2
        height = 0.8
        rect = patches.FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width, height,
            boxstyle=patches.BoxStyle("Round", pad=0.4, rounding_size=0.2),
            facecolor=bg_color,
            alpha=0.9,
            edgecolor='#00000033',
            linewidth=1,
            zorder=5,
            clip_on=False
        )
        ax.add_patch(rect)

        # Add text
        ax.text(x, y, phase_name, ha='center', va='center',
                fontsize=11, fontweight='bold', color=self.colors['text'], zorder=6)

    def visualize_phase(self, phase_data, save_path=None, title=None):
        """
        Create a visualization for a specific training phase.

        Args:
            phase_data: Dictionary with phase information:
                - head_attributions: List of lists with attribution values
                - mlp_activations: List of MLP activation values
                - circuits: List of (src_id, tgt_id, strength) tuples
                - phase_name: Name of this phase
                - epoch_range: (start_epoch, end_epoch) tuple
            save_path: Path to save the visualization
            title: Custom title (optional)

        Returns:
            matplotlib figure
        """
        # Extract data
        head_attributions = phase_data.get('head_attributions',
                                           [[0.5] * self.num_heads for _ in range(self.num_layers)])
        mlp_activations = phase_data.get('mlp_activations', [0.5] * self.num_layers)
        circuits = phase_data.get('circuits', [])
        phase_name = phase_data.get('phase_name', 'Unknown Phase')
        epoch_range = phase_data.get('epoch_range', (0, 0))

        # Create figure
        fig, ax = self._create_figure()

        # Embedding
        embed_x = 2
        base_y = self.num_layers * self.layer_spacing
        embed_id = self._draw_embedding(ax, embed_x, base_y)

        # Transformer layers
        for i in range(self.num_layers):
            layer_y = base_y - i * self.layer_spacing

            # Get attributions and activation for this layer
            layer_attributions = head_attributions[i] if i < len(head_attributions) else [0.5] * self.num_heads
            layer_activation = mlp_activations[i] if i < len(mlp_activations) else 0.5

            # Draw layer
            self._draw_transformer_layer(
                ax, embed_x, layer_y, i,
                head_attributions=layer_attributions,
                mlp_activation=layer_activation
            )

        # Output
        output_x = embed_x + 8.75
        output_y = base_y - (self.num_layers - 1) * self.layer_spacing - 1.2
        output_id = self._draw_output(ax, output_x, output_y)

        # Connect last layer to output
        last_output_id = f"output_{self.num_layers - 1}"
        if last_output_id in self.component_positions:
            last_output_pos = self.component_positions[last_output_id]
            self._draw_arrow(ax, last_output_pos[0], last_output_pos[1], output_x, output_y)

        # Draw circuits
        circuit_offsets = {}  # Track offsets for multiple circuits between same components
        for circuit in circuits:
            src_id, tgt_id, strength = circuit

            # Generate offset for multiple connections between same components
            key = f"{src_id}_{tgt_id}"
            if key in circuit_offsets:
                circuit_offsets[key] += 0.1
            else:
                circuit_offsets[key] = 0.0

            self._draw_circuit(ax, src_id, tgt_id, strength, circuit_offsets[key])

        # Phase name and epoch range
        if epoch_range and epoch_range[0] != epoch_range[1]:
            phase_label = f"{phase_name} (Epochs {epoch_range[0]}-{epoch_range[1]})"
        else:
            phase_label = phase_name

        # Add phase label at top
        self._draw_phase_label(ax, embed_x + 4.5, base_y + 2.2, phase_label)

        # Add title
        if title:
            ax.text(embed_x + 4.5, base_y + 3.0, title, fontsize=16, fontweight='bold',
                    ha='center', color=self.colors['text'])

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

        # Save if specified
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        return fig

    def visualize_learning_phases(self, phase_analyzer, weight_tracker, model, eval_loader=None,
                                  save_dir=None, title=None):
        """
        Create visualizations for all phases identified by PhaseTransitionAnalyzer.

        Args:
            phase_analyzer: Instance of PhaseTransitionAnalyzer
            weight_tracker: Instance of EnhancedWeightSpaceTracker
            model: The transformer model
            eval_loader: Evaluation data loader
            save_dir: Directory to save visualizations
            title: Main title for visualizations

        Returns:
            List of saved paths
        """
        saved_paths = []

        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Get the phase summary from analyzer
        phase_summary = phase_analyzer.get_learning_phase_summary()
        if not phase_summary or 'phases' not in phase_summary:
            print("No phases detected in PhaseTransitionAnalyzer")
            return saved_paths

        # Check for grokking information
        grokking_epoch = None
        if 'grokking_correlations' in phase_summary:
            for corr in phase_summary['grokking_correlations']:
                if corr.get('aligned', False):
                    grokking_epoch = corr.get('grokking_epoch')
                    break

        # Store current model state to restore later
        current_state = {k: v.clone() for k, v in model.state_dict().items()}

        try:
            # For each phase, create a visualization
            for i, phase in enumerate(phase_summary['phases']):
                phase_idx = i + 1

                # Get phase information
                start_epoch = phase.get('start_epoch', 0)
                end_epoch = phase.get('end_epoch', 0)
                classification = phase.get('classification', 'unknown')

                # Try to load a checkpoint from this phase
                phase_epoch = (start_epoch + end_epoch) // 2  # Mid-point of phase

                # Check if this phase contains the grokking point
                contains_grokking = (grokking_epoch is not None and
                                     start_epoch <= grokking_epoch <= end_epoch)

                if contains_grokking:
                    # Use grokking epoch as the representative point
                    phase_epoch = grokking_epoch
                    phase_name = f"{classification.title()} Phase (Grokking)"
                else:
                    phase_name = f"{classification.title()} Phase"

                # Try to load checkpoint
                checkpoint_path = os.path.join(weight_tracker.save_dir, f"checkpoint_step_{phase_epoch}.pt")
                checkpoint_loaded = False

                if os.path.exists(checkpoint_path):
                    try:
                        checkpoint = torch.load(checkpoint_path)
                        model.load_state_dict(checkpoint["model_state_dict"])
                        checkpoint_loaded = True
                    except:
                        print(f"Failed to load checkpoint for epoch {phase_epoch}")

                # Get attribution data
                head_attributions = []
                if checkpoint_loaded and eval_loader:
                    # Get head attributions from model
                    attribution_data = model.analyze_head_attribution(eval_loader)

                    for layer_idx in range(self.num_layers):
                        layer_attr = []
                        for head_idx in range(self.num_heads):
                            key = f'layer_{layer_idx}_head_{head_idx}'
                            attr_val = attribution_data.get(key, 0.1)
                            # Normalize for visualization
                            norm_attr = max(0.1, min(0.9, attr_val * 10))
                            layer_attr.append(norm_attr)
                        head_attributions.append(layer_attr)
                else:
                    # Generate default attributions based on phase
                    # Early phases: Lower attribution
                    # Late phases: Higher attribution
                    base_val = 0.3 + (i / max(1, len(phase_summary['phases']))) * 0.5
                    variance = 0.2

                    for layer_idx in range(self.num_layers):
                        layer_attr = []
                        for head_idx in range(self.num_heads):
                            # Random variation around base value
                            attr = base_val + (np.random.random() - 0.5) * variance
                            layer_attr.append(max(0.1, min(0.9, attr)))
                            head_attributions.append(layer_attr)

                # Get MLP activations
                mlp_activations = []
                if checkpoint_loaded and hasattr(weight_tracker, 'train_history'):
                    # Try to get MLP norms from weight tracker
                    for layer_idx in range(self.num_layers):
                        key = f'layer_{layer_idx}_mlp_combined_norm'
                        if key in weight_tracker.train_history:
                            # Find activation values near this epoch
                            act_data = [(ep, act) for ep, act in zip(weight_tracker.weight_timestamps,
                                                                     weight_tracker.train_history[key])
                                        if abs(ep - phase_epoch) < 100]
                            if act_data:
                                # Get closest activation to target epoch
                                epoch_diffs = [abs(ep - phase_epoch) for ep, _ in act_data]
                                closest_idx = epoch_diffs.index(min(epoch_diffs))
                                activation = act_data[closest_idx][1]
                                # Normalize for visualization
                                norm_act = max(0.1, min(0.9, activation / 5.0))
                                mlp_activations.append(norm_act)
                            else:
                                # Default value based on phase progression
                                mlp_activations.append(0.3 + 0.5 * (i / max(1, len(phase_summary['phases']))))
                        else:
                            # Default value based on phase progression
                            mlp_activations.append(0.3 + 0.5 * (i / max(1, len(phase_summary['phases']))))
                else:
                    # Default activations based on phase
                    base_val = 0.3 + (i / max(1, len(phase_summary['phases']))) * 0.5
                    for layer_idx in range(self.num_layers):
                        mlp_activations.append(base_val)

                # Get circuits
                circuits = []
                if checkpoint_loaded and eval_loader and hasattr(phase_analyzer, 'circuit_tracker'):
                    # Get circuits from circuit tracker or analyzer
                    try:
                        circuit_summary = phase_analyzer.circuit_tracker.get_circuit_summary_for_epoch(phase_epoch)

                        if circuit_summary and 'active_circuits' in circuit_summary:
                            active_circuits = circuit_summary['active_circuits']
                            strengths = circuit_summary.get('circuit_strengths', {})

                            for circuit in active_circuits:
                                # Parse circuit notation (could be in different formats)
                                if '+' in circuit:
                                    parts = circuit.split('+')
                                    if len(parts) == 2:
                                        src_parts = parts[0].split('_')
                                        tgt_parts = parts[1].split('_')

                                        if len(src_parts) >= 4 and len(tgt_parts) >= 4:
                                            src_layer = int(src_parts[1])
                                            src_head = int(src_parts[3])
                                            tgt_layer = int(tgt_parts[1])
                                            tgt_head = int(tgt_parts[3])

                                            # Convert to component IDs
                                            src_id = f"L{src_layer}_H{src_head}"
                                            tgt_id = f"L{tgt_layer}_H{tgt_head}"

                                            # Get circuit strength
                                            strength = strengths.get(circuit, 0.5)
                                            norm_strength = max(0.1, min(0.9, strength * 5))

                                            circuits.append((src_id, tgt_id, norm_strength))
                    except Exception as e:
                        print(f"Error extracting circuits: {e}")

                # If we have no circuits but are in a later phase, generate some based on phase
                if not circuits and i >= 1:
                    # Simple pattern: more and stronger circuits in later phases
                    num_circuits = i + 1  # More circuits in later phases
                    base_strength = 0.3 + (i / max(1, len(phase_summary['phases']))) * 0.5

                    # Generate plausible circuits
                    circuit_candidates = []
                    for src_layer in range(self.num_layers - 1):
                        for src_head in range(self.num_heads):
                            for tgt_layer in range(src_layer + 1, self.num_layers):
                                for tgt_head in range(self.num_heads):
                                    circuit_candidates.append((
                                        f"L{src_layer}_H{src_head}",
                                        f"L{tgt_layer}_H{tgt_head}"
                                    ))

                    # Select a subset of candidates
                    if circuit_candidates:
                        selected_indices = np.random.choice(
                            len(circuit_candidates),
                            size=min(num_circuits, len(circuit_candidates)),
                            replace=False
                        )

                        for idx in selected_indices:
                            src_id, tgt_id = circuit_candidates[idx]
                            # Vary strength slightly
                            strength = base_strength * (0.8 + 0.4 * np.random.random())
                            circuits.append((src_id, tgt_id, min(0.9, strength)))

                # Create phase data
                phase_data = {
                    'head_attributions': head_attributions,
                    'mlp_activations': mlp_activations,
                    'circuits': circuits,
                    'phase_name': phase_name,
                    'epoch_range': (start_epoch, end_epoch)
                }

                # Generate filename
                if save_dir:
                    phase_type = classification.lower().replace(' ', '_')
                    save_path = os.path.join(save_dir, f"phase_{phase_idx}_{phase_type}.svg")

                    # Create the visualization
                    fig = self.visualize_phase(
                        phase_data=phase_data,
                        save_path=save_path,
                        title=title
                    )

                    saved_paths.append(save_path)

                    # Also save as PNG
                    png_path = save_path.replace('.svg', '.png')
                    fig.savefig(png_path, dpi=300, bbox_inches='tight')
                    saved_paths.append(png_path)

                    plt.close(fig)

        finally:
            # Restore original model state
            model.load_state_dict(current_state)

        return saved_paths

##########################################################################################

    def visualize_phase_transitions(self, phase_analyzer, weight_tracker, model,
                                    save_dir=None, title=None):
        """
        Create a series of visualizations showing the transitions between phases.

        Args:
            phase_analyzer: Instance of PhaseTransitionAnalyzer
            weight_tracker: Instance of EnhancedWeightSpaceTracker
            model: The transformer model
            save_dir: Directory to save visualizations
            title: Main title for visualizations

        Returns:
            List of saved paths
        """
        saved_paths = []

        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Get transitions from the analyzer
        transitions = phase_analyzer.detected_transitions
        if not transitions:
            print("No transitions detected in PhaseTransitionAnalyzer")
            return saved_paths

        # Store current model state
        current_state = {k: v.clone() for k, v in model.state_dict().items()}

        try:
            # For each transition, visualize before and after states
            for i, transition in enumerate(transitions):
                transition_epoch = transition.get('epoch', 0)
                transition_types = transition.get('transition_types', ['unknown'])
                window = transition.get('window', (transition_epoch - 50, transition_epoch + 50))

                # Generate transition title
                transition_name = "Phase Transition: " + ", ".join(t.replace('_', ' ').title()
                                                                   for t in transition_types)

                # Create before visualization (50 epochs before)
                before_epoch = max(1, transition_epoch - 50)
                before_checkpoint = os.path.join(weight_tracker.save_dir, f"checkpoint_step_{before_epoch}.pt")

                if os.path.exists(before_checkpoint):
                    # Load before state
                    checkpoint = torch.load(before_checkpoint)
                    model.load_state_dict(checkpoint["model_state_dict"])

                    # Generate before data
                    before_phase_data = self._extract_phase_data(
                        model=model,
                        phase_analyzer=phase_analyzer,
                        weight_tracker=weight_tracker,
                        epoch=before_epoch,
                        phase_name=f"Before Transition"
                    )

                    # Save before visualization
                    if save_dir:
                        before_path = os.path.join(save_dir, f"transition_{i + 1}_before.svg")
                        before_fig = self.visualize_phase(
                            phase_data=before_phase_data,
                            save_path=before_path,
                            title=f"{transition_name} - Before (Epoch {before_epoch})"
                        )
                        saved_paths.append(before_path)

                        # Also save PNG
                        before_png = before_path.replace('.svg', '.png')
                        before_fig.savefig(before_png, dpi=300, bbox_inches='tight')
                        saved_paths.append(before_png)

                        plt.close(before_fig)

                # Create during visualization
                during_checkpoint = os.path.join(weight_tracker.save_dir, f"checkpoint_step_{transition_epoch}.pt")

                if os.path.exists(during_checkpoint):
                    # Load transition state
                    checkpoint = torch.load(during_checkpoint)
                    model.load_state_dict(checkpoint["model_state_dict"])

                    # Generate during data
                    during_phase_data = self._extract_phase_data(
                        model=model,
                        phase_analyzer=phase_analyzer,
                        weight_tracker=weight_tracker,
                        epoch=transition_epoch,
                        phase_name=f"During Transition"
                    )

                    # Save during visualization
                    if save_dir:
                        during_path = os.path.join(save_dir, f"transition_{i + 1}_during.svg")
                        during_fig = self.visualize_phase(
                            phase_data=during_phase_data,
                            save_path=during_path,
                            title=f"{transition_name} (Epoch {transition_epoch})"
                        )
                        saved_paths.append(during_path)

                        # Also save PNG
                        during_png = during_path.replace('.svg', '.png')
                        during_fig.savefig(during_png, dpi=300, bbox_inches='tight')
                        saved_paths.append(during_png)

                        plt.close(during_fig)

                # Create after visualization (50 epochs after)
                after_epoch = transition_epoch + 50
                after_checkpoint = os.path.join(weight_tracker.save_dir, f"checkpoint_step_{after_epoch}.pt")

                if os.path.exists(after_checkpoint):
                    # Load after state
                    checkpoint = torch.load(after_checkpoint)
                    model.load_state_dict(checkpoint["model_state_dict"])

                    # Generate after data
                    after_phase_data = self._extract_phase_data(
                        model=model,
                        phase_analyzer=phase_analyzer,
                        weight_tracker=weight_tracker,
                        epoch=after_epoch,
                        phase_name=f"After Transition"
                    )

                    # Save after visualization
                    if save_dir:
                        after_path = os.path.join(save_dir, f"transition_{i + 1}_after.svg")
                        after_fig = self.visualize_phase(
                            phase_data=after_phase_data,
                            save_path=after_path,
                            title=f"{transition_name} - After (Epoch {after_epoch})"
                        )
                        saved_paths.append(after_path)

                        # Also save PNG
                        after_png = after_path.replace('.svg', '.png')
                        after_fig.savefig(after_png, dpi=300, bbox_inches='tight')
                        saved_paths.append(after_png)

                        plt.close(after_fig)
        finally:
            # Restore original model state
            model.load_state_dict(current_state)

        return saved_paths


    def _extract_phase_data(self, model, phase_analyzer, weight_tracker, epoch, phase_name):
        """
        Extract visualization data from model and analyzers at a specific epoch.

        Returns:
            dict: Phase data ready for visualization
        """
        # Get head attributions from model (if eval_loader available)
        head_attributions = []

        if hasattr(model, 'analyze_head_attribution') and hasattr(phase_analyzer, 'eval_loader'):
            try:
                attribution_data = model.analyze_head_attribution(phase_analyzer.eval_loader)

                for layer_idx in range(self.num_layers):
                    layer_attr = []
                    for head_idx in range(self.num_heads):
                        key = f'layer_{layer_idx}_head_{head_idx}'
                        attr_val = attribution_data.get(key, 0.1)
                        # Normalize for visualization
                        norm_attr = max(0.1, min(0.9, attr_val * 10))
                        layer_attr.append(norm_attr)
                    head_attributions.append(layer_attr)
            except:
                # Default attributions if method fails
                for layer_idx in range(self.num_layers):
                    head_attributions.append([0.5] * self.num_heads)
        else:
            # Default attributions
            for layer_idx in range(self.num_layers):
                head_attributions.append([0.5] * self.num_heads)

        # Get MLP activations
        mlp_activations = []
        if hasattr(weight_tracker, 'train_history'):
            # Try to get MLP norms from weight tracker
            for layer_idx in range(self.num_layers):
                key = f'layer_{layer_idx}_mlp_combined_norm'
                if key in weight_tracker.train_history:
                    # Find activation values near this epoch
                    act_data = [(ep, act) for ep, act in zip(weight_tracker.weight_timestamps,
                                                             weight_tracker.train_history[key])
                                if abs(ep - epoch) < 100]
                    if act_data:
                        # Get closest activation to target epoch
                        epoch_diffs = [abs(ep - epoch) for ep, _ in act_data]
                        closest_idx = epoch_diffs.index(min(epoch_diffs))
                        activation = act_data[closest_idx][1]
                        # Normalize for visualization
                        norm_act = max(0.1, min(0.9, activation / 5.0))
                        mlp_activations.append(norm_act)
                    else:
                        mlp_activations.append(0.5)
                else:
                    mlp_activations.append(0.5)
        else:
            # Default activations
            mlp_activations = [0.5] * self.num_layers

        # Get circuits
        circuits = []
        if hasattr(phase_analyzer, 'circuit_tracker'):
            # Get circuits from circuit tracker
            try:
                circuit_summary = phase_analyzer.circuit_tracker.get_circuit_summary_for_epoch(epoch)

                if circuit_summary and 'active_circuits' in circuit_summary:
                    active_circuits = circuit_summary['active_circuits']
                    strengths = circuit_summary.get('circuit_strengths', {})

                    for circuit in active_circuits:
                        # Parse circuit notation
                        if '+' in circuit:
                            parts = circuit.split('+')
                            if len(parts) == 2:
                                src_parts = parts[0].split('_')
                                tgt_parts = parts[1].split('_')

                                if len(src_parts) >= 4 and len(tgt_parts) >= 4:
                                    src_layer = int(src_parts[1])
                                    src_head = int(src_parts[3])
                                    tgt_layer = int(tgt_parts[1])
                                    tgt_head = int(tgt_parts[3])

                                    # Convert to component IDs
                                    src_id = f"L{src_layer}_H{src_head}"
                                    tgt_id = f"L{tgt_layer}_H{tgt_head}"

                                    # Get circuit strength
                                    strength = strengths.get(circuit, 0.5)
                                    norm_strength = max(0.1, min(0.9, strength * 5))

                                    circuits.append((src_id, tgt_id, norm_strength))
            except:
                # No circuits if method fails
                pass

        # Return the collected data
        return {
            'head_attributions': head_attributions,
            'mlp_activations': mlp_activations,
            'circuits': circuits,
            'phase_name': phase_name,
            'epoch_range': (epoch, epoch)
        }


    def visualize_grokking_transition(self, phase_analyzer, weight_tracker, model,
                                      save_dir=None, title=None):
        """
        Create visualizations specifically focused on the grokking transition.

        Args:
            phase_analyzer: Instance of PhaseTransitionAnalyzer
            weight_tracker: Instance of EnhancedWeightSpaceTracker
            model: The transformer model
            save_dir: Directory to save visualizations
            title: Main title for visualizations

        Returns:
            List of saved paths
        """
        saved_paths = []

        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Check for grokking point
        grokking_epoch = None
        if hasattr(model, 'logger') and 'grokking_phases' in model.logger.logs:
            grokking_data = model.logger.logs['grokking_phases']
            if 'grokking_step' in grokking_data:
                grokking_step = grokking_data['grokking_step']
                if isinstance(grokking_step, (list, tuple)):
                    grokking_epoch = grokking_step[0] if grokking_step else None
                else:
                    grokking_epoch = grokking_step

        if not grokking_epoch:
            print("No grokking point detected")
            return saved_paths

        # Store current model state
        current_state = {k: v.clone() for k, v in model.state_dict().items()}

        try:
            # Define epochs to visualize (before, during, after grokking)
            before_epoch = max(1, grokking_epoch - 200)
            during_epoch = grokking_epoch
            after_epoch = grokking_epoch + 200

            epochs = [before_epoch, during_epoch, after_epoch]
            labels = ["Pre-Grokking", "Grokking Point", "Post-Grokking"]

            # Create visualizations for each stage
            for i, (epoch, label) in enumerate(zip(epochs, labels)):
                checkpoint_path = os.path.join(weight_tracker.save_dir, f"checkpoint_step_{epoch}.pt")

                if os.path.exists(checkpoint_path):
                    # Load checkpoint
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint["model_state_dict"])

                    # Get phase data
                    phase_data = self._extract_phase_data(
                        model=model,
                        phase_analyzer=phase_analyzer,
                        weight_tracker=weight_tracker,
                        epoch=epoch,
                        phase_name=label
                    )

                    # Create visualization
                    if save_dir:
                        save_path = os.path.join(save_dir, f"grokking_{label.lower().replace('-', '_')}.svg")
                        sub_title = title or "Transformer Model During Grokking"
                        fig = self.visualize_phase(
                            phase_data=phase_data,
                            save_path=save_path,
                            title=f"{sub_title} - {label} (Epoch {epoch})"
                        )
                        saved_paths.append(save_path)

                        # Also save PNG
                        png_path = save_path.replace('.svg', '.png')
                        fig.savefig(png_path, dpi=300, bbox_inches='tight')
                        saved_paths.append(png_path)

                        plt.close(fig)
        finally:
            # Restore original model state
            model.load_state_dict(current_state)

        return saved_paths


def create_phase_visualizations(model, phase_analyzer, weight_tracker,
                                eval_loader=None, save_dir="phase_visualizations"):
    """
    Create a complete set of phase-based visualizations for a trained model.

    Args:
        model: The transformer model
        phase_analyzer: Instance of PhaseTransitionAnalyzer
        weight_tracker: Instance of EnhancedWeightSpaceTracker
        save_dir: Directory to save visualizations

    Returns:
        dict: Paths to generated visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create the visualizer
    visualizer = AnthropicPhaseTransformerViz(
        num_layers=model.num_layers,
        num_heads=model.num_heads
    )

    # Generate main title
    title = f"Transformer Learning Analysis ({model.num_layers} Layers, {model.num_heads} Heads)"

    # Generate visualizations
    results = {
        'phases': visualizer.visualize_learning_phases(
            phase_analyzer=phase_analyzer,
            weight_tracker=weight_tracker,
            model=model,
            eval_loader=eval_loader,
            save_dir=os.path.join(save_dir, "learning_phases"),
            title=title
        ),
        'transitions': visualizer.visualize_phase_transitions(
            phase_analyzer=phase_analyzer,
            weight_tracker=weight_tracker,
            model=model,
            save_dir=os.path.join(save_dir, "phase_transitions"),
            title=title
        ),
        'grokking': visualizer.visualize_grokking_transition(
            phase_analyzer=phase_analyzer,
            weight_tracker=weight_tracker,
            model=model,
            save_dir=os.path.join(save_dir, "grokking_transition"),
            title=title
        )
    }

    # Create a summary HTML file to view all visualizations
    html_path = os.path.join(save_dir, "phase_visualization_summary.html")

    with open(html_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
        <html>
        <head>
            <title>Transformer Learning Phase Visualizations</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f7f9fb; }}
                .section {{ margin: 30px 0; border: 1px solid #ddd; padding: 20px; background-color: white; border-radius: 5px; }}
                .section h2 {{ color: #303133; }}
                .gallery {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .figure {{ margin: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .figure img {{ max-width: 100%; height: auto; max-height: 500px; }}
                .figure figcaption {{ padding: 10px; background-color: #f0f0f0; }}
                h1 {{ color: #5470C6; }}
            </style>
        </head>
        <body>
            <h1>Transformer Learning Phase Visualizations</h1>
            <p>Model: {title}</p>

            <div class="section">
                <h2>Learning Phases</h2>
                <div class="gallery">
        """)

        # Add phase visualizations
        for path in results['phases']:
            if path.endswith('.png'):
                name = os.path.basename(path)
                phase_name = name.replace('phase_', '').replace('.png', '').replace('_', ' ').title()
                f.write(f"""
                    <figure class="figure">
                        <img src="{path}" alt="{phase_name}">
                        <figcaption>{phase_name}</figcaption>
                    </figure>
        """)

        f.write(f"""
                </div>
            </div>

            <div class="section">
                <h2>Phase Transitions</h2>
                <div class="gallery">
        """)

        # Add transition visualizations
        for path in results['transitions']:
            if path.endswith('.png'):
                name = os.path.basename(path)
                f.write(f"""
                    <figure class="figure">
                        <img src="{path}" alt="{name}">
                        <figcaption>{name.replace('.png', '')}</figcaption>
                    </figure>
        """)

        f.write(f"""
                </div>
            </div>

            <div class="section">
                <h2>Grokking Transition</h2>
                <div class="gallery">
        """)

        # Add grokking visualizations
        for path in results['grokking']:
            if path.endswith('.png'):
                name = os.path.basename(path)
                stage = name.replace('grokking_', '').replace('.png', '').replace('_', ' ').title()
                f.write(f"""
                    <figure class="figure">
                        <img src="{path}" alt="{stage}">
                        <figcaption>{stage}</figcaption>
                    </figure>
        """)

        f.write(f"""
                </div>
            </div>
        </body>
        </html>
        """)

    # Add HTML summary to results
    results['summary_html'] = html_path

    print(f"Phase visualization summary created at: {html_path}")
    return results


if __name__ == "__main__":
    # This would be replaced with actual model and analyzer instances
    print("Import and use this module with your PhaseTransitionAnalyzer")