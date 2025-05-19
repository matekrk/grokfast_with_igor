import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import json
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from .base_visualizer import BaseVisualizer


class TimeSeriesVisualizer(BaseVisualizer):
    """Visualizes time series data as line charts."""

    @property
    def name(self):
        return "time_series"

    @property
    def description(self):
        return "Creates line charts for time series data with epoch as x-axis"

    def can_visualize(self, data):
        """Check if data contains time series information"""
        for field, field_data in data.items():
            if isinstance(field_data, dict):
                # Check for dictionaries with array/list values
                for key, values in field_data.items():
                    if isinstance(values, list) and len(values) > 1:
                        # Check if values look like numeric time series
                        if all(isinstance(v, (int, float)) for v in values[:5]):
                            return True
            elif isinstance(field_data, list) and len(field_data) > 1:
                # Direct list might be a time series
                if all(isinstance(v, (int, float)) for v in field_data[:5]):
                    return True
        return False

    def get_supported_fields(self, logger_data):
        """Get logger fields that can be visualized as time series."""
        supported = []

        # Recursively check for time series data in the logger
        def check_dict(data, path=""):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                # Skip special keys like 'grokking_step' that would be plotted differently
                if key in ['grokking_step', 'phase', 'transition_types']:
                    continue

                # If we find epoch data, look for related fields
                if key == 'epoch' and isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    parent_dict = data
                    for metric_key, metric_values in parent_dict.items():
                        if metric_key != 'epoch' and isinstance(metric_values, (list, np.ndarray)):
                            # Only include if lengths match and values are numeric
                            if (len(metric_values) == len(value) and
                                    all(isinstance(v, (int, float, np.number)) or v is None
                                        for v in metric_values)):
                                metric_path = f"{path}.{metric_key}" if path else metric_key
                                category_path = path  # The parent path becomes the category
                                supported.append(metric_path)

                # Recursively check nested dictionaries
                elif isinstance(value, dict):
                    check_dict(value, current_path)

        # Start the recursive check
        check_dict(logger_data)
        return supported

    def visualize(self, data, config=None):
        """Create a line chart visualization."""
        config = config or {}

        # Extract styling options
        figsize = config.get('figsize', (10, 6))
        title = config.get('title', 'Time Series Data')
        xlabel = config.get('xlabel', 'Epoch')
        ylabel = config.get('ylabel', 'Value')
        grid = config.get('grid', True)
        style = config.get('style', {})
        legend_loc = config.get('legend_loc', 'best')

        # Apply matplotlib style if specified
        plt_style = config.get('matplotlib_style', 'seaborn-v0_8-darkgrid')
        with plt.style.context(plt_style):
            fig, ax = plt.subplots(figsize=figsize)

            # If data is a dictionary with 'epoch'
            if isinstance(data, dict) and 'epoch' in data:
                x_values = data['epoch']

                # Plot each metric (except epoch)
                for key, values in data.items():
                    if key != 'epoch' and isinstance(values, (list, np.ndarray)) and len(values) == len(x_values):
                        # Filter out None values
                        valid_indices = [i for i, v in enumerate(values) if v is not None]
                        if valid_indices:
                            valid_x = [x_values[i] for i in valid_indices]
                            valid_y = [values[i] for i in valid_indices]

                            # Apply any specific style for this line
                            line_style = style.get(key, {})
                            label = line_style.get('label', key)
                            color = line_style.get('color', None)
                            linestyle = line_style.get('linestyle', '-')
                            marker = line_style.get('marker', 'o')
                            linewidth = line_style.get('linewidth', 1.5)
                            markersize = line_style.get('markersize', 5)
                            alpha = line_style.get('alpha', 0.8)

                            ax.plot(valid_x, valid_y,
                                    label=label,
                                    color=color,
                                    linestyle=linestyle,
                                    marker=marker,
                                    linewidth=linewidth,
                                    markersize=markersize,
                                    alpha=alpha)

            # If data is a list or array, plot directly
            elif isinstance(data, (list, np.ndarray)):
                x_values = list(range(len(data)))
                ax.plot(x_values, data, **style)

            # Set labels and title
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

            # Add grid if specified
            if grid:
                ax.grid(True, alpha=0.3)

            # Add legend if we have multiple lines
            if isinstance(data, dict) and len([k for k in data.keys() if k != 'epoch']) > 1:
                ax.legend(loc=legend_loc)

            # Add horizontal and vertical markers if specified
            h_markers = config.get('horizontal_markers', [])
            for marker in h_markers:
                y_value = marker.get('value', 0)
                color = marker.get('color', 'red')
                linestyle = marker.get('linestyle', '--')
                label = marker.get('label', None)
                ax.axhline(y=y_value, color=color, linestyle=linestyle, label=label)

            v_markers = config.get('vertical_markers', [])
            for marker in v_markers:
                x_value = marker.get('value', 0)
                color = marker.get('color', 'green')
                linestyle = marker.get('linestyle', '--')
                label = marker.get('label', None)
                ax.axvline(x=x_value, color=color, linestyle=linestyle, label=label)

            # Special handling for grokking points if provided
            grokking_points = config.get('grokking_points', [])
            if grokking_points:
                for point in grokking_points:
                    ax.axvline(x=point, color='green', linestyle='-',
                               label=f'Grokking point: {point}', linewidth=2)

            plt.tight_layout()
            return fig

    def save(self, visualization, path):
        """Save the visualization to a file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save with high quality
        visualization.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(visualization)
        return path


class HeatmapVisualizer(BaseVisualizer):
    """Visualizes 2D data as heatmaps."""

    @property
    def name(self):
        return "heatmap"

    @property
    def description(self):
        return "Creates heatmap visualizations for 2D data matrices"

    def can_visualize(self, data):
        """Check if data can be visualized as a heatmap."""
        # Check for 2D numpy array
        if isinstance(data, np.ndarray) and data.ndim == 2:
            return True

        # Check for 2D torch tensor
        if isinstance(data, torch.Tensor) and data.dim() == 2:
            return True

        # Check for 2D list of lists
        if (isinstance(data, list) and
                len(data) > 0 and
                isinstance(data[0], list) and
                all(isinstance(row, list) and len(row) == len(data[0]) for row in data)):
            return True

        return False

    def get_supported_fields(self, logger_data):
        """Get logger fields that can be visualized as heatmaps."""
        supported = []

        # Recursively check for 2D data in the logger
        def check_dict(data, path=""):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                # Check if value is 2D data
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    supported.append(current_path)

                elif isinstance(value, torch.Tensor) and value.dim() == 2:
                    supported.append(current_path)

                elif (isinstance(value, list) and len(value) > 0 and
                      isinstance(value[0], list) and
                      all(isinstance(row, list) and len(row) == len(value[0]) for row in value)):
                    supported.append(current_path)

                # Recursively check nested dictionaries
                elif isinstance(value, dict):
                    check_dict(value, current_path)

                # Check if this is a patterns dictionary with 2D arrays
                elif (isinstance(value, dict) and
                      all(isinstance(v, (np.ndarray, torch.Tensor)) for v in value.values()) and
                      any(v.ndim == 2 if isinstance(v, np.ndarray) else v.dim() == 2
                          for v in value.values())):
                    supported.append(current_path)

        # Start the recursive check
        check_dict(logger_data)
        return supported

    def visualize(self, data, config=None):
        """Create a heatmap visualization."""
        config = config or {}

        # Extract styling options
        figsize = config.get('figsize', (10, 8))
        title = config.get('title', 'Heatmap')
        cmap = config.get('cmap', 'viridis')
        center = config.get('center', None)
        annot = config.get('annot', False)
        fmt = config.get('fmt', '.2f')
        xlabel = config.get('xlabel', None)
        ylabel = config.get('ylabel', None)
        xticklabels = config.get('xticklabels', True)
        yticklabels = config.get('yticklabels', True)

        # Apply matplotlib style if specified
        plt_style = config.get('matplotlib_style', 'seaborn-v0_8-darkgrid')
        with plt.style.context(plt_style):
            # Convert to numpy array if needed
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            elif isinstance(data, list):
                data = np.array(data)

            # Check if data is a dictionary of patterns
            if isinstance(data, dict) and all(isinstance(v, (np.ndarray, torch.Tensor)) for v in data.values()):
                # Create a figure with multiple heatmaps
                keys = list(data.keys())
                n = len(keys)

                # Calculate grid dimensions
                cols = min(3, n)
                rows = (n + cols - 1) // cols

                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                if rows == 1 and cols == 1:
                    axes = np.array([axes])
                axes = axes.flatten()

                for i, (key, matrix) in enumerate(data.items()):
                    if i < len(axes):
                        # Convert to numpy if needed
                        if isinstance(matrix, torch.Tensor):
                            matrix = matrix.detach().cpu().numpy()

                        # Create heatmap
                        sns.heatmap(matrix, cmap=cmap, center=center,
                                    annot=annot, fmt=fmt,
                                    xticklabels=xticklabels,
                                    yticklabels=yticklabels,
                                    ax=axes[i])

                        axes[i].set_title(key)
                        if xlabel:
                            axes[i].set_xlabel(xlabel)
                        if ylabel:
                            axes[i].set_ylabel(ylabel)

                # Hide any unused subplots
                for i in range(n, len(axes)):
                    axes[i].axis('off')

                plt.suptitle(title, fontsize=16)
                plt.tight_layout()

            else:
                # Create a single heatmap
                fig, ax = plt.subplots(figsize=figsize)
                sns.heatmap(data, cmap=cmap, center=center,
                            annot=annot, fmt=fmt,
                            xticklabels=xticklabels,
                            yticklabels=yticklabels,
                            ax=ax)

                ax.set_title(title)
                if xlabel:
                    ax.set_xlabel(xlabel)
                if ylabel:
                    ax.set_ylabel(ylabel)

                plt.tight_layout()

            return fig

    def save(self, visualization, path):
        """Save the visualization to a file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save with high quality
        visualization.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(visualization)
        return path


class NetworkGraphVisualizer(BaseVisualizer):
    """Visualizes network graphs and connectivity."""

    @property
    def name(self):
        return "network_graph"

    @property
    def description(self):
        return "Creates network graph visualizations for connectivity data"

    def can_visualize(self, data):
        """Check if data contains network/graph information"""
        # Don't return True for everything!
        for field, field_data in data.items():
            field_name = field.lower()
            # Check field name for graph-related terms
            if any(term in field_name for term in ['circuit', 'network', 'graph', 'connection']):
                return True

            # Check for data that looks like a graph/network
            if isinstance(field_data, dict):
                # Look for nested dictionaries that could represent connections
                for key, value in field_data.items():
                    if isinstance(value, dict) and len(value) > 0:
                        return True
        return False

    def get_supported_fields(self, logger_data):
        """
        Get list of logger fields that this visualizer supports.
        Fixed to handle integer dictionary keys properly.

        Args:
            logger_data: The logger data dictionary

        Returns:
            list: List of field paths that this visualizer supports
        """
        supported = []

        # Check for circuit analysis data
        if 'circuit_analysis' in logger_data:
            supported.append('circuit_analysis')

        # Check for attention-MLP interaction data
        if 'attn_mlp_interaction' in logger_data:
            supported.append('attn_mlp_interaction')

        # Check for sparsity history with proper key handling
        if 'mlp_sparsity' in logger_data:
            supported.append('mlp_sparsity')

        # Check for neuron activations
        if 'neuron_activations' in logger_data:
            supported.append('neuron_activations')

        # Properly handle sparsity_history with integer keys
        for category in logger_data:
            # Check if this category could contain a sparsity_history
            if 'sparsity' in category.lower() or 'history' in category.lower():
                # For each key in the category
                for key in logger_data[category]:
                    # Check if the value is a dictionary (potential sparsity_history)
                    if isinstance(logger_data[category][key], dict):
                        # Check if it has numeric keys (common for epoch-indexed data)
                        has_numeric_keys = any(isinstance(k, (int, float))
                                               for k in logger_data[category][key].keys())

                        if has_numeric_keys:
                            # This might be a sparsity_history with integer keys
                            supported.append(f"{category}.{key}")

        # Add phase transition edges if available
        if 'phase_transitions' in logger_data:
            supported.append('phase_transitions')

        # Check for weight space jumps
        if 'weight_space_jumps' in logger_data:
            supported.append('weight_space_jumps')

        return supported

    def _convert_to_networkx(self, data):
        """Convert various data formats to a NetworkX graph."""
        if isinstance(data, (nx.Graph, nx.DiGraph)):
            return data

        # Create a directed graph by default
        G = nx.DiGraph()

        # Convert from adjacency matrix
        if isinstance(data, (np.ndarray, torch.Tensor)):
            # Convert torch tensor to numpy if needed
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()

            # Create nodes
            for i in range(data.shape[0]):
                G.add_node(i)

            # Add edges with weights from the matrix
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if data[i, j] != 0:
                        G.add_edge(i, j, weight=float(data[i, j]))

        # Convert from nodes/edges dictionary
        elif isinstance(data, dict) and 'nodes' in data and 'edges' in data:
            # Add nodes with attributes
            for node in data['nodes']:
                if isinstance(node, dict):
                    # If node is a dictionary with 'id' key
                    node_id = node.get('id', None)
                    if node_id is not None:
                        G.add_node(node_id, **{k: v for k, v in node.items() if k != 'id'})
                else:
                    # Simple node ID
                    G.add_node(node)

            # Add edges with attributes
            for edge in data['edges']:
                if isinstance(edge, dict):
                    # If edge is a dictionary with 'source' and 'target' keys
                    source = edge.get('source', None)
                    target = edge.get('target', None)
                    if source is not None and target is not None:
                        G.add_edge(source, target, **{k: v for k, v in edge.items()
                                                      if k not in ['source', 'target']})
                elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    # Edge as a list/tuple of [source, target] or [source, target, weight]
                    source, target = edge[0], edge[1]
                    if len(edge) > 2:
                        G.add_edge(source, target, weight=edge[2])
                    else:
                        G.add_edge(source, target)

        # Convert from connection dictionary
        elif isinstance(data, dict):
            # Add nodes for all keys
            for node in data.keys():
                G.add_node(node)

            # Add edges based on connections
            for source, targets in data.items():
                if isinstance(targets, list):
                    # List of target nodes
                    for target in targets:
                        G.add_edge(source, target)
                elif isinstance(targets, dict):
                    # Dictionary of target: weight pairs
                    for target, weight in targets.items():
                        G.add_edge(source, target, weight=weight)

        return G

    def visualize(self, data, config=None):
        """
        Create and return a network graph visualization from various data sources.

        This visualizer can handle:
        - Circuit analysis data (head connections, circuit strengths)
        - Attention-MLP interactions
        - Component relationships in phase transitions
        - Neuron activation networks
        - Sparsity patterns with numerical (epoch) keys

        Args:
            data: The data to visualize (dict with categories as keys)
            config: Optional configuration dictionary with:
                - title: Graph title
                - figsize: Figure dimensions as tuple (width, height)
                - layout: Graph layout algorithm ('spring', 'kamada_kawai', 'circular', etc.)
                - node_size_scale: Scaling factor for node sizes
                - edge_width_scale: Scaling factor for edge widths
                - show_labels: Whether to show node labels (default: True)
                - label_font_size: Font size for node labels
                - color_scheme: Color palette for nodes ('default', 'pastel', 'deep', etc.)

        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import numpy as np
        import seaborn as sns

        # Default configuration
        default_config = {
            'title': 'Network Graph Visualization',
            'figsize': (12, 10),
            'layout': 'spring',
            'node_size_scale': 500,
            'edge_width_scale': 2,
            'show_labels': True,
            'label_font_size': 9,
            'color_scheme': 'tab10'
        }

        # Update with user config
        if config is None:
            config = {}
        config = {**default_config, **config}

        # Create directed graph (can represent both directed and undirected relationships)
        G = nx.DiGraph()

        # Track node types for coloring
        node_types = {}
        node_weights = {}
        edge_weights = {}
        node_data = {}

        # Track legend items
        legend_elements = []

        # ========== Process Circuit Analysis Data ==========
        if 'circuit_analysis' in data:
            circuit_data = data['circuit_analysis']

            # Process jump-specific circuit analysis
            jump_circuits = {}
            for key in circuit_data:
                if key.startswith('jump_') and '_circuit' in key:
                    # Extract jump epoch and circuit number
                    parts = key.split('_')
                    jump_epoch = int(parts[1])

                    if jump_epoch not in jump_circuits:
                        jump_circuits[jump_epoch] = []

                    # Extract circuit components
                    if isinstance(circuit_data[key], list):
                        # List format
                        components = circuit_data[key]
                    elif isinstance(circuit_data[key], str) and '+' in circuit_data[key]:
                        # String format with + separator
                        components = circuit_data[key].split('+')
                    else:
                        # Single component or other format
                        components = [circuit_data[key]]

                    # Get circuit strength if available
                    strength_key = key.replace('circuit', 'interaction')
                    strength = circuit_data.get(strength_key, 0.5)  # Default strength

                    jump_circuits[jump_epoch].append((components, strength))

            # Add circuit connections to graph
            for jump_epoch, circuits in jump_circuits.items():
                for components, strength in circuits:
                    if len(components) >= 2:
                        # Create edges between components
                        for i in range(len(components) - 1):
                            src = str(components[i])
                            dst = str(components[i + 1])

                            # Add nodes if they don't exist
                            if src not in G:
                                G.add_node(src)
                                node_types[src] = 'circuit_component'
                                node_weights[src] = 1.0
                                node_data[src] = {'jump_epoch': jump_epoch}

                            if dst not in G:
                                G.add_node(dst)
                                node_types[dst] = 'circuit_component'
                                node_weights[dst] = 1.0
                                node_data[dst] = {'jump_epoch': jump_epoch}

                            # Add or update edge
                            if G.has_edge(src, dst):
                                # Strengthen existing edge
                                G[src][dst]['weight'] += float(strength)
                            else:
                                G.add_edge(src, dst, weight=float(strength))

                            edge_weights[(src, dst)] = float(strength)

                            # Add circuit metadata
                            G[src][dst]['circuit_epoch'] = jump_epoch
                            G[src][dst]['circuit_type'] = 'jump_circuit'

            # Add legend item for circuits
            legend_elements.append(
                Line2D([0], [0], color='blue', lw=2, label='Circuit Connection')
            )

        # ========== Process Attention-MLP Interactions ==========
        if 'attn_mlp_interaction' in data:
            interaction_data = data['attn_mlp_interaction']

            # Process correlation data
            for key in interaction_data:
                if 'correlation' in key and not key.endswith('_correlation'):
                    # Extract layer from key
                    parts = key.split('_')
                    try:
                        layer_idx = int(parts[1])

                        # Create nodes for attention and MLP components
                        attn_node = f"attention_layer_{layer_idx}"
                        mlp_node = f"mlp_layer_{layer_idx}"

                        # Add nodes
                        if attn_node not in G:
                            G.add_node(attn_node)
                            node_types[attn_node] = 'attention'
                            node_weights[attn_node] = 0.7

                        if mlp_node not in G:
                            G.add_node(mlp_node)
                            node_types[mlp_node] = 'mlp'
                            node_weights[mlp_node] = 0.7

                        # Add interaction edge with correlation as weight
                        correlation = float(interaction_data[key][-1])  # Use most recent value
                        if correlation > 0.1:  # Only add significant interactions
                            G.add_edge(attn_node, mlp_node, weight=correlation)
                            edge_weights[(attn_node, mlp_node)] = correlation
                            G[attn_node][mlp_node]['interaction_type'] = 'attn_mlp'
                    except (ValueError, IndexError):
                        continue

            # Add legend items for attention-MLP
            legend_elements.extend([
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=10, label='Attention'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                       markersize=10, label='MLP')
            ])

        # ========== Process Sparsity History with Numeric Keys ==========
        for category in data:
            if 'sparsity' in category.lower() or 'history' in category.lower():
                # Look for dictionary of epoch -> data mappings
                for key, value in data[category].items():
                    if isinstance(value, dict):
                        # Check for numeric keys (epochs)
                        numeric_keys = [k for k in value.keys()
                                        if isinstance(k, (int, float))]

                        if numeric_keys:
                            # Process the most recent epoch
                            latest_epoch = max(numeric_keys)
                            epoch_data = value[latest_epoch]

                            # Handle different sparsity data formats
                            if isinstance(epoch_data, dict):
                                # If we have layer-specific data
                                if 'layer_changes' in epoch_data:
                                    layer_data = epoch_data['layer_changes']
                                    # Add layers as nodes
                                    for layer_name, layer_info in layer_data.items():
                                        if layer_name not in G:
                                            G.add_node(layer_name)
                                            node_types[layer_name] = 'layer'

                                            # Use sparsity or change metric for node weight
                                            if 'normalized_change' in layer_info:
                                                weight = layer_info['normalized_change']
                                            else:
                                                weight = 0.5

                                            node_weights[layer_name] = weight
                                            node_data[layer_name] = {'epoch': latest_epoch}

                                # If we have connections data
                                if 'connections' in epoch_data:
                                    for src, targets in epoch_data['connections'].items():
                                        if isinstance(targets, dict):
                                            for dst, weight in targets.items():
                                                if weight > 0:  # Only add positive connections
                                                    if src not in G:
                                                        G.add_node(src)
                                                        node_types[src] = 'neuron'
                                                    if dst not in G:
                                                        G.add_node(dst)
                                                        node_types[dst] = 'neuron'

                                                    G.add_edge(src, dst, weight=float(weight))
                                                    edge_weights[(src, dst)] = float(weight)

        # ========== Process Phase Transition Data ==========
        if 'phase_transitions' in data:
            transition_data = data['phase_transitions']

            # Process transitions
            for key in transition_data:
                if 'transition_' in key and '_epoch' in key:
                    try:
                        # Get transition epoch
                        transition_epoch = transition_data[key][-1]  # Most recent

                        # Look for transition types and components
                        type_key = key.replace('_epoch', '_types')
                        if type_key in transition_data:
                            transition_types = transition_data[type_key][-1]

                            # Create nodes for each transition type
                            if isinstance(transition_types, str):
                                transition_types = transition_types.split('+')

                            for t_type in transition_types:
                                node_id = f"{t_type}_{transition_epoch}"
                                if node_id not in G:
                                    G.add_node(node_id)
                                    node_types[node_id] = 'transition'
                                    node_weights[node_id] = 0.8
                                    node_data[node_id] = {'epoch': transition_epoch}
                    except (IndexError, TypeError):
                        continue

        # ========== Handle Empty Graph ==========
        if len(G) == 0:
            # Create placeholder graph if no data was successfully processed
            fig, ax = plt.subplots(figsize=config['figsize'])
            ax.text(0.5, 0.5, "No network graph data found",
                    ha='center', va='center', fontsize=14)
            ax.set_title(config['title'])
            ax.axis('off')
            return fig

        # ========== Create Graph Visualization ==========
        fig, ax = plt.subplots(figsize=config['figsize'])

        # Choose layout algorithm
        if config['layout'] == 'spring':
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        elif config['layout'] == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif config['layout'] == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        # Create color map based on node types
        unique_types = list(set(node_types.values()))
        color_map = plt.cm.get_cmap(config['color_scheme'], max(8, len(unique_types)))
        type_to_color = {t: color_map(i) for i, t in enumerate(unique_types)}

        # Create node colors
        node_colors = [type_to_color.get(node_types.get(n, 'other'), 'gray') for n in G.nodes()]

        # Create node sizes based on weights
        node_sizes = [config['node_size_scale'] * node_weights.get(n, 0.5) + 100 for n in G.nodes()]

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                               node_size=node_sizes,
                               node_color=node_colors,
                               alpha=0.8,
                               ax=ax)

        # Create edge widths based on weights
        edge_widths = [config['edge_width_scale'] * edge_weights.get(e, 0.5) + 1 for e in G.edges()]

        # Draw edges with arrows for directed connections
        nx.draw_networkx_edges(G, pos,
                               width=edge_widths,
                               alpha=0.6,
                               edge_color='gray',
                               arrows=True,
                               arrowsize=15,
                               ax=ax)

        # Draw labels if requested
        if config['show_labels']:
            # Generate shorter labels for better readability
            labels = {}
            for node in G.nodes():
                # Create more readable labels
                if isinstance(node, str):
                    if 'layer_' in node and 'head_' in node:
                        # For attention heads, use format "L0H1"
                        parts = node.split('_')
                        try:
                            layer = int(parts[parts.index('layer') + 1])
                            head = int(parts[parts.index('head') + 1])
                            labels[node] = f"L{layer}H{head}"
                        except (ValueError, IndexError):
                            labels[node] = node
                    elif 'layer_' in node:
                        # For layers, use format "Layer 0"
                        parts = node.split('_')
                        try:
                            layer = int(parts[parts.index('layer') + 1])
                            labels[node] = f"Layer {layer}"
                        except (ValueError, IndexError):
                            labels[node] = node
                    else:
                        # For other nodes, use the original label but truncated
                        labels[node] = str(node)[:20] + ('...' if len(str(node)) > 20 else '')
                else:
                    # For non-string nodes
                    labels[node] = str(node)

            nx.draw_networkx_labels(G, pos, labels=labels,
                                    font_size=config['label_font_size'],
                                    font_family='sans-serif')

        # Add legend for node types
        for node_type, color in type_to_color.items():
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markersize=10,
                       label=node_type.replace('_', ' ').title())
            )

        # Add the legend
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right',
                      bbox_to_anchor=(1.1, 1.05))

        # Set title and remove axes
        ax.set_title(config['title'])
        ax.axis('off')

        plt.tight_layout()
        return fig

    def save(self, visualization, path):
        """Save the visualization to a file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save with high quality
        visualization.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(visualization)
        return path


class CircuitAnalysisVisualizer(BaseVisualizer):
    """Specialized visualizer for circuit analysis results."""

    @property
    def name(self):
        return "circuit_analysis"

    @property
    def description(self):
        return "Creates visualizations for transformer circuit analysis results"

    def can_visualize(self, data):
        """Check if data can be visualized as circuit analysis."""
        # Circuit analysis typically has specific keys
        if isinstance(data, dict):
            # Check for common circuit analysis keys
            circuit_keys = ['active_circuits', 'circuit_strengths', 'head_attributions',
                            'significant_heads', 'circuits', 'pairwise_interactions']

            return any(key in data for key in circuit_keys)

        return False

    def get_supported_fields(self, logger_data):
        """Get logger fields that can be visualized as circuit analysis."""
        supported = []

        # Recursively check for circuit analysis data in the logger
        def check_dict(data, path=""):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                # Check for circuit analysis keys
                if key in ['circuit_analysis', 'circuit_tracker', 'circuits']:
                    if isinstance(value, dict):
                        # If value is a dictionary, it's likely circuit analysis data
                        circuit_keys = ['active_circuits', 'circuit_strengths', 'head_attributions',
                                        'significant_heads', 'pairwise_interactions']
                        if any(ck in value for ck in circuit_keys):
                            supported.append(current_path)

                # Check for circuit history
                elif key == 'circuit_history' and isinstance(value, dict):
                    supported.append(current_path)

                # Recursively check nested dictionaries
                elif isinstance(value, dict):
                    check_dict(value, current_path)

        # Start the recursive check
        check_dict(logger_data)
        return supported

    def visualize(self, data, config=None):
        """Create circuit analysis visualizations."""
        config = config or {}

        # Extract styling options
        figsize = config.get('figsize', (15, 15))
        title = config.get('title', 'Circuit Analysis')

        # Create a multi-panel visualization based on available data
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(3, 2)  # 3 rows, 2 columns

        # 1. Head attribution barplot (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'head_attributions' in data:
            self._plot_head_attributions(data['head_attributions'], ax1)
        elif 'significant_heads' in data:
            self._plot_head_attributions(data['significant_heads'], ax1)

        # 2. Circuit strength barplot (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'circuit_strengths' in data:
            self._plot_circuit_strengths(data['circuit_strengths'], ax2)
        elif 'circuits' in data and isinstance(data['circuits'], dict):
            circuit_strengths = {
                circuit: info.get('circuit_strength', 0)
                for circuit, info in data['circuits'].items()
            }
            self._plot_circuit_strengths(circuit_strengths, ax2)

        # 3. Circuit network graph (bottom-left)
        ax3 = fig.add_subplot(gs[1:, 0])
        if 'active_circuits' in data and 'head_attributions' in data:
            self._plot_circuit_network(data, ax3)

        # 4. Circuit interaction matrix (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        if 'pairwise_interactions' in data:
            self._plot_interaction_matrix(data['pairwise_interactions'], ax4)

        # 5. Circuit stability (if available)
        ax5 = fig.add_subplot(gs[2, 1])
        if 'circuit_stability' in data:
            self._plot_circuit_stability(data['circuit_stability'], ax5)
        elif 'epochs' in data and 'active_circuits' in data:
            # Try to extract stability information from history
            self._plot_circuit_evolution(data, ax5)

        plt.suptitle(title, fontsize=16)
        return fig

    def _plot_head_attributions(self, attributions, ax):
        """Plot head attribution scores."""
        # Convert dictionary to sorted list of tuples
        if isinstance(attributions, dict):
            items = sorted(attributions.items(), key=lambda x: x[1], reverse=True)
            heads = [head for head, _ in items[:10]]  # Top 10 heads
            values = [value for _, value in items[:10]]
        else:
            # If it's already a list
            heads = [item[0] for item in attributions[:10]]
            values = [item[1] for item in attributions[:10]]

        # Create horizontal bar chart
        y_pos = range(len(heads))
        ax.barh(y_pos, values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(heads)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Attribution Score')
        ax.set_title('Top Head Attributions')

    def _plot_circuit_strengths(self, strengths, ax):
        """Plot circuit strength scores."""
        # Convert dictionary to sorted list of tuples
        if isinstance(strengths, dict):
            items = sorted(strengths.items(), key=lambda x: x[1], reverse=True)
            circuits = [circuit for circuit, _ in items[:10]]  # Top 10 circuits
            values = [value for _, value in items[:10]]
        else:
            # If it's already a list
            circuits = [item[0] for item in strengths[:10]]
            values = [item[1] for item in strengths[:10]]

        # Create horizontal bar chart
        y_pos = range(len(circuits))
        ax.barh(y_pos, values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(circuits)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Circuit Strength')
        ax.set_title('Top Circuit Strengths')

    def _plot_circuit_network(self, data, ax):
        """Plot the circuit network graph."""
        # Create a graph
        G = nx.Graph()

        # Add nodes for each head with attribution scores
        for head, score in data['head_attributions'].items():
            G.add_node(head, size=score, type='head')

        # Add edges for active circuits
        if isinstance(data['active_circuits'], list):
            for circuit in data['active_circuits']:
                if '+' in circuit:
                    head1, head2 = circuit.split('+')
                    # Get circuit strength if available
                    strength = data.get('circuit_strengths', {}).get(circuit, 1.0)
                    G.add_edge(head1, head2, weight=strength)

        # Compute layout
        pos = nx.spring_layout(G, seed=42)

        # Get node sizes based on attribution scores
        node_sizes = [G.nodes[n].get('size', 0.1) * 1000 + 100 for n in G.nodes()]

        # Get edge widths based on circuit strengths
        edge_widths = [G[u][v].get('weight', 1.0) * 2 + 1 for u, v in G.edges()]

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        ax.set_title('Circuit Network')
        ax.axis('off')

    def _plot_interaction_matrix(self, interactions, ax):
        """Plot the pairwise interaction matrix."""
        # Extract heads and create interaction matrix
        if isinstance(interactions, dict):
            # Get all unique heads
            heads = set()
            for pair in interactions.keys():
                if isinstance(pair, tuple):
                    heads.add(pair[0])
                    heads.add(pair[1])
                elif isinstance(pair, str) and '+' in pair:
                    head1, head2 = pair.split('+')
                    heads.add(head1)
                    heads.add(head2)

            heads = sorted(list(heads))
            n = len(heads)

            # Create interaction matrix
            matrix = np.zeros((n, n))
            for i, head1 in enumerate(heads):
                for j, head2 in enumerate(heads):
                    if i > j:  # Only fill lower triangle
                        # Try both orderings of the pair
                        pair1 = f"{head1}+{head2}"
                        pair2 = f"{head2}+{head1}"
                        pair_tuple1 = (head1, head2)
                        pair_tuple2 = (head2, head1)

                        if pair1 in interactions:
                            matrix[i, j] = interactions[pair1].get('circuit_strength', 0)
                        elif pair2 in interactions:
                            matrix[i, j] = interactions[pair2].get('circuit_strength', 0)
                        elif pair_tuple1 in interactions:
                            matrix[i, j] = interactions[pair_tuple1].get('circuit_strength', 0)
                        elif pair_tuple2 in interactions:
                            matrix[i, j] = interactions[pair_tuple2].get('circuit_strength', 0)

            # Make it symmetric
            matrix = matrix + matrix.T

            # Plot heatmap
            sns.heatmap(matrix, xticklabels=heads, yticklabels=heads, cmap='RdBu_r',
                        center=0, square=True, ax=ax)

            ax.set_title('Pairwise Interactions')
        else:
            ax.text(0.5, 0.5, 'Interaction data not available',
                    horizontalalignment='center', verticalalignment='center')
            ax.axis('off')

    def _plot_circuit_stability(self, stability_data, ax):
        """Plot circuit stability over time."""
        for circuit, stability in stability_data.items():
            epochs = list(range(len(stability)))
            ax.plot(epochs, stability, label=circuit)

        ax.set_xlabel('Time')
        ax.set_ylabel('Stability')
        ax.set_title('Circuit Stability')
        ax.legend()

    def _plot_circuit_evolution(self, data, ax):
        """Plot circuit evolution over time."""
        epochs = data['epochs']

        # Count active circuits per epoch
        if isinstance(data['active_circuits'], list) and all(
                isinstance(item, list) for item in data['active_circuits']):
            circuit_counts = [len(circuits) for circuits in data['active_circuits']]
            ax.plot(epochs, circuit_counts, 'b-', label='Active Circuits')

        # Plot emerging circuits if available
        if 'emerging_circuits' in data:
            emerging_counts = [len(circuits) for circuits in data['emerging_circuits']]
            ax.plot(epochs, emerging_counts, 'g--', label='Emerging')

        # Plot declining circuits if available
        if 'declining_circuits' in data:
            declining_counts = [len(circuits) for circuits in data['declining_circuits']]
            ax.plot(epochs, declining_counts, 'r--', label='Declining')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Count')
        ax.set_title('Circuit Evolution')
        ax.legend()

    def save(self, visualization, path):
        """Save the visualization to a file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save with high quality
        visualization.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(visualization)
        return path


class PhaseTransitionVisualizer(BaseVisualizer):
    """Specialized visualizer for phase transition analysis."""

    @property
    def name(self):
        return "phase_transitions"

    @property
    def description(self):
        return "Creates visualizations for phase transitions in transformer learning"

    def can_visualize(self, data):
        """Check if data can be visualized as phase transitions."""
        # Phase transition data typically has specific keys
        if isinstance(data, dict):
            phase_keys = ['phases', 'transitions', 'insights']
            return any(key in data for key in phase_keys)

        return False

    def get_supported_fields(self, logger_data):
        """Get logger fields that can be visualized as phase transitions."""
        supported = []

        # Recursively check for phase transition data in the logger
        def check_dict(data, path=""):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                # Check for phase transition keys
                if key in ['phase_transitions', 'phase_structure', 'phases']:
                    if isinstance(value, dict):
                        # If value is a dictionary with phase data
                        phase_keys = ['phases', 'transitions', 'insights']
                        if any(pk in value for pk in phase_keys):
                            supported.append(current_path)

                # Recursively check nested dictionaries
                elif isinstance(value, dict):
                    check_dict(value, current_path)

        # Start the recursive check
        check_dict(logger_data)
        return supported

    def visualize(self, data, config=None):
        """Create phase transition visualizations."""
        config = config or {}

        # Extract styling options
        figsize = config.get('figsize', (15, 10))
        title = config.get('title', 'Learning Phase Analysis')

        # Extract additional data for visualization
        performance_data = config.get('performance_data', None)
        grokking_points = config.get('grokking_points', [])

        # Create figure with two rows
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Plot performance data on top plot if available
        if performance_data is not None:
            self._plot_performance(performance_data, data, ax1, grokking_points)
        else:
            # Create an empty placeholder plot
            ax1.text(0.5, 0.5, 'Performance data not available',
                     horizontalalignment='center', verticalalignment='center')
            ax1.set_title('Training Performance')
            ax1.axis('on')

        # Plot phase structure on bottom plot
        self._plot_phases(data, ax2, grokking_points)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig

    def _plot_performance(self, performance_data, phase_data, ax, grokking_points):
        """Plot performance metrics with phase transitions."""
        # Plot training and evaluation metrics
        if 'training' in performance_data and 'evaluation' in performance_data:
            train_data = performance_data['training']
            eval_data = performance_data['evaluation']

            # Check if required keys exist
            if ('epoch' in train_data and 'accuracy' in train_data and
                    'epoch' in eval_data and 'accuracy' in eval_data):
                train_epochs = train_data['epoch']
                train_accs = train_data['accuracy']

                eval_epochs = eval_data['epoch']
                eval_accs = eval_data['accuracy']

                # Plot training and evaluation accuracy
                ax.plot(train_epochs, train_accs, 'b-', label='Train Accuracy')
                ax.plot(eval_epochs, eval_accs, 'r-', label='Eval Accuracy')

        # Mark phase transitions
        if 'transitions' in phase_data:
            for transition in phase_data['transitions']:
                epoch = transition.get('epoch')
                if epoch:
                    # Add vertical line for transition
                    ax.axvline(x=epoch, color='purple', linestyle='--', alpha=0.7)

                    # Add label with transition type
                    transition_types = transition.get('transition_types', [])
                    if transition_types:
                        label = ", ".join(transition_types)
                        ax.text(epoch, 0.1, f"T:{label}", rotation=90,
                                transform=ax.get_xaxis_transform(), fontsize=8)

        # Mark grokking points
        for point in grokking_points:
            ax.axvline(x=point, color='green', linestyle='-', linewidth=2, alpha=0.7)
            ax.text(point, 0.95, f"Grokking: {point}", rotation=90,
                    transform=ax.get_xaxis_transform(), fontsize=10, color='green')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Performance and Phase Transitions')
        ax.legend()

    def _plot_phases(self, data, ax, grokking_points):
        """Plot phase structure as colored blocks."""
        if 'phases' in data:
            phases = data['phases']

            # Define colors for different phase classifications
            phase_colors = {
                'exploration': 'lightblue',
                'consolidation': 'lightgreen',
                'stability': 'lightyellow',
                'pruning': 'salmon',
                'transition': 'lightgray',
                'unknown': 'white'
            }

            # Plot phases as colored blocks
            for i, phase in enumerate(phases):
                start = phase.get('start_epoch', 0)
                end = phase.get('end_epoch', 0)

                classification = phase.get('classification', 'unknown')
                color = phase_colors.get(classification, 'white')

                # Add rectangle for this phase
                rect = plt.Rectangle((start, 0), end - start, 1,
                                     facecolor=color, alpha=0.7, edgecolor='black')
                ax.add_patch(rect)

                # Add phase label
                ax.text((start + end) / 2, 0.5, f"Phase {i + 1}\n{classification.title()}",
                        ha='center', va='center', fontsize=10)

            # Set axis limits
            min_epoch = min(phase['start_epoch'] for phase in phases)
            max_epoch = max(phase['end_epoch'] for phase in phases)
            ax.set_xlim(min_epoch, max_epoch)
            ax.set_ylim(0, 1)

            # Mark grokking points
            for point in grokking_points:
                ax.axvline(x=point, color='green', linestyle='-', linewidth=2, alpha=0.7)
                ax.text(point, 0.85, f"G", rotation=0, fontsize=10, color='white',
                        bbox=dict(facecolor='green', alpha=0.7))

        else:
            # No phase data available
            ax.text(0.5, 0.5, 'Phase data not available',
                    horizontalalignment='center', verticalalignment='center')

        # Remove y-axis ticks and labels
        ax.set_yticks([])
        ax.set_ylabel('Phases')
        ax.set_xlabel('Epoch')
        ax.set_title('Learning Phases')

    def save(self, visualization, path):
        """Save the visualization to a file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save with high quality
        visualization.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(visualization)
        return path


class MLPSparsityVisualizer(BaseVisualizer):
    """Specialized visualizer for MLP sparsity analysis."""

    @property
    def name(self):
        return "mlp_sparsity"

    @property
    def description(self):
        return "Creates visualizations for MLP neuron sparsity and selectivity"

    def can_visualize(self, data):
        """Check if data can be visualized as MLP sparsity analysis."""
        if isinstance(data, dict):
            # Look for common keys in sparsity analysis
            sparsity_keys = ['avg_sparsity', 'selectivity_summary', 'class_selectivity', 'neuron_activity']
            return any(key in data for key in sparsity_keys)

        return False

    def get_supported_fields(self, logger_data):
        """
        Get list of logger fields that this visualizer supports.
        Fixed to handle integer dictionary keys properly.

        Args:
            logger_data: The logger data dictionary

        Returns:
            list: List of field paths that this visualizer supports
        """
        supported = []

        # List of key substrings that indicate sparsity-related data
        sparsity_keys = ['sparsity', 'sparse', 'activation', 'neuron', 'mlp']

        # Check regular logger categories
        for category in logger_data:
            # Check for direct sparsity field matches
            if any(sk in category.lower() for sk in sparsity_keys):
                supported.append(category)

            # Check for nested dictionary structures that might contain sparsity data
            if isinstance(logger_data[category], dict):
                for key in logger_data[category]:
                    # Handle integer keys properly
                    if isinstance(key, (int, float)):
                        # For integer keys, can't check if string is in them
                        # Instead, check if the value is a dict with sparsity-related keys
                        if (isinstance(logger_data[category][key], dict) and
                                any(sk in str(k).lower() for k in logger_data[category][key] for sk in sparsity_keys)):
                            supported.append(f"{category}.{key}")
                    else:
                        # For string keys, check directly
                        if any(sk in str(key).lower() for sk in sparsity_keys):
                            supported.append(f"{category}.{key}")

                        # Also check nested dictionary values
                        if isinstance(logger_data[category][key], dict):
                            # Check if there are integer keys in the nested dict
                            numeric_keys = any(isinstance(k, (int, float)) for k in logger_data[category][key].keys())

                            if numeric_keys:
                                # This could be an epoch-indexed data structure
                                supported.append(f"{category}.{key}")

                            # Also check for sparsity keys in the nested dict
                            contains_sparsity = any(
                                sk in str(k).lower() for k in logger_data[category][key].keys()
                                for sk in sparsity_keys
                            )

                            if contains_sparsity:
                                supported.append(f"{category}.{key}")

        # Check for mlp_sparsity_tracker's sparsity_history structure
        if any(cat.startswith('sparsity_') for cat in logger_data.keys()):
            for cat in logger_data:
                if cat.startswith('sparsity_'):
                    supported.append(cat)

        # Check for enhanced_analysis data
        for category in logger_data:
            if 'enhanced_analysis' in category:
                supported.append(category)

        return supported

    def visualize(self, data, config=None):
        """Create MLP sparsity visualizations."""
        config = config or {}

        # Extract styling options
        figsize = config.get('figsize', (15, 12))
        title = config.get('title', 'MLP Neuron Sparsity Analysis')

        # Check what kind of data we have to determine visualization layout
        has_history = False
        has_selectivity = False

        # Check for sparsity history
        if isinstance(data, dict) and all(isinstance(k, (int, float)) for k in data.keys()):
            has_history = True

        # Check for selectivity data
        if (isinstance(data, dict) and
                ('selectivity_summary' in data or
                 any('selectivity' in k for k in data.keys()))):
            has_selectivity = True

        # Create appropriate layout based on available data
        if has_history and has_selectivity:
            # 2x2 grid
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            gs = fig.add_gridspec(2, 2)

            # 1. Sparsity by layer (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_sparsity_by_layer(data, ax1)

            # 2. Sparsity evolution (top-right)
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_sparsity_evolution(data, ax2)

            # 3. Neuron selectivity (bottom-left)
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_neuron_selectivity(data, ax3)

            # 4. Class distribution (bottom-right)
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_class_distribution(data, ax4)

        elif has_history:
            # 1x2 grid focusing on sparsity trends
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # 1. Sparsity by layer
            self._plot_sparsity_by_layer(data, ax1)

            # 2. Sparsity evolution
            self._plot_sparsity_evolution(data, ax2)

        elif has_selectivity:
            # 1x2 grid focusing on selectivity
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # 1. Neuron selectivity
            self._plot_neuron_selectivity(data, ax1)

            # 2. Class distribution
            self._plot_class_distribution(data, ax2)

        else:
            # Single plot for whatever data we have
            fig, ax = plt.subplots(figsize=(10, 6))
            self._plot_sparsity_by_layer(data, ax)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig

    def _plot_sparsity_by_layer(self, data, ax):
        """Plot sparsity values by layer."""
        # Check for different data formats
        avg_sparsity = None

        # Format 1: Direct avg_sparsity dictionary
        if 'avg_sparsity' in data:
            avg_sparsity = data['avg_sparsity']

        # Format 2: Latest entry in sparsity_history
        elif isinstance(data, dict) and all(isinstance(k, (int, float)) for k in data.keys()):
            latest_epoch = max(data.keys())
            if 'avg_sparsity' in data[latest_epoch]:
                avg_sparsity = data[latest_epoch]['avg_sparsity']

        # If we have sparsity data, plot it
        if avg_sparsity:
            # Sort layers by their index
            layer_names = sorted(avg_sparsity.keys(),
                                 key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)

            sparsity_values = [avg_sparsity[layer] for layer in layer_names]

            # Create bar chart
            x_pos = range(len(layer_names))
            ax.bar(x_pos, sparsity_values)

            # Clean up layer names for display
            display_names = [layer.replace('layer_', 'L').replace('_mlp_expanded', '')
                             for layer in layer_names]

            ax.set_xticks(x_pos)
            ax.set_xticklabels(display_names)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Sparsity (% inactive neurons)')
            ax.set_title('MLP Sparsity by Layer')

            # Add value labels on bars
            for i, v in enumerate(sparsity_values):
                ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
        else:
            # No sparsity data available
            ax.text(0.5, 0.5, 'Sparsity data not available',
                    horizontalalignment='center', verticalalignment='center')
            ax.axis('on')

    def _plot_sparsity_evolution(self, data, ax):
        """Plot sparsity evolution over time."""
        # Check if we have sparsity history
        if isinstance(data, dict) and all(isinstance(k, (int, float)) for k in data.keys()):
            # This is a sparsity history dictionary
            epochs = sorted(data.keys())

            # Get all layer names from the first entry
            if epochs and 'avg_sparsity' in data[epochs[0]]:
                layer_names = data[epochs[0]]['avg_sparsity'].keys()

                # Plot evolution for each layer
                for layer in layer_names:
                    sparsity_values = []
                    valid_epochs = []

                    for epoch in epochs:
                        if ('avg_sparsity' in data[epoch] and
                                layer in data[epoch]['avg_sparsity']):
                            valid_epochs.append(epoch)
                            sparsity_values.append(data[epoch]['avg_sparsity'][layer])

                    if valid_epochs:
                        # Clean up layer name for display
                        display_name = layer.replace('layer_', 'L').replace('_mlp_expanded', '')
                        ax.plot(valid_epochs, sparsity_values, 'o-', label=display_name)

            # Add transitions if provided in config
            transitions = data.get('transitions', [])
            if isinstance(transitions, list):
                for transition in transitions:
                    if isinstance(transition, dict) and 'epoch' in transition:
                        ax.axvline(x=transition['epoch'], color='red', linestyle='--', alpha=0.7)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Sparsity (% inactive neurons)')
            ax.set_title('MLP Sparsity Evolution')
            ax.legend()
        else:
            # No evolution data available
            ax.text(0.5, 0.5, 'Sparsity evolution data not available',
                    horizontalalignment='center', verticalalignment='center')
            ax.axis('on')

    def _plot_neuron_selectivity(self, data, ax):
        """Plot neuron selectivity data."""
        selectivity_data = None

        # Format 1: Direct selectivity_summary
        if 'selectivity_summary' in data:
            selectivity_data = data['selectivity_summary']

        # Format 2: Latest entry in sparsity_history
        elif isinstance(data, dict) and all(isinstance(k, (int, float)) for k in data.keys()):
            latest_epoch = max(data.keys())
            if 'selectivity_summary' in data[latest_epoch]:
                selectivity_data = data[latest_epoch]['selectivity_summary']

        # If we have selectivity data, plot it
        if selectivity_data:
            # Create data for plotting
            layer_names = []
            total_neurons = []
            selective_neurons = []
            selectivity_ratios = []

            for layer, info in selectivity_data.items():
                layer_names.append(layer.replace('layer_', 'L').replace('_mlp_expanded', ''))
                total_neurons.append(info.get('total_neurons', 0))
                selective_neurons.append(info.get('selective_neurons', 0))
                ratio = info.get('selective_neurons', 0) / info.get('total_neurons', 1)
                selectivity_ratios.append(ratio)

            # Create stacked bar chart
            x_pos = range(len(layer_names))

            # Plot non-selective neurons
            non_selective = [total - selective for total, selective in zip(total_neurons, selective_neurons)]
            ax.bar(x_pos, non_selective, label='Non-selective neurons')

            # Plot selective neurons on top
            ax.bar(x_pos, selective_neurons, bottom=non_selective, label='Selective neurons')

            ax.set_xticks(x_pos)
            ax.set_xticklabels(layer_names)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Neuron Count')
            ax.set_title('Neuron Selectivity by Layer')
            ax.legend()

            # Add selectivity ratio labels
            for i, ratio in enumerate(selectivity_ratios):
                ax.text(i, total_neurons[i] + 1, f"{ratio:.2f}", ha='center')
        else:
            # No selectivity data available
            ax.text(0.5, 0.5, 'Neuron selectivity data not available',
                    horizontalalignment='center', verticalalignment='center')
            ax.axis('on')

    def _plot_class_distribution(self, data, ax):
        """Plot class distribution of selective neurons."""
        class_distribution = None

        # Format 1: Direct selectivity_summary with class_distribution
        if ('selectivity_summary' in data and
                isinstance(data['selectivity_summary'], dict)):

            # Find first entry with class_distribution
            for layer, info in data['selectivity_summary'].items():
                if 'class_distribution' in info:
                    class_distribution = info['class_distribution']
                    break

        # Format 2: Latest entry in sparsity_history
        elif isinstance(data, dict) and all(isinstance(k, (int, float)) for k in data.keys()):
            latest_epoch = max(data.keys())
            if 'selectivity_summary' in data[latest_epoch]:
                # Find first entry with class_distribution
                for layer, info in data[latest_epoch]['selectivity_summary'].items():
                    if 'class_distribution' in info:
                        class_distribution = info['class_distribution']
                        break

        # If we have class distribution data, plot it
        if class_distribution:
            # Convert dict to sorted lists
            classes = sorted(class_distribution.keys())
            neuron_counts = [class_distribution[c] for c in classes]

            # Create bar chart
            x_pos = range(len(classes))
            ax.bar(x_pos, neuron_counts)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(classes)
            ax.set_xlabel('Class')
            ax.set_ylabel('Selective Neuron Count')
            ax.set_title('Class-Selective Neurons Distribution')

            # Add value labels on bars
            for i, v in enumerate(neuron_counts):
                ax.text(i, v + 0.5, str(v), ha='center')
        else:
            # No class distribution data available
            ax.text(0.5, 0.5, 'Class distribution data not available',
                    horizontalalignment='center', verticalalignment='center')
            ax.axis('on')

    def save(self, visualization, path):
        """Save the visualization to a file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save with high quality
        visualization.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(visualization)
        return path