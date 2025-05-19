# logger_visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd
import json
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import networkx as nx


class LoggerVisualizer:
    """Visualization utilities for DataLogger metrics"""

    def __init__(self, logger, save_dir=None):
        """
        Initialize the visualizer

        Args:
            logger: DataLogger instance containing metrics
            save_dir: Directory to save visualization outputs
        """
        self.logger = logger
        self.save_dir = Path(save_dir) if save_dir else Path("visualizations")
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # Set up plotting style
        sns.set_theme(
            style="whitegrid",
            font_scale=1.0,
            palette=sns.color_palette("pastel"),
            rc={
                "lines.linewidth": 1.0,
                "axes.spines.right": False,
                "axes.spines.top": False,
            }
        )

    def visualize_training_metrics(self, figsize=(12, 8), highlight_grokking=True,
                                   highlight_jumps=True, highlight_phases=True):
        """
        Create a comprehensive visualization of training metrics

        Args:
            figsize: Figure size tuple
            highlight_grokking: Whether to highlight grokking points
            highlight_jumps: Whether to highlight weight space jumps
            highlight_phases: Whether to highlight phase transitions

        Returns:
            fig: Matplotlib figure object
        """
        # Check if training metrics exist
        if not self.logger.logs.get('training') or not self.logger.logs.get('evaluation'):
            print("No training or evaluation metrics found in logger")
            return None

        # Extract training data
        train_logs = self.logger.logs['training']
        eval_logs = self.logger.logs['evaluation']

        epochs = train_logs.get('epoch', [])
        if not epochs:
            print("No epoch data found")
            return None

        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot accuracy
        ax1 = axs[0]
        if 'accuracy' in train_logs and 'accuracy' in eval_logs:
            ax1.plot(epochs, train_logs['accuracy'], 'b-', marker='.', alpha=0.7,
                     label='Train Accuracy')
            ax1.plot(epochs, eval_logs['accuracy'], 'r-', marker='.', alpha=0.7,
                     label='Validation Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1.05)
            ax1.legend(loc='lower right')

            # Add grokking point if it exists and highlight_grokking is True
            if highlight_grokking and 'grokking_phases' in self.logger.logs and 'grokking_step' in self.logger.logs[
                'grokking_phases']:
                grokking_step = self.logger.logs['grokking_phases']['grokking_step']
                if isinstance(grokking_step, list):
                    for step in grokking_step:
                        ax1.axvline(x=step, color='g', linestyle='-', alpha=0.5)
                        ax1.text(step, 0.95, f'Grokking', rotation=90, verticalalignment='top')
                else:
                    ax1.axvline(x=grokking_step, color='g', linestyle='-', alpha=0.5)
                    ax1.text(grokking_step, 0.95, f'Grokking', rotation=90, verticalalignment='top')

        # Plot loss
        ax2 = axs[1]
        if 'loss' in train_logs and 'loss' in eval_logs:
            ax2.plot(epochs, train_logs['loss'], 'b-', marker='.', alpha=0.7,
                     label='Train Loss')
            ax2.plot(epochs, eval_logs['loss'], 'r-', marker='.', alpha=0.7,
                     label='Validation Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epochs')
            ax2.legend(loc='upper right')

        # Highlight weight space jumps
        if highlight_jumps and 'weight_space_jumps' in self.logger.logs and 'jump_epochs' in self.logger.logs[
            'weight_space_jumps']:
            jump_epochs = self.logger.logs['weight_space_jumps']['jump_epochs']
            for ax in axs:
                for jump in jump_epochs:
                    ax.axvline(x=jump, color='r', linestyle='--', alpha=0.5)
                    ax.text(jump, 0.1, f'Jump', rotation=90, verticalalignment='bottom')

        # Highlight phase transitions
        if highlight_phases and 'phase_transitions' in self.logger.logs:
            # Iterate through phase transition records
            if 'transition_' in self.logger.logs['phase_transitions']:
                for key in self.logger.logs['phase_transitions']:
                    if key.startswith('transition_') and '_epoch' in key:
                        epoch = self.logger.logs['phase_transitions'][key]
                        for ax in axs:
                            ax.axvline(x=epoch, color='purple', linestyle='-.', alpha=0.5)
                            ax.text(epoch, 0.5, 'Phase\nTransition', rotation=90,
                                    verticalalignment='center', color='purple')

        plt.tight_layout()
        return fig

    def visualize_fitting_score(self, figsize=(10, 6)):
        """
        Visualize the fitting score evolution

        Returns:
            fig: Matplotlib figure object
        """
        # Check if we can calculate fitting score
        if not all(k in self.logger.logs for k in ['training', 'evaluation']):
            print("Missing training or evaluation data for fitting score")
            return None

        train_logs = self.logger.logs['training']
        eval_logs = self.logger.logs['evaluation']

        if not all(k in train_logs and k in eval_logs for k in ['accuracy', 'loss', 'epoch']):
            print("Missing accuracy, loss or epoch data")
            return None

        epochs = train_logs['epoch']
        train_acc = train_logs['accuracy']
        eval_acc = eval_logs['accuracy']
        train_loss = train_logs['loss']
        eval_loss = eval_logs['loss']

        # Normalize losses for visualization
        min_train_loss = min(train_loss)
        max_train_loss = max(train_loss)
        min_eval_loss = min(eval_loss)
        max_eval_loss = max(eval_loss)

        # Calculate fitting score
        epsilon = 1e-6
        fitting_scores = []

        for i in range(len(epochs)):
            # Normalize losses
            norm_train_loss = (train_loss[i] - min_train_loss) / (max_train_loss - min_train_loss + epsilon)
            norm_eval_loss = (eval_loss[i] - min_eval_loss) / (max_eval_loss - min_eval_loss + epsilon)

            # Calculate score as in the FittingScore class
            fitting_score = ((1 - abs(train_acc[i] - eval_acc[i])) *
                             (1 - abs(norm_train_loss - norm_eval_loss)))
            fitting_scores.append(fitting_score)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(epochs, fitting_scores, 'g-', marker='.', alpha=0.7)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Fitting Score')
        ax.set_title('Fitting Score Evolution')

        # Add grokking point if it exists
        if 'grokking_phases' in self.logger.logs and 'grokking_step' in self.logger.logs['grokking_phases']:
            grokking_step = self.logger.logs['grokking_phases']['grokking_step']
            if isinstance(grokking_step, list):
                for step in grokking_step:
                    ax.axvline(x=step, color='g', linestyle='-', alpha=0.5)
                    ax.text(step, max(fitting_scores) / 2, f'Grokking', rotation=90)
            else:
                ax.axvline(x=grokking_step, color='g', linestyle='-', alpha=0.5)
                ax.text(grokking_step, max(fitting_scores) / 2, f'Grokking', rotation=90)

        plt.tight_layout()
        return fig

    def visualize_weight_space_jumps(self, figsize=(12, 8)):
        """
        Visualize weight space jumps

        Returns:
            fig: Matplotlib figure object
        """
        # Check if jump data is available
        if 'weight_space_jumps' not in self.logger.logs:
            print("No weight space jump data found")
            return None

        jump_logs = self.logger.logs['weight_space_jumps']

        if not ('jump_epochs' in jump_logs and 'jump_velocity_norms' in jump_logs):
            print("\tLoggerVisualizer.visualize_fitting_score()\tMissing jump epoch or velocity norm data")
            return None

        jump_epochs = jump_logs['jump_epochs']
        jump_velocities = jump_logs['jump_velocity_norms']
        jump_z_scores = jump_logs.get('jump_z_scores', [])  # Optional
        jump_types = jump_logs.get('jump_types', [])  # Optional

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)

        # Plot jump velocity norms
        ax.bar(range(len(jump_epochs)), jump_velocities, alpha=0.7, color='b')
        ax.set_xlabel('Jump Index')
        ax.set_ylabel('Jump Velocity Norm')
        ax.set_title('Weight Space Jumps')

        # Add jump epochs as x-tick labels
        ax.set_xticks(range(len(jump_epochs)))
        ax.set_xticklabels([f"E{e}" for e in jump_epochs], rotation=45)

        # Add z-scores if available
        if jump_z_scores:
            ax2 = ax.twinx()
            ax2.plot(range(len(jump_epochs)), jump_z_scores, 'ro-', alpha=0.7)
            ax2.set_ylabel('Z-Score', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

        # Add jump types if available
        if jump_types:
            prev_type = None
            colors = {'strong': 'darkred', 'medium': 'orange', 'gradual': 'green'}

            for i, jtype in enumerate(jump_types):
                if jtype != prev_type:
                    ax.text(i, jump_velocities[i], jtype, rotation=45, va='bottom',
                            color=colors.get(jtype, 'blue'))
                    prev_type = jtype

        plt.tight_layout()
        return fig

    def visualize_phase_transitions(self, figsize=(14, 8)):
        """
        Visualize phase transitions with performance metrics

        Returns:
            fig: Matplotlib figure object
        """
        # Check if phase transition data is available
        found_transitions = False
        transition_epochs = []
        transition_types = []

        # Search for transition data in logger logs
        if 'phase_transitions' in self.logger.logs:
            phase_logs = self.logger.logs['phase_transitions']

            # Extract transition epochs and types
            for key in phase_logs:
                if key.endswith('_epoch'):
                    base_key = key[:-6]  # Remove "_epoch" suffix
                    if base_key + '_types' in phase_logs:
                        transition_epochs.append(phase_logs[key])
                        transition_types.append(phase_logs[base_key + '_types'])
                        found_transitions = True

        if not found_transitions:
            print("No phase transition data found")
            return None

        # Get performance data if available
        has_performance = 'training' in self.logger.logs and 'evaluation' in self.logger.logs

        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 1])

        # Create performance plot
        ax1 = fig.add_subplot(gs[0])

        if has_performance:
            train_logs = self.logger.logs['training']
            eval_logs = self.logger.logs['evaluation']

            if 'epoch' in train_logs and 'accuracy' in train_logs and 'accuracy' in eval_logs:
                epochs = train_logs['epoch']
                ax1.plot(epochs, train_logs['accuracy'], 'b-', marker='.', alpha=0.7,
                         label='Train Accuracy')
                ax1.plot(epochs, eval_logs['accuracy'], 'r-', marker='.', alpha=0.7,
                         label='Validation Accuracy')
                ax1.set_ylabel('Accuracy')
                ax1.set_ylim(0, 1.05)
                ax1.legend(loc='lower right')

        # Add transitions as vertical lines
        for i, epoch in enumerate(transition_epochs):
            type_str = transition_types[i]
            ax1.axvline(x=epoch, color='purple', linestyle='--', alpha=0.7)
            ax1.text(epoch, 0.1, f"T: {type_str}", rotation=90, va='bottom', color='purple')

        # Create phase visualization in the bottom subplot
        ax2 = fig.add_subplot(gs[1])

        # If we have phase structure data, visualize it
        if 'phases' in self.logger.logs:
            phases = self.logger.logs['phases']

            # Define colors for different phase classifications
            phase_colors = {
                'exploration': 'lightblue',
                'consolidation': 'lightgreen',
                'stability': 'lightyellow',
                'pruning': 'salmon',
                'transition': 'lightgray'
            }

            # Find min and max epochs for plot limits
            if has_performance and 'epoch' in train_logs:
                min_epoch = min(train_logs['epoch'])
                max_epoch = max(train_logs['epoch'])
            else:
                min_epoch = 0
                max_epoch = max(transition_epochs) + 100  # Add some padding

            # Add phase information as colored blocks
            for phase_id, phase_data in phases.items():
                if 'start_epoch' in phase_data and 'end_epoch' in phase_data:
                    start = phase_data['start_epoch']
                    end = phase_data['end_epoch']
                    classification = phase_data.get('classification', 'transition')
                    color = phase_colors.get(classification, 'lightgray')

                    # Draw rectangle for this phase
                    rect = Rectangle((start, 0), end - start, 1,
                                     facecolor=color, alpha=0.7, edgecolor='black')
                    ax2.add_patch(rect)

                    # Add phase label
                    ax2.text((start + end) / 2, 0.5, f"Phase {phase_id}\n{classification.title()}",
                             ha='center', va='center', fontsize=10)

            # Set limits
            ax2.set_xlim(min_epoch, max_epoch)
            ax2.set_ylim(0, 1)

            # Remove y-axis ticks
            ax2.set_yticks([])
            ax2.set_ylabel('Phases')

        ax2.set_xlabel('Epochs')

        plt.tight_layout()
        return fig

    def visualize_circuit_evolution(self, figsize=(12, 8)):
        """
        Visualize circuit evolution over time

        Returns:
            fig: Matplotlib figure object or None if no data is available
        """
        # Check for circuit data
        if 'circuit_tracking' not in self.logger.logs:
            print("No circuit tracking data found")
            return None

        circuit_logs = self.logger.logs['circuit_tracking']

        # Check for required data fields
        required_fields = ['epoch', 'active_heads_count', 'active_circuits_count']
        if not all(any(field in key for key in circuit_logs) for field in required_fields):
            print("Missing required circuit evolution data")
            return None

        # Extract epochs, active heads count, and circuit counts
        epochs = []
        head_counts = []
        circuit_counts = []

        # Organize data by epoch
        epoch_data = {}

        for key, value in circuit_logs.items():
            parts = key.split('_')
            if 'epoch' in parts:
                epoch_idx = parts.index('epoch')
                if epoch_idx + 1 < len(parts):
                    epoch = int(parts[epoch_idx + 1])
                    if epoch not in epoch_data:
                        epoch_data[epoch] = {}

                    # Store data by type
                    if 'active_heads_count' in key:
                        epoch_data[epoch]['head_count'] = value
                    elif 'active_circuits_count' in key:
                        epoch_data[epoch]['circuit_count'] = value

        # Sort by epoch and extract data
        sorted_epochs = sorted(epoch_data.keys())

        for epoch in sorted_epochs:
            epochs.append(epoch)
            head_counts.append(epoch_data[epoch].get('head_count', 0))
            circuit_counts.append(epoch_data[epoch].get('circuit_count', 0))

        # Check if we have enough data
        if len(epochs) < 2:
            print("Not enough circuit evolution data points")
            return None

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)

        # Plot counts
        ax.plot(epochs, head_counts, 'b-', marker='o', label='Active Heads')
        ax.plot(epochs, circuit_counts, 'r-', marker='s', label='Active Circuits')

        # Add connectivity evolution if available
        if 'connectivity_change' in circuit_logs:
            connectivity_changes = []
            for epoch in sorted_epochs:
                key = f'epoch_{epoch}_connectivity_change'
                if key in circuit_logs:
                    connectivity_changes.append(circuit_logs[key])
                else:
                    # If missing, use 0 or interpolate
                    connectivity_changes.append(0)

            # Scale connectivity for better visualization
            max_count = max(max(head_counts), max(circuit_counts))
            max_connectivity = max(connectivity_changes) if connectivity_changes else 1

            if max_connectivity > 0:  # Avoid division by zero
                scaled_connectivity = [c * max_count / max_connectivity for c in connectivity_changes]
                ax.plot(epochs, scaled_connectivity, 'g-', marker='^', label='Connection Changes (scaled)')

        # Add emerging and declining circuits if available
        emerging_counts = []
        declining_counts = []

        for epoch in sorted_epochs:
            # Check for emerging circuits
            e_key = f'epoch_{epoch}_emerging_circuits_count'
            d_key = f'epoch_{epoch}_declining_circuits_count'

            if e_key in circuit_logs:
                emerging_counts.append(circuit_logs[e_key])
            else:
                emerging_counts.append(0)

            if d_key in circuit_logs:
                declining_counts.append(circuit_logs[d_key])
            else:
                declining_counts.append(0)

        if any(emerging_counts) or any(declining_counts):
            ax.plot(epochs, emerging_counts, 'g--', alpha=0.7, label='Emerging Circuits')
            ax.plot(epochs, declining_counts, 'r--', alpha=0.7, label='Declining Circuits')

        # Add grokking points if available
        if 'grokking_phases' in self.logger.logs and 'grokking_step' in self.logger.logs['grokking_phases']:
            grokking_step = self.logger.logs['grokking_phases']['grokking_step']
            if isinstance(grokking_step, list):
                for step in grokking_step:
                    ax.axvline(x=step, color='g', linestyle='-', alpha=0.5)
                    ax.text(step, max(max(head_counts), max(circuit_counts)) * 0.9,
                            f'Grokking', rotation=90, va='top')
            else:
                ax.axvline(x=grokking_step, color='g', linestyle='-', alpha=0.5)
                ax.text(grokking_step, max(max(head_counts), max(circuit_counts)) * 0.9,
                        f'Grokking', rotation=90, va='top')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Count')
        ax.set_title('Circuit Evolution Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_head_attributions(self, epoch=None, figsize=(12, 8)):
        """
        Visualize head attributions at a specific epoch

        Args:
            epoch: Specific epoch to visualize (uses latest if None)

        Returns:
            fig: Matplotlib figure object
        """
        # Check for attribution data
        if 'circuit_analysis' not in self.logger.logs:
            print("No circuit analysis data found")
            return None

        circuit_logs = self.logger.logs['circuit_analysis']

        # Find relevant epoch data
        attribution_data = {}

        if epoch is None:
            # Find the latest epoch with attribution data
            latest_epoch = -1

            for key in circuit_logs:
                if 'jump_' in key and '_top' in key and '_component' in key:
                    parts = key.split('_')
                    jump_idx = parts.index('jump')
                    if jump_idx + 1 < len(parts):
                        current_epoch = int(parts[jump_idx + 1])
                        if current_epoch > latest_epoch:
                            latest_epoch = current_epoch

            if latest_epoch >= 0:
                epoch = latest_epoch
            else:
                print("No attribution data found")
                return None

        # Collect attribution data for the epoch
        for key in circuit_logs:
            if f'jump_{epoch}_top' in key:
                if '_component' in key:
                    # Extract component name
                    component_key = f"{key.replace('_component', '')}"
                    attribution_key = f"{component_key}_attribution"

                    if attribution_key in circuit_logs:
                        component = circuit_logs[key]
                        attribution = circuit_logs[attribution_key]
                        attribution_data[component] = attribution

        # Check if we found data
        if not attribution_data:
            print(f"No attribution data found for epoch {epoch}")
            return None

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by attribution score
        sorted_components = sorted(attribution_data.items(), key=lambda x: x[1], reverse=True)

        # Create bar chart
        components = [comp for comp, _ in sorted_components]
        scores = [score for _, score in sorted_components]

        # Color by component type
        colors = []
        for comp in components:
            if 'layer_' in comp and 'head_' in comp:
                colors.append('skyblue')  # Attention head
            elif 'mlp' in comp:
                colors.append('salmon')  # MLP component
            else:
                colors.append('lightgray')  # Other

        # Plot bars
        ax.bar(range(len(components)), scores, color=colors)

        # Set x-tick labels
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components, rotation=45, ha='right')

        ax.set_xlabel('Component')
        ax.set_ylabel('Attribution Score')
        ax.set_title(f'Head/Component Attribution Scores at Epoch {epoch}')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='Attention Head'),
            Patch(facecolor='salmon', label='MLP Component')
        ]
        ax.legend(handles=legend_elements)

        plt.tight_layout()
        return fig

    def visualize_mlp_sparsity(self, figsize=(12, 6)):
        """
        Visualize MLP sparsity evolution

        Returns:
            fig: Matplotlib figure object
        """
        # Check for sparsity data
        if 'mlp_sparsity' not in self.logger.logs:
            print("No MLP sparsity data found")
            return None

        sparsity_logs = self.logger.logs['mlp_sparsity']

        # Group data by layer and epoch
        layer_data = {}
        epochs = []

        for key, value in sparsity_logs.items():
            if '_sparsity' in key:
                layer = key.replace('_sparsity', '')
                if layer not in layer_data:
                    layer_data[layer] = []

                layer_data[layer].append(value)

                # Track epochs (assuming they are added in order)
                if len(layer_data[layer]) > len(epochs):
                    epochs.append(len(epochs))

        # Check if we have data
        if not layer_data or not epochs:
            print("No valid sparsity data found")
            return None

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)

        # Plot each layer's sparsity
        for layer, sparsity_values in layer_data.items():
            # Ensure lengths match
            values = sparsity_values[:len(epochs)]
            ax.plot(epochs, values, 'o-', label=layer.replace('layer_', '').replace('_mlp_expanded', ''))

        # Mark phase transitions if available
        if 'phase_transitions' in self.logger.logs:
            transitions = []

            for key, value in self.logger.logs['phase_transitions'].items():
                if key.endswith('_epoch'):
                    transitions.append(value)

            for t in transitions:
                if t <= max(epochs):
                    ax.axvline(x=t, color='r', linestyle='--', alpha=0.5)
                    ax.text(t, 0.95, f"T:{t}", transform=ax.get_xaxis_transform(),
                            rotation=90, va='top')

        # Mark grokking points if available
        if 'grokking_phases' in self.logger.logs and 'grokking_step' in self.logger.logs['grokking_phases']:
            grokking_step = self.logger.logs['grokking_phases']['grokking_step']
            if isinstance(grokking_step, list):
                for step in grokking_step:
                    if step <= max(epochs):
                        ax.axvline(x=step, color='g', linestyle='-', alpha=0.5)
                        ax.text(step, 0.95, f"G:{step}", transform=ax.get_xaxis_transform(),
                                rotation=90, va='top', color='green')
            elif grokking_step <= max(epochs):
                ax.axvline(x=grokking_step, color='g', linestyle='-', alpha=0.5)
                ax.text(grokking_step, 0.95, f"G:{grokking_step}", transform=ax.get_xaxis_transform(),
                        rotation=90, va='top', color='green')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Sparsity (% inactive neurons)')
        ax.set_title('MLP Neuron Sparsity Evolution')
        ax.legend()

        plt.tight_layout()
        return fig

    def visualize_circuit_network(self, epoch=None, figsize=(12, 10)):
        """
        Visualize the circuit network at a specific epoch

        Args:
            epoch: Specific epoch to visualize (uses latest available if None)

        Returns:
            fig: Matplotlib figure object
        """
        # Check if we have network data
        if 'circuit_analysis' not in self.logger.logs:
            print("No circuit analysis data found")
            return None

        # Find relevant epoch with circuit data
        circuit_logs = self.logger.logs['circuit_analysis']

        # Find epochs with circuit data
        epochs_with_circuits = set()

        for key in circuit_logs:
            if 'jump_' in key and '_circuit' in key and '_pair' in key:
                parts = key.split('_')
                jump_idx = parts.index('jump')
                if jump_idx + 1 < len(parts):
                    epochs_with_circuits.add(int(parts[jump_idx + 1]))

        if not epochs_with_circuits:
            print("No circuit network data found")
            return None

        # Use specified epoch or latest available
        if epoch is None:
            epoch = max(epochs_with_circuits)
        elif epoch not in epochs_with_circuits:
            print(f"No circuit data for epoch {epoch}")
            print(f"Available epochs: {sorted(epochs_with_circuits)}")
            return None

        # Collect circuit pairs and interactions
        circuit_pairs = []

        for key in circuit_logs:
            if f'jump_{epoch}_circuit' in key and '_pair' in key:
                # Find corresponding interaction key
                interaction_key = key.replace('_pair', '_interaction')
                if interaction_key in circuit_logs:
                    pair = circuit_logs[key]
                    interaction = circuit_logs[interaction_key]
                    circuit_pairs.append((pair, interaction))

        # Create visualization if we have data
        if not circuit_pairs:
            print(f"No circuit connections found for epoch {epoch}")
            return None

        # Create networkx graph
        G = nx.Graph()

        # Add nodes and edges
        for pair, interaction in circuit_pairs:
            # Parse the pair string (could be various formats)
            if isinstance(pair, str):
                if '+' in pair:
                    head1, head2 = pair.split('+')
                else:
                    parts = pair.split('_')
                    if len(parts) >= 2:
                        # Try to extract two components
                        midpoint = len(parts) // 2
                        head1 = '_'.join(parts[:midpoint])
                        head2 = '_'.join(parts[midpoint:])
                    else:
                        continue  # Skip if can't parse
            else:
                continue  # Skip if not a valid pair

            # Add nodes
            if head1 not in G.nodes():
                G.add_node(head1, type='head')
            if head2 not in G.nodes():
                G.add_node(head2, type='head')

            # Add edge with interaction weight
            G.add_edge(head1, head2, weight=interaction)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Set positions using spring layout
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

        # Get node sizes based on degree centrality
        node_centrality = nx.degree_centrality(G)
        node_sizes = [node_centrality[n] * 3000 + 200 for n in G.nodes()]

        # Color nodes by layer
        node_colors = []
        for n in G.nodes():
            if 'layer_' in n:
                # Extract layer number
                try:
                    layer = int(n.split('_')[1])
                    # Create a colormap based on layer
                    node_colors.append(plt.cm.viridis(layer / 10))  # Assuming max 10 layers
                except (ValueError, IndexError):
                    node_colors.append('gray')
            else:
                node_colors.append('gray')

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

        # Draw edges with width proportional to interaction strength
        edge_widths = [G[u][v]['weight'] * 2 + 1 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')

        # Draw labels with smaller font size for better readability
        nx.draw_networkx_labels(G, pos, font_size=8)

        ax.set_title(f'Circuit Network at Epoch {epoch}')
        ax.axis('off')

        plt.tight_layout()
        return fig

    def save_visualization(self, fig, filename=None, dpi=300):
        """
        Save a visualization figure to file

        Args:
            fig: Matplotlib figure object
            filename: Output filename (will be auto-generated if None)
            dpi: Resolution in dots per inch

        Returns:
            str: Path to saved file
        """
        if fig is None:
            return None

        # Generate filename if not provided
        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"visualization_{timestamp}.png"

        # Ensure directory exists
        filepath = self.save_dir / filename

        # Save figure
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        return str(filepath)

    def create_dashboard(self, output_dir=None):
        """
        Create a comprehensive dashboard with all available visualizations

        Args:
            output_dir: Directory to save dashboard files (uses save_dir if None)

        Returns:
            str: Path to dashboard HTML file
        """
        if output_dir is None:
            output_dir = self.save_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

        # Create dashboard directory
        dashboard_dir = output_dir / "dashboard"
        dashboard_dir.mkdir(exist_ok=True, parents=True)

        # Generate all visualizations
        visualizations = {}

        # Training metrics
        fig = self.visualize_training_metrics()
        if fig:
            path = self.save_visualization(fig, "training_metrics.png")
            visualizations['training_metrics'] = path

        # Fitting score
        fig = self.visualize_fitting_score()
        if fig:
            path = self.save_visualization(fig, "fitting_score.png")
            visualizations['fitting_score'] = path

        # Weight space jumps
        fig = self.visualize_weight_space_jumps()
        if fig:
            path = self.save_visualization(fig, "weight_space_jumps.png")
            visualizations['weight_space_jumps'] = path

        # Phase transitions
        fig = self.visualize_phase_transitions()
        if fig:
            path = self.save_visualization(fig, "phase_transitions.png")
            visualizations['phase_transitions'] = path

        # Circuit evolution
        fig = self.visualize_circuit_evolution()
        if fig:
            path = self.save_visualization(fig, "circuit_evolution.png")
            visualizations['circuit_evolution'] = path

        # Head attributions
        fig = self.visualize_head_attributions()
        if fig:
            path = self.save_visualization(fig, "head_attributions.png")
            visualizations['head_attributions'] = path

        # MLP sparsity
        fig = self.visualize_mlp_sparsity()
        if fig:
            path = self.save_visualization(fig, "mlp_sparsity.png")
            visualizations['mlp_sparsity'] = path

        # Circuit network
        fig = self.visualize_circuit_network()
        if fig:
            path = self.save_visualization(fig, "circuit_network.png")
            visualizations['circuit_network'] = path

        # Generate HTML dashboard
        dashboard_html = self._generate_dashboard_html(visualizations)

        # Write HTML to file
        dashboard_path = dashboard_dir / "dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)

        return str(dashboard_path)

    def _generate_dashboard_html(self, visualizations):
        """
        Generate HTML for the dashboard

        Args:
            visualizations: Dictionary of visualization paths

        Returns:
            str: HTML content
        """
        # Experiment name
        experiment_name = self.logger.id if hasattr(self.logger, 'id') else "Transformer Learning Analysis"

        # HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{experiment_name} Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .dashboard-header {{
                    background-color: #343a40;
                    color: white;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }}
                .dashboard-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .visualization-card {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                    flex: 1 1 calc(50% - 20px);
                    min-width: 500px;
                    margin-bottom: 20px;
                }}
                .card-header {{
                    background-color: #007bff;
                    color: white;
                    padding: 10px 15px;
                }}
                .card-body {{
                    padding: 15px;
                    text-align: center;
                }}
                .visualization-img {{
                    max-width: 100%;
                    height: auto;
                }}
                .metrics-container {{
                    margin-top: 20px;
                    background-color: white;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                .metrics-header {{
                    background-color: #28a745;
                    color: white;
                    padding: 10px 15px;
                    border-radius: 5px 5px 0 0;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 15px;
                    padding: 15px;
                }}
                .metric-item {{
                    background-color: #e9ecef;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #007bff;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #6c757d;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>{experiment_name}</h1>
                <p>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        """

        # Add key metrics section if we have training data
        if 'training' in self.logger.logs and 'evaluation' in self.logger.logs:
            train_logs = self.logger.logs['training']
            eval_logs = self.logger.logs['evaluation']

            if 'accuracy' in train_logs and 'accuracy' in eval_logs:
                final_train_acc = train_logs['accuracy'][-1] if train_logs['accuracy'] else 0
                final_eval_acc = eval_logs['accuracy'][-1] if eval_logs['accuracy'] else 0
                best_eval_acc = max(eval_logs['accuracy']) if eval_logs['accuracy'] else 0
                best_epoch = eval_logs['epoch'][eval_logs['accuracy'].index(best_eval_acc)] if eval_logs[
                    'accuracy'] else 0

                html += f"""
                <div class="metrics-container">
                    <div class="metrics-header">
                        <h2>Key Metrics</h2>
                    </div>
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <div class="metric-value">{final_train_acc:.4f}</div>
                            <div class="metric-label">Final Train Accuracy</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">{final_eval_acc:.4f}</div>
                            <div class="metric-label">Final Eval Accuracy</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">{best_eval_acc:.4f}</div>
                            <div class="metric-label">Best Eval Accuracy</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">{best_epoch}</div>
                            <div class="metric-label">Best Epoch</div>
                        </div>
                """

                # Add grokking metric if available
                if 'grokking_phases' in self.logger.logs and 'grokking_step' in self.logger.logs['grokking_phases']:
                    grokking_step = self.logger.logs['grokking_phases']['grokking_step']
                    if isinstance(grokking_step, list):
                        grokking_steps_str = ", ".join([str(step) for step in grokking_step])
                        html += f"""
                        <div class="metric-item">
                            <div class="metric-value">{grokking_steps_str}</div>
                            <div class="metric-label">Grokking Epochs</div>
                        </div>
                        """
                    else:
                        html += f"""
                        <div class="metric-item">
                            <div class="metric-value">{grokking_step}</div>
                            <div class="metric-label">Grokking Epoch</div>
                        </div>
                        """

                # Add phase transition count if available
                if 'phase_transitions' in self.logger.logs:
                    transition_count = sum(1 for key in self.logger.logs['phase_transitions'] if key.endswith('_epoch'))
                    html += f"""
                    <div class="metric-item">
                        <div class="metric-value">{transition_count}</div>
                        <div class="metric-label">Phase Transitions</div>
                    </div>
                    """

                # Add weight space jump count if available
                if 'weight_space_jumps' in self.logger.logs and 'jump_epochs' in self.logger.logs['weight_space_jumps']:
                    jump_count = len(self.logger.logs['weight_space_jumps']['jump_epochs'])
                    html += f"""
                    <div class="metric-item">
                        <div class="metric-value">{jump_count}</div>
                        <div class="metric-label">Weight Space Jumps</div>
                    </div>
                    """

                html += """
                    </div>
                </div>
                """

        # Add visualization section
        html += """
        <div class="dashboard-container">
        """

        # Add visualizations
        for viz_type, viz_path in visualizations.items():
            if viz_path:
                title = viz_type.replace('_', ' ').title()
                # Convert absolute path to relative
                rel_path = Path(viz_path).relative_to(self.save_dir)
                html += f"""
                <div class="visualization-card">
                    <div class="card-header">
                        <h3>{title}</h3>
                    </div>
                    <div class="card-body">
                        <img class="visualization-img" src="../{rel_path}" alt="{title}">
                    </div>
                </div>
                """

        # Close container and body
        html += """
        </div>
        </body>
        </html>
        """

        return html


# Example usage:
"""
# Create a logger
from analysis.core.logger import DataLogger
logger = DataLogger(id="experiment1")

# Add some metrics
for epoch in range(100):
    logger.log_data('training', 'epoch', epoch)
    logger.log_data('training', 'accuracy', 0.5 + 0.005 * epoch)
    logger.log_data('training', 'loss', 1.0 - 0.01 * epoch)

    logger.log_data('evaluation', 'epoch', epoch)
    logger.log_data('evaluation', 'accuracy', 0.4 + 0.006 * epoch)
    logger.log_data('evaluation', 'loss', 1.2 - 0.01 * epoch)

# Create a visualizer
visualizer = LoggerVisualizer(logger, save_dir="visualizations")

# Generate dashboard
dashboard_path = visualizer.create_dashboard()
print(f"Dashboard created at {dashboard_path}")
"""