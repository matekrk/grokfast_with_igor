from pathlib import Path

import numpy as np
import torch


class ContinuousCircuitTracker:
    """
    Extension for CircuitAnalyzer that continuously tracks circuit formation and evolution
    throughout the training process, rather than only at detected jumps.

    This tracker works alongside the existing CircuitAnalyzer to provide a temporal
    perspective on circuit development.
    """

    def __init__(self, model, save_dir, logger=None, sampling_freq=20,
                 min_attribution=0.01, history_length=10):
        """
        Initialize the continuous circuit tracker.

        Args:
            model: The transformer model being analyzed
            save_dir: Directory to save analysis results
            logger: Optional logger instance
            sampling_freq: Frequency (in epochs) to sample circuit properties
            min_attribution: Minimum attribution score to consider a head part of a circuit
            history_length: Number of past circuit states to maintain
        """
        self.model = model
        self.save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logger if logger else (model.logger if hasattr(model, 'logger') else None)

        # Analysis settings
        self.sampling_freq = sampling_freq
        self.min_attribution = min_attribution
        self.history_length = history_length

        # Create directory for circuit tracking
        self.circuits_dir = self.save_dir / "continuous_circuits"
        self.circuits_dir.mkdir(exist_ok=True, parents=True)

        # Initialize storage for circuit history with all required fields
        self.circuit_history = {
            'epochs': [],
            'active_heads': [],  # Heads that exceed min_attribution at each epoch
            'head_attributions': [],  # Attribution scores for all heads at each epoch
            'active_circuits': [],  # Detected circuits at each epoch
            'circuit_strengths': [],  # Strengths of each active circuit at each epoch
            'head_stability': {},  # Tracks how stable each head's attribution is
            'circuit_stability': {},  # Tracks how stable each circuit's existence is
            'emerging_circuits': [],  # Circuits that are beginning to form
            'declining_circuits': [],  # Circuits that are weakening
            'stable_circuits': []  # Circuits that have remained consistent
        }

        # Track connection matrices for spectral analysis
        self.connection_matrices = []
        self.connectivity_evolution = []

        # For tracking behavioral impact of circuits
        self.behavioral_impact = {
            'performance_deltas': [],
            'circuit_to_performance': {}
        }

    def _ensure_history_consistency(self):
        """
        Ensure that all lists in the circuit history are of consistent length.
        This helps prevent index errors when accessing history elements.
        """
        if not self.circuit_history['epochs']:
            return  # No history yet

        # Get reference length from epochs
        ref_length = len(self.circuit_history['epochs'])

        # Check and fix all list elements
        for key, value in self.circuit_history.items():
            if isinstance(value, list):
                # Skip non-list elements and the reference (epochs)
                if key == 'epochs':
                    continue

                current_length = len(value)

                if current_length < ref_length:
                    # Too short - pad with empty/default values
                    if key == 'active_heads' or key == 'active_circuits':
                        # Pad with empty lists
                        value.extend([[] for _ in range(ref_length - current_length)])
                    elif key == 'head_attributions' or key == 'circuit_strengths':
                        # Pad with empty dictionaries
                        value.extend([{} for _ in range(ref_length - current_length)])
                    elif key == 'emerging_circuits' or key == 'declining_circuits' or key == 'stable_circuits':
                        # Pad with empty lists
                        value.extend([[] for _ in range(ref_length - current_length)])
                    else:
                        # General case - pad with None
                        value.extend([None for _ in range(ref_length - current_length)])

                elif current_length > ref_length:
                    # Too long - truncate, but keep the MOST RECENT elements
                    # We want to keep elements [-ref_length:] not [:ref_length]
                    del value[:-ref_length]

        # Also fix connection matrices and connectivity evolution - keep most recent elements
        if len(self.connection_matrices) > ref_length:
            self.connection_matrices = self.connection_matrices[-ref_length:]

        if len(self.connectivity_evolution) > ref_length:
            self.connectivity_evolution = self.connectivity_evolution[-ref_length:]

    def sample_circuits(self, epoch, eval_loader, baseline_acc=None,
                        sample_behavioral=True, circuit_threshold=0.01):
        """
        Sample the current state of circuits in the model.

        Args:
            epoch: Current training epoch
            eval_loader: Evaluation data loader
            baseline_acc: Optional baseline accuracy (will be computed if None)
            sample_behavioral: Whether to measure behavioral impact of circuits
            circuit_threshold: Threshold for circuit identification

        Returns:
            dict: Current circuit analysis results
        """
        # Check if we've already sampled this epoch
        if self.circuit_history['epochs'] and epoch in self.circuit_history['epochs']:
            # Find the index of this epoch
            epoch_idx = self.circuit_history['epochs'].index(epoch)

            # Return the existing data
            return {
                'epoch': epoch,
                'active_heads': self.circuit_history['active_heads'][epoch_idx],
                'head_attributions': self.circuit_history['head_attributions'][epoch_idx],
                'active_circuits': self.circuit_history['active_circuits'][epoch_idx],
                'circuit_strengths': (self.circuit_history['circuit_strengths'][epoch_idx]
                                      if epoch_idx < len(self.circuit_history['circuit_strengths'])
                                      else {})
            }

        # Only sample at specified frequency
        if epoch % self.sampling_freq != 0:
            return None

        # Ensure we don't exceed history length
        if len(self.circuit_history['epochs']) >= self.history_length:
            for key in self.circuit_history:
                if isinstance(self.circuit_history[key], list):
                    if len(self.circuit_history[key]) > 0:
                        self.circuit_history[key].pop(0)

            # Also trim connection matrices
            if len(self.connection_matrices) > self.history_length:
                self.connection_matrices.pop(0)
                self.connectivity_evolution.pop(0)

        # Store the current epoch
        self.circuit_history['epochs'].append(epoch)

        # Analyze model using core circuit analysis methods
        individual_results, pairwise_results = self._analyze_current_circuits(
            eval_loader, baseline_acc, circuit_threshold)

        # Track active heads
        active_heads = [head for head, score in individual_results.items()
                        if score >= self.min_attribution]
        self.circuit_history['active_heads'].append(active_heads)
        self.circuit_history['head_attributions'].append(individual_results)

        # Identify active circuits
        active_circuits = []
        circuit_strengths = {}

        for pair, data in pairwise_results.items():
            if data['circuit_strength'] > circuit_threshold and data['is_circuit']:
                # Convert tuple to string format if needed
                if isinstance(pair, tuple):
                    circuit_key = '+'.join(pair)
                else:
                    circuit_key = pair

                active_circuits.append(circuit_key)
                circuit_strengths[circuit_key] = data['circuit_strength']

        self.circuit_history['active_circuits'].append(active_circuits)
        self.circuit_history['circuit_strengths'].append(circuit_strengths)

        # Update stability metrics
        self._update_stability_metrics()

        # Perform connection matrix analysis
        connection_matrix = self._build_connection_matrix(individual_results, pairwise_results)
        self.connection_matrices.append(connection_matrix)

        # Track connectivity evolution
        if len(self.connection_matrices) > 1:
            connectivity_delta = np.sum(np.abs(
                self.connection_matrices[-1] - self.connection_matrices[-2]))
            self.connectivity_evolution.append(connectivity_delta)
        else:
            self.connectivity_evolution.append(0.0)

        # Optionally measure behavioral impact
        if sample_behavioral:
            self._measure_behavioral_impact(active_circuits, circuit_strengths, eval_loader)

        # Ensure history consistency
        self._ensure_history_consistency()

        # Generate visualization if there's enough history
        if len(self.circuit_history['epochs']) >= 2:
            self._visualize_circuit_evolution()

        # Log key metrics
        if self.logger:
            # Log count of active heads and circuits
            self.logger.log_data('circuit_tracking', f'epoch_{epoch}_active_heads_count',
                                 len(active_heads))
            self.logger.log_data('circuit_tracking', f'epoch_{epoch}_active_circuits_count',
                                 len(active_circuits))

            # Log top circuit strength
            if active_circuits:
                top_circuit = max(circuit_strengths.items(), key=lambda x: x[1])
                self.logger.log_data('circuit_tracking', f'epoch_{epoch}_top_circuit',
                                     top_circuit[0])
                self.logger.log_data('circuit_tracking', f'epoch_{epoch}_top_circuit_strength',
                                     top_circuit[1])

            # Log connectivity change
            if len(self.connectivity_evolution) > 0:
                self.logger.log_data('circuit_tracking', f'epoch_{epoch}_connectivity_change',
                                     self.connectivity_evolution[-1])

        if active_circuits:
            print(f"\tContinuousCircuitTracker.sample_circuits() @ {epoch}: \t{active_circuits}")
        return {
            'epoch': epoch,
            'active_heads': active_heads,
            'head_attributions': individual_results,
            'active_circuits': active_circuits,
            'circuit_strengths': circuit_strengths,
            'connectivity_change': self.connectivity_evolution[-1] if self.connectivity_evolution else 0.0
        }

    def _analyze_current_circuits(self, eval_loader, baseline_acc=None, threshold=0.01):
        """
        Analyze the current state of circuits in the model.

        Args:
            eval_loader: Evaluation data loader
            baseline_acc: Optional baseline accuracy (will be computed if None)
            threshold: Threshold for significance

        Returns:
            tuple: (individual_results, pairwise_results)
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

        # Analyze pairwise interactions if we have enough significant heads
        pairwise_results = {}
        if len(significant_heads) >= 2:
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
                    pair_key = (head1, head2)
                    pairwise_results[pair_key] = {
                        'expected_drop': expected_drop,
                        'actual_drop': actual_drop,
                        'circuit_strength': circuit_strength,
                        'is_circuit': circuit_strength > threshold  # Positive interaction indicates circuit
                    }

                    # Restore weights
                    with torch.no_grad():
                        self.model.layers[layer_idx1].attn.out_proj.weight.copy_(original_weights1)
                        self.model.layers[layer_idx2].attn.out_proj.weight.copy_(original_weights2)

        return individual_results, pairwise_results

    def _update_stability_metrics(self):
        """Update metrics that track the stability of heads and circuits over time"""
        # Need at least 2 timepoints to measure stability
        if len(self.circuit_history['epochs']) < 2:
            return

        # Ensure history lists have consistent lengths
        self._ensure_history_consistency()

        # Get the most recent and previous index
        latest_idx = len(self.circuit_history['epochs']) - 1
        prev_idx = latest_idx - 1

        # Safely access attribution data
        if latest_idx < len(self.circuit_history['head_attributions']) and prev_idx < len(
                self.circuit_history['head_attributions']):
            latest_attributions = self.circuit_history['head_attributions'][latest_idx]
            previous_attributions = self.circuit_history['head_attributions'][prev_idx]

            # Update head stability metrics
            for head in latest_attributions:
                if head in previous_attributions:
                    # Calculate relative change in attribution
                    prev_value = previous_attributions[head]
                    curr_value = latest_attributions[head]

                    if prev_value > 0:
                        rel_change = abs(curr_value - prev_value) / prev_value
                    else:
                        rel_change = 1.0 if curr_value > 0 else 0.0

                    # Update stability metric (1.0 = completely stable, 0.0 = completely unstable)
                    stability = max(0.0, 1.0 - rel_change)

                    if head not in self.circuit_history['head_stability']:
                        self.circuit_history['head_stability'][head] = []

                    self.circuit_history['head_stability'][head].append(stability)
                else:
                    # New head that wasn't active before
                    if head not in self.circuit_history['head_stability']:
                        self.circuit_history['head_stability'][head] = []

                    self.circuit_history['head_stability'][head].append(0.0)  # New head is unstable

        # Safely access circuit data
        if latest_idx < len(self.circuit_history['active_circuits']) and prev_idx < len(
                self.circuit_history['active_circuits']):
            # Update circuit stability metrics
            latest_circuits = set(self.circuit_history['active_circuits'][latest_idx])
            previous_circuits = set(self.circuit_history['active_circuits'][prev_idx])

            # Identify circuits by their changes
            emerging = latest_circuits - previous_circuits
            declining = previous_circuits - latest_circuits
            stable = latest_circuits.intersection(previous_circuits)

            # Ensure these lists exist
            if 'emerging_circuits' not in self.circuit_history:
                self.circuit_history['emerging_circuits'] = []
            if 'declining_circuits' not in self.circuit_history:
                self.circuit_history['declining_circuits'] = []
            if 'stable_circuits' not in self.circuit_history:
                self.circuit_history['stable_circuits'] = []

            # Append the new data
            self.circuit_history['emerging_circuits'].append(list(emerging))
            self.circuit_history['declining_circuits'].append(list(declining))
            self.circuit_history['stable_circuits'].append(list(stable))

            # Ensure these lists have the correct length
            while len(self.circuit_history['emerging_circuits']) < len(self.circuit_history['epochs']):
                self.circuit_history['emerging_circuits'].insert(0, [])

            while len(self.circuit_history['declining_circuits']) < len(self.circuit_history['epochs']):
                self.circuit_history['declining_circuits'].insert(0, [])

            while len(self.circuit_history['stable_circuits']) < len(self.circuit_history['epochs']):
                self.circuit_history['stable_circuits'].insert(0, [])

            # For stable circuits, calculate strength stability
            if latest_idx < len(self.circuit_history['circuit_strengths']) and prev_idx < len(
                    self.circuit_history['circuit_strengths']):
                latest_strengths = self.circuit_history['circuit_strengths'][latest_idx]
                previous_strengths = self.circuit_history['circuit_strengths'][prev_idx]

                for circuit in stable:
                    if circuit in latest_strengths and circuit in previous_strengths:
                        # Calculate relative change in strength
                        prev_strength = previous_strengths[circuit]
                        curr_strength = latest_strengths[circuit]

                        if prev_strength > 0:
                            rel_change = abs(curr_strength - prev_strength) / prev_strength
                        else:
                            rel_change = 1.0

                        # Update stability metric
                        stability = max(0.0, 1.0 - rel_change)

                        if circuit not in self.circuit_history['circuit_stability']:
                            self.circuit_history['circuit_stability'][circuit] = []

                        self.circuit_history['circuit_stability'][circuit].append(stability)

    def _build_connection_matrix(self, individual_results, pairwise_results):
        """
        Build a connection matrix representing the circuit structure.

        This matrix can be used for spectral analysis of circuit topology.

        Args:
            individual_results: Individual head attribution scores
            pairwise_results: Pairwise interaction results

        Returns:
            numpy.ndarray: Connection matrix
        """
        # Determine the number of heads
        heads = list(individual_results.keys())
        n_heads = len(heads)

        # Create a mapping from head name to index
        head_to_idx = {head: i for i, head in enumerate(heads)}

        # Initialize connection matrix with individual attributions on diagonal
        connection_matrix = np.zeros((n_heads, n_heads))
        for head, score in individual_results.items():
            if head in head_to_idx:
                idx = head_to_idx[head]
                connection_matrix[idx, idx] = score

        # Add pairwise interactions to off-diagonal elements
        for pair, data in pairwise_results.items():
            # Handle both tuple and string formats
            if isinstance(pair, tuple):
                head1, head2 = pair
            else:
                # For string format with '+' separator
                parts = pair.split('+')
                if len(parts) == 2:
                    head1, head2 = parts
                else:
                    continue  # Skip if format is unexpected

            if head1 in head_to_idx and head2 in head_to_idx:
                idx1 = head_to_idx[head1]
                idx2 = head_to_idx[head2]

                # Use circuit strength for off-diagonal elements
                # Only if there's a positive circuit effect (super-additivity)
                if data['circuit_strength'] > 0 and data.get('is_circuit', False):
                    connection_matrix[idx1, idx2] = data['circuit_strength']
                    connection_matrix[idx2, idx1] = data['circuit_strength']  # Symmetric

        return connection_matrix

    def _measure_behavioral_impact(self, active_circuits, circuit_strengths, eval_loader):
        """
        Measure the behavioral impact of each circuit on model performance.

        Args:
            active_circuits: List of active circuits
            circuit_strengths: Dictionary mapping circuits to their strengths
            eval_loader: Evaluation data loader
        """
        if not active_circuits:
            return

        # Get baseline performance
        baseline_acc, _ = self.model.evaluate(eval_loader)

        # Store original model state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Measure impact of top circuits
        circuit_impacts = {}

        try:
            # Sort circuits by strength
            sorted_circuits = sorted(circuit_strengths.items(), key=lambda x: x[1], reverse=True)

            # Test top 5 circuits or all if fewer
            top_circuits = sorted_circuits[:min(5, len(sorted_circuits))]

            for circuit_key, strength in top_circuits:
                # Parse the circuit components
                if '+' in circuit_key:
                    head1, head2 = circuit_key.split('+')
                else:
                    head1, head2 = circuit_key

                # Get layer and head indices
                layer_idx1 = int(head1.split('_')[1])
                head_idx1 = int(head1.split('_')[3])
                layer_idx2 = int(head2.split('_')[1])
                head_idx2 = int(head2.split('_')[3])

                # Functionally remove this circuit by zeroing head connections
                head_dim = self.model.dim // self.model.num_heads

                with torch.no_grad():
                    # Zero first head
                    start_idx1 = head_idx1 * head_dim
                    end_idx1 = (head_idx1 + 1) * head_dim
                    self.model.layers[layer_idx1].attn.out_proj.weight[:, start_idx1:end_idx1] = 0

                    # Zero second head
                    start_idx2 = head_idx2 * head_dim
                    end_idx2 = (head_idx2 + 1) * head_dim
                    self.model.layers[layer_idx2].attn.out_proj.weight[:, start_idx2:end_idx2] = 0

                # Measure performance without this circuit
                circuit_acc, _ = self.model.evaluate(eval_loader)

                # Calculate impact
                impact = baseline_acc - circuit_acc
                circuit_impacts[circuit_key] = impact

                # Restore original state after each test
                self.model.load_state_dict(original_state)

        finally:
            # Ensure original state is restored
            self.model.load_state_dict(original_state)

        # Store circuit impacts
        self.behavioral_impact['performance_deltas'].append(circuit_impacts)

        # Update mapping from circuits to performance
        for circuit, impact in circuit_impacts.items():
            if circuit not in self.behavioral_impact['circuit_to_performance']:
                self.behavioral_impact['circuit_to_performance'][circuit] = []

            self.behavioral_impact['circuit_to_performance'][circuit].append(impact)

    def _visualize_circuit_evolution(self):
        """Create visualizations of circuit evolution over time"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import networkx as nx
        from matplotlib.lines import Line2D

        # Only create visualizations if we have enough history
        if len(self.circuit_history['epochs']) < 2:
            return

        # Ensure history consistency before visualization
        self._ensure_history_consistency()

        current_epoch = self.circuit_history['epochs'][-1]

        # 1. Create circuit evolution timeline
        fig1, ax1 = plt.subplots(figsize=(12, 8))

        # Extract data for plotting
        epochs = self.circuit_history['epochs']
        active_heads_counts = [len(heads) for heads in self.circuit_history['active_heads']]
        active_circuit_counts = [len(circuits) for circuits in self.circuit_history['active_circuits']]

        # Plot counts
        ax1.plot(epochs, active_heads_counts, 'b-', marker='o', label='Active Heads')
        ax1.plot(epochs, active_circuit_counts, 'r-', marker='s', label='Active Circuits')

        # Add connectivity evolution if available
        if self.connectivity_evolution:
            # Make sure connectivity evolution has the same length as epochs
            if len(self.connectivity_evolution) < len(epochs):
                # Pad with zeros at the beginning if needed
                self.connectivity_evolution = [0] * (
                        len(epochs) - len(self.connectivity_evolution)) + self.connectivity_evolution
            elif len(self.connectivity_evolution) > len(epochs):
                # Truncate if too long
                self.connectivity_evolution = self.connectivity_evolution[-len(epochs):]

            # Scale connectivity for better visualization
            max_count = max(max(active_heads_counts) if active_heads_counts else 1,
                            max(active_circuit_counts) if active_circuit_counts else 1)
            max_connectivity = max(self.connectivity_evolution) if self.connectivity_evolution else 1

            if max_connectivity > 0:  # Avoid division by zero
                scaled_connectivity = [c * max_count / max_connectivity for c in self.connectivity_evolution]
                ax1.plot(epochs, scaled_connectivity, 'g-', marker='^', label='Connection Changes (scaled)')

        # Add emerging and declining circuits if available
        emerging_counts = []
        declining_counts = []

        # Ensure emerging_circuits and declining_circuits have correct lengths
        if 'emerging_circuits' in self.circuit_history and 'declining_circuits' in self.circuit_history:
            # Make sure these lists are the same length as epochs
            for key in ['emerging_circuits', 'declining_circuits']:
                if len(self.circuit_history[key]) < len(epochs):
                    # Pad with empty lists at the beginning
                    self.circuit_history[key] = [[]] * (len(epochs) - len(self.circuit_history[key])) + \
                                                self.circuit_history[key]
                elif len(self.circuit_history[key]) > len(epochs):
                    # Truncate if too long
                    self.circuit_history[key] = self.circuit_history[key][-len(epochs):]

            # Now extract counts safely
            emerging_counts = [len(circuits) for circuits in self.circuit_history['emerging_circuits']]
            declining_counts = [len(circuits) for circuits in self.circuit_history['declining_circuits']]

            # Plot emerging and declining counts
            ax1.plot(epochs, emerging_counts, 'g--', alpha=0.7, label='Emerging Circuits')
            ax1.plot(epochs, declining_counts, 'r--', alpha=0.7, label='Declining Circuits')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Circuit Evolution Timeline (up to epoch {current_epoch})')
        ax1.set_yscale('log')

        # info display y-scale in decimal, not log, scale
        actual_ticks = [1, 2, 3, 5, 8, 10, 12, 14]
        ax1.set_yticks(actual_ticks)
        ax1.set_yticklabels([f"{i}" for i in range(1, len(actual_ticks) + 1)])

        ax1.legend()
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(f"{self.model.plot_prefix}")
        plt.savefig(self.circuits_dir / f'circuit_evolution_timeline_epoch_{current_epoch}.png')
        plt.close(fig1)

        # 2. Create circuit network visualization for most recent epoch
        if active_circuit_counts and active_circuit_counts[-1] > 0:
            fig2, ax2 = plt.subplots(figsize=(12, 10))

            # Create a graph representation
            G = nx.Graph()

            # Add nodes for active heads
            head_attributions = self.circuit_history['head_attributions'][-1]
            active_heads = self.circuit_history['active_heads'][-1]

            for head in active_heads:
                attribution = head_attributions.get(head, 0.0)
                G.add_node(head, size=attribution, type='head')

            # Add edges for active circuits
            active_circuits = self.circuit_history['active_circuits'][-1]
            circuit_strengths = self.circuit_history['circuit_strengths'][-1] if len(
                self.circuit_history['circuit_strengths']) > 0 else {}

            for circuit in active_circuits:
                if '+' in circuit:
                    head1, head2 = circuit.split('+')
                else:
                    # Handle tuple format if encountered
                    head1, head2 = circuit

                strength = circuit_strengths.get(circuit, 0.0)
                G.add_edge(head1, head2, weight=strength)

            # Proceed only if we have a non-empty graph
            if G.number_of_nodes() > 0:
                # Set positions using spring layout
                pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

                # Get node attributes for visualization
                node_sizes = [G.nodes[n].get('size', 0.0) * 3000 + 100 for n in G.nodes()]

                # Color nodes by layer
                node_colors = []
                for n in G.nodes():
                    layer = int(n.split('_')[1])
                    # Create a colormap based on layer
                    node_colors.append(plt.cm.viridis(layer / max(1, self.model.num_layers - 1)))

                # Draw nodes
                nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

                # Draw edges if any
                if G.number_of_edges() > 0:
                    # Draw edges with width proportional to circuit strength
                    edge_widths = [G[u][v].get('weight', 1.0) * 5 + 1 for u, v in G.edges()]
                    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')

                # Draw labels
                nx.draw_networkx_labels(G, pos, font_size=8)

                # Add legends for interpretability
                layer_legend_elements = []
                for i in range(self.model.num_layers):
                    color = plt.cm.viridis(i / max(1, self.model.num_layers - 1))
                    layer_legend_elements.append(
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                               markersize=10, label=f'Layer {i}')
                    )

                size_legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                           markersize=8, label='Low Attribution'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                           markersize=15, label='High Attribution')
                ]

                # Create two separate legends
                layer_legend = plt.legend(handles=layer_legend_elements, loc='upper left',
                                          title="Layers", bbox_to_anchor=(1.01, 1))
                plt.gca().add_artist(layer_legend)
                plt.legend(handles=size_legend_elements, loc='upper left',
                           title="Attribution", bbox_to_anchor=(1.01, 0.8))

                plt.title(f'Circuit Network at Epoch {current_epoch}')
                plt.axis('off')
                plt.tight_layout()
                plt.suptitle(f"{self.model.plot_prefix}")
                plt.savefig(self.circuits_dir / f'circuit_network_epoch_{current_epoch}.png')

            plt.close(fig2)

        # 3. Create heatmap of connection matrix
        if self.connection_matrices:
            fig3, ax3 = plt.subplots(figsize=(10, 8))

            # Get most recent connection matrix
            conn_matrix = self.connection_matrices[-1]

            # Make sure the matrix isn't empty
            if conn_matrix.size > 0:
                # Create heatmap
                sns.heatmap(conn_matrix, cmap='coolwarm', center=0, ax=ax3)

            # Get active heads for labels
            active_heads = self.circuit_history['active_heads'][-1]

            # Simplify labels for readability
            labels = []
            for head in active_heads:
                parts = head.split('_')
                simplified = f"L{parts[1]}H{parts[3]}"
                labels.append(simplified)

            # Set labels if not too many
            if len(labels) <= 20:  # Only set labels if not too crowded
                ax3.set_xticks(np.arange(len(labels)) + 0.5)
                ax3.set_yticks(np.arange(len(labels)) + 0.5)
                ax3.set_xticklabels(labels, rotation=45, ha='right')
                ax3.set_yticklabels(labels)

            ax3.set_title(f'Circuit Connection Matrix at Epoch {current_epoch}')

            plt.tight_layout()
            plt.suptitle(f"{self.model.plot_prefix}")
            plt.savefig(self.circuits_dir / f'connection_matrix_epoch_{current_epoch}.png')

            plt.close(fig3)

        return self.circuits_dir

    def get_circuit_evolution_summary(self):
        """
        Generate a summary of circuit evolution over the tracked history.

        Returns:
            dict: Summary statistics and insights
        """
        if not self.circuit_history['epochs']:
            return {"status": "No circuit data collected yet"}

        summary = {
            "epochs_analyzed": self.circuit_history['epochs'],
            "active_heads_trend": [],
            "active_circuits_trend": [],
            "most_stable_heads": [],
            "most_stable_circuits": [],
            "recently_emerged_circuits": [],
            "behavioral_impact": {}
        }

        # Calculate trends in active components
        for i in range(len(self.circuit_history['epochs'])):
            summary["active_heads_trend"].append({
                "epoch": self.circuit_history['epochs'][i],
                "count": len(self.circuit_history['active_heads'][i])
            })

            summary["active_circuits_trend"].append({
                "epoch": self.circuit_history['epochs'][i],
                "count": len(self.circuit_history['active_circuits'][i])
            })

        # Find most stable heads (if we have stability data)
        if self.circuit_history['head_stability']:
            head_avg_stability = {}
            for head, stability_values in self.circuit_history['head_stability'].items():
                if stability_values:
                    head_avg_stability[head] = sum(stability_values) / len(stability_values)

            # Sort by stability and get top 5
            sorted_heads = sorted(head_avg_stability.items(), key=lambda x: x[1], reverse=True)
            summary["most_stable_heads"] = sorted_heads[:5]

        # Find most stable circuits
        if self.circuit_history['circuit_stability']:
            circuit_avg_stability = {}
            for circuit, stability_values in self.circuit_history['circuit_stability'].items():
                if stability_values:
                    circuit_avg_stability[circuit] = sum(stability_values) / len(stability_values)

            # Sort by stability and get top 5
            sorted_circuits = sorted(circuit_avg_stability.items(), key=lambda x: x[1], reverse=True)
            summary["most_stable_circuits"] = sorted_circuits[:5]

        # Find recently emerged circuits
        if len(self.circuit_history['emerging_circuits']) > 0:
            # Get most recent newly emerged circuits
            recent_emerged = self.circuit_history['emerging_circuits'][-1]

            # Check if we have corresponding circuit strengths for the latest epoch
            if len(self.circuit_history['circuit_strengths']) > 0:
                latest_strengths = self.circuit_history['circuit_strengths'][-1]

            # Add details about these circuits
            for circuit in recent_emerged:
                if circuit in latest_strengths:
                    summary["recently_emerged_circuits"].append({
                        "circuit": circuit,
                        "strength": latest_strengths[circuit],
                        "epoch": self.circuit_history['epochs'][-1]
                    })
                else:
                    # If we don't have strength data for this circuit, still include it
                    summary["recently_emerged_circuits"].append({
                        "circuit": circuit,
                        "strength": None,
                        "epoch": self.circuit_history['epochs'][-1]
                    })
            else:
                # If no strength data available, just include the circuit names
                for circuit in recent_emerged:
                    summary["recently_emerged_circuits"].append({
                        "circuit": circuit,
                        "strength": None,
                        "epoch": self.circuit_history['epochs'][-1]
                    })

        # Summarize behavioral impact
        if self.behavioral_impact['circuit_to_performance']:
            for circuit, impacts in self.behavioral_impact['circuit_to_performance'].items():
                if impacts:
                    avg_impact = sum(impacts) / len(impacts)
                    summary["behavioral_impact"][circuit] = {
                        "average_impact": avg_impact,
                        "latest_impact": impacts[-1],
                        "trend": "increasing" if len(impacts) > 1 and impacts[-1] > impacts[-2] else
                        "decreasing" if len(impacts) > 1 and impacts[-1] < impacts[-2] else
                        "stable"
                    }

        # Add eigenvalue analysis of connection matrix if available
        if self.connection_matrices:
            latest_matrix = self.connection_matrices[-1]
            try:
                # Compute eigenvalues of the connection matrix
                eigenvalues, _ = np.linalg.eig(latest_matrix)

                # Sort by magnitude
                sorted_eigenvalues = sorted([(i, abs(ev)) for i, ev in enumerate(eigenvalues)],
                                            key=lambda x: x[1], reverse=True)

                # Extract top eigenvalues
                top_eigenvalues = [(i, eigenvalues[i]) for i, _ in sorted_eigenvalues[:3]]

                summary["spectral_analysis"] = {
                    "top_eigenvalues": top_eigenvalues,
                    "spectral_gap": sorted_eigenvalues[0][1] - sorted_eigenvalues[1][1] if len(
                        sorted_eigenvalues) > 1 else 0,
                    "connectivity_change": self.connectivity_evolution[-1] if self.connectivity_evolution else 0
                }
            except:
                # Skip eigenvalue analysis if it fails
                pass

        return summary

    def detect_phase_transitions(self, window_size=3):
        """
        Detect potential phase transitions in circuit formation and structure.

        Args:
            window_size: Size of the sliding window for change detection

        Returns:
            list: Detected phase transitions with epochs and characteristics
        """
        if len(self.circuit_history['epochs']) < window_size + 1:
            return []

        transitions = []

        # Create sliding windows over circuit history
        for i in range(len(self.circuit_history['epochs']) - window_size):
            window_start = i
            window_end = i + window_size

            # Extract window data
            window_epochs = self.circuit_history['epochs'][window_start:window_end + 1]
            window_head_counts = [len(h) for h in self.circuit_history['active_heads'][window_start:window_end + 1]]
            window_circuit_counts = [len(c) for c in
                                     self.circuit_history['active_circuits'][window_start:window_end + 1]]

            # Look for significant changes in the number of active heads or circuits
            head_change_ratio = max(window_head_counts) / (min(window_head_counts) + 1e-6)
            circuit_change_ratio = max(window_circuit_counts) / (min(window_circuit_counts) + 1e-6)

            # Get connectivity changes if available
            connectivity_change = None
            if len(self.connectivity_evolution) > i + window_size:
                connectivity_change = self.connectivity_evolution[window_start:window_end + 1]

            # Detect significant changes
            significant_head_change = head_change_ratio > 1.5  # 50% change
            significant_circuit_change = circuit_change_ratio > 1.5  # 50% change

            # Check for connectivity spikes
            connectivity_spike = False
            if connectivity_change:
                mean_connectivity = sum(connectivity_change) / len(connectivity_change)
                max_connectivity = max(connectivity_change)
                connectivity_spike = max_connectivity > 2 * mean_connectivity

            # If significant changes detected, add to transitions
            if significant_head_change or significant_circuit_change or connectivity_spike:
                # Find the epoch with the maximum change
                max_change_idx = 0  # Default to first point

                if connectivity_change and connectivity_spike:
                    # Find where connectivity changes most
                    max_change_idx = connectivity_change.index(max(connectivity_change))
                elif significant_circuit_change and len(window_circuit_counts) > 1:
                    # Find where circuit count changes most
                    circuit_diffs = []
                    for j in range(len(window_circuit_counts) - 1):
                        circuit_diffs.append(abs(window_circuit_counts[j + 1] - window_circuit_counts[j]))
                    if circuit_diffs:  # Check for non-empty list
                        max_change_idx = circuit_diffs.index(max(circuit_diffs))
                elif significant_head_change and len(window_head_counts) > 1:
                    # Find where head count changes most
                    head_diffs = []
                    for j in range(len(window_head_counts) - 1):
                        head_diffs.append(abs(window_head_counts[j + 1] - window_head_counts[j]))
                    if head_diffs:  # Check for non-empty list
                        max_change_idx = head_diffs.index(max(head_diffs))

                # Determine the transition epoch - carefully handle max_change_idx
                transition_epoch = window_epochs[min(max_change_idx, len(window_epochs) - 1)]

                # Characterize the transition - ensure we don't go out of bounds
                transition_type = []
                if significant_head_change and max_change_idx < len(window_head_counts) - 1:
                    if window_head_counts[max_change_idx + 1] > window_head_counts[max_change_idx]:
                        transition_type.append("head_emergence")
                    else:
                        transition_type.append("head_pruning")

                if significant_circuit_change and max_change_idx < len(window_circuit_counts) - 1:
                    if window_circuit_counts[max_change_idx + 1] > window_circuit_counts[max_change_idx]:
                        transition_type.append("circuit_formation")
                    else:
                        transition_type.append("circuit_dissolution")

                if connectivity_spike:
                    transition_type.append("connectivity_reorganization")

                # If we couldn't determine transition types, use defaults based on ratio changes
                if not transition_type:
                    if significant_circuit_change:
                        # Just check the overall trend
                        if window_circuit_counts[-1] > window_circuit_counts[0]:
                            transition_type.append("circuit_formation")
                        else:
                            transition_type.append("circuit_dissolution")

                    if significant_head_change:
                        # Just check the overall trend
                        if window_head_counts[-1] > window_head_counts[0]:
                            transition_type.append("head_emergence")
                        else:
                            transition_type.append("head_pruning")

                # Add to detected transitions
                transitions.append({
                    "epoch": transition_epoch,
                    "window": (window_epochs[0], window_epochs[-1]),
                    "transition_types": transition_type,
                    "head_change_ratio": head_change_ratio,
                    "circuit_change_ratio": circuit_change_ratio,
                    "connectivity_spike": connectivity_spike if connectivity_change else None
                })

        # Remove duplicates and sort by epoch
        unique_transitions = []
        transition_epochs = set()

        for t in transitions:
            if t["epoch"] not in transition_epochs:
                transition_epochs.add(t["epoch"])
                unique_transitions.append(t)

        unique_transitions.sort(key=lambda x: x["epoch"])
        return unique_transitions

    def analyze_head_interactions(self):
        """
        Analyze how head interactions evolve over time.

        Returns:
            dict: Analysis of head interaction patterns
        """
        if len(self.circuit_history['epochs']) < 2:
            return {}

        results = {
            "epochs": self.circuit_history['epochs'],
            "interaction_patterns": [],
            "head_roles": {},
            "head_centrality": {}
        }

        # Analyze interaction patterns
        for i, epoch in enumerate(self.circuit_history['epochs']):
            if not self.connection_matrices or i >= len(self.connection_matrices):
                continue

            conn_matrix = self.connection_matrices[i]

            # Calculate network metrics if we have networkx
            try:
                import networkx as nx

                # Convert connection matrix to a weighted graph
                G = nx.from_numpy_array(conn_matrix)

                # Map node indices to head names
                active_heads = self.circuit_history['active_heads'][i]
                mapping = {j: head for j, head in enumerate(active_heads)}
                G = nx.relabel_nodes(G, mapping)

                # Calculate centrality metrics
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
                eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')

                # Store centrality for this epoch
                epoch_centrality = {
                    "epoch": epoch,
                    "degree": degree_centrality,
                    "betweenness": betweenness_centrality,
                    "eigenvector": eigenvector_centrality
                }

                # Update running centrality metrics for each head
                for head in active_heads:
                    if head not in results["head_centrality"]:
                        results["head_centrality"][head] = []

                    results["head_centrality"][head].append({
                        "epoch": epoch,
                        "degree": degree_centrality.get(head, 0),
                        "betweenness": betweenness_centrality.get(head, 0),
                        "eigenvector": eigenvector_centrality.get(head, 0)
                    })

                # Detect communities using Louvain method
                try:
                    communities = nx.community.louvain_communities(G)

                    # Map communities to head roles
                    community_map = {}
                    for comm_idx, community in enumerate(communities):
                        for head in community:
                            community_map[head] = comm_idx

                    # Store community information
                    results["interaction_patterns"].append({
                        "epoch": epoch,
                        "communities": [list(c) for c in communities],
                        "community_map": community_map,
                        "modularity": nx.community.modularity(G, communities)
                    })

                    # Update head roles
                    for head, comm_idx in community_map.items():
                        if head not in results["head_roles"]:
                            results["head_roles"][head] = []

                        results["head_roles"][head].append({
                            "epoch": epoch,
                            "community": comm_idx
                        })
                except:
                    # Skip community detection if it fails
                    pass
            except:
                # Skip network analysis if networkx is not available
                pass

        # Analyze role stability for each head
        head_stability = {}
        for head, roles in results["head_roles"].items():
            if len(roles) < 2:
                continue

            # Count how often the head changes communities
            community_changes = sum(1 for i in range(1, len(roles))
                                    if roles[i]["community"] != roles[i - 1]["community"])

            # Calculate stability (0 = always changing, 1 = never changing)
            stability = 1.0 - (community_changes / (len(roles) - 1))

            head_stability[head] = stability

        results["head_role_stability"] = head_stability

        # Identify heads with consistent high centrality
        consistent_central_heads = {}
        for head, centrality_history in results["head_centrality"].items():
            if len(centrality_history) < 2:
                continue

            # Calculate average centrality metrics
            avg_eigenvector = sum(c["eigenvector"] for c in centrality_history) / len(centrality_history)
            avg_betweenness = sum(c["betweenness"] for c in centrality_history) / len(centrality_history)

            # Score is a combination of centrality and stability
            stability = head_stability.get(head, 0.5)  # Default to 0.5 if not available
            centrality_score = (avg_eigenvector + avg_betweenness) * stability

            consistent_central_heads[head] = centrality_score

        # Get top central heads
        top_central_heads = sorted(consistent_central_heads.items(), key=lambda x: x[1], reverse=True)
        results["top_central_heads"] = top_central_heads[:5]  # Top 5 most central heads

        return results

    def predict_emerging_circuits(self):
        """
        Predict potential circuit formation based on current trends.

        Returns:
            list: Predicted emerging circuits with confidence scores
        """
        if len(self.circuit_history['epochs']) < 3:
            return []

        # Get head attributions and active circuits from the most recent epoch
        latest_attribution = self.circuit_history['head_attributions'][-1]
        active_circuits = set(self.circuit_history['active_circuits'][-1])

        # Identify heads with increasing attribution but not yet in circuits
        increasing_heads = {}
        for head, curr_attr in latest_attribution.items():
            # Skip if attribution is too low
            if curr_attr < self.min_attribution:
                continue

            # Check if this head is already in any active circuit
            in_active_circuit = False
            for circuit in active_circuits:
                if head in circuit:
                    in_active_circuit = True
                    break

            # Skip heads already in active circuits
            if in_active_circuit:
                continue

            # Check attribution trend
            found_in_history = False
            for i in range(len(self.circuit_history['head_attributions']) - 2, -1, -1):
                prev_attributions = self.circuit_history['head_attributions'][i]
                if head in prev_attributions:
                    found_in_history = True
                    # Check if attribution is increasing
                    if curr_attr > prev_attributions[head]:
                        # Calculate growth rate
                        growth_rate = (curr_attr - prev_attributions[head]) / (prev_attributions[head] + 1e-6)
                        increasing_heads[head] = growth_rate
                    break

            # If head was not in history, it's newly significant
            if not found_in_history and curr_attr >= self.min_attribution:
                increasing_heads[head] = 1.0  # Maximum growth rate for new heads

        # If we have a connection matrix, use it to predict potential connections
        potential_circuits = []

        if self.connection_matrices and increasing_heads:
            latest_matrix = self.connection_matrices[-1]
            active_heads = self.circuit_history['active_heads'][-1]

            # Create mapping from head name to index
            head_to_idx = {head: i for i, head in enumerate(active_heads)}

            # Analyze potential connections for each increasing head
            for head, growth_rate in increasing_heads.items():
                if head not in head_to_idx:
                    continue

                head_idx = head_to_idx[head]

                # Look for strongest connection potentials
                for i, other_head in enumerate(active_heads):
                    if other_head == head or i == head_idx:
                        continue

                    # Skip if this pair is already an active circuit
                    pair_key = f"{head}+{other_head}"
                    reverse_pair_key = f"{other_head}+{head}"
                    if pair_key in active_circuits or reverse_pair_key in active_circuits:
                        continue

                    # Check connection strength
                    connection_strength = latest_matrix[head_idx, i]

                    # Only consider positive connections
                    if connection_strength <= 0:
                        continue

                    # Calculate confidence based on connection strength and growth rate
                    confidence = min(1.0, connection_strength * growth_rate * 5.0)

                    if confidence > 0.2:  # Threshold for prediction
                        potential_circuits.append({
                            "heads": [head, other_head],
                            "confidence": confidence,
                            "connection_strength": connection_strength,
                            "growth_rate": growth_rate
                        })

        # Sort by confidence
        potential_circuits.sort(key=lambda x: x["confidence"], reverse=True)

        return potential_circuits

    def analyze_phase_structure(self):
        """
        Analyze the structure of different learning phases based on circuit dynamics.

        Returns:
            dict: Analysis of learning phases and their circuit characteristics
        """
        # Detect phase transitions first
        transitions = self.detect_phase_transitions()

        if not transitions or len(self.circuit_history['epochs']) < 3:
            return {"status": "Not enough data for phase analysis"}

        # Define phases based on detected transitions
        transitions_epochs = [t["epoch"] for t in transitions]
        phase_boundaries = [self.circuit_history['epochs'][0]] + transitions_epochs + [
            self.circuit_history['epochs'][-1]]

        phases = []
        for i in range(len(phase_boundaries) - 1):
            start_epoch = phase_boundaries[i]
            end_epoch = phase_boundaries[i + 1]

            # Define phase
            phases.append({
                "phase_id": i + 1,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "transition_in": transitions[i - 1] if i > 0 else None,
                "transition_out": transitions[i] if i < len(transitions) else None,
                "characteristics": {}
            })

        # Analyze characteristics of each phase
        for phase in phases:
            start_idx = self.circuit_history['epochs'].index(phase["start_epoch"]) if phase["start_epoch"] in \
                                                                                      self.circuit_history[
                                                                                          'epochs'] else 0
            end_idx = self.circuit_history['epochs'].index(phase["end_epoch"]) if phase["end_epoch"] in \
                                                                                  self.circuit_history[
                                                                                      'epochs'] else len(
                self.circuit_history['epochs']) - 1

            # Skip phases that don't map well to our data
            if end_idx <= start_idx:
                continue

            # Extract phase-specific data
            phase_epochs = self.circuit_history['epochs'][start_idx:end_idx + 1]
            phase_head_counts = [len(h) for h in self.circuit_history['active_heads'][start_idx:end_idx + 1]]
            phase_circuit_counts = [len(c) for c in self.circuit_history['active_circuits'][start_idx:end_idx + 1]]

            def safe_mean(values, default=0.0):
                """Calculate mean safely for empty arrays"""
                if not values or len(values) == 0:
                    return default
                return np.mean(values)

            # Calculate phase metrics
            phase["characteristics"] = {
                "duration": len(phase_epochs),
                "avg_active_heads": sum(phase_head_counts) / len(phase_head_counts),
                "avg_active_circuits": sum(phase_circuit_counts) / len(phase_circuit_counts),
                "head_trend": "increasing" if phase_head_counts[-1] > phase_head_counts[0] else
                "decreasing" if phase_head_counts[-1] < phase_head_counts[0] else
                "stable",
                "circuit_trend": "increasing" if phase_circuit_counts[-1] > phase_circuit_counts[0] else
                "decreasing" if phase_circuit_counts[-1] < phase_circuit_counts[0] else
                "stable",
                # "variability": np.std(phase_circuit_counts) / (np.mean(phase_circuit_counts) + 1e-6),
                "variability": np.std(phase_circuit_counts) / (safe_mean(phase_circuit_counts, 1.) + 1e-6),
            }

            # Identify key circuits in this phase
            phase_circuits = set()
            for circuits in self.circuit_history['active_circuits'][start_idx:end_idx + 1]:
                phase_circuits.update(circuits)

            # Calculate circuit consistency (how frequently they appear in the phase)
            circuit_consistency = {}
            for circuit in phase_circuits:
                appearances = sum(1 for circuits in self.circuit_history['active_circuits'][start_idx:end_idx + 1]
                                  if circuit in circuits)
                consistency = appearances / len(phase_epochs)
                circuit_consistency[circuit] = consistency

            # Get most consistent circuits
            top_circuits = sorted(circuit_consistency.items(), key=lambda x: x[1], reverse=True)
            phase["characteristics"]["core_circuits"] = [c for c, _ in top_circuits[:min(5, len(top_circuits))]]

            # Calculate average circuit strength in this phase
            circuit_strengths = []
            for i, epoch_idx in enumerate(range(start_idx, end_idx + 1)):
                if epoch_idx < len(self.circuit_history['circuit_strengths']):
                    strengths = self.circuit_history['circuit_strengths'][epoch_idx]
                    if strengths:
                        circuit_strengths.extend(list(strengths.values()))

            if circuit_strengths:
                phase["characteristics"]["avg_circuit_strength"] = sum(circuit_strengths) / len(circuit_strengths)

            # Analyze connectivity changes if available
            if self.connectivity_evolution and start_idx < len(self.connectivity_evolution) and end_idx < len(
                    self.connectivity_evolution):
                phase_connectivity = self.connectivity_evolution[start_idx:end_idx + 1]
                phase["characteristics"]["connectivity_change_rate"] = sum(phase_connectivity) / len(phase_connectivity)

            # Classify phase by its characteristics
            # This is a simplified classification and could be made more sophisticated
            if phase["characteristics"]["circuit_trend"] == "increasing" and phase["characteristics"][
                "head_trend"] == "increasing":
                phase["classification"] = "exploration"
            elif phase["characteristics"]["circuit_trend"] == "decreasing" and phase["characteristics"][
                "variability"] < 0.2:
                phase["classification"] = "consolidation"
            elif phase["characteristics"]["circuit_trend"] == "stable" and phase["characteristics"][
                "head_trend"] == "stable":
                phase["classification"] = "stability"
            elif phase["characteristics"]["circuit_trend"] == "decreasing" and phase["characteristics"][
                "head_trend"] == "decreasing":
                phase["classification"] = "pruning"
            else:
                phase["classification"] = "transition"

        return {
            "phases": phases,
            "transitions": transitions
        }

    def get_circuit_summary_for_epoch(self, epoch):
        """
        Get a summary of circuit state at a specific epoch.

        Args:
            epoch: The epoch to summarize

        Returns:
            dict: Summary of circuit state at the specified epoch
        """
        # Ensure history consistency before accessing
        self._ensure_history_consistency()

        if not self.circuit_history['epochs'] or epoch not in self.circuit_history['epochs']:
            return {"status": f"No data available for epoch {epoch}"}

        # Find the index for this epoch
        epoch_idx = self.circuit_history['epochs'].index(epoch)

        # Safely access history elements
        active_heads = (self.circuit_history['active_heads'][epoch_idx]
                        if epoch_idx < len(self.circuit_history['active_heads']) else [])

        active_circuits = (self.circuit_history['active_circuits'][epoch_idx]
                           if epoch_idx < len(self.circuit_history['active_circuits']) else [])

        emerging_circuits = (self.circuit_history['emerging_circuits'][epoch_idx]
                             if epoch_idx < len(self.circuit_history['emerging_circuits']) else [])

        declining_circuits = (self.circuit_history['declining_circuits'][epoch_idx]
                              if epoch_idx < len(self.circuit_history['declining_circuits']) else [])

        circuit_strengths = (self.circuit_history['circuit_strengths'][epoch_idx]
                             if epoch_idx < len(self.circuit_history['circuit_strengths']) else {})

        summary = {
            "epoch": epoch,
            "active_heads": active_heads,
            "active_circuits": active_circuits,
            "emerging_circuits": emerging_circuits,
            "declining_circuits": declining_circuits,
            "circuit_strengths": circuit_strengths
        }

        # Add top circuits by strength
        if summary["circuit_strengths"]:
            top_circuits = sorted(summary["circuit_strengths"].items(), key=lambda x: x[1], reverse=True)
            summary["top_circuits"] = top_circuits[:min(5, len(top_circuits))]

        # Add connectivity change if available
        if self.connectivity_evolution and epoch_idx < len(self.connectivity_evolution):
            summary["connectivity_change"] = self.connectivity_evolution[epoch_idx]

        return summary

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
