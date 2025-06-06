# integrated_token_discovery.py
import random

import torch
import numpy as np
# from pathlib import Path
# from typing import Dict, List, Optional, Any, Union, Tuple

from analysis.core.circuit_schema import Circuit, CircuitType
from analysis.core.circuit_registry import EnhancedCircuitRegistry as CircuitRegistry
from analysis.analyzers.token_circuit_discovery import TokenCircuitDiscovery
from analysis.analyzers.circuit_evolution_tracker import CircuitEvolutionTracker
# Fix the import path to match your file structure
# from analysis.analyzers.attention_pattern_analyzer import AttentionAnalyzer
from analysis.utils.utils import get_current_callable_info


class IntegratedTokenCircuitDiscovery(TokenCircuitDiscovery):
    """Token circuit discovery that integrates with existing analysis tools"""

    def __init__(self, model, save_dir=None, logger=None, circuit_registry=None,
                 attention_analyzer=None, circuit_tracker=None, weight_tracker=None):
        """
        Initialize the integrated token circuit discovery

        Args:
            model: The transformer model
            save_dir: Directory to save analysis results
            circuit_registry: Optional circuit registry
            attention_analyzer: Optional attention analyzer
            circuit_tracker: Optional circuit tracker
            weight_tracker: Optional weight tracker
        """
        super().__init__(model=model, save_dir=save_dir, circuit_registry=circuit_registry)

        # Store references to existing analyzers
        self.attention_analyzer = attention_analyzer
        self.circuit_tracker = circuit_tracker
        self.weight_tracker = weight_tracker

        # Initialize evolution tracker
        self.evolution_tracker = CircuitEvolutionTracker(
            registry=self.registry,
            save_dir=save_dir / "evolution" if save_dir else None,
            logger=logger,
        )

        # Storage for integrated analysis results
        self.integrated_results = {}


    # Add to integrated_token_discovery.py
    def analyze_epoch(self, epoch, eval_loader, baseline_acc=None):
        """
        Integrated analysis without needing a representative batch
        """

        # Token analysis with multiple random batches (as above)
        total_batches = len(eval_loader)
        num_batches_to_analyze = min(3, total_batches)
        selected_batch_indices = random.sample(range(total_batches), num_batches_to_analyze)
        selected_batch_indices.sort()

        all_token_results = []

        max_examples = 5
        # print(f"\t{get_current_callable_info()} @ {epoch}:\tprocess [analyze token relationships({len(selected_batch_indices)} batches x {max_examples} examples)]")

        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            if batch_idx in selected_batch_indices:
                batch_token_results = self.analyze_token_relationships(
                    inputs=inputs,
                    targets=targets,
                    epoch=epoch,
                    analyze_multiple_examples=True,
                    max_examples=max_examples,
                    random_sampling=True,
                    random_seed=epoch + batch_idx,
                )
                all_token_results.append(batch_token_results)
                selected_batch_indices.remove(batch_idx)
                if not selected_batch_indices:
                    break

        # Combine token results
        combined_token_results = self.combine_batch_results(all_token_results, epoch)
        # print(f"\t\t {len(combined_token_results['copy_mechanisms'])} copy mechanisms,\t{len(combined_token_results['induction_patterns'])} induction patterns")

        # Run other analyses independently (they can sample their own data)
        component_results = None
        if self.circuit_tracker:
            component_results = self.circuit_tracker.sample_circuits(
                epoch=epoch,
                eval_loader=eval_loader,  # 👈 Let circuit_tracker sample its own data
                baseline_acc=baseline_acc
            )

        attention_results = None
        if self.attention_analyzer:
            attention_results = self.attention_analyzer.analyze(
                eval_loader=eval_loader  # 👈 Let attention_analyzer sample its own data
            )

        # Update evolution tracking
        evolution_results = self.evolution_tracker.update_circuit_evolution(
            epoch=epoch,
            circuits=combined_token_results["circuits"],
            token_attribution=combined_token_results["token_attribution"]
        )

        return {
            "token_results": combined_token_results,
            "component_results": component_results,
            "attention_results": attention_results,
            "evolution_results": evolution_results
        }


    def combine_batch_results(self, all_token_results, epoch):
        """Combine token circuit results from multiple batches"""
        # Combine all circuits
        all_circuits = []
        all_copy_mechanisms = []
        all_induction_patterns = []
        all_attributions = []

        for batch_results in all_token_results:
            all_circuits.extend(batch_results.get("circuits", []))
            all_copy_mechanisms.extend(batch_results.get("copy_mechanisms", []))
            all_induction_patterns.extend(batch_results.get("induction_patterns", []))

            # Collect attributions
            if "token_attribution" in batch_results:
                all_attributions.append(batch_results["token_attribution"])

        # Average attribution across batches
        if all_attributions:
            avg_attribution = np.mean(all_attributions, axis=0)
        else:
            seq_len = 4  # Your sequence length
            avg_attribution = np.eye(seq_len)

        # Find circuits that appear across multiple batches (more robust)
        robust_circuits = self._find_robust_cross_batch_circuits(
            all_circuits, min_batch_appearances=2)

        return {
            "circuits": robust_circuits,
            "copy_mechanisms": all_copy_mechanisms,
            "induction_patterns": all_induction_patterns,
            "token_attribution": avg_attribution,
            "batches_analyzed": len(all_token_results),
            "total_examples": sum(r.get("num_examples_analyzed", 0) for r in all_token_results)
        }


    def _find_robust_cross_batch_circuits(self, all_circuits, min_batch_appearances=2):
        """Find circuits that appear consistently across multiple batches"""
        circuit_patterns = {}

        # Group circuits by their pattern/structure
        for circuit in all_circuits:
            # Create a pattern key based on circuit structure
            pattern_key = self._get_circuit_pattern_key(circuit)

            if pattern_key not in circuit_patterns:
                circuit_patterns[pattern_key] = []
            circuit_patterns[pattern_key].append(circuit)

        # Keep only circuits that appear in multiple batches
        robust_circuits = []
        for pattern_key, circuits in circuit_patterns.items():
            if len(circuits) >= min_batch_appearances:
                # Use the circuit with highest attribution as representative
                best_circuit = max(circuits, key=lambda c: c.attribution)
                best_circuit.metadata['cross_batch_count'] = len(circuits)
                robust_circuits.append(best_circuit)

        return robust_circuits


    def _get_circuit_pattern_key(self, circuit):
        """Create a pattern key for circuit similarity comparison"""
        # Use operation type and head information to create pattern
        op_type = circuit.metadata.get('operation_type', 'unknown')
        head = circuit.metadata.get('head', 'unknown_head')

        # For copy circuits, include relative offset
        if op_type == 'copy':
            offset = circuit.metadata.get('relative_offset', 0)
            return f"{op_type}_{head}_offset_{offset}"
        else:
            return f"{op_type}_{head}"

    def analyze_epoch_single_repeating_batch(self, epoch, eval_loader, baseline_acc=None):
        """
        Integrated analysis for a specific epoch

        Args:
            epoch: Current training epoch
            eval_loader: Evaluation data loader
            baseline_acc: Optional baseline accuracy

        Returns:
            Dict with analysis results
        """
        # print(f"\t{get_current_callable_info()} @ {epoch}: \t")

        # Run component-level circuit tracking if available
        component_results = None
        if self.circuit_tracker:
            component_results = self.circuit_tracker.sample_circuits(
                epoch=epoch,
                eval_loader=eval_loader,
                baseline_acc=baseline_acc
            )

        # Run attention analysis if available
        attention_results = None
        if self.attention_analyzer:
            attention_results = self.attention_analyzer.analyze(
                eval_loader=eval_loader
            )

        # Run token-level analysis on a sample batch
        batch = next(iter(eval_loader))
        inputs, targets = batch

        token_results = self.analyze_token_relationships(
            inputs=inputs,
            targets=targets,
            epoch=epoch,
            analyze_multiple_examples=True
        )

        # Update evolution tracking
        evolution_results = self.evolution_tracker.update_circuit_evolution(
            epoch=epoch,
            circuits=token_results["circuits"],
            token_attribution=token_results["token_attribution"]
        )

        # Cross-reference with component circuits
        if component_results and hasattr(component_results, 'active_circuits'):
            component_circuit_ids = component_results.get('active_circuits', [])
            token_circuit_ids = [c.id for c in token_results["circuits"]]

            # Register potential relationships between token and component circuits
            for token_id in token_circuit_ids:
                token_circuit = self.registry.get_circuit(token_id)
                if token_circuit:
                    for comp_id in component_circuit_ids:
                        comp_circuit = self.registry.get_circuit(comp_id)
                        if comp_circuit:
                            # Check for potential relationship based on elements
                            if self._check_circuit_overlap(token_circuit, comp_circuit):
                                # Register relationship
                                self.registry.register_relation(token_id, comp_id)

        # Store integrated results
        self.integrated_results[epoch] = {
            "token_results": token_results,
            "component_results": component_results,
            "attention_results": attention_results,
            "evolution_results": evolution_results
        }

        # Return combined results
        return {
            "token_results": token_results,
            "component_results": component_results,
            "attention_results": attention_results,
            "evolution_results": evolution_results
        }

    def _check_circuit_overlap(self, circuit1, circuit2):
        """
        Check if two circuits have overlapping elements

        Args:
            circuit1: First circuit
            circuit2: Second circuit

        Returns:
            bool: True if circuits overlap
        """
        # Extract element IDs from both circuits
        elements1 = set(e.id for e in circuit1.elements)
        elements2 = set(e.id for e in circuit2.elements)

        # Check for overlap
        return len(elements1.intersection(elements2)) > 0

    def analyze_jumps_and_circuits(self, jump_results, epoch, eval_loader):
        """
        Analyze the relationship between weight space jumps and circuit formation

        Args:
            jump_results: Results from jump analysis
            epoch: Current epoch
            eval_loader: Evaluation data loader

        Returns:
            Dict with jump-circuit analysis
        """
        results = {}

        for jump_result in jump_results:
            jump_epoch = jump_result['jump_epoch']

            # Find circuits that emerged near this jump
            nearby_circuits = []
            # print(f"\t\t{get_current_callable_info()} @ {epoch}: \t")
            for circuit_id, emergence_epoch in self.evolution_tracker.emergence_epochs.items():
                if abs(emergence_epoch - jump_epoch) <= 10:  # Within 10 epochs
                    circuit = self.registry.get_circuit(circuit_id)
                    if circuit:
                        nearby_circuits.append({
                            'circuit_id': circuit_id,
                            'circuit_type': circuit.type.value,
                            'emergence_epoch': emergence_epoch,
                            'relation': 'before_jump' if emergence_epoch < jump_epoch else 'after_jump',
                            'distance': abs(emergence_epoch - jump_epoch)
                        })

            # Analyze circuit examples around jump
            sample_batch = next(iter(eval_loader))
            inputs, targets = sample_batch

            # Get pre-jump and post-jump states
            pre_state = jump_result.get('pre_jump_snapshot', {}).get('state_dict')
            post_state = jump_result.get('post_jump_snapshot', {}).get('state_dict')

            # Analyze circuit behavior change if we have pre and post states
            behavior_change = {}
            if pre_state and post_state:
                behavior_change = self._analyze_circuit_behavior_change(
                    circuit_ids=[c['circuit_id'] for c in nearby_circuits],
                    pre_state=pre_state,
                    post_state=post_state,
                    inputs=inputs,
                    targets=targets
                )

            # Store results for this jump
            results[jump_epoch] = {
                'nearby_circuits': nearby_circuits,
                'circuit_count_before': len([c for c in nearby_circuits if c['relation'] == 'before_jump']),
                'circuit_count_after': len([c for c in nearby_circuits if c['relation'] == 'after_jump']),
                'behavior_change': behavior_change
            }

            # Log insights
            # print(f"\t{get_current_callable_info()} @ {epoch}: \tjump at Epoch {jump_epoch} and Circuit Formation:")
            # print(f"\t\tircuits emerging before jump: {results[jump_epoch]['circuit_count_before']}")
            # print(f"\t\tcircuits emerging after jump: {results[jump_epoch]['circuit_count_after']}")

            if behavior_change:
                print(f"\t{get_current_callable_info()} @ {epoch}: \tcircuit behavior changes:")
                for circuit_id, change in behavior_change.items():
                    print(f"\t{get_current_callable_info()} @ {epoch}: \t{circuit_id}: attribution {change['attribution_change']:.3f}")

        return results

    def _analyze_circuit_behavior_change(self, circuit_ids, pre_state, post_state, inputs, targets):
        """
        Analyze how circuits behave differently before and after a jump

        Args:
            circuit_ids: List of circuit IDs to analyze
            pre_state: Model state before jump
            post_state: Model state after jump
            inputs: Input tensor for analysis
            targets: Target tensor for analysis

        Returns:
            Dict with behavior change analysis
        """
        # Store original state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        # print(f"\t{get_current_callable_info()}: \t")
        try:
            # Analyze behavior with pre-jump state
            self.model.load_state_dict(pre_state)
            pre_behavior = self._analyze_circuit_behavior(circuit_ids, inputs, targets)

            # Analyze behavior with post-jump state
            self.model.load_state_dict(post_state)
            post_behavior = self._analyze_circuit_behavior(circuit_ids, inputs, targets)

            # Calculate changes
            behavior_change = {}

            for circuit_id in circuit_ids:
                if circuit_id in pre_behavior and circuit_id in post_behavior:
                    pre = pre_behavior[circuit_id]
                    post = post_behavior[circuit_id]

                    # Calculate changes in key metrics
                    attribution_change = post['attribution'] - pre['attribution']

                    behavior_change[circuit_id] = {
                        'attribution_change': attribution_change,
                        'pre_attribution': pre['attribution'],
                        'post_attribution': post['attribution']
                    }

            return behavior_change

        finally:
            # Restore original state
            self.model.load_state_dict(original_state)

    def _analyze_circuit_behavior(self, circuit_ids, inputs, targets):
        """
        Analyze circuit behavior on specific inputs

        Args:
            circuit_ids: List of circuit IDs to analyze
            inputs: Input tensor for analysis
            targets: Target tensor for analysis

        Returns:
            Dict with behavior analysis
        """
        # Run forward pass to get attention patterns
        outputs = self.model(inputs, store_attention=True)

        # Get attention patterns
        attention_patterns = {}
        if hasattr(self.model, 'get_attention_patterns'):
            attention_patterns = self.model.get_attention_patterns()

        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == targets).float().mean().item()

        # Analyze each circuit
        behavior = {}

        for circuit_id in circuit_ids:
            circuit = self.registry.get_circuit(circuit_id)
            if not circuit:
                continue

            # For token circuits, compute attribution based on attention patterns
            if circuit.type == CircuitType.TOKEN:
                attribution = self._compute_token_circuit_attribution(circuit, attention_patterns)

                behavior[circuit_id] = {
                    'attribution': attribution,
                    'accuracy': accuracy
                }
        # print(f"\t{get_current_callable_info()}: \t{behavior}")
        return behavior

    def analyze_circuit_cooperation(self, epoch, eval_loader):
        """
        Analyze how different circuits cooperate

        Args:
            epoch: Current epoch
            eval_loader: Evaluation data loader

        Returns:
            Dict with cooperation analysis
        """
        # Get active circuits
        active_circuit_ids = self.evolution_tracker.epoch_to_circuits.get(epoch, set())

        # Get batch of data
        batch = next(iter(eval_loader))
        inputs, targets = batch

        # Run forward pass with attention tracking
        outputs = self.model(inputs, store_attention=True)

        # Get attention patterns
        attention_patterns = {}
        if hasattr(self.model, 'get_attention_patterns'):
            attention_patterns = self.model.get_attention_patterns()

        # Analyze activation patterns for each circuit
        circuit_activations = {}

        for circuit_id in active_circuit_ids:
            circuit = self.registry.get_circuit(circuit_id)
            if not circuit:
                continue

            # Calculate activation for this circuit
            activation = self._calculate_circuit_activation(circuit, attention_patterns, inputs)
            circuit_activations[circuit_id] = activation

        # Calculate correlation between circuit activations
        cooperation_matrix = np.zeros((len(circuit_activations), len(circuit_activations)))
        circuit_ids = list(circuit_activations.keys())

        for i, cid1 in enumerate(circuit_ids):
            for j, cid2 in enumerate(circuit_ids):
                if i == j:
                    cooperation_matrix[i, j] = 1.0  # Self-correlation
                else:
                    # Calculate activation correlation
                    act1 = circuit_activations[cid1]
                    act2 = circuit_activations[cid2]

                    if isinstance(act1, torch.Tensor):
                        act1 = act1.cpu().numpy()
                    if isinstance(act2, torch.Tensor):
                        act2 = act2.cpu().numpy()

                    # Flatten if needed
                    act1 = act1.flatten()
                    act2 = act2.flatten()

                    # Ensure same length by truncating or padding
                    min_len = min(len(act1), len(act2))
                    act1 = act1[:min_len]
                    act2 = act2[:min_len]

                    # Calculate correlation
                    if min_len > 1:
                        correlation = np.corrcoef(act1, act2)[0, 1]
                        cooperation_matrix[i, j] = correlation if not np.isnan(correlation) else 0
                    else:
                        cooperation_matrix[i, j] = 0

        # Identify cooperating circuit groups
        cooperating_groups = self._identify_cooperating_groups(cooperation_matrix, circuit_ids)

        # Identify sequential processing chains
        processing_chains = self._identify_processing_chains(circuit_activations, circuit_ids)

        return {
            'cooperation_matrix': cooperation_matrix,
            'circuit_ids': circuit_ids,
            'cooperating_groups': cooperating_groups,
            'processing_chains': processing_chains
        }

    def _calculate_circuit_activation(self, circuit, attention_patterns, inputs):
        """Calculate activation values for a circuit"""
        # Different calculation based on circuit type
        if circuit.type == CircuitType.TOKEN:
            return self._calculate_token_circuit_activation(circuit, attention_patterns)
        else:
            # For other circuit types, use a placeholder
            # This would need to be implemented based on your circuit definitions
            return np.random.random(10)  # Placeholder

    def _calculate_token_circuit_activation(self, circuit, attention_patterns):
        """Calculate activation for a token circuit"""
        # Extract head elements
        head_elements = [e for e in circuit.elements if e.type.name == 'HEAD']

        if not head_elements:
            return np.zeros(10)  # Placeholder for empty circuit

        # Calculate activation as attention scores
        activations = []

        for head_elem in head_elements:
            head_id = head_elem.id
            if head_id in attention_patterns:
                pattern = attention_patterns[head_id]
                if isinstance(pattern, torch.Tensor):
                    pattern = pattern.cpu().numpy()

                # Flatten pattern for activation signature
                activations.append(pattern.flatten())

        # Combine activations from all heads
        if activations:
            return np.mean(activations, axis=0)
        else:
            return np.zeros(10)  # Placeholder

    def _identify_cooperating_groups(self, cooperation_matrix, circuit_ids, threshold=0.6):
        """Identify groups of cooperating circuits using community detection"""
        import networkx as nx

        # Create graph from cooperation matrix
        G = nx.Graph()

        # Add nodes
        for i, cid in enumerate(circuit_ids):
            G.add_node(cid)

        # Add edges for cooperating circuits
        for i in range(len(circuit_ids)):
            for j in range(i + 1, len(circuit_ids)):
                if cooperation_matrix[i, j] >= threshold:
                    G.add_edge(circuit_ids[i], circuit_ids[j], weight=cooperation_matrix[i, j])

        # Find communities
        try:
            communities = nx.community.louvain_communities(G)

            # Format results
            results = []
            for i, community in enumerate(communities):
                results.append({
                    'group_id': i,
                    'circuit_ids': list(community),
                    'size': len(community)
                })

            return results
        except:
            # Fallback if community detection fails
            return []

    def _identify_processing_chains(self, circuit_activations, circuit_ids):
        """Identify sequential processing chains between circuits"""
        chains = []

        # Identify circuits with token elements
        token_circuits = []
        for cid in circuit_ids:
            circuit = self.registry.get_circuit(cid)
            if circuit and circuit.type == CircuitType.TOKEN:
                token_circuits.append(cid)

        # Look for potential chains based on token positions
        for cid1 in token_circuits:
            for cid2 in token_circuits:
                if cid1 != cid2:
                    circuit1 = self.registry.get_circuit(cid1)
                    circuit2 = self.registry.get_circuit(cid2)

                    # Get token positions
                    positions1 = [e.properties.get('position', -1)
                                  for e in circuit1.elements
                                  if e.type.name == 'TOKEN']

                    positions2 = [e.properties.get('position', -1)
                                  for e in circuit2.elements
                                  if e.type.name == 'TOKEN']

                    # Check for sequential relationship
                    if any(p1 in positions2 for p1 in positions1):
                        chains.append({
                            'from': cid1,
                            'to': cid2,
                            'type': 'token_sharing'
                        })

        return chains


    def _compute_token_circuit_attribution(self, circuit, attention_patterns):
        """
        Compute attribution score for a token circuit based on attention patterns

        Args:
            circuit: The token circuit
            attention_patterns: Attention patterns from the model

        Returns:
            float: Attribution score
        """
        # Extract head names from circuit
        head_elements = [e for e in circuit.elements if e.type.name == 'HEAD']
        head_names = [e.id for e in head_elements]

        # Calculate average attention score
        attribution = 0.0
        count = 0

        for head_name in head_names:
            if head_name in attention_patterns:
                pattern = attention_patterns[head_name]

                # Find source and target token positions
                source_tokens = [e for e in circuit.elements if e.type.name == 'TOKEN']
                token_positions = [e.properties.get('position', -1) for e in source_tokens]

                # Filter valid positions
                valid_positions = [pos for pos in token_positions if pos >= 0 and pos < pattern.shape[0]]

                if valid_positions:
                    # Calculate average attention at these positions
                    for pos in valid_positions:
                        # Attention from this position to others
                        pos_attention = pattern[pos].mean().item()
                        attribution += pos_attention
                        count += 1

        # Return average attribution
        return attribution / max(1, count)

    # Add to integrated_token_discovery.py
    def analyze_circuit_competition(self, epoch, eval_loader):
        """
        Analyze how circuits compete or interfere with each other

        Args:
            epoch: Current epoch
            eval_loader: Evaluation data loader

        Returns:
            Dict with competition analysis
        """
        # Get active circuits
        active_circuit_ids = self.evolution_tracker.epoch_to_circuits.get(epoch, set())

        # Get batch of data
        batch = next(iter(eval_loader))
        inputs, targets = batch

        # Store original model state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        try:
            # 1. Negative Correlation Analysis
            cooperation = self.analyze_circuit_cooperation(epoch, eval_loader)
            cooperation_matrix = cooperation.get('cooperation_matrix', [])
            circuit_ids = cooperation.get('circuit_ids', [])

            competing_pairs = []
            if len(cooperation_matrix) > 0:
                # Find strongly anti-correlated circuit pairs (negative correlation)
                for i in range(len(circuit_ids)):
                    for j in range(i + 1, len(circuit_ids)):
                        correlation = cooperation_matrix[i, j]

                        # Negative correlation suggests competition
                        if correlation < -0.4:
                            competing_pairs.append({
                                'circuit1': circuit_ids[i],
                                'circuit2': circuit_ids[j],
                                'negative_correlation': correlation
                            })

            # 2. Resource Contention Analysis
            resource_contention = []
            for cid1 in active_circuit_ids:
                for cid2 in active_circuit_ids:
                    if cid1 != cid2:
                        circuit1 = self.registry.get_circuit(cid1)
                        circuit2 = self.registry.get_circuit(cid2)

                        if circuit1 and circuit2:
                            # Check if circuits share components but have different functions
                            component_overlap = self.evolution_tracker._calculate_component_overlap(circuit1, circuit2)
                            functional_similarity = self.evolution_tracker._calculate_functional_similarity(circuit1,
                                                                                                            circuit2)

                            # High component overlap but low functional similarity suggests contention
                            if component_overlap > 0.6 and functional_similarity < 0.3:
                                resource_contention.append({
                                    'circuit1': cid1,
                                    'circuit2': cid2,
                                    'component_overlap': component_overlap,
                                    'functional_similarity': functional_similarity
                                })

            # 3. Interference Testing
            # Test if ablating one circuit improves another's performance
            interference_relationships = []

            # Select top circuits to test (to limit computational cost)
            test_circuits = list(active_circuit_ids)[:10]  # Limit to 10 circuits

            for cid1 in test_circuits:
                circuit1 = self.registry.get_circuit(cid1)
                if not circuit1:
                    continue

                # Get baseline performance for this circuit
                baseline_attr = self._compute_circuit_attribution(circuit1, inputs)

                for cid2 in test_circuits:
                    if cid1 == cid2:
                        continue

                    circuit2 = self.registry.get_circuit(cid2)
                    if not circuit2:
                        continue

                    # Ablate circuit2 to see effect on circuit1
                    self._ablate_circuit(circuit2)

                    # Measure circuit1's performance without circuit2
                    ablated_attr = self._compute_circuit_attribution(circuit1, inputs)

                    # Restore model state
                    self.model.load_state_dict(original_state)

                    # Calculate interference
                    # Positive difference means circuit2 was interfering with circuit1
                    interference = ablated_attr - baseline_attr

                    if abs(interference) > 0.1:  # Significant interference
                        interference_relationships.append({
                            'circuit1': cid1,
                            'circuit2': cid2,
                            'interference': interference
                        })

            # 4. Gradient Conflict Analysis
            # This is more complex and would require backprop through the model
            # Here we'll use a simplified approach based on weight updates
            gradient_conflicts = []

            if self.weight_tracker and len(self.weight_tracker.weight_snapshots) >= 2:
                # Get recent weight snapshots
                latest_idx = len(self.weight_tracker.weight_snapshots) - 1
                latest = self.weight_tracker.weight_snapshots[latest_idx]
                previous = self.weight_tracker.weight_snapshots[latest_idx - 1]

                # Calculate weight updates
                updates = {}
                for param_name in latest['state_dict']:
                    if param_name in previous['state_dict']:
                        latest_param = latest['state_dict'][param_name]
                        prev_param = previous['state_dict'][param_name]

                        if isinstance(latest_param, torch.Tensor) and isinstance(prev_param, torch.Tensor):
                            updates[param_name] = latest_param - prev_param

                # Check for gradient conflicts for each circuit pair
                for cid1 in test_circuits:
                    circuit1 = self.registry.get_circuit(cid1)
                    if not circuit1:
                        continue

                    for cid2 in test_circuits:
                        if cid1 == cid2:
                            continue

                        circuit2 = self.registry.get_circuit(cid2)
                        if not circuit2:
                            continue

                        # Get components from each circuit
                        components1 = self._get_circuit_components(circuit1)
                        components2 = self._get_circuit_components(circuit2)

                        # Calculate weight update correlation for shared components
                        shared_components = set(components1).intersection(set(components2))

                        if shared_components:
                            update_corrs = []

                            for comp in shared_components:
                                # Find weight parameters for this component
                                comp_params = [p for p in updates if comp in p]

                                for param in comp_params:
                                    # Flatten update
                                    flat_update = updates[param].flatten()

                                    # Calculate contribution to circuit1 vs circuit2
                                    # This is a simplification - in a real implementation
                                    # you'd need to calculate the gradients specifically for each circuit
                                    update_norm = torch.norm(flat_update).item()

                                    if update_norm > 0:
                                        # Add to correlation list - a placeholder value for this simplified example
                                        update_corrs.append(0.1)  # Placeholder

                            if update_corrs:
                                avg_corr = sum(update_corrs) / len(update_corrs)

                                # Negative correlation suggests competing gradient updates
                                if avg_corr < -0.3:
                                    gradient_conflicts.append({
                                        'circuit1': cid1,
                                        'circuit2': cid2,
                                        'gradient_correlation': avg_corr
                                    })

            return {
                'competing_pairs': competing_pairs,
                'resource_contention': resource_contention,
                'interference_relationships': interference_relationships,
                'gradient_conflicts': gradient_conflicts
            }

        finally:
            # Always restore original state
            self.model.load_state_dict(original_state)

    def _ablate_circuit(self, circuit):
        """
        Ablate a circuit by masking its components

        Args:
            circuit: The circuit to ablate
        """
        # Identify components to ablate
        head_elements = [e for e in circuit.elements if e.type.name == 'HEAD']

        # Ablate each head
        for head_elem in head_elements:
            head_id = head_elem.id

            # Parse layer and head indices
            if '_' in head_id:
                parts = head_id.split('_')
                if len(parts) >= 4 and parts[0] == 'layer' and parts[2] == 'head':
                    try:
                        layer_idx = int(parts[1])
                        head_idx = int(parts[3])

                        # Check if indices are valid
                        if layer_idx < len(self.model.layers):
                            layer = self.model.layers[layer_idx]

                            # Ablate this head by zeroing its output weights
                            head_dim = self.model.dim // self.model.num_heads
                            start_idx = head_idx * head_dim
                            end_idx = (head_idx + 1) * head_dim

                            with torch.no_grad():
                                layer.attn.out_proj.weight[:, start_idx:end_idx] = 0
                    except:
                        # Skip if parsing fails
                        pass

    def _compute_circuit_attribution(self, circuit, inputs):
        """
        Compute attribution score for a circuit on specific inputs

        Args:
            circuit: The circuit to analyze
            inputs: Input tensor

        Returns:
            float: Attribution score
        """
        # Run forward pass to get attention patterns
        outputs = self.model(inputs, store_attention=True)

        # Get attention patterns
        attention_patterns = {}
        if hasattr(self.model, 'get_attention_patterns'):
            attention_patterns = self.model.get_attention_patterns()

        # For token circuits, compute attribution based on attention patterns
        if circuit.type == CircuitType.TOKEN:
            return self._compute_token_circuit_attribution(circuit, attention_patterns)

        # Default attribution for other circuit types
        return 0.5  # Placeholder

    def _get_circuit_components(self, circuit):
        """
        Get component names used by a circuit

        Args:
            circuit: The circuit to analyze

        Returns:
            list: Component names
        """
        components = []

        # Extract head names
        head_elements = [e for e in circuit.elements if e.type.name == 'HEAD']
        components.extend([e.id for e in head_elements])

        # Extract MLP components if present
        mlp_elements = [e for e in circuit.elements if e.type.name == 'MLP']
        components.extend([e.id for e in mlp_elements])

        return components