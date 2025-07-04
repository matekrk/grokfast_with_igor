# token_circuit_discovery.py
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from analysis.analyzers.adaptive_token_operations import AdaptiveTokenOperationDetector
from analysis.core.circuit_registry import EnhancedCircuitRegistry as CircuitRegistry
from analysis.core.circuit_schema import Circuit, CircuitType, Element, ElementType, Connection, ConnectionType
from analysis.utils.utils import get_current_callable_info


class TokenCircuitDiscovery:
    """Discover, track, and analyze token-level circuits in transformer models"""

    def __init__(self, model, save_dir: Optional[Union[str, Path]] = None,
                 circuit_registry: Optional[CircuitRegistry] = None):
        """
        Initialize the token circuit discovery system

        Args:
            model: The transformer model to analyze
            save_dir: Directory to save analysis results
            circuit_registry: Optional existing circuit registry
        """
        self.model = model

        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        # Initialize registry
        self.registry = circuit_registry

        # Initialize operation detector
        self.operation_detector = AdaptiveTokenOperationDetector(model, self.registry)

        # Storage for token-level analysis
        self.token_attribution_maps = {}  # Maps output tokens to input token influences
        self.token_circuits = {}  # Identified token relationship patterns
        self.token_circuit_evolution = {}  # How token circuits evolve over training

    def analyze_token_relationships(self, inputs, targets=None, batch_idx=0,
                                    store_attention=True, epoch=None,
                                    analyze_multiple_examples=True, max_examples=10,
                                    random_sampling=True, random_seed=None):
        """
        Analyze token relationships across multiple examples to find robust circuits

        Args:
            inputs: Input tensor [batch_size, seq_len]
            targets: Optional target tensor [batch_size]
            batch_idx: Starting index if not analyzing multiple examples
            store_attention: Whether to store attention patterns
            epoch: Current training epoch (for tracking)
            analyze_multiple_examples: Whether to analyze multiple examples
            max_examples: Maximum number of examples to analyze
            random_sampling: Whether to randomly sample examples instead of taking first ones
            random_seed: Optional seed for reproducible randomization
        Returns:
            Dict with token relationship analysis across examples
        """

        import random
        if random_seed is not None:
            random.seed(random_seed)

        self.model.eval()

        # Standard PyTorch format: [batch_size, seq_len]
        batch_size, seq_len = inputs.shape

        # Validate inputs
        if len(inputs.shape) != 2:
            raise ValueError(f"\t{get_current_callable_info()} @ {epoch}:\tExpected 2D input [batch_size, seq_len], got shape {inputs.shape}")

        # Limit number of examples to analyze
        if analyze_multiple_examples:
            num_examples = min(max_examples, batch_size)
            if random_sampling:
                example_indices = random.sample(range(batch_size), num_examples)
            else:
                example_indices = list(range(num_examples))
        else:
            num_examples = 1
            if random_sampling and batch_idx > 1:
                example_indices = [random.randint(0, batch_size - 1)]
            else:
                example_indices = [min(batch_idx, batch_size - 1)]  # Ensure valid index

        # print(f"\t{get_current_callable_info()} @ {epoch}:\t\tanalyze {num_examples} examples (batch_size={batch_size}, seq_len={seq_len})")

        # Storage for cross-example analysis
        all_copy_mechanisms = []
        all_induction_patterns = []
        all_token_attributions = []
        example_tokens = []

        # Analyze each selected example
        for i, example_idx in enumerate(example_indices):
            if example_idx >= batch_size:
                continue

            # Get single example - keep as [1, seq_len] to maintain batch dimension
            single_input = inputs[example_idx:example_idx + 1]
            single_target = targets[example_idx:example_idx + 1] if targets is not None else None

            # Forward pass with attention storage
            if store_attention:
                outputs = self.model(single_input, store_attention=True)
            else:
                outputs = self.model(single_input)

            # Get attention patterns for this example
            attention_patterns = {}
            if store_attention and hasattr(self.model, 'get_attention_patterns'):
                attention_patterns = self.model.get_attention_patterns()

            # Convert to tokens (use the single example)
            if hasattr(self.model, 'tokenize'):
                tokens = self.model.tokenize(single_input[0])  # Remove batch dimension for tokenization
            else:
                tokens = [f"token_{j}" for j in range(seq_len)]

            example_tokens.append(tokens)

            # Compute token attribution for this example
            token_attribution = self._compute_token_attribution(
                single_input, outputs, attention_patterns, batch_idx=0)
            all_token_attributions.append(token_attribution)

            # Detect operations for this example
            copy_mechanisms = self.operation_detector.detect_copy_mechanisms(attention_patterns)
            induction_patterns = self.operation_detector.detect_induction_patterns(attention_patterns)

            # Store with example context
            for copy in copy_mechanisms:
                copy['example_idx'] = example_idx
                copy['tokens'] = tokens

            for induction in induction_patterns:
                induction['example_idx'] = example_idx
                induction['tokens'] = tokens

            all_copy_mechanisms.extend(copy_mechanisms)
            all_induction_patterns.extend(induction_patterns)

        # Find consistent circuits across examples
        consistent_circuits = self._find_consistent_circuits(
            all_copy_mechanisms, all_induction_patterns,
            all_token_attributions, example_tokens, epoch)

        # Create average token attribution
        if all_token_attributions:
            avg_token_attribution = np.mean(all_token_attributions, axis=0)
        else:
            avg_token_attribution = np.eye(seq_len)

        # Store attribution map
        if epoch is not None:
            self.token_attribution_maps[epoch] = {
                'average': avg_token_attribution,
                'individual': all_token_attributions,
                'num_examples': num_examples
            }

        # Return comprehensive analysis
        return {
            "tokens": example_tokens[0] if example_tokens else [],
            "all_example_tokens": example_tokens,
            "copy_mechanisms": all_copy_mechanisms,
            "induction_patterns": all_induction_patterns,
            "token_attribution": avg_token_attribution,
            "individual_attributions": all_token_attributions,
            "circuits": consistent_circuits,
            "num_examples_analyzed": num_examples
        }

    def _find_consistent_circuits(self, all_copy_mechanisms, all_induction_patterns,
                                  all_token_attributions, example_tokens, epoch):
        """
        Find circuits that appear consistently across multiple examples

        Args:
            all_copy_mechanisms: Copy mechanisms from all examples
            all_induction_patterns: Induction patterns from all examples
            all_token_attributions: Token attributions from all examples
            example_tokens: Tokens from all examples
            epoch: Current epoch

        Returns:
            List of consistent circuits
        """
        circuits = []

        # Group mechanisms by type and structural pattern
        copy_groups = self._group_mechanisms_by_pattern(all_copy_mechanisms, 'copy')
        induction_groups = self._group_mechanisms_by_pattern(all_induction_patterns, 'induction')

        # Create circuits for patterns that appear in multiple examples
        consistency_threshold = max(1, len(example_tokens) // 3)  # Appear in at least 1/3 of examples

        for pattern_key, mechanisms in copy_groups.items():
            if len(mechanisms) >= consistency_threshold:
                # Create circuit for this consistent pattern
                circuit = self._create_consistent_circuit(
                    mechanisms, 'copy', pattern_key, epoch)
                if circuit:
                    circuits.append(circuit)

                    # Register in registry
                    self.registry.register_circuit(circuit, source="token_discovery")
                    self.token_circuits[circuit.id] = circuit

        for pattern_key, mechanisms in induction_groups.items():
            if len(mechanisms) >= consistency_threshold:
                # Create circuit for this consistent pattern
                circuit = self._create_consistent_circuit(
                    mechanisms, 'induction', pattern_key, epoch)
                if circuit:
                    circuits.append(circuit)

                    # Register in registry
                    self.registry.register_circuit(circuit, source="token_discovery")
                    self.token_circuits[circuit.id] = circuit

        return circuits

    def _group_mechanisms_by_pattern(self, mechanisms, mech_type):
        """Group mechanisms by their structural pattern"""
        groups = {}

        for mech in mechanisms:
            if mech_type == 'copy':
                # Group by head and relative position pattern
                head = mech.get('head', 'unknown')
                # Use relative positions instead of absolute to find general patterns
                src_pos = mech.get('source_pos', -1)
                tgt_pos = mech.get('target_pos', -1)
                relative_offset = tgt_pos - src_pos if src_pos >= 0 and tgt_pos >= 0 else 0

                pattern_key = f"{head}_copy_offset_{relative_offset}"

            elif mech_type == 'induction':
                # Group by head and induction pattern structure
                head = mech.get('head', 'unknown')
                # For induction, the pattern is more about the head than positions
                pattern_key = f"{head}_induction"

            else:
                pattern_key = 'unknown'

            if pattern_key not in groups:
                groups[pattern_key] = []
            groups[pattern_key].append(mech)

        return groups

    def _create_consistent_circuit(self, mechanisms, circuit_type, pattern_key, epoch):
        """Create a circuit from consistent mechanisms across examples"""
        if not mechanisms:
            return None

        # Use the first mechanism as template
        template = mechanisms[0]

        # Calculate average strength
        avg_strength = sum(m.get('strength', 0.5) for m in mechanisms) / len(mechanisms)

        # Create representative tokens (use most common pattern)
        if circuit_type == 'copy':
            # For copy circuits, create a general pattern
            head = template.get('head', 'unknown')

            # Find most common relative offset AND position patterns
            offsets = []
            source_positions = []
            target_positions = []

            for m in mechanisms:
                src_pos = m.get('source_pos', -1)
                tgt_pos = m.get('target_pos', -1)
                if src_pos >= 0 and tgt_pos >= 0:
                    offsets.append(tgt_pos - src_pos)
                    source_positions.append(src_pos)
                    target_positions.append(tgt_pos)

            most_common_offset = max(set(offsets), key=offsets.count) if offsets else 1

            # Get representative positions (most common source position)
            most_common_source = max(set(source_positions), key=source_positions.count) if source_positions else -1
            representative_target = most_common_source + most_common_offset if most_common_source >= 0 else -1

            # ✅ FIXED: Pass all necessary kwargs for position_info generation
            circuit_id = self.registry.generate_circuit_id(
                operation_type="copy",
                component_info=head,
                epoch=epoch,
                # Position information
                relative_offset=most_common_offset,
                source_pos=most_common_source,  # ✅ ADD: for absolute position info
                target_pos=representative_target,  # ✅ ADD: for absolute position info
                # Pattern information
                consistency=len(mechanisms),
                examples_found=len(mechanisms),
                avg_strength=avg_strength,
                source="consistent"
            )

            # Create abstract circuit (not tied to specific tokens)
            circuit = Circuit(
                id=circuit_id,
                type=CircuitType.TOKEN,
                elements=[
                    Element(id=f"source_token", type=ElementType.TOKEN,
                            properties={"relative_position": 0, "role": "source"}),
                    Element(id=f"target_token", type=ElementType.TOKEN,
                            properties={"relative_position": most_common_offset, "role": "target"}),
                    Element(id=head, type=ElementType.HEAD,
                            properties={"name": head})
                ],
                connections=[
                    Connection(source="source_token", target=head,
                               strength=avg_strength, type=ConnectionType.ATTENTION),
                    Connection(source=head, target="target_token",
                               strength=avg_strength, type=ConnectionType.ATTENTION)
                ],
                attribution=avg_strength,
                metadata={
                    "operation_type": "copy",
                    "head": head,
                    "relative_offset": most_common_offset,
                    "most_common_source_pos": most_common_source,
                    "representative_target_pos": representative_target,
                    "examples_found": len(mechanisms),
                    "consistency": len(mechanisms),
                    "avg_strength": avg_strength
                },
                discovered_at=epoch
            )
            # print(f"\t{get_current_callable_info()} @ {epoch}:\tconsistent COPY\t{circuit_id} ")
            return circuit

        elif circuit_type == 'induction':
            head = template.get('head', 'unknown')

            # Extract pattern information from all mechanisms
            pattern_distances = []  # Distance from inducer to target
            induction_spans = []  # Distance from inducer to induced
            inducer_positions = []
            induced_positions = []
            target_positions = []

            for mech in mechanisms:
                inducer_pos = mech.get('inducer_pos', -1)
                induced_pos = mech.get('induced_pos', -1)
                target_pos = mech.get('target_pos', -1)

                if inducer_pos >= 0 and target_pos >= 0:
                    pattern_distances.append(target_pos - inducer_pos)
                    inducer_positions.append(inducer_pos)
                    target_positions.append(target_pos)

                if inducer_pos >= 0 and induced_pos >= 0:
                    induction_spans.append(induced_pos - inducer_pos)
                    induced_positions.append(induced_pos)

            # Find most common patterns
            most_common_distance = max(set(pattern_distances), key=pattern_distances.count) if pattern_distances else 1
            most_common_span = max(set(induction_spans), key=induction_spans.count) if induction_spans else 1

            # Representative positions
            most_common_inducer = max(set(inducer_positions), key=inducer_positions.count) if inducer_positions else -1
            most_common_induced = max(set(induced_positions), key=induced_positions.count) if induced_positions else -1
            most_common_target = max(set(target_positions), key=target_positions.count) if target_positions else -1

            # Determine pattern type based on distances
            pattern_type = f"dist_{most_common_distance}_span_{most_common_span}"

            # ✅ FIXED: Pass all necessary kwargs for induction circuits
            circuit_id = self.registry.generate_circuit_id(
                operation_type="induction",
                component_info=head,
                epoch=epoch,
                # info parameters for induction type
                pattern_type=pattern_type,
                pattern_distance=most_common_distance,
                induction_span=most_common_span,
                # Position information
                inducer_pos=most_common_inducer,  # ✅ ADD: for position info
                induced_pos=most_common_induced,  # ✅ ADD: for position info
                target_pos=most_common_target,  # ✅ ADD: for position info
                # Consistency information
                consistency=len(mechanisms),
                examples_found=len(mechanisms),
                avg_strength=avg_strength,
                source="consistent"
            )

            circuit = Circuit(
                id=circuit_id,
                type=CircuitType.TOKEN,
                elements=[
                    Element(id=f"inducer_token", type=ElementType.TOKEN,
                            properties={"role": "inducer", "relative_position": 0}),
                    Element(id=f"induced_token", type=ElementType.TOKEN,
                            properties={"role": "induced", "relative_position": most_common_span}),
                    Element(id=f"target_token", type=ElementType.TOKEN,
                            properties={"role": "target", "relative_position": most_common_distance}),
                    Element(id=head, type=ElementType.HEAD,
                            properties={"name": head})
                ],
                connections=[
                    Connection(source="inducer_token", target=head,
                               strength=avg_strength, type=ConnectionType.ATTENTION),
                    Connection(source=head, target="target_token",
                               strength=avg_strength, type=ConnectionType.ATTENTION)
                ],
                attribution=avg_strength,
                metadata={
                    "operation_type": "induction",
                    "head": head,
                    "pattern_distance": most_common_distance,
                    "induction_span": most_common_span,
                    "pattern_type": pattern_type,
                    "most_common_inducer_pos": most_common_inducer,
                    "most_common_induced_pos": most_common_induced,
                    "most_common_target_pos": most_common_target,
                    "examples_found": len(mechanisms),
                    "consistency": len(mechanisms),
                    "avg_strength": avg_strength
                },
                discovered_at=epoch
            )

            return circuit

        return None

    def analyze_token_relationships_old_backup(self, inputs, targets=None, batch_idx=0,
                                    store_attention=True, epoch=None, analyze_all_batch=False):
        """
        Analyze token relationships for the given inputs

        Args:
            inputs: Input tensor of token IDs
            targets: Optional target tensor
            batch_idx: Which item in the batch to analyze (if analyze_all_batch=False)
            store_attention: Whether to store attention patterns
            epoch: Current training epoch (for tracking)
            analyze_all_batch: Whether to analyze all examples in batch or just one

        Returns:
            Dict with token relationship analysis
        """
        self.model.eval()
        # print(f"\t{get_current_callable_info()} @ {epoch}: \t")
        # Forward pass with attention storage
        if store_attention:
            outputs = self.model(inputs, store_attention=True)
        else:
            outputs = self.model(inputs)

        # Get attention patterns if stored
        attention_patterns = {}
        if store_attention and hasattr(self.model, 'get_attention_patterns'):
            attention_patterns = self.model.get_attention_patterns()

        # Convert inputs to tokens for better interpretability
        if hasattr(self.model, 'tokenize'):
            tokens = self.model.tokenize(inputs[batch_idx] if not analyze_all_batch else inputs[0])
        else:
            # Use positional ids if we can't get actual tokens
            seq_len = min(inputs.shape) if len(inputs.shape) >= 2 else inputs.shape[0]
            tokens = [f"token_{i}" for i in range(seq_len)]

        # Compute token attribution
        if analyze_all_batch:
            token_attribution = self._compute_batch_token_attribution(inputs, outputs, attention_patterns)
            # Use first example for circuit analysis
            single_attribution = token_attribution[0] if len(token_attribution) > 0 else np.eye(len(tokens))
        else:
            token_attribution = self._compute_token_attribution(inputs, outputs, attention_patterns, batch_idx)
            single_attribution = token_attribution

        # Run operation detection on the single attribution matrix
        copy_mechanisms = self.operation_detector.detect_copy_mechanisms(attention_patterns)
        induction_patterns = self.operation_detector.detect_induction_patterns(attention_patterns)

        # Create circuits from detected operations
        circuits = []

        for op_data in copy_mechanisms:
            circuit = self.operation_detector.create_token_operation_circuit(
                op_data, tokens, epoch or 0)
            circuits.append(circuit)

            # Register in registry
            self.registry.register_circuit(circuit, source="token_discovery")
            self.token_circuits[circuit.id] = circuit

        for op_data in induction_patterns:
            circuit = self.operation_detector.create_token_operation_circuit(
                op_data, tokens, epoch or 0)
            circuits.append(circuit)

            # Register in registry
            self.registry.register_circuit(circuit, source="token_discovery")

            # Store in internal tracking
            self.token_circuits[circuit.id] = circuit

        # print(f"\t{get_current_callable_info()} @ {epoch}:\tcreated circuits\t{shorten_layer_head(circuits)}")

        # Return analysis results
        return {
            "tokens": tokens,
            "copy_mechanisms": copy_mechanisms,
            "induction_patterns": induction_patterns,
            "token_attribution": token_attribution,
            "circuits": circuits
        }

    def _compute_token_attribution(self, inputs, outputs, attention_patterns, batch_idx=0):
        """
        Compute token attribution map for a specific example in the batch

        Args:
            inputs: Input tensor [batch_size, seq_len]
            outputs: Output tensor [batch_size, num_tokens]
            attention_patterns: Dict of attention patterns from all heads
            batch_idx: Which example in the batch to analyze

        Returns:
            numpy.ndarray: [seq_len, seq_len] attribution matrix for the specific example
        """

        # Determine sequence length from inputs (standard format: [batch_size, seq_len])
        if len(inputs.shape) >= 2:
            batch_size, seq_len = inputs.shape
        else:
            batch_size, seq_len = 1, inputs.shape[0]

        # Validate batch_idx
        if batch_idx >= batch_size:
            # print(f"\t{get_current_callable_info()}: \twarning\tbatch_idx {batch_idx} >= batch_size {batch_size}, using batch_idx=0")
            batch_idx = 0

        # Double-check with attention patterns if available
        if attention_patterns:
            first_pattern = next(iter(attention_patterns.values()))
            if isinstance(first_pattern, torch.Tensor):
                if len(first_pattern.shape) == 3:  # [batch_size, seq_len, seq_len]
                    pattern_seq_len = first_pattern.shape[1]
                    if pattern_seq_len != seq_len:
                        # print(f"\t{get_current_callable_info()}: \twarning: Input seq_len {seq_len} doesn't match attention pattern seq_len {pattern_seq_len}")
                        seq_len = pattern_seq_len  # Use the attention pattern's dimension
                elif len(first_pattern.shape) == 2:  # [seq_len, seq_len] (single example)
                    seq_len = first_pattern.shape[0]

        # Initialize attribution matrix for this specific example
        token_attribution = np.zeros((seq_len, seq_len))

        # Sum attention weights across all heads for the specific batch example
        valid_patterns = 0
        for head_name, pattern in attention_patterns.items():
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.detach().cpu().numpy()

            # Extract pattern for the specific batch example
            if len(pattern.shape) == 3:  # [batch_size, seq_len, seq_len]
                if batch_idx < pattern.shape[0]:
                    example_pattern = pattern[batch_idx]  # [seq_len, seq_len]
                else:
                    print(f"\t{get_current_callable_info()}:\twarning: batch_idx {batch_idx} >= pattern batch size {pattern.shape[0]}")
                    continue
            elif len(pattern.shape) == 2:  # [seq_len, seq_len] (single example or averaged)
                example_pattern = pattern
            else:
                print(f"\t{get_current_callable_info()}:\tunexpected pattern shape for {head_name}: {pattern.shape}")
                continue

            # Verify dimensions
            if example_pattern.shape == (seq_len, seq_len):
                token_attribution += example_pattern
                valid_patterns += 1
            else:
                # print(f"\t{get_current_callable_info()}: \tskipping pattern {head_name}: shape {example_pattern.shape}, expected ({seq_len}, {seq_len})")
                pass

        # Normalize by number of valid patterns
        if valid_patterns > 0:
            token_attribution = token_attribution / valid_patterns
        else:
            # print(f"\t{get_current_callable_info()}:\twarning: No valid attention patterns found")
            return np.eye(seq_len)  # Identity matrix as fallback

        # Normalize rows to create probability distribution
        row_sums = token_attribution.sum(axis=1, keepdims=True)
        token_attribution = token_attribution / (row_sums + 1e-10)

        return token_attribution

    def _compute_batch_token_attribution(self, inputs, outputs, attention_patterns):
        """
        info compute token attribution maps for all examples in the batch to be used in
         analyze_token_relationship with analyze_all_batch = True

        Args:
            inputs: Input tensor
            outputs: Output tensor
            attention_patterns: Dict of attention patterns from all heads

        Returns:
            numpy.ndarray: [batch_size, seq_len, seq_len] attribution matrices
        """

        # Determine batch size and sequence length
        if attention_patterns:
            first_pattern = next(iter(attention_patterns.values()))
            if isinstance(first_pattern, torch.Tensor):
                if len(first_pattern.shape) == 3:  # [batch_size, seq_len, seq_len]
                    batch_size, seq_len = first_pattern.shape[0], first_pattern.shape[1]
                elif len(first_pattern.shape) == 2:  # [seq_len, seq_len] (single example)
                    batch_size, seq_len = 1, first_pattern.shape[0]
                else:
                    raise ValueError(f"Unexpected attention pattern shape: {first_pattern.shape}")
        else:
            # Determine from inputs
            if len(inputs.shape) >= 2:
                batch_size = max(inputs.shape)
                seq_len = min(inputs.shape)
            else:
                batch_size, seq_len = 1, inputs.shape[0]

        # Initialize attribution matrices for all examples
        batch_token_attribution = np.zeros((batch_size, seq_len, seq_len))

        # Process each example in the batch
        for batch_idx in range(batch_size):
            valid_patterns = 0

            # Sum attention weights across all heads for this example
            for head_name, pattern in attention_patterns.items():
                if isinstance(pattern, torch.Tensor):
                    pattern = pattern.detach().cpu().numpy()

                # Extract pattern for this batch example
                if len(pattern.shape) == 3:  # [batch_size, seq_len, seq_len]
                    if batch_idx < pattern.shape[0]:
                        example_pattern = pattern[batch_idx]
                    else:
                        continue  # Skip if batch_idx out of range
                elif len(pattern.shape) == 2:  # [seq_len, seq_len] (same for all examples)
                    example_pattern = pattern
                else:
                    continue  # Skip invalid patterns

                # Add to attribution if dimensions match
                if example_pattern.shape == (seq_len, seq_len):
                    batch_token_attribution[batch_idx] += example_pattern
                    valid_patterns += 1

            # Normalize this example's attribution
            if valid_patterns > 0:
                batch_token_attribution[batch_idx] /= valid_patterns

                # Normalize rows to create probability distribution
                row_sums = batch_token_attribution[batch_idx].sum(axis=1, keepdims=True)
                batch_token_attribution[batch_idx] = batch_token_attribution[batch_idx] / (row_sums + 1e-10)
            else:
                # Fallback to identity matrix
                batch_token_attribution[batch_idx] = np.eye(seq_len)

        return batch_token_attribution


    def _compute_token_attribution_backup_old(self, inputs, outputs, attention_patterns, batch_idx=0):
        """info old version computing attributions for only one token"""
        """Compute token attribution map to determine influence between tokens"""

        # Handle different input formats
        if len(inputs.shape) == 2:
            # Determine which dimension is sequence length vs batch size
            # Convention: sequence length is usually smaller and first in transformer models
            if hasattr(self.model, 'seq_len'):
                seq_len = self.model.seq_len
            else:
                # Heuristic: sequence length is typically the smaller dimension
                # and usually the first dimension in transformer implementations
                seq_len = inputs.shape[0] if inputs.shape[0] <= inputs.shape[1] else inputs.shape[1]
        else:
            seq_len = inputs.shape[0]

        # Double-check with attention patterns if available
        if attention_patterns:
            first_pattern = next(iter(attention_patterns.values()))
            if isinstance(first_pattern, torch.Tensor):
                pattern_seq_len = first_pattern.shape[0]
                if pattern_seq_len != seq_len:
                    # print(f"\t{get_current_callable_info()}: \twarning: Inferred seq_len {seq_len} doesn't match attention pattern seq_len {pattern_seq_len}")
                    seq_len = pattern_seq_len  # Use the attention pattern's dimension

        # Initialize token attribution matrix with correct dimensions
        token_attribution = np.zeros((seq_len, seq_len))

        # Sum attention weights across all heads
        valid_patterns = 0
        for head_name, pattern in attention_patterns.items():
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.detach().cpu().numpy()

            # Verify dimensions match
            if pattern.shape == (seq_len, seq_len):
                token_attribution += pattern
                valid_patterns += 1
            else:
                print(f"\t{get_current_callable_info()}:\tskipping pattern {head_name}: shape {pattern.shape}, expected ({seq_len}, {seq_len})")

        # Normalize by number of valid patterns
        if valid_patterns > 0:
            token_attribution = token_attribution / valid_patterns
        else:
            # print(f"\t{get_current_callable_info()}:\twarning: No valid attention patterns found for token attribution")
            return np.eye(seq_len)  # Return identity matrix as fallback

        # Normalize rows to sum to 1 (probability distribution)
        row_sums = token_attribution.sum(axis=1, keepdims=True)
        token_attribution = token_attribution / (row_sums + 1e-10)

        return token_attribution
    def perform_token_interventions(self, model, inputs, targets, circuit_spec):
        """Perform token-level interventions to validate circuits"""
        # To be implemented - would modify token representations and measure output changes
        pass

    def identify_token_circuits(self, min_influence=0.1, consistency_threshold=0.8):
        """Identify consistent token circuits from the attribution maps"""
        # Placeholder for the main circuit identification algorithm
        pass

    def compose_with_component_circuits(self, component_circuits):
        """Compose token circuits with component-level circuits"""
        # Placeholder for composition logic - would connect token patterns with implementing components
        pass

    def save_token_circuits(self, filepath: Optional[Path] = None):
        """Save discovered token circuits to file"""
        if filepath is None:
            if self.save_dir is None:
                raise ValueError("No save directory or filepath specified")
            filepath = self.save_dir / "token_circuits.json"

        circuits = list(self.token_circuits.values())
        from analysis.core.circuit_schema import save_circuits
        save_circuits(circuits, filepath)

    def load_token_circuits(self, filepath: Optional[Path] = None):
        """Load token circuits from file"""
        if filepath is None:
            if self.save_dir is None:
                raise ValueError("No save directory or filepath specified")
            filepath = self.save_dir / "token_circuits.json"

        from analysis.core.circuit_schema import load_circuits
        circuits = load_circuits(filepath)

        for circuit in circuits:
            self.token_circuits[circuit.id] = circuit
            self.registry.register_circuit(circuit, source="token_discovery")