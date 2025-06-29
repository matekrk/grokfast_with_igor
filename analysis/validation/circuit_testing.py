# analysis/validation/circuit_testing.py
"""
Real Implementation of Circuit Quality Testing

Implements actual circuit validation through activation patching, behavioral testing,
and functional analysis for modular arithmetic tasks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import copy


class CircuitFunctionalTester:
    """
    info test what circuits actually compute through activation patching and behavioral analysis
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

        # Store original activations for patching
        self.stored_activations = {}
        self.hooks = []

    def test_circuit_on_example(self, circuit_id: str, canonical_circuit,
                                inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """
        Test what a specific circuit actually does on a given example

        Uses activation patching to measure circuit's causal effect
        """

        # Get circuit metadata
        operation_type = canonical_circuit.computational_signature.operation_type

        if operation_type == 'copy':
            return self._test_copy_circuit(circuit_id, canonical_circuit, inputs, targets)
        elif operation_type == 'induction':
            return self._test_induction_circuit(circuit_id, canonical_circuit, inputs, targets)
        else:
            return self._test_generic_circuit(circuit_id, canonical_circuit, inputs, targets)

    def _test_copy_circuit(self, circuit_id: str, canonical_circuit,
                           inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """
        Test copy circuit: Does it actually help copy tokens from source to target positions?
        """

        # Get circuit-specific information
        head_name = canonical_circuit.metadata.get('head', 'unknown')

        # Extract positions from recent instances
        recent_instance = canonical_circuit.instances[-1] if canonical_circuit.instances else None
        if not recent_instance:
            return {'behavior': 'no_instances', 'strength': 0.0}

        source_pos = recent_instance.positions.get('source', 0)
        target_pos = recent_instance.positions.get('target', -1)
        tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]

        # 1. Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(inputs)
            baseline_logits = baseline_output.logits if hasattr(baseline_output, 'logits') else baseline_output
            # baseline_pred = torch.argmax(baseline_logits[0, target_pos], dim=-1)
            baseline_pred = torch.argmax(baseline_logits[0], dim=-1)

        # 2. Test with circuit ablated (zero out attention for this head at these positions)
        ablated_output = self._run_with_ablated_circuit(inputs, head_name, source_pos, target_pos)
        ablated_logits = ablated_output.logits if hasattr(ablated_output, 'logits') else ablated_output
        # ablated_pred = torch.argmax(ablated_logits[0, target_pos], dim=-1)
        ablated_pred = torch.argmax(ablated_logits[0], dim=-1)

        # 3. Measure copy-specific behavior
        # source_token = inputs[0, source_pos].item()
        # target_token = inputs[0, target_pos].item() if target_pos < len(tokens) else -1
        source_token = inputs[0, source_pos].item()
        correct_answer = target_token = targets[0].item() if target_pos < len(tokens) else -1

        # Check if circuit helps predict the source token at target position
        # source_token_prob_baseline = F.softmax(baseline_logits[0], dim=-1)[source_token].item()
        # source_token_prob_ablated = F.softmax(ablated_logits[0], dim=-1)[source_token].item()
        correct_prob_baseline = source_token_prob_baseline = F.softmax(baseline_logits[0], dim=-1)[target_token].item()
        correct_prob_ablated = source_token_prob_ablated = F.softmax(ablated_logits[0], dim=-1)[target_token].item()

        copy_effect = source_token_prob_baseline - source_token_prob_ablated

        # 4. Test on modular arithmetic copy pattern (for your context)
        modular_copy_score = self._test_modular_copy_pattern(
            tokens, source_pos, target_pos, baseline_logits, ablated_logits
        )

        return {
            'behavior': 'copy_circuit',
            'strength': abs(copy_effect),
            'copy_effect': copy_effect,
            'helps_copy': copy_effect > 0.01,  # Threshold for meaningful effect
            'source_token': source_token,
            'target_token': target_token,
            'source_token_prob_change': copy_effect,
            'modular_copy_score': modular_copy_score,
            'positions': {'source': source_pos, 'target': target_pos},
            'baseline_pred': baseline_pred.item(),
            'ablated_pred': ablated_pred.item(),
            'prediction_changed': baseline_pred != ablated_pred
        }

    def _test_induction_circuit(self, circuit_id: str, canonical_circuit,
                                inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """
        Test induction circuit: Does it help with A-B-A-? → B pattern completion?
        """

        head_name = canonical_circuit.metadata.get('head', 'unknown')

        # Get positions from recent instance
        recent_instance = canonical_circuit.instances[-1] if canonical_circuit.instances else None
        if not recent_instance:
            return {'behavior': 'no_instances', 'strength': 0.0}

        inducer_pos = recent_instance.positions.get('inducer', 0)
        target_pos = recent_instance.positions.get('target', -1)
        tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]

        # 1. Get baseline and ablated predictions
        with torch.no_grad():
            baseline_output = self.model(inputs)
            baseline_logits = baseline_output.logits if hasattr(baseline_output, 'logits') else baseline_output

        ablated_output = self._run_with_ablated_circuit(inputs, head_name, inducer_pos, target_pos)
        ablated_logits = ablated_output.logits if hasattr(ablated_output, 'logits') else ablated_output

        # 2. Look for induction pattern in the sequence
        induction_pattern = self._detect_induction_pattern_in_sequence(tokens, inducer_pos, target_pos)

        if induction_pattern['found']:
            # Test if circuit helps predict the expected "B" token
            expected_token = induction_pattern['expected_token']

            # expected_prob_baseline = F.softmax(baseline_logits[0, target_pos], dim=-1)[expected_token].item()
            # expected_prob_ablated = F.softmax(ablated_logits[0, target_pos], dim=-1)[expected_token].item()
            expected_prob_baseline = F.softmax(baseline_logits[0], dim=-1)[expected_token].item()
            expected_prob_ablated = F.softmax(ablated_logits[0], dim=-1)[expected_token].item()

            induction_effect = expected_prob_baseline - expected_prob_ablated

            # Test modular arithmetic induction (for your context)
            modular_induction_score = self._test_modular_induction_pattern(
                tokens, inducer_pos, target_pos, baseline_logits, ablated_logits
            )
        else:
            induction_effect = 0.0
            expected_token = -1
            modular_induction_score = 0.0

        return {
            'behavior': 'induction_circuit',
            'strength': abs(induction_effect),
            'induction_effect': induction_effect,
            'helps_induction': induction_effect > 0.01,
            'pattern_found': induction_pattern['found'],
            'expected_token': expected_token,
            'inducer_pos': inducer_pos,
            'target_pos': target_pos,
            'pattern_distance': target_pos - inducer_pos,
            'modular_induction_score': modular_induction_score,
            'induction_pattern': induction_pattern
        }

    def _test_generic_circuit(self, circuit_id: str, canonical_circuit,
                              inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """
        Test generic circuit by measuring its overall causal effect
        """

        head_name = canonical_circuit.metadata.get('head', 'unknown')

        # Measure overall effect by ablating the entire head
        with torch.no_grad():
            baseline_output = self.model(inputs)
            baseline_logits = baseline_output.logits if hasattr(baseline_output, 'logits') else baseline_output

        # Ablate entire head
        ablated_output = self._run_with_ablated_head(inputs, head_name)
        ablated_logits = ablated_output.logits if hasattr(ablated_output, 'logits') else ablated_output

        # Measure overall effect on accuracy
        if targets.dim() > 1:
            targets = targets.squeeze()

        baseline_loss = F.cross_entropy(baseline_logits.view(-1, baseline_logits.size(-1)),
                                        targets.view(-1), reduction='none')
        ablated_loss = F.cross_entropy(ablated_logits.view(-1, ablated_logits.size(-1)),
                                       targets.view(-1), reduction='none')

        effect_on_loss = (ablated_loss - baseline_loss).mean().item()

        return {
            'behavior': 'generic_circuit',
            'strength': abs(effect_on_loss),
            'effect_on_loss': effect_on_loss,
            'helps_performance': effect_on_loss > 0.01,  # Higher loss when ablated = circuit helps
            'avg_baseline_loss': baseline_loss.mean().item(),
            'avg_ablated_loss': ablated_loss.mean().item()
        }

    def _run_with_ablated_circuit(self, inputs: torch.Tensor, head_name: str,
                                  source_pos: int, target_pos: int) -> torch.Tensor:
        """
        Run model with specific attention connections ablated
        """

        def ablation_hook(module, input, output):
            # output is typically (batch, seq_len, seq_len) attention matrix
            if hasattr(output, 'shape') and len(output.shape) >= 2:
                # Zero out the specific attention connection
                if target_pos < output.shape[-1] and source_pos < output.shape[-1]:
                    output[..., target_pos, source_pos] = 0.0
            return output

        # Parse head name to get layer and head indices
        layer_idx, head_idx = self._parse_head_name(head_name)

        # Register hook on the attention module
        if layer_idx < len(self.model.layers):
            attention_module = self.model.layers[layer_idx].attn
            hook = attention_module.register_forward_hook(ablation_hook)

            try:
                with torch.no_grad():
                    output = self.model(inputs)
            finally:
                hook.remove()
        else:
            # Fallback: run without ablation
            with torch.no_grad():
                output = self.model(inputs)

        return output

    def _run_with_ablated_head(self, inputs: torch.Tensor, head_name: str) -> torch.Tensor:
        """
        Run model with entire attention head ablated
        """

        def head_ablation_hook(module, input, output):
            # Zero out the entire head's output
            layer_idx, head_idx = self._parse_head_name(head_name)
            if hasattr(output, 'shape') and len(output.shape) >= 3:
                # Assuming output shape is (batch, seq_len, num_heads, head_dim)
                if head_idx < output.shape[-2]:
                    output[..., head_idx, :] = 0.0
            return output

        layer_idx, head_idx = self._parse_head_name(head_name)

        if layer_idx < len(self.model.layers):
            attention_module = self.model.layers[layer_idx].attn
            hook = attention_module.register_forward_hook(head_ablation_hook)

            try:
                with torch.no_grad():
                    output = self.model(inputs)
            finally:
                hook.remove()
        else:
            with torch.no_grad():
                output = self.model(inputs)

        return output

    def _parse_head_name(self, head_name: str) -> Tuple[int, int]:
        """
        Parse head name like 'layer_1_head_3' into (layer_idx, head_idx)
        """
        try:
            parts = head_name.split('_')
            layer_idx = int(parts[1]) if 'layer' in parts[0] else 0
            head_idx = int(parts[3]) if len(parts) > 3 and 'head' in parts[2] else 0
            return layer_idx, head_idx
        except (ValueError, IndexError):
            return 0, 0

    def _test_modular_copy_pattern(self, tokens: List[str], source_pos: int, target_pos: int,
                                   baseline_logits: torch.Tensor, ablated_logits: torch.Tensor) -> float:
        """
        Test copy circuit specifically for modular arithmetic patterns

        For modular arithmetic, copy circuits often help with:
        - Copying operands: a + b = ? (copying 'a' or 'b')
        - Copying intermediate results
        """

        if source_pos >= len(tokens) or target_pos >= len(tokens):
            return 0.0

        try:
            source_token = int(tokens[source_pos])

            # In modular arithmetic, check if this looks like operand copying
            # Look for patterns like: a + b = ? where we're copying 'a'
            if target_pos > source_pos + 2:  # Enough space for "a + b"
                # Check if there's an operator between source and target
                has_operator = any(token in ['+', '-', '*', '%']
                                   for token in tokens[source_pos + 1:target_pos])

                if has_operator:
                    # This could be operand copying in arithmetic expression
                    # source_token_prob_baseline = F.softmax(baseline_logits[0, target_pos], dim=-1)[source_token].item()
                    # source_token_prob_ablated = F.softmax(ablated_logits[0, target_pos], dim=-1)[source_token].item()
                    source_token_prob_baseline = F.softmax(baseline_logits[0], dim=-1)[source_token].item()
                    source_token_prob_ablated = F.softmax(ablated_logits[0], dim=-1)[source_token].item()

                    copy_effect = source_token_prob_baseline - source_token_prob_ablated
                    return max(0.0, copy_effect)  # Only positive effects count

            return 0.0

        except (ValueError, IndexError):
            return 0.0

    def _test_modular_induction_pattern(self, tokens: List[str], inducer_pos: int, target_pos: int,
                                        baseline_logits: torch.Tensor, ablated_logits: torch.Tensor) -> float:
        """
        Test induction circuit for modular arithmetic patterns

        For modular arithmetic, induction might help with:
        - Recognizing repeated calculation patterns
        - Learning from previous similar equations
        """

        if inducer_pos >= len(tokens) or target_pos >= len(tokens):
            return 0.0

        try:
            # Look for arithmetic pattern repetition
            # Example: "3 + 5 = 8, 3 + 7 = ?" where we might induce from seeing "3 +" before

            # Simple heuristic: check if inducer position contains a number that appears again
            inducer_token = int(tokens[inducer_pos])

            # Look for the same number appearing later in the sequence
            for i in range(inducer_pos + 1, target_pos):
                try:
                    if int(tokens[i]) == inducer_token:
                        # Found repeated number - this could be induction pattern
                        # Test if circuit helps predict what should come after the repeated pattern

                        # What typically comes after this number in this position?
                        # This is domain-specific to your modular arithmetic setup

                        # For now, just measure the circuit's effect on the final prediction
                        # baseline_probs = F.softmax(baseline_logits[0, target_pos], dim=-1)
                        # ablated_probs = F.softmax(ablated_logits[0, target_pos], dim=-1)
                        baseline_probs = F.softmax(baseline_logits[0], dim=-1)
                        ablated_probs = F.softmax(ablated_logits[0], dim=-1)

                        # Measure KL divergence as proxy for circuit importance
                        kl_div = F.kl_div(ablated_probs.log(), baseline_probs, reduction='sum').item()
                        return min(1.0, kl_div)  # Cap at 1.0

                except (ValueError, IndexError):
                    continue

            return 0.0

        except (ValueError, IndexError):
            return 0.0

    def _detect_induction_pattern_in_sequence(self, tokens: List[str], inducer_pos: int,
                                              target_pos: int) -> Dict[str, Any]:
        """
        Detect if there's an actual A-B-A-? → B induction pattern in the sequence
        """

        if inducer_pos >= len(tokens) or target_pos >= len(tokens) or target_pos <= inducer_pos + 2:
            return {'found': False, 'expected_token': -1}

        try:
            # Look for A-B-A pattern where inducer_pos is first A, target_pos is where B should go
            inducer_token = tokens[inducer_pos]

            # Look for B (next token after first A)
            if inducer_pos + 1 < len(tokens):
                b_token = tokens[inducer_pos + 1]

                # Look for second A before target position
                for i in range(inducer_pos + 2, target_pos):
                    if i < len(tokens) and tokens[i] == inducer_token:
                        # Found A-B-A pattern, expect B at target position
                        try:
                            expected_token = int(b_token)
                            return {
                                'found': True,
                                'expected_token': expected_token,
                                'pattern': f"{inducer_token}-{b_token}-{inducer_token}-?",
                                'positions': [inducer_pos, inducer_pos + 1, i, target_pos]
                            }
                        except ValueError:
                            continue

            return {'found': False, 'expected_token': -1}

        except (ValueError, IndexError):
            return {'found': False, 'expected_token': -1}


class CircuitConsistencyMeasurer:
    """
    info measure how consistent a circuit's behavior is with its expected operation type
    """

    def measure_behavior_consistency(self, behavior: Dict[str, Any], expected_type: str) -> float:
        """
        Measure how consistent the observed behavior is with the expected circuit type

        Args:
            behavior: Output from test_circuit_on_example
            expected_type: Expected operation type ('copy', 'induction', etc.)

        Returns:
            Consistency score (0.0 to 1.0, higher is better)
        """

        if expected_type == 'copy':
            return self._measure_copy_consistency(behavior)
        elif expected_type == 'induction':
            return self._measure_induction_consistency(behavior)
        else:
            return self._measure_generic_consistency(behavior)

    def _measure_copy_consistency(self, behavior: Dict[str, Any]) -> float:
        """
        Measure consistency for copy circuits

        Good copy circuits should:
        1. Actually help with copying (positive copy_effect)
        2. Change predictions in meaningful ways
        3. Have reasonable source/target positions
        """

        if behavior.get('behavior') != 'copy_circuit':
            return 0.0

        consistency_factors = []

        # Factor 1: Does it actually help with copying?
        if behavior.get('helps_copy', False):
            copy_effect = behavior.get('copy_effect', 0.0)
            consistency_factors.append(min(1.0, abs(copy_effect) * 10))  # Scale up small effects
        else:
            consistency_factors.append(0.0)

        # Factor 2: Does it change predictions meaningfully?
        if behavior.get('prediction_changed', False):
            consistency_factors.append(1.0)
        else:
            consistency_factors.append(0.3)  # Partial credit

        # Factor 3: Are the positions reasonable?
        positions = behavior.get('positions', {})
        source_pos = positions.get('source', -1)
        target_pos = positions.get('target', -1)

        if source_pos >= 0 and target_pos >= 0 and source_pos != target_pos:
            consistency_factors.append(1.0)
        else:
            consistency_factors.append(0.0)

        # Factor 4: Modular arithmetic specific score
        modular_score = behavior.get('modular_copy_score', 0.0)
        consistency_factors.append(modular_score)

        # Weighted average
        weights = [0.4, 0.3, 0.2, 0.1]
        consistency = sum(f * w for f, w in zip(consistency_factors, weights))

        return min(1.0, consistency)

    def _measure_induction_consistency(self, behavior: Dict[str, Any]) -> float:
        """
        Measure consistency for induction circuits

        Good induction circuits should:
        1. Help with induction patterns (positive induction_effect)
        2. Find actual induction patterns in the data
        3. Have reasonable pattern distances
        """

        if behavior.get('behavior') != 'induction_circuit':
            return 0.0

        consistency_factors = []

        # Factor 1: Does it help with induction?
        if behavior.get('helps_induction', False):
            induction_effect = behavior.get('induction_effect', 0.0)
            consistency_factors.append(min(1.0, abs(induction_effect) * 10))
        else:
            consistency_factors.append(0.0)

        # Factor 2: Does it find actual patterns?
        if behavior.get('pattern_found', False):
            consistency_factors.append(1.0)
        else:
            consistency_factors.append(0.2)  # Low but not zero

        # Factor 3: Is the pattern distance reasonable?
        pattern_distance = behavior.get('pattern_distance', 0)
        if 1 <= pattern_distance <= 10:  # Reasonable induction distances
            consistency_factors.append(1.0)
        else:
            consistency_factors.append(0.3)

        # Factor 4: Modular arithmetic specific score
        modular_score = behavior.get('modular_induction_score', 0.0)
        consistency_factors.append(modular_score)

        # Weighted average
        weights = [0.4, 0.3, 0.2, 0.1]
        consistency = sum(f * w for f, w in zip(consistency_factors, weights))

        return min(1.0, consistency)

    def _measure_generic_consistency(self, behavior: Dict[str, Any]) -> float:
        """
        Measure consistency for generic circuits
        """

        # For generic circuits, just check if they have meaningful effects
        if behavior.get('helps_performance', False):
            effect_magnitude = behavior.get('strength', 0.0)
            return min(1.0, effect_magnitude * 5)  # Scale up
        else:
            return 0.1  # Very low but not zero


class CircuitInstanceSimilarityCalculator:
    """
    info calculate similarity between different instances of the same circuit
    """

    def calculate_instance_similarity(self, instance1, instance2) -> float:
        """
        Calculate similarity between two circuit instances

        Args:
            instance1, instance2: CircuitInstance objects

        Returns:
            Similarity score (0.0 to 1.0, higher is more similar)
        """

        similarity_factors = []

        # Factor 1: Position similarity
        pos_similarity = self._calculate_position_similarity(instance1, instance2)
        similarity_factors.append(pos_similarity)

        # Factor 2: Strength similarity
        strength_similarity = self._calculate_strength_similarity(instance1, instance2)
        similarity_factors.append(strength_similarity)

        # Factor 3: Token context similarity
        token_similarity = self._calculate_token_similarity(instance1, instance2)
        similarity_factors.append(token_similarity)

        # Factor 4: Temporal proximity
        temporal_similarity = self._calculate_temporal_similarity(instance1, instance2)
        similarity_factors.append(temporal_similarity)

        # Weighted average
        weights = [0.3, 0.25, 0.25, 0.2]
        overall_similarity = sum(f * w for f, w in zip(similarity_factors, weights))

        return min(1.0, overall_similarity)

    def _calculate_position_similarity(self, instance1, instance2) -> float:
        """
        Calculate similarity based on positions used by the circuit
        """

        pos1 = instance1.positions
        pos2 = instance2.positions

        if not pos1 or not pos2:
            return 0.0

        # Compare relative positions rather than absolute
        similarities = []

        common_roles = set(pos1.keys()) & set(pos2.keys())
        if not common_roles:
            return 0.0

        for role in common_roles:
            # For same-length sequences, exact position match is good
            if pos1[role] == pos2[role]:
                similarities.append(1.0)
            else:
                # For different positions, calculate relative similarity
                max_pos = max(len(instance1.tokens), len(instance2.tokens))
                if max_pos > 0:
                    rel_diff = abs(pos1[role] - pos2[role]) / max_pos
                    similarities.append(max(0.0, 1.0 - rel_diff))
                else:
                    similarities.append(0.0)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_strength_similarity(self, instance1, instance2) -> float:
        """
        Calculate similarity based on circuit strength/attribution
        """

        strength1 = instance1.strength
        strength2 = instance2.strength

        if strength1 == 0 and strength2 == 0:
            return 1.0

        # Calculate relative similarity in strength
        max_strength = max(strength1, strength2)
        min_strength = min(strength1, strength2)

        if max_strength > 0:
            return min_strength / max_strength
        else:
            return 0.0

    def _calculate_token_similarity(self, instance1, instance2) -> float:
        """
        Calculate similarity based on token context
        """

        tokens1 = set(instance1.tokens)
        tokens2 = set(instance2.tokens)

        if not tokens1 and not tokens2:
            return 1.0

        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        if union > 0:
            return intersection / union
        else:
            return 0.0

    def _calculate_temporal_similarity(self, instance1, instance2) -> float:
        """
        Calculate similarity based on when instances were detected
        """

        epoch_diff = abs(instance1.epoch - instance2.epoch)

        # Closer in time = more similar
        # Use exponential decay with half-life of 50 epochs
        similarity = np.exp(-epoch_diff / 50.0)

        return similarity


# ============================================================================
# INTEGRATION WITH CircuitQualityAnalyzer
# ============================================================================

def integrate_real_testing_with_quality_analyzer(quality_analyzer):
    """
    Replace mock functions in CircuitQualityAnalyzer with real implementations
    """

    # Create the testing components
    functional_tester = CircuitFunctionalTester(quality_analyzer.model)
    consistency_measurer = CircuitConsistencyMeasurer()
    similarity_calculator = CircuitInstanceSimilarityCalculator()

    # Replace the mock methods
    quality_analyzer.test_circuit_on_example = functional_tester.test_circuit_on_example
    quality_analyzer.measure_behavior_consistency = consistency_measurer.measure_behavior_consistency
    quality_analyzer.calculate_instance_similarity = similarity_calculator.calculate_instance_similarity

    print("✅ CircuitQualityAnalyzer updated with real circuit testing functions")

    return quality_analyzer


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage_with_real_testing():
    """
    Example of how to use the real circuit testing
    """

    # In your training loop:
    # analyzer = CircuitQualityAnalyzer(model, canonical_registry, save_dir)

    # # Replace mock functions with real ones
    # analyzer = integrate_real_testing_with_quality_analyzer(analyzer)

    # # Now run analysis with real testing
    # analysis_results = analyzer.analyze_all_circuits(eval_loader, max_circuits=50)

    # # The results will now contain real behavioral testing data
    # for circuit_id, analysis in analysis_results['circuit_analyses'].items():
    #     functional = analysis['functional_analysis']
    #     if functional.get('status') != 'test_failed':
    #         print(f"Circuit {circuit_id}:")
    #         print(f"  Functional quality: {functional['functional_quality']:.3f}")
    #         print(f"  Behavior consistency: {functional['avg_consistency']:.3f}")
    #         print(f"  Examples tested: {functional['examples_tested']}")

    pass