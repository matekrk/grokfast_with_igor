# token_operations.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from analysis.core.circuit_schema import Circuit, Element, Connection, ElementType, ConnectionType, CircuitType


class TokenOperationDetector:
    """Detector for common token-level operations in transformer models"""

    def __init__(self, model):
        self.model = model

    def detect_copy_mechanisms(self, attention_patterns: Dict[str, torch.Tensor],
                               threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Detect heads that perform token copying operations

        Args:
            attention_patterns: Dictionary mapping head names to attention patterns
            threshold: Minimum attention weight to consider as copying

        Returns:
            List of copying mechanisms with metadata
        """
        copy_mechanisms = []

        for head_name, pattern in attention_patterns.items():
            # Convert to numpy for easier analysis
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.detach().cpu().numpy()

            # For each query position, find positions it attends to strongly
            for query_pos in range(pattern.shape[0]):
                for key_pos in range(query_pos):  # Only look at previous positions
                    if pattern[query_pos, key_pos] > threshold:
                        # Found potential copy mechanism
                        copy_mechanisms.append({
                            "head": head_name,
                            "source_pos": key_pos,
                            "target_pos": query_pos,
                            "strength": float(pattern[query_pos, key_pos]),
                            "type": "copy"
                        })

        return copy_mechanisms

    def detect_induction_patterns(self, attention_patterns: Dict[str, torch.Tensor],
                                  threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Identify induction heads (if A followed by B previously, predict B after A now)

        Args:
            attention_patterns: Dictionary mapping head names to attention patterns
            threshold: Minimum attention weight to consider as induction

        Returns:
            List of induction mechanisms with metadata
        """
        induction_patterns = []

        for head_name, pattern in attention_patterns.items():
            # Convert to numpy for easier analysis
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.detach().cpu().numpy()

            # Induction typically has a specific pattern:
            # For token sequence [A, B, ..., A], the second A attends to the first A

            # Check sequence length
            seq_len = pattern.shape[0]
            if seq_len < 4:  # Need at least [A, B, ..., A] for induction
                continue

            # Look for positions that attend strongly to earlier positions
            for query_pos in range(2, seq_len):  # Start from 3rd position
                max_attended_pos = np.argmax(pattern[query_pos, :query_pos])

                # Check if attention is strong enough
                if pattern[query_pos, max_attended_pos] > threshold:
                    # Check for induction pattern: look at the next position after max_attended_pos
                    next_after_attended = max_attended_pos + 1
                    if next_after_attended < query_pos:
                        induction_patterns.append({
                            "head": head_name,
                            "inducer_pos": max_attended_pos,
                            "induced_pos": next_after_attended,
                            "target_pos": query_pos,
                            "strength": float(pattern[query_pos, max_attended_pos]),
                            "type": "induction"
                        })

        return induction_patterns

    def create_token_operation_circuit(self, operation_data: Dict[str, Any],
                                       tokens: List[str], epoch: int) -> Circuit:
        """Convert an operation detection result into a formal circuit"""
        op_type = operation_data["type"]

        if op_type == "copy":
            return self._create_copy_circuit(operation_data, tokens, epoch)
        elif op_type == "induction":
            return self._create_induction_circuit(operation_data, tokens, epoch)
        else:
            raise ValueError(f"Unknown operation type: {op_type}")

    def _create_copy_circuit(self, operation_data: Dict[str, Any],
                             tokens: List[str], epoch: int) -> Circuit:
        """Create a circuit representing a copy operation"""
        head = operation_data["head"]
        source_pos = operation_data["source_pos"]
        target_pos = operation_data["target_pos"]
        strength = operation_data["strength"]

        # Create unique circuit ID
        circuit_id = f"copy_{head}_{source_pos}_{target_pos}_{epoch}"

        # Create elements for source and target tokens
        source_token = Element(
            id=f"token_{source_pos}",
            type=ElementType.TOKEN,
            properties={"position": source_pos, "token": tokens[source_pos]}
        )

        target_token = Element(
            id=f"token_{target_pos}",
            type=ElementType.TOKEN,
            properties={"position": target_pos, "token": tokens[target_pos]}
        )

        # Create element for the attention head
        attention_head = Element(
            id=head,
            type=ElementType.HEAD,
            properties={"name": head}
        )

        # Create connections
        source_to_head = Connection(
            source=source_token.id,
            target=attention_head.id,
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "read"}
        )

        head_to_target = Connection(
            source=attention_head.id,
            target=target_token.id,
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "write"}
        )

        # Create the circuit
        circuit = Circuit(
            id=circuit_id,
            type=CircuitType.TOKEN,
            elements=[source_token, target_token, attention_head],
            connections=[source_to_head, head_to_target],
            attribution=strength,
            metadata={
                "operation_type": "copy",
                "head": head,
                "source_position": source_pos,
                "target_position": target_pos
            },
            discovered_at=epoch
        )

        return circuit

    def _create_induction_circuit(self, operation_data: Dict[str, Any],
                                  tokens: List[str], epoch: int) -> Circuit:
        """Create a circuit representing an induction operation"""
        head = operation_data["head"]
        inducer_pos = operation_data["inducer_pos"]
        induced_pos = operation_data["induced_pos"]
        target_pos = operation_data["target_pos"]
        strength = operation_data["strength"]

        # Create unique circuit ID
        circuit_id = f"induction_{head}_{inducer_pos}_{induced_pos}_{target_pos}_{epoch}"

        # Create elements for tokens
        inducer_token = Element(
            id=f"token_{inducer_pos}",
            type=ElementType.TOKEN,
            properties={"position": inducer_pos, "token": tokens[inducer_pos]}
        )

        induced_token = Element(
            id=f"token_{induced_pos}",
            type=ElementType.TOKEN,
            properties={"position": induced_pos, "token": tokens[induced_pos]}
        )

        target_token = Element(
            id=f"token_{target_pos}",
            type=ElementType.TOKEN,
            properties={"position": target_pos, "token": tokens[target_pos]}
        )

        # Create element for the attention head
        attention_head = Element(
            id=head,
            type=ElementType.HEAD,
            properties={"name": head}
        )

        # Create connections
        inducer_to_head = Connection(
            source=inducer_token.id,
            target=attention_head.id,
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "read"}
        )

        head_to_target = Connection(
            source=attention_head.id,
            target=target_token.id,
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "write"}
        )

        # "Induced" connection (the semantic relationship)
        induced_relation = Connection(
            source=induced_token.id,
            target=target_token.id,
            strength=strength * 0.8,  # Slightly weaker
            type=ConnectionType.COMPOSITE,
            properties={"operation": "predict"}
        )

        # Create the circuit
        circuit = Circuit(
            id=circuit_id,
            type=CircuitType.TOKEN,
            elements=[inducer_token, induced_token, target_token, attention_head],
            connections=[inducer_to_head, head_to_target, induced_relation],
            attribution=strength,
            metadata={
                "operation_type": "induction",
                "head": head,
                "inducer_position": inducer_pos,
                "induced_position": induced_pos,
                "target_position": target_pos
            },
            discovered_at=epoch
        )

        return circuit