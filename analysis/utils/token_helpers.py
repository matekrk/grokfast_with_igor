# analysis/helpers/token_helpers.py
"""
Helper functions for token-level circuit analysis
"""
# import torch
# import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from analysis.core.circuit_schema import Circuit, CircuitType, Element, ElementType, Connection, ConnectionType



def analyze_tokens_adaptively(detector, inputs, targets, epoch, total_epochs, accuracy):
    """Enhanced token analysis with adaptive detector"""
    # Convert inputs to tokens (simplified)
    tokens = [f"token_{i}" for i in inputs[0].cpu().numpy()]

    # Run forward pass with attention storage
    outputs = detector.model(inputs, store_attention=True)
    attention_patterns = detector.model.get_attention_patterns()

    # Adaptive copy detection
    copy_mechanisms = detector.detect_copy_mechanisms_adaptive(
        attention_patterns=attention_patterns,
        tokens=tokens,
        epoch=epoch,
        total_epochs=total_epochs,
        model_accuracy=accuracy,
        content_aware=True
    )

    # Adaptive induction detection
    induction_patterns = detector.detect_induction_patterns_adaptive(
        attention_patterns=attention_patterns,
        tokens=tokens,
        epoch=epoch,
        total_epochs=total_epochs,
        model_accuracy=accuracy
    )

    # Prune unstable circuits
    stable_copy = detector.prune_unstable_circuits(copy_mechanisms, epoch)
    stable_induction = detector.prune_unstable_circuits(induction_patterns, epoch)

    return {
        "copy_mechanisms": stable_copy,
        "induction_patterns": stable_induction,
        "emergence_timeline": detector.get_emergence_timeline(),
        "total_detected": len(stable_copy) + len(stable_induction)
    }


def convert_mechanisms_to_circuits(mechanisms, operation_type: str, epoch: int, registry) -> List[Circuit]:
    """Convert detected mechanisms to formal circuits"""
    circuits = []

    for mechanism in mechanisms:
        if operation_type == "copy":
            circuit = create_copy_circuit_from_mechanism(mechanism, epoch, registry)
        elif operation_type == "induction":
            circuit = create_induction_circuit_from_mechanism(mechanism, epoch, registry)
        else:
            continue

        if circuit:
            circuits.append(circuit)

    return circuits


def create_copy_circuit_from_mechanism(mechanism: Dict[str, Any], epoch: int, registry) -> Optional[Circuit]:
    """Create a copy circuit from a detected mechanism"""
    head = mechanism["head"]
    source_pos = mechanism["source_pos"]
    target_pos = mechanism["target_pos"]
    strength = mechanism["attention_strength"]

    # Generate circuit ID using registry method
    circuit_id = registry.generate_circuit_id(
        operation_type="copy",
        component_info=head,
        epoch=epoch,
        relative_offset=target_pos - source_pos,
        source_pos=source_pos,
        target_pos=target_pos,
        source="adaptive_detector",
        consistency=1
    )

    # Create elements
    source_token = Element(
        id=f"token_{source_pos}",
        type=ElementType.TOKEN,
        properties={"position": source_pos, "role": "source"}
    )

    target_token = Element(
        id=f"token_{target_pos}",
        type=ElementType.TOKEN,
        properties={"position": target_pos, "role": "target"}
    )

    attention_head = Element(
        id=head,
        type=ElementType.HEAD,
        properties={"name": head}
    )

    # Create connections
    connections = [
        Connection(
            source=source_token.id,
            target=attention_head.id,
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "read"}
        ),
        Connection(
            source=attention_head.id,
            target=target_token.id,
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "write"}
        )
    ]

    # Create circuit with enhanced metadata
    circuit = Circuit(
        id=circuit_id,
        type=CircuitType.TOKEN,
        elements=[source_token, target_token, attention_head],
        connections=connections,
        attribution=strength,
        metadata={
            "operation_type": "copy",
            "head": head,
            "source_position": source_pos,
            "target_position": target_pos,
            "relative_offset": target_pos - source_pos,
            "copy_type": mechanism.get("copy_type", "positional_only"),
            "content_strength": mechanism.get("content_strength", strength),
            "reliability": mechanism.get("reliability", 0.5),
            "detection_threshold": mechanism.get("detection_threshold", 0.8)
        },
        discovered_at=epoch
    )

    return circuit


def create_induction_circuit_from_mechanism(mechanism: Dict[str, Any], epoch: int, registry) -> Optional[Circuit]:
    """Create an induction circuit from a detected mechanism"""
    head = mechanism["head"]
    inducer_pos = mechanism["inducer_pos"]
    induced_pos = mechanism["induced_pos"]
    target_pos = mechanism["target_pos"]
    strength = mechanism["attention_strength"]

    # Generate circuit ID
    pattern_distance = target_pos - induced_pos
    pattern_type = f"dist_{pattern_distance}"

    circuit_id = registry.generate_circuit_id(
        operation_type="induction",
        component_info=head,
        epoch=epoch,
        pattern_type=pattern_type,
        inducer_pos=inducer_pos,
        induced_pos=induced_pos,
        target_pos=target_pos,
        strength=strength,
        source="adaptive_detector",
        consistency=1
    )

    # Create elements
    elements = [
        Element(
            id=f"token_{inducer_pos}",
            type=ElementType.TOKEN,
            properties={"position": inducer_pos, "role": "inducer"}
        ),
        Element(
            id=f"token_{induced_pos}",
            type=ElementType.TOKEN,
            properties={"position": induced_pos, "role": "induced"}
        ),
        Element(
            id=f"token_{target_pos}",
            type=ElementType.TOKEN,
            properties={"position": target_pos, "role": "target"}
        ),
        Element(
            id=head,
            type=ElementType.HEAD,
            properties={"name": head}
        )
    ]

    # Create connections
    connections = [
        Connection(
            source=f"token_{inducer_pos}",
            target=head,
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "read"}
        ),
        Connection(
            source=head,
            target=f"token_{target_pos}",
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "write"}
        ),
        Connection(
            source=f"token_{induced_pos}",
            target=f"token_{target_pos}",
            strength=strength * 0.8,
            type=ConnectionType.COMPOSITE,
            properties={"operation": "predict"}
        )
    ]

    # Create circuit
    circuit = Circuit(
        id=circuit_id,
        type=CircuitType.TOKEN,
        elements=elements,
        connections=connections,
        attribution=strength,
        metadata={
            "operation_type": "induction",
            "head": head,
            "inducer_position": inducer_pos,
            "induced_position": induced_pos,
            "target_position": target_pos,
            "pattern_distance": pattern_distance,
            "pattern_type": pattern_type,
            "reliability": mechanism.get("reliability", 0.5),
            "detection_threshold": mechanism.get("detection_threshold", 0.7)
        },
        discovered_at=epoch
    )

    return circuit


def analyze_content_vs_positional_patterns(mechanisms: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the balance between content-based and positional patterns"""
    if not mechanisms:
        return {"total": 0}

    content_types = {}
    for mechanism in mechanisms:
        copy_type = mechanism.get("copy_type", "unknown")
        content_types[copy_type] = content_types.get(copy_type, 0) + 1

    total = len(mechanisms)
    analysis = {
        "total": total,
        "content_types": content_types,
        "content_aware_ratio": (content_types.get("exact_token_copy", 0) +
                                content_types.get("pattern_completion", 0)) / max(1, total),
        "positional_ratio": content_types.get("positional_only", 0) / max(1, total)
    }

    return analysis