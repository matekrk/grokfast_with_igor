# Day 4: Cross-Level Circuit Composition Implementation Plan

# ============================================================================
# STEP 4.1: Create Cross-Level Circuit Detector
# ============================================================================

# File: analysis/analyzers/cross_level_detector.py (NEW)
"""
Cross-Level Circuit Detector for:
1. Functional Circuits (Token + Component combinations)
2. Sparse Feature Circuits (Component + Subspace interactions)
3. Representation Circuits (Token + Subspace relationships)
4. Multi-Level Validation Framework
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import torch

from analysis.core.circuit_schema import Circuit, CircuitType, Element, ElementType, Connection, ConnectionType
from analysis.core.circuit_registry import EnhancedCircuitRegistry


class CrossLevelCircuitDetector:
    """Detector for cross-level circuit compositions"""

    def __init__(self, model, registry: EnhancedCircuitRegistry,
                 save_dir: Optional[Path] = None, logger=None):
        self.model = model
        self.registry = registry
        self.save_dir = save_dir
        self.logger = logger

        # Storage for cross-level analysis
        self.functional_circuits = {}
        self.sparse_feature_circuits = {}
        self.representation_circuits = {}

    def analyze_cross_level_circuits(self, epoch: int, eval_loader, thresholds,
                                     current_accuracy: float) -> Dict[str, Any]:
        """Comprehensive cross-level circuit analysis"""

        results = {}

        # 1. Functional Circuits (Token + Component)
        if self.logger:
            self.logger.debug("🔍 Analyzing functional circuits (Token + Component)")

        functional_results = self.analyze_functional_circuits(
            epoch, eval_loader, thresholds, current_accuracy
        )
        results["functional_circuits"] = functional_results

        # 2. Sparse Feature Circuits (Component + Subspace)
        if self.logger:
            self.logger.debug("🔍 Analyzing sparse feature circuits (Component + Subspace)")

        sparse_feature_results = self.analyze_sparse_feature_circuits(
            epoch, eval_loader, thresholds, current_accuracy
        )
        results["sparse_feature_circuits"] = sparse_feature_results

        # 3. Representation Circuits (Token + Subspace)
        if self.logger:
            self.logger.debug("🔍 Analyzing representation circuits (Token + Subspace)")

        representation_results = self.analyze_representation_circuits(
            epoch, eval_loader, thresholds, current_accuracy
        )
        results["representation_circuits"] = representation_results

        # 4. Multi-Level Validation
        all_cross_level_circuits = (
                functional_results["circuits"] +
                sparse_feature_results["circuits"] +
                representation_results["circuits"]
        )

        validation_results = self.validate_cross_level_circuits(
            all_cross_level_circuits, eval_loader
        )
        results["validation_results"] = validation_results

        # Log comprehensive results
        if self.logger:
            cross_level_metrics = {
                "functional_circuits": len(functional_results["circuits"]),
                "sparse_feature_circuits": len(sparse_feature_results["circuits"]),
                "representation_circuits": len(representation_results["circuits"]),
                "total_cross_level": len(all_cross_level_circuits),
                "validation_accuracy": np.mean([v["accuracy_impact"] for v in validation_results])
            }
            self.logger.log_metrics(cross_level_metrics, step=epoch, category="cross_level_analysis")

        return results

    # ========================================================================
    # FUNCTIONAL CIRCUITS (Token + Component)
    # ========================================================================

    def analyze_functional_circuits(self, epoch: int, eval_loader, thresholds,
                                    current_accuracy: float) -> Dict[str, Any]:
        """Analyze functional circuits (Token + Component combinations)"""

        # Get existing circuits by type
        token_circuits = self.registry.get_circuits_by_type(CircuitType.TOKEN)
        component_circuits = self.registry.get_circuits_by_type(CircuitType.COMPONENT)

        functional_circuits = []
        functional_threshold = thresholds.get_threshold("functional", epoch, 1000, current_accuracy)

        # Find token-component combinations that work together
        for token_circuit in token_circuits:
            for component_circuit in component_circuits:
                interaction_strength = self._calculate_token_component_interaction(
                    token_circuit, component_circuit
                )

                if interaction_strength > functional_threshold:
                    functional_circuit = self._create_functional_circuit(
                        token_circuit, component_circuit, interaction_strength, epoch
                    )
                    functional_circuits.append(functional_circuit)

                    # Register with enhanced metadata
                    self.registry.register_circuit_enhanced(
                        circuit=functional_circuit,
                        source="cross_level_functional_analysis",
                        epoch=epoch,
                        detection_method="functional_detector",
                        confidence=interaction_strength,
                        total_epochs=1000
                    )

        return {
            "circuits": functional_circuits,
            "token_circuits_analyzed": len(token_circuits),
            "component_circuits_analyzed": len(component_circuits),
            "functional_threshold": functional_threshold,
            "interactions_found": len(functional_circuits)
        }

    def _calculate_token_component_interaction(self, token_circuit: Circuit,
                                               component_circuit: Circuit) -> float:
        """Calculate interaction strength between token and component circuits"""

        # Check for shared components (attention heads, etc.)
        token_elements = set(e.id for e in token_circuit.elements)
        component_elements = set(e.id for e in component_circuit.elements)

        shared_elements = token_elements.intersection(component_elements)
        if not shared_elements:
            return 0.0

        # Base interaction from shared components
        base_interaction = len(shared_elements) / max(len(token_elements), len(component_elements))

        # Boost for functional compatibility
        functional_boost = self._calculate_functional_compatibility(token_circuit, component_circuit)

        # Temporal alignment boost
        temporal_boost = self._calculate_temporal_alignment(token_circuit, component_circuit)

        total_interaction = min(1.0, base_interaction + functional_boost + temporal_boost)
        return total_interaction

    def _calculate_functional_compatibility(self, token_circuit: Circuit,
                                            component_circuit: Circuit) -> float:
        """Calculate functional compatibility between circuits"""

        token_type = token_circuit.metadata.get('operation_type', '')
        comp_type = component_circuit.metadata.get('operation_type', '')

        # Define compatibility matrix
        compatibility_matrix = {
            ('copy', 'head_head_interaction'): 0.4,
            ('copy', 'head_mlp_interaction'): 0.3,
            ('induction', 'head_head_interaction'): 0.5,
            ('induction', 'multi_head_cooperation'): 0.6,
        }

        return compatibility_matrix.get((token_type, comp_type), 0.0)

    def _calculate_temporal_alignment(self, token_circuit: Circuit,
                                      component_circuit: Circuit) -> float:
        """Calculate temporal alignment between circuit discoveries"""

        token_epoch = token_circuit.discovered_at or 0
        comp_epoch = component_circuit.discovered_at or 0

        # Circuits discovered close in time are more likely to be related
        epoch_diff = abs(token_epoch - comp_epoch)

        if epoch_diff <= 10:
            return 0.2
        elif epoch_diff <= 50:
            return 0.1
        else:
            return 0.0

    def _create_functional_circuit(self, token_circuit: Circuit, component_circuit: Circuit,
                                   strength: float, epoch: int) -> Circuit:
        """Create a functional circuit from token + component combination"""

        # Generate unique ID
        circuit_id = f"functional_{token_circuit.id}_{component_circuit.id}_{epoch}"

        # Combine elements from both circuits
        combined_elements = list(token_circuit.elements)

        # Add component elements that aren't already included
        for element in component_circuit.elements:
            if not any(e.id == element.id for e in combined_elements):
                combined_elements.append(element)

        # Create functional connections
        combined_connections = list(token_circuit.connections) + list(component_circuit.connections)

        # Add meta-connection showing functional relationship
        if token_circuit.elements and component_circuit.elements:
            functional_connection = Connection(
                source=token_circuit.elements[0].id,
                target=component_circuit.elements[0].id,
                strength=strength,
                type=ConnectionType.COMPOSITE,
                properties={"relationship": "implements", "cross_level": True}
            )
            combined_connections.append(functional_connection)

        # Create functional circuit
        circuit = Circuit(
            id=circuit_id,
            type=CircuitType.FUNCTIONAL,
            elements=combined_elements,
            connections=combined_connections,
            attribution=strength,
            metadata={
                "operation_type": "functional_composition",
                "token_circuit": token_circuit.id,
                "component_circuit": component_circuit.id,
                "composition_type": "token_plus_component",
                "interaction_strength": strength,
                "functional_compatibility": self._calculate_functional_compatibility(token_circuit, component_circuit)
            },
            discovered_at=epoch
        )

        return circuit

    # ========================================================================
    # SPARSE FEATURE CIRCUITS (Component + Subspace)
    # ========================================================================

    def analyze_sparse_feature_circuits(self, epoch: int, eval_loader, thresholds,
                                        current_accuracy: float) -> Dict[str, Any]:
        """Analyze sparse feature circuits (Component + Subspace interactions)"""

        # Get existing circuits
        component_circuits = self.registry.get_circuits_by_type(CircuitType.COMPONENT)
        subspace_circuits = self.registry.get_circuits_by_type(CircuitType.SUBSPACE)

        sparse_feature_circuits = []
        sparse_threshold = thresholds.get_threshold("sparse_feature", epoch, 1000, current_accuracy)

        # Find component-subspace interactions
        for component_circuit in component_circuits:
            for subspace_circuit in subspace_circuits:
                interaction_strength = self._calculate_component_subspace_interaction(
                    component_circuit, subspace_circuit
                )

                if interaction_strength > sparse_threshold:
                    sparse_circuit = self._create_sparse_feature_circuit(
                        component_circuit, subspace_circuit, interaction_strength, epoch
                    )
                    sparse_feature_circuits.append(sparse_circuit)

                    # Register with enhanced metadata
                    self.registry.register_circuit_enhanced(
                        circuit=sparse_circuit,
                        source="cross_level_sparse_feature_analysis",
                        epoch=epoch,
                        detection_method="sparse_feature_detector",
                        confidence=interaction_strength,
                        total_epochs=1000
                    )

        return {
            "circuits": sparse_feature_circuits,
            "component_circuits_analyzed": len(component_circuits),
            "subspace_circuits_analyzed": len(subspace_circuits),
            "sparse_threshold": sparse_threshold,
            "interactions_found": len(sparse_feature_circuits)
        }

    def _calculate_component_subspace_interaction(self, component_circuit: Circuit,
                                                  subspace_circuit: Circuit) -> float:
        """Calculate interaction between component and subspace circuits"""

        # Check layer alignment
        comp_layer = self._extract_layer_info(component_circuit)
        subspace_layer = subspace_circuit.metadata.get("layer", -1)

        if comp_layer != subspace_layer:
            return 0.0  # Must be in same layer

        # Check for MLP component involvement
        has_mlp_component = any(e.type == ElementType.MLP for e in component_circuit.elements)
        if not has_mlp_component:
            return 0.0  # Component must involve MLP

        # Calculate interaction based on feature importance and component strength
        component_strength = component_circuit.attribution
        subspace_importance = subspace_circuit.metadata.get("importance", 0.0)

        interaction = component_strength * subspace_importance

        # Boost for temporal alignment
        temporal_boost = self._calculate_temporal_alignment(component_circuit, subspace_circuit)

        return min(1.0, interaction + temporal_boost)

    def _extract_layer_info(self, circuit: Circuit) -> int:
        """Extract layer information from circuit"""

        # Look for layer info in metadata
        if "layer" in circuit.metadata:
            return circuit.metadata["layer"]

        # Extract from element IDs
        for element in circuit.elements:
            if "layer_" in element.id:
                try:
                    layer_num = int(element.id.split("layer_")[1].split("_")[0])
                    return layer_num
                except:
                    continue

        return -1  # Unknown layer

    def _create_sparse_feature_circuit(self, component_circuit: Circuit, subspace_circuit: Circuit,
                                       strength: float, epoch: int) -> Circuit:
        """Create sparse feature circuit from component + subspace combination"""

        # Generate unique ID
        circuit_id = f"sparse_feature_{component_circuit.id}_{subspace_circuit.id}_{epoch}"

        # Combine elements
        combined_elements = list(component_circuit.elements) + list(subspace_circuit.elements)

        # Combine connections
        combined_connections = list(component_circuit.connections) + list(subspace_circuit.connections)

        # Add sparse feature connection
        if component_circuit.elements and subspace_circuit.elements:
            sparse_connection = Connection(
                source=component_circuit.elements[0].id,
                target=subspace_circuit.elements[0].id,
                strength=strength,
                type=ConnectionType.COMPOSITE,
                properties={"relationship": "activates_sparse_feature", "cross_level": True}
            )
            combined_connections.append(sparse_connection)

        circuit = Circuit(
            id=circuit_id,
            type=CircuitType.HYBRID,  # Use HYBRID for cross-level circuits
            elements=combined_elements,
            connections=combined_connections,
            attribution=strength,
            metadata={
                "operation_type": "sparse_feature_interaction",
                "component_circuit": component_circuit.id,
                "subspace_circuit": subspace_circuit.id,
                "composition_type": "component_plus_subspace",
                "layer": subspace_circuit.metadata.get("layer", -1),
                "interaction_strength": strength
            },
            discovered_at=epoch
        )

        return circuit

    # ========================================================================
    # REPRESENTATION CIRCUITS (Token + Subspace)
    # ========================================================================

    def analyze_representation_circuits(self, epoch: int, eval_loader, thresholds,
                                        current_accuracy: float) -> Dict[str, Any]:
        """Analyze representation circuits (Token + Subspace relationships)"""

        # Get existing circuits
        token_circuits = self.registry.get_circuits_by_type(CircuitType.TOKEN)
        subspace_circuits = self.registry.get_circuits_by_type(CircuitType.SUBSPACE)

        representation_circuits = []
        representation_threshold = thresholds.get_threshold("representation", epoch, 1000, current_accuracy)

        # Find token-subspace relationships through embedding analysis
        for token_circuit in token_circuits:
            for subspace_circuit in subspace_circuits:
                interaction_strength = self._calculate_token_subspace_interaction(
                    token_circuit, subspace_circuit, eval_loader
                )

                if interaction_strength > representation_threshold:
                    representation_circuit = self._create_representation_circuit(
                        token_circuit, subspace_circuit, interaction_strength, epoch
                    )
                    representation_circuits.append(representation_circuit)

                    # Register with enhanced metadata
                    self.registry.register_circuit_enhanced(
                        circuit=representation_circuit,
                        source="cross_level_representation_analysis",
                        epoch=epoch,
                        detection_method="representation_detector",
                        confidence=interaction_strength,
                        total_epochs=1000
                    )

        return {
            "circuits": representation_circuits,
            "token_circuits_analyzed": len(token_circuits),
            "subspace_circuits_analyzed": len(subspace_circuits),
            "representation_threshold": representation_threshold,
            "interactions_found": len(representation_circuits)
        }

    def _calculate_token_subspace_interaction(self, token_circuit: Circuit,
                                              subspace_circuit: Circuit, eval_loader) -> float:
        """Calculate interaction between token patterns and subspace representations"""

        # Simplified analysis - in practice, would analyze embedding clusters
        # and how token patterns correlate with subspace activations

        # Check if token pattern involves the same layer as subspace
        token_elements = [e.id for e in token_circuit.elements if e.type == ElementType.HEAD]
        subspace_layer = subspace_circuit.metadata.get("layer", -1)

        token_layers = set()
        for element_id in token_elements:
            if "layer_" in element_id:
                try:
                    layer_num = int(element_id.split("layer_")[1].split("_")[0])
                    token_layers.add(layer_num)
                except:
                    continue

        if subspace_layer not in token_layers:
            return 0.0

        # Base interaction from layer alignment
        base_interaction = 0.3

        # Boost from attribution alignment
        attribution_similarity = min(token_circuit.attribution, subspace_circuit.attribution)

        # Boost from operation type compatibility
        operation_boost = 0.0
        token_op = token_circuit.metadata.get('operation_type', '')
        if token_op in ['copy', 'induction']:
            operation_boost = 0.2

        total_interaction = base_interaction + attribution_similarity + operation_boost
        return min(1.0, total_interaction)

    def _create_representation_circuit(self, token_circuit: Circuit, subspace_circuit: Circuit,
                                       strength: float, epoch: int) -> Circuit:
        """Create representation circuit from token + subspace combination"""

        # Generate unique ID
        circuit_id = f"representation_{token_circuit.id}_{subspace_circuit.id}_{epoch}"

        # Combine elements
        combined_elements = list(token_circuit.elements) + list(subspace_circuit.elements)

        # Combine connections
        combined_connections = list(token_circuit.connections) + list(subspace_circuit.connections)

        # Add representation connection
        if token_circuit.elements and subspace_circuit.elements:
            representation_connection = Connection(
                source=token_circuit.elements[0].id,
                target=subspace_circuit.elements[0].id,
                strength=strength,
                type=ConnectionType.COMPOSITE,
                properties={"relationship": "represented_in_subspace", "cross_level": True}
            )
            combined_connections.append(representation_connection)

        circuit = Circuit(
            id=circuit_id,
            type=CircuitType.HYBRID,  # Use HYBRID for cross-level circuits
            elements=combined_elements,
            connections=combined_connections,
            attribution=strength,
            metadata={
                "operation_type": "representation_clustering",
                "token_circuit": token_circuit.id,
                "subspace_circuit": subspace_circuit.id,
                "composition_type": "token_plus_subspace",
                "interaction_strength": strength
            },
            discovered_at=epoch
        )

        return circuit

    # ========================================================================
    # MULTI-LEVEL VALIDATION FRAMEWORK
    # ========================================================================

    def validate_cross_level_circuits(self, circuits: List[Circuit], eval_loader) -> List[Dict[str, Any]]:
        """Validate cross-level circuits through multi-level ablation"""

        validation_results = []

        for circuit in circuits:
            validation_result = self._validate_cross_level_circuit(circuit, eval_loader)
            validation_results.append(validation_result)

        return validation_results

    def _validate_cross_level_circuit(self, circuit: Circuit, eval_loader) -> Dict[str, Any]:
        """Validate a single cross-level circuit"""

        # Get baseline performance
        baseline_accuracy = self._evaluate_model(eval_loader)

        # Perform multi-level ablation
        accuracy_after_ablation = self._perform_multi_level_ablation(circuit, eval_loader)

        accuracy_impact = baseline_accuracy - accuracy_after_ablation

        return {
            "circuit_id": circuit.id,
            "circuit_type": circuit.type.value,
            "baseline_accuracy": baseline_accuracy,
            "ablated_accuracy": accuracy_after_ablation,
            "accuracy_impact": accuracy_impact,
            "validation_passed": accuracy_impact > 0.005,  # 0.5% impact threshold
            "composition_type": circuit.metadata.get("composition_type", "unknown")
        }

    def _perform_multi_level_ablation(self, circuit: Circuit, eval_loader) -> float:
        """Perform ablation across multiple levels of the circuit"""

        # Store original model state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        try:
            # Ablate different components based on circuit composition
            composition_type = circuit.metadata.get("composition_type", "")

            if composition_type == "token_plus_component":
                self._ablate_token_component_circuit(circuit)
            elif composition_type == "component_plus_subspace":
                self._ablate_component_subspace_circuit(circuit)
            elif composition_type == "token_plus_subspace":
                self._ablate_token_subspace_circuit(circuit)

            # Evaluate with ablations
            ablated_accuracy = self._evaluate_model(eval_loader)

            return ablated_accuracy

        finally:
            # Restore original state
            self.model.load_state_dict(original_state)

    def _ablate_token_component_circuit(self, circuit: Circuit):
        """
        Ablate a functional circuit (token + component combination)

        Args:
            circuit: Functional circuit to ablate
        """
        # Extract circuit information
        token_circuit_id = circuit.metadata.get("token_circuit")
        component_circuit_id = circuit.metadata.get("component_circuit")

        # Get the original circuits from registry
        token_circuit = self.registry.get_circuit(token_circuit_id) if token_circuit_id else None
        component_circuit = self.registry.get_circuit(component_circuit_id) if component_circuit_id else None

        # Ablate token circuit components (attention heads)
        if token_circuit:
            self._ablate_token_circuit_elements(token_circuit)

        # Ablate component circuit elements (head interactions, MLP interactions)
        if component_circuit:
            self._ablate_component_circuit_elements(component_circuit)

        # Additionally, ablate any shared elements in the functional circuit
        for element in circuit.elements:
            if element.type == ElementType.HEAD:
                self._ablate_attention_head(element.id)
            elif element.type == ElementType.MLP:
                self._ablate_mlp_component(element.id)

    def _ablate_component_subspace_circuit(self, circuit: Circuit):
        """
        Ablate a sparse feature circuit (component + subspace combination)

        Args:
            circuit: Sparse feature circuit to ablate
        """
        # Extract circuit information
        component_circuit_id = circuit.metadata.get("component_circuit")
        subspace_circuit_id = circuit.metadata.get("subspace_circuit")
        layer_idx = circuit.metadata.get("layer", -1)

        # Get the original circuits from registry
        component_circuit = self.registry.get_circuit(component_circuit_id) if component_circuit_id else None
        subspace_circuit = self.registry.get_circuit(subspace_circuit_id) if subspace_circuit_id else None

        # Ablate component circuit elements
        if component_circuit:
            self._ablate_component_circuit_elements(component_circuit)

        # Ablate subspace circuit elements (specific neurons/directions)
        if subspace_circuit and layer_idx >= 0:
            self._ablate_subspace_circuit_elements(subspace_circuit, layer_idx)

        # Ablate any circuit-specific elements
        for element in circuit.elements:
            if element.type == ElementType.MLP:
                # Ablate specific MLP neurons involved in sparse features
                self._ablate_mlp_component(element.id)
            elif element.type == ElementType.SUBSPACE:
                # Ablate subspace directions
                self._ablate_subspace_element(element, layer_idx)

    def _ablate_token_subspace_circuit(self, circuit: Circuit):
        """
        Ablate a representation circuit (token + subspace combination)

        Args:
            circuit: Representation circuit to ablate
        """
        # Extract circuit information
        token_circuit_id = circuit.metadata.get("token_circuit")
        subspace_circuit_id = circuit.metadata.get("subspace_circuit")

        # Get the original circuits from registry
        token_circuit = self.registry.get_circuit(token_circuit_id) if token_circuit_id else None
        subspace_circuit = self.registry.get_circuit(subspace_circuit_id) if subspace_circuit_id else None

        # Ablate token circuit components (attention patterns)
        if token_circuit:
            self._ablate_token_circuit_elements(token_circuit)

        # Ablate subspace representation components
        if subspace_circuit:
            layer_idx = subspace_circuit.metadata.get("layer", -1)
            if layer_idx >= 0:
                self._ablate_subspace_circuit_elements(subspace_circuit, layer_idx)

        # Ablate embedding/representation connections
        for element in circuit.elements:
            if element.type == ElementType.TOKEN:
                # Ablate token embedding contributions
                self._ablate_token_embedding_contribution(element)
            elif element.type == ElementType.SUBSPACE:
                # Ablate subspace representation
                layer_idx = element.properties.get("layer", -1)
                if layer_idx >= 0:
                    self._ablate_subspace_element(element, layer_idx)

    # Helper methods for specific ablation operations

    def _ablate_token_circuit_elements(self, token_circuit: Circuit):
        """Ablate elements of a token circuit (attention heads)"""
        for element in token_circuit.elements:
            if element.type == ElementType.HEAD:
                self._ablate_attention_head(element.id)

    def _ablate_component_circuit_elements(self, component_circuit: Circuit):
        """Ablate elements of a component circuit (head interactions, MLP components)"""
        for element in component_circuit.elements:
            if element.type == ElementType.HEAD:
                self._ablate_attention_head(element.id)
            elif element.type == ElementType.MLP:
                self._ablate_mlp_component(element.id)

    def _ablate_subspace_circuit_elements(self, subspace_circuit: Circuit, layer_idx: int):
        """Ablate elements of a subspace circuit (feature directions, neurons)"""
        for element in subspace_circuit.elements:
            if element.type == ElementType.MLP:
                # Ablate specific neurons
                neuron_idx = element.properties.get("neuron_idx")
                if neuron_idx is not None:
                    self._ablate_specific_neuron(layer_idx, neuron_idx)
            elif element.type == ElementType.SUBSPACE:
                # Ablate subspace directions
                self._ablate_subspace_element(element, layer_idx)

    def _ablate_attention_head(self, head_id: str):
        """
        Ablate a specific attention head

        Args:
            head_id: Head identifier (e.g., "layer_0_head_1")
        """
        try:
            # Parse head identifier
            parts = head_id.split('_')
            if len(parts) >= 4 and parts[0] == 'layer' and parts[2] == 'head':
                layer_idx = int(parts[1])
                head_idx = int(parts[3])

                if layer_idx < len(self.model.layers):
                    layer = self.model.layers[layer_idx]
                    head_dim = self.model.dim // self.model.num_heads
                    start_idx = head_idx * head_dim
                    end_idx = (head_idx + 1) * head_dim

                    # Zero out the head's output projection
                    with torch.no_grad():
                        layer.attn.out_proj.weight[:, start_idx:end_idx] = 0

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to ablate attention head {head_id}: {e}")

    def _ablate_mlp_component(self, component_id: str):
        """
        Ablate an MLP component

        Args:
            component_id: MLP component identifier
        """
        try:
            # Parse component identifier
            if 'layer_' in component_id:
                # Extract layer index
                layer_idx = int(component_id.split('layer_')[1].split('_')[0])

                if layer_idx < len(self.model.layers):
                    layer = self.model.layers[layer_idx]

                    # Ablate entire MLP for this layer
                    with torch.no_grad():
                        for param in layer.mlp.parameters():
                            param.data *= 0.1  # Reduce but don't completely zero

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to ablate MLP component {component_id}: {e}")

    def _ablate_specific_neuron(self, layer_idx: int, neuron_idx: int):
        """
        Ablate a specific neuron in an MLP layer

        Args:
            layer_idx: Index of the layer
            neuron_idx: Index of the neuron to ablate
        """
        try:
            if layer_idx < len(self.model.layers):
                layer = self.model.layers[layer_idx]

                # Zero out the neuron's weights
                with torch.no_grad():
                    # Up projection (input to hidden)
                    layer.mlp[0].weight[neuron_idx, :] = 0
                    if hasattr(layer.mlp[0], 'bias') and layer.mlp[0].bias is not None:
                        layer.mlp[0].bias[neuron_idx] = 0

                    # Down projection (hidden to output)
                    layer.mlp[2].weight[:, neuron_idx] = 0

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to ablate neuron {neuron_idx} in layer {layer_idx}: {e}")

    def _ablate_subspace_element(self, element: Element, layer_idx: int):
        """
        Ablate a subspace element (feature direction)

        Args:
            element: Subspace element to ablate
            layer_idx: Layer index
        """
        try:
            # Get subspace properties
            direction_type = element.properties.get("direction_type", "unknown")
            neuron_idx = element.properties.get("neuron_idx")

            if neuron_idx is not None:
                # Ablate specific neuron
                self._ablate_specific_neuron(layer_idx, neuron_idx)
            else:
                # General subspace ablation - perturb multiple neurons
                if layer_idx < len(self.model.layers):
                    layer = self.model.layers[layer_idx]

                    # Get importance score to determine ablation strength
                    importance = element.properties.get("importance", 0.5)
                    ablation_strength = min(0.8, importance * 2.0)  # Scale ablation by importance

                    with torch.no_grad():
                        # Randomly ablate a fraction of neurons based on importance
                        hidden_size = layer.mlp[0].weight.shape[0]
                        num_to_ablate = int(hidden_size * ablation_strength * 0.1)  # Up to 8% of neurons

                        if num_to_ablate > 0:
                            # Select neurons to ablate based on weight magnitudes
                            neuron_norms = torch.norm(layer.mlp[0].weight, dim=1)
                            _, top_indices = torch.topk(neuron_norms, num_to_ablate)

                            # Ablate top neurons
                            for idx in top_indices:
                                layer.mlp[0].weight[idx, :] *= 0.2  # Reduce but don't zero
                                layer.mlp[2].weight[:, idx] *= 0.2

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to ablate subspace element {element.id}: {e}")

    def _ablate_token_embedding_contribution(self, token_element: Element):
        """
        Ablate token embedding contributions

        Args:
            token_element: Token element to ablate
        """
        try:
            # Get token position information
            position = token_element.properties.get("position", -1)

            if position >= 0:
                # Ablate position embeddings for this position
                if hasattr(self.model, 'position_embeddings'):
                    with torch.no_grad():
                        self.model.position_embeddings.weight[position, :] *= 0.5

            # For token embeddings, we could ablate specific token representations
            # but this is more complex and depends on the specific tokens involved
            # in the circuit. For now, we focus on positional information.

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to ablate token embedding for {token_element.id}: {e}")

    def _evaluate_model(self, eval_loader) -> float:
        """Quick model evaluation"""

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in eval_loader:
                outputs = self.model(inputs)
                predicted = outputs.argmax(dim=-1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                # Only evaluate a few batches for speed
                if total >= 100:
                    break

        return correct / max(1, total)
