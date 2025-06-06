# analysis/analyzers/adaptive_token_operations.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from analysis.core.circuit_schema import Circuit, Element, Connection, ElementType, ConnectionType, CircuitType
from analysis.analyzers.token_operations import TokenOperationDetector


class AdaptiveTokenOperationDetector(TokenOperationDetector):
    """Enhanced token operation detector with adaptive thresholds and content-awareness"""

    def __init__(self, model, registry=None, thresholds=None, content_analyzer=None):
        super().__init__(model, registry)

        self.thresholds = thresholds
        # Content-aware analysis
        if content_analyzer is None:
            from analysis.analyzers.content_aware_circuit_analyzer import ContentAwareCircuitAnalyzer
            self.content_analyzer = ContentAwareCircuitAnalyzer(model)
        else:
            self.content_analyzer = content_analyzer

        # Track circuit history for stability analysis
        self.circuit_history = defaultdict(list)
        self.false_positive_patterns = set()

        # Threshold handling
        if self.thresholds is None:
            # Create default thresholds if none provided
            from analysis.core.circuit_thresholds import CircuitThresholds
            self.thresholds = CircuitThresholds()
            print("⚠️  No thresholds provided, using default CircuitThresholds")
        print("✅ Adaptive token operation detector initialized")

    def get_adaptive_threshold(self, operation_type: str, epoch: int,
                               total_epochs: int, model_accuracy: float) -> float:
        """Get adaptive threshold for specific operation type"""
        if self.thresholds:
            return self.thresholds.get_threshold(operation_type, epoch, total_epochs, model_accuracy)
        else:
            # Fallback to static thresholds
            static_thresholds = {
                "copy": 0.8,
                "induction": 0.7,
                "component": 0.6
            }
            return static_thresholds.get(operation_type, 0.7)


    def detect_copy_mechanisms_adaptive(self,
                                        attention_patterns: Dict[str, torch.Tensor],
                                        tokens: List[str] = None,
                                        epoch: int = 0,
                                        total_epochs: int = 1000,
                                        model_accuracy: float = 0.0,
                                        content_aware: bool = True) -> List[Dict[str, Any]]:
        """Enhanced copy detection with proper threshold and content analysis"""

        # Get adaptive threshold
        threshold = self.get_adaptive_threshold("copy", epoch, total_epochs, model_accuracy)

        copy_mechanisms = []

        for head_name, pattern in attention_patterns.items():
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.detach().cpu().numpy()

            for query_pos in range(pattern.shape[0]):
                for key_pos in range(query_pos):  # Causal attention only
                    attention_strength = pattern[query_pos, key_pos]

                    if attention_strength > threshold:
                        # Basic copy detection
                        copy_candidate = {
                            "head": head_name,
                            "source_pos": key_pos,
                            "target_pos": query_pos,
                            "attention_strength": float(attention_strength),
                            "type": "copy",
                            "epoch_detected": epoch,
                            "detection_threshold": threshold
                        }

                        # Enhanced content-aware analysis
                        if content_aware and tokens and len(tokens) > max(query_pos, key_pos):
                            content_analysis = self._enhanced_content_analysis(
                                tokens, key_pos, query_pos, attention_strength
                            )
                            copy_candidate.update(content_analysis)

                        # Calculate reliability score
                        reliability = self._calculate_copy_reliability(
                            copy_candidate, epoch, attention_strength, threshold
                        )
                        copy_candidate["reliability"] = reliability

                        # Only include if reliability meets minimum standards
                        min_reliability = 0.3 if epoch < 100 else 0.2  # More lenient later in training
                        if reliability > min_reliability:
                            copy_mechanisms.append(copy_candidate)

        return copy_mechanisms

    def _enhanced_content_analysis(self, tokens: List[str], source_pos: int,
                                   target_pos: int, attention_strength: float) -> Dict[str, Any]:
        """Enhanced content analysis using ContentAwareCircuitAnalyzer"""

        # Analyze source context
        source_context = self.content_analyzer._analyze_token_context(tokens, source_pos)
        target_context = self.content_analyzer._analyze_token_context(tokens, target_pos)

        # Analyze copy pattern
        target_context_tokens = tokens[max(0, target_pos - 2):target_pos + 3]
        copy_analysis = self.content_analyzer._analyze_generic_copy(
            tokens[source_pos], target_context_tokens, attention_strength
        )

        # Calculate semantic strength
        semantic_strength = self.content_analyzer.analyze_copy_semantic_strength(
            {
                "source_pos": source_pos,
                "target_pos": target_pos,
                "attention_strength": attention_strength
            },
            tokens
        )

        return {
            "source_context": source_context,
            "target_context": target_context,
            "copy_analysis": copy_analysis,
            "semantic_strength": semantic_strength,
            "copy_type": copy_analysis["copy_type"],
            "content_strength": copy_analysis["confidence"],
            "source_token": tokens[source_pos],
            "target_context_tokens": target_context_tokens
        }

    def detect_induction_patterns_adaptive(self,
                                           attention_patterns: Dict[str, torch.Tensor],
                                           tokens: List[str] = None,
                                           epoch: int = 0,
                                           total_epochs: int = 1000,
                                           model_accuracy: float = 0.0) -> List[Dict[str, Any]]:
        """
        Detect induction patterns with adaptive thresholds
        """
        if self.thresholds is None:
            return self.detect_induction_patterns(attention_patterns, threshold=0.7)

        threshold = self.thresholds.get_threshold("induction", epoch, total_epochs, model_accuracy)

        induction_patterns = []

        for head_name, pattern in attention_patterns.items():
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.detach().cpu().numpy()

            seq_len = pattern.shape[0]
            if seq_len < 4:
                continue

            for query_pos in range(2, seq_len):
                max_attended_pos = np.argmax(pattern[query_pos, :query_pos])

                if pattern[query_pos, max_attended_pos] > threshold:
                    next_after_attended = max_attended_pos + 1
                    if next_after_attended < query_pos:
                        induction_candidate = {
                            "head": head_name,
                            "inducer_pos": max_attended_pos,
                            "induced_pos": next_after_attended,
                            "target_pos": query_pos,
                            "strength": float(pattern[query_pos, max_attended_pos]),
                            "type": "induction",
                            "epoch_detected": epoch,
                            "detection_threshold": threshold
                        }

                        # Calculate reliability
                        reliability = self._calculate_induction_reliability(
                            induction_candidate, epoch, tokens
                        )
                        induction_candidate["reliability"] = reliability

                        if reliability > 0.3:
                            induction_patterns.append(induction_candidate)

        return induction_patterns

    def _analyze_copy_content(self, tokens: List[str], source_pos: int, target_pos: int,
                              attention_strength: float) -> Tuple[str, float]:
        """Analyze what type of copying based on token content"""
        if source_pos >= len(tokens) or target_pos >= len(tokens):
            return "positional_only", attention_strength

        source_token = tokens[source_pos]

        # Check for exact token copying
        if target_pos < len(tokens) - 1:
            next_token = tokens[target_pos + 1]
            if source_token == next_token:
                return "exact_token_copy", attention_strength * 1.2

        # Check for pattern completion (A B ... A -> B)
        if source_pos < len(tokens) - 1:
            source_next = tokens[source_pos + 1]
            if target_pos < len(tokens) - 1:
                target_next = tokens[target_pos + 1]
                if source_next == target_next:
                    return "pattern_completion", attention_strength * 1.1

        # Check for content similarity
        content_similarity = self._calculate_token_similarity(source_token, tokens[target_pos])
        if content_similarity > 0.7:
            return "content_similar", attention_strength * (0.8 + 0.4 * content_similarity)

        return "positional_only", attention_strength * 0.9

    def _calculate_token_similarity(self, token1: str, token2: str) -> float:
        """Calculate similarity between tokens"""
        if token1 == token2:
            return 1.0

        # Check if both are numbers
        try:
            float(token1)
            float(token2)
            return 0.8
        except:
            pass

        # Same length similarity
        if len(token1) == len(token2):
            return 0.5

        return 0.1

    def _calculate_copy_reliability(self, copy_candidate: Dict, epoch: int,
                                    attention_strength: float, threshold: float) -> float:
        """Calculate reliability score for copy mechanism"""
        # Base reliability from attention strength
        attention_margin = (attention_strength - threshold) / (1.0 - threshold)
        base_reliability = min(1.0, attention_margin * 2.0)

        # Training stage factor
        if epoch < 50:
            stage_factor = 0.5
        elif epoch < 200:
            stage_factor = 0.7
        elif epoch < 500:
            stage_factor = 0.9
        else:
            stage_factor = 1.0

        # Historical consistency
        pattern = f"{copy_candidate['head']}_{copy_candidate['source_pos']}_{copy_candidate['target_pos']}"

        if pattern in self.circuit_history:
            epochs_seen = len(self.circuit_history[pattern])
            consistency_factor = min(1.0, epochs_seen / 5.0)
        else:
            consistency_factor = 0.5
            self.circuit_history[pattern] = [epoch]

        # Content boost
        content_factor = 1.0
        if "copy_type" in copy_candidate:
            copy_type = copy_candidate["copy_type"]
            if copy_type == "exact_token_copy":
                content_factor = 1.3
            elif copy_type == "pattern_completion":
                content_factor = 1.2
            elif copy_type == "content_similar":
                content_factor = 1.1

        reliability = base_reliability * stage_factor * consistency_factor * content_factor
        return min(1.0, reliability)

    def _calculate_induction_reliability(self, induction_candidate: Dict, epoch: int,
                                         tokens: List[str] = None) -> float:
        """Calculate reliability for induction patterns"""
        base_strength = induction_candidate["strength"]

        # Training stage factor
        if epoch < 100:
            stage_factor = 0.4  # Induction heads emerge later
        elif epoch < 300:
            stage_factor = 0.7
        else:
            stage_factor = 1.0

        # Pattern consistency
        pattern = f"{induction_candidate['head']}_induction"
        if pattern in self.circuit_history:
            consistency_factor = min(1.0, len(self.circuit_history[pattern]) / 3.0)
        else:
            consistency_factor = 0.5
            self.circuit_history[pattern] = [epoch]

        return base_strength * stage_factor * consistency_factor

    def prune_unstable_circuits(self, current_circuits: List[Dict], epoch: int) -> List[Dict]:
        """Remove circuits that haven't been seen recently"""
        stable_circuits = []

        for circuit in current_circuits:
            circuit_pattern = f"{circuit['head']}_{circuit.get('source_pos', 0)}_{circuit.get('target_pos', 0)}"

            # Update history
            if circuit_pattern not in self.circuit_history:
                self.circuit_history[circuit_pattern] = []

            if epoch not in self.circuit_history[circuit_pattern]:
                self.circuit_history[circuit_pattern].append(epoch)

            # Check stability
            history = self.circuit_history[circuit_pattern]
            recent_sightings = sum(1 for e in history if epoch - e <= 20)
            total_sightings = len(history)

            # Stability criteria
            is_stable = (
                    total_sightings >= 2 and
                    recent_sightings >= 1 and
                    (recent_sightings / total_sightings) > 0.3
            )

            if is_stable:
                circuit["stability_score"] = recent_sightings / max(1, total_sightings)
                stable_circuits.append(circuit)
            else:
                self.false_positive_patterns.add(circuit_pattern)

        return stable_circuits

    def get_emergence_timeline(self) -> Dict[str, Any]:
        """Get timeline of circuit emergence"""
        emergence_data = {}

        for pattern, epochs in self.circuit_history.items():
            if epochs:
                emergence_data[pattern] = {
                    "first_seen": min(epochs),
                    "last_seen": max(epochs),
                    "total_sightings": len(epochs),
                    "epochs": sorted(epochs)
                }

        return emergence_data




# info adaptive token copy mechanism analysis functions

