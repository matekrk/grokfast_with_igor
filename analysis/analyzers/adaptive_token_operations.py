# analysis/analyzers/adaptive_token_operations.py
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

from analysis.analyzers.token_operations import TokenOperationDetector
from analysis.core import CanonicalCircuitRegistry, CanonicalRegistryAdapter
from analysis.core.circuit_schema import (Circuit)
from analysis.core.dynamic_thresholds import DynamicThresholdManager


class Obsolete_AdaptiveTokenOperationDetector_obsolete(TokenOperationDetector):
    """Enhanced token operation detector with adaptive thresholds and content-awareness"""

    def __init__(self, model, registry=None, thresholds=None, content_analyzer=None):
        super().__init__(model, registry)

        self.model = model
        self.registry = registry
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


class AdaptiveTokenOperationDetector(TokenOperationDetector):
    """Fixed version with more lenient detection and better debugging"""

    def __init__(self, model, registry=None, thresholds=None, content_analyzer=None,
                 enable_dynamic_thresholds=True, threshold_schedule=None):
        super().__init__(model, registry)

        from analysis.core.circuit_thresholds import CircuitThresholds
        if enable_dynamic_thresholds:
            initial_thresholds = thresholds or CircuitThresholds()
            self.threshold_manager = DynamicThresholdManager(
                initial_thresholds,
                schedule=threshold_schedule)
            self.thresholds = self.threshold_manager.current_thresholds
        else:
            self.thresholds = thresholds or CircuitThresholds(
                copy_attention_min=0.2,  # ✅ FIX: Lower minimum threshold
                copy_attention_max=0.8,
                induction_attention_min=0.3,  # ✅ FIX: Lower minimum threshold
                induction_attention_max=0.7,
                warmup_epochs=50,
                min_accuracy_threshold=0.1  # ✅ FIX: Lower accuracy threshold
            )
            self.threshold_manager = None

        # Content analyzer setup
        if content_analyzer is None:
            try:
                from analysis.analyzers.content_aware_circuit_analyzer import ContentAwareCircuitAnalyzer
                self.content_analyzer = ContentAwareCircuitAnalyzer(model)
            except:
                self.content_analyzer = None
                print("⚠️ ContentAwareCircuitAnalyzer not available, using simplified analysis")
        else:
            self.content_analyzer = content_analyzer

        # Circuit tracking
        self.circuit_history = defaultdict(list)
        self.false_positive_patterns = set()

    def update_thresholds_for_epoch(self, epoch: int, model_accuracy: float = None):
        """Update thresholds for current epoch"""
        if self.threshold_manager:
            old_phase = self.threshold_manager.current_phase
            self.thresholds = self.threshold_manager.update_thresholds_for_epoch(epoch, model_accuracy)

            # Log if phase changed
            if self.threshold_manager.current_phase != old_phase:
                print(f"📊 Threshold update @ epoch {epoch}: {old_phase} → {self.threshold_manager.current_phase}")
                self._log_current_thresholds()


    def manually_switch_threshold_phase(self, phase_name: str):
        """Manually switch to a different threshold phase"""
        if self.threshold_manager:
            self.threshold_manager.manually_set_phase(phase_name)
            self.thresholds = self.threshold_manager.current_thresholds
            self._log_current_thresholds()
        else:
            print("⚠️ Dynamic thresholds not enabled")

    def _log_current_thresholds(self):
        """Log current threshold values for debugging"""
        print(f"📈 Current thresholds:")
        print(f"   Copy: {self.thresholds.copy_attention_min:.3f} - {self.thresholds.copy_attention_max:.3f}")
        print(
            f"   Induction: {self.thresholds.induction_attention_min:.3f} - {self.thresholds.induction_attention_max:.3f}")
        print(f"   Min accuracy: {self.thresholds.min_accuracy_threshold:.3f}")

    def get_adaptive_threshold(self, operation_type, epoch, total_epochs, model_accuracy):
        """Get adaptive threshold (existing method - now uses updated thresholds)"""
        if self.thresholds:
            return self.thresholds.get_threshold(operation_type, epoch, total_epochs, model_accuracy)
        else:
            # Fallback (existing code)
            static_thresholds = {"copy": 0.3, "induction": 0.4, "component": 0.3}
            return static_thresholds.get(operation_type, 0.4)

    def detect_copy_mechanisms_adaptive(self, attention_patterns, tokens=None, epoch=0,
                                        total_epochs=1000, model_accuracy=0.0, content_aware=True):
        """Fixed copy detection with more lenient criteria and debugging"""

        # print(f"🔍 Starting copy detection @ epoch {epoch}")

        # Get threshold
        threshold = self.get_adaptive_threshold("copy", epoch, total_epochs, model_accuracy)

        copy_mechanisms = []
        patterns_checked = 0
        attention_scores_found = []

        for head_name, pattern in attention_patterns.items():
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.detach().cpu().numpy()

            # print(f"  Checking {head_name}: shape {pattern.shape}")

            for query_pos in range(pattern.shape[0]):
                for key_pos in range(query_pos):  # Causal attention only
                    patterns_checked += 1
                    attention_strength = pattern[query_pos, key_pos]
                    attention_scores_found.append(attention_strength)

                    # ✅ FIX: More lenient initial check
                    if attention_strength > threshold * 0.7:  # Allow 30% below threshold initially
                        copy_candidate = {
                            "head": head_name,
                            "source_pos": key_pos,
                            "target_pos": query_pos,
                            "attention_strength": float(attention_strength),
                            "type": "copy",
                            "epoch_detected": epoch,
                            "detection_threshold": threshold
                        }

                        # Content analysis if available and requested
                        if content_aware and self.content_analyzer and tokens:
                            try:
                                content_analysis = self._enhanced_content_analysis(
                                    tokens, key_pos, query_pos, attention_strength
                                )
                                copy_candidate.update(content_analysis)

                                # ✅ FIX: Boost strength based on content
                                content_boost = content_analysis.get("content_strength", 1.0)
                                boosted_strength = attention_strength * content_boost
                                copy_candidate["boosted_strength"] = boosted_strength

                                # Use boosted strength for threshold check
                                effective_strength = boosted_strength
                            except Exception as e:
                                print(f"    ⚠️ Content analysis failed: {e}")
                                effective_strength = attention_strength
                        else:
                            effective_strength = attention_strength

                        # ✅ FIX: More lenient final threshold check
                        if effective_strength > threshold * 0.8:  # 20% below threshold allowed
                            # Calculate reliability
                            reliability = self._calculate_copy_reliability_lenient(
                                copy_candidate, epoch, effective_strength, threshold
                            )
                            copy_candidate["reliability"] = reliability

                            # ✅ FIX: Lower reliability threshold
                            min_reliability = 0.1 if epoch < 100 else 0.05
                            if reliability > min_reliability:
                                copy_mechanisms.append(copy_candidate)
                                # print(f"    ✅ Found copy: {head_name} {key_pos}→{query_pos} strength={effective_strength:.3f}")

        # Debug statistics
        if attention_scores_found:
            max_score = max(attention_scores_found)
            mean_score = sum(attention_scores_found) / len(attention_scores_found)
            scores_above_threshold = sum(1 for s in attention_scores_found if s > threshold)

            # print(f"  📊 Checked {patterns_checked} position pairs")
            # print(f"  📊 Attention scores: max={max_score:.4f}, mean={mean_score:.4f}")
            # print(f"  📊 {scores_above_threshold} scores above threshold {threshold:.3f}")
            # print(f"  📊 Found {len(copy_mechanisms)} copy mechanisms | above threshold {threshold:.3f}")

        return copy_mechanisms

    def _enhanced_content_analysis(self, tokens, source_pos, target_pos, attention_strength):
        """Enhanced content analysis with fallback"""
        if self.content_analyzer:
            try:
                # Use the sophisticated content analyzer
                source_context = self.content_analyzer._analyze_token_context(tokens, source_pos)
                target_context = self.content_analyzer._analyze_token_context(tokens, target_pos)

                target_context_tokens = tokens[max(0, target_pos - 2):target_pos + 3]
                copy_analysis = self.content_analyzer._analyze_generic_copy(
                    tokens[source_pos], target_context_tokens, attention_strength
                )

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
                    "copy_type": copy_analysis.get("copy_type", "positional_only"),
                    "content_strength": copy_analysis.get("confidence", 1.0),
                    "source_token": tokens[source_pos] if source_pos < len(tokens) else "OOB",
                    "target_context_tokens": target_context_tokens
                }
            except Exception as e:
                print(f"    ⚠️ Sophisticated content analysis failed: {e}")

        # ✅ FIX: Fallback to simple content analysis
        return self._simple_content_analysis(tokens, source_pos, target_pos, attention_strength)

    def _simple_content_analysis(self, tokens, source_pos, target_pos, attention_strength):
        """Simple fallback content analysis"""
        if source_pos >= len(tokens) or target_pos >= len(tokens):
            return {
                "copy_type": "positional_only",
                "content_strength": 0.8,  # Default boost
                "source_token": "OOB",
                "semantic_strength": 0.5
            }

        source_token = tokens[source_pos]

        # Simple pattern detection
        copy_type = "positional_only"
        content_strength = 0.8  # Default

        # Check for exact repetition
        if target_pos + 1 < len(tokens) and source_token == tokens[target_pos + 1]:
            copy_type = "exact_token_copy"
            content_strength = 1.3  # Strong boost

        # Check for similar tokens (same value in modular arithmetic)
        elif source_token == tokens[target_pos]:
            copy_type = "content_similar"
            content_strength = 1.1  # Moderate boost

        return {
            "copy_type": copy_type,
            "content_strength": content_strength,
            "source_token": source_token,
            "semantic_strength": content_strength * 0.7
        }

    def _calculate_copy_reliability_lenient(self, copy_candidate, epoch, attention_strength, threshold):
        """More lenient reliability calculation"""
        # Base reliability from attention strength
        attention_margin = (attention_strength - threshold * 0.5) / (1.0 - threshold * 0.5)  # ✅ Lower base threshold
        base_reliability = min(1.0, max(0.1, attention_margin))  # Ensure minimum 0.1

        # Training stage factor - more lenient early on
        if epoch < 50:
            stage_factor = 0.8  # ✅ Higher than original 0.5
        elif epoch < 200:
            stage_factor = 0.9  # ✅ Higher than original 0.7
        else:
            stage_factor = 1.0

        # Historical consistency
        pattern = f"{copy_candidate['head']}_{copy_candidate['source_pos']}_{copy_candidate['target_pos']}"

        if pattern in self.circuit_history:
            epochs_seen = len(self.circuit_history[pattern])
            consistency_factor = min(1.0, 0.3 + epochs_seen / 10.0)  # ✅ Higher base consistency
        else:
            consistency_factor = 0.7  # ✅ Higher base for new patterns
            self.circuit_history[pattern] = [epoch]

        # Content boost
        content_factor = copy_candidate.get("content_strength", 1.0)

        reliability = base_reliability * stage_factor * consistency_factor * content_factor
        return min(1.0, reliability)

    def prune_unstable_circuits(self, current_circuits, epoch, analyze_interval=2):
        """
        Fixed stability calculation that accounts for analysis intervals

        Args:
            current_circuits: List of circuits detected this epoch
            epoch: Current epoch
            analyze_interval: How often analysis is performed (default: 2)

        Returns:
            List of stable circuits
        """
        stable_circuits = []

        # ✅ FIX: Calculate realistic analysis windows
        recent_epoch_window = 30  # Look back 30 epochs
        analysis_opportunities_in_window = recent_epoch_window // analyze_interval  # ~15 with interval=2

        print(f"🔍 Pruning circuits @ epoch {epoch} (interval={analyze_interval})")
        print(
            f"   📊 Recent window: {recent_epoch_window} epochs = ~{analysis_opportunities_in_window} analysis opportunities")

        for circuit in current_circuits:
            circuit_pattern = f"{circuit['head']}_{circuit.get('source_pos', 0)}_{circuit.get('target_pos', 0)}"

            # Update history
            if circuit_pattern not in self.circuit_history:
                self.circuit_history[circuit_pattern] = []

            if epoch not in self.circuit_history[circuit_pattern]:
                self.circuit_history[circuit_pattern].append(epoch)

            # ✅ FIX: Corrected stability analysis
            history = self.circuit_history[circuit_pattern]

            # Recent sightings in the window
            recent_sightings = sum(1 for e in history if epoch - e <= recent_epoch_window)

            # ✅ FIX: Calculate expected vs actual analysis opportunities in recent window
            first_analysis_in_window = max(0, epoch - recent_epoch_window)
            last_analysis_in_window = epoch

            # Count actual analysis epochs in the recent window
            actual_analysis_epochs_in_window = []
            for analysis_epoch in range(first_analysis_in_window, last_analysis_in_window + 1, analyze_interval):
                if analysis_epoch <= epoch:
                    actual_analysis_epochs_in_window.append(analysis_epoch)

            expected_analyses_in_window = len(actual_analysis_epochs_in_window)

            # ✅ FIX: Calculate realistic stability metrics
            if expected_analyses_in_window > 0:
                recent_consistency = recent_sightings / expected_analyses_in_window
            else:
                recent_consistency = 0.0

            # ✅ FIX: Alternative long-term stability (last 20 analysis opportunities)
            last_n_analyses = 20
            recent_analysis_epochs = [e for e in history if e >= epoch - (last_n_analyses * analyze_interval)]
            long_term_consistency = len(recent_analysis_epochs) / min(last_n_analyses, len(history))

            # ✅ FIX: More reasonable stability criteria
            total_sightings = len(history)

            # Multiple stability criteria (any one can qualify)
            is_stable = False
            stability_reason = ""

            # Criterion 1: High recent consistency (appears in most recent analyses)
            if recent_consistency >= 0.4:  # 40% of recent analyses
                is_stable = True
                stability_reason = f"recent_consistency_{recent_consistency:.2f}"

            # Criterion 2: Long-term consistency (reliable over time)
            elif long_term_consistency >= 0.3 and total_sightings >= 5:
                is_stable = True
                stability_reason = f"long_term_consistency_{long_term_consistency:.2f}"

            # Criterion 3: Frequent recent detection (absolute count)
            elif recent_sightings >= 3 and total_sightings >= 5:
                is_stable = True
                stability_reason = f"frequent_recent_{recent_sightings}_of_{expected_analyses_in_window}"

            # Criterion 4: Very new but promising pattern
            elif total_sightings <= 3 and recent_consistency >= 0.6:
                is_stable = True
                stability_reason = f"new_promising_{recent_consistency:.2f}"

            # Debug logging
            if is_stable:
                circuit["stability_score"] = max(recent_consistency, long_term_consistency)
                circuit["stability_reason"] = stability_reason
                stable_circuits.append(circuit)

                print(f"   ✅ STABLE: {circuit_pattern[:20]}... | {stability_reason} | "
                      f"recent={recent_sightings}/{expected_analyses_in_window} | "
                      f"total={total_sightings}")
            else:
                print(f"   ❌ PRUNED: {circuit_pattern[:20]}... | "
                      f"recent={recent_sightings}/{expected_analyses_in_window}={recent_consistency:.2f} | "
                      f"long_term={long_term_consistency:.2f} | total={total_sightings}")

        print(f"   📈 Result: {len(stable_circuits)}/{len(current_circuits)} circuits kept as stable")
        return stable_circuits

    def detect_induction_patterns_adaptive(self, attention_patterns, tokens=None, epoch=0,
                                           total_epochs=1000, model_accuracy=0.0):
        """
        Enhanced induction pattern detection with adaptive thresholds

        Args:
            attention_patterns: Dictionary mapping head names to attention patterns
            tokens: List of token strings for analysis
            epoch: Current training epoch
            total_epochs: Total training epochs
            model_accuracy: Current model accuracy

        Returns:
            List of induction patterns with enhanced metadata
        """

        # print(f"🔄 Starting induction detection @ epoch {epoch}")

        # Get adaptive threshold
        threshold = self.get_adaptive_threshold("induction", epoch, total_epochs, model_accuracy)

        induction_patterns = []
        patterns_checked = 0
        attention_scores_found = []

        for head_name, pattern in attention_patterns.items():
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.detach().cpu().numpy()

            seq_len = pattern.shape[0]
            if seq_len < 4:  # Need at least [A, B, ..., A] for induction
                continue

            # print(f"  Checking {head_name} for induction: shape {pattern.shape}")

            # Look for induction patterns: A B ... A → attend to first A to predict B
            for query_pos in range(2, seq_len):  # Start from 3rd position
                patterns_checked += 1

                # Find position with strongest attention
                max_attended_pos = np.argmax(pattern[query_pos, :query_pos])
                max_attention = pattern[query_pos, max_attended_pos]
                attention_scores_found.append(max_attention)

                # ✅ FIX: More lenient threshold check
                if max_attention > threshold * 0.7:  # 30% below threshold initially
                    # Check for valid induction pattern structure
                    next_after_attended = max_attended_pos + 1
                    if next_after_attended < query_pos:

                        induction_candidate = {
                            "head": head_name,
                            "inducer_pos": max_attended_pos,
                            "induced_pos": next_after_attended,
                            "target_pos": query_pos,
                            "strength": float(max_attention),
                            "type": "induction",
                            "epoch_detected": epoch,
                            "detection_threshold": threshold
                        }

                        # Enhanced pattern analysis
                        if tokens:
                            pattern_analysis = self._analyze_induction_pattern(
                                tokens, max_attended_pos, next_after_attended, query_pos, max_attention
                            )
                            induction_candidate.update(pattern_analysis)

                            # Use boosted strength for final check
                            effective_strength = max_attention * pattern_analysis.get("pattern_strength_boost", 1.0)
                            induction_candidate["effective_strength"] = effective_strength
                        else:
                            effective_strength = max_attention

                        # ✅ FIX: More lenient final threshold
                        if effective_strength > threshold * 0.6:  # 40% below threshold allowed
                            # Calculate reliability
                            reliability = self._calculate_induction_reliability_lenient(
                                induction_candidate, epoch, effective_strength, threshold
                            )
                            induction_candidate["reliability"] = reliability

                            # ✅ FIX: Lower reliability threshold for induction (harder to detect)
                            min_reliability = 0.05 if epoch < 100 else 0.02
                            if reliability > min_reliability:
                                induction_patterns.append(induction_candidate)
                                # print(f"    ✅ Found induction: {head_name} {max_attended_pos}→{next_after_attended}→{query_pos} strength={effective_strength:.3f}")

        # Debug statistics
        if attention_scores_found:
            max_score = max(attention_scores_found)
            mean_score = sum(attention_scores_found) / len(attention_scores_found)
            scores_above_threshold = sum(1 for s in attention_scores_found if s > threshold)

            # print(f"  📊 Checked {patterns_checked} induction positions")
            # print(f"  📊 Max attention: {max_score:.4f}, mean: {mean_score:.4f}")
            # print(f"  📊 {scores_above_threshold} scores above threshold {threshold:.3f}")
            # print(f"  📊 Found {len(induction_patterns)} induction patterns | above threshold {threshold:.3f}")

        return induction_patterns

    def _analyze_induction_pattern(self, tokens, inducer_pos, induced_pos, target_pos, attention_strength):
        """Analyze the semantic content of an induction pattern"""

        # Ensure positions are valid
        if (inducer_pos >= len(tokens) or induced_pos >= len(tokens) or
                target_pos >= len(tokens) or inducer_pos < 0):
            return {
                "pattern_type": "positional_only",
                "pattern_strength_boost": 0.8,
                "semantic_quality": 0.5
            }

        inducer_token = tokens[inducer_pos]
        induced_token = tokens[induced_pos]

        # Default values
        pattern_type = "positional_only"
        strength_boost = 0.8
        semantic_quality = 0.5

        # Check for exact A-B-A pattern
        if target_pos < len(tokens):
            target_context = tokens[max(0, target_pos - 1):target_pos + 2]

            # Look for the "B" token in target context
            if induced_token in target_context:
                pattern_type = "exact_induction"
                strength_boost = 1.4  # Strong boost for exact pattern
                semantic_quality = 0.9

            # Check for A-B-...-A-? pattern where ? should be B
            elif inducer_token == tokens[target_pos - 1] if target_pos > 0 else False:
                pattern_type = "adjacent_induction"
                strength_boost = 1.2  # Good boost for adjacent pattern
                semantic_quality = 0.8

        # Check for numerical patterns (in modular arithmetic)
        try:
            inducer_val = int(inducer_token)
            induced_val = int(induced_token)

            # Check for arithmetic relationships
            if abs(induced_val - inducer_val) == 1:  # Sequential numbers
                pattern_type = "sequential_numeric"
                strength_boost = max(strength_boost, 1.1)
                semantic_quality = max(semantic_quality, 0.7)

        except (ValueError, TypeError):
            pass  # Not numeric tokens

        return {
            "pattern_type": pattern_type,
            "pattern_strength_boost": strength_boost,
            "semantic_quality": semantic_quality,
            "inducer_token": inducer_token,
            "induced_token": induced_token,
            "distance": target_pos - inducer_pos
        }

    def _calculate_induction_reliability_lenient(self, induction_candidate, epoch, attention_strength, threshold):
        """More lenient reliability calculation for induction patterns"""

        # Base reliability from attention strength (more lenient for induction)
        attention_margin = (attention_strength - threshold * 0.4) / (1.0 - threshold * 0.4)
        base_reliability = min(1.0, max(0.05, attention_margin))  # Very lenient base

        # Training stage factor - induction emerges later
        if epoch < 100:
            stage_factor = 0.6  # Lower early factor since induction emerges later
        elif epoch < 300:
            stage_factor = 0.9
        else:
            stage_factor = 1.0

        # Historical consistency
        pattern = f"{induction_candidate['head']}_induction_{induction_candidate['distance']}"

        if pattern in self.circuit_history:
            epochs_seen = len(self.circuit_history[pattern])
            consistency_factor = min(1.0, 0.4 + epochs_seen / 8.0)  # Higher base for induction
        else:
            consistency_factor = 0.8  # High base for new induction patterns
            self.circuit_history[pattern] = [epoch]

        # Pattern quality boost
        semantic_quality = induction_candidate.get("semantic_quality", 0.5)
        pattern_boost = induction_candidate.get("pattern_strength_boost", 1.0)
        quality_factor = (semantic_quality + pattern_boost) / 2.0

        reliability = base_reliability * stage_factor * consistency_factor * quality_factor
        return min(1.0, reliability)

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


class RegistrationAwareAdaptiveTokenOperationDetector(AdaptiveTokenOperationDetector):
    """Enhanced detector that properly creates and registers circuits"""

    def __init__(self, model, registry=None, thresholds=None, content_analyzer=None):
        super().__init__(model, registry, thresholds, content_analyzer)

        # Import circuit creation utilities
        # from analysis.analyzers.token_operations import TokenOperationDetector
        # self.circuit_creator = TokenOperationDetector(model, registry)

        # from analysis.analyzers.fixed_adaptive_token_operations import ModernCircuitCreator
        self.circuit_creator = ModernCircuitCreator(model, registry)

    def detect_and_register_copy_mechanisms(self, attention_patterns, tokens=None, epoch=0,
                                            total_epochs=1000, model_accuracy=0.0,
                                            content_aware=True, register_circuits=False):
        """Enhanced copy detection with modern circuit creation"""

        # Get raw copy mechanisms (existing logic)
        copy_mechanisms = self.detect_copy_mechanisms_adaptive(
            attention_patterns, tokens, epoch, total_epochs, model_accuracy, content_aware
        )

        created_circuits = []

        if register_circuits and copy_mechanisms and self.registry:
            print(f"🔧 Creating {len(copy_mechanisms)} copy circuits with modern creator...")

            for mechanism in copy_mechanisms:
                try:
                    # ✅ FIX: Use modern circuit creator
                    circuit = self.circuit_creator.create_circuit_from_adaptive_detection(
                        mechanism, tokens or [], epoch
                    )

                    # Register with enhanced metadata
                    self.registry.register_circuit_enhanced(
                        circuit=circuit,
                        source="adaptive_token_detection",
                        epoch=epoch,
                        detection_method="adaptive_copy",
                        confidence=mechanism.get("reliability", 0.5),
                        total_epochs=total_epochs
                    )

                    created_circuits.append(circuit)

                except Exception as e:
                    print(f"    ⚠️ Failed to create copy circuit: {e}")
                    import traceback
                    traceback.print_exc()

        return {
            "raw_mechanisms": copy_mechanisms,
            "created_circuits": created_circuits,
            "registration_summary": {
                "attempted": len(copy_mechanisms),
                "succeeded": len(created_circuits),
                "failed": len(copy_mechanisms) - len(created_circuits)
            }
        }

    def detect_and_register_induction_patterns(self, attention_patterns, tokens=None, epoch=0,
                                               total_epochs=1000, model_accuracy=0.0,
                                               register_circuits=False):
        """Enhanced induction detection with modern circuit creation"""

        # Get raw induction patterns
        induction_patterns = self.detect_induction_patterns_adaptive(
            attention_patterns, tokens, epoch, total_epochs, model_accuracy
        )

        created_circuits = []

        if register_circuits and induction_patterns and self.registry:
            print(f"🔄 Creating {len(induction_patterns)} induction circuits with modern creator...")

            for pattern in induction_patterns:
                try:
                    # ✅ FIX: Use modern circuit creator
                    circuit = self.circuit_creator.create_circuit_from_adaptive_detection(
                        pattern, tokens or [], epoch
                    )

                    # Register with enhanced metadata
                    self.registry.register_circuit_enhanced(
                        circuit=circuit,
                        source="adaptive_token_detection",
                        epoch=epoch,
                        detection_method="adaptive_induction",
                        confidence=pattern.get("reliability", 0.5),
                        total_epochs=total_epochs
                    )

                    created_circuits.append(circuit)

                except Exception as e:
                    print(f"    ⚠️ Failed to create induction circuit: {e}")
                    import traceback
                    traceback.print_exc()

        return {
            "raw_patterns": induction_patterns,
            "created_circuits": created_circuits,
            "registration_summary": {
                "attempted": len(induction_patterns),
                "succeeded": len(created_circuits),
                "failed": len(induction_patterns) - len(created_circuits)
            }
        }


class ModernCircuitCreator:
    """
    Modern circuit creator compatible with:
    - FixedAdaptiveTokenOperationDetector output format
    - EnhancedCircuitRegistry interface
    - Current circuit schema
    """

    def __init__(self, model, registry):
        self.model = model
        self.registry = registry

    def create_circuit_from_adaptive_detection(self, detection_data, tokens, epoch):
        """
        Create circuit from adaptive detector output

        Args:
            detection_data: Output from detect_copy_mechanisms_adaptive or detect_induction_patterns_adaptive
            tokens: List of token strings
            epoch: Current epoch

        Returns:
            Circuit: Formal circuit object
        """
        detection_type = detection_data.get("type", "unknown")

        if detection_type == "copy":
            return self._create_copy_circuit_modern(detection_data, tokens, epoch)
        elif detection_type == "induction":
            return self._create_induction_circuit_modern(detection_data, tokens, epoch)
        else:
            raise ValueError(f"Unknown detection type: {detection_type}")

    def _create_copy_circuit_modern(self, copy_data, tokens, epoch):
        """Create copy circuit compatible with current systems"""

        from analysis.core.circuit_schema import Circuit, CircuitType, Element, ElementType, Connection, ConnectionType

        # ✅ FIX: Handle both field name variants
        strength = copy_data.get("strength", copy_data.get("attention_strength", 0.5))

        head = copy_data.get("head", "unknown")
        source_pos = copy_data.get("source_pos", -1)
        target_pos = copy_data.get("target_pos", -1)
        relative_offset = target_pos - source_pos if source_pos >= 0 and target_pos >= 0 else 0

        # ✅ FIX: Use correct registry method name
        circuit_id = self.registry.generate_circuit_id(
            operation_type="copy",
            component_info=head,
            epoch=epoch,
            source_pos=source_pos,
            target_pos=target_pos,
            relative_offset=relative_offset,
            source="adaptive_individual",
            consistency=1
        )

        # Create elements
        source_token = Element(
            id=f"source_token_{source_pos}",
            type=ElementType.TOKEN,
            properties={
                "position": source_pos,
                "token": tokens[source_pos] if 0 <= source_pos < len(tokens) else "OOB",
                "role": "source"
            }
        )

        target_token = Element(
            id=f"target_token_{target_pos}",
            type=ElementType.TOKEN,
            properties={
                "position": target_pos,
                "token": tokens[target_pos] if 0 <= target_pos < len(tokens) else "OOB",
                "role": "target"
            }
        )

        attention_head = Element(
            id=head,
            type=ElementType.HEAD,
            properties={"name": head, "operation": "copy"}
        )

        # Create connections
        source_to_head = Connection(
            source=source_token.id,
            target=attention_head.id,
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "read", "position_type": "source"}
        )

        head_to_target = Connection(
            source=attention_head.id,
            target=target_token.id,
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "write", "position_type": "target"}
        )

        # Enhanced metadata from adaptive detection
        metadata = {
            "operation_type": "copy",
            "head": head,
            "source_position": source_pos,
            "target_position": target_pos,
            "relative_offset": relative_offset,
            # ✅ ADD: Modern adaptive detection metadata
            "detection_method": copy_data.get("detection_method", "adaptive_copy"),
            "reliability": copy_data.get("reliability", 0.5),
            "content_aware": copy_data.get("content_aware", False),
            "copy_type": copy_data.get("copy_type", "positional_only"),
            "semantic_strength": copy_data.get("semantic_strength", 0.5),
            "detection_threshold": copy_data.get("detection_threshold", 0.5),
            "boosted_strength": copy_data.get("boosted_strength", strength)
        }

        # Create circuit
        circuit = Circuit(
            id=circuit_id,
            type=CircuitType.TOKEN,
            elements=[source_token, target_token, attention_head],
            connections=[source_to_head, head_to_target],
            attribution=strength,
            metadata=metadata,
            discovered_at=epoch
        )

        return circuit

    def _create_induction_circuit_modern(self, induction_data, tokens, epoch):
        """Create induction circuit compatible with current systems"""

        from analysis.core.circuit_schema import Circuit, CircuitType, Element, ElementType, Connection, ConnectionType

        # ✅ FIX: Handle both field name variants
        strength = induction_data.get("strength", induction_data.get("attention_strength", 0.5))

        head = induction_data.get("head", "unknown")
        inducer_pos = induction_data.get("inducer_pos", -1)
        induced_pos = induction_data.get("induced_pos", -1)
        target_pos = induction_data.get("target_pos", -1)

        # Calculate pattern metrics
        pattern_distance = target_pos - inducer_pos if inducer_pos >= 0 and target_pos >= 0 else 0
        induction_span = induced_pos - inducer_pos if inducer_pos >= 0 and induced_pos >= 0 else 0
        pattern_type = f"dist_{pattern_distance}_span_{induction_span}"

        # ✅ FIX: Use correct registry method name
        circuit_id = self.registry.generate_circuit_id(
            operation_type="induction",
            component_info=head,
            epoch=epoch,
            pattern_type=pattern_type,
            pattern_distance=pattern_distance,
            induction_span=induction_span,
            inducer_pos=inducer_pos,
            induced_pos=induced_pos,
            target_pos=target_pos,
            source="adaptive_individual",
            consistency=1
        )

        # Create elements
        inducer_token = Element(
            id=f"inducer_token_{inducer_pos}",
            type=ElementType.TOKEN,
            properties={
                "position": inducer_pos,
                "token": tokens[inducer_pos] if 0 <= inducer_pos < len(tokens) else "OOB",
                "role": "inducer"
            }
        )

        induced_token = Element(
            id=f"induced_token_{induced_pos}",
            type=ElementType.TOKEN,
            properties={
                "position": induced_pos,
                "token": tokens[induced_pos] if 0 <= induced_pos < len(tokens) else "OOB",
                "role": "induced"
            }
        )

        target_token = Element(
            id=f"target_token_{target_pos}",
            type=ElementType.TOKEN,
            properties={
                "position": target_pos,
                "token": tokens[target_pos] if 0 <= target_pos < len(tokens) else "OOB",
                "role": "target"
            }
        )

        attention_head = Element(
            id=head,
            type=ElementType.HEAD,
            properties={"name": head, "operation": "induction"}
        )

        # Create connections
        inducer_to_head = Connection(
            source=inducer_token.id,
            target=attention_head.id,
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "read", "induction_role": "inducer"}
        )

        head_to_target = Connection(
            source=attention_head.id,
            target=target_token.id,
            strength=strength,
            type=ConnectionType.ATTENTION,
            properties={"operation": "write", "induction_role": "target"}
        )

        # Semantic induction connection
        induced_relation = Connection(
            source=induced_token.id,
            target=target_token.id,
            strength=strength * 0.8,
            type=ConnectionType.COMPOSITE,
            properties={"operation": "predict", "induction_role": "pattern"}
        )

        # Enhanced metadata from adaptive detection
        metadata = {
            "operation_type": "induction",
            "head": head,
            "inducer_position": inducer_pos,
            "induced_position": induced_pos,
            "target_position": target_pos,
            "pattern_distance": pattern_distance,
            "induction_span": induction_span,
            "pattern_type": pattern_type,
            # ✅ ADD: Modern adaptive detection metadata
            "detection_method": induction_data.get("detection_method", "adaptive_induction"),
            "reliability": induction_data.get("reliability", 0.5),
            "pattern_type_semantic": induction_data.get("pattern_type", "positional_only"),
            "semantic_quality": induction_data.get("semantic_quality", 0.5),
            "effective_strength": induction_data.get("effective_strength", strength),
            "detection_threshold": induction_data.get("detection_threshold", 0.5)
        }

        # Create circuit
        circuit = Circuit(
            id=circuit_id,
            type=CircuitType.TOKEN,
            elements=[inducer_token, induced_token, target_token, attention_head],
            connections=[inducer_to_head, head_to_target, induced_relation],
            attribution=strength,
            metadata=metadata,
            discovered_at=epoch
        )

        return circuit


class CanonicalAwareAdaptiveTokenOperationDetector(AdaptiveTokenOperationDetector):
    """
    Updated adaptive detector that uses canonical circuit system
    Extends RegistrationAwareAdaptiveTokenOperationDetector
    """

    def __init__(self, model, enhanced_registry, canonical_registry=None, thresholds=None,
                 content_analyzer=None, enable_dynamic_thresholds=True,
                 threshold_schedule=None):
        super().__init__(model=model, registry=enhanced_registry, thresholds=thresholds,
                         content_analyzer=content_analyzer,
                         enable_dynamic_thresholds=enable_dynamic_thresholds,
                         threshold_schedule=threshold_schedule)

        # info store both registries for canonical functionality
        self.enhanced_registry = enhanced_registry
        self.canonical_registry = canonical_registry or CanonicalCircuitRegistry()

        # info create adapter between systems for seamless integration
        self.canonical_adapter = CanonicalRegistryAdapter(self.enhanced_registry,
                                                          self.canonical_registry)

        # info circuit creator uses inherited registry, which is an enhanced_registry
        self.circuit_creator = ModernCircuitCreator(model, enhanced_registry)

        # Circuit tracking (now tracks canonical IDs)
        self.circuit_history = {}  # canonical_id -> detection_epochs
        self.false_positive_patterns = set()

    def detect_and_register_copy_mechanisms(self, attention_patterns, tokens=None, epoch=0,
                                            total_epochs=1000, model_accuracy=0.0,
                                            content_aware=True, register_circuits=True):
        """Enhanced copy detection with canonical registration"""

        # Existing detection logic (unchanged)
        copy_mechanisms = self._detect_copy_mechanisms_fixed(
            attention_patterns, tokens, epoch, total_epochs, model_accuracy, content_aware
        )

        canonical_circuits = []
        registration_summary = {'attempted': 0, 'succeeded': 0, 'aggregated': 0}

        if register_circuits and copy_mechanisms:
            print(f"🔧 Registering {len(copy_mechanisms)} copy circuits with canonical system...")

            for mechanism in copy_mechanisms:
                try:
                    registration_summary['attempted'] += 1

                    # Create circuit using existing logic
                    circuit = self.circuit_creator.create_circuit_from_adaptive_detection(
                        mechanism, tokens or [], epoch
                    )

                    # ✅ NEW: Register using canonical system
                    canonical_id, legacy_id = self.canonical_adapter.register_circuit_detection(
                        circuit=circuit,
                        epoch=epoch,
                        tokens=tokens or [],
                        detection_confidence=mechanism.get("reliability", 0.5),
                        detection_method="adaptive_copy",
                        example_metadata={
                            'attention_strength': mechanism.get('attention_strength', 0.0),
                            'copy_type': mechanism.get('copy_type', 'unknown'),
                            'source_pos': mechanism.get('source_pos', -1),
                            'target_pos': mechanism.get('target_pos', -1)
                        }
                    )

                    # Track canonical circuit
                    canonical_circuits.append(canonical_id)

                    # Update history tracking (now uses canonical IDs)
                    if canonical_id not in self.circuit_history:
                        self.circuit_history[canonical_id] = []
                        registration_summary['succeeded'] += 1
                    else:
                        registration_summary['aggregated'] += 1

                    self.circuit_history[canonical_id].append(epoch)

                except Exception as e:
                    print(f"    ⚠️ Failed to register copy circuit: {e}")

        return {
            "raw_mechanisms": copy_mechanisms,
            "canonical_circuits": canonical_circuits,
            "registration_summary": registration_summary,
            "canonical_registry_summary": self.canonical_registry.get_registry_summary()
        }

    def detect_and_register_induction_patterns(self, attention_patterns, tokens=None, epoch=0,
                                               total_epochs=1000, model_accuracy=0.0,
                                               register_circuits=True):
        """Enhanced induction detection with canonical registration"""

        # Similar pattern to copy detection
        induction_patterns = self._detect_induction_patterns_fixed(
            attention_patterns, tokens, epoch, total_epochs, model_accuracy
        )

        canonical_circuits = []
        registration_summary = {'attempted': 0, 'succeeded': 0, 'aggregated': 0}

        if register_circuits and induction_patterns:
            print(f"🔄 Registering {len(induction_patterns)} induction circuits with canonical system...")

            for pattern in induction_patterns:
                try:
                    registration_summary['attempted'] += 1

                    circuit = self.circuit_creator.create_circuit_from_adaptive_detection(
                        pattern, tokens or [], epoch
                    )

                    canonical_id, legacy_id = self.canonical_adapter.register_circuit_detection(
                        circuit=circuit,
                        epoch=epoch,
                        tokens=tokens or [],
                        detection_confidence=pattern.get("reliability", 0.5),
                        detection_method="adaptive_induction",
                        example_metadata={
                            'strength': pattern.get('strength', 0.0),
                            'pattern_type': pattern.get('pattern_type', 'unknown'),
                            'inducer_pos': pattern.get('inducer_pos', -1),
                            'target_pos': pattern.get('target_pos', -1),
                            'distance': pattern.get('distance', 0)
                        }
                    )

                    canonical_circuits.append(canonical_id)

                    if canonical_id not in self.circuit_history:
                        self.circuit_history[canonical_id] = []
                        registration_summary['succeeded'] += 1
                    else:
                        registration_summary['aggregated'] += 1

                    self.circuit_history[canonical_id].append(epoch)

                except Exception as e:
                    print(f"    ⚠️ Failed to register induction circuit: {e}")

        return {
            "raw_patterns": induction_patterns,
            "canonical_circuits": canonical_circuits,
            "registration_summary": registration_summary
        }

    def get_canonical_circuit_stability(self, canonical_id: str) -> Dict[str, float]:
        """Get stability metrics for canonical circuit"""
        canonical = self.canonical_registry.get_canonical_circuit(canonical_id)

        if canonical:
            return {
                'stability_score': canonical.stability_score,
                'consistency_score': canonical.consistency_score,
                'persistence_score': canonical.persistence_score,
                'total_detections': canonical.total_detections,
                'temporal_span': canonical.last_seen - canonical.first_seen + 1
            }
        return {'stability_score': 0.0, 'consistency_score': 0.0}

    def prune_unstable_canonical_circuits(self, current_epoch: int, min_stability: float = 0.3):
        """Prune based on canonical stability scores"""
        stable_circuits = self.canonical_registry.get_stable_circuits(
            min_stability=min_stability, min_detections=2
        )

        stable_canonical_ids = set(circuit.canonical_id for circuit in stable_circuits)

        print(f"🔍 Canonical pruning @ epoch {current_epoch}: "
              f"{len(stable_circuits)} stable circuits from {len(self.canonical_registry.canonical_circuits)} total")

        return stable_canonical_ids

    def _detect_copy_mechanisms_fixed(self, attention_patterns, tokens, epoch, total_epochs, model_accuracy,
                                      content_aware):
        """Existing copy detection logic (from FixedAdaptiveTokenOperationDetector)"""
        # Import and use existing logic
        # from analysis.analyzers.fixed_adaptive_token_operations import FixedAdaptiveTokenOperationDetector
        base_detector = AdaptiveTokenOperationDetector(self.model, None, self.thresholds)
        return base_detector.detect_copy_mechanisms_adaptive(
            attention_patterns, tokens, epoch, total_epochs, model_accuracy, content_aware
        )

    def _detect_induction_patterns_fixed(self, attention_patterns, tokens, epoch, total_epochs, model_accuracy):
        """Existing induction detection logic"""
        # from analysis.analyzers.fixed_adaptive_token_operations import FixedAdaptiveTokenOperationDetector
        base_detector = AdaptiveTokenOperationDetector(self.model, None, self.thresholds)
        return base_detector.detect_induction_patterns_adaptive(
            attention_patterns, tokens, epoch, total_epochs, model_accuracy
        )
