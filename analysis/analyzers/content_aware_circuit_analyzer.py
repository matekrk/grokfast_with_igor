# analysis/analyzers/content_aware_circuit_analyzer.py
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter


class ContentAwareCircuitAnalyzer:
    """Analyzer for content-aware circuit detection and validation"""

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

        # Token type classifications
        self.number_tokens = set()
        self.operator_tokens = set()
        self.special_tokens = set()

        self._initialize_token_types()

    def _initialize_token_types(self):
        """Initialize token type classifications"""
        # For modular arithmetic tasks
        self.number_tokens = {str(i) for i in range(100)}  # 0-99
        self.operator_tokens = {'+', '-', '*', '=', '%'}
        self.special_tokens = {'[BOS]', '[EOS]', '[PAD]', '<pad>'}

    def _analyze_token_context(self, tokens: List[str], position: int, window_size: int = 3) -> Dict[str, Any]:
        """
        Analyze semantic context around a token position

        Args:
            tokens: List of token strings
            position: Position to analyze
            window_size: Context window size on each side

        Returns:
            Dict with context analysis
        """
        if position >= len(tokens) or position < 0:
            return {"valid": False, "reason": "position_out_of_bounds"}

        # Extract context window
        start_pos = max(0, position - window_size)
        end_pos = min(len(tokens), position + window_size + 1)
        context_tokens = tokens[start_pos:end_pos]

        # Analyze token types in context
        context_analysis = {
            "valid": True,
            "position": position,
            "context_tokens": context_tokens,
            "context_window": (start_pos, end_pos),
            "target_token": tokens[position],
            "token_types": self._classify_context_tokens(context_tokens),
            "semantic_features": self._extract_semantic_features(context_tokens, position - start_pos),
            "pattern_type": self._identify_pattern_type(context_tokens, position - start_pos)
        }

        return context_analysis

    def _classify_context_tokens(self, context_tokens: List[str]) -> Dict[str, List[int]]:
        """Classify tokens in context by type"""
        classification = {
            "numbers": [],
            "operators": [],
            "special": [],
            "unknown": []
        }

        for i, token in enumerate(context_tokens):
            if token in self.number_tokens:
                classification["numbers"].append(i)
            elif token in self.operator_tokens:
                classification["operators"].append(i)
            elif token in self.special_tokens:
                classification["special"].append(i)
            else:
                classification["unknown"].append(i)

        return classification

    def _extract_semantic_features(self, context_tokens: List[str], target_idx: int) -> Dict[str, Any]:
        """Extract semantic features from context"""
        features = {
            "has_arithmetic": any(op in context_tokens for op in ['+', '-', '*']),
            "has_equation": '=' in context_tokens,
            "number_count": sum(1 for token in context_tokens if token in self.number_tokens),
            "operator_count": sum(1 for token in context_tokens if token in self.operator_tokens),
            "target_token_type": self._get_token_type(
                context_tokens[target_idx] if target_idx < len(context_tokens) else ""),
            "sequence_type": self._identify_sequence_type(context_tokens)
        }

        return features

    def _get_token_type(self, token: str) -> str:
        """Get type of a single token"""
        if token in self.number_tokens:
            return "number"
        elif token in self.operator_tokens:
            return "operator"
        elif token in self.special_tokens:
            return "special"
        else:
            return "unknown"

    def _identify_sequence_type(self, tokens: List[str]) -> str:
        """Identify the type of sequence"""
        token_types = [self._get_token_type(token) for token in tokens]

        if "operator" in token_types and "number" in token_types:
            if "=" in tokens:
                return "arithmetic_equation"
            else:
                return "arithmetic_expression"
        elif all(t == "number" for t in token_types):
            return "number_sequence"
        else:
            return "mixed_sequence"

    def _identify_pattern_type(self, context_tokens: List[str], target_idx: int) -> str:
        """Identify the pattern type at target position"""
        if target_idx >= len(context_tokens):
            return "unknown"

        # Look for common patterns
        if target_idx > 0:
            prev_token = context_tokens[target_idx - 1]
            curr_token = context_tokens[target_idx]

            # Pattern: number + operator → likely arithmetic
            if (self._get_token_type(prev_token) == "number" and
                    self._get_token_type(curr_token) == "operator"):
                return "arithmetic_continuation"

            # Pattern: operator + number → likely operand
            if (self._get_token_type(prev_token) == "operator" and
                    self._get_token_type(curr_token) == "number"):
                return "operand_after_operator"

        # Check for equation completion
        if "=" in context_tokens[:target_idx] and target_idx == len(context_tokens) - 1:
            return "equation_result"

        return "generic_continuation"

    def _analyze_generic_copy(self, source_token: str, target_context: List[str],
                              attention_strength: float) -> Dict[str, Any]:
        """
        Analyze generic copying patterns beyond exact matches

        Args:
            source_token: Token being copied from
            target_context: Context around target position
            attention_strength: Attention weight

        Returns:
            Dict with copy analysis
        """
        analysis = {
            "source_token": source_token,
            "target_context": target_context,
            "attention_strength": attention_strength,
            "copy_type": "unknown",
            "confidence": 0.0,
            "semantic_relevance": 0.0
        }

        # Analyze source token
        source_type = self._get_token_type(source_token)

        # Analyze target context
        if not target_context:
            analysis["copy_type"] = "no_context"
            return analysis

        target_types = [self._get_token_type(token) for token in target_context]

        # Type-consistent copying (number → number position)
        if source_type == "number":
            if "number" in target_types:
                analysis["copy_type"] = "number_to_number"
                analysis["confidence"] = 0.8
                analysis["semantic_relevance"] = 0.9
            elif any(op in target_context for op in ['+', '-', '*']):
                analysis["copy_type"] = "number_to_arithmetic"
                analysis["confidence"] = 0.7
                analysis["semantic_relevance"] = 0.8
            else:
                analysis["copy_type"] = "number_to_other"
                analysis["confidence"] = 0.4
                analysis["semantic_relevance"] = 0.3

        # Operator copying
        elif source_type == "operator":
            if source_token in target_context:
                analysis["copy_type"] = "operator_repetition"
                analysis["confidence"] = 0.9
                analysis["semantic_relevance"] = 0.8
            elif "=" in target_context and source_token in ['+', '-', '*']:
                analysis["copy_type"] = "operator_to_equation"
                analysis["confidence"] = 0.6
                analysis["semantic_relevance"] = 0.7
            else:
                analysis["copy_type"] = "operator_to_other"
                analysis["confidence"] = 0.3
                analysis["semantic_relevance"] = 0.2

        # Pattern completion copying
        elif self._is_pattern_completion(source_token, target_context):
            analysis["copy_type"] = "pattern_completion"
            analysis["confidence"] = 0.8
            analysis["semantic_relevance"] = 0.9

        else:
            analysis["copy_type"] = "generic_positional"
            analysis["confidence"] = 0.2
            analysis["semantic_relevance"] = 0.1

        # Boost confidence based on attention strength
        attention_boost = min(0.3, (attention_strength - 0.5) * 0.6)
        analysis["confidence"] = min(1.0, analysis["confidence"] + attention_boost)

        return analysis

    def _is_pattern_completion(self, source_token: str, target_context: List[str]) -> bool:
        """Check if this represents pattern completion"""
        # Simple pattern completion check
        # Could be enhanced with more sophisticated pattern recognition
        return (source_token in target_context or
                any(token == source_token for token in target_context[-3:]))

    def analyze_copy_semantic_strength(self, copy_mechanism: Dict[str, Any],
                                       tokens: List[str]) -> float:
        """Calculate semantic strength of a copy mechanism"""
        source_pos = copy_mechanism.get("source_pos", -1)
        target_pos = copy_mechanism.get("target_pos", -1)

        if source_pos < 0 or target_pos < 0 or source_pos >= len(tokens) or target_pos >= len(tokens):
            return 0.0

        # Analyze source context
        source_context = self._analyze_token_context(tokens, source_pos)
        target_context = self._analyze_token_context(tokens, target_pos)

        # Analyze copy pattern
        copy_analysis = self._analyze_generic_copy(
            tokens[source_pos],
            tokens[max(0, target_pos - 2):target_pos + 3],
            copy_mechanism.get("attention_strength", 0.0)
        )

        # Calculate semantic strength
        semantic_strength = (
                copy_analysis["semantic_relevance"] * 0.5 +
                copy_analysis["confidence"] * 0.3 +
                source_context["semantic_features"]["has_arithmetic"] * 0.2
        )

        return min(1.0, semantic_strength)