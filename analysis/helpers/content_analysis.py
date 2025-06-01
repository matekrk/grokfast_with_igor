# analysis/helpers/content_analysis.py
from collections import defaultdict
from typing import List, Tuple, Dict  #, Any
# import re


class ContentAwareCircuitAnalyzer:
    """Analyze circuit behavior based on token content and context"""

    def __init__(self, task="modular_arithmetic", model=None, tokenizer=None):
        """
        Initialize content-aware circuit analyzer

        Args:
            task: Task name (for backward compatibility) - can be "modular_arithmetic", etc.
            model: Optional model reference (for advanced analysis)
            tokenizer: Optional tokenizer (for token processing)
        """
        self.task = task
        self.model = model
        self.tokenizer = tokenizer

        # Initialize token classifications based on task
        self._initialize_token_types_for_task(task)

        print(f"✅ ContentAwareCircuitAnalyzer initialized for task: {task}")

    def _initialize_token_types_for_task(self, task):
        """Initialize token type classifications based on task using table format"""
        if task == "modular_arithmetic" or task.startswith("modular"):
            # For modular arithmetic tasks - use table format for compatibility
            self.token_types = {
                "numbers": {str(i): True for i in range(100)},  # 0-99 as table
                "operators": {'+': True, '-': True, '*': True, '=': True, '%': True},
                "special": {'[BOS]': True, '[EOS]': True, '[PAD]': True, '<pad>': True}
            }

        elif task == "language_modeling" or task.startswith("language"):
            # For language modeling tasks
            self.token_types = {
                "numbers": {str(i): True for i in range(1000)},  # More numbers
                "operators": {'+': True, '-': True, '*': True, '/': True, '=': True,
                              '%': True, '<': True, '>': True, '&': True, '|': True},
                "special": {'[BOS]': True, '[EOS]': True, '[PAD]': True, '<pad>': True,
                            '[UNK]': True, '[MASK]': True}
            }

        else:
            # Default/generic task
            self.token_types = {
                "numbers": {str(i): True for i in range(100)},
                "operators": {'+': True, '-': True, '*': True, '=': True},
                "special": {'[BOS]': True, '[EOS]': True, '[PAD]': True}
            }

        # Create convenience sets for fast lookup (maintain both formats)
        self.number_tokens = set(self.token_types["numbers"].keys())
        self.operator_tokens = set(self.token_types["operators"].keys())
        self.special_tokens = set(self.token_types["special"].keys())

        print(f"\t\tToken types initialized for {task}: {len(self.number_tokens)} numbers, "
              f"{len(self.operator_tokens)} operators, {len(self.special_tokens)} special")

    def analyze_copy_semantics(self, copy_mechanism, tokens, attention_patterns=None):
        """
        EXISTING METHOD - analyze semantic content of copy operations
        This method calls _analyze_generic_copy with full token list
        """
        source_pos = copy_mechanism.get("source_pos", -1)
        target_pos = copy_mechanism.get("target_pos", -1)
        attention_strength = copy_mechanism.get("attention_strength", 0.0)

        if source_pos < 0 or target_pos < 0 or source_pos >= len(tokens) or target_pos >= len(tokens):
            return {"valid": False, "reason": "invalid_positions"}

        # Analyze source context
        source_context = self._analyze_token_context(tokens, source_pos)

        # Analyze target context
        target_context = self._analyze_token_context(tokens, target_pos)

        # THIS IS THE CORRECTED CALL - matching the actual usage
        copy_analysis = self._analyze_generic_copy(tokens, source_pos, target_pos, target_context)

        return {
            "valid": True,
            "source_context": source_context,
            "target_context": target_context,
            "copy_analysis": copy_analysis,
            "semantic_strength": copy_analysis.get("semantic_relevance", 0.0),
            "copy_type": copy_analysis.get("copy_type", "unknown")
        }

    def _analyze_arithmetic_copy(self, tokens: List[str], source_pos: int,
                                 target_pos: int, context: Dict) -> Tuple[str, float, Dict]:
        """Analyze copy in modular arithmetic context"""
        source_token = tokens[source_pos]

        # Number copying patterns
        if self.is_number(source_token):
            # Check if copying operand to result position
            if self.is_result_position(tokens, target_pos):
                if self.is_operand_position(tokens, source_pos):
                    return "operand_to_result", 0.9, {
                        "operation_type": "arithmetic_result_copy",
                        "source_value": source_token,
                        "context": "operand_copying"
                    }

            # Check if copying between operand positions
            if (self.is_operand_position(tokens, source_pos) and
                    self.is_operand_position(tokens, target_pos)):
                return "operand_to_operand", 0.8, {
                    "operation_type": "operand_copying",
                    "source_value": source_token,
                    "context": "operand_replication"
                }

    # Operator copying
        if self.is_operator(source_token):
            return "operator_copy", 0.7, {
                "operation_type": "operator_copying",
                "operator": source_token
            }

        return "positional_arithmetic", 0.6, {"context": "arithmetic_but_unclear"}

    def is_number(self, token):
        """Check if token is a number using table format"""
        return token in self.token_types["numbers"]

    def is_operator(self, token):
        """Check if token is an operator using table format"""
        return token in self.token_types["operators"]

    def is_special(self, token):
        """Check if token is special using table format"""
        return token in self.token_types["special"]

    def _get_token_type(self, token):
        """Get type of a single token using corrected methods"""
        if self.is_number(token):
            return "number"
        elif self.is_operator(token):
            return "operator"
        elif self.is_special(token):
            return "special"
        else:
            return "unknown"

    def is_result_position(self, tokens: List[str], pos: int) -> bool:
        """Check if position is after equals sign (result position)"""
        # Find equals sign
        try:
            equals_pos = tokens.index('=')
            return pos > equals_pos
        except ValueError:
            return False

    def is_operand_position(self, tokens: List[str], pos: int) -> bool:
        """Check if position contains an operand (before equals)"""
        try:
            equals_pos = tokens.index('=')
            return pos < equals_pos and self.is_number(tokens[pos])
        except ValueError:
            return self.is_number(tokens[pos])

    def detect_algorithmic_patterns(self, copy_mechanisms: List[Dict]) -> List[Dict]:
        """Detect higher-level algorithmic patterns from copy mechanisms"""
        patterns = []

        # Group by semantic type
        semantic_groups = self._group_by_semantics(copy_mechanisms)

        # Detect algorithmic patterns
        for semantic_type, mechanisms in semantic_groups.items():
            if len(mechanisms) >= 2:  # Need multiple instances
                pattern = self._detect_pattern_in_group(semantic_type, mechanisms)
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _group_by_semantics(self, mechanisms: List[Dict]) -> Dict[str, List[Dict]]:
        """Group mechanisms by their semantic type"""
        groups = defaultdict(list)

        for mech in mechanisms:
            semantic_type = mech.get("copy_type", "unknown")
            groups[semantic_type].append(mech)

        return dict(groups)

    def _detect_pattern_in_group(self, semantic_type: str, mechanisms: List[Dict]) -> Dict:
        """Detect patterns within a semantic group"""
        if semantic_type == "operand_to_result":
            # This could be the core arithmetic computation pattern
            return {
                "pattern_type": "arithmetic_computation",
                "pattern_name": "operand_to_result_pattern",
                "mechanism_count": len(mechanisms),
                "confidence": min(1.0, len(mechanisms) / 5.0),
                "description": "Copying operands to result position (arithmetic computation)"
            }

        elif semantic_type == "operand_to_operand":
            return {
                "pattern_type": "operand_manipulation",
                "pattern_name": "operand_copying_pattern",
                "mechanism_count": len(mechanisms),
                "confidence": min(1.0, len(mechanisms) / 3.0),
                "description": "Copying between operand positions (intermediate computation)"
            }

        return None


    def _analyze_token_context(self, tokens, position, window_size=3):
        """
        Analyze semantic context around a token position
        """
        if position >= len(tokens) or position < 0:
            return {
                "valid": False,
                "reason": "position_out_of_bounds",
                "semantic_type": "unknown",
                "context_strength": 0.0
            }

        # Extract context window
        start_pos = max(0, position - window_size)
        end_pos = min(len(tokens), position + window_size + 1)
        context_tokens = tokens[start_pos:end_pos]

        # Analyze token types in context (uses corrected token classification)
        token_types = self._classify_tokens_in_context(context_tokens)

        # Determine semantic type based on context
        semantic_type = self._determine_semantic_type(context_tokens, position - start_pos)

        # Calculate context strength
        context_strength = self._calculate_context_strength(context_tokens, token_types)

        return {
            "valid": True,
            "position": position,
            "context_tokens": context_tokens,
            "context_window": (start_pos, end_pos),
            "target_token": tokens[position],
            "token_types": token_types,
            "semantic_type": semantic_type,
            "context_strength": context_strength,
            "task": self.task,
            "has_arithmetic": any(self.is_operator(token) for token in context_tokens),
            "number_density": sum(1 for t in context_tokens if self.is_number(t)) / len(context_tokens)
        }


    def _analyze_generic_copy(self, tokens, source_pos, target_pos, context_analysis):
        """
        CORRECTED: Analyze generic copying patterns with full token list

        Args:
            tokens: Full list of tokens (List[str])
            source_pos: Source position index (int)
            target_pos: Target position index (int)
            context_analysis: Context analysis from _analyze_token_context

        Returns:
            Dict with copy analysis
        """
        # Validate positions
        if (source_pos < 0 or target_pos < 0 or
                source_pos >= len(tokens) or target_pos >= len(tokens)):
            return {
                "copy_type": "invalid_positions",
                "confidence": 0.0,
                "semantic_relevance": 0.0,
                "functional_purpose": "unknown",
                "task": self.task
            }

        # Extract source token and target context
        source_token = tokens[source_pos]

        # Get target context (window around target)
        target_window_start = max(0, target_pos - 2)
        target_window_end = min(len(tokens), target_pos + 3)
        target_context = tokens[target_window_start:target_window_end]

        # Get attention strength from context analysis or default
        attention_strength = context_analysis.get("context_strength", 0.5)

        # Determine source token type using corrected methods
        source_type = self._get_token_type(source_token)

        # Analyze target context
        if not target_context or len(target_context) == 0:
            return {
                "copy_type": "no_context",
                "confidence": 0.1,
                "semantic_relevance": 0.0,
                "functional_purpose": "unknown",
                "task": self.task
            }

        # Classify tokens in target context
        target_types = [self._get_token_type(token) for token in target_context]

        # Determine copy type and confidence
        copy_analysis = self._classify_copy_pattern(
            source_token, source_type, target_context, target_types, attention_strength
        )

        # Add positional and task info
        copy_analysis.update({
            "task": self.task,
            "source_pos": source_pos,
            "target_pos": target_pos,
            "source_token": source_token,
            "target_context": target_context
        })

        return copy_analysis


    def _classify_tokens_in_context(self, context_tokens):
        """Classify tokens in context by type using task-specific classifications"""
        classification = {
            "numbers": [],
            "operators": [],
            "special": [],
            "unknown": []
        }
        for i, token in enumerate(context_tokens):
            if self.is_number(token):
                classification["numbers"].append(i)
            elif self.is_operator(token):
                classification["operators"].append(i)
            elif self.is_special(token):
                classification["special"].append(i)
            else:
                classification["unknown"].append(i)

        return classification

    """
    def _get_token_type(self, token):
        "Get type of a single token using task-specific classifications"
        if token in self.number_tokens:
            return "number"
        elif token in self.operator_tokens:
            return "operator"
        elif token in self.special_tokens:
            return "special"
        else:
            return "unknown"
    """

    def _determine_semantic_type(self, context_tokens, target_idx):
        """Determine semantic type based on task and context"""
        # Use task-specific logic
        has_arithmetic = any(op in context_tokens for op in self.operator_tokens)
        has_equation = '=' in context_tokens
        has_numbers = any(token in self.number_tokens for token in context_tokens)

        if self.task == "modular_arithmetic":
            if has_equation and has_arithmetic:
                return "modular_equation_context"
            elif has_arithmetic and has_numbers:
                return "modular_arithmetic_context"
            elif has_numbers:
                return "modular_numeric_context"
            else:
                return "generic_context"
        else:
            # Generic task logic
            if has_equation and has_arithmetic:
                return "equation_context"
            elif has_arithmetic and has_numbers:
                return "arithmetic_context"
            elif has_numbers:
                return "numeric_context"
            else:
                return "generic_context"

    def _calculate_context_strength(self, context_tokens, token_types):
        """Calculate strength of semantic context (same implementation)"""
        if not context_tokens:
            return 0.0

        # Strong context indicators
        arithmetic_density = len(token_types["operators"]) / len(context_tokens)
        number_density = len(token_types["numbers"]) / len(context_tokens)

        # Context is stronger if it has clear arithmetic structure
        structure_score = 0.0
        if arithmetic_density > 0.2:  # At least 20% operators
            structure_score += 0.4
        if number_density > 0.4:  # At least 40% numbers
            structure_score += 0.3
        if '=' in context_tokens:  # Equation structure
            structure_score += 0.3

        return min(1.0, structure_score)

    def _classify_copy_pattern(self, source_token, source_type, target_context, target_types, attention_strength):
        """Classify copy pattern with task-aware logic"""
        copy_analysis = {
            "copy_type": "generic_copy",
            "confidence": 0.5,
            "semantic_relevance": 0.5,
            "functional_purpose": "unknown"
        }

        # Task-specific pattern classification
        if self.task == "modular_arithmetic":
            copy_analysis = self._classify_modular_arithmetic_copy(
                source_token, source_type, target_context, target_types, attention_strength
            )
        else:
            copy_analysis = self._classify_generic_copy_pattern(
                source_token, source_type, target_context, target_types, attention_strength
            )

        return copy_analysis

    def _classify_modular_arithmetic_copy(self, source_token, source_type, target_context, target_types,
                                          attention_strength):
        """Classify copy patterns specific to modular arithmetic"""
        copy_analysis = {
            "copy_type": "modular_generic",
            "confidence": 0.5,
            "semantic_relevance": 0.5,
            "functional_purpose": "unknown"
        }

        # Modular arithmetic specific patterns
        if source_type == "number":
            if "number" in target_types:
                copy_analysis.update({
                    "copy_type": "modular_number_to_number",
                    "confidence": 0.8,
                    "semantic_relevance": 0.9,
                    "functional_purpose": "modular_operand"
                })
            elif any(op in target_context for op in ['+', '-', '*', '%']):
                copy_analysis.update({
                    "copy_type": "modular_number_to_arithmetic",
                    "confidence": 0.7,
                    "semantic_relevance": 0.8,
                    "functional_purpose": "modular_computation"
                })
            elif '=' in target_context:
                copy_analysis.update({
                    "copy_type": "modular_number_to_result",
                    "confidence": 0.9,
                    "semantic_relevance": 0.95,
                    "functional_purpose": "modular_result"
                })

        # Boost for strong attention
        attention_factor = min(1.5, attention_strength / 0.5)
        copy_analysis["confidence"] = min(1.0, copy_analysis["confidence"] * attention_factor)

        return copy_analysis

    def _classify_generic_copy_pattern(self, source_token, source_type, target_context, target_types,
                                       attention_strength):
        """Generic copy pattern classification (fallback)"""
        # Same logic as before but simplified
        copy_analysis = {
            "copy_type": "generic_positional",
            "confidence": 0.3,
            "semantic_relevance": 0.2,
            "functional_purpose": "positional_pattern"
        }

        if source_type == "number" and "number" in target_types:
            copy_analysis.update({
                "copy_type": "number_to_number",
                "confidence": 0.6,
                "semantic_relevance": 0.7
            })

        return copy_analysis

    def is_pattern_completion(self, source_token, target_context):
        """Check for pattern completion (same implementation)"""
        if source_token in target_context:
            return True

        if source_token in self.number_tokens:
            numbers_in_context = [t for t in target_context if t in self.number_tokens]
            if len(numbers_in_context) >= 2:
                return True

        return False