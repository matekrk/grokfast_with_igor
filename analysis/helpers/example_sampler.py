# analysis/helpers/example_sampler.py
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional     #, Callable
from collections import defaultdict
# from pathlib import Path
import random


class ExampleSampler:
    """
    Intelligent example sampling for circuit analysis with multiple strategies

    Supports:
    - Random sampling with reproducible seeds
    - Diverse sampling based on token patterns
    - Rotating systematic sampling
    - Pattern-aware caching
    - Adaptive budget based on training phase
    """

    def __init__(self, eval_loader, base_budget: int = 3, max_cache_size: int = 50,
                 diversity_metrics: Optional[List[str]] = None, logger=None):
        """
        Initialize ExampleSampler

        Args:
            eval_loader: DataLoader for evaluation data
            base_budget: Default number of examples to sample
            max_cache_size: Maximum number of cached examples
            diversity_metrics: List of diversity metrics to use
        """
        self.eval_loader = eval_loader
        self.base_budget = base_budget
        self.max_cache_size = max_cache_size

        # Diversity metrics to use
        self.diversity_metrics = diversity_metrics or ["entropy", "repetition", "unique_tokens"]

        # State tracking
        self.rotation_state = {"batch_idx": 0, "example_idx": 0}
        self.sampling_history = defaultdict(list)

        # Pattern-aware cache
        self.cached_examples = []
        self.pattern_signatures = set()

        # Pre-computed diversity scores for efficiency
        self.diversity_cache = {}
        self.cache_valid = False

        # info logger
        self.logger = logger

    def sample_examples(self, epoch: int, total_epochs: int = 1000,
                        strategy: str = "diverse_random",
                        budget: Optional[int] = None,
                        seed_offset: int = 0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample examples using specified strategy

        Args:
            epoch: Current training epoch
            total_epochs: Total training epochs
            strategy: Sampling strategy ("random", "diverse_random", "rotating", "cached", "adaptive_budget")
            budget: Override default budget
            seed_offset: Additional seed offset for reproducibility

        Returns:
            List of (inputs, targets) tuples
        """
        # Determine budget
        if budget is None:
            budget = self._get_adaptive_budget(epoch, total_epochs)

        # Set reproducible seed
        seed = epoch + seed_offset
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Route to appropriate sampling method
        if strategy == "random":
            examples = self._sample_random(budget, seed)
        elif strategy == "diverse_random":
            examples = self._sample_diverse_random(budget, seed)
        elif strategy == "rotating":
            examples = self._sample_rotating(budget)
        elif strategy == "cached":
            examples = self._sample_from_cache(budget)
        elif strategy == "adaptive_budget":
            examples = self._sample_adaptive_budget(epoch, total_epochs, seed)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Record sampling history
        self.sampling_history[epoch] = {
            "strategy": strategy,
            "budget": len(examples),
            "examples_sampled": len(examples)
        }

        return examples

    def _get_adaptive_budget(self, epoch: int, total_epochs: int) -> int:
        """Calculate adaptive budget based on training phase"""
        training_progress = epoch / max(total_epochs, 1)

        if training_progress < 0.2:  # Early training - more exploration
            return self.base_budget * 2
        elif training_progress < 0.6:  # Middle training - standard budget
            return self.base_budget
        else:  # Late training - focused sampling
            return max(1, self.base_budget // 2)

    def _sample_random(self, budget: int, seed: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample random examples from the dataset"""
        examples = []
        dataset_size = len(self.eval_loader)

        # Convert DataLoader to list for random access (cache if needed)
        if not hasattr(self, '_cached_batches'):
            self._cached_batches = list(self.eval_loader)

        for i in range(budget):
            # Random batch
            batch_idx = random.randint(0, len(self._cached_batches) - 1)
            inputs, targets = self._cached_batches[batch_idx]

            # Random example within batch
            example_idx = random.randint(0, inputs.shape[0] - 1)

            examples.append((
                inputs[example_idx:example_idx + 1],
                targets[example_idx:example_idx + 1]
            ))

        return examples

    def _sample_diverse_random(self, budget: int, seed: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample examples with maximum diversity in token patterns"""
        # Ensure diversity cache is valid
        if not self.cache_valid:
            self._build_diversity_cache()

        # Get candidates sorted by diversity
        candidates = []
        for (batch_idx, example_idx), diversity_score in self.diversity_cache.items():
            candidates.append((diversity_score, batch_idx, example_idx))

        # Sort by diversity score (highest first)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Select top diverse examples with some randomness
        selected_examples = []

        # Take top 30% deterministically
        deterministic_count = max(1, budget // 3)
        for i in range(min(deterministic_count, len(candidates))):
            _, batch_idx, example_idx = candidates[i]
            inputs, targets = self._cached_batches[batch_idx]
            selected_examples.append((
                inputs[example_idx:example_idx + 1],
                targets[example_idx:example_idx + 1]
            ))

        # Fill remaining budget with random selection from top 50%
        remaining_budget = budget - len(selected_examples)
        if remaining_budget > 0:
            top_half_size = len(candidates) // 2
            random_candidates = random.sample(candidates[:top_half_size],
                                              min(remaining_budget, top_half_size))

            for _, batch_idx, example_idx in random_candidates:
                inputs, targets = self._cached_batches[batch_idx]
                selected_examples.append((
                    inputs[example_idx:example_idx + 1],
                    targets[example_idx:example_idx + 1]
                ))

        return selected_examples

    def _sample_rotating(self, budget: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample examples using systematic rotation through dataset"""
        examples = []

        for _ in range(budget):
            # Get current batch and example
            batch_idx = self.rotation_state["batch_idx"]
            example_idx = self.rotation_state["example_idx"]

            # Ensure we have cached batches
            if not hasattr(self, '_cached_batches'):
                self._cached_batches = list(self.eval_loader)

            # Get the example
            inputs, targets = self._cached_batches[batch_idx]
            if example_idx < inputs.shape[0]:
                examples.append((
                    inputs[example_idx:example_idx + 1],
                    targets[example_idx:example_idx + 1]
                ))

            # Advance rotation state
            self._advance_rotation_state()

        return examples

    def _advance_rotation_state(self):
        """Advance the rotation state to next example"""
        if not hasattr(self, '_cached_batches'):
            self._cached_batches = list(self.eval_loader)

        self.rotation_state["example_idx"] += 1

        # Check if we need to move to next batch
        current_batch = self._cached_batches[self.rotation_state["batch_idx"]]
        if self.rotation_state["example_idx"] >= current_batch[0].shape[0]:
            self.rotation_state["example_idx"] = 0
            self.rotation_state["batch_idx"] = (self.rotation_state["batch_idx"] + 1) % len(self._cached_batches)

    def _sample_from_cache(self, budget: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample from pattern-aware cache"""
        if len(self.cached_examples) == 0:
            # Cache is empty, fall back to diverse sampling
            return self._sample_diverse_random(budget, 0)

        # Sample from cache
        selected_count = min(budget, len(self.cached_examples))
        selected_indices = random.sample(range(len(self.cached_examples)), selected_count)

        examples = []
        for idx in selected_indices:
            examples.append(self.cached_examples[idx])

        # Fill remaining budget with new diverse examples if needed
        remaining_budget = budget - len(examples)
        if remaining_budget > 0:
            new_examples = self._sample_diverse_random(remaining_budget, 1)
            examples.extend(new_examples)

            # Add new examples to cache
            self._update_cache(new_examples)

        return examples

    def _sample_adaptive_budget(self, epoch: int, total_epochs: int, seed: int) -> List[
        Tuple[torch.Tensor, torch.Tensor]]:
        """Sample with adaptive budget and strategy based on training phase"""
        training_progress = epoch / max(total_epochs, 1)

        if training_progress < 0.2:
            # Early training: diverse exploration
            budget = self.base_budget * 2
            return self._sample_diverse_random(budget, seed)
        elif training_progress < 0.6:
            # Middle training: balanced approach
            budget = self.base_budget
            # Mix of diverse and rotating
            diverse_count = budget // 2
            rotating_count = budget - diverse_count

            diverse_examples = self._sample_diverse_random(diverse_count, seed)
            rotating_examples = self._sample_rotating(rotating_count)

            return diverse_examples + rotating_examples
        else:
            # Late training: focused on cached patterns
            budget = max(1, self.base_budget // 2)
            return self._sample_from_cache(budget)

    def _build_diversity_cache(self):
        """Build cache of diversity scores for all examples"""
        if not hasattr(self, '_cached_batches'):
            self._cached_batches = list(self.eval_loader)

        self.diversity_cache = {}

        for batch_idx, (inputs, targets) in enumerate(self._cached_batches):
            # Limit analysis to avoid startup delay
            max_examples_per_batch = min(10, inputs.shape[0])

            for example_idx in range(max_examples_per_batch):
                example = inputs[example_idx]
                diversity_score = self._calculate_diversity_score(example)
                self.diversity_cache[(batch_idx, example_idx)] = diversity_score

        self.cache_valid = True

    def _calculate_diversity_score(self, example: torch.Tensor) -> float:
        """Calculate diversity score for an example"""
        tokens = example.cpu().numpy().flatten()
        score = 0.0

        # Entropy-based diversity
        if "entropy" in self.diversity_metrics:
            unique_tokens, counts = np.unique(tokens, return_counts=True)
            probs = counts / len(tokens)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            score += entropy

        # Repetition patterns (good for copy mechanisms)
        if "repetition" in self.diversity_metrics:
            has_repetitions = len(tokens) != len(set(tokens))
            score += 1.0 if has_repetitions else 0.0

        # Unique token count
        if "unique_tokens" in self.diversity_metrics:
            unique_count = len(set(tokens))
            score += unique_count * 0.1

        # Sequential patterns (good for induction)
        if "sequential" in self.diversity_metrics:
            sequential_score = self._calculate_sequential_score(tokens)
            score += sequential_score

        return score

    def _calculate_sequential_score(self, tokens: np.ndarray) -> float:
        """Calculate score based on sequential patterns"""
        score = 0.0

        # Look for A-B-A patterns (induction-like)
        for i in range(len(tokens) - 2):
            for j in range(i + 2, len(tokens)):
                if tokens[i] == tokens[j]:  # Found A-?-A pattern
                    score += 0.5

        return score

    def _update_cache(self, new_examples: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Update pattern-aware cache with new examples"""
        for inputs, targets in new_examples:
            # Calculate pattern signature
            example = inputs[0]  # Single example
            pattern_signature = self._get_pattern_signature(example)

            # Add if novel pattern
            if pattern_signature not in self.pattern_signatures:
                self.cached_examples.append((inputs, targets))
                self.pattern_signatures.add(pattern_signature)

                # Maintain cache size limit
                if len(self.cached_examples) > self.max_cache_size:
                    # Remove oldest example
                    removed_example = self.cached_examples.pop(0)
                    # Would need to track signatures to remove properly

    def _get_pattern_signature(self, example: torch.Tensor) -> str:
        """Generate a signature for pattern matching"""
        tokens = example.cpu().numpy().flatten()

        # Create signature based on token patterns
        unique_tokens = sorted(set(tokens))
        token_map = {token: i for i, token in enumerate(unique_tokens)}

        # Map to normalized pattern
        pattern = [token_map[token] for token in tokens]

        return str(pattern)

    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get statistics about sampling behavior"""
        return {
            "total_epochs_sampled": len(self.sampling_history),
            "cached_examples": len(self.cached_examples),
            "unique_patterns": len(self.pattern_signatures),
            "diversity_cache_size": len(self.diversity_cache),
            "recent_sampling": dict(list(self.sampling_history.items())[-5:])  # Last 5 epochs
        }

    def reset_cache(self):
        """Reset all caches and state"""
        self.cached_examples = []
        self.pattern_signatures = set()
        self.diversity_cache = {}
        self.cache_valid = False
        self.rotation_state = {"batch_idx": 0, "example_idx": 0}
        if hasattr(self, '_cached_batches'):
            delattr(self, '_cached_batches')



class FixedExampleSampler:
    """Fixed version of ExampleSampler with proper randomization"""

    def __init__(self, eval_loader, base_budget=3, max_cache_size=50, diversity_metrics=None):
        self.eval_loader = eval_loader
        self.base_budget = base_budget
        self.max_cache_size = max_cache_size
        self.diversity_metrics = diversity_metrics or ["entropy", "repetition", "unique_tokens"]

        # State tracking
        self.rotation_state = {"batch_idx": 0, "example_idx": 0}
        self.sampling_history = defaultdict(list)

        # Convert eval_loader to list once for random access
        self._cached_batches = list(eval_loader)
        print(f"📦 Cached {len(self._cached_batches)} batches with total examples")

        # Pattern-aware cache
        self.cached_examples = []
        self.pattern_signatures = set()

        # Build diversity cache once
        self.diversity_cache = {}
        self.cache_valid = False
        self._build_diversity_cache()

    def sample_examples(self, epoch, total_epochs=1000, strategy="diverse_random",
                        budget=None, seed_offset=0):
        """Fixed sampling with proper randomization"""

        if budget is None:
            budget = self._get_adaptive_budget(epoch, total_epochs)

        # ✅ FIX: Create unique seed for each strategy and epoch
        base_seed = epoch * 1000 + seed_offset
        strategy_seeds = {
            "random": base_seed + 1,
            "diverse_random": base_seed + 2,
            "rotating": base_seed + 3,  # Rotating shouldn't use random, but just in case
            "cached": base_seed + 4,
            "adaptive_budget": base_seed + 5
        }

        seed = strategy_seeds.get(strategy, base_seed)

        # ✅ FIX: Set seeds properly
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        print(f"🎲 Strategy '{strategy}' @ epoch {epoch} using seed {seed}")

        # Route to fixed methods
        if strategy == "random":
            examples = self._sample_random_fixed(budget, seed)
        elif strategy == "diverse_random":
            examples = self._sample_diverse_random_fixed(budget, seed)
        elif strategy == "rotating":
            examples = self._sample_rotating_fixed(budget)  # No randomness
        elif strategy == "cached":
            examples = self._sample_from_cache_fixed(budget, seed)
        elif strategy == "adaptive_budget":
            examples = self._sample_adaptive_budget_fixed(epoch, total_epochs, seed)
        else:
            examples = self._sample_random_fixed(budget, seed)

        # Debug output
        example_signatures = []
        for inputs, targets in examples:
            tokens = inputs[0].cpu().numpy().flatten()
            signature = tuple(tokens[:3])
            example_signatures.append(signature)

        print(f"    📋 Returned {len(examples)} examples: {example_signatures}")

        return examples

    def _sample_random_fixed(self, budget, seed):
        """Fixed random sampling"""
        examples = []

        for i in range(budget):
            # ✅ FIX: Use different random state for each sample
            local_random = random.Random(seed + i * 37)  # Different seed each iteration

            # Random batch
            batch_idx = local_random.randint(0, len(self._cached_batches) - 1)
            inputs, targets = self._cached_batches[batch_idx]

            # Random example within batch
            example_idx = local_random.randint(0, inputs.shape[0] - 1)

            examples.append((
                inputs[example_idx:example_idx + 1],
                targets[example_idx:example_idx + 1]
            ))

        return examples

    def _sample_diverse_random_fixed(self, budget, seed):
        """Fixed diverse sampling"""
        if not self.cache_valid:
            self._build_diversity_cache()

        # Get all candidates with diversity scores
        candidates = [(score, batch_idx, example_idx)
                      for (batch_idx, example_idx), score in self.diversity_cache.items()]

        # Sort by diversity (highest first)
        candidates.sort(key=lambda x: x[0], reverse=True)

        examples = []
        local_random = random.Random(seed + 100)  # Consistent but different from random strategy

        # Take top candidates with some randomness
        if len(candidates) >= budget:
            # Take from top 50% with randomness
            top_half_size = len(candidates) // 2
            top_candidates = candidates[:max(top_half_size, budget * 2)]

            selected_candidates = local_random.sample(top_candidates, min(budget, len(top_candidates)))
        else:
            selected_candidates = candidates

        for _, batch_idx, example_idx in selected_candidates:
            inputs, targets = self._cached_batches[batch_idx]
            examples.append((
                inputs[example_idx:example_idx + 1],
                targets[example_idx:example_idx + 1]
            ))

        return examples

    def _sample_rotating_fixed(self, budget):
        """Fixed rotating sampling - deterministic, no randomness"""
        examples = []

        for _ in range(budget):
            # Get current position
            batch_idx = self.rotation_state["batch_idx"]
            example_idx = self.rotation_state["example_idx"]

            # Get example
            inputs, targets = self._cached_batches[batch_idx]
            if example_idx < inputs.shape[0]:
                examples.append((
                    inputs[example_idx:example_idx + 1],
                    targets[example_idx:example_idx + 1]
                ))

            # Advance state
            self._advance_rotation_state()

        return examples

    def _sample_from_cache_fixed(self, budget, seed):
        """Fixed cache sampling"""
        if len(self.cached_examples) == 0:
            # No cache, fall back to diverse sampling
            return self._sample_diverse_random_fixed(budget, seed + 200)

        examples = []
        local_random = random.Random(seed + 300)

        # Sample from existing cache
        cache_budget = min(budget, len(self.cached_examples))
        if cache_budget > 0:
            selected_cache_examples = local_random.sample(self.cached_examples, cache_budget)
            examples.extend(selected_cache_examples)

        # Fill remaining with new diverse examples
        remaining = budget - len(examples)
        if remaining > 0:
            new_examples = self._sample_diverse_random_fixed(remaining, seed + 400)
            examples.extend(new_examples)

        return examples

    def _sample_adaptive_budget_fixed(self, epoch, total_epochs, seed):
        """Fixed adaptive budget sampling"""
        progress = epoch / max(total_epochs, 1)

        if progress < 0.2:
            # Early: diverse exploration
            budget = self.base_budget * 2
            return self._sample_diverse_random_fixed(budget, seed + 500)
        elif progress < 0.6:
            # Middle: mixed approach
            budget = self.base_budget
            diverse_count = budget // 2
            random_count = budget - diverse_count

            diverse_examples = self._sample_diverse_random_fixed(diverse_count, seed + 600)
            random_examples = self._sample_random_fixed(random_count, seed + 700)

            return diverse_examples + random_examples
        else:
            # Late: cached patterns
            budget = max(1, self.base_budget // 2)
            return self._sample_from_cache_fixed(budget, seed + 800)

    def _get_adaptive_budget(self, epoch, total_epochs):
        """Calculate adaptive budget"""
        progress = epoch / max(total_epochs, 1)
        if progress < 0.2:
            return self.base_budget * 2
        elif progress < 0.6:
            return self.base_budget
        else:
            return max(1, self.base_budget // 2)

    def _build_diversity_cache(self):
        """Build diversity cache"""
        self.diversity_cache = {}

        for batch_idx, (inputs, targets) in enumerate(self._cached_batches):
            max_examples = min(10, inputs.shape[0])

            for example_idx in range(max_examples):
                example = inputs[example_idx]
                diversity_score = self._calculate_diversity_score(example)
                self.diversity_cache[(batch_idx, example_idx)] = diversity_score

        self.cache_valid = True
        print(f"📊 Built diversity cache with {len(self.diversity_cache)} examples")

    def _calculate_diversity_score(self, example):
        """Calculate diversity score"""
        tokens = example.cpu().numpy().flatten()
        score = 0.0

        # Entropy
        unique_tokens, counts = np.unique(tokens, return_counts=True)
        if len(counts) > 1:
            probs = counts / len(tokens)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            score += entropy

        # Repetitions
        has_repetitions = len(tokens) != len(set(tokens))
        score += 1.0 if has_repetitions else 0.0

        # Unique count
        score += len(set(tokens)) * 0.1

        return score

    def _advance_rotation_state(self):
        """Advance rotation state"""
        self.rotation_state["example_idx"] += 1

        current_batch = self._cached_batches[self.rotation_state["batch_idx"]]
        if self.rotation_state["example_idx"] >= current_batch[0].shape[0]:
            self.rotation_state["example_idx"] = 0
            self.rotation_state["batch_idx"] = (self.rotation_state["batch_idx"] + 1) % len(self._cached_batches)

    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get statistics about sampling behavior"""
        return {
            "total_epochs_sampled": len(self.sampling_history),
            "cached_examples": len(self.cached_examples),
            "unique_patterns": len(self.pattern_signatures),
            "diversity_cache_size": len(self.diversity_cache),
            "recent_sampling": dict(list(self.sampling_history.items())[-5:])  # Last 5 epochs
        }



# ============================================================================
# 🔧 QUICK FIX: Replace ExampleSampler in your code
# ============================================================================

def create_fixed_example_sampler(eval_loader, strategy_config=None):
    """Create fixed example sampler"""
    config = strategy_config or {}

    return FixedExampleSampler(
        eval_loader=eval_loader,
        base_budget=config.get("base_budget", 3),
        max_cache_size=config.get("max_cache_size", 50),
        diversity_metrics=config.get("diversity_metrics", ["entropy", "repetition", "unique_tokens"])
    )

# Integration helper for AdaptiveTokenOperationDetector
def create_example_sampler(eval_loader, strategy_config: Optional[Dict[str, Any]] = None) -> ExampleSampler:
    """
    Factory function to create configured ExampleSampler

    Args:
        eval_loader: DataLoader for evaluation
        strategy_config: Configuration for sampling strategies

    Returns:
        Configured ExampleSampler instance
    """
    config = strategy_config or {}

    return ExampleSampler(
        eval_loader=eval_loader,
        base_budget=config.get("base_budget", 3),
        max_cache_size=config.get("max_cache_size", 50),
        diversity_metrics=config.get("diversity_metrics", ["entropy", "repetition", "unique_tokens"])
    )
