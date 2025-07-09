# analysis/sampling/diversity_enhanced_sampling.py
"""
whatis Enhanced Diversity Sampling for Circuit Discovery

info Addresses the issue where diverse_random sampling becomes too repetitive,
 limiting circuit discovery to ~40 patterns. Implements aggressive diversity
 strategies to discover rare computational patterns.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import random
from collections import defaultdict, Counter


class AggressiveDiversitySampler:
    """
    info sampler focused on maximizing token pattern diversity for circuit discovery

    info ses anti-repetition mechanisms and forced exploration of rare patterns
    """

    def __init__(self, eval_loader, config: Dict[str, Any]):
        self.eval_loader = eval_loader
        self.config = config

        # info diversity tracking
        self.seen_token_patterns = set()
        self.seen_value_combinations = set()
        self.seen_operations = set()
        self.pattern_frequency = defaultdict(int)

        # info anti-repetition tracking
        self.recent_samples = []  # Recent sample history
        self.max_recent_history = config.get('max_recent_history', 50)
        self.repetition_penalty = config.get('repetition_penalty', 0.8)

        # info token value exploration
        self.token_ranges = config.get('token_ranges', {'min': 0, 'max': 97})
        self.force_rare_tokens = config.get('force_rare_tokens', True)
        self.rare_token_probability = config.get('rare_token_probability', 0.3)

        print(f"🎲 AggressiveDiversitySampler initialized with anti-repetition and forced exploration")

    def sample_examples(self, epoch: int, total_epochs: int, strategy: str = "aggressive_random",
                        num_samples: int = None, seed_offset: int = 0) -> List[Tuple]:
        """
        info sample with aggressive diversity maximization

        Args:
            strategy: 'aggressive_random', 'forced_diversity', 'curriculum_exploration'
        """
        if num_samples is None:
            num_samples = self.config.get('base_budget', 8)  # Increase default budget

        if strategy == "aggressive_random":
            return self._aggressive_random_sampling(num_samples, epoch, seed_offset)
        elif strategy == "forced_diversity":
            return self._forced_diversity_sampling(num_samples, epoch, seed_offset)
        elif strategy == "curriculum_exploration":
            return self._curriculum_exploration_sampling(num_samples, epoch, total_epochs, seed_offset)
        elif strategy == "pure_random":
            return self._pure_random_sampling(num_samples, seed_offset)
        else:
            return self._aggressive_random_sampling(num_samples, epoch, seed_offset)

    def _aggressive_random_sampling(self, num_samples: int, epoch: int, seed_offset: int) -> List[Tuple]:
        random.seed(epoch * 1000 + seed_offset)

        all_examples = self._get_all_examples()  # Now you have 4656 individual examples
        candidates = []
        max_attempts = num_samples * 20
        attempts = 0

        while len(candidates) < num_samples and attempts < max_attempts:
            idx = random.randint(0, len(all_examples) - 1)
            inputs, targets = all_examples[idx]

            tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]
            token_pattern = ' '.join(tokens)

            if self._should_include_sample(token_pattern, tokens, epoch):
                candidates.append((inputs, targets))
                self._update_diversity_tracking(token_pattern, tokens)

            attempts += 1

        return candidates

    def _get_all_examples(self):
        """Extract all individual examples from the DataLoader"""
        all_examples = []
        for batch_inputs, batch_targets in self.eval_loader:
            # Iterate through each example in the batch
            for i in range(batch_inputs.size(0)):  # batch_size
                single_input = batch_inputs[i:i+1]  # Keep batch dimension
                single_target = batch_targets[i:i+1]
                all_examples.append((single_input, single_target))
        return all_examples

    def _forced_diversity_sampling(self, num_samples: int, epoch: int, seed_offset: int) -> List[Tuple]:
        """whatis forced diversity with stratified token sampling"""
        random.seed(epoch * 1000 + seed_offset)
        np.random.seed(epoch * 1000 + seed_offset + 1)

        candidates = []
        all_examples = list(self.eval_loader)

        # info strategy 1: Sample across different token value ranges
        token_strata = self._create_token_strata()
        samples_per_stratum = max(1, num_samples // len(token_strata))

        for stratum_name, token_range in token_strata.items():
            stratum_candidates = []

            # info find examples in this token range
            for inputs, targets in all_examples:
                tokens = [int(x) for x in inputs[0].cpu().numpy()]
                if self._tokens_in_range(tokens, token_range):
                    token_pattern = ' '.join(map(str, tokens))
                    if self._should_include_sample(token_pattern, tokens, epoch):
                        stratum_candidates.append((inputs, targets))

                if len(stratum_candidates) >= samples_per_stratum * 3:  # Enough candidates
                    break

            # info random sample from this stratum
            selected = random.sample(stratum_candidates,
                                     min(samples_per_stratum, len(stratum_candidates)))
            candidates.extend(selected)

            for inputs, targets in selected:
                tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]
                self._update_diversity_tracking(' '.join(tokens), tokens)

        # info strategy 2: Fill remaining slots with anti-repetition random
        while len(candidates) < num_samples:
            idx = random.randint(0, len(all_examples) - 1)
            inputs, targets = all_examples[idx]
            tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]
            token_pattern = ' '.join(tokens)

            if self._should_include_sample(token_pattern, tokens, epoch):
                candidates.append((inputs, targets))
                self._update_diversity_tracking(token_pattern, tokens)

        print(f"  🎯 Forced diversity: {len(candidates)} samples across {len(token_strata)} strata")
        return candidates[:num_samples]

    def _curriculum_exploration_sampling(self, num_samples: int, epoch: int,
                                         total_epochs: int, seed_offset: int) -> List[Tuple]:
        """whatis curriculum that gradually explores rarer patterns"""
        random.seed(epoch * 1000 + seed_offset)

        # info exploration pressure increases over training
        exploration_factor = min(1.0, epoch / (total_epochs * 0.3))  # Full exploration by 30% of training
        rare_sample_ratio = 0.2 + exploration_factor * 0.5  # 20% → 70% rare samples

        candidates = []
        all_examples = list(self.eval_loader)

        # info classify examples by rarity
        common_examples = []
        rare_examples = []

        for inputs, targets in all_examples:
            tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]
            token_pattern = ' '.join(tokens)

            # info classify as rare or common based on previous observations
            if self._is_rare_pattern(token_pattern, tokens):
                rare_examples.append((inputs, targets))
            else:
                common_examples.append((inputs, targets))

        # info sample according to curriculum
        num_rare = int(num_samples * rare_sample_ratio)
        num_common = num_samples - num_rare

        # info sample rare examples
        if rare_examples and num_rare > 0:
            selected_rare = random.sample(rare_examples, min(num_rare, len(rare_examples)))
            candidates.extend(selected_rare)

        # info sample common examples
        if common_examples and num_common > 0:
            selected_common = random.sample(common_examples, min(num_common, len(common_examples)))
            candidates.extend(selected_common)

        # info update tracking
        for inputs, targets in candidates:
            tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]
            self._update_diversity_tracking(' '.join(tokens), tokens)

        print(f"  🎯 Curriculum exploration: {len(candidates)} samples "
              f"({num_rare} rare, {num_common} common, exploration={exploration_factor:.2f})")
        return candidates

    def _pure_random_sampling(self, num_samples: int, seed_offset: int) -> List[Tuple]:
        """whatis pure random sampling without any diversity considerations"""
        random.seed(seed_offset)

        all_examples = list(self.eval_loader)
        candidates = random.sample(all_examples, min(num_samples, len(all_examples)))

        # info still track for statistics
        for inputs, targets in candidates:
            tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]
            self._update_diversity_tracking(' '.join(tokens), tokens)

        print(f"  🎯 Pure random: {len(candidates)} samples")
        return candidates

    def _should_include_sample(self, token_pattern: str, tokens: List[str], epoch: int) -> bool:
        """whatis decide whether to include sample based on diversity and anti-repetition"""

        # info check recent repetition
        if token_pattern in self.recent_samples:
            recent_index = self.recent_samples.index(token_pattern)
            recency_penalty = (len(self.recent_samples) - recent_index) / len(self.recent_samples)
            if random.random() < recency_penalty * self.repetition_penalty:
                return False  # Skip recently seen patterns

        # info frequency-based inclusion (favor less seen patterns)
        frequency = self.pattern_frequency[token_pattern]
        if frequency > 0:
            frequency_penalty = min(0.9, frequency * 0.1)  # Higher frequency = higher penalty
            if random.random() < frequency_penalty:
                return False

        # info force inclusion of truly novel patterns
        if token_pattern not in self.seen_token_patterns:
            return True  # Always include novel patterns

        # info force inclusion of rare token combinations
        if self.force_rare_tokens and self._has_rare_tokens(tokens):
            if random.random() < self.rare_token_probability:
                return True

        # info default inclusion based on diversity metrics
        return random.random() < 0.7  # 70% inclusion rate for non-novel patterns

    def _create_token_strata(self) -> Dict[str, Dict[str, int]]:
        """whatis create token value strata for diverse sampling"""
        token_min = self.token_ranges['min']
        token_max = self.token_ranges['max']
        range_size = token_max - token_min + 1

        strata = {
            'low_values': {'min': token_min, 'max': token_min + range_size // 4},
            'mid_low_values': {'min': token_min + range_size // 4, 'max': token_min + range_size // 2},
            'mid_high_values': {'min': token_min + range_size // 2, 'max': token_min + 3 * range_size // 4},
            'high_values': {'min': token_min + 3 * range_size // 4, 'max': token_max},
            'edge_cases': {'min': token_min, 'max': token_min + 5},  # Very low values
            'boundary_cases': {'min': token_max - 5, 'max': token_max}  # Very high values
        }

        return strata

    def _tokens_in_range(self, tokens: List[int], token_range: Dict[str, int]) -> bool:
        """info check if tokens fall within specified range"""
        min_val, max_val = token_range['min'], token_range['max']
        return any(min_val <= token <= max_val for token in tokens)

    def _is_rare_pattern(self, token_pattern: str, tokens: List[str]) -> bool:
        """info determine if pattern is rare based on various criteria"""

        # info novel patterns are rare
        if token_pattern not in self.seen_token_patterns:
            return True

        # info infrequently seen patterns are rare
        if self.pattern_frequency[token_pattern] <= 2:
            return True

        # info patterns with unusual token combinations are rare
        if self._has_rare_tokens(tokens):
            return True

        # info patterns with specific mathematical properties
        numeric_tokens = [int(t) for t in tokens if t.isdigit()]
        if len(numeric_tokens) >= 3:
            # info check for unusual arithmetic patterns
            if self._has_unusual_arithmetic(numeric_tokens):
                return True

        return False

    def _has_rare_tokens(self, tokens: List[str]) -> bool:
        """whatis check if tokens contain rare values"""
        numeric_tokens = [int(t) for t in tokens if t.isdigit()]
        if not numeric_tokens:
            return False

        # info edge values are rare
        min_val, max_val = self.token_ranges['min'], self.token_ranges['max']
        edge_threshold = 5

        has_low_edge = any(t <= min_val + edge_threshold for t in numeric_tokens)
        has_high_edge = any(t >= max_val - edge_threshold for t in numeric_tokens)

        # info large value differences are rare
        if len(numeric_tokens) >= 2:
            max_diff = max(numeric_tokens) - min(numeric_tokens)
            has_large_spread = max_diff > (max_val - min_val) * 0.7
            return has_low_edge or has_high_edge or has_large_spread

        return has_low_edge or has_high_edge

    def _has_unusual_arithmetic(self, numeric_tokens: List[int]) -> bool:
        """info check for unusual arithmetic patterns"""
        if len(numeric_tokens) < 3:
            return False

        # info check for large products, edge case divisions, etc.
        for i in range(len(numeric_tokens) - 2):
            a, b, c = numeric_tokens[i], numeric_tokens[i + 1], numeric_tokens[i + 2]

            # info large products
            if a * b > 80 or b * c > 80:
                return True

            # info edge case modular arithmetic
            if b != 0 and (a % b == 0 or c % b == 0):
                if max(a, c) > 70:  # Large exact divisions
                    return True

        return False

    def _update_diversity_tracking(self, token_pattern: str, tokens: List[str]):
        """whatis update diversity tracking with new sample"""

        # info track pattern
        self.seen_token_patterns.add(token_pattern)
        self.pattern_frequency[token_pattern] += 1

        # info track recent samples (with size limit)
        self.recent_samples.append(token_pattern)
        if len(self.recent_samples) > self.max_recent_history:
            self.recent_samples.pop(0)

        # info track value combinations
        numeric_tokens = tuple(int(t) for t in tokens if t.isdigit())
        if numeric_tokens:
            self.seen_value_combinations.add(numeric_tokens)

        # info track operation types (inferred)
        if '+' in tokens:
            self.seen_operations.add('addition')
        if '-' in tokens:
            self.seen_operations.add('subtraction')
        if '*' in tokens:
            self.seen_operations.add('multiplication')
        if '=' in tokens:
            self.seen_operations.add('equation')

    def get_diversity_statistics(self) -> Dict[str, Any]:
        """whatis get comprehensive diversity statistics"""
        return {
            'unique_token_patterns': len(self.seen_token_patterns),
            'unique_value_combinations': len(self.seen_value_combinations),
            'operation_types_seen': list(self.seen_operations),
            'pattern_frequency_distribution': dict(Counter(self.pattern_frequency.values())),
            'most_common_patterns': sorted(self.pattern_frequency.items(),
                                           key=lambda x: x[1], reverse=True)[:10],
            'recent_pattern_count': len(set(self.recent_samples))
        }


# ============================================================================
# CIRCUIT CAPACITY ANALYSIS
# ============================================================================

def analyze_circuit_capacity_limits(model_architecture: Dict[str, int],
                                    discovered_circuits: Dict[str, Any]) -> Dict[str, Any]:
    """
    whatis analyze whether 40 circuits represents a capacity limit for the model architecture

    Args:
        model_architecture: {'num_layers': 2, 'num_heads': 4, 'embedding_dim': 128}
        discovered_circuits: Results from canonical circuit analysis
    """

    num_layers = model_architecture['num_layers']
    num_heads = model_architecture['num_heads']
    total_heads = num_layers * num_heads


    # info theoretical capacity analysis
    theoretical_analysis = {
        'total_attention_heads': total_heads,
        'theoretical_copy_circuits_per_head': estimate_copy_circuits_per_head(),
        'theoretical_induction_circuits_per_head': estimate_induction_circuits_per_head(),
        'theoretical_max_circuits': None,
        'specialization_factor': 0.7  # info heads don't usually learn all possible patterns
    }

    # info calculate theoretical maximum
    copy_max = total_heads * theoretical_analysis['theoretical_copy_circuits_per_head']
    induction_max = total_heads * theoretical_analysis['theoretical_induction_circuits_per_head']
    theoretical_max = (copy_max + induction_max) * theoretical_analysis['specialization_factor']
    theoretical_analysis['theoretical_max_circuits'] = int(theoretical_max)

    # info observed circuit analysis
    observed_circuits = discovered_circuits.get('registry_summary', {})
    observed_total = observed_circuits.get('total_canonical_circuits', 0)
    copy_circuits = len([c for c in discovered_circuits.get('stable_circuits', [])
                         if 'copy' in c.get('canonical_id', '')])
    induction_circuits = observed_total - copy_circuits

    observed_analysis = {
        'observed_total_circuits': observed_total,
        'observed_copy_circuits': copy_circuits,
        'observed_induction_circuits': induction_circuits,
        'circuits_per_head': observed_total / total_heads if total_heads > 0 else 0,
        'copy_circuits_per_head': copy_circuits / total_heads if total_heads > 0 else 0,
        'induction_circuits_per_head': induction_circuits / total_heads if total_heads > 0 else 0
    }

    # info capacity utilization analysis
    capacity_analysis = {
        'capacity_utilization': observed_total / theoretical_analysis['theoretical_max_circuits']
        if theoretical_analysis['theoretical_max_circuits'] > 0 else 0,
        'is_near_capacity': observed_total >= theoretical_analysis['theoretical_max_circuits'] * 0.8,
        'remaining_capacity': max(0, theoretical_analysis['theoretical_max_circuits'] - observed_total),
        'bottleneck_analysis': analyze_circuit_bottlenecks(model_architecture, observed_analysis)
    }

    # info head specialization analysis
    specialization_analysis = analyze_head_specialization(total_heads, observed_analysis)

    print(f"🏗️ Circuit capacity {num_layers}L × {num_heads}H = {total_heads} total heads "
        f" | Theoretical max: {theoretical_analysis['theoretical_max_circuits']} circuits "
        f" | Observed: {observed_total} circuits ({capacity_analysis['capacity_utilization']:.1%} capacity)"
        f" | Per head: {observed_analysis['circuits_per_head']:.1f} circuits")

    if capacity_analysis['is_near_capacity']:
        print(f"  🔴 Near capacity limit - architectural bottleneck likely")
    else:
        print(f"  🟡 Capacity available - sampling diversity may be the bottleneck")

    return {
        'theoretical_analysis': theoretical_analysis,
        'observed_analysis': observed_analysis,
        'capacity_analysis': capacity_analysis,
        'specialization_analysis': specialization_analysis,
        'conclusion': generate_capacity_conclusion(theoretical_analysis, observed_analysis, capacity_analysis)
    }


def estimate_copy_circuits_per_head() -> int:
    """whatis estimate how many copy circuits a single head can learn"""
    # Each head can learn copy operations at different relative offsets
    # Typical range: -4 to +4 relative positions, but heads specialize
    # Empirical observation: heads usually learn 2-4 copy patterns each
    return 3


def estimate_induction_circuits_per_head() -> int:
    """whatis estimate how many induction circuits a single head can learn"""
    # Induction heads learn patterns at different distances
    # Typical range: distance 1-5, but more complex than copy
    # Empirical observation: induction heads learn 1-3 patterns each
    return 2


def analyze_circuit_bottlenecks(architecture: Dict, observed: Dict) -> List[str]:
    """whatis identify potential bottlenecks in circuit formation"""
    bottlenecks = []

    total_heads = architecture['num_layers'] * architecture['num_heads']
    circuits_per_head = observed['circuits_per_head']

    # info head saturation bottleneck
    if circuits_per_head >= 4.5:  # High circuits per head
        bottlenecks.append("head_saturation")

    # info layer bottleneck (shallow networks)
    if architecture['num_layers'] <= 2 and observed['observed_total_circuits'] >= 30:
        bottlenecks.append("shallow_architecture")

    # info specialization bottleneck
    copy_per_head = observed['copy_circuits_per_head']
    induction_per_head = observed['induction_circuits_per_head']

    if copy_per_head >= 3.0:
        bottlenecks.append("copy_head_saturation")
    if induction_per_head >= 2.5:
        bottlenecks.append("induction_head_saturation")

    return bottlenecks


def analyze_head_specialization(total_heads: int, observed: Dict) -> Dict[str, Any]:
    """whatis analyze head specialization patterns"""

    copy_circuits = observed['observed_copy_circuits']
    induction_circuits = observed['observed_induction_circuits']
    total_circuits = observed['observed_total_circuits']

    # info estimate number of specialized heads
    estimated_copy_heads = min(total_heads, copy_circuits // 2)  # Assume 2+ circuits per specialized head
    estimated_induction_heads = min(total_heads, induction_circuits // 1)  # Assume 1+ circuits per specialized head
    estimated_specialized_heads = estimated_copy_heads + estimated_induction_heads

    specialization_analysis = {
        'estimated_copy_heads': estimated_copy_heads,
        'estimated_induction_heads': estimated_induction_heads,
        'estimated_specialized_heads': min(total_heads, estimated_specialized_heads),
        'estimated_unused_heads': max(0, total_heads - estimated_specialized_heads),
        'specialization_efficiency': estimated_specialized_heads / total_heads if total_heads > 0 else 0
    }

    return specialization_analysis


def generate_capacity_conclusion(theoretical: Dict, observed: Dict, capacity: Dict) -> str:
    """whatis generate conclusion about circuit capacity limits"""

    if capacity['is_near_capacity']:
        return (f"ARCHITECTURAL BOTTLENECK: {observed['observed_total_circuits']} circuits "
                f"approaches theoretical limit of {theoretical['theoretical_max_circuits']}. "
                f"Model architecture (2L×4H) is likely constraining circuit discovery.")

    elif capacity['capacity_utilization'] < 0.5:
        return (f"SAMPLING BOTTLENECK: Only {capacity['capacity_utilization']:.1%} of theoretical "
                f"capacity used. Sampling diversity likely limiting circuit discovery.")

    else:
        return (f"MIXED BOTTLENECK: {capacity['capacity_utilization']:.1%} capacity utilization "
                f"suggests both architectural and sampling constraints.")


# ============================================================================
# ENHANCED SAMPLING INTEGRATION
# ============================================================================

def run_canonical_circuit_analysis_with_enhanced_sampling(
        canonical_detector, eval_loader, epoch: int, total_epochs: int, accuracy: float,
        sampling_strategy: str = "forced_diversity",
        logger=None, **kwargs) -> Dict[str, Any]:
    """
    whatis enhanced canonical circuit analysis with aggressive diversity sampling
    """

    # info create enhanced sampler
    sampling_config = {
        'base_budget': kwargs.get('num_samples', 12),  # Increased budget
        'max_recent_history': 100,
        'repetition_penalty': 0.9,
        'token_ranges': {'min': 0, 'max': 97},
        'force_rare_tokens': True,
        'rare_token_probability': 0.4
    }

    enhanced_sampler = AggressiveDiversitySampler(eval_loader, sampling_config)

    # info sample with enhanced diversity
    sampled_examples = enhanced_sampler.sample_examples(
        epoch=epoch,
        total_epochs=total_epochs,
        strategy=sampling_strategy,
        num_samples=sampling_config['base_budget'],
        seed_offset=epoch % 1000
    )

    # info get diversity statistics
    diversity_stats = enhanced_sampler.get_diversity_statistics()

    logger.info(f"🎲 Enhanced sampling @ epoch {epoch}: {len(sampled_examples)} examples, "
                f"{diversity_stats['unique_token_patterns']} unique patterns seen")

    # Continue with existing canonical analysis logic...
    # (Use the sampled_examples in the existing circuit analysis pipeline)

    # info return enhanced results with diversity metrics
    return {
        'sampling_diversity': diversity_stats,
        'enhanced_sampling_strategy': sampling_strategy,
        'samples_analyzed': len(sampled_examples),
        # ... existing canonical analysis results
    }


# ============================================================================
# RECOMMENDATIONS
# ============================================================================

def get_sampling_strategy_recommendations(current_results: Dict[str, Any]) -> List[str]:
    """whatis get recommendations for improving sampling diversity"""

    recommendations = []

    total_circuits = current_results.get('registry_summary', {}).get('total_canonical_circuits', 0)

    # info strategy recommendations based on current state
    if total_circuits >= 35:
        recommendations.extend([
            "🎯 Use 'forced_diversity' strategy to explore rare token combinations",
            "🎲 Increase sampling budget to 15-20 examples per analysis",
            "🔄 Implement anti-repetition with 90% penalty for recent patterns",
            "📊 Add stratified sampling across token value ranges",
            "🚀 Use 'curriculum_exploration' to gradually increase pattern complexity"
        ])

    # info architecture recommendations
    recommendations.extend([
        "🏗️ Consider analyzing larger models (3-4 layers) to verify capacity limits",
        "🔍 Implement circuit ablation studies to confirm discovered circuits",
        "📈 Track diversity metrics to monitor sampling effectiveness",
        "⚡ Use pure random sampling occasionally to establish baseline diversity"
    ])

    return recommendations