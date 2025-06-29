# analysis/sampling/fast_subset_sampler.py
"""
Fast Circuit Discovery Sampling using PyTorch SubsetRandomSampler

Simple, efficient approach that leverages PyTorch's native sampling capabilities
for circuit discovery without complex diversity strategies.
"""
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from analysis.core import CanonicalCircuitRegistry


class FastCircuitSampler:
    """
    Fast sampler using PyTorch SubsetRandomSampler for circuit discovery

    Optimized for speed over complex diversity - lets PyTorch handle the sampling
    """

    def __init__(self, eval_loader: DataLoader, config: Dict[str, Any] = {}):
        self.dataset = eval_loader.dataset
        self.batch_size = eval_loader.batch_size
        self.num_workers = eval_loader.num_workers
        self.device = next(iter(eval_loader))[0].device if len(eval_loader) > 0 else 'cpu'

        # Configuration
        self.base_budget = config.get('base_budget', 8)
        self.max_cache_size = config.get('max_cache_size', 50)

        # Simple tracking for basic diversity
        self.seen_patterns = set()

        # print(f"🚀 FastCircuitSampler initialized: {len(self.dataset)} examples available")

    def sample_examples(self, epoch: int, total_epochs: int,
                        strategy: str = "fast_random",
                        num_samples: int = None,
                        seed_offset: int = 0) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Fast sampling using SubsetRandomSampler

        Args:
            epoch: Current epoch
            total_epochs: Total training epochs
            strategy: Sampling strategy (simplified to just 'fast_random')
            num_samples: Number of samples to get
            seed_offset: Seed offset for reproducibility

        Returns:
            List of (input, target) tuples ready for circuit analysis
        """
        if num_samples is None:
            num_samples = self.base_budget

        # Ensure we don't sample more than available
        num_samples = min(num_samples, len(self.dataset))

        # Set seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(epoch * 1000 + seed_offset)

        # Create random subset indices
        indices = torch.randperm(len(self.dataset), generator=generator)[:num_samples].tolist()

        # Create SubsetRandomSampler
        subset_sampler = SubsetRandomSampler(indices)

        # Create temporary DataLoader with our subset
        sample_loader = DataLoader(
            self.dataset,
            batch_size=1,  # Get individual examples
            sampler=subset_sampler,
            num_workers=0,  # Avoid multiprocessing overhead for small samples
            pin_memory=False  # Not needed for small samples
        )

        # Collect samples
        samples = []
        for inputs, targets in sample_loader:
            samples.append((inputs, targets))

            # Optional: Track basic diversity
            tokens = [str(int(x)) for x in inputs[0].cpu().numpy()]
            pattern = ' '.join(tokens)
            self.seen_patterns.add(pattern)

        # print(f"  ⚡ Fast sampling: {len(samples)} examples in epoch {epoch}")
        return samples

    def get_diversity_stats(self) -> Dict[str, Any]:
        """Get basic diversity statistics"""
        return {
            'unique_patterns_seen': len(self.seen_patterns),
            'total_samples_processed': len(self.seen_patterns)  # Approximate
        }


# ============================================================================
# INTEGRATION WITH CANONICAL CIRCUIT ANALYSIS
# ============================================================================

# ============================================================================
# SIMPLIFIED TRAINING INTEGRATION
# ============================================================================
'''
def update_training_loop_with_fast_sampling(
        # Add this to your training loop
        canonical_detector, eval_loader, epoch, total_epochs, accuracy,
        analyze_interval=2, num_samples=8, logger=None):
    """
    Simple integration for training loops - just replace the sampling call
    """

    if epoch % analyze_interval == 0:
        canonical_results = run_canonical_circuit_analysis_with_fast_sampling(
            canonical_detector=canonical_detector,
            eval_loader=eval_loader,
            epoch=epoch,
            total_epochs=total_epochs,
            accuracy=accuracy,
            num_samples=num_samples,
            logger=logger
        )

        return canonical_results

    return None
'''

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Example of how to use in your training loop
    """

    # In your training loop, replace:
    # OLD:
    # canonical_results = run_canonical_circuit_analysis_with_sampling(...)

    # NEW:
    # canonical_results = run_canonical_circuit_analysis_with_fast_sampling(
    #     canonical_detector=canonical_detector,
    #     eval_loader=eval_loader,
    #     epoch=epoch,
    #     total_epochs=epochs,
    #     accuracy=current_accuracy,
    #     num_samples=12,  # Adjust as needed
    #     logger=logger
    # )

    pass