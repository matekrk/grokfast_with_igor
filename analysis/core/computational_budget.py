# analysis/core/computational_budget.py
import time
from typing import Dict, DefaultDict
from collections import defaultdict, deque


class ComputationalBudget:
    """Manage computational resources across circuit detection methods"""

    def __init__(self, max_time_per_epoch: float = 30.0):
        """
        Initialize computational budget manager

        Args:
            max_time_per_epoch: Maximum time (seconds) to spend on analysis per epoch
        """
        self.max_time_per_epoch = max_time_per_epoch

        # Track execution times for each method (rolling window)
        self.method_times: DefaultDict[str, deque] = defaultdict(lambda: deque(maxlen=10))

        # Priority allocation for different analysis methods
        self.method_priorities = {
            "adaptive_token_detection": 1.0,  # Highest priority
            "token_detection": 1.0,  # Highest priority
            "component_detection": 0.8,  # High priority
            "subspace_detection": 0.6,  # Medium priority
            "cross_level_detection": 0.4,  # Lower priority
            "validation": 0.3,  # Lowest priority (expensive)
            "visualization": 0.2  # Optional
        }

        # Track current epoch usage
        self.current_epoch_start: float = 0.0
        self.current_epoch_used: float = 0.0
        self.method_usage_current_epoch: Dict[str, float] = {}

    def start_epoch(self):
        """Mark the start of a new epoch"""
        self.current_epoch_start = time.time()
        self.current_epoch_used = 0.0
        self.method_usage_current_epoch = {}

    def record_execution_time(self, method: str, execution_time: float):
        """Record execution time for a method"""
        self.method_times[method].append(execution_time)
        self.current_epoch_used += execution_time

        if method not in self.method_usage_current_epoch:
            self.method_usage_current_epoch[method] = 0.0
        self.method_usage_current_epoch[method] += execution_time

    def get_time_budget(self, method: str) -> float:
        """Get allocated time budget for a method"""
        priority = self.method_priorities.get(method, 0.5)
        return self.max_time_per_epoch * priority

    def get_remaining_budget(self) -> float:
        """Get remaining time budget for current epoch"""
        return max(0.0, self.max_time_per_epoch - self.current_epoch_used)

    def can_run_method(self, method: str, estimated_time: float = None) -> bool:
        """Check if we have budget to run a method"""
        if estimated_time is None:
            # Use average of recent executions as estimate
            if method in self.method_times and len(self.method_times[method]) > 0:
                estimated_time = sum(self.method_times[method]) / len(self.method_times[method])
            else:
                # Default estimates based on method type
                default_estimates = {
                    "adaptive_token_detection": 5.0,
                    "token_detection": 5.0,
                    "component_detection": 8.0,
                    "subspace_detection": 6.0,
                    "cross_level_detection": 10.0,
                    "validation": 15.0,
                    "visualization": 3.0
                }
                estimated_time = default_estimates.get(method, 5.0)

        # Check if we have enough budget remaining
        remaining_budget = self.get_remaining_budget()
        method_budget = self.get_time_budget(method)

        # Allow method if either condition is met:
        # 1. It fits in remaining total budget, OR
        # 2. It fits in method-specific budget and we haven't used much time yet
        if not( (estimated_time <= remaining_budget or
                (estimated_time <= method_budget and self.current_epoch_used < self.max_time_per_epoch * 0.5))):
            pass
            # info not enough time

        return (estimated_time <= remaining_budget or
                (estimated_time <= method_budget and self.current_epoch_used < self.max_time_per_epoch * 0.5))

    def suggest_sampling_rate(self, method: str, base_samples: int) -> int:
        """Suggest sampling rate based on computational budget"""
        method_budget = self.get_time_budget(method)

        if method in self.method_times and len(self.method_times[method]) > 0:
            # Estimate time per sample based on recent history
            recent_times = list(self.method_times[method])
            avg_total_time = sum(recent_times) / len(recent_times)
            avg_time_per_sample = avg_total_time / base_samples if base_samples > 0 else 1.0

            # Calculate how many samples we can afford
            max_samples = int(method_budget / (avg_time_per_sample + 1e-6))
            return min(base_samples, max(1, max_samples))

        return base_samples  # No history yet, use full base samples

    def get_usage_summary(self) -> Dict[str, float]:
        """Get summary of current epoch usage"""
        return {
            "total_time": self.current_epoch_used,
            "remaining_budget": self.get_remaining_budget(),
            "budget_utilization": self.current_epoch_used / self.max_time_per_epoch,
            "method_breakdown": dict(self.method_usage_current_epoch)
        }

    def is_over_budget(self) -> bool:
        """Check if we've exceeded the time budget"""
        return self.current_epoch_used > self.max_time_per_epoch * 1.1  # 10% tolerance