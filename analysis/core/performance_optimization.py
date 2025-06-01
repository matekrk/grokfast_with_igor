# analysis/core/performance_optimization.py
# import time
from typing import Dict, Any  #, List
from collections import deque
# import numpy as np


class PerformanceOptimizer:
    """Optimize circuit analysis performance based on training dynamics"""

    def __init__(self, budget_manager):
        self.budget = budget_manager
        self.performance_history = deque(maxlen=100)
        self.analysis_effectiveness = {}  # method -> effectiveness score

    def optimize_analysis_schedule(self, epoch: int, model_accuracy: float,
                                   recent_results: Dict[str, Any]) -> Dict[str, bool]:
        """Decide which analysis methods to run based on effectiveness"""

        # Calculate method effectiveness
        self._update_effectiveness_scores(recent_results)

        # Get remaining budget
        remaining_budget = self.budget.get_remaining_budget()

        # Prioritize methods by effectiveness and cost
        method_priorities = self._calculate_method_priorities()

        # Decide which methods to run
        methods_to_run = {}
        estimated_total_time = 0

        for method, priority in sorted(method_priorities.items(), key=lambda x: x[1], reverse=True):
            estimated_time = self._estimate_method_time(method)

            if estimated_total_time + estimated_time <= remaining_budget:
                methods_to_run[method] = True
                estimated_total_time += estimated_time
            else:
                methods_to_run[method] = False

        return methods_to_run

    def _update_effectiveness_scores(self, results: Dict[str, Any]):
        """Update effectiveness scores based on results quality"""

        for method, result in results.items():
            if method.endswith("_results"):
                method_name = method.replace("_results", "")

                # Calculate effectiveness based on result quality
                if isinstance(result, dict):
                    circuits_found = len(result.get("stable_circuits", []))
                    detection_quality = result.get("detection_summary", {}).get("stability_rate", 0)

                    # Higher effectiveness for methods that find stable, high-quality circuits
                    effectiveness = circuits_found * detection_quality

                    self.analysis_effectiveness[method_name] = effectiveness

    def _calculate_method_priorities(self) -> Dict[str, float]:
        """Calculate priority scores for each analysis method"""
        priorities = {}

        base_priorities = {
            "adaptive_token_detection": 1.0,
            "token_discovery": 0.8,
            "component_analysis": 0.7,
            "validation": 0.3
        }

        for method, base_priority in base_priorities.items():
            effectiveness = self.analysis_effectiveness.get(method, 0.5)
            # Combine base priority with effectiveness
            priorities[method] = base_priority * (0.5 + 0.5 * effectiveness)

        return priorities

    def _estimate_method_time(self, method: str) -> float:
        """Estimate execution time for a method"""
        return self.budget.get_time_budget(method) * 0.8  # Conservative estimate
