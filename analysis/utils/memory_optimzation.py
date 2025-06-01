# analysis/utils/memory_optimization.py
import torch
import gc
from typing import Dict, Any


class CircuitAnalysisMemoryManager:
    """Manage memory usage during circuit analysis"""

    def __init__(self):
        self.memory_checkpoints = []

    def start_analysis_block(self, block_name: str):
        """Start a memory-managed analysis block"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.memory_checkpoints.append({
            "block": block_name,
            "start_memory": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        })

    def end_analysis_block(self, block_name: str):
        """End analysis block and clean up"""
        # Force garbage collection
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log memory usage
        if self.memory_checkpoints:
            checkpoint = self.memory_checkpoints[-1]
            if checkpoint["block"] == block_name:
                end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_used = end_memory - checkpoint["start_memory"]

                if memory_used > 100 * 1024 * 1024:  # > 100MB
                    print(f"⚠️  High memory usage in {block_name}: {memory_used / (1024 * 1024):.1f}MB")

    def cleanup_circuit_data(self, circuit_data: Dict[str, Any]):
        """Clean up circuit data to free memory"""
        for key, value in circuit_data.items():
            if isinstance(value, torch.Tensor):
                # Move large tensors to CPU to free GPU memory
                if value.is_cuda and value.numel() > 1000:
                    circuit_data[key] = value.cpu()
