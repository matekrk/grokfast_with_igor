# circuit_analysis_pipeline.py
# from typing import Dict, List
from typing import Optional
# from typing import Any
from typing import Union
from pathlib import Path

from analysis.core.circuit_registry import EnhancedCircuitRegistry as CircuitRegistry
from analysis.analyzers.token_circuit_discovery import TokenCircuitDiscovery
from analysis.visualization.token_visualizer import TokenCircuitVisualizer


class CircuitAnalysisPipeline:
    """Manages multi-stage circuit analysis"""

    def __init__(self, model, save_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the circuit analysis pipeline

        Args:
            model: The transformer model to analyze
            save_dir: Directory to save analysis results
        """
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        self.model = model

        # Initialize the circuit registry
        self.registry = CircuitRegistry(self.save_dir / "circuit_registry" if self.save_dir else None)

        # Initialize analyzers
        self.token_discovery = TokenCircuitDiscovery(
            model=model,
            save_dir=self.save_dir / "token_circuits" if self.save_dir else None,
            circuit_registry=self.registry
        )

        # Initialize visualizers
        self.token_visualizer = TokenCircuitVisualizer(
            save_dir=self.save_dir / "visualizations/tokens" if self.save_dir else None
        )

        # Stage results storage
        self.results = {}

    def run_token_discovery(self, inputs, targets=None, epoch=None, **params):
        """
        Run token circuit discovery

        Args:
            inputs: Input tensor
            targets: Optional target tensor
            epoch: Current training epoch
            **params: Additional parameters for token discovery

        Returns:
            Analysis results
        """
        results = self.token_discovery.analyze_token_relationships(
            inputs=inputs,
            targets=targets,
            epoch=epoch,
            **params
        )

        # Store results
        stage_key = f"token_discovery_{epoch}" if epoch is not None else "token_discovery"
        self.results[stage_key] = results

        # Visualize results
        if self.save_dir:
            # Visualize token attribution
            self.token_visualizer.visualize_token_attribution(
                attribution_map=results["token_attribution"],
                tokens=results["tokens"],
                title=f"Token Attribution Map (Epoch {epoch})" if epoch is not None else "Token Attribution Map",
                show=False,
                save_path=f"token_attribution_epoch_{epoch}.png" if epoch is not None else "token_attribution.png"
            )

            # Visualize discovered circuits
            for i, circuit in enumerate(results["circuits"]):
                self.token_visualizer.visualize_token_circuit(
                    circuit=circuit,
                    title=f"Token Circuit {i + 1} (Epoch {epoch})" if epoch is not None else f"Token Circuit {i + 1}",
                    show=False,
                    save_path=f"token_circuit_{i + 1}_epoch_{epoch}.png" if epoch is not None else f"token_circuit_{i + 1}.png"
                )

                self.token_visualizer.visualize_token_flow(
                    circuit=circuit,
                    tokens=results["tokens"],
                    title=f"Token Flow {i + 1} (Epoch {epoch})" if epoch is not None else f"Token Flow {i + 1}",
                    show=False,
                    save_path=f"token_flow_{i + 1}_epoch_{epoch}.png" if epoch is not None else f"token_flow_{i + 1}.png"
                )

        return results

    def run_pipeline(self, inputs, targets=None, epoch=None, stages=None):
        """
        Run multiple analysis stages

        Args:
            inputs: Input tensor
            targets: Optional target tensor
            epoch: Current training epoch
            stages: List of stages to run (defaults to all)

        Returns:
            Dict with results from all stages
        """
        all_results = {}

        # Default to token discovery
        if stages is None:
            stages = ["token_discovery"]

        # Run each requested stage
        for stage in stages:
            if stage == "token_discovery":
                stage_results = self.run_token_discovery(inputs, targets, epoch)
                all_results[stage] = stage_results
            # Add other stages as they're implemented

        return all_results

    def save_results(self, filepath: Optional[Path] = None):
        """Save analysis results and registry"""
        # Save registry
        self.registry.save()

        # Save token circuits
        self.token_discovery.save_token_circuits()

        # Additional saving logic could be added here

    def load_results(self, filepath: Optional[Path] = None):
        """Load previously saved analysis results and registry"""
        # Load registry
        self.registry.load()

        # Load token circuits
        self.token_discovery.load_token_circuits()

        # Additional loading logic could be added here