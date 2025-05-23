# Example script showing how to use the token circuit discovery system
import torch
from pathlib import Path

# Import your model implementation
from your_model_module import YourModel

# Import the circuit analysis pipeline
from circuit_analysis_pipeline import CircuitAnalysisPipeline

# Initialize your model
model = YourModel(...)

# Initialize the analysis pipeline
save_dir = Path("./analysis_results")
pipeline = CircuitAnalysisPipeline(model=model, save_dir=save_dir)

# Generate or load some input data
inputs = torch.tensor(...)  # Your input data
targets = torch.tensor(...)  # Optional target data

# Run token circuit discovery
token_results = pipeline.run_token_discovery(
    inputs=inputs,
    targets=targets,
    epoch=100  # Current training epoch
)

# Print discovered circuits
print(f"Discovered {len(token_results['circuits'])} token circuits:")
for circuit in token_results['circuits']:
    print(f"- {circuit.id}: {circuit.metadata['operation_type']} operation")
    print(f"  Attribution: {circuit.attribution:.4f}")

# Run full pipeline
all_results = pipeline.run_pipeline(
    inputs=inputs,
    targets=targets,
    epoch=100
)

# Save results
pipeline.save_results()