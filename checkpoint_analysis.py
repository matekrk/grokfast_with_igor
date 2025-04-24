import argparse
import torch
from pathlib import Path

from analysis.models.analysis_transformer import Decoder
from analysis.visualization import visualize_model_analysis
from analysis.models.modular_data import create_modular_dataloaders


def analyze_model_checkpoints(checkpoint_dir, output_dir, modulus=97, op='multiply', epochs=None):
    """
    Analyze saved model checkpoints and generate comprehensive visualizations.

    Parameters:
    -----------
    checkpoint_dir : str
        Directory containing checkpoint files
    output_dir : str
        Directory to save analysis results
    modulus : int
        Modulus value used for the dataset
    op : str
        Operation used in modular arithmetic
    epochs : list, optional
        Specific epochs to analyze, if None analyze all checkpoints
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Find checkpoint files
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_path.glob("checkpoint_step_*.pt"))

    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))

    # Filter for specific epochs if requested
    if epochs is not None:
        checkpoint_files = [f for f in checkpoint_files
                            if int(f.stem.split('_')[-1]) in epochs]

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    # Load first checkpoint to get model configuration
    print(f"Loading checkpoint {checkpoint_files[0]}")
    checkpoint = torch.load(checkpoint_files[0])

    # Extract model configuration from checkpoint
    model_config = checkpoint.get('model_config', {})
    if not model_config:
        # Try to infer from the state dict
        state_dict = checkpoint['model_state_dict']
        embedding_dim = state_dict['token_embeddings.weight'].size(1)
        num_layers = sum(1 for k in state_dict if 'layers' in k and 'ln_1.weight' in k)
        num_heads = model_config.get('num_heads', 4)  # Default if not available
        num_tokens = state_dict['token_embeddings.weight'].size(0)
        seq_len = state_dict['position_embeddings.weight'].size(0)
    else:
        embedding_dim = model_config.get('dim', 128)
        num_layers = model_config.get('num_layers', 2)
        num_heads = model_config.get('num_heads', 4)
        num_tokens = model_config.get('num_tokens', 97)
        seq_len = model_config.get('seq_len', 5)

    # Recreate dataloaders
    try:
        train_loader, eval_loader, vocab_size, dataset_indices = create_modular_dataloaders(
            modulus=modulus,
            op=op,
            train_ratio=0.5,  # Default value
            batch_size=32,  # Small batch is fine for analysis
            sequence_format=True,
            seed=42
        )
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = Decoder(
        dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_tokens=num_tokens,
        seq_len=seq_len,
        id=checkpoint_path.parent.name
    )
    model.to(device)

    # Process each checkpoint
    for checkpoint_file in checkpoint_files:
        epoch = int(checkpoint_file.stem.split('_')[-1])
        print(f"Analyzing epoch {epoch}...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])

        # If the checkpoint has stats, add them to the model's logger
        if 'stats' in checkpoint:
            stats = checkpoint['stats']
            if 'train_loss' in stats and 'train_accuracy' in stats:
                for i, (loss, acc, ep) in enumerate(zip(stats['train_loss'], stats['train_accuracy'], stats['epoch'])):
                    model.log_stats('training', {'loss': loss, 'accuracy': acc, 'epoch': ep})

            if 'val_loss' in stats and 'val_accuracy' in stats:
                for i, (loss, acc, ep) in enumerate(zip(stats['val_loss'], stats['val_accuracy'], stats['epoch'])):
                    model.log_stats('evaluation', {'loss': loss, 'accuracy': acc, 'epoch': ep})

        # Run analyses to populate logger
        model.analyze_head_attribution(eval_loader)
        model.compute_attention_entropy(eval_loader)

        # Create visualization
        save_path = output_path / f"analysis_epoch_{epoch}.png"
        try:
            visualize_model_analysis(
                model=model,
                epoch=epoch,
                eval_loader=eval_loader,
                save_path=save_path,
                title_prefix=f"Checkpoint Analysis"
            )
            print(f"Saved visualization to {save_path}")
        except Exception as e:
            print(f"Error creating visualization: {e}")

    print("Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory with checkpoint files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save analysis results")
    parser.add_argument("--modulus", type=int, default=97, help="Modulus value for dataset")
    parser.add_argument("--operation", type=str, default="multiply", choices=["add", "subtract", "multiply", "divide"],
                        help="Operation for modular arithmetic")
    parser.add_argument("--epochs", type=int, nargs="+", help="Specific epochs to analyze")

    args = parser.parse_args()

    analyze_model_checkpoints(
        args.checkpoint_dir,
        args.output_dir,
        args.modulus,
        args.operation,
        args.epochs
    )