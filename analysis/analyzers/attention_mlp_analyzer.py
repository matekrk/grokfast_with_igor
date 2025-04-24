import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict


class AttentionMLPAnalyzer:
    """
    Analyzer for studying the relationship between attention heads and MLPs in transformers
    during grokking phase transitions.
    """

    def __init__(self, model, eval_loader, device=None, save_dir=None):
        """
        Initialize the analyzer.

        Args:
            model: The transformer model to analyze
            eval_loader: DataLoader for evaluation data
            device: Computation device
            save_dir: Directory to save analysis results
        """
        self.model = model
        self.eval_loader = eval_loader
        self.device = device or next(model.parameters()).device
        self.save_dir = Path(save_dir) if save_dir else Path("analysis_results")
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # info heatmap min/max values
        self.heatmap_min = 0.0
        self.heatmap_max = 0.65

    def collect_activations(self, num_batches=3):
        """
        Collect activations from attention heads and MLPs.

        Args:
            num_batches: Number of batches to process

        Returns:
            dict: Collected activations
        """
        model = self.model
        activations = {
            'attn_inputs': defaultdict(list),  # Inputs to attention blocks
            'attn_outputs': defaultdict(list),  # Outputs from attention blocks
            'mlp_inputs': defaultdict(list),  # Inputs to MLP blocks
            'mlp_outputs': defaultdict(list),  # Outputs from MLP blocks
            'layer_outputs': defaultdict(list),  # Outputs from full layers
            'attn_patterns': defaultdict(list),  # Attention patterns
            'inputs': [],  # Original inputs
            'targets': []  # Targets
        }

        # Set up hooks
        hooks = []

        # Attention input hook
        def get_attn_input_hook(layer_idx):
            def hook(module, input, output):
                # input is a tuple, we usually want the first element
                if isinstance(input, tuple) and len(input) > 0:
                    # Ensure it's a tensor we can detach
                    if torch.is_tensor(input[0]):
                        activations['attn_inputs'][layer_idx].append(input[0].detach().cpu())
                    else:
                        print(f"Warning: input[0] is not a tensor in layer {layer_idx} attention input")

            return hook

        # Attention output hook
        def get_attn_output_hook(layer_idx):
            def hook(module, input, output):
                # Sometimes output can be a tuple in MultiheadAttention
                if isinstance(output, tuple):
                    # MultiheadAttention returns (attn_output, attn_weights)
                    if len(output) > 0 and torch.is_tensor(output[0]):
                        activations['attn_outputs'][layer_idx].append(output[0].detach().cpu())
                elif torch.is_tensor(output):
                    activations['attn_outputs'][layer_idx].append(output.detach().cpu())
                else:
                    print(f"Warning: Unexpected output type in layer {layer_idx} attention output")

            return hook

        # MLP input hook
        def get_mlp_input_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    if torch.is_tensor(input[0]):
                        activations['mlp_inputs'][layer_idx].append(input[0].detach().cpu())

            return hook

        # MLP output hook
        def get_mlp_output_hook(layer_idx):
            def hook(module, input, output):
                if torch.is_tensor(output):
                    activations['mlp_outputs'][layer_idx].append(output.detach().cpu())
                elif isinstance(output, tuple) and len(output) > 0:
                    if torch.is_tensor(output[0]):
                        activations['mlp_outputs'][layer_idx].append(output[0].detach().cpu())

            return hook

        # Layer output hook
        def get_layer_output_hook(layer_idx):
            def hook(module, input, output):
                if torch.is_tensor(output):
                    activations['layer_outputs'][layer_idx].append(output.detach().cpu())
                elif isinstance(output, tuple) and len(output) > 0:
                    if torch.is_tensor(output[0]):
                        activations['layer_outputs'][layer_idx].append(output[0].detach().cpu())

            return hook

        # Register hooks
        for i, layer in enumerate(model.layers):
            # Layernorm hooks
            hooks.append(layer.ln_1.register_forward_hook(get_attn_input_hook(i)))
            hooks.append(layer.ln_2.register_forward_hook(get_mlp_input_hook(i)))

            # Attention and MLP hooks
            hooks.append(layer.attn.register_forward_hook(get_attn_output_hook(i)))
            hooks.append(layer.mlp.register_forward_hook(get_mlp_output_hook(i)))

            # Layer output hook
            hooks.append(layer.register_forward_hook(get_layer_output_hook(i)))

        # Run forward pass
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.eval_loader):
                if batch_idx >= num_batches:
                    break

                # Store inputs and targets
                activations['inputs'].append(inputs.cpu())
                activations['targets'].append(targets.cpu())

                # Forward pass
                inputs = inputs.to(self.device)
                _ = model(inputs, store_attention=True)

                # Store attention patterns
                patterns = model.get_attention_patterns()
                for pattern_name, pattern in patterns.items():
                    activations['attn_patterns'][pattern_name].append(pattern.cpu())

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Concatenate batches
        for key in activations:
            if key in ['inputs', 'targets']:
                activations[key] = torch.cat(activations[key], dim=0)
            elif key == 'attn_patterns':
                for pattern_name in activations[key]:
                    activations[key][pattern_name] = torch.cat(activations[key][pattern_name], dim=0)
            else:
                for layer_idx in activations[key]:
                    activations[key][layer_idx] = torch.cat(activations[key][layer_idx], dim=0)

        return activations

    def analyze_activation_flow(self, activations=None):
        """
        Analyze how information flows from attention heads to MLP components.

        Args:
            activations: Pre-collected activations or None to collect new ones

        Returns:
            dict: Flow analysis results
        """
        if activations is None:
            activations = self.collect_activations()

        model = self.model
        results = {}

        # For each layer, analyze the flow from attention to MLP
        for layer_idx in range(model.num_layers):
            # Skip if we don't have data for this layer
            if (layer_idx not in activations['attn_outputs'] or
                    layer_idx not in activations['mlp_inputs']):
                continue

            # Get attention outputs and MLP inputs for this layer
            attn_out = activations['attn_outputs'][layer_idx]
            mlp_in = activations['mlp_inputs'][layer_idx]

            # Calculate correlation between attention output and MLP input
            # We need to reshape to 2D: [batch_size * seq_len, hidden_dim]
            attn_out_flat = attn_out.reshape(-1, attn_out.size(-1))
            mlp_in_flat = mlp_in.reshape(-1, mlp_in.size(-1))

            # Compute correlation for each dimension
            dim_correlations = []
            for dim_idx in range(min(attn_out_flat.size(1), 20)):  # Limit to 20 dims for efficiency
                corr, _ = pearsonr(
                    attn_out_flat[:, dim_idx].numpy(),
                    mlp_in_flat[:, dim_idx].numpy()
                )
                dim_correlations.append(corr)

            # Compute overall correlation matrix (sample up to 100 dimensions)
            sample_dims = min(attn_out_flat.size(1), 100)
            corr_matrix = np.corrcoef(
                attn_out_flat[:, :sample_dims].numpy().T,
                mlp_in_flat[:, :sample_dims].numpy().T
            )

            # Extract the cross-correlation block
            cross_corr = corr_matrix[:sample_dims, sample_dims:]

            # Store results for this layer
            results[f'layer_{layer_idx}'] = {
                'dimension_correlations': dim_correlations,
                'mean_correlation': np.mean(dim_correlations),
                'cross_correlation_matrix': cross_corr,
                'mean_cross_correlation': np.mean(np.abs(cross_corr))
            }

        return results

    def analyze_attention_mlp_specialization(self, activations=None, pca_dims=20):
        """
        Analyze how attention heads and MLP components specialize.

        Args:
            activations: Pre-collected activations or None to collect new ones
            pca_dims: Number of PCA dimensions to use

        Returns:
            dict: Specialization analysis results
        """
        if activations is None:
            activations = self.collect_activations()

        model = self.model
        results = {}

        # Extract per-head attention outputs
        head_outputs = {}

        for layer_idx in range(model.num_layers):
            # Skip if we don't have data for this layer
            if layer_idx not in activations['attn_outputs']:
                continue

            # Get attention outputs
            attn_out = activations['attn_outputs'][layer_idx]
            batch_size, seq_len, hidden_dim = attn_out.shape
            head_dim = hidden_dim // model.num_heads

            # Reshape to separate heads: [batch_size, seq_len, num_heads, head_dim]
            reshaped = attn_out.view(batch_size, seq_len, model.num_heads, head_dim)

            # Store each head's output separately
            for head_idx in range(model.num_heads):
                head_out = reshaped[:, :, head_idx]  # [batch_size, seq_len, head_dim]
                head_outputs[f'layer_{layer_idx}_head_{head_idx}'] = head_out

        # Analyze head patterns
        head_pattern_stats = {}
        for pattern_name, patterns in activations['attn_patterns'].items():
            # Calculate average pattern
            avg_pattern = torch.mean(patterns, dim=0)

            # Calculate entropy (lower means more specialized/focused)
            entropy = -torch.sum(avg_pattern * torch.log(avg_pattern + 1e-10)).item()

            # Calculate diagonal strength (higher means more focus on the token itself)
            diag_strength = torch.mean(torch.diag(avg_pattern)).item()

            # Calculate position bias (where the head focuses most)
            row_means = torch.mean(patterns, dim=1)  # Average across columns to get row importance
            pos_bias = torch.argmax(row_means).item()

            head_pattern_stats[pattern_name] = {
                'entropy': entropy,
                'diagonal_strength': diag_strength,
                'position_bias': pos_bias,
                'mean_pattern': avg_pattern.numpy()
            }

        # Analyze MLP specialization
        mlp_specialization = {}
        for layer_idx in range(model.num_layers):
            # Skip if we don't have data for this layer
            if layer_idx not in activations['mlp_outputs']:
                continue

            # Get MLP outputs
            mlp_out = activations['mlp_outputs'][layer_idx]
            batch_size, seq_len, hidden_dim = mlp_out.shape

            # Reshape to 2D: [batch_size * seq_len, hidden_dim]
            mlp_out_flat = mlp_out.reshape(-1, hidden_dim)

            # Run PCA to find the principal components of MLP output
            pca = PCA(n_components=min(pca_dims, mlp_out_flat.shape[1], mlp_out_flat.shape[0]))
            pca.fit(mlp_out_flat.numpy())

            # Calculate explained variance ratio
            explained_variance = pca.explained_variance_ratio_

            # Calculate sparsity of activations
            act_sparsity = (mlp_out_flat.abs() < 0.01).float().mean().item()

            mlp_specialization[f'layer_{layer_idx}'] = {
                'pca_explained_variance': explained_variance,
                'concentration': explained_variance[0],  # How much first component explains
                'activation_sparsity': act_sparsity
            }

        results['head_pattern_stats'] = head_pattern_stats
        results['mlp_specialization'] = mlp_specialization

        return results

    def visualize_attention_mlp_relation(self, epoch, activation_flow=None, specialization=None):
        """
        Create visualizations of attention-MLP relationships.

        Args:
            epoch: Current training epoch
            activation_flow: Pre-computed activation flow results
            specialization: Pre-computed specialization results

        Returns:
            dict: Paths to generated visualizations
        """
        if activation_flow is None or specialization is None:
            activations = self.collect_activations()
            if activation_flow is None:
                activation_flow = self.analyze_activation_flow(activations)
            if specialization is None:
                specialization = self.analyze_attention_mlp_specialization(activations)

        vis_paths = {}

        # 1. Visualize correlation between attention outputs and MLP inputs
        corr_fig, corr_axes = plt.subplots(self.model.num_layers, 1,
                                           figsize=(10, 5 * self.model.num_layers))

        if self.model.num_layers == 1:
            corr_axes = [corr_axes]

        for layer_idx in range(self.model.num_layers):
            layer_key = f'layer_{layer_idx}'
            if layer_key not in activation_flow:
                continue

            # Plot correlation matrix
            sns.heatmap(
                activation_flow[layer_key]['cross_correlation_matrix'],
                cmap='coolwarm',
                center=0,
                ax=corr_axes[layer_idx],
                vmin=self.heatmap_min, vmax=self.heatmap_max,
            )
            corr_axes[layer_idx].set_title(f'Layer {layer_idx} Attention-MLP Correlation')
            corr_axes[layer_idx].set_xlabel('MLP Input Dimension')
            corr_axes[layer_idx].set_ylabel('Attention Output Dimension')

        corr_path = self.save_dir / f'attn_mlp_correlation_epoch_{epoch}.png'
        corr_fig.tight_layout()
        corr_fig.savefig(corr_path)
        plt.close(corr_fig)
        vis_paths['correlation'] = str(corr_path)

        # 2. Visualize head pattern statistics
        pattern_fig, pattern_axes = plt.subplots(1, 2, figsize=(15, 6))

        # Extract pattern stats
        pattern_data = []
        for pattern_name, stats in specialization['head_pattern_stats'].items():
            layer_idx = int(pattern_name.split('_')[1])
            head_idx = int(pattern_name.split('_')[3])
            pattern_data.append({
                'layer': layer_idx,
                'head': head_idx,
                'entropy': stats['entropy'],
                'diagonal_strength': stats['diagonal_strength']
            })

        pattern_df = pd.DataFrame(pattern_data)

        # Plot entropy
        sns.barplot(
            data=pattern_df,
            x='head',
            y='entropy',
            hue='layer',
            ax=pattern_axes[0]
        )
        pattern_axes[0].set_title('Attention Head Entropy (lower = more specialized)')
        pattern_axes[0].set_xlabel('Head Index')
        pattern_axes[0].set_ylabel('Entropy')

        # Plot diagonal strength
        sns.barplot(
            data=pattern_df,
            x='head',
            y='diagonal_strength',
            hue='layer',
            ax=pattern_axes[1]
        )
        pattern_axes[1].set_title('Attention Head Diagonal Strength')
        pattern_axes[1].set_xlabel('Head Index')
        pattern_axes[1].set_ylabel('Diagonal Strength')

        pattern_path = self.save_dir / f'head_pattern_stats_epoch_{epoch}.png'
        pattern_fig.tight_layout()
        pattern_fig.savefig(pattern_path)
        plt.close(pattern_fig)
        vis_paths['pattern_stats'] = str(pattern_path)

        # 3. Visualize MLP specialization
        mlp_fig, mlp_axes = plt.subplots(1, 2, figsize=(15, 6))

        # Extract MLP stats
        mlp_data = []
        for layer_key, stats in specialization['mlp_specialization'].items():
            layer_idx = int(layer_key.split('_')[1])
            mlp_data.append({
                'layer': layer_idx,
                'concentration': stats['concentration'],
                'activation_sparsity': stats['activation_sparsity']
            })

        mlp_df = pd.DataFrame(mlp_data)

        # Plot concentration
        sns.barplot(
            data=mlp_df,
            x='layer',
            y='concentration',
            ax=mlp_axes[0]
        )
        mlp_axes[0].set_title('MLP Representation Concentration (higher = less distributed)')
        mlp_axes[0].set_xlabel('Layer Index')
        mlp_axes[0].set_ylabel('PCA Concentration')

        # Plot sparsity
        sns.barplot(
            data=mlp_df,
            x='layer',
            y='activation_sparsity',
            ax=mlp_axes[1]
        )
        mlp_axes[1].set_title('MLP Activation Sparsity')
        mlp_axes[1].set_xlabel('Layer Index')
        mlp_axes[1].set_ylabel('Activation Sparsity')

        mlp_path = self.save_dir / f'mlp_specialization_epoch_{epoch}.png'
        mlp_fig.tight_layout()
        mlp_fig.savefig(mlp_path)
        plt.close(mlp_fig)
        vis_paths['mlp_specialization'] = str(mlp_path)

        # 4. Visualize sample attention patterns
        pattern_vis_fig, pattern_vis_axes = plt.subplots(
            self.model.num_layers,
            self.model.num_heads,
            figsize=(4 * self.model.num_heads, 4 * self.model.num_layers)
        )

        if self.model.num_layers == 1 and self.model.num_heads == 1:
            pattern_vis_axes = np.array([[pattern_vis_axes]])
        elif self.model.num_layers == 1:
            pattern_vis_axes = np.array([pattern_vis_axes])
        elif self.model.num_heads == 1:
            pattern_vis_axes = np.array([[ax] for ax in pattern_vis_axes])

        for layer_idx in range(self.model.num_layers):
            for head_idx in range(self.model.num_heads):
                pattern_name = f'layer_{layer_idx}_head_{head_idx}'
                if pattern_name not in specialization['head_pattern_stats']:
                    continue

                # Get mean pattern
                mean_pattern = specialization['head_pattern_stats'][pattern_name]['mean_pattern']

                # Check dimensionality and reshape if needed
                if len(mean_pattern.shape) == 1:
                    # Assuming it should be a square matrix
                    seq_len = int(math.sqrt(len(mean_pattern)))
                    if seq_len ** 2 == len(mean_pattern):  # Perfect square check
                        mean_pattern = mean_pattern.reshape(seq_len, seq_len)
                    else:
                        # If not a perfect square, just make it a row vector
                        mean_pattern = mean_pattern.reshape(1, -1)
                sns.heatmap(
                    mean_pattern,
                    cmap='viridis',
                    ax=pattern_vis_axes[layer_idx, head_idx],
                    vmin=self.heatmap_min, vmax=self.heatmap_max,
                )
                pattern_vis_axes[layer_idx, head_idx].set_title(f'Layer {layer_idx}, Head {head_idx}')

        pattern_vis_path = self.save_dir / f'attention_patterns_epoch_{epoch}.png'
        pattern_vis_fig.tight_layout()
        pattern_vis_fig.savefig(pattern_vis_path)
        plt.close(pattern_vis_fig)
        vis_paths['attention_patterns'] = str(pattern_vis_path)

        return vis_paths

    def create_sparse_autoencoder(self, hidden_dim, code_dim, l1_coef=0.001):
        """
        Create a sparse autoencoder for analyzing representations.

        Args:
            hidden_dim: Dimension of the input features
            code_dim: Dimension of the sparse code
            l1_coef: L1 regularization coefficient for sparsity

        Returns:
            SparseAutoencoder: The autoencoder model
        """
        return SparseAutoencoder(hidden_dim, code_dim, l1_coef)

    def analyze_with_sparse_autoencoder(self, autoencoder, activations=None,
                                        component='mlp', layer_idx=1, train_epochs=100):
        """
        Analyze representations using a sparse autoencoder.

        Args:
            autoencoder: SparseAutoencoder model
            activations: Pre-collected activations or None to collect new ones
            component: Which component to analyze ('mlp' or 'attention')
            layer_idx: Which layer to analyze
            train_epochs: Number of epochs to train the autoencoder

        Returns:
            dict: Analysis results
        """
        if activations is None:
            activations = self.collect_activations()

        # Get the activations to analyze
        if component == 'mlp':
            if layer_idx not in activations['mlp_outputs']:
                raise ValueError(f"No MLP outputs available for layer {layer_idx}")
            act_data = activations['mlp_outputs'][layer_idx]
        else:  # attention
            if layer_idx not in activations['attn_outputs']:
                raise ValueError(f"No attention outputs available for layer {layer_idx}")
            act_data = activations['attn_outputs'][layer_idx]

        # Flatten activations
        act_flat = act_data.reshape(-1, act_data.size(-1))

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(act_flat)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=True
        )

        # Train the autoencoder
        autoencoder.to(self.device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

        for epoch in range(train_epochs):
            total_loss = 0
            reconstruction_loss = 0
            sparsity_loss = 0

            for batch in dataloader:
                x = batch[0].to(self.device)

                # Forward pass
                x_recon, codes = autoencoder(x)

                # Calculate losses
                recon_loss = torch.nn.functional.mse_loss(x_recon, x)
                sparse_loss = autoencoder.l1_coef * torch.abs(codes).mean()
                loss = recon_loss + sparse_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                reconstruction_loss += recon_loss.item()
                sparsity_loss += sparse_loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.6f} "
                      f"(Recon = {reconstruction_loss / len(dataloader):.6f}, "
                      f"Sparse = {sparsity_loss / len(dataloader):.6f})")

        # Analyze the learned features
        all_codes = []
        autoencoder.eval()
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                _, codes = autoencoder(x)
                all_codes.append(codes.cpu())

        all_codes = torch.cat(all_codes, dim=0)

        # Calculate statistics
        avg_activation = torch.mean(all_codes, dim=0)
        activation_frequency = torch.mean((all_codes > 0.1).float(), dim=0)
        feature_usage = activation_frequency.numpy()

        # Get decoder weights for feature visualization
        decoder_weights = autoencoder.decoder.weight.detach().cpu().numpy().T

        results = {
            'avg_activation': avg_activation.numpy(),
            'activation_frequency': activation_frequency.numpy(),
            'decoder_weights': decoder_weights,
            'all_codes': all_codes.numpy()
        }

        return results

    def visualize_autoencoder_results(self, results, component, layer_idx, epoch):
        """
        Visualize sparse autoencoder results.

        Args:
            results: Results from analyze_with_sparse_autoencoder
            component: Which component was analyzed ('mlp' or 'attention')
            layer_idx: Which layer was analyzed
            epoch: Current training epoch

        Returns:
            dict: Paths to generated visualizations
        """
        vis_paths = {}

        # 1. Feature usage distribution
        usage_fig, usage_ax = plt.subplots(figsize=(12, 6))

        sns.histplot(
            results['activation_frequency'],
            bins=50,
            ax=usage_ax
        )
        usage_ax.set_title(f'{component.upper()} Layer {layer_idx} Feature Usage Distribution')
        usage_ax.set_xlabel('Activation Frequency')
        usage_ax.set_ylabel('Count')

        usage_path = self.save_dir / f'{component}_layer{layer_idx}_feature_usage_epoch_{epoch}.png'
        usage_fig.tight_layout()
        usage_fig.savefig(usage_path)
        plt.close(usage_fig)
        vis_paths['feature_usage'] = str(usage_path)

        # 2. Top features visualization (decoder weights)
        # Find top activated features
        top_features = np.argsort(results['activation_frequency'])[-20:]

        top_fig, top_axes = plt.subplots(4, 5, figsize=(15, 12))
        top_axes = top_axes.flatten()

        for i, feature_idx in enumerate(top_features):
            # Get decoder weights for this feature
            feature_weights = results['decoder_weights'][feature_idx]

            # Plot as a heatmap reshaped to match input dimension structure if possible
            # Simple case: just plot as a bar chart
            top_axes[i].bar(
                range(len(feature_weights)),
                feature_weights,
                width=1.0
            )
            top_axes[i].set_title(f'Feature {feature_idx} (freq: {results["activation_frequency"][feature_idx]:.3f})')
            top_axes[i].set_xticks([])

        top_path = self.save_dir / f'{component}_layer{layer_idx}_top_features_epoch_{epoch}.png'
        top_fig.tight_layout()
        top_fig.savefig(top_path)
        plt.close(top_fig)
        vis_paths['top_features'] = str(top_path)

        # 3. Code correlation heatmap
        code_fig, code_ax = plt.subplots(figsize=(12, 10))

        # Calculate correlation matrix on a sample of codes (up to 1000 for efficiency)
        sample_size = min(1000, results['all_codes'].shape[0])
        sample_indices = np.random.choice(results['all_codes'].shape[0], sample_size, replace=False)
        sample_codes = results['all_codes'][sample_indices]

        code_corr = np.corrcoef(sample_codes.T)
        sns.heatmap(
            code_corr,
            cmap='coolwarm',
            center=0,
            ax=code_ax,
            vmin=self.heatmap_min, vmax=self.heatmap_max,
        )
        code_ax.set_title(f'{component.upper()} Layer {layer_idx} Feature Correlations')

        code_path = self.save_dir / f'{component}_layer{layer_idx}_code_correlations_epoch_{epoch}.png'
        code_fig.tight_layout()
        code_fig.savefig(code_path)
        plt.close(code_fig)
        vis_paths['code_correlations'] = str(code_path)

        # 4. t-SNE visualization of codes
        tsne_fig, tsne_ax = plt.subplots(figsize=(10, 8))

        # Run t-SNE on a sample of codes
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(sample_codes)

        sns.scatterplot(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            alpha=0.5,
            ax=tsne_ax
        )
        tsne_ax.set_title(f'{component.upper()} Layer {layer_idx} t-SNE Visualization')
        tsne_ax.set_xlabel('t-SNE 1')
        tsne_ax.set_ylabel('t-SNE 2')

        tsne_path = self.save_dir / f'{component}_layer{layer_idx}_tsne_epoch_{epoch}.png'
        tsne_fig.tight_layout()
        tsne_fig.savefig(tsne_path)
        plt.close(tsne_fig)
        vis_paths['tsne'] = str(tsne_path)

        return vis_paths

    # Completing the analyze_across_grokking_phases method

    def analyze_across_grokking_phases(self, save_dir=None, phase_checkpoints=None):
        """
        Compare attention-MLP relationships across grokking phases.

        Args:
            save_dir: Directory to save results
            phase_checkpoints: List of (phase_name, checkpoint_path) tuples

        Returns:
            dict: Comparison results
        """
        if phase_checkpoints is None or len(phase_checkpoints) < 2:
            raise ValueError("Need at least two phase checkpoints to compare")

        save_dir = Path(save_dir) if save_dir else self.save_dir / "phase_comparison"
        save_dir.mkdir(exist_ok=True, parents=True)

        # Storage for results across phases
        phase_results = {}

        for phase_name, checkpoint_path in phase_checkpoints:
            print(f"Analyzing {phase_name} phase from {checkpoint_path}...")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Collect activations and analyze
            activations = self.collect_activations()
            flow_results = self.analyze_activation_flow(activations)
            spec_results = self.analyze_attention_mlp_specialization(activations)

            # Store results
            phase_results[phase_name] = {
                'flow': flow_results,
                'specialization': spec_results
            }

            # Visualize individual phase
            self.visualize_attention_mlp_relation(
                epoch=phase_name,  # Use phase name instead of epoch
                activation_flow=flow_results,
                specialization=spec_results
            )

        # Compare across phases - create comparative visualizations
        comparison_results = self.create_phase_comparison_visualizations(phase_results, save_dir)

        return {
            'phase_results': phase_results,
            'comparison': comparison_results
        }

    def create_phase_comparison_visualizations(self, phase_results, save_dir):
        """
        Create visualizations comparing metrics across grokking phases.

        Args:
            phase_results: Dictionary of results per phase
            save_dir: Directory to save visualizations

        Returns:
            dict: Paths to generated visualizations
        """
        vis_paths = {}

        # 1. Compare attention-MLP correlation across phases
        corr_fig, corr_ax = plt.subplots(figsize=(12, 6))

        # Extract correlation data
        corr_data = []
        for phase_name, results in phase_results.items():
            for layer_key, layer_results in results['flow'].items():
                layer_idx = int(layer_key.split('_')[1])
                corr_data.append({
                    'phase': phase_name,
                    'layer': layer_idx,
                    'correlation': layer_results['mean_cross_correlation']
                })

        corr_df = pd.DataFrame(corr_data)

        # Plot correlation comparison
        sns.barplot(
            data=corr_df,
            x='layer',
            y='correlation',
            hue='phase',
            ax=corr_ax
        )
        corr_ax.set_title('Attention-MLP Correlation Across Grokking Phases')
        corr_ax.set_xlabel('Layer')
        corr_ax.set_ylabel('Mean Absolute Correlation')

        corr_path = save_dir / 'correlation_comparison.png'
        corr_fig.tight_layout()
        corr_fig.savefig(corr_path)
        plt.close(corr_fig)
        vis_paths['correlation_comparison'] = str(corr_path)

        # 2. Compare head specialization (entropy) across phases
        entropy_fig, entropy_ax = plt.subplots(figsize=(14, 7))

        # Extract entropy data
        entropy_data = []
        for phase_name, results in phase_results.items():
            for pattern_name, pattern_stats in results['specialization']['head_pattern_stats'].items():
                layer_idx = int(pattern_name.split('_')[1])
                head_idx = int(pattern_name.split('_')[3])
                entropy_data.append({
                    'phase': phase_name,
                    'layer': layer_idx,
                    'head': head_idx,
                    'entropy': pattern_stats['entropy']
                })

        entropy_df = pd.DataFrame(entropy_data)

        # Plot entropy comparison
        sns.barplot(
            data=entropy_df,
            x='head',
            y='entropy',
            hue='phase',
            ax=entropy_ax
        )
        entropy_ax.set_title('Attention Head Entropy Across Grokking Phases')
        entropy_ax.set_xlabel('Head Index')
        entropy_ax.set_ylabel('Entropy (lower = more specialized)')

        entropy_path = save_dir / 'entropy_comparison.png'
        entropy_fig.tight_layout()
        entropy_fig.savefig(entropy_path)
        plt.close(entropy_fig)
        vis_paths['entropy_comparison'] = str(entropy_path)

        # 3. Compare MLP specialization across phases
        mlp_fig, mlp_ax = plt.subplots(figsize=(12, 6))

        # Extract MLP specialization data
        mlp_data = []
        for phase_name, results in phase_results.items():
            for layer_key, layer_results in results['specialization']['mlp_specialization'].items():
                layer_idx = int(layer_key.split('_')[1])
                mlp_data.append({
                    'phase': phase_name,
                    'layer': layer_idx,
                    'concentration': layer_results['concentration']
                })

        mlp_df = pd.DataFrame(mlp_data)

        # Plot MLP concentration comparison
        sns.barplot(
            data=mlp_df,
            x='layer',
            y='concentration',
            hue='phase',
            ax=mlp_ax
        )
        mlp_ax.set_title('MLP Representation Concentration Across Grokking Phases')
        mlp_ax.set_xlabel('Layer')
        mlp_ax.set_ylabel('PCA Concentration (higher = less distributed)')

        mlp_path = save_dir / 'mlp_concentration_comparison.png'
        mlp_fig.tight_layout()
        mlp_fig.savefig(mlp_path)
        plt.close(mlp_fig)
        vis_paths['mlp_concentration_comparison'] = str(mlp_path)

        # 4. Compare attention patterns visually across phases
        # Create a grid of attention patterns for each phase and head
        phases = list(phase_results.keys())
        num_phases = len(phases)

        # Find number of layers and heads
        num_layers = 0
        num_heads = 0
        for phase_name, results in phase_results.items():
            for pattern_name in results['specialization']['head_pattern_stats'].keys():
                layer_idx = int(pattern_name.split('_')[1])
                head_idx = int(pattern_name.split('_')[3])
                num_layers = max(num_layers, layer_idx + 1)
                num_heads = max(num_heads, head_idx + 1)

        # Create a large figure with all patterns
        pattern_fig, pattern_axes = plt.subplots(
            num_phases * num_layers,
            num_heads,
            figsize=(4 * num_heads, 4 * num_layers * num_phases)
        )

        # Adjust pattern_axes based on dimensions
        if num_layers * num_phases == 1 and num_heads == 1:
            pattern_axes = np.array([[pattern_axes]])
        elif num_layers * num_phases == 1:
            pattern_axes = np.array([pattern_axes])
        elif num_heads == 1:
            pattern_axes = np.array([[ax] for ax in pattern_axes])

        for phase_idx, phase_name in enumerate(phases):
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    pattern_name = f'layer_{layer_idx}_head_{head_idx}'
                    if pattern_name not in phase_results[phase_name]['specialization']['head_pattern_stats']:
                        continue

                    # Get mean pattern
                    mean_pattern = phase_results[phase_name]['specialization']['head_pattern_stats'][pattern_name][
                        'mean_pattern']

                    # Get axis index
                    ax_idx = phase_idx * num_layers + layer_idx

                    # Plot pattern
                    sns.heatmap(
                        mean_pattern,
                        cmap='viridis',
                        ax=pattern_axes[ax_idx, head_idx],
                        cbar=False,
                        vmin=self.heatmap_min, vmax=self.heatmap_max,
                    )
                    pattern_axes[ax_idx, head_idx].set_title(f'{phase_name}: L{layer_idx}, H{head_idx}')
                    # Remove x and y labels for cleaner look
                    pattern_axes[ax_idx, head_idx].set_xlabel('')
                    pattern_axes[ax_idx, head_idx].set_ylabel('')
                    pattern_axes[ax_idx, head_idx].set_xticks([])
                    pattern_axes[ax_idx, head_idx].set_yticks([])

        pattern_path = save_dir / 'attention_patterns_comparison.png'
        pattern_fig.tight_layout()
        pattern_fig.savefig(pattern_path)
        plt.close(pattern_fig)
        vis_paths['attention_patterns_comparison'] = str(pattern_path)

        return vis_paths

    def analyze_across_jumps(self, weight_tracker, jump_results, num_batches=3):
        """
        Analyze attention-MLP relationships across detected jumps.

        Parameters:
        -----------
        weight_tracker : EnhancedWeightSpaceTracker
            The weight tracker containing jump information
        jump_results : list
            Results from weight_tracker.analyze_pending_jumps()
        num_batches : int
            Number of batches to use for activation collection

        Returns:
        --------
        dict : Comparative analysis across jumps
        """
        jump_analyses = {}

        for result in jump_results:
            jump_epoch = result['jump_epoch']
            jump_char = result['characterization']

            # Store the current model state
            original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            # Get pre-jump, jump, and post-jump states
            pre_jump_state = result.get('pre_jump_snapshot', {}).get('state_dict')
            jump_state = result.get('jump_snapshot', {}).get('state_dict')
            post_jump_state = result.get('post_jump_snapshot', {}).get('state_dict')

            jump_analyses[jump_epoch] = {}

            try:
                # Analyze each state
                for state_name, state in [
                    ('pre_jump', pre_jump_state),
                    ('jump', jump_state),
                    ('post_jump', post_jump_state)
                ]:
                    if state is None:
                        continue

                    # Load this state
                    self.model.load_state_dict(state)

                    # Focus on top changing layers and heads from characterization
                    top_layers = [int(layer.split('_')[1]) for layer in jump_char.get('top_layers', [])]
                    top_heads = []
                    for head in jump_char.get('top_heads', []):
                        parts = head.split('_')
                        if len(parts) >= 4:
                            layer_idx = int(parts[1])
                            head_idx = int(parts[3])
                            top_heads.append((layer_idx, head_idx))

                    # Collect activations
                    activations = self.collect_activations(num_batches=num_batches)

                    # Analyze activation flow for top layers
                    flow_analysis = {}
                    for layer_idx in top_layers:
                        layer_key = f'layer_{layer_idx}'
                        if layer_idx in activations.get('attn_outputs', {}) and layer_idx in activations.get(
                                'mlp_inputs',
                                {}):
                            # Get attention outputs and MLP inputs
                            attn_out = activations['attn_outputs'][layer_idx]
                            mlp_in = activations['mlp_inputs'][layer_idx]

                            # Reshape for correlation analysis
                            attn_out_flat = attn_out.reshape(-1, attn_out.size(-1))
                            mlp_in_flat = mlp_in.reshape(-1, mlp_in.size(-1))

                            # Calculate correlation
                            corr_matrix = np.corrcoef(
                                attn_out_flat[:, :100].numpy().T,  # Sample up to 100 dims
                                mlp_in_flat[:, :100].numpy().T
                            )

                            # Extract cross-correlation block
                            cross_corr = corr_matrix[:100, 100:]

                            flow_analysis[layer_key] = {
                                'mean_correlation': np.mean(np.abs(cross_corr)),
                                'cross_correlation_matrix': cross_corr
                            }

                    # Analyze patterns for top heads
                    pattern_analysis = {}
                    for layer_idx, head_idx in top_heads:
                        pattern_key = f'layer_{layer_idx}_head_{head_idx}'
                        if pattern_key in activations.get('attn_patterns', {}):
                            patterns = activations['attn_patterns'][pattern_key]

                            # Calculate entropy (lower means more specialized)
                            entropy = -torch.sum(patterns * torch.log(patterns + 1e-10)).item()

                            # Calculate diagonal strength (focus on token itself)
                            diag_strength = torch.mean(torch.diag(patterns)).item()

                            pattern_analysis[pattern_key] = {
                                'entropy': entropy,
                                'diagonal_strength': diag_strength,
                                'mean_pattern': patterns.mean(0).numpy()
                            }

                    # Store results for this state
                    jump_analyses[jump_epoch][state_name] = {
                        'flow_analysis': flow_analysis,
                        'pattern_analysis': pattern_analysis
                    }

                    # Apply sparse autoencoder to analyze top layer representations
                    if top_layers and 'mlp' in state_name:  # Only for specific states to save time
                        primary_layer = top_layers[0]
                        layer_key = f'layer_{primary_layer}'

                        if primary_layer in activations.get('mlp_outputs', {}):
                            autoencoder = SparseAutoencoder(
                                input_dim=self.model.dim,
                                code_dim=self.model.dim * 2,
                                l1_coef=0.001
                            )

                            mlp_out = activations['mlp_outputs'][primary_layer]
                            mlp_out_flat = mlp_out.reshape(-1, mlp_out.size(-1))

                            # Train autoencoder on this layer's activations
                            sparse_results = self._train_autoencoder(
                                autoencoder,
                                mlp_out_flat,
                                epochs=50
                            )

                            jump_analyses[jump_epoch][state_name]['sparse_analysis'] = {
                                layer_key: sparse_results
                            }
            finally:
                # Restore original model state
                self.model.load_state_dict(original_state)

            # Calculate changes across states
            if all(k in jump_analyses[jump_epoch] for k in ['pre_jump', 'jump', 'post_jump']):
                # Analyze changes in flow correlation
                for layer_key in top_layers:
                    layer_key = f'layer_{layer_key}'
                    pre_corr = jump_analyses[jump_epoch]['pre_jump']['flow_analysis'].get(layer_key, {}).get(
                        'mean_correlation', 0)
                    jump_corr = jump_analyses[jump_epoch]['jump']['flow_analysis'].get(layer_key, {}).get(
                        'mean_correlation', 0)
                    post_corr = jump_analyses[jump_epoch]['post_jump']['flow_analysis'].get(layer_key, {}).get(
                        'mean_correlation', 0)

                    jump_analyses[jump_epoch]['correlation_changes'] = {
                        layer_key: {
                            'pre_to_jump': jump_corr - pre_corr,
                            'jump_to_post': post_corr - jump_corr,
                            'overall': post_corr - pre_corr
                        }
                    }

                # Analyze changes in pattern entropy
                for pattern_key in pattern_analysis:
                    pre_entropy = jump_analyses[jump_epoch]['pre_jump']['pattern_analysis'].get(pattern_key, {}).get(
                        'entropy', 0)
                    jump_entropy = jump_analyses[jump_epoch]['jump']['pattern_analysis'].get(pattern_key, {}).get(
                        'entropy',
                        0)
                    post_entropy = jump_analyses[jump_epoch]['post_jump']['pattern_analysis'].get(pattern_key, {}).get(
                        'entropy', 0)

                    if 'entropy_changes' not in jump_analyses[jump_epoch]:
                        jump_analyses[jump_epoch]['entropy_changes'] = {}

                    jump_analyses[jump_epoch]['entropy_changes'][pattern_key] = {
                        'pre_to_jump': jump_entropy - pre_entropy,
                        'jump_to_post': post_entropy - jump_entropy,
                        'overall': post_entropy - pre_entropy
                    }

        # Create visualizations for these analyses
        self._visualize_jump_comparisons(jump_analyses)

        return jump_analyses

    def _train_autoencoder(self, autoencoder, activations, epochs=50, batch_size=128):
        """Helper method to train a sparse autoencoder on activations"""
        device = next(self.model.parameters()).device
        autoencoder = autoencoder.to(device)

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(activations)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Train
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss = 0
            recon_loss = 0
            sparse_loss = 0

            for x, in dataloader:
                x = x.to(device)

                # Forward pass
                x_recon, codes = autoencoder(x)

                # Losses
                r_loss = torch.nn.functional.mse_loss(x_recon, x)
                s_loss = autoencoder.l1_coef * torch.abs(codes).mean()
                loss = r_loss + s_loss

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                recon_loss += r_loss.item()
                sparse_loss += s_loss.item()

        # Analyze the learned features
        autoencoder.eval()
        all_codes = []

        with torch.no_grad():
            for x, in dataloader:
                x = x.to(device)
                _, codes = autoencoder(x)
                all_codes.append(codes.cpu())

        all_codes = torch.cat(all_codes, dim=0)

        # Calculate feature statistics
        activation_freq = torch.mean((all_codes > 0.1).float(), dim=0).numpy()
        decoder_weights = autoencoder.decoder.weight.detach().cpu().numpy().T

        return {
            'activation_frequency': activation_freq,
            'decoder_weights': decoder_weights,
            'autoencoder': autoencoder
        }

    def _visualize_jump_comparisons(self, jump_analyses):
        """Create visualizations comparing attention-MLP relationships across jumps"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        viz_dir = self.save_dir / "jump_comparisons"
        viz_dir.mkdir(exist_ok=True, parents=True)

        for jump_epoch, analysis in jump_analyses.items():
            # 1. Compare correlation changes
            if 'correlation_changes' in analysis:
                fig1, ax1 = plt.subplots(figsize=(10, 6))

                # Extract data for plotting
                layers = []
                pre_to_jump = []
                jump_to_post = []
                overall = []

                for layer, changes in analysis['correlation_changes'].items():
                    layers.append(layer)
                    pre_to_jump.append(changes['pre_to_jump'])
                    jump_to_post.append(changes['jump_to_post'])
                    overall.append(changes['overall'])

                # Set up x-positions for grouped bars
                x = np.arange(len(layers))
                width = 0.25

                # Plot grouped bars
                ax1.bar(x - width, pre_to_jump, width, label='Pre→Jump')
                ax1.bar(x, jump_to_post, width, label='Jump→Post')
                ax1.bar(x + width, overall, width, label='Overall')

                ax1.set_xlabel('Layer')
                ax1.set_ylabel('Change in Correlation')
                ax1.set_title(f'Changes in Attention-MLP Correlation at Jump {jump_epoch}')
                ax1.set_xticks(x)
                ax1.set_xticklabels(layers)
                ax1.legend()

                plt.tight_layout()
                plt.savefig(viz_dir / f"jump_{jump_epoch}_correlation_changes.png")
                plt.close(fig1)

            # 2. Compare entropy changes
            if 'entropy_changes' in analysis:
                fig2, ax2 = plt.subplots(figsize=(12, 6))

                # Extract data for plotting
                heads = []
                pre_to_jump = []
                jump_to_post = []

                for head, changes in analysis['entropy_changes'].items():
                    heads.append(head)
                    pre_to_jump.append(changes['pre_to_jump'])
                    jump_to_post.append(changes['jump_to_post'])

                # Convert to DataFrame for grouped bar plot
                import pandas as pd
                data = []
                for i, head in enumerate(heads):
                    data.append({'Head': head, 'Change Type': 'Pre→Jump', 'Entropy Change': pre_to_jump[i]})
                    data.append({'Head': head, 'Change Type': 'Jump→Post', 'Entropy Change': jump_to_post[i]})

                df = pd.DataFrame(data)

                # Create grouped bar plot
                sns.barplot(x='Head', y='Entropy Change', hue='Change Type', data=df, ax=ax2)

                ax2.set_title(f'Changes in Attention Head Entropy at Jump {jump_epoch}')
                ax2.set_xlabel('Attention Head')
                ax2.set_ylabel('Entropy Change')
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

                plt.tight_layout()
                plt.savefig(viz_dir / f"jump_{jump_epoch}_entropy_changes.png")
                plt.close(fig2)

            # 3. Compare attention patterns across states
            for state_name in ['pre_jump', 'jump', 'post_jump']:
                if state_name in analysis and 'pattern_analysis' in analysis[state_name]:
                    patterns = analysis[state_name]['pattern_analysis']

                    if patterns:
                        # Create a grid of patterns
                        n_patterns = len(patterns)
                        n_cols = min(n_patterns, 4)
                        n_rows = (n_patterns + n_cols - 1) // n_cols

                        fig3, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
                        axes = axes.flatten() if n_patterns > 1 else [axes]

                        for i, (head_key, pattern_data) in enumerate(patterns.items()):
                            if i < len(axes):
                                mean_pattern = pattern_data['mean_pattern']
                                if len(mean_pattern.shape) == 1:
                                    mean_pattern_2d = mean_pattern.reshape(1, -1)
                                sns.heatmap(mean_pattern_2d, ax=axes[i], cmap='viridis',
                                            vmin=self.heatmap_min, vmax=self.heatmap_max,)
                                axes[i].set_title(f"{head_key}\nEntropy: {pattern_data['entropy']:.3f}")

                        # Hide unused axes
                        for i in range(n_patterns, len(axes)):
                            axes[i].axis('off')

                        plt.tight_layout()
                        plt.savefig(viz_dir / f"jump_{jump_epoch}_{state_name}_patterns.png")
                        plt.close(fig3)

            # 4. Feature visualization for sparse analysis
            for state_name in ['pre_jump', 'jump', 'post_jump']:
                if state_name in analysis and 'sparse_analysis' in analysis[state_name]:
                    for layer_key, results in analysis[state_name]['sparse_analysis'].items():
                        # Visualize feature usage distribution
                        fig4, ax4 = plt.subplots(figsize=(10, 6))

                        sns.histplot(results['activation_frequency'], bins=30, ax=ax4)
                        ax4.set_title(f'Feature Usage Distribution - {layer_key} at {state_name}')
                        ax4.set_xlabel('Activation Frequency')
                        ax4.set_ylabel('Count')

                        plt.tight_layout()
                        plt.savefig(viz_dir / f"jump_{jump_epoch}_{state_name}_{layer_key}_feature_dist.png")
                        plt.close(fig4)

                        # Visualize top features
                        top_indices = np.argsort(results['activation_frequency'])[-16:]
                        fig5, axes = plt.subplots(4, 4, figsize=(12, 10))
                        axes = axes.flatten()

                        for i, feat_idx in enumerate(top_indices):
                            weights = results['decoder_weights'][feat_idx]
                            axes[i].bar(range(len(weights)), weights)
                            axes[i].set_title(
                                f'Feature {feat_idx}\nFreq: {results["activation_frequency"][feat_idx]:.3f}')
                            axes[i].set_xticks([])

                        plt.tight_layout()
                        plt.savefig(viz_dir / f"jump_{jump_epoch}_{state_name}_{layer_key}_top_features.png")
                        plt.close(fig5)

    def discover_circuits(self, weight_tracker, jump_results, eval_loader, criterion, circuit_threshold=0.1):
        """
        Discover circuits that form during jump transitions.

        Parameters:
        -----------
        weight_tracker : EnhancedWeightSpaceTracker
            The weight tracker with jump information
        jump_results : list
            Results from weight_tracker.analyze_pending_jumps()
        eval_loader : DataLoader
            Evaluation data loader
        criterion : loss function
            Loss function for evaluation
        circuit_threshold : float
            Threshold for identifying circuit components

        Returns:
        --------
        dict : Discovered circuits and their properties
        """
        circuit_analysis = {}

        for result in jump_results:
            jump_epoch = result['jump_epoch']
            jump_char = result['characterization']

            # Store original model state
            original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            try:
                # Get pre-jump, jump, and post-jump states
                pre_jump_state = result.get('pre_jump_snapshot', {}).get('state_dict')
                jump_state = result.get('jump_snapshot', {}).get('state_dict')
                post_jump_state = result.get('post_jump_snapshot', {}).get('state_dict')

                # Focus on top changing components from characterization
                top_layers = [int(layer.split('_')[1]) for layer in jump_char.get('top_layers', [])]
                top_heads = []
                for head in jump_char.get('top_heads', []):
                    parts = head.split('_')
                    if len(parts) >= 4:
                        layer_idx = int(parts[1])
                        head_idx = int(parts[3])
                        top_heads.append((layer_idx, head_idx))

                # Identify potential circuit components
                potential_components = []

                # Add top heads to potential components
                for layer_idx, head_idx in top_heads:
                    potential_components.append(('attention', layer_idx, head_idx))

                # Add top MLP layers to potential components
                for layer_idx in top_layers:
                    potential_components.append(('mlp', layer_idx, None))

                # Evaluate baseline performance with post-jump state
                self.model.load_state_dict(post_jump_state)
                baseline_acc, baseline_loss = self.model.evaluate(eval_loader)

                # Analyze each potential component
                component_attribution = {}

                for comp_type, layer_idx, head_idx in potential_components:
                    # Load post-jump state (restored state)
                    self.model.load_state_dict(post_jump_state)

                    # Mask the component
                    if comp_type == 'attention' and head_idx is not None:
                        # Mask specific attention head
                        head_dim = self.model.dim // self.model.num_heads
                        start_idx = head_idx * head_dim
                        end_idx = (head_idx + 1) * head_dim

                        with torch.no_grad():
                            # Zero out this head's contribution in the output projection
                            self.model.layers[layer_idx].attn.out_proj.weight[:, start_idx:end_idx] = 0

                    elif comp_type == 'mlp':
                        # Mask entire MLP
                        with torch.no_grad():
                            # Zero out MLP weights
                            if hasattr(self.model.layers[layer_idx].mlp, '0'):
                                # Sequential MLP
                                self.model.layers[layer_idx].mlp[0].weight.fill_(0)
                                self.model.layers[layer_idx].mlp[2].weight.fill_(0)
                            else:
                                # Named MLP
                                self.model.layers[layer_idx].mlp.up_proj.weight.fill_(0)
                                self.model.layers[layer_idx].mlp.down_proj.weight.fill_(0)

                    # Evaluate with component masked
                    ablated_acc, ablated_loss = self.model.evaluate(eval_loader)

                    # Calculate attribution
                    attribution = baseline_acc - ablated_acc

                    # Store results
                    component_key = f"{comp_type}_{layer_idx}"
                    if head_idx is not None:
                        component_key += f"_head_{head_idx}"

                    component_attribution[component_key] = attribution

                # Identify circuit components based on attribution threshold
                circuit_components = {k: v for k, v in component_attribution.items()
                                      if v >= circuit_threshold}

                # Sort by attribution
                sorted_components = sorted(circuit_components.items(),
                                           key=lambda x: x[1],
                                           reverse=True)

                # Analyze pairwise interactions
                pairwise_interactions = {}

                # Only analyze if we have at least 2 components
                component_keys = list(circuit_components.keys())
                if len(component_keys) >= 2:
                    for i, comp1 in enumerate(component_keys):
                        for comp2 in component_keys[i + 1:]:
                            # Load post-jump state
                            self.model.load_state_dict(post_jump_state)

                            # Parse component keys
                            comp1_parts = comp1.split('_')
                            comp2_parts = comp2.split('_')

                            # Mask both components
                            self._mask_component(self.model, comp1_parts)
                            self._mask_component(self.model, comp2_parts)

                            # Evaluate with both components masked
                            pair_ablated_acc, pair_ablated_loss = self.model.evaluate(eval_loader)

                            # Calculate interaction effect
                            expected_attribution = circuit_components[comp1] + circuit_components[comp2]
                            actual_attribution = baseline_acc - pair_ablated_acc

                            # Measure super-additive effects (positive indicates circuit-like behavior)
                            interaction = actual_attribution - expected_attribution

                            pairwise_interactions[f"{comp1}+{comp2}"] = {
                                'expected_attribution': expected_attribution,
                                'actual_attribution': actual_attribution,
                                'interaction': interaction,
                                'is_circuit': interaction > 0.01  # Positive interaction indicates circuit
                            }

                # Store circuit analysis
                circuit_analysis[jump_epoch] = {
                    'component_attribution': component_attribution,
                    'circuit_components': circuit_components,
                    'sorted_components': sorted_components,
                    'pairwise_interactions': pairwise_interactions
                }

                # Create circuit visualization
                self._visualize_circuit_analysis(jump_epoch, circuit_analysis[jump_epoch])

            finally:
                # Restore original model state
                self.model.load_state_dict(original_state)

        return circuit_analysis

    def _mask_component(self, model, component_parts):
        """Helper to mask a specific component based on parsed key"""
        if 'attention' in component_parts:
            layer_idx = int(component_parts[component_parts.index('attention') + 1])
            if 'head' in component_parts:
                head_idx = int(component_parts[component_parts.index('head') + 1])

                # Mask specific attention head
                head_dim = model.dim // model.num_heads
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim

                with torch.no_grad():
                    model.layers[layer_idx].attn.out_proj.weight[:, start_idx:end_idx] = 0

        elif 'mlp' in component_parts:
            layer_idx = int(component_parts[component_parts.index('mlp') + 1])

            # Mask entire MLP
            with torch.no_grad():
                # Zero out MLP weights
                if hasattr(model.layers[layer_idx].mlp, '0'):
                    # Sequential MLP
                    model.layers[layer_idx].mlp[0].weight.fill_(0)
                    model.layers[layer_idx].mlp[2].weight.fill_(0)
                else:
                    # Named MLP
                    model.layers[layer_idx].mlp.up_proj.weight.fill_(0)
                    model.layers[layer_idx].mlp.down_proj.weight.fill_(0)

    def _visualize_circuit_analysis(self, jump_epoch, circuit_data):
        """Create visualizations for circuit analysis"""
        import matplotlib.pyplot as plt
        import networkx as nx
        import seaborn as sns

        viz_dir = self.save_dir / "circuit_analysis"
        viz_dir.mkdir(exist_ok=True, parents=True)

        # 1. Component Attribution Bar Chart
        if circuit_data['component_attribution']:
            components = []
            attributions = []

            for comp, attr in sorted(circuit_data['component_attribution'].items(),
                                     key=lambda x: x[1], reverse=True):
                components.append(comp)
                attributions.append(attr)

            # Create bar chart
            plt.figure(figsize=(12, 6))
            plt.bar(components, attributions)
            plt.axhline(y=0.01, color='r', linestyle='--', label='Threshold')
            plt.xlabel('Component')
            plt.ylabel('Attribution (Accuracy Decrease When Ablated)')
            plt.title(f'Component Attribution at Jump {jump_epoch}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(viz_dir / f"jump_{jump_epoch}_component_attribution.png")
            plt.close()

        # 2. Circuit Interaction Network Graph
        if circuit_data['pairwise_interactions']:
            plt.figure(figsize=(10, 8))

            # Create a graph
            G = nx.Graph()

            # Add nodes for circuit components
            for comp in circuit_data['circuit_components']:
                attr_score = circuit_data['component_attribution'][comp]
                G.add_node(comp, weight=attr_score)

            # Add edges for interactions
            for pair, data in circuit_data['pairwise_interactions'].items():
                comp1, comp2 = pair.split('+')

                if data['is_circuit']:
                    # Only add edges for positive interactions (circuit-like behavior)
                    G.add_edge(comp1, comp2, weight=data['interaction'])

            # Set node sizes based on attribution
            node_sizes = [G.nodes[node]['weight'] * 5000 for node in G.nodes]

            # Set edge widths based on interaction strength
            edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges]

            # Position nodes using spring layout
            pos = nx.spring_layout(G, seed=42)

            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8,
                                   node_color='lightblue')
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5,
                                   edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_size=10)

            plt.title(f'Circuit Interaction Network at Jump {jump_epoch}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(viz_dir / f"jump_{jump_epoch}_circuit_network.png")
            plt.close()

        # 3. Heatmap of Pairwise Interactions
        if circuit_data['pairwise_interactions']:
            import pandas as pd

            # Extract interaction values
            pairs = []
            interactions = []

            for pair, data in circuit_data['pairwise_interactions'].items():
                comp1, comp2 = pair.split('+')
                pairs.append(pair)
                interactions.append(data['interaction'])

            # Create a matrix representation
            unique_comps = sorted(list(circuit_data['circuit_components'].keys()))
            n_comps = len(unique_comps)

            if n_comps > 1:
                interaction_matrix = np.zeros((n_comps, n_comps))

                # Fill in the matrix
                for pair, data in circuit_data['pairwise_interactions'].items():
                    comp1, comp2 = pair.split('+')

                    i = unique_comps.index(comp1)
                    j = unique_comps.index(comp2)

                    interaction_matrix[i, j] = data['interaction']
                    interaction_matrix[j, i] = data['interaction']  # Symmetric

                # Create heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(interaction_matrix, annot=True, fmt=".3f",
                            xticklabels=unique_comps, yticklabels=unique_comps,
                            cmap='coolwarm', center=0, vmin=self.heatmap_min, vmax=self.heatmap_max,)

                plt.title(f'Pairwise Component Interactions at Jump {jump_epoch}')
                plt.tight_layout()
                plt.savefig(viz_dir / f"jump_{jump_epoch}_interaction_heatmap.png")
                plt.close()



# Define the SparseAutoencoder class that was referenced
class SparseAutoencoder(torch.nn.Module):
    """
    Sparse autoencoder for analyzing neural network representations.
    """

    def __init__(self, input_dim, code_dim, l1_coef=0.001):
        """
        Initialize the sparse autoencoder.

        Args:
            input_dim: Dimension of input features
            code_dim: Dimension of sparse code (typically larger than input_dim)
            l1_coef: L1 regularization coefficient for sparsity
        """
        super().__init__()

        self.input_dim = input_dim
        self.code_dim = code_dim
        self.l1_coef = l1_coef

        # Encoder (input -> code)
        self.encoder = torch.nn.Linear(input_dim, code_dim, bias=True)

        # Decoder (code -> reconstruction)
        self.decoder = torch.nn.Linear(code_dim, input_dim, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values"""
        torch.nn.init.xavier_normal_(self.encoder.weight, gain=1.0)
        torch.nn.init.xavier_normal_(self.decoder.weight, gain=1.0)
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            tuple: (reconstruction, code)
        """
        # Encode (ReLU activation for non-negative codes)
        code = torch.nn.functional.relu(self.encoder(x))

        # Decode
        reconstruction = self.decoder(code)

        return reconstruction, code

    def l1_loss(self, codes):
        """
        Calculate L1 sparsity loss on codes.

        Args:
            codes: Activation codes

        Returns:
            torch.Tensor: L1 loss
        """
        return self.l1_coef * torch.sum(torch.abs(codes))

    def encode(self, x):
        """
        Encode inputs to sparse codes.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Sparse codes
        """
        return torch.nn.functional.relu(self.encoder(x))

    def decode(self, code):
        """
        Decode sparse codes to reconstructions.

        Args:
            code: Code tensor

        Returns:
            torch.Tensor: Reconstructed inputs
        """
        return self.decoder(code)

# Example usage of the complete framework
def analyze_model_across_grokking(model_path, pre_grokking_ckpt, during_grokking_ckpt,
                                  post_grokking_ckpt, train_loader, eval_loader, device="cuda"):
    """
    Analyze a model across different grokking phases.

    Args:
        model_path: Path to the model definition
        pre_grokking_ckpt: Path to checkpoint before grokking
        during_grokking_ckpt: Path to checkpoint during grokking transition
        post_grokking_ckpt: Path to checkpoint after grokking
        train_loader: Training data loader
        eval_loader: Evaluation data loader
        device: Device to run analysis on

    Returns:
        dict: Analysis results
    """
    # Load the model
    model = torch.load(model_path)
    model.to(device)

    # Create analyzer
    analyzer = AttentionMLPAnalyzer(
        model=model,
        eval_loader=eval_loader,
        device=device,
        save_dir="grokking_analysis_results"
    )

    # Define phase checkpoints
    phase_checkpoints = [
        ("pre_grokking", pre_grokking_ckpt),
        ("during_transition", during_grokking_ckpt),
        ("post_grokking", post_grokking_ckpt)
    ]

    # Run analysis across phases
    phase_analysis = analyzer.analyze_across_grokking_phases(
        phase_checkpoints=phase_checkpoints
    )

    # Create sparse autoencoders for each phase
    autoencoder_results = {}

    for phase_name, ckpt_path in phase_checkpoints:
        print(f"Training sparse autoencoder for {phase_name} phase...")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Create autoencoder
        hidden_dim = model.dim  # Assuming this is the hidden dimension
        code_dim = hidden_dim * 2  # Make code dimension larger
        autoencoder = analyzer.create_sparse_autoencoder(hidden_dim, code_dim)

        # Analyze MLP layer
        mlp_results = analyzer.analyze_with_sparse_autoencoder(
            autoencoder=autoencoder,
            component='mlp',
            layer_idx=0,  # Analyze first layer
            train_epochs=50
        )

        # Visualize results
        mlp_vis = analyzer.visualize_autoencoder_results(
            results=mlp_results,
            component='mlp',
            layer_idx=0,
            epoch=phase_name
        )

        # Store results
        autoencoder_results[f"{phase_name}_mlp"] = {
            'analysis': mlp_results,
            'visualizations': mlp_vis
        }

        # Create new autoencoder for attention
        autoencoder = analyzer.create_sparse_autoencoder(hidden_dim, code_dim)

        # Analyze attention layer
        attn_results = analyzer.analyze_with_sparse_autoencoder(
            autoencoder=autoencoder,
            component='attention',
            layer_idx=0,  # Analyze first layer
            train_epochs=50
        )

        # Visualize results
        attn_vis = analyzer.visualize_autoencoder_results(
            results=attn_results,
            component='attention',
            layer_idx=0,
            epoch=phase_name
        )

        # Store results
        autoencoder_results[f"{phase_name}_attention"] = {
            'analysis': attn_results,
            'visualizations': attn_vis
        }

    return {
        'phase_analysis': phase_analysis,
        'autoencoder_results': autoencoder_results
    }
