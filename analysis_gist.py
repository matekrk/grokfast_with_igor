import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class Block(nn.Module):
    """Causal transformer block with analysis capabilities
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Store attention weights for analysis
        self.attention_weights = None
        self.store_attention_weights = False

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask[torch.isnan(attn_mask)] = 0.0  # fixes all 'nan' on 'mps' device

        x_norm = self.ln_1(x)

        # Get attention outputs and optionally store weights
        if self.store_attention_weights:
            a, self.attention_weights = self.attn(
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False  # Get per-head weights
            )
        else:
            a, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)

        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x

    def get_head_norms(self):
        """Calculate norms for each attention head"""
        head_norms = {}

        # Extract norms for query, key, value projections
        # Assumes standard PyTorch MultiheadAttention implementation
        q_weight = self.attn.in_proj_weight[:self.dim]
        k_weight = self.attn.in_proj_weight[self.dim:2 * self.dim]
        v_weight = self.attn.in_proj_weight[2 * self.dim:3 * self.dim]

        # Calculate norms for each head
        for head_idx in range(self.num_heads):
            head_start = head_idx * self.head_dim
            head_end = (head_idx + 1) * self.head_dim

            # Extract head-specific weights
            q_head = q_weight[head_start:head_end, :]
            k_head = k_weight[head_start:head_end, :]
            v_head = v_weight[head_start:head_end, :]

            # Calculate norms
            q_norm = torch.norm(q_head).item()
            k_norm = torch.norm(k_head).item()
            v_norm = torch.norm(v_head).item()

            # Get output projection norm
            out_norm = torch.norm(self.attn.out_proj.weight[:, head_start:head_end]).item()

            # Store combined and individual norms
            head_norms[f'head_{head_idx}_combined'] = q_norm * k_norm * v_norm * out_norm
            head_norms[f'head_{head_idx}_q'] = q_norm
            head_norms[f'head_{head_idx}_k'] = k_norm
            head_norms[f'head_{head_idx}_v'] = v_norm
            head_norms[f'head_{head_idx}_out'] = out_norm

        return head_norms

    def get_mlp_norms(self):
        """Calculate norms for MLP weights"""
        mlp_norms = {}

        # Get weights for the two linear layers
        mlp_up = self.mlp[0].weight  # dim*4 x dim
        mlp_down = self.mlp[2].weight  # dim x dim*4

        # Calculate Frobenius norms
        mlp_norms['up'] = torch.norm(mlp_up).item()
        mlp_norms['down'] = torch.norm(mlp_down).item()
        mlp_norms['combined'] = mlp_norms['up'] * mlp_norms['down']

        return mlp_norms


class Decoder(nn.Module):
    """Causal Transformer decoder with analysis capabilities
    """

    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(dim=dim, num_heads=num_heads))

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

        # Analysis settings
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.seq_len = seq_len

        # History tracking for analysis
        self.train_history = defaultdict(list)
        self.current_step = 0

    def forward(self, x, store_attention=False):
        # Set attention storage for this forward pass
        for layer in self.layers:
            layer.store_attention_weights = store_attention

        h = self.token_embeddings(x)
        positions = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)

        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits

    def get_attention_patterns(self):
        """Extract attention patterns from all layers"""
        patterns = {}
        for layer_idx, layer in enumerate(self.layers):
            if layer.attention_weights is not None:
                # Reshape to separate the heads
                # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
                for head_idx in range(self.num_heads):
                    patterns[f'layer_{layer_idx}_head_{head_idx}'] = layer.attention_weights[0, head_idx].detach().cpu()
        return patterns

    def log_training_stats(self, loss=None, accuracy=None):
        """Log training statistics and model norms for the current step"""
        self.train_history['step'].append(self.current_step)

        if loss is not None:
            self.train_history['loss'].append(loss)

        if accuracy is not None:
            self.train_history['accuracy'].append(accuracy)

        # Log attention head norms
        for layer_idx, layer in enumerate(self.layers):
            head_norms = layer.get_head_norms()
            for head_name, norm in head_norms.items():
                self.train_history[f'layer_{layer_idx}_{head_name}_norm'].append(norm)

            # Log MLP norms
            mlp_norms = layer.get_mlp_norms()
            for mlp_name, norm in mlp_norms.items():
                self.train_history[f'layer_{layer_idx}_mlp_{mlp_name}_norm'].append(norm)

        self.current_step += 1

    def analyze_head_attribution(self, eval_loader):
        """Measure impact of each attention head by masking"""
        # Evaluate baseline performance
        baseline_accuracy = self.evaluate(eval_loader)
        attribution_scores = {}

        # For each layer and head
        for layer_idx, layer in enumerate(self.layers):
            for head_idx in range(self.num_heads):
                # Store original weights
                original_weights = layer.attn.out_proj.weight.clone()

                # Create a mask for this specific head
                head_dim = self.dim // self.num_heads
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim

                # Zero out this head's contribution
                with torch.no_grad():
                    layer.attn.out_proj.weight[:, start_idx:end_idx] = 0

                # Evaluate with this head masked
                masked_accuracy = self.evaluate(eval_loader)

                # Restore original weights
                with torch.no_grad():
                    layer.attn.out_proj.weight.copy_(original_weights)

                # Record attribution (impact of removing this head)
                attribution_scores[f'layer_{layer_idx}_head_{head_idx}'] = baseline_accuracy - masked_accuracy

        return attribution_scores

    def compute_attention_entropy(self, eval_loader):
        """Measure entropy of attention distributions as a specialization metric"""
        entropies = defaultdict(list)

        # Set to evaluation mode
        self.eval()

        for inputs, _ in eval_loader:
            # Forward pass with attention storage
            _ = self(inputs, store_attention=True)

            # Get attention patterns
            patterns = self.get_attention_patterns()

            # Calculate entropy for each layer and head
            for pattern_name, pattern in patterns.items():
                # Compute entropy for each query position
                for query_pos in range(pattern.shape[0]):
                    probs = pattern[query_pos]
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                    entropies[pattern_name].append(entropy)

        # Calculate average entropy per head
        avg_entropies = {head: sum(vals) / len(vals) for head, vals in entropies.items()}
        return avg_entropies

    def evaluate(self, eval_loader):
        """Simple evaluation function that returns accuracy"""
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in eval_loader:
                outputs = self(inputs)
                # Assuming last token prediction
                last_token_preds = outputs[-1].argmax(dim=-1)
                correct += (last_token_preds == targets).sum().item()
                total += targets.size(0)

        return correct / total if total > 0 else 0

    def visualize_attention(self, sample_input, title=None):
        """Visualize attention patterns for a sample input"""
        # Forward pass with attention storage
        _ = self(sample_input, store_attention=True)

        # Get attention patterns
        patterns = self.get_attention_patterns()

        if not patterns:
            print("No attention patterns were stored during forward pass.")
            return

        # Create a figure with a subplot for each layer and head
        n_layers = self.num_layers
        n_heads = self.num_heads
        fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 3, n_layers * 3))

        if n_layers == 1 and n_heads == 1:
            axes = np.array([[axes]])
        elif n_layers == 1:
            axes = axes.reshape(1, -1)
        elif n_heads == 1:
            axes = axes.reshape(-1, 1)

        # Plot each attention pattern
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                pattern_key = f'layer_{layer_idx}_head_{head_idx}'

                if pattern_key in patterns:
                    ax = axes[layer_idx, head_idx]
                    sns.heatmap(patterns[pattern_key], cmap='viridis', ax=ax)
                    ax.set_title(f'Layer {layer_idx}, Head {head_idx}')
                    ax.set_xlabel('Key position')
                    ax.set_ylabel('Query position')

        plt.tight_layout()
        if title:
            fig.suptitle(title, fontsize=16)
            plt.subplots_adjust(top=0.9)

        plt.show()
        return fig

    def plot_training_dynamics(self):
        """Plot training dynamics from recorded history"""
        history = self.train_history

        if not history:
            print("No training history recorded yet.")
            return

        # Create figures for different aspects
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        steps = history['step']

        # Loss and accuracy
        if 'loss' in history and 'accuracy' in history:
            ax1.plot(steps, history['loss'], 'b-', label='Loss')
            ax2 = ax1.twinx()
            ax2.plot(steps, history['accuracy'], 'r-', label='Accuracy')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss', color='b')
            ax2.set_ylabel('Accuracy', color='r')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            plt.title('Training Dynamics')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
            plt.tight_layout()
            plt.show()

        # Plot head norms for each layer
        for layer_idx in range(self.num_layers):
            fig2, ax = plt.subplots(figsize=(10, 6))

            for head_idx in range(self.num_heads):
                key = f'layer_{layer_idx}_head_{head_idx}_combined_norm'
                if key in history:
                    ax.plot(steps, history[key], label=f'Head {head_idx}')

            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Head Norm')
            ax.set_title(f'Layer {layer_idx} Head Norms')
            ax.legend()
            plt.tight_layout()
            plt.show()

        # Plot MLP norms for each layer
        fig3, ax = plt.subplots(figsize=(10, 6))
        for layer_idx in range(self.num_layers):
            key = f'layer_{layer_idx}_mlp_combined_norm'
            if key in history:
                ax.plot(steps, history[key], label=f'Layer {layer_idx}')

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('MLP Norm')
        ax.set_title('MLP Norms')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def identify_grokking_phase(self):
        """Identify pre-grokking, grokking transition, and post-grokking phases"""
        if 'accuracy' not in self.train_history or len(self.train_history['accuracy']) < 10:
            print("Not enough training data to identify grokking phases.")
            return None

        # Convert to numpy for analysis
        steps = np.array(self.train_history['step'])
        accs = np.array(self.train_history['accuracy'])

        # Compute moving average for smoothing
        window = min(10, len(accs) // 5)
        smoothed_accs = np.convolve(accs, np.ones(window) / window, mode='valid')
        smoothed_steps = steps[window - 1:]

        # Calculate rate of change (discrete derivative)
        acc_change = np.diff(smoothed_accs)
        change_steps = smoothed_steps[:-1]  # One fewer point after diff

        # Find where rate of change is highest (potential grokking point)
        if len(acc_change) > 0:
            grokking_idx = np.argmax(acc_change)
            grokking_step = change_steps[grokking_idx]

            # Define the phases
            phase_width = max(len(steps) // 10, 5)  # Use at least 5 steps or 10% of data

            pre_grokking = steps < grokking_step - phase_width
            transition = (steps >= grokking_step - phase_width) & (steps <= grokking_step + phase_width)
            post_grokking = steps > grokking_step + phase_width

            phases = {
                'pre_grokking_mask': pre_grokking,
                'transition_mask': transition,
                'post_grokking_mask': post_grokking,
                'grokking_step': grokking_step,
                'pre_grokking_steps': steps[pre_grokking],
                'transition_steps': steps[transition],
                'post_grokking_steps': steps[post_grokking]
            }

            # Plot the identified phases
            plt.figure(figsize=(12, 6))
            plt.plot(steps, accs, 'b-', alpha=0.5, label='Raw Accuracy')
            plt.plot(smoothed_steps, smoothed_accs, 'b-', linewidth=2, label='Smoothed Accuracy')

            # Highlight the phases
            if np.any(pre_grokking):
                plt.axvspan(min(steps[pre_grokking]), max(steps[pre_grokking]),
                            alpha=0.2, color='red', label='Pre-Grokking')

            if np.any(transition):
                plt.axvspan(min(steps[transition]), max(steps[transition]),
                            alpha=0.2, color='yellow', label='Transition')

            if np.any(post_grokking):
                plt.axvspan(min(steps[post_grokking]), max(steps[post_grokking]),
                            alpha=0.2, color='green', label='Post-Grokking')

            plt.axvline(x=grokking_step, color='r', linestyle='--', label=f'Grokking Point: Step {grokking_step}')

            plt.xlabel('Training Steps')
            plt.ylabel('Accuracy')
            plt.title('Identified Grokking Phases')
            plt.legend()
            plt.tight_layout()
            plt.show()

            return phases
        else:
            print("Not enough data points to identify grokking after smoothing.")
            return None


# Training loop with analysis
def train_with_analysis(model, train_loader, eval_loader, criterion, optimizer,
                        epochs, device, log_interval=10, analyze_interval=100):
    """
    Train the model with periodic analysis and logging

    Parameters:
    -----------
    model : Decoder
        The transformer model with analysis capabilities
    train_loader : DataLoader
        Training data loader
    eval_loader : DataLoader
        Evaluation data loader
    criterion : loss function
        Loss function for training
    optimizer : optimizer
        Optimizer for training
    epochs : int
        Number of training epochs
    device : torch.device
        Device to train on
    log_interval : int
        How often to log basic metrics (steps)
    analyze_interval : int
        How often to perform detailed analysis (steps)
    """
    model.to(device)
    step = 0

    for epoch in range(epochs):
        model.train()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Basic logging at regular intervals
            if step % log_interval == 0:
                # Evaluate
                model.eval()
                with torch.no_grad():
                    accuracy = model.evaluate(eval_loader)
                model.train()

                # Log stats
                model.log_training_stats(loss=loss.item(), accuracy=accuracy)

                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

            # Detailed analysis at less frequent intervals
            if step % analyze_interval == 0:
                # Store a sample input for visualization
                sample_input = next(iter(eval_loader))[0].to(device)

                # Visualize attention patterns
                model.visualize_attention(sample_input, title=f"Attention Patterns at Step {step}")

                # Calculate head attribution if we're past the early training stage
                if step > 0 and step % (analyze_interval * 5) == 0:
                    print("Analyzing head attribution...")
                    attribution = model.analyze_head_attribution(eval_loader)

                    # Plot attribution scores
                    plt.figure(figsize=(10, 6))
                    heads = list(attribution.keys())
                    scores = [attribution[h] for h in heads]
                    plt.bar(heads, scores)
                    plt.xlabel('Attention Head')
                    plt.ylabel('Attribution Score (decrease in accuracy when masked)')
                    plt.title(f'Head Attribution at Step {step}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()

                    # Also calculate attention entropy
                    entropies = model.compute_attention_entropy(eval_loader)

                    # Plot entropy
                    plt.figure(figsize=(10, 6))
                    heads = list(entropies.keys())
                    entropy_vals = [entropies[h] for h in heads]
                    plt.bar(heads, entropy_vals)
                    plt.xlabel('Attention Head')
                    plt.ylabel('Attention Entropy')
                    plt.title(f'Attention Entropy at Step {step} (lower = more specialized)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()

            step += 1

        # At the end of each epoch, try to identify grokking phases
        if epoch > 0 and len(model.train_history['accuracy']) > 20:
            print(f"\nAnalyzing training dynamics after epoch {epoch}...")
            model.plot_training_dynamics()
            phases = model.identify_grokking_phase()

            if phases and 'grokking_step' in phases:
                print(f"Potential grokking detected at step {phases['grokking_step']}")

        print(f"Completed epoch {epoch}")

    # Final detailed analysis
    print("\nFinal Analysis:")
    model.plot_training_dynamics()
    phases = model.identify_grokking_phase()

    # Compare attention patterns before and after grokking
    if phases:
        # Get sample input for visualization
        sample_input = next(iter(eval_loader))[0].to(device)

        if np.any(phases['pre_grokking_mask']):
            # Find a representative pre-grokking step
            pre_step = phases['pre_grokking_steps'][-1]
            print(f"Analyzing pre-grokking attention (step {pre_step})...")

            # We can't go back in time, so this is just illustrative
            model.visualize_attention(sample_input, title=f"Pre-Grokking Attention (Step {pre_step})")

        print(f"Analyzing post-grokking attention (latest step)...")
        model.visualize_attention(sample_input, title="Post-Grokking Attention (Latest)")

    return model