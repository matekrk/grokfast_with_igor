import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from logger import DataLogger


class Block(nn.Module):
    """Causal transformer block with analysis capabilities
    """

    def __init__(self, dim, num_heads, mlp_hidden_mult=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_hidden_mult),
            nn.GELU(),
            nn.Linear(dim * mlp_hidden_mult, dim),
        )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mlp_hidden_mult = mlp_hidden_mult

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

    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97,
                 seq_len=5, ratio=0.5,
                 criterion=nn.CrossEntropyLoss(), device='cpu', id=None,
                 save_dir=None, checkpoint_dir=None):
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
        self.ratio = ratio

        self.device = device

        # loss function for evaluation
        self.criterion = criterion

        # History tracking for analysis
        self.id = id if id else f"decoder_l{num_layers}_h{num_heads}_e{dim}_t{num_tokens}_s{seq_len}_r{ratio:.1f}"
        self.train_history = defaultdict(list)
        self.logger = DataLogger(id=self.id)
        self.current_step = 0
        self.current_epoch = 0

        # plotting name
        self.plot_prefix = f"{self.id}"
        print(f"plot_prefix: {self.plot_prefix}")

        # savedir
        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir

        # info init seaborn theme
        sns.set_theme(
            style="whitegrid",
            font="sans-serif",
            font_scale=1.0,
            palette=sns.color_palette("pastel"),

            rc={
                "lines.linewidth": 1.0,
                "axes.spines.right": False,
                "axes.spines.top": False,
            }, )

    def forward(self, x, store_attention=False):
        # Set attention storage for this forward pass
        for layer in self.layers:
            layer.store_attention_weights = store_attention

        # Check if input is batch-first and transpose if needed
        # x should be in format [seq_len, batch_size] for the rest of the model
        if len(x.shape) > 1 and x.shape[0] > x.shape[1]:  # Likely batch-first format
            x = x.transpose(0, 1)  # Convert to sequence-first

        h = self.token_embeddings(x)
        positions = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)

        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits[-1]

    def get_id(self):
        return self.id

    def apply_xavier_init(self):
        """
        Apply Xavier/Glorot initialization to all weights in the model.
        This can help speed up grokking by providing a better initialization.
        """

        def _xavier_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Apply Xavier uniform initialization to Linear and Embedding layers
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    # Initialize biases to zero
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Initialize LayerNorm weights to 1 and biases to 0
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Apply initialization to all modules
        self.apply(_xavier_init)

        # Special handling for attention weights which need to be divided by sqrt(head_dim)
        for layer in self.layers:
            if hasattr(layer.attn, 'in_proj_weight') and layer.attn.in_proj_weight is not None:
                # For nn.MultiheadAttention, divide by sqrt(head_dim)
                nn.init.xavier_uniform_(layer.attn.in_proj_weight, gain=1 / np.sqrt(self.dim // self.num_heads))

            if hasattr(layer.attn, 'out_proj') and layer.attn.out_proj.weight is not None:
                nn.init.xavier_uniform_(layer.attn.out_proj.weight)
                if layer.attn.out_proj.bias is not None:
                    nn.init.zeros_(layer.attn.out_proj.bias)

        print(f"Applied Xavier initialization to all weights")

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

    def log_training_stats(self, epoch, loss=None, accuracy=None):
        self.current_epoch = epoch
        """Log training statistics and model norms for the current step"""
        # todo move to Datalogger
        self.train_history['step'].append(self.current_step)
        self.train_history['epoch'].append(epoch)

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


    def log_stats(self, category, data_dict):
        self.logger.update_category(category, data_dict=data_dict)

    def analyze_head_attribution(self, eval_loader):
        """Measure impact of each attention head by masking"""
        # Evaluate baseline performance
        baseline_accuracy, baseline_loss = self.evaluate(eval_loader)
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
                masked_accuracy, masked_loss = self.evaluate(eval_loader)

                # Restore original weights
                with torch.no_grad():
                    layer.attn.out_proj.weight.copy_(original_weights)

                # Record attribution (impact of removing this head)
                attribution_scores[f'layer_{layer_idx}_head_{head_idx}'] = baseline_accuracy - masked_accuracy

        return attribution_scores

    def analyze_head_cross_attribution(self, eval_loader):
        """
        Measure impact of pairs of attention heads by masking - optimized version.
        Only computes the lower triangular portion of the matrix and copies values for symmetry.

        Returns: numpy array with columns (layer_f_idx, head_f_idx, layer_s_idx, head_s_idx, attribution_score)
        """
        baseline_accuracy, baseline_loss = self.evaluate(eval_loader)

        # Calculate total number of heads
        total_heads = self.num_layers * self.num_heads

        # Pre-allocate result array for all unique pairs (including diagonal)
        # Number of unique pairs = (n * (n + 1)) / 2
        num_pairs = (total_heads * (total_heads + 1)) // 2
        attr_scores = np.zeros((num_pairs, 5))

        # Create a lookup grid to easily map head pairs to their index in the result array
        # This will help us retrieve results when building the full matrix
        lookup_grid = np.full((total_heads, total_heads), -1)

        k = 0
        for i in range(total_heads):
            # Extract layer and head indices for first head
            layer_f_idx = i // self.num_heads
            head_f_idx = i % self.num_heads

            # Get the layer
            layer_f = self.layers[layer_f_idx]

            # Calculate head dimensions
            head_f_dim = self.dim // self.num_heads
            start_f_idx = head_f_idx * head_f_dim
            end_f_idx = (head_f_idx + 1) * head_f_dim

            # Store original weights
            original_f_weights = layer_f.attn.out_proj.weight.clone()

            # Iterate only through the lower triangular portion (including diagonal)
            for j in range(i + 1):
                # Extract layer and head indices for second head
                layer_s_idx = j // self.num_heads
                head_s_idx = j % self.num_heads

                # Get the layer
                layer_s = self.layers[layer_s_idx]

                # Calculate head dimensions
                head_s_dim = self.dim // self.num_heads
                start_s_idx = head_s_idx * head_s_dim
                end_s_idx = (head_s_idx + 1) * head_s_dim

                # Store original weights
                original_s_weights = layer_s.attn.out_proj.weight.clone()

                # Zero out both heads
                with torch.no_grad():
                    layer_f.attn.out_proj.weight[:, start_f_idx:end_f_idx] = 0
                    layer_s.attn.out_proj.weight[:, start_s_idx:end_s_idx] = 0

                # Evaluate and record
                masked_accuracy, masked_loss = self.evaluate(eval_loader)

                # Store results: layer_f, head_f, layer_s, head_s, score
                attr_scores[k] = (layer_f_idx, head_f_idx, layer_s_idx, head_s_idx, baseline_accuracy - masked_accuracy)

                # Store lookup index
                lookup_grid[i, j] = k
                lookup_grid[j, i] = k  # Store symmetric lookup

                # Restore weights for second head
                with torch.no_grad():
                    layer_s.attn.out_proj.weight.copy_(original_s_weights)

                k += 1

            # Restore weights for first head
            with torch.no_grad():
                layer_f.attn.out_proj.weight.copy_(original_f_weights)

        # Now expand to full matrix if needed
        full_matrix = np.zeros((total_heads * total_heads, 5))
        full_k = 0

        for i in range(total_heads):
            layer_i = i // self.num_heads
            head_i = i % self.num_heads

            for j in range(total_heads):
                layer_j = j // self.num_heads
                head_j = j % self.num_heads

                # Get the lookup index from our grid
                lookup_idx = int(lookup_grid[i, j])

                # Copy the score from the lookup
                score = attr_scores[lookup_idx][4]

                # Store in full matrix
                full_matrix[full_k] = (layer_i, head_i, layer_j, head_j, score)
                full_k += 1

        return full_matrix[:full_k]


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
        total_elem = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self(inputs, store_attention=True)
                loss = self.criterion(logits, targets)
                # Assuming last token prediction
                last_token_preds = logits.argmax(dim=-1)
                correct += (last_token_preds == targets).sum().item()
                total_elem += targets.size(0)
                total_loss += loss.item() * targets.size(0)

        # info probably training mode should be the default (?) make it so
        self.train()

        return (correct / total_elem if total_elem > 0 else 0.0,
                total_loss / total_elem if total_elem > 0 else 0.0)


    def visualize_attention(self, sample_input, title=None):
        """Visualize attention patterns for a sample input"""
        # Forward pass with attention storage
        _ = self(sample_input, store_attention=True)

        # Get attention patterns
        patterns = self.get_attention_patterns()

        if not patterns:
            print("\tNo attention patterns were stored during forward pass.")
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

    def identify_grokking_phase(self, save=True):
        """Identify pre-grokking, grokking transition, and post-grokking phases"""

        if self.logger.get_length('evaluation', 'accuracy') < 10:
            print("Not enough training data to identify grokking phases.")
            return None

        evl_log = self.logger.get_logs("evaluation")
        evl_steps = np.array(evl_log['epoch'])
        evl_accs = np.array(evl_log['accuracy'])

        # Compute moving average for smoothing
        evl_window = min(10, len(evl_accs) // 5)
        evl_smoothed_accs = np.convolve(evl_accs, np.ones(evl_window) / evl_window, mode='valid')
        evl_smoothed_steps = evl_steps[evl_window - 1:]

        # Calculate rate of change (discrete derivative)
        evl_acc_change = np.diff(evl_smoothed_accs)
        evl_change_steps = evl_smoothed_steps[:-1]  # One fewer point after diff

        # Find where rate of change is highest (potential grokking point)
        if len(evl_acc_change) > 0:
            grokking_idx = np.argmax(evl_acc_change)
            grokking_step = evl_change_steps[grokking_idx]

            # Define the phases
            phase_width = max(len(evl_steps) // 10, 5)  # Use at least 5 steps or 10% of data

            pre_grokking = evl_steps < grokking_step - phase_width
            transition = (evl_steps >= grokking_step - phase_width) & (evl_steps <= grokking_step + phase_width)
            post_grokking = evl_steps > grokking_step + phase_width

            phases = {
                'pre_grokking_mask': pre_grokking,
                'transition_mask': transition,
                'post_grokking_mask': post_grokking,
                'grokking_step': grokking_step,
                'pre_grokking_steps': evl_steps[pre_grokking],
                'transition_steps': evl_steps[transition],
                'post_grokking_steps': evl_steps[post_grokking]
            }
            self.log_stats('grokking_phases', phases)

            # Plot the identified phases
            plt.figure(figsize=(12, 6))
            plt.plot(evl_steps, evl_accs, 'b-', alpha=0.5, label='Raw eval accuracy')
            plt.plot(evl_smoothed_steps, evl_smoothed_accs, 'b-', linewidth=2, label='Smoothed eval accuracy')

            # Highlight the phases
            if np.any(pre_grokking):
                plt.axvspan(min(evl_steps[pre_grokking]), max(evl_steps[pre_grokking]),
                            alpha=0.15, color='red', label='Pre-Grokking')

            if np.any(transition):
                plt.axvspan(min(evl_steps[transition]), max(evl_steps[transition]),
                            alpha=0.2, color='yellow', label='Transition')

            if np.any(post_grokking):
                plt.axvspan(min(evl_steps[post_grokking]), max(evl_steps[post_grokking]),
                            alpha=0.2, color='green', label='Post-Grokking')

            plt.axvline(x=grokking_step, color='r', linestyle='--', label=f'Grokking Point: Step {grokking_step}')

            # info get training history too to plot, if available
            if self.logger.get_length('training', 'accuracy') > 100:
                trn_log = self.logger.get_logs("training")
                trn_steps = np.array(trn_log['epoch'])
                trn_accs = np.array(trn_log['accuracy'])

                # Compute moving average for smoothing
                trn_window = min(10, len(trn_accs) // 5)
                trn_smoothed_accs = np.convolve(trn_accs, np.ones(evl_window) / trn_window, mode='valid')
                trn_smoothed_steps = evl_steps[trn_window - 1:]

                plt.plot(trn_steps, trn_accs, 'r-', alpha=0.5, label='Raw train accuracy')
                plt.plot(trn_smoothed_steps, trn_smoothed_accs, 'r-', linewidth=2, label='Smoothed train accuracy')

            plt.xlabel('Training epochs')
            plt.ylabel('Accuracy')
            plt.title('Identified Grokking Phases')
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(f'results/{self.plot_prefix}_grokking_phases.png')
            else:
                plt.show()

            return phases
        else:
            print("Not enough data points to identify grokking after smoothing.")
            return None


    def analyze_head_patterns_changes(self, eval_loader, checkpoint_steps):
        """
        Track changes in attention patterns across critical checkpoints

        Parameters:
        -----------
        eval_loader : DataLoader
            Evaluation data loader
        checkpoint_steps : list
            List of checkpoint steps to analyze
        """
        # Get sample input
        sample_input, _ = next(iter(eval_loader))
        sample_input = sample_input.to(next(self.parameters()).device)

        pattern_changes = {}
        head_specialization = {}

        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                head_id = f'layer_{layer_idx}_head_{head_idx}'
                head_specialization[head_id] = []

        # TODO: Load checkpoints and analyze patterns at each
        for step in checkpoint_steps:
            # Load checkpoint
            # self.load_state_dict(checkpoint_state_dict)

            # Get patterns
            _ = self(sample_input, store_attention=True)
            patterns = self.get_attention_patterns()

            # Analyze pattern structure for each head
            for head_id, pattern in patterns.items():
                # Calculate several pattern metrics

                # 1. Diagonal strength (local attention)
                diagonal_strength = torch.mean(torch.diag(pattern)).item()

                # 2. Entropy (specialization)
                entropy = -torch.sum(pattern * torch.log(pattern + 1e-10)).item()

                # 3. Position bias (where does this head focus)
                position_bias = torch.argmax(torch.mean(pattern, dim=0)).item()

                head_specialization[head_id].append({
                    'step': step,
                    'diagonal_strength': diagonal_strength,
                    'entropy': entropy,
                    'position_bias': position_bias,
                })

        return head_specialization
