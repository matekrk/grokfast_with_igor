import re

import numpy as np
import torch
from scipy.stats import entropy


def get_parameter_norms(model):
    norms = {
        # 'attention': [],
        # 'mlp': [],
        # 'total': [],
        # 'head': []
    }

    for name, param in model.named_parameters():
        # Compute Frobenius norm
        param_norm = torch.norm(param).item()
        if 'layers.0' in name:
            lll = "lay.0"
        elif 'layers.1' in name:
            lll = "lay.1"
        elif 'layers.2' in name:
            lll = "lay.2"
        else:
            lll = None

        # Categorize based on parameter name
        if 'attention' in name or 'attn' in name:
            key = 'attn'
            norms = add_key(norms, key)
            norms[key].append(param_norm)
            if lll is not None:
                key = f"{key}_{lll}"
                norms = add_key(norms, key)
                norms[key].append(param_norm)
        elif 'mlp' in name or 'ffn' in name:  # depending on your naming
            key = 'mlp'
            norms = add_key(norms, key)
            norms[key].append(param_norm)
            if lll is not None:
                key = f"{key}_{lll}"
                norms = add_key(norms, key)
                norms[key].append(param_norm)
        # elif 'head' in name:
        #     norms['head'].append(param_norm)

        # if not ('head' in name or 'ln_f' in name or 'ln_1' in name or 'ln_2' in name):
        #     norms['total'].append(param_norm)

    mean_norms = {}
    for key in norms.keys():
        mean_norms[key] = np.mean(norms[key])

    return mean_norms


def add_key(d, key):
    if key not in d:
        d[key] = []
    return d



def get_detailed_norms(model):
    details = {}

    # Per-layer norms
    # for name, param in model.named_parameters():
    #     details[f'{name}_norm'] = torch.norm(param).item()

    # Per-head attention norms (if needed)
    for name, module in model.named_modules():
        if hasattr(module, 'query'):  # assuming attention heads
            q_norm = torch.norm(module.query.weight).item()
            k_norm = torch.norm(module.key.weight).item()
            v_norm = torch.norm(module.value.weight).item()
            details[f'{name}_qkv_norms'] = (q_norm, k_norm, v_norm)

    return details

def get_detailed_joined_attn_norms(model):
    attn_norms = {}
    # Per-layer norms
    # for name, param in model.named_parameters():
    #     details[f'{name}_norm'] = torch.norm(param).item()

    # Per-head attention norms (if needed)
    for name, module in model.named_modules():
        if 'layers' in name and ('attention' in name or 'attn' in name):
            if hasattr(module, 'in_proj_weight'):
                m = re.search(r'layers\.([0-9]+)\.', name)
                lay_no = int(m.group(1))
                weight = module.in_proj_weight[lay_no]
                q_weight, k_weight, v_weight = weight.chunk(3)
                attn_norms[f"q_norm_{lay_no}"] = torch.norm(q_weight).item()
                attn_norms[f"k_norm_{lay_no}"] = torch.norm(k_weight).item()
                attn_norms[f"v_norm_{lay_no}"] = torch.norm(v_weight).item()

        # if hasattr(module, 'query'):  # assuming attention heads
        #     q_norm = torch.norm(module.query.weight).item()
        #     k_norm = torch.norm(module.key.weight).item()
        #     v_norm = torch.norm(module.value.weight).item()
        #     details[f'{name}_qkv_norms'] = (q_norm, k_norm, v_norm)

    return attn_norms


def analyze_attention_heads(model, sample_input):
    # Get attention weights for all heads
    # Assuming model outputs attention weights or has hooks to access them
    attention_weights = get_model_attention_weights(model, sample_input)

    # For each layer and each head
    for layer_idx, layer_attn in enumerate(attention_weights):
        for head_idx, head_attn in enumerate(layer_attn):
            # Calculate metrics
            attn_norm = torch.norm(head_attn, p='fro').item()
            attn_entropy = entropy(head_attn.flatten().cpu().numpy())
            attn_sparsity = gini_coefficient(head_attn.cpu().numpy())

            print(f"Layer {layer_idx}, Head {head_idx}:")
            print(f"  Norm: {attn_norm:.4f}")
            print(f"  Entropy: {attn_entropy:.4f}")
            print(f"  Sparsity: {attn_sparsity:.4f}")

            # Visualize attention pattern
            plt.figure(figsize=(8, 6))
            plt.imshow(head_attn.cpu().numpy(), cmap='viridis')
            plt.title(f"Layer {layer_idx}, Head {head_idx}")
            plt.colorbar()
            plt.savefig(f"attn_layer{layer_idx}_head{head_idx}.png")
            plt.close()
