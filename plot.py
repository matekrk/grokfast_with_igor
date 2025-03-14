import re

import pandas as pd
import torch
import numpy as np
from numpy.ma.core import indices
from torch import nn
import torch.nn.functional as F




def get_matrix_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.dim() == 2:  # Only for matrices
            norms[f'{name}_frobenius'] = torch.norm(param, p='fro').item()
            norms[f'{name}_nuclear'] = torch.linalg.norm(param, ord='nuc').item()
            norms[f'{name}_spectral'] = torch.linalg.norm(param, ord=2).item()

            # Rank analysis
            U, S, V = torch.linalg.svd(param)
            effective_rank = torch.sum(S > 1e-5).item()  # threshold as needed
            norms[f'{name}_rank'] = effective_rank

            # Condition number
            norms[f'{name}_condition'] = S[0] / S[-1]

    return norms


def analyze_activations(model, dataloader):
    activations = []

    def hook_fn(module, input, output):
        # Store activations
        activations.append(output.detach().cpu())

    # Register hooks for layers you want to analyze
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  # or other layer types
            hooks.append(module.register_forward_hook(hook_fn))

    # Collect activations
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            _ = model(batch)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze activation patterns
    for act in activations:
        # Compute covariance matrix
        act_flat = act.reshape(-1, act.shape[-1])
        cov = torch.cov(act_flat.T)

        # Get eigenvalues for dimensionality analysis
        eigenvals = torch.linalg.eigvalsh(cov)

        # Participation ratio (measure of how many dimensions are effectively used)
        PR = torch.sum(eigenvals) ** 2 / torch.sum(eigenvals ** 2)

    return


def analyze_superposition(model, features):
    # Get weight matrix (e.g., from MLP or attention)
    W = model.mlp.weight.detach()

    # SVD analysis
    U, S, V = torch.linalg.svd(W)

    # Compute feature-direction alignment
    alignments = torch.mm(features, V.T)

    # Analyze how features are distributed across singular directions
    feature_distribution = torch.sum(alignments ** 2, dim=0)

    # Measure feature interference
    interference_matrix = torch.mm(features, features.T)

    return {
        'singular_values': S,
        'feature_distribution': feature_distribution,
        'interference': interference_matrix
    }


def geometric_analysis(representations):
    # Cosine similarities between different representations
    cos_sim = F.cosine_similarity(representations.unsqueeze(1),
                                  representations.unsqueeze(0), dim=2)

    # Analyze clustering structure
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10)  # adjust number as needed
    clusters = kmeans.fit_predict(representations.numpy())

    # Measure isotropy (how uniformly distributed the representations are)
    mean_direction = torch.mean(representations, dim=0)
    isotropy = torch.norm(mean_direction) / torch.mean(torch.norm(representations, dim=1))

    return cos_sim, isotropy


import seaborn as sns
import matplotlib.pyplot as plt


def visualize_analysis(analysis_results):
    # Plot singular value spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(analysis_results['singular_values'])
    plt.yscale('log')
    plt.title('Singular Value Spectrum')
    plt.show()

    # Visualize feature interference
    sns.heatmap(analysis_results['interference'])
    plt.title('Feature Interference Matrix')
    plt.show()

def plot_single_dict(ax, ddd, this_ddd_key, smooth=True):
    # info plot dict dd to ax
    d = {}
    ddd_keys = sorted(ddd.keys())
    ddd_n = range(len(ddd_keys))
    for key, n in zip(ddd_keys, ddd_n):
        if key == 'epoch':
            continue
        d[key] = np.zeros((len(ddd['epoch']), 3))
        d[key][:, 0], d[key][:, 1] = np.array(ddd['epoch']), np.array(ddd[key])
        d[key][:, 2] = n
    nrm_np = np.concatenate(list(d.values()), axis=0)
    nrm_df = pd.DataFrame(nrm_np, columns=['step', 'val', 'key'])
    nrm_df = nrm_df.astype({'step': int, 'val': float, 'key': int})
    reversed_dict = {n: k for k, n in zip(ddd_keys, ddd_n)}
    nrm_df['key'] = nrm_df['key'].replace(reversed_dict).values
    if smooth:
        window_size=5
        smoothed_df = nrm_df.copy()
        for category, group in smoothed_df.groupby('key'):
            group = group.sort_values(by='step')
            indices = group.index
            smoothed_values = group['val'].rolling(window=window_size, center=True, min_periods=1).mean()
            smoothed_df.loc[indices, 'smoothed_val'] = smoothed_values
        g_res = sns.lineplot(x='step', y='smoothed_val', hue='key', data=smoothed_df, ci=95, ax=ax)
    else:
        g_res = sns.lineplot(data=nrm_df, x="step", y="val", hue="key", errorbar=('ci', 95), estimator='mean', ax=ax)
    g_res.set(xscale='log')
    if this_ddd_key is not None and this_ddd_key != 'accuracy':
        g_res.set(yscale='log')

    return ax


def plot_dicts(dres, plot_infix):
    # info plot a series of results in dictionaries that share a common x value steps
    # info into an array of rows x 2
    sns.set_theme(
        style="whitegrid",
        font="sans-serif",
        font_scale=0.5,
        palette=sns.color_palette('pastel'),

        rc={
            "lines.linewidth": 1.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
        },)

    nof = len(dres.keys())
    nrows = nof % 2 + nof // 2
    height, width = 7, 8
    fig, axs = plt.subplots(nrows=nrows, ncols=2,
                            sharex=True, sharey=False, figsize=(width, width))
    axs_list = axs.flatten().tolist()
    indx = 0
    for k, v in dres.items():
        if not isinstance(v, dict):
            print(f"plot_dicts(): {k} is not a dict")
        ax = axs_list[indx]
        ax = plot_single_dict(ax=ax, ddd=v, this_ddd_key=k)
        ax.set_title(k)
        ax.set_ylabel('')
        if indx % 2 != 0:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=8)
        indx += 1
    plt.tight_layout(pad=0.5)
    plt.suptitle("Transformer grokking")
    plt.savefig(f"results/dicts_{plot_infix}.png", dpi=300)
    plt.close()

def train_eval_to_dict():
    pass



