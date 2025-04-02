import math
from argparse import ArgumentParser
from datetime import datetime
from itertools import permutations
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import xscale
from tqdm import tqdm
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from analyze import get_parameter_norms, get_detailed_norms, add_key
from plot import plot_dicts, get_matrix_norms
from grokfast import *


class Block(nn.Module):
    """Causal transformer block
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

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask[torch.isnan(attn_mask)] = 0.0 # fixes all 'nan' on 'mps' device

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class Decoder(nn.Module):
    """Causal Transformer decoder
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

    def forward(self, x):
        h = self.token_embeddings(x)
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits


def multiplication_mod_p_data(p, eq_token, op_token):
    """x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p
    """
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = x * y % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result])


def plot_train_eval(steps_per_epoch, train_acc, val_acc, train_loss, val_loss, plot_infix):
    steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
    sns_plot = 'sns'
    sns.set_theme()

    trn_np = np.zeros((len(train_acc), 3))
    trn_np[:, 0], trn_np[:, 1] = steps, np.array(train_acc)
    trn_np[:, 2] = 0
    val_np = np.zeros((len(val_acc), 3))
    val_np[:, 0], val_np[:, 1] = steps, np.array(val_acc)
    val_np[:, 2] = 1
    acc_np = np.vstack((trn_np, val_np))
    acc_df = pd.DataFrame(acc_np, columns=['step', 'val', 'type'])
    acc_df = acc_df.astype({'step': int, 'val': float, 'type': int})
    acc_df['type'] = acc_df['type'].replace({0: 'train', 1: 'valid'}).values
    g_res = sns.lineplot(data=acc_df, x="step", y="val", hue="type")
    g_res.set(xscale='log')
    plt.title("Modular Multiplication (training on 50% of data)")
    plt.ylim(-0.05, 1.05)
    plt.savefig(f"results/acc_{plot_infix}.png")
    plt.close()

    trn_np = np.zeros((len(train_loss), 3))
    trn_np[:, 0], trn_np[:, 1] = steps, np.array(train_loss)
    trn_np[:, 2] = 0
    val_np = np.zeros((len(val_acc), 3))
    val_np[:, 0], val_np[:, 1] = steps, np.array(val_loss)
    val_np[:, 2] = 1
    lss_np = np.vstack((trn_np, val_np[2:]))
    lss_df = pd.DataFrame(lss_np, columns=['step', 'val', 'type'])
    lss_df = lss_df.astype({'step': int, 'val': float, 'type': int})
    lss_df['type'] = lss_df['type'].replace({0: 'train', 1: 'valid'}).values
    g_res = sns.lineplot(data=lss_df, x="step", y="val", hue='type')
    g_res.set(xscale='log')
    plt.title("Modular Multiplication (training on 50% of data)")
    # plt.ylim(0.0, 1.0)
    # plt.show()
    plt.savefig(f"results/loss_{plot_infix}.png")
    plt.close()


def plot_norms(norms_history, plot_infix):
    # steps = torch.arange(len(norms_history['epoch'])).numpy() * steps_per_epoch
    sns_plot = 'sns'
    sns.set_theme()

    # nrm_np = np.zeros((len(norms_history['epoch']) * 3, 4))
    # nrm_np[:, 0], nrm_np[:, 1], nrm_np[:, 2], nrm_np[:, 3] = (steps, np.array(norms_history['attention']),
    #                                                           np.array(norms_history['mlp']), np.array(norms_history['total']))
    d = {}
    # for key in ('attention', 'mlp', 'total'):
    for key, n in zip(norms_history.keys(), range(len(norms_history.keys()))):
        if key == 'epoch':
            continue
        d[key] = np.zeros((len(norms_history['epoch']), 3))
        d[key][:, 0], d[key][:, 1] = (norms_history['epoch'],
                                      np.array(norms_history[key]))
        d[key][:, 2] = n
    # nrm_np = np.vstack(d.values())
    nrm_np = np.concatenate(list(d.values()), axis=0)
    nrm_df = pd.DataFrame(nrm_np, columns=['step', 'val', 'key'])
    nrm_df = nrm_df.astype({'step': int, 'val': float, 'key': int})
    reversed_dict = {n: k for k, n in zip(norms_history.keys(),
                                          range(len(norms_history.keys())))}
    nrm_df['key'] = nrm_df['key'].replace(reversed_dict).values
    g_res = sns.lineplot(data=nrm_df, x="step", y="val", hue="key")
    g_res.set(xscale='log')
    plt.title("Modular Multiplication Transformer norms (on 50% of data)")
    # plt.ylim(-0.05, 1.05)
    plt.savefig(f"results/norms_{plot_infix}.png")
    plt.close()

# def plot_some_results(steps_per_epoch, train_acc, val_acc, train_loss, val_loss):
#     steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
#     sns_plot = 'sns'
#     sns.set_theme()
#
#     steps_np = np.concatenate((steps[:-2], steps[:-2], steps[:-2]))
#
#     trn_np = np.zeros((len(train_acc), 3))
#     trn_np[:, 0], trn_np[:, 1] = steps, np.array(train_acc)
#     trn_np[:, 2] = 0
#     trn_np = np.vstack((trn_np[0:-2, :], trn_np[1:-1, :], trn_np[2:, :]))
#     trn_np[:, 0] = steps_np
#
#     val_np = np.zeros((len(val_acc), 3))
#     val_np[:, 0], val_np[:, 1] = steps, np.array(val_acc)
#     val_np[:, 2] = 1
#     val_np = np.vstack((val_np[0:-2, :], val_np[1:-1, :], val_np[2:, :]))
#     val_np[:, 0] = steps_np
#
#     acc_np = np.vstack((trn_np, val_np))
#
#     acc_df = pd.DataFrame(acc_np, columns=['step', 'val', 'type'])
#     acc_df = acc_df.astype({'step': int, 'val': float, 'type': int})
#     acc_df['type'] = acc_df['type'].replace({0: 'train', 1: 'valid'}).values
#
#     g_res = sns.lineplot(data=acc_df, x="step", y="val", hue="type")
#     g_res.set(xscale='log')
#     plt.title("Modular Multiplication (training on 50% of data)")
#     plt.ylim(-0.025, 1.025)
#     # plt.show()
#     plt.savefig(f"results/acc_{sns_plot}.png")
#     plt.close()
#
#     trn_np = np.zeros((len(train_loss), 3))
#     trn_np[:, 0], trn_np[:, 1] = steps, np.array(train_loss)
#     trn_np[:, 2] = 0
#     trn_np = np.vstack((trn_np[0:-2, :], trn_np[1:-1, :], trn_np[2:, :]))
#     trn_np[:, 0] = steps_np
#
#     val_np = np.zeros((len(val_acc), 3))
#     val_np[:, 0], val_np[:, 1] = steps, np.array(val_loss)
#     val_np[:, 2] = 1
#     val_np = np.vstack((val_np[0:-2, :], val_np[1:-1, :], val_np[2:, :]))
#     val_np[:, 0] = steps_np
#
#     lss_np = np.vstack((trn_np, val_np[2:]))
#
#     lss_df = pd.DataFrame(lss_np, columns=['step', 'val', 'type'])
#     lss_df = lss_df.astype({'step': int, 'val': float, 'type': int})
#     lss_df['type'] = lss_df['type'].replace({0: 'train', 1: 'valid'}).values
#
#     g_res = sns.lineplot(data=lss_df, x="step", y="val", hue='type')
#     g_res.set(xscale='log')
#     plt.title("Modular Multiplication (training on 50% of data)")
#     # plt.ylim(0.0, 1.0)
#     # plt.show()
#     plt.savefig(f"results/loss_{sns_plot}.png")
#     plt.close()
#
#     pass

def get_plot_infix(args):
    # plot model architecture infix
    ff = datetime.now().strftime("%f")
    plot_infix = f"l{args.num_layers}_h{args.num_heads}_e{args.embedding}_{ff}"
    return plot_infix

def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokens for <op> and <=>. It's not clear why <=> is needed at all since it
    # has no effect on the output, but we'll leave it in to best follow the
    # paper.
    eq_token = args.p
    op_token = args.p + 1

    # "We trained a standard decoder-only transformer (Vaswani et al., 2017)
    # with causal attention masking, and calculated loss and accuracy only on
    # the answer part of the equation. For all experiments we used a
    # transformer with 2 layers, width 128, and 4 attention heads"
    model = Decoder(
        dim=args.embedding,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_tokens=args.p + 2,
        seq_len=5
    ).to(device)
    nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(model)
    print(f'Total number of parameters: {nparams}')

    # plot model architecture infix
    plot_infix = get_plot_infix(args)

    data = multiplication_mod_p_data(args.p, eq_token, op_token)

    train_idx, valid_idx = torch.randperm(data.shape[1]).split(data.shape[1] // 2)
    train_data, valid_data = data[:, train_idx], data[:, valid_idx]

    # For most experiments we used AdamW optimizer with learning rate 10−3,
    # weight decay 1, β1 = 0.9, β2 = 0.98
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    #  linear learning rate warmup over the first 10 updates
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )

    steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)

    its, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    grads = None
    i = 0

    # For logging network weights.
    net_its, nets = [], []
    # In your training loop:
    norms_history = {
        # 'attention': [],
        # 'mlp': [],
        # 'total': [],
        'epoch': []
    }
    acc_dict = {
        'train': [],
        'valid': [],
        'epoch': []
    }
    loss_dict = {
        'train': [],
        'valid': [],
        'epoch': []
    }
    all_history = {}

    for e in tqdm(range(int(args.budget) // steps_per_epoch)):

        # randomly shuffle train data
        train_data = train_data[:, torch.randperm(train_data.shape[1])]

        for data, is_train in [(train_data, True), (valid_data, False)]:

            model.train(is_train)
            total_loss = 0
            total_acc = 0

            # torch.split faster than dataloader with tensor
            dl = torch.split(data, args.batch_size, dim=1)
            for input in dl:
                input = input.to(device)

                with torch.set_grad_enabled(is_train):
                    logits = model(input[:-1])
                    # calculate loss only on the answer part of the equation (last element
                    loss = F.cross_entropy(logits[-1], input[-1])
                    total_loss += loss.item() * input.shape[-1]

                if is_train:
                    model.zero_grad()
                    loss.backward()

                    #######

                    trigger = i < 500 if args.two_stage else False

                    if args.filter == "none":
                        pass
                    elif args.filter == "ma":
                        grads = gradfilter_ma(model, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
                    elif args.filter == "ema":
                        grads = gradfilter_ema(model, grads=grads, alpha=args.alpha, lamb=args.lamb)
                    else:
                        raise ValueError(f"Invalid gradient filter type `{args.filter}`")

                    #######

                    optimizer.step()
                    scheduler.step()
                    i += 1

                acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
                total_acc += acc.item() * input.shape[-1]

            if is_train:
                acc_dict['train'].append(total_acc / train_data.shape[-1])
                loss_dict['train'].append(total_loss / train_data.shape[-1])
                train_acc.append(total_acc / train_data.shape[-1])
                train_loss.append(total_loss / train_data.shape[-1])
                its.append(i)

                if e % 2 == 0:
                    with torch.no_grad():  # important for efficiency
                        norms = get_parameter_norms(model)
                        for key in norms.keys():
                            if key not in norms_history.keys():
                                norms_history[key] = []
                            norms_history[key].append(norms[key])
                        norms_history = add_key(norms_history, 'epoch')
                        norms_history['epoch'].append(i)
                        all_history["norms"] = norms_history.copy()

                        # detailed_norms = get_detailed_norms(model)
                        # if 'epoch' not in detailed_norms.keys():
                        #     detailed_norms['epoch'] = []
                        # detailed_norms["epoch"].append(i)
                        # all_history["detailed_norms"] = detailed_norms.copy()
                        # matrix_norms = get_matrix_norms(model)

                        # det_norms = get_detailed_norms(model)
            else:
                acc_dict['valid'].append(total_acc / valid_data.shape[-1])
                loss_dict['valid'].append(total_loss / valid_data.shape[-1])
                val_acc.append(total_acc / valid_data.shape[-1])
                val_loss.append(total_loss / valid_data.shape[-1])
                acc_dict = add_key(acc_dict, 'epoch')
                acc_dict['epoch'].append(i)
                loss_dict = add_key(loss_dict, 'epoch')
                loss_dict['epoch'].append(i)
            all_history["accuracy"] = acc_dict.copy()
            all_history["loss"] = loss_dict.copy()

        if args.save_weights:
            do_save = e <= 500 or (e > 500 and (e + 1) % 100 == 0) or e == int(args.budget) // steps_per_epoch - 1
        else:
            do_save = (e + 1) % 100 == 0
        if do_save:
            plot_dicts(dres=all_history, plot_infix=plot_infix)

            results = {
                'its': its,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
            }

        if do_save:
            if args.save_weights:
                net_its.append(e)
                nets.append(copy.deepcopy(model.state_dict()))
                results['net_its'] = net_its
                results['net'] = nets

            torch.save(results, f"results/res_{args.label}.pt")
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    # architecture parameters
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)

    # run params
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=3e5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optimizer", default="Adam")

    # Grokfast
    parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir"], default="none")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)

    # Ablation studies
    parser.add_argument("--two_stage", action='store_true')
    parser.add_argument("--save_weights", action='store_true')
    args = parser.parse_args()

    filter_str = ('_' if args.label != '' else '') + args.filter
    window_size_str = f'_w{args.window_size}'
    alpha_str = f'_a{args.alpha:.3f}'.replace('.', '')
    lamb_str = f'_l{int(args.lamb)}'

    if args.filter == 'none':
        filter_suffix = ''
    elif args.filter == 'ma':
        filter_suffix = window_size_str + lamb_str
    elif args.filter == 'ema':
        filter_suffix = alpha_str + lamb_str
    else:
        raise ValueError(f"Unrecognized filter type {args.filter}")

    optim_suffix = ''
    if args.weight_decay != 0:
        optim_suffix = optim_suffix + f'_wd{args.weight_decay:.1e}'.replace('.', '')
    if args.lr != 1e-3:
        optim_suffix = optim_suffix + f'_lrx{int(args.lr / 1e-3)}'

    args.label = args.label + filter_str + filter_suffix + optim_suffix
    print(f'Experiment results saved under name: {args.label}')

    main(args)
