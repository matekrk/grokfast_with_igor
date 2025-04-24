import math
import sys
from argparse import ArgumentParser
from datetime import datetime

from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.functional.classification import multiclass_accuracy
# from torcheval.metrics import BinaryAccuracy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from analysis.core.logger import DataLogger
from analyze import get_parameter_norms, get_detailed_joined_attn_norms
from plot import plot_dicts


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
        positions = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits[-1]

def get_plot_infix(args):
    # plot model architecture infix
    ff = datetime.now().strftime("%f")
    plot_infix = f"l{args.num_layers}_h{args.num_heads}_e{args.embedding}_{ff}"
    return plot_infix


# replace with read_args(sys.argv[1:]) in python
def read_args(args):
    parser = ArgumentParser(description="Grokfast")

    print(f"provided args: {args}")

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

    args = parser.parse_args(args=args)

    args.plot_infix = get_plot_infix(args=args)

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

    return args

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

def print_model(model):
    nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(model)
    print(f'Total number of parameters: {nparams}')
    return

def build_model(args):
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
    print_model(model)
    return model

def set_optimizer(model, args):
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
    return optimizer, scheduler

class TransformerDataset(Dataset):
    def __init__(self, token_array, labels=None):
        self.data = torch.tensor(token_array.clone().detach().transpose(0, 1))  # Transpose to make each row an example
        self.labels = None if labels is None else torch.tensor(labels)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]


def create_train_test_dataloaders(token_array, labels=None, train_ratio=0.8, batch_size=32):
    dataset = TransformerDataset(token_array, labels)

    # Calculate lengths for split
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Split dataset
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader



def build_dataloader(token_array, labels=None):
    # Create dataset and dataloader
    dataset = TransformerDataset(token_array, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    return dataloader


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    # accuracy = BinaryAccuracy()
    model.train()
    metric = MulticlassAccuracy(num_classes=args.p+1)
    loss_sum = torch.zeros(1, device=device)
    all_inputs = torch.zeros(1, device=device)
    total_acc = torch.zeros(1, device=device)
    with torch.set_grad_enabled(True):
        for k, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs.transpose(0, 1))
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * inputs.size(0)
            all_inputs += inputs.size(0)
            acc_metr_fn_micro = multiclass_accuracy(logits.argmax(-1), targets, args.p+1, average='micro')
            total_acc += acc_metr_fn_micro.item() * inputs.size(0)
    return total_acc / all_inputs, loss_sum / all_inputs

def validate_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = torch.zeros(1, device=device)
    all_inputs = torch.zeros(1, device=device)
    total_acc = torch.zeros(1, device=device)
    with torch.no_grad():
        for k, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs.transpose(0, 1))
            loss = criterion(logits[-1], targets)
            loss_sum += loss.item() * inputs.size(0)
            all_inputs += inputs.size(0)
            acc = (logits.argmax(-1) == targets).float().mean()
            acc_metr_fn_micro = multiclass_accuracy(logits.argmax(-1), targets, args.p+1, average='micro')
            total_acc += acc_metr_fn_micro.item() * inputs.size(0)
    return total_acc / all_inputs, loss_sum / all_inputs


def main(args):
    # info data logging
    # Example usage
    logger = DataLogger()

    # # Retrieve and print logs
    # temperature_logs = logger.get_logs('temperature')
    # humidity_logs = logger.get_logs('humidity')
    #
    # print("Temperature Logs:", temperature_logs)
    # print("Humidity Logs:", humidity_logs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eq_token = args.p
    op_token = args.p + 1
    data = multiplication_mod_p_data(p=args.p, eq_token=eq_token, op_token=op_token)
    train_loader, test_loader = create_train_test_dataloaders(token_array=data[:-1, :],
                                                              labels=data[-1, :],
                                                              train_ratio=0.5, batch_size=args.batch_size)
    # plot_infix = get_plot_infix(args=args)
    model = build_model(args=args)
    print(f"plot index is {args.plot_infix}")
    optimizer, scheduler = set_optimizer(model=model, args=args)
    steps_per_epoch = math.ceil(len(train_loader) / args.batch_size)
    plot_interval = 100

    # for epoch in trange(int(args.budget // steps_per_epoch)):
    for epoch in tqdm(range(int(args.budget // steps_per_epoch))):
        trn_acc, trn_loss = train_one_epoch(model=model, loader=train_loader,
                                            criterion=torch.nn.CrossEntropyLoss(),
                                            optimizer=optimizer, scheduler=scheduler, device=device)
        if epoch % 2 == 0:
            vld_acc, vld_loss = validate_epoch(model=model, loader=test_loader,
                                               criterion=torch.nn.CrossEntropyLoss(), device=device)
            # info log accuracy and loss for both train and validate
            logger.log_data('accuracy', 'train', trn_acc.item())
            logger.log_data('accuracy', 'valid', vld_acc.item())
            logger.log_data('accuracy', 'epoch', epoch)
            logger.log_data('loss', 'train', trn_loss.item())
            logger.log_data('loss', 'valid', vld_loss.item())
            logger.log_data('loss', 'epoch', epoch)
            norms = get_parameter_norms(model)
            logger.update_category_means('norms', norms)
            logger.log_data('norms', 'epoch', epoch)
            qkv_norms = get_detailed_joined_attn_norms(model=model)
            for k, v in qkv_norms.items():
                logger.log_data('qkv_norms', k, v)
            logger.log_data('qkv_norms', 'epoch', epoch)
        if epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch} - Train Acc: {trn_acc:.3f}, Val Acc: {vld_acc:.3f}")
        if epoch > 0 and ((epoch * steps_per_epoch))% plot_interval == 0:
            plot_dicts(logger.get_all_logs(), plot_infix=args.plot_infix)

    pass

if __name__ == '__main__':
    args = read_args(sys.argv[1:])
    main(args)