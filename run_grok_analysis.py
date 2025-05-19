# File: run_grok_analysis.py
# Place this file at the root level (same level as the analysis folder)

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the main function from the module
from analysis.experiments.grok_transformer_analysis import main

if __name__ == "__main__":
    import argparse

    # Create argument parser - copy the same parser from grok_transformer_analysis.py
    parser = argparse.ArgumentParser()
    # architecture parameters
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)

    # run params
    parser.add_argument("--label", default="")  # fixme ?
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p", type=int, default=97)  # fixme ?
    parser.add_argument("--budget", type=int, default=3e5)  # fixme ?
    parser.add_argument("--batch_size", type=int, default=256)  # 512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)  # fixme ?
    parser.add_argument("--beta2", type=float, default=0.98)  # fixme ?
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.5)

    # parser.add_argument("--enhanced", action="store_true", help="Run enhanced analysis")
    # parser.add_argument("--phase", action="store_true", help="Run phase transition analysis")
    parser.add_argument('--mode',
                        choices=['enhanced', 'default'],  # The three possible values
                        default='enhanced',  # Default value
                        help='Set the analysis type: enhanced [default] or standard')
    parser.add_argument('--type',
                        choices=['phase', 'weight'],  # The three possible values
                        default='phase',  # Default value
                        help='Set the analysis mode: phase [default] or weight')

    # analysis intervals
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--analyze_interval", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=4)
    parser.add_argument("--checkpoint_interval", type=int, default=200)

    parser.add_argument("--operation", type=str, default='multiply')

    # Ablation studies
    parser.add_argument("--two_stage", action='store_true')  # fixme ?
    parser.add_argument("--save_weights", action='store_true')  # fixme ?
    args = parser.parse_args()

    # Run main function with parsed arguments
    main(args)