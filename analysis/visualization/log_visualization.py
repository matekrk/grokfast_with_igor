# Example Usage of the Visualization System

import os
from dataclasses import fields
from pathlib import Path
import numpy as np
import torch
from analysis.models.analysis_transformer import Decoder
from analysis.core.logger import DataLogger
from analysis.utils.utils import find_best_matching_file, find_matching_files, read_json
from analysis.visualization.base_visualizer import BaseVisualizer
from analysis.visualization.visualization_manager import VisualizationManager
from analysis.visualization.concrete_visualizers import (
    TimeSeriesVisualizer,
    HeatmapVisualizer,
    NetworkGraphVisualizer,
    CircuitAnalysisVisualizer,
    PhaseTransitionVisualizer,
    MLPSparsityVisualizer
)


def read_logger(file):
    if file is None or file == '':
        return DataLogger()
    else:
        logger = read_json(file)
        return logger


def main(experiment_name, epoch):

    base_dir = Path.home() / "results"
    # Load or create sample logger data
    if experiment_name is None or experiment_name == "":
        logger_data = get_sample_logger()
    else:
        # fixme we need logger data, not the logger as such # todo this would probably be the logger.logs?
        log_dir = base_dir / f"{experiment_name}/stats/logs/"
        log_pattern = f"logs_epoch_*.json"
        get_file = find_best_matching_file(log_dir, log_pattern, criteria="largest_number")
        log_pattern = "logs_epoch_*.json"
        get_file_list_best_best = find_matching_files(log_dir, log_pattern, criteria="largest_number", return_sorted=False)
        get_file_list_sorted = find_matching_files(log_dir, log_pattern, criteria="largest_number", return_sorted=True)

        logger_data = read_logger(get_file_list_best_best.absolute().as_posix())

    save_dir = base_dir / f"{experiment_name}/visualization_outputs"
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create visualization manager and register visualizers
    manager = VisualizationManager(save_dir=save_dir)
    manager.register_visualizer(TimeSeriesVisualizer())
    manager.register_visualizer(HeatmapVisualizer())
    manager.register_visualizer(NetworkGraphVisualizer())
    manager.register_visualizer(CircuitAnalysisVisualizer())
    manager.register_visualizer(PhaseTransitionVisualizer())
    manager.register_visualizer(MLPSparsityVisualizer())


    manager.set_data(logger_data["data"])  # todo logger_data.data?

    fields = manager.get_available_fields()
    print(f"Availabale fields are\n{fields}")

    results = manager.create_example_visualizations(
        logger_data=logger_data["data"],
        # output_dir="example_visualizations",
        fields_to_visualize=[
            "spectral_properties",
            "weight_space_tracking",
            "training",
            "evaluation",
            "circuit_tracking",
            # "weight_velocities",            # warning saved at different moments, but why??
            "mlp_sparcity",
            "mlp_selectivity",
            # "attn_mlp_interaction",         # warning only last moment; save history?
            "grokking_phases",
            "phase_transitions",
            # "attention_entropy",            # warning only last value
            "circuit_analysis",
            "weight_space_jumps",
            # "jump_analysis",                # warning only single values (several)
        ]
    )

    print(f"Created {len(results)} visualizations")
    for field, path in results.items():
        print(f"  - {field}: {path}")

    # Save results
    manager.save_visualization_results(results, "example_visualizations/results.json")

    print(f"All visualizations have been saved to {manager.save_dir.absolute()}")


def create_example_visualizations(vis_manager, output_dir):
    """Create a set of example visualizations."""
    visualizations_to_create = [
        # Performance metrics (time series)
        {
            'field': 'evaluation.accuracy',
            'vis_type': 'time_series',
            'config': {
                'title': 'Evaluation Accuracy Over Training',
                'xlabel': 'Epoch',
                'ylabel': 'Accuracy',
                'style': {'color': 'blue', 'marker': 'o', 'linestyle': '-'}
            },
            'filename': 'eval_accuracy.png'
        },
        {
            'field': 'training.loss',
            'vis_type': 'time_series',
            'config': {
                'title': 'Training Loss',
                'xlabel': 'Epoch',
                'ylabel': 'Loss',
                'style': {'color': 'red', 'marker': '.', 'linestyle': '-'}
            },
            'filename': 'train_loss.png'
        },

        # Combined training metrics
        {
            'field': 'training',
            'vis_type': 'time_series',
            'config': {
                'title': 'Training Metrics',
                'xlabel': 'Epoch',
                'ylabel': 'Value',
                'style': {
                    'accuracy': {'color': 'green', 'label': 'Accuracy'},
                    'loss': {'color': 'red', 'label': 'Loss'}
                }
            },
            'filename': 'training_metrics.png'
        },

        # MLP Sparsity visualization
        {
            'field': 'mlp_sparsity_tracker.sparsity_history',
            'vis_type': 'mlp_sparsity',
            'config': {
                'title': 'MLP Neuron Sparsity Analysis',
                'figsize': (14, 10)
            },
            'filename': 'mlp_sparsity.png'
        },

        # Circuit Analysis
        {
            'field': 'circuit_tracker.circuit_history',
            'vis_type': 'circuit_analysis',
            'config': {
                'title': 'Circuit Analysis Results',
                'figsize': (15, 12)
            },
            'filename': 'circuit_analysis.png'
        },

        # Phase transitions
        {
            'field': 'phase_analyzer.phase_structure',
            'vis_type': 'phase_transitions',
            'config': {
                'title': 'Learning Phase Analysis',
                'figsize': (15, 10),
                'performance_data': {
                    'training': {
                        'epoch': vis_manager.get_field_data('training.epoch'),
                        'accuracy': vis_manager.get_field_data('training.accuracy')
                    },
                    'evaluation': {
                        'epoch': vis_manager.get_field_data('evaluation.epoch'),
                        'accuracy': vis_manager.get_field_data('evaluation.accuracy')
                    }
                },
                'grokking_points': vis_manager.get_field_data('grokking_phases.grokking_step')
            },
            'filename': 'phase_transitions.png'
        },

        # Attention patterns
        {
            'field': 'attention_patterns',
            'vis_type': 'heatmap',
            'config': {
                'title': 'Attention Patterns',
                'figsize': (16, 10)
            },
            'filename': 'attention_patterns.png'
        },

        # Interaction network
        {
            'field': 'interaction_analyzer.interaction_history.1000.interaction_graph',
            'vis_type': 'network_graph',
            'config': {
                'title': 'Attention-MLP Interaction Network (Epoch 1000)',
                'figsize': (12, 10),
                'layout': 'spring'
            },
            'filename': 'interaction_network.png'
        }
    ]

    # Create each visualization if possible
    for viz_config in visualizations_to_create:
        field = viz_config['field']
        vis_type = viz_config['vis_type']
        config = viz_config['config']
        filename = viz_config['filename']

        # Check if the field exists and can be visualized
        if vis_manager.can_visualize_field(field):
            try:
                print(f"Creating visualization for {field} as {vis_type}...")
                visualization = vis_manager.visualize_field(field, vis_type, config)
                save_path = output_dir / filename
                vis_manager.save_visualization(visualization, save_path, vis_type)
                print(f"✓ Saved to {save_path}")
            except Exception as e:
                print(f"✗ Failed to create visualization for {field}: {e}")
        else:
            print(f"✗ Cannot visualize field: {field}")


def get_sample_logger():
    """
    Create a sample logger with data for testing visualizations.
    In a real application, this would be loaded from saved experiment data.
    """
    logger = DataLogger()

    # Add training and evaluation metrics
    epochs = list(range(0, 2000, 20))

    # Simulated training metrics
    train_acc = [min(0.99, 0.5 + 0.5 * (1 - np.exp(-epoch / 200))) for epoch in epochs]
    train_loss = [max(0.01, 0.5 * np.exp(-epoch / 300)) for epoch in epochs]

    # Simulated evaluation metrics with grokking pattern
    eval_acc = []
    for epoch in epochs:
        if epoch < 500:
            # Random fluctuation at start
            acc = 0.05 + 0.05 * np.random.random()
        elif 500 <= epoch < 1000:
            # Slow improvement
            acc = 0.1 + 0.1 * (epoch - 500) / 500
        elif 1000 <= epoch < 1200:
            # Grokking jump
            progress = (epoch - 1000) / 200
            acc = 0.2 + 0.7 * progress
        else:
            # Final plateau with slight improvement
            acc = 0.9 + 0.09 * (epoch - 1200) / 800
        eval_acc.append(acc)

    eval_loss = [max(0.005, 1.0 * np.exp(-epoch / 500)) for epoch in epochs]

    # Log training metrics
    for i, epoch in enumerate(epochs):
        logger.log_data('training', 'epoch', epoch)
        logger.log_data('training', 'accuracy', train_acc[i])
        logger.log_data('training', 'loss', train_loss[i])

        logger.log_data('evaluation', 'epoch', epoch)
        logger.log_data('evaluation', 'accuracy', eval_acc[i])
        logger.log_data('evaluation', 'loss', eval_loss[i])

    # Add grokking detection
    logger.log_data('grokking_phases', 'grokking_step', 1100)

    # Simulate phase transitions
    phase_structure = {
        'phases': [
            {
                'phase_id': 1,
                'start_epoch': 0,
                'end_epoch': 500,
                'classification': 'exploration'
            },
            {
                'phase_id': 2,
                'start_epoch': 500,
                'end_epoch': 1000,
                'classification': 'transition'
            },
            {
                'phase_id': 3,
                'start_epoch': 1000,
                'end_epoch': 1200,
                'classification': 'consolidation'
            },
            {
                'phase_id': 4,
                'start_epoch': 1200,
                'end_epoch': 2000,
                'classification': 'stability'
            }
        ],
        'transitions': [
            {
                'epoch': 500,
                'transition_types': ['exploration_to_transition']
            },
            {
                'epoch': 1000,
                'transition_types': ['transition_to_consolidation']
            },
            {
                'epoch': 1200,
                'transition_types': ['consolidation_to_stability']
            }
        ]
    }

    logger.logs['phase_analyzer'] = {'phase_structure': phase_structure}

    # Simulate MLP sparsity data
    sparsity_history = {}
    for epoch in [0, 500, 1000, 1200, 1500, 1800]:
        # More sparsity as training progresses
        base_sparsity = 0.2 + 0.6 * epoch / 2000

        sparsity_history[epoch] = {
            'avg_sparsity': {
                'layer_0_mlp_expanded': base_sparsity - 0.1,
                'layer_1_mlp_expanded': base_sparsity,
                'layer_2_mlp_expanded': base_sparsity + 0.1,
                'layer_3_mlp_expanded': base_sparsity + 0.15
            },
            'selectivity_summary': {
                'layer_0_mlp_expanded': {
                    'total_neurons': 100,
                    'selective_neurons': int(20 + 60 * epoch / 2000),
                    'class_distribution': {
                        '0': 5,
                        '1': 8,
                        '2': 15,
                        '3': 7
                    }
                },
                'layer_1_mlp_expanded': {
                    'total_neurons': 100,
                    'selective_neurons': int(10 + 70 * epoch / 2000)
                },
                'layer_2_mlp_expanded': {
                    'total_neurons': 100,
                    'selective_neurons': int(5 + 75 * epoch / 2000)
                },
                'layer_3_mlp_expanded': {
                    'total_neurons': 100,
                    'selective_neurons': int(2 + 80 * epoch / 2000)
                }
            }
        }

    logger.logs['mlp_sparsity_tracker'] = {'sparsity_history': sparsity_history}

    # Simulate circuit analysis data
    circuit_history = {
        'epochs': [0, 500, 1000, 1500, 1800],
        'active_heads': [
            # Epoch 0
            ['layer_0_head_0', 'layer_1_head_1'],
            # Epoch 500
            ['layer_0_head_0', 'layer_0_head_1', 'layer_1_head_1'],
            # Epoch 1000
            ['layer_0_head_0', 'layer_0_head_1', 'layer_1_head_0', 'layer_1_head_1', 'layer_2_head_0'],
            # Epoch 1500
            ['layer_0_head_0', 'layer_0_head_1', 'layer_1_head_0', 'layer_1_head_1', 'layer_2_head_0',
             'layer_2_head_1'],
            # Epoch 1800
            ['layer_0_head_0', 'layer_0_head_1', 'layer_1_head_0', 'layer_1_head_1', 'layer_2_head_0', 'layer_2_head_1',
             'layer_3_head_0']
        ],
        'head_attributions': [
            # Epoch 0
            {'layer_0_head_0': 0.1, 'layer_1_head_1': 0.15},
            # Epoch 500
            {'layer_0_head_0': 0.2, 'layer_0_head_1': 0.15, 'layer_1_head_1': 0.25},
            # Epoch 1000
            {'layer_0_head_0': 0.3, 'layer_0_head_1': 0.25, 'layer_1_head_0': 0.2, 'layer_1_head_1': 0.35,
             'layer_2_head_0': 0.15},
            # Epoch 1500
            {'layer_0_head_0': 0.4, 'layer_0_head_1': 0.35, 'layer_1_head_0': 0.3, 'layer_1_head_1': 0.45,
             'layer_2_head_0': 0.25, 'layer_2_head_1': 0.2},
            # Epoch 1800
            {'layer_0_head_0': 0.5, 'layer_0_head_1': 0.45, 'layer_1_head_0': 0.4, 'layer_1_head_1': 0.55,
             'layer_2_head_0': 0.35, 'layer_2_head_1': 0.3, 'layer_3_head_0': 0.25}
        ],
        'active_circuits': [
            # Epoch 0
            [],
            # Epoch 500
            ['layer_0_head_0+layer_1_head_1'],
            # Epoch 1000
            ['layer_0_head_0+layer_1_head_1', 'layer_0_head_1+layer_1_head_0'],
            # Epoch 1500
            ['layer_0_head_0+layer_1_head_1', 'layer_0_head_1+layer_1_head_0', 'layer_1_head_1+layer_2_head_0'],
            # Epoch 1800
            ['layer_0_head_0+layer_1_head_1', 'layer_0_head_1+layer_1_head_0', 'layer_1_head_1+layer_2_head_0',
             'layer_2_head_0+layer_3_head_0']
        ],
        'circuit_strengths': [
            # Epoch 0
            {},
            # Epoch 500
            {'layer_0_head_0+layer_1_head_1': 0.05},
            # Epoch 1000
            {'layer_0_head_0+layer_1_head_1': 0.15, 'layer_0_head_1+layer_1_head_0': 0.1},
            # Epoch 1500
            {'layer_0_head_0+layer_1_head_1': 0.25, 'layer_0_head_1+layer_1_head_0': 0.2,
             'layer_1_head_1+layer_2_head_0': 0.15},
            # Epoch 1800
            {'layer_0_head_0+layer_1_head_1': 0.35, 'layer_0_head_1+layer_1_head_0': 0.3,
             'layer_1_head_1+layer_2_head_0': 0.25, 'layer_2_head_0+layer_3_head_0': 0.2}
        ],
        'emerging_circuits': [
            # Epoch 0
            [],
            # Epoch 500
            ['layer_0_head_0+layer_1_head_1'],
            # Epoch 1000
            ['layer_0_head_1+layer_1_head_0'],
            # Epoch 1500
            ['layer_1_head_1+layer_2_head_0'],
            # Epoch 1800
            ['layer_2_head_0+layer_3_head_0']
        ],
        'declining_circuits': [
            # Epoch 0
            [],
            # Epoch 500
            [],
            # Epoch 1000
            [],
            # Epoch 1500
            [],
            # Epoch 1800
            []
        ]
    }

    logger.logs['circuit_tracker'] = {'circuit_history': circuit_history}

    # Simulated attention patterns
    attention_patterns = {
        'layer_0_head_0': np.zeros((5, 5)),
        'layer_0_head_1': np.zeros((5, 5)),
        'layer_1_head_0': np.zeros((5, 5)),
        'layer_1_head_1': np.zeros((5, 5))
    }

    # Fill with some sample patterns
    for i in range(5):
        for j in range(i + 1):  # Lower triangular - causal attention
            attention_patterns['layer_0_head_0'][i, j] = 1.0 if i == j else 0.1
            attention_patterns['layer_0_head_1'][i, j] = 1.0 if j == 0 else 0.1
            attention_patterns['layer_1_head_0'][i, j] = 1.0 / (abs(i - j) + 1)
            attention_patterns['layer_1_head_1'][i, j] = 1.0 if i == j else 0.0

    # Normalize
    for key in attention_patterns:
        attention_patterns[key] = attention_patterns[key] / attention_patterns[key].sum(axis=1, keepdims=True)

    logger.logs['attention_patterns'] = attention_patterns

    # Simulated interaction analysis
    interaction_graph = {
        'nodes': [
            {'id': 'layer_0_head_0', 'type': 'attention_head', 'correlation': 0.8},
            {'id': 'layer_0_head_1', 'type': 'attention_head', 'correlation': 0.7},
            {'id': 'layer_1_head_0', 'type': 'attention_head', 'correlation': 0.6},
            {'id': 'layer_1_head_1', 'type': 'attention_head', 'correlation': 0.9},
            {'id': 'layer_0_mlp', 'type': 'mlp'},
            {'id': 'layer_1_mlp', 'type': 'mlp'}
        ],
        'edges': [
            {'source': 'layer_0_head_0', 'target': 'layer_0_mlp', 'weight': 0.7, 'type': 'attn_to_mlp'},
            {'source': 'layer_0_head_1', 'target': 'layer_0_mlp', 'weight': 0.5, 'type': 'attn_to_mlp'},
            {'source': 'layer_1_head_0', 'target': 'layer_1_mlp', 'weight': 0.6, 'type': 'attn_to_mlp'},
            {'source': 'layer_1_head_1', 'target': 'layer_1_mlp', 'weight': 0.8, 'type': 'attn_to_mlp'},
            {'source': 'layer_0_head_0', 'target': 'layer_1_head_1', 'weight': 0.4, 'type': 'head_to_head'},
            {'source': 'layer_0_head_1', 'target': 'layer_1_head_0', 'weight': 0.3, 'type': 'head_to_head'}
        ]
    }

    logger.logs['interaction_analyzer'] = {
        'interaction_history': {
            1000: {'interaction_graph': interaction_graph}
        }
    }

    return logger


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="l2_h4_e128_b256_mult-97_AdamW_lr0.001_wd0.01_r0.5_750401")  # fixme ?
    parser.add_argument("--epoch", default="400")
    args = parser.parse_args()
    main(experiment_name=args.experiment, epoch=args.epoch)