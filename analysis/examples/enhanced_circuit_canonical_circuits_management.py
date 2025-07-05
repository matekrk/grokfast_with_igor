# examples/enhanced_circuit_training_example.py
"""
Complete Example: Enhanced Circuit Management in Training

Shows how to migrate from the old system to the new unified approach.
"""

import torch
import torch.nn as nn
from pathlib import Path

# NEW: Import the enhanced system
from analysis.trainers.train_with_enhanced_canonical_circuits import (
    train_with_enhanced_circuit_management
)
from analysis.trainers.utils import get_default_circuit_config, get_aggressive_circuit_config, \
    get_conservative_circuit_config

'''
def main_training_example():
    """
    Complete example showing how to use the enhanced circuit management system
    """

    # Setup model and data (your existing setup)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize your model, data loaders, etc.
    model = create_your_model()  # Your transformer model
    train_loader = create_train_loader()  # Your training data
    eval_loader = create_eval_loader()  # Your evaluation data

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    # Configure circuit management
    # Choose based on your experimental needs:

    # For exploratory experiments (faster iteration)
    circuit_config = get_aggressive_circuit_config()

    # For important experiments (preserve more circuits)
    # circuit_config = get_conservative_circuit_config()

    # For balanced approach
    # circuit_config = get_default_circuit_config()

    # Customize configuration if needed
    circuit_config.update({
        'max_circuits': 120,  # Adjust based on your needs
        'min_quality_threshold': 0.3,  # Adjust quality bar
        'probation_epochs': 75,  # Adjust protection period
    })

    print("🚀 Starting enhanced circuit management training...")
    print(f"Configuration: {circuit_config}")

    # NEW: Use the enhanced training function
    results = train_with_enhanced_circuit_management(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=2000,
        log_interval=10,
        circuit_assessment_interval=25,  # Assess circuits every 25 epochs
        circuit_management_config=circuit_config,
        enable_circuit_removal=True,  # Actually remove low-quality circuits
        enable_diversity_protection=True,  # Protect for diversity
        enable_real_testing=True,  # 🧪 Use real functional testing (activation patching)
        enable_wandb_logging=False,  # Set to True if you use wandb
        enable_file_logging=True,
        log_level="INFO"
    )

    # Analyze results
    analyze_training_results(results)

    return results
'''

def analyze_training_results(results):
    """
    Analyze the results of enhanced circuit training
    """
    model = results['model']
    canonical_registry = results['canonical_registry']
    quality_system = results['quality_system']
    circuit_history = results['circuit_management_history']
    df_circuit = results['circuit_dataframe']  # 📊 NEW: Access to comprehensive DataFrame
    circuit_snapshots = results['circuit_snapshots']  # 📊 NEW: Detailed snapshots

    print("\n🔍 Training Results Analysis:")
    print("=" * 50)

    # 1. Final circuit population
    final_stats = results['final_stats']
    print(f"📊 Circuit Population:")
    print(f"   Total circuits discovered: {final_stats['total_canonical_circuits']}")
    print(f"   Aggregation rate: {final_stats['aggregation_rate']:.2%}")
    print(f"   Stable circuits: {final_stats['stable_circuits_count']}")

    # 📊 2. ENHANCED DATAFRAME ANALYSIS
    if not df_circuit.empty:
        print(f"\n📈 DataFrame Analysis ({len(df_circuit)} records):")
        print(f"   Epochs covered: {df_circuit['epoch'].min()}-{df_circuit['epoch'].max()}")
        print(f"   Unique circuits tracked: {df_circuit['circuit_id'].nunique()}")

        # Quality evolution analysis
        quality_evolution = df_circuit.groupby('epoch')['overall_quality'].agg(['mean', 'std', 'count'])
        print(
            f"   Quality evolution: {quality_evolution.iloc[-1]['mean']:.3f} ± {quality_evolution.iloc[-1]['std']:.3f}")

        # Lifecycle distribution
        if 'lifecycle_state' in df_circuit.columns:
            lifecycle_dist = df_circuit['lifecycle_state'].value_counts()
            print(f"   Final lifecycle distribution: {dict(lifecycle_dist)}")

        # Operation type diversity
        op_type_dist = df_circuit['operation_type'].value_counts()
        print(f"   Operation types: {dict(op_type_dist)}")

        # Circuit longevity analysis
        circuit_lifespans = df_circuit.groupby('circuit_id')['circuit_age'].max()
        print(f"   Average circuit lifespan: {circuit_lifespans.mean():.1f} epochs")
        print(f"   Longest surviving circuit: {circuit_lifespans.max()} epochs")

        # Quality trend analysis
        if 'quality_trend' in df_circuit.columns:
            improving_circuits = (df_circuit['quality_trend'] > 0.1).sum()
            declining_circuits = (df_circuit['quality_trend'] < -0.1).sum()
            print(f"   Circuits improving: {improving_circuits}, declining: {declining_circuits}")

    # 3. Circuit lifecycle analysis
    if circuit_history:
        removal_count = sum(len(h['removed_circuits']) for h in circuit_history)
        protection_count = sum(len(h['protected_circuits']) for h in circuit_history)

        print(f"\n🔄 Circuit Lifecycle:")
        print(f"   Total removals: {removal_count}")
        print(f"   Total protections: {protection_count}")

        # Analyze removal reasons
        removal_reasons = {}
        for history in circuit_history:
            for removal in history['removed_circuits']:
                reason = removal['reason']
                removal_reasons[reason] = removal_reasons.get(reason, 0) + 1

        print(f"   Removal reasons: {dict(removal_reasons)}")

    # 4. Quality distribution analysis
    quality_assessment = quality_system.quality_analyzer.assess_circuit_quality(
        eval_loader=None,  # You'd pass your eval_loader here
        max_circuits=1000
    )

    quality_stats = quality_assessment['quality_report']['quality_statistics']
    print(f"\n📈 Quality Distribution:")
    print(f"   Mean quality: {quality_stats['mean_quality']:.3f}")
    print(f"   Quality std: {quality_stats['std_quality']:.3f}")

    # 5. Diversity analysis
    diversity = quality_system.diversity_tracker.get_diversity_summary(
        canonical_registry.canonical_circuits
    )
    print(f"\n🌈 Circuit Diversity:")
    print(f"   Operation types: {diversity['operation_type_count']}")
    print(f"   Diversity score: {diversity['diversity_score']:.3f}")
    print(f"   Distribution: {diversity['layer_distribution']}")

    # 6. Circuit removal history analysis
    removal_history = quality_system.get_removal_history()
    if removal_history:
        print(f"\n🗑️  Removal History:")

        # Group by operation type
        removed_by_type = {}
        for removal in removal_history:
            op_type = removal['operation_type']
            removed_by_type[op_type] = removed_by_type.get(op_type, 0) + 1

        print(f"   Removed by type: {dict(removed_by_type)}")

        # Average age at removal
        avg_age = sum(r['age'] for r in removal_history) / len(removal_history)
        print(f"   Average age at removal: {avg_age:.1f} epochs")

    # 📊 7. DATAFRAME-SPECIFIC INSIGHTS
    analyze_dataframe_insights(df_circuit)


def analyze_dataframe_insights(df_circuit):
    """
    Advanced analysis using the comprehensive DataFrame
    """
    if df_circuit.empty:
        return

    import pandas as pd
    import numpy as np

    print(f"\n🔬 Advanced DataFrame Insights:")
    print("=" * 35)

    # Circuit survival analysis
    if 'lifecycle_state' in df_circuit.columns:
        # Find circuits that were removed
        final_epoch = df_circuit['epoch'].max()
        final_circuits = df_circuit[df_circuit['epoch'] == final_epoch]['circuit_id'].unique()
        all_circuits = df_circuit['circuit_id'].unique()
        removed_circuits = set(all_circuits) - set(final_circuits)

        survival_rate = len(final_circuits) / len(all_circuits) if all_circuits else 0
        print(f"   Circuit survival rate: {survival_rate:.2%} ({len(final_circuits)}/{len(all_circuits)})")

    # Quality predictors analysis
    if 'overall_quality' in df_circuit.columns and 'circuit_age' in df_circuit.columns:
        # Correlation between age and quality
        age_quality_corr = df_circuit['circuit_age'].corr(df_circuit['overall_quality'])
        print(f"   Age-Quality correlation: {age_quality_corr:.3f}")

        # Quality by operation type
        quality_by_type = df_circuit.groupby('operation_type')['overall_quality'].mean().sort_values(ascending=False)
        print(f"   Best operation types: {dict(quality_by_type.head(3))}")

    # Protection effectiveness
    if 'is_protected' in df_circuit.columns:
        protected_circuits = df_circuit[df_circuit['is_protected'] == True]['circuit_id'].nunique()
        total_unique_circuits = df_circuit['circuit_id'].nunique()
        protection_rate = protected_circuits / total_unique_circuits if total_unique_circuits > 0 else 0
        print(f"   Circuits ever protected: {protection_rate:.2%}")

    # Quality evolution patterns
    if len(df_circuit) > 100:  # Only if we have enough data
        # Find circuits that improved significantly
        circuit_quality_trends = df_circuit.groupby('circuit_id').agg({
            'overall_quality': ['first', 'last', 'count'],
            'epoch': ['first', 'last']
        }).round(3)

        # Flatten column names
        circuit_quality_trends.columns = ['_'.join(col).strip() for col in circuit_quality_trends.columns.values]

        # Calculate improvement
        circuit_quality_trends['quality_improvement'] = (
                circuit_quality_trends['overall_quality_last'] - circuit_quality_trends['overall_quality_first']
        )

        # Find most improved circuits
        most_improved = circuit_quality_trends.nlargest(3, 'quality_improvement')
        print(f"   Most improved circuits (Δquality):")
        for circuit_id, row in most_improved.iterrows():
            print(f"     {circuit_id[:30]}...: +{row['quality_improvement']:.3f}")

    # Diversity trends over time
    if 'operation_rarity' in df_circuit.columns:
        final_diversity = df_circuit[df_circuit['epoch'] == df_circuit['epoch'].max()]
        rare_circuits = (final_diversity['operation_rarity'] > 0.7).sum()
        common_circuits = (final_diversity['operation_rarity'] < 0.3).sum()
        print(f"   Final diversity: {rare_circuits} rare, {common_circuits} common circuits")


def dataframe_analysis_examples():
    """
    Show examples of advanced DataFrame analysis
    """
    print("\n📊 DataFrame Analysis Examples:")
    print("=" * 32)

    example_code = '''
    # Example analyses you can perform with the comprehensive DataFrame:

    # 1. Circuit evolution over time
    quality_evolution = df_circuit.groupby(['epoch', 'operation_type'])['overall_quality'].mean().unstack()
    quality_evolution.plot(title='Quality Evolution by Operation Type')

    # 2. Survival analysis
    circuit_lifespans = df_circuit.groupby('circuit_id').agg({
        'epoch': ['min', 'max'],
        'overall_quality': 'mean',
        'lifecycle_state': 'last'
    })

    # 3. Quality predictors
    from sklearn.ensemble import RandomForestRegressor
    features = ['circuit_age', 'total_detections', 'temporal_quality', 'attention_quality']
    X = df_circuit[features].fillna(0)
    y = df_circuit['overall_quality']
    rf = RandomForestRegressor().fit(X, y)
    feature_importance = dict(zip(features, rf.feature_importances_))

    # 4. Removal prediction
    removal_risk = df_circuit[df_circuit['removal_votes'] > 0]['circuit_id'].unique()

    # 5. Protection effectiveness
    protected_outcomes = df_circuit[df_circuit['is_protected'] == True].groupby('circuit_id')['overall_quality'].agg(['first', 'last', 'mean'])

    # 6. Circuit clustering by behavior
    from sklearn.cluster import KMeans
    behavioral_features = ['attention_entropy', 'pattern_consistency', 'behavioral_impact']
    X_behavior = df_circuit[behavioral_features].fillna(0)
    clusters = KMeans(n_clusters=3).fit_predict(X_behavior)
    df_circuit['behavior_cluster'] = clusters

    # 7. Time series analysis
    circuit_counts_over_time = df_circuit.groupby('epoch')['circuit_id'].nunique()
    quality_trends = df_circuit.groupby('epoch')['overall_quality'].mean()
    '''

    print(example_code)


def real_testing_examples():
    """
    Examples of using real functional testing
    """
    print("\n🧪 Real Functional Testing Examples:")
    print("=" * 37)

    print("""
    DEVELOPMENT vs PRODUCTION WORKFLOWS:
    ───────────────────────────────────────

    # DEVELOPMENT: Fast iteration with mock testing
    dev_results = train_with_enhanced_circuit_management(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        enable_real_testing=False,           # Mock testing for speed
        circuit_assessment_interval=10,      # Frequent assessment
        circuit_management_config={
            'max_circuits': 200,              # Larger population for exploration
            'min_quality_threshold': 0.2,     # Lower bar for development
        }
    )

    # PRODUCTION: Accurate validation with real testing  
    final_results = train_with_enhanced_circuit_management(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        enable_real_testing=True,            # Real activation patching
        circuit_assessment_interval=50,      # Less frequent but thorough
        circuit_management_config={
            'max_circuits': 100,              # Smaller, higher-quality population
            'min_quality_threshold': 0.4,     # Higher bar for final results
            'probation_epochs': 150,          # Longer protection period
        }
    )


    COMPARING MOCK vs REAL TESTING RESULTS:
    ──────────────────────────────────────────

    # Compare circuit quality scores
    mock_df = dev_results['circuit_dataframe']
    real_df = final_results['circuit_dataframe']

    # Real testing typically finds fewer but higher-quality circuits
    print(f"Mock testing: {mock_df['circuit_id'].nunique()} circuits, "
          f"avg quality: {mock_df['overall_quality'].mean():.3f}")
    print(f"Real testing: {real_df['circuit_id'].nunique()} circuits, "
          f"avg quality: {real_df['overall_quality'].mean():.3f}")

    # Functional quality is more meaningful with real testing
    real_functional = real_df['functional_quality'].mean()
    print(f"Real functional quality: {real_functional:.3f} (based on activation patching)")


    TASK-SPECIFIC REAL TESTING:
    ───────────────────────────────

    # For custom tasks, you can integrate your own testing methods
    from analysis.validation.circuit_testing import CircuitFunctionalTester

    class MyTaskCircuitTester(CircuitFunctionalTester):
        def __init__(self, model, task_type='my_task'):
            super().__init__(model)
            self.task_type = task_type

        def _test_modular_copy_pattern(self, tokens, source_pos, target_pos, 
                                     baseline_logits, ablated_logits):
            # Override with your task-specific copy testing
            if self.task_type == 'my_task':
                return self._test_my_task_copy_pattern(tokens, source_pos, target_pos,
                                                      baseline_logits, ablated_logits)
            else:
                return super()._test_modular_copy_pattern(tokens, source_pos, target_pos,
                                                        baseline_logits, ablated_logits)

    # Then integrate your custom tester (you'd need to modify the integration function)
    """)


def migration_from_old_system():
    """
    Guide for migrating from the old train_with_canonical_circuits
    """

    print("🔄 Migration Guide from Old System:")
    print("=" * 40)

    print("""
    OLD APPROACH (train_with_canonical_circuits.py):
    ──────────────────────────────────────────────────

    # Multiple separate systems
    canonical_detector = CanonicalAwareAdaptiveTokenOperationDetector(...)
    canonical_registry = CanonicalCircuitRegistry(...)

    # Manual integration of real testing
    from analysis.validation.circuit_testing import integrate_real_testing_with_quality_analyzer
    circuits_quality_analyzer = CircuitQualityAnalyzer(model, canonical_registry, save_dir)
    circuits_quality_analyzer = integrate_real_testing_with_quality_analyzer(circuits_quality_analyzer)

    # Misleading "pruning" that doesn't actually remove
    stable_ids, most_stable, least_stable = canonical_detector.prune_unstable_canonical_circuits(epoch)

    # Quality analysis separate from action
    quality_results = circuits_quality_analyzer.analyze_all_circuits(...)

    # Manual DataFrame logging
    batch_dict = {col: [] for col in df_circuit.columns}
    # ... manual data collection ...
    new_batch = pd.DataFrame(batch_dict)
    df_circuit = pd.concat([df_circuit, new_batch], ignore_index=True)


    NEW APPROACH (Enhanced System):
    ──────────────────────────────────────────

    # Single unified system with automatic real testing integration
    results = train_with_enhanced_circuit_management(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        circuit_management_config=config,
        enable_circuit_removal=True,        # Actually removes circuits!
        enable_real_testing=True,           # 🧪 Automatic real testing integration
        ...
    )

    # Automatic lifecycle management with real behavioral validation
    # Quality assessment drives removal decisions using activation patching
    # Diversity protection built-in
    # Comprehensive DataFrame logging automatic
    # Real functional testing integrated seamlessly
    """)

    print("\nSTEP-BY-STEP MIGRATION:")
    print("1. Replace train_with_canonical_circuits with train_with_enhanced_circuit_management")
    print("2. Remove manual integrate_real_testing_with_quality_analyzer calls")
    print("3. Remove manual 'pruning' calls (they don't actually prune)")
    print("4. Remove separate CircuitQualityAnalyzer instantiation (handled internally)")
    print("5. Remove manual DataFrame logging (automatic in new system)")
    print("6. Add circuit_management_config parameter")
    print("7. Set enable_real_testing=True for activation patching")
    print("8. Update logging to use the new comprehensive metrics")


def dataframe_analysis_examples():
    """
    Show examples of advanced DataFrame analysis
    """
    print("\n📊 DataFrame Analysis Examples:")
    print("=" * 32)

    example_code = '''
    # Example analyses you can perform with the comprehensive DataFrame:

    # 1. Circuit evolution over time
    quality_evolution = df_circuit.groupby(['epoch', 'operation_type'])['overall_quality'].mean().unstack()
    quality_evolution.plot(title='Quality Evolution by Operation Type')

    # 2. Real vs Mock testing comparison
    real_circuits = df_circuit[df_circuit['functional_quality'] > 0]  # Real testing data
    mock_circuits = df_circuit[df_circuit['functional_quality'] == 0]  # Mock testing data

    print(f"Real testing circuits: {len(real_circuits)} (avg quality: {real_circuits['overall_quality'].mean():.3f})")
    print(f"Mock testing circuits: {len(mock_circuits)} (avg quality: {mock_circuits['overall_quality'].mean():.3f})")

    # 3. Behavioral validation effectiveness
    consistent_behavior = df_circuit[df_circuit['pattern_consistency'] > 0.7]
    print(f"Behaviorally consistent circuits: {len(consistent_behavior)}")

    # 4. Functional testing insights
    copy_circuits = df_circuit[df_circuit['operation_type'] == 'copy']
    effective_copy = copy_circuits[copy_circuits['functional_quality'] > 0.6]
    print(f"Effective copy circuits (real testing): {len(effective_copy)}/{len(copy_circuits)}")

    # 5. Circuit survival analysis by functional quality
    final_epoch_data = df_circuit[df_circuit['epoch'] == df_circuit['epoch'].max()]
    survival_by_quality = final_epoch_data.groupby(pd.cut(final_epoch_data['functional_quality'], 
                                                          bins=[0, 0.3, 0.6, 1.0]))['circuit_id'].count()

    # 6. Protection effectiveness with real testing
    protected_and_functional = df_circuit[
        (df_circuit['is_protected'] == True) & 
        (df_circuit['functional_quality'] > 0.5)
    ]
    print(f"Protected circuits that are actually functional: {len(protected_and_functional)}")
    '''


def migration_from_old_system():
    """
    Guide for migrating from the old train_with_canonical_circuits
    """

    print("🔄 Migration Guide from Old System:")
    print("=" * 40)

    print("""
    OLD APPROACH (train_with_canonical_circuits.py):
    ──────────────────────────────────────────────────

    # Multiple separate systems
    canonical_detector = CanonicalAwareAdaptiveTokenOperationDetector(...)
    canonical_registry = CanonicalCircuitRegistry(...)

    # Misleading "pruning" that doesn't actually remove
    stable_ids, most_stable, least_stable = canonical_detector.prune_unstable_canonical_circuits(epoch)

    # Quality analysis separate from action
    quality_results = CircuitQualityAnalyzer.analyze_all_circuits(...)

    # No actual circuit removal
    # Manual tracking of circuit states


    NEW APPROACH (Enhanced System):
    ──────────────────────────────────────────

    # Single unified system
    results = train_with_enhanced_circuit_management(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        circuit_management_config=config,
        enable_circuit_removal=True,  # Actually removes circuits!
        ...
    )

    # Automatic lifecycle management
    # Quality assessment drives removal decisions
    # Diversity protection built-in
    # Comprehensive logging and analysis
    """)

    print("\nSTEP-BY-STEP MIGRATION:")
    print("1. Replace train_with_canonical_circuits with train_with_enhanced_circuit_management")
    print("2. Remove manual 'pruning' calls (they don't actually prune)")
    print("3. Remove separate CircuitQualityAnalyzer instantiation (handled internally)")
    print("4. Add circuit_management_config parameter")
    print("5. Set enable_circuit_removal=True if you want actual removal")
    print("6. Update logging to use the new metrics structure")

'''
def advanced_usage_examples():
    """
    Advanced usage patterns for the enhanced system
    """

    print("\n🔬 Advanced Usage Patterns:")
    print("=" * 30)

    # Example 1: Custom circuit protection
    def custom_protection_example():
        """Manually protect important circuits"""

        # During training, protect specific circuits
        results = train_with_enhanced_circuit_management(...)
        quality_system = results['quality_system']

        # Protect a specific circuit from removal
        important_circuit_id = "copy_token_operation_abc123"
        quality_system.force_protect_circuit(
            circuit_id=important_circuit_id,
            reason="Critical for main task performance"
        )

    # Example 2: Dynamic configuration adjustment
    def dynamic_config_example():
        """Adjust circuit management during training"""

        # Start conservative, become more aggressive
        def get_epoch_config(epoch):
            if epoch < 500:
                return get_conservative_circuit_config()
            elif epoch < 1000:
                return get_default_circuit_config()
            else:
                return get_aggressive_circuit_config()

        # You'd implement this in a custom training loop

    # Example 3: Analysis-driven decisions
    def analysis_driven_example():
        """Use quality analysis to make training decisions"""

        results = train_with_enhanced_circuit_management(...)
        quality_system = results['quality_system']

        # Analyze circuit quality distribution
        quality_assessment = quality_system.quality_analyzer.assess_circuit_quality(
            eval_loader, max_circuits=1000
        )

        # Make decisions based on circuit quality
        mean_quality = quality_assessment['quality_report']['quality_statistics']['mean_quality']

        if mean_quality < 0.4:
            print("⚠️  Low mean circuit quality - consider adjusting detection thresholds")
        elif mean_quality > 0.8:
            print("✅ High mean circuit quality - system working well")

        # Check diversity
        diversity = quality_system.diversity_tracker.get_diversity_summary(
            quality_system.canonical_registry.canonical_circuits
        )

        if diversity['diversity_score'] < 0.5:
            print("⚠️  Low circuit diversity - consider protecting more circuit types")
'''

'''
def debugging_and_monitoring():
    """
    How to debug and monitor the enhanced system
    """

    print("\n🔧 Debugging and Monitoring:")
    print("=" * 28)

    # Access internal state for debugging
    def debug_circuit_state(quality_system):
        """Debug current circuit state"""

        # Check lifecycle distribution
        lifecycle_counts = {}
        for circuit_info in quality_system.circuit_lifecycles.values():
            state = circuit_info.state.value
            lifecycle_counts[state] = lifecycle_counts.get(state, 0) + 1

        print(f"Circuit lifecycle distribution: {lifecycle_counts}")

        # Check quality trends
        for circuit_id, lifecycle in quality_system.circuit_lifecycles.items():
            if len(lifecycle.quality_history) > 5:
                recent_trend = np.mean(lifecycle.quality_history[-3:]) - np.mean(lifecycle.quality_history[:3])
                if recent_trend < -0.2:
                    print(f"⚠️  {circuit_id} shows declining quality trend: {recent_trend:.3f}")

        # Check for circuits approaching removal
        removal_candidates = [
            circuit_id for circuit_id, lifecycle in quality_system.circuit_lifecycles.items()
            if lifecycle.state.value in ['declining', 'marked_for_removal']
        ]

        if removal_candidates:
            print(f"🚨 {len(removal_candidates)} circuits at risk of removal")

    # Monitor during training
    def monitor_during_training():
        """Add monitoring hooks during training"""

        # You can add custom monitoring by modifying the training loop
        # or by checking the circuit_management_history after each assessment
        pass
'''
'''
if __name__ == "__main__":
    # Run the complete example
    results = main_training_example()

    # Show migration guide
    migration_from_old_system()

    # Show advanced usage
    advanced_usage_examples()

    # Show debugging tips
    debugging_and_monitoring()

    # 📊 NEW: Show DataFrame analysis examples
    dataframe_analysis_examples()

    # 🧪 NEW: Show real testing examples
    real_testing_examples()

    # 📊 Access the DataFrame for your own analysis
    df_circuit = results['circuit_dataframe']
    circuit_snapshots = results['circuit_snapshots']

    print(f"\n💾 Training complete! Circuit DataFrame saved with {len(df_circuit)} records")
    print(f"Columns available: {list(df_circuit.columns)}")

    # Example: Save for external analysis
    df_circuit.to_csv("my_circuit_analysis.csv", index=False)

    # Example: Quick quality analysis
    if not df_circuit.empty:
        final_epoch_data = df_circuit[df_circuit['epoch'] == df_circuit['epoch'].max()]
        best_circuit = final_epoch_data.loc[final_epoch_data['overall_quality'].idxmax()]
        print(f"🏆 Best final circuit: {best_circuit['circuit_id']} (quality: {best_circuit['overall_quality']:.3f})")
'''