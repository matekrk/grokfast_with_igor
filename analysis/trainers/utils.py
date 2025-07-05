import torch

from analysis.core import CanonicalCircuitRegistry
from analysis.core.canonical_circuit_system import FunctionalCircuitSignatureExtractor
from analysis.core.json_safe_canonical_circuits import JSONSafeCanonicalCircuitRegistry


def train_epoch(model, train_loader, criterion, optimizer, epoch, device, scheduler=None):
    """Train for one epoch with standard batch format"""
    model.train()
    train_correct = train_total = 0
    train_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Inputs should be [batch_size, seq_len], targets should be [batch_size]
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass - model expects [batch_size, seq_len] and returns [batch_size, num_tokens]
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        train_total += targets.size(0)
        train_correct += (predicted == targets).sum().item()
        train_loss += loss.item() * targets.size(0)

    if scheduler is not None:
        scheduler.step()

    # Return averages
    train_accuracy = train_correct / train_total if train_total > 0 else 0.0
    train_loss = train_loss / train_total if train_total > 0 else 0.0

    return {'accuracy': train_accuracy, 'loss': train_loss}


def evaluate(model, eval_loader, criterion, device):
    """Evaluate the model on the provided data"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in eval_loader:
            # Move to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update statistics
            total_loss += loss.item() * targets.size(0)
            predicted = outputs.argmax(dim=-1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Calculate metrics
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def log_metrics(model, epoch, train_stats, eval_stats):
    """Log training and evaluation metrics"""
    # Use model logger if available
    if hasattr(model, 'logger'):
        logger = model.logger

        # Log training metrics
        if train_stats:
            logger.log_data('training', 'epoch', epoch)
            logger.log_data('training', 'loss', train_stats['loss'])
            logger.log_data('training', 'accuracy', train_stats['accuracy'])

        # Log evaluation metrics
        if eval_stats:
            logger.log_data('evaluation', 'epoch', epoch)
            logger.log_data('evaluation', 'loss', eval_stats['loss'])
            logger.log_data('evaluation', 'accuracy', eval_stats['accuracy'])

    # Print metrics
    print(f"Epoch  {epoch:5d}: "
          f"\tTrain Loss={train_stats['loss']:.4g}, "
          f"\tAcc={train_stats['accuracy']:5.3f}, "
          f"\t\tVal Loss={eval_stats['loss']:6.4g}, "
          f"\tAcc={eval_stats['accuracy']:5.3f}")


def detect_grokking(model, epoch, train_stats, eval_stats):
    """Detect if grokking is occurring at this epoch"""
    # Skip if model has no logger or if stats are missing
    if not hasattr(model, 'logger') or not train_stats or not eval_stats:
        return False

    logger = model.logger

    # Check for grokking conditions:
    # 1. Training accuracy is high (memorization)
    # 2. Evaluation accuracy is rapidly improving

    # Check if we have enough history
    if logger.get_length('evaluation', 'accuracy') >= 5:
        # Get recent history
        recent_eval_accs = logger.logs['evaluation']['accuracy'][-5:]

        # Check if training accuracy is high
        train_high = train_stats['accuracy'] > 0.9

        # Check if evaluation accuracy is improving rapidly
        prev_eval_accs = recent_eval_accs[:-1]  # All but the latest
        prev_avg = sum(prev_eval_accs) / len(prev_eval_accs) if prev_eval_accs else 0

        significant_improvement = (eval_stats['accuracy'] > prev_avg * 1.2)

        # Detect potential grokking
        if train_high and significant_improvement:
            print(f"\tdetect_grokking()\tPotential grokking detected at epoch {epoch}")

            # Log the grokking point
            logger.log_data('grokking_phases', 'grokking_step', epoch)
            return True

    return False


def process_jumps(model, weight_tracker, eval_loader, criterion, optimizer):
    """Process pending jumps detected by the weight tracker"""
    # Get a batch of data for analysis
    sample_inputs, sample_targets = next(iter(eval_loader))
    sample_inputs = sample_inputs.to(next(model.parameters()).device)
    sample_targets = sample_targets.to(next(model.parameters()).device)

    jump_results = weight_tracker.analyze_pending_jumps(
        inputs=sample_inputs,
        targets=sample_targets,
        criterion=criterion,
        optimizer=optimizer,
        jump_analyzer=None,  # We'll handle this separately
        eval_loader=eval_loader,
        mini_train_steps=weight_tracker.sliding_window_size - 1,
    )
    # Process jumps

    # Print summary of processed jumps
    if jump_results:
        print(f"\tprocess_jumps()\tProcessed {len(jump_results)} weight space jumps:")
        for result in jump_results:
            jump_epoch = result['jump_epoch']
            jump_char = result['characterization']

            print(f"\t\tJump at epoch {jump_epoch}: "
                  f"Magnitude={jump_char['total_magnitude']['pre_to_jump']:.4f}, "
                  f"Top layers: {', '.join(jump_char['top_layers'][:2])}, "
                  f"Top heads: {', '.join(jump_char['top_heads'][:2])}")

        # Visualize jump timeline
        weight_tracker.visualize_jumps_timeline()

    return jump_results


def extend_canonical_registry_for_new_types(registry: CanonicalCircuitRegistry):
    """Example of how to extend for new circuit types"""

    # Add new extractors
    registry.extractors.append(FunctionalCircuitSignatureExtractor())

    # Could add more extractors for:
    # - Sparse autoencoder features
    # - Superposition circuits
    # - Meta-learning circuits
    # - Compositional reasoning circuits
    # etc.

    return registry


def _get_stats(epoch, train_stats, eval_stats, optimizer):
    return {
        'epoch': epoch,
        "train_loss": train_stats.get('loss', 1000.0),
        "train_accuracy": train_stats.get('accuracy', 0.0),
        "eval_loss": eval_stats.get('loss', 1000.0),
        "eval_accuracy": eval_stats.get('accuracy', 0.0),
        "learning_rate": optimizer.param_groups[0]['lr'],
        # "weight decay": optimizer.param_groups[0]['weight_decay'],
    }


'''
def perform_circuit_detection(canonical_detector, eval_loader, epoch, total_epochs,
                              accuracy, logger):
    """
    Perform circuit detection using existing logic
    """
    # Get sample for analysis
    sample_input, sample_target = next(iter(eval_loader))
    tokens = [str(int(x)) for x in sample_input[0].cpu().numpy()]

    # Forward pass with attention storage
    outputs = canonical_detector.model(sample_input, store_attention=True)
    attention_patterns = canonical_detector.model.get_attention_patterns()

    # Detect circuits
    copy_results = canonical_detector.detect_and_register_copy_mechanisms(
        attention_patterns=attention_patterns,
        tokens=tokens,
        epoch=epoch,
        total_epochs=total_epochs,
        model_accuracy=accuracy,
        register_circuits=True
    )

    induction_results = canonical_detector.detect_and_register_induction_patterns(
        attention_patterns=attention_patterns,
        tokens=tokens,
        epoch=epoch,
        total_epochs=total_epochs,
        model_accuracy=accuracy,
        register_circuits=True
    )

    return {
        'copy_results': copy_results,
        'induction_results': induction_results
    }
'''


def create_standard_thresholds(stage="early_training"):
    """Create standardized threshold configuration"""

    from analysis import CircuitThresholds
    configs = {
        "early_training": CircuitThresholds(
            copy_attention_min=0.3, copy_attention_max=0.85,
            induction_attention_min=0.4, induction_attention_max=0.8,
            warmup_epochs=100, min_accuracy_threshold=0.15
        ),
        "late_training": CircuitThresholds(
            copy_attention_min=0.5, copy_attention_max=0.9,
            induction_attention_min=0.6, induction_attention_max=0.85,
            warmup_epochs=50, min_accuracy_threshold=0.3
        )
    }

    return configs[stage]


def validate_canonical_system_health(circuit_system):
    """Validate system before proceeding with Phase 1"""

    issues = []

    # Check canonical registry type
    if not isinstance(circuit_system['canonical_registry'], JSONSafeCanonicalCircuitRegistry):
        issues.append("Not using JSON-safe canonical registry")

    # Check detector type
    from analysis.analyzers.adaptive_token_operations import CanonicalAwareAdaptiveTokenOperationDetector
    if not isinstance(circuit_system['canonical_detector'], CanonicalAwareAdaptiveTokenOperationDetector):
        issues.append("Not using canonical-aware detector")

    # Test JSON serialization
    try:
        test_data = circuit_system['canonical_registry'].get_registry_summary()
        from analysis.utils.utils import CircuitJSONEncoder
        import json
        json.dumps(test_data, cls=CircuitJSONEncoder)
    except Exception as e:
        issues.append(f"JSON serialization test failed: {e}")

    if issues:
        print("❌ Fix these issues before continuing:")
        for issue in issues:
            print(f"  - {issue}")
        exit(1)

    return issues


# ============================================================================
# 📊 ENHANCED DATAFRAME LOGGING FUNCTIONS
# ============================================================================

def initialize_circuit_dataframe():
    """
    Initialize DataFrame with comprehensive circuit tracking columns
    """
    import pandas as pd

    return pd.DataFrame({
        # Basic identification
        'epoch': pd.Series(dtype='int'),
        'circuit_id': pd.Series(dtype='str'),
        'operation_type': pd.Series(dtype='str'),

        # Lifecycle information
        'lifecycle_state': pd.Series(dtype='str'),
        'circuit_age': pd.Series(dtype='int'),
        'first_seen': pd.Series(dtype='int'),
        'last_seen': pd.Series(dtype='int'),
        'total_detections': pd.Series(dtype='int'),

        # Quality metrics (comprehensive)
        'overall_quality': pd.Series(dtype='float'),
        'temporal_quality': pd.Series(dtype='float'),
        'functional_quality': pd.Series(dtype='float'),
        'attention_quality': pd.Series(dtype='float'),
        'stability_score': pd.Series(dtype='float'),

        # Management status
        'recommendation': pd.Series(dtype='str'),
        'removal_votes': pd.Series(dtype='int'),
        'is_protected': pd.Series(dtype='bool'),
        'protection_reason': pd.Series(dtype='str'),

        # Trend analysis
        'quality_trend': pd.Series(dtype='float'),  # Recent trend in quality
        'strengthening': pd.Series(dtype='bool'),  # Is circuit getting stronger?

        # Diversity and uniqueness
        'operation_rarity': pd.Series(dtype='float'),  # How rare is this operation type?
        'uniqueness_score': pd.Series(dtype='float'),  # How unique is this circuit?

        # Behavioral metrics
        'attention_entropy': pd.Series(dtype='float'),
        'pattern_consistency': pd.Series(dtype='float'),
        'cross_example_robustness': pd.Series(dtype='float'),

        # Performance impact
        'attribution_strength': pd.Series(dtype='float'),
        'behavioral_impact': pd.Series(dtype='float'),

        # Meta information
        'assessment_count': pd.Series(dtype='int'),
        'last_assessment_epoch': pd.Series(dtype='int'),
    })


def log_circuits_to_dataframe(quality_system, management_results, epoch,
                              df_circuit, eval_loader):
    """
    🔧 FIXED: Pass eval_loader instead of None
    """
    import pandas as pd
    import numpy as np

    # ✅ FIXED: Pass eval_loader instead of None
    quality_results = quality_system.quality_analyzer.analyze_all_circuits(
        eval_loader, max_circuits=1000  # ✅ Now passes the actual eval_loader
    )

    # Get diversity analysis
    diversity_analysis = quality_system.diversity_tracker.analyze_diversity(
        quality_system.canonical_registry.canonical_circuits
    )

    # Prepare batch data (same as before)
    batch_dict = {col: [] for col in df_circuit.columns}

    # Process each circuit
    for circuit_id, analysis in quality_results['circuit_analyses'].items():
        # Basic identification
        batch_dict['epoch'].append(epoch)
        batch_dict['circuit_id'].append(circuit_id)
        batch_dict['operation_type'].append(analysis['operation_type'])

        # Get lifecycle info
        lifecycle_info = quality_system.circuit_lifecycles.get(circuit_id)
        canonical_circuit = quality_system.canonical_registry.canonical_circuits.get(circuit_id)

        if lifecycle_info:
            batch_dict['lifecycle_state'].append(lifecycle_info.state.value)
            batch_dict['circuit_age'].append(epoch - lifecycle_info.first_detected)
            batch_dict['removal_votes'].append(lifecycle_info.removal_votes)
            batch_dict['is_protected'].append(lifecycle_info.state.value == 'protected')
            batch_dict['protection_reason'].append(lifecycle_info.protection_reason or '')
            batch_dict['assessment_count'].append(lifecycle_info.assessment_count)
            batch_dict['last_assessment_epoch'].append(lifecycle_info.last_assessment_epoch)

            # Calculate quality trend
            if len(lifecycle_info.quality_history) >= 3:
                recent = np.mean(lifecycle_info.quality_history[-3:])
                earlier = np.mean(lifecycle_info.quality_history[:3])
                quality_trend = recent - earlier
            else:
                quality_trend = 0.0
            batch_dict['quality_trend'].append(quality_trend)
        else:
            # Default values for missing lifecycle info
            batch_dict['lifecycle_state'].append('unknown')
            batch_dict['circuit_age'].append(0)
            batch_dict['removal_votes'].append(0)
            batch_dict['is_protected'].append(False)
            batch_dict['protection_reason'].append('')
            batch_dict['assessment_count'].append(0)
            batch_dict['last_assessment_epoch'].append(epoch)
            batch_dict['quality_trend'].append(0.0)

        if canonical_circuit:
            batch_dict['first_seen'].append(canonical_circuit.first_seen)
            batch_dict['last_seen'].append(canonical_circuit.last_seen)
            batch_dict['total_detections'].append(canonical_circuit.total_detections)
            batch_dict['attribution_strength'].append(canonical_circuit.best_attribution)
        else:
            batch_dict['first_seen'].append(epoch)
            batch_dict['last_seen'].append(epoch)
            batch_dict['total_detections'].append(1)
            batch_dict['attribution_strength'].append(0.0)

        # Quality metrics (comprehensive breakdown)
        batch_dict['overall_quality'].append(analysis['quality_score'])
        batch_dict['stability_score'].append(analysis['stability_score'])

        # Extract component scores safely
        temporal_analysis = analysis.get('temporal_analysis', {})
        functional_analysis = analysis.get('functional_analysis', {})
        attention_analysis = analysis.get('attention_analysis', {})

        batch_dict['temporal_quality'].append(temporal_analysis.get('temporal_consistency', 0.0))
        batch_dict['functional_quality'].append(functional_analysis.get('functional_quality', 0.0))
        batch_dict['attention_quality'].append(attention_analysis.get('attention_quality', 0.0))

        # Behavioral metrics
        batch_dict['attention_entropy'].append(attention_analysis.get('avg_attention_entropy', 0.0))
        batch_dict['pattern_consistency'].append(attention_analysis.get('pattern_consistency', 0.0))
        batch_dict['behavioral_impact'].append(functional_analysis.get('avg_consistency', 0.0))
        batch_dict['cross_example_robustness'].append(functional_analysis.get('behavior_variance', 1.0))

        # Management status
        batch_dict['recommendation'].append(analysis.get('recommendation', 'UNKNOWN').split(' ')[0])

        # Calculate operation rarity and uniqueness
        op_type = analysis['operation_type']
        total_circuits = len(quality_results['circuit_analyses'])
        same_type_count = len(diversity_analysis['by_operation_type'].get(op_type, []))
        operation_rarity = 1.0 - (same_type_count / total_circuits) if total_circuits > 0 else 0.0
        batch_dict['operation_rarity'].append(operation_rarity)

        # Uniqueness score (combination of rarity and quality)
        uniqueness_score = operation_rarity * analysis['quality_score']
        batch_dict['uniqueness_score'].append(uniqueness_score)

        # Strengthening trend
        temporal_trend = temporal_analysis.get('strengthening_trend', False)
        batch_dict['strengthening'].append(temporal_trend)

    # Add batch to DataFrame
    if batch_dict['epoch']:  # Only if we have data
        new_batch = pd.DataFrame(batch_dict)
        df_circuit = pd.concat([df_circuit, new_batch], ignore_index=True)

    # Create summary snapshot for this epoch
    snapshot = {
        'epoch': epoch,
        'total_circuits': len(batch_dict['epoch']),
        'mean_quality': np.mean(batch_dict['overall_quality']) if batch_dict['overall_quality'] else 0.0,
        'circuits_removed': management_results['circuit_management']['circuits_removed'],
        'circuits_protected': management_results['circuit_management']['circuits_protected'],
        'diversity_score': management_results['diversity_analysis']['diversity_score'],
        'lifecycle_distribution': management_results['lifecycle_distribution'],
    }

    return df_circuit, snapshot


def log_circuits_to_dataframe_assessment_only(quality_system, quality_results, epoch, df_circuit):
    """
    Simplified DataFrame logging when circuit removal is disabled
    """
    import pandas as pd
    import numpy as np

    # Simplified logging without lifecycle management
    batch_dict = {col: [] for col in df_circuit.columns}

    for circuit_id, analysis in quality_results['circuit_analyses'].items():
        # Basic info
        batch_dict['epoch'].append(epoch)
        batch_dict['circuit_id'].append(circuit_id)
        batch_dict['operation_type'].append(analysis['operation_type'])

        # Quality info
        batch_dict['overall_quality'].append(analysis['quality_score'])
        batch_dict['stability_score'].append(analysis['stability_score'])
        batch_dict['recommendation'].append(analysis.get('recommendation', 'UNKNOWN').split(' ')[0])

        # Registry info
        canonical_circuit = quality_system.canonical_registry.canonical_circuits.get(circuit_id)
        if canonical_circuit:
            batch_dict['first_seen'].append(canonical_circuit.first_seen)
            batch_dict['last_seen'].append(canonical_circuit.last_seen)
            batch_dict['total_detections'].append(canonical_circuit.total_detections)
            batch_dict['circuit_age'].append(epoch - canonical_circuit.first_seen)
        else:
            # Defaults
            batch_dict['first_seen'].append(epoch)
            batch_dict['last_seen'].append(epoch)
            batch_dict['total_detections'].append(1)
            batch_dict['circuit_age'].append(0)

        # Fill other columns with defaults
        for col in df_circuit.columns:
            if col not in batch_dict or len(batch_dict[col]) < len(batch_dict['epoch']):
                if df_circuit[col].dtype == 'float64':
                    batch_dict.setdefault(col, []).append(0.0)
                elif df_circuit[col].dtype == 'int64':
                    batch_dict.setdefault(col, []).append(0)
                elif df_circuit[col].dtype == 'bool':
                    batch_dict.setdefault(col, []).append(False)
                else:
                    batch_dict.setdefault(col, []).append('')

    # Add to DataFrame
    if batch_dict['epoch']:
        new_batch = pd.DataFrame(batch_dict)
        df_circuit = pd.concat([df_circuit, new_batch], ignore_index=True)

    # Simple snapshot
    snapshot = {
        'epoch': epoch,
        'total_circuits': len(batch_dict['epoch']),
        'mean_quality': np.mean(batch_dict['overall_quality']) if batch_dict['overall_quality'] else 0.0,
        'assessment_only': True,
    }

    return snapshot


def analyze_training_results(results):
    """
    Analyze the results of enhanced circuit training
    """
    model = results['model']
    eval_loader = results['eval_loader']
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
    quality_assessment = quality_system.quality_analyzer.analyze_all_circuits(
        eval_loader=eval_loader,  # You'd pass your eval_loader here
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

    print(f"\n🔬 Advanced DataFrame Insights:")
    print("=" * 35)

    # Circuit survival analysis
    if 'lifecycle_state' in df_circuit.columns:
        # Find circuits that were removed
        final_epoch = df_circuit['epoch'].max()
        final_circuits = df_circuit[df_circuit['epoch'] == final_epoch]['circuit_id'].unique()
        all_circuits = df_circuit['circuit_id'].unique()
        removed_circuits = set(all_circuits) - set(final_circuits)

        survival_rate = len(final_circuits) / len(all_circuits) if len(all_circuits) > 0 else 0
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
        print(f"   Most improved circuits (Δ quality):")
        for circuit_id, row in most_improved.iterrows():
            print(f"     {circuit_id[:30]}...: +{row['quality_improvement']:.3f}")

    # Diversity trends over time
    if 'operation_rarity' in df_circuit.columns:
        final_diversity = df_circuit[df_circuit['epoch'] == df_circuit['epoch'].max()]
        rare_circuits = (final_diversity['operation_rarity'] > 0.7).sum()
        common_circuits = (final_diversity['operation_rarity'] < 0.3).sum()
        print(f"   Final diversity: {rare_circuits} rare, {common_circuits} common circuits")


def get_sampling_configs(strategy="aggressive", num_samples=12):
    # info aggressive exploration whatis (early, higher computational cost)
    sampling_configs = {
        "exploration": {
            "base_budget": 5,
            "max_cache_size": 100,
            "diversity_metrics": ["entropy", "repetition", "unique_tokens", "sequential"]
        },
        # info conservative sampling whatis (middle, low computational cost)
        "conservative": {
            "base_budget": 2,
            "max_cache_size": 20,
            "diversity_metrics": ["entropy", "repetition"]
        },
        # info late-training focused whatis (late, minimal cost)
        "focused": {
            "base_budget": 1,
            "max_cache_size": 10,
            "diversity_metrics": ["entropy"]
        },
        "aggressive": {
            'base_budget': num_samples,  # Increased budget
            'max_recent_history': 500,
            'repetition_penalty': 0.925,
            'token_ranges': {'min': 0, 'max': 97},
            'force_rare_tokens': True,
            'rare_token_probability': 0.4
        },
    }
    return sampling_configs[strategy]


def get_default_circuit_config():
    """
    Get default configuration for circuit management
    """
    return {
        'probation_epochs': 100,  # Can't remove circuits younger than this
        'grace_period_epochs': 50,  # Extra time for borderline circuits
        'min_quality_threshold': 0.25,  # Below this = candidate for removal
        'removal_votes_required': 3,  # Need multiple bad assessments
        'max_circuits': 150,  # Maximum circuits to maintain
        'diversity_protection_ratio': 0.15,  # Protect 15% for diversity
        'assessment_interval': 10,  # Epochs between internal assessments
    }


def get_aggressive_circuit_config():
    """
    More aggressive circuit management for faster experimentation
    """
    return {
        'probation_epochs': 50,  # Shorter probation
        'grace_period_epochs': 25,  # Less grace time
        'min_quality_threshold': 0.4,  # Higher quality bar
        'removal_votes_required': 2,  # Faster removal
        'max_circuits': 100,  # Smaller population
        'diversity_protection_ratio': 0.1,  # Less diversity protection
        'assessment_interval': 5,  # More frequent assessment
    }


def get_conservative_circuit_config():
    """
    Conservative circuit management for important experiments
    """
    return {
        'probation_epochs': 200,  # Longer probation
        'grace_period_epochs': 100,  # More grace time
        'min_quality_threshold': 0.15,  # Lower quality bar
        'removal_votes_required': 5,  # Require more votes
        'max_circuits': 300,  # Larger population
        'diversity_protection_ratio': 0.25,  # Strong diversity protection
        'assessment_interval': 20,  # Less frequent assessment
    }


def _should_log_metrics(epoch, circuit_assessment_interval, pre_post_width):
    should_log_metrics = (
            (epoch % circuit_assessment_interval) in range(circuit_assessment_interval - pre_post_width,
                                                           circuit_assessment_interval) or
            (epoch % circuit_assessment_interval) in range(pre_post_width + 1))
    return should_log_metrics


def _should_assess_circuits(epoch, current_accuracy, access_circuits_threshold, circuit_assessment_interval):
    return ((epoch % circuit_assessment_interval) == 0
            and epoch > 50 and current_accuracy > access_circuits_threshold)
