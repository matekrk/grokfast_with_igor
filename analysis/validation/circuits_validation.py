def validate_circuit_importance(canonical_detector, circuit_id, min_epochs=50, min_strength_growth=0.1):
    """
    Distinguish real circuits from spurious ones using temporal dynamics
    """
    evolution = canonical_detector.canonical_registry.analyze_circuit_evolution(circuit_id)

    # Real circuits show strengthening over time during grokking
    if len(evolution['attribution_history']) < min_epochs:
        return False, "insufficient_history"

    # Check for strength growth (real circuits get stronger)
    early_strength = np.mean([attr for epoch, attr in evolution['attribution_history'][:10]])
    late_strength = np.mean([attr for epoch, attr in evolution['attribution_history'][-10:]])

    if late_strength < early_strength + min_strength_growth:
        return False, "no_strengthening"

    # Check for consistency (real circuits are more stable)
    strengths = [attr for _, attr in evolution['attribution_history']]
    cv = np.std(strengths) / max(np.mean(strengths), 0.1)

    if cv > 0.5:  # High coefficient of variation = unstable
        return False, "high_variability"

    return True, "validated"


# Use in your training loop
def prune_spurious_circuits(canonical_detector, epoch):
    if epoch < 100:  # Don't prune too early
        return

    all_circuits = canonical_detector.canonical_registry.canonical_circuits
    spurious_circuits = []

    for circuit_id in all_circuits:
        is_valid, reason = validate_circuit_importance(canonical_detector, circuit_id)
        if not is_valid:
            spurious_circuits.append((circuit_id, reason))

    # Remove spurious circuits to make room for real ones
    for circuit_id, reason in spurious_circuits:
        canonical_detector.canonical_registry.mark_circuit_spurious(circuit_id, reason)