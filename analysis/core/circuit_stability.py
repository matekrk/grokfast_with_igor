# ============================================================================
# 📊 CURRENT STABILITY MEASURES IN THE SYSTEM
# ============================================================================

"""
EXISTING STABILITY MEASURES:

1. In FixedAdaptiveTokenOperationDetector:
   - reliability: Based on attention strength, training stage, historical consistency
   - stability_score: Frequency of appearance in recent vs expected analyses
   - recent_consistency: % of recent analysis opportunities where circuit appeared

2. In EnhancedCircuitRegistry:
   - consistency_score: Cross-method consistency
   - stability_score: Recency + consistency over time span
   - strength_history: List of (epoch, attribution) tuples

3. In CircuitMetadata:
   - stability: TRANSIENT/EMERGING/STABLE/PERSISTENT/DECLINING/DEFUNCT
   - detection_epochs: All epochs where circuit was seen
   - strength_history: Historical strength evolution
"""


# ============================================================================
# 🔍 ENHANCED STABILITY ANALYSIS
# ============================================================================

class CircuitStabilityAnalyzer:
    """Enhanced circuit stability analysis and lifecycle management"""

    def __init__(self, registry, stability_window=50, decline_threshold=0.3):
        self.registry = registry
        self.stability_window = stability_window  # Epochs to look back for stability
        self.decline_threshold = decline_threshold  # Threshold for considering decline

    def analyze_circuit_stability(self, circuit_id, current_epoch):
        """
        Comprehensive stability analysis for a specific circuit

        Args:
            circuit_id: ID of circuit to analyze
            current_epoch: Current training epoch

        Returns:
            dict: Comprehensive stability metrics
        """

        # Get circuit and its metadata
        circuit = self.registry.get_circuit(circuit_id)
        if not circuit or circuit_id not in self.registry.circuit_metadata:
            return {"status": "not_found"}

        metadata = self.registry.circuit_metadata[circuit_id]

        # Basic metrics
        first_detected = metadata.first_detected
        last_seen = metadata.last_seen
        detection_epochs = metadata.detection_epochs
        strength_history = metadata.strength_history

        # Temporal metrics
        circuit_age = current_epoch - first_detected
        epochs_since_last_seen = current_epoch - last_seen
        total_detections = len(detection_epochs)

        # Recent activity analysis
        recent_epochs = [e for e in detection_epochs if current_epoch - e <= self.stability_window]
        recent_detections = len(recent_epochs)
        recent_activity_rate = recent_detections / min(self.stability_window, circuit_age + 1)

        # Strength evolution analysis
        strength_metrics = self._analyze_strength_evolution(strength_history, current_epoch)

        # Detection frequency analysis
        frequency_metrics = self._analyze_detection_frequency(detection_epochs, current_epoch)

        # Overall stability classification
        stability_class = self._classify_overall_stability(
            circuit_age, epochs_since_last_seen, recent_activity_rate,
            strength_metrics, frequency_metrics
        )

        return {
            "circuit_id": circuit_id,
            "current_epoch": current_epoch,

            # Temporal metrics
            "age": circuit_age,
            "epochs_since_last_seen": epochs_since_last_seen,
            "total_detections": total_detections,
            "recent_detections": recent_detections,
            "recent_activity_rate": recent_activity_rate,

            # Strength evolution
            "strength_metrics": strength_metrics,

            # Detection frequency
            "frequency_metrics": frequency_metrics,

            # Overall assessment
            "stability_class": stability_class,
            "current_strength": circuit.attribution,
            "registry_stability": metadata.stability.value,

            # Lifecycle recommendations
            "lifecycle_recommendation": self._get_lifecycle_recommendation(
                stability_class, strength_metrics, epochs_since_last_seen
            )
        }

    def _analyze_strength_evolution(self, strength_history, current_epoch):
        """Analyze how circuit strength has evolved over time"""

        if len(strength_history) < 2:
            return {
                "trend": "insufficient_data",
                "current_strength": strength_history[-1][1] if strength_history else 0.0,
                "peak_strength": strength_history[-1][1] if strength_history else 0.0,
                "strength_decline": 0.0,
                "recent_trend": "unknown"
            }

        # Sort by epoch
        sorted_history = sorted(strength_history, key=lambda x: x[0])

        # Current and peak strength
        current_strength = sorted_history[-1][1]
        peak_strength = max(s[1] for s in sorted_history)

        # Recent trend (last 25% of history)
        recent_count = max(2, len(sorted_history) // 4)
        recent_history = sorted_history[-recent_count:]

        if len(recent_history) >= 2:
            recent_trend_slope = (recent_history[-1][1] - recent_history[0][1]) / len(recent_history)

            if recent_trend_slope > 0.05:
                recent_trend = "strengthening"
            elif recent_trend_slope < -0.05:
                recent_trend = "weakening"
            else:
                recent_trend = "stable"
        else:
            recent_trend = "unknown"

        # Overall trend
        if len(sorted_history) >= 3:
            overall_slope = (sorted_history[-1][1] - sorted_history[0][1]) / len(sorted_history)

            if overall_slope > 0.02:
                overall_trend = "strengthening"
            elif overall_slope < -0.02:
                overall_trend = "weakening"
            else:
                overall_trend = "stable"
        else:
            overall_trend = "unknown"

        # Calculate decline from peak
        strength_decline = (peak_strength - current_strength) / peak_strength if peak_strength > 0 else 0

        return {
            "trend": overall_trend,
            "recent_trend": recent_trend,
            "current_strength": current_strength,
            "peak_strength": peak_strength,
            "strength_decline": strength_decline,
            "trend_slope": overall_slope if len(sorted_history) >= 3 else 0,
            "recent_slope": recent_trend_slope if len(recent_history) >= 2 else 0,
            "measurements": len(sorted_history)
        }

    def _analyze_detection_frequency(self, detection_epochs, current_epoch):
        """Analyze frequency of circuit detection over time"""

        if len(detection_epochs) < 2:
            return {
                "frequency_trend": "insufficient_data",
                "recent_frequency": 0.0,
                "overall_frequency": 0.0
            }

        sorted_epochs = sorted(detection_epochs)

        # Overall frequency (detections per epoch over lifetime)
        circuit_lifespan = current_epoch - sorted_epochs[0] + 1
        overall_frequency = len(detection_epochs) / circuit_lifespan

        # Recent frequency (last stability_window epochs)
        recent_start = max(sorted_epochs[0], current_epoch - self.stability_window)
        recent_window_size = current_epoch - recent_start + 1
        recent_detections = sum(1 for e in detection_epochs if e >= recent_start)
        recent_frequency = recent_detections / recent_window_size

        # Frequency trend
        if recent_frequency > overall_frequency * 1.2:
            frequency_trend = "increasing"
        elif recent_frequency < overall_frequency * 0.8:
            frequency_trend = "decreasing"
        else:
            frequency_trend = "stable"

        return {
            "frequency_trend": frequency_trend,
            "recent_frequency": recent_frequency,
            "overall_frequency": overall_frequency,
            "frequency_ratio": recent_frequency / max(overall_frequency, 0.001)
        }

    def _classify_overall_stability(self, age, epochs_since_last_seen, recent_activity_rate,
                                    strength_metrics, frequency_metrics):
        """Classify overall circuit stability"""

        # Age-based classification
        if age < 10:
            age_class = "very_young"
        elif age < 50:
            age_class = "young"
        elif age < 200:
            age_class = "mature"
        else:
            age_class = "old"

        # Activity-based classification
        if epochs_since_last_seen > 100:
            activity_class = "dormant"
        elif epochs_since_last_seen > 50:
            activity_class = "declining"
        elif recent_activity_rate > 0.1:
            activity_class = "active"
        else:
            activity_class = "sparse"

        # Strength-based classification
        strength_decline = strength_metrics.get("strength_decline", 0)
        recent_trend = strength_metrics.get("recent_trend", "unknown")

        if strength_decline > 0.5:
            strength_class = "severely_declined"
        elif strength_decline > 0.3:
            strength_class = "moderately_declined"
        elif recent_trend == "strengthening":
            strength_class = "strengthening"
        elif recent_trend == "weakening":
            strength_class = "weakening"
        else:
            strength_class = "stable"

        return {
            "age_class": age_class,
            "activity_class": activity_class,
            "strength_class": strength_class,
            "composite_stability": self._get_composite_stability(age_class, activity_class, strength_class)
        }

    def _get_composite_stability(self, age_class, activity_class, strength_class):
        """Get composite stability assessment"""

        # Dormant circuits
        if activity_class == "dormant":
            return "defunct"

        # Severely declined circuits
        if strength_class == "severely_declined":
            return "failing"

        # Young circuits
        if age_class in ["very_young", "young"]:
            if activity_class == "active" and strength_class in ["stable", "strengthening"]:
                return "emerging_stable"
            else:
                return "emerging_unstable"

        # Mature/old circuits
        if activity_class == "active":
            if strength_class in ["stable", "strengthening"]:
                return "highly_stable"
            elif strength_class == "weakening":
                return "stable_but_declining"
            else:
                return "stable"

        # Default cases
        if activity_class == "declining":
            return "declining"
        else:
            return "unstable"

    def _get_lifecycle_recommendation(self, stability_class, strength_metrics, epochs_since_last_seen):
        """Get lifecycle management recommendation"""

        composite_stability = stability_class["composite_stability"]
        strength_decline = strength_metrics.get("strength_decline", 0)

        if composite_stability == "defunct":
            return "remove"
        elif composite_stability == "failing":
            return "remove"
        elif composite_stability == "declining" and epochs_since_last_seen > 50:
            return "remove"
        elif composite_stability == "stable_but_declining" and strength_decline > 0.4:
            return "downgrade"
        elif composite_stability in ["emerging_unstable", "unstable"]:
            return "monitor"
        elif composite_stability in ["highly_stable", "stable", "emerging_stable"]:
            return "keep"
        else:
            return "monitor"


# ============================================================================
# 🔄 ENHANCED REGISTRY LIFECYCLE MANAGEMENT
# ============================================================================

class EnhancedRegistryLifecycleManager:
    """Manages circuit lifecycle including removal of declining circuits"""

    def __init__(self, registry, stability_analyzer=None):
        self.registry = registry
        self.stability_analyzer = stability_analyzer or CircuitStabilityAnalyzer(registry)

        # Lifecycle tracking
        self.removal_history = []
        self.downgrade_history = []

    def perform_lifecycle_maintenance(self, current_epoch, logger=None):
        """
        Perform comprehensive lifecycle maintenance

        Args:
            current_epoch: Current training epoch
            logger: Optional logger for maintenance actions

        Returns:
            dict: Summary of maintenance actions taken
        """

        if logger:
            logger.info(f"🔄 Starting circuit lifecycle maintenance @ epoch {current_epoch}")

        # Analyze all circuits
        maintenance_actions = {
            "analyzed": 0,
            "kept": 0,
            "removed": 0,
            "downgraded": 0,
            "monitored": 0,
            "removal_details": [],
            "downgrade_details": []
        }

        circuit_ids = list(self.registry.circuits.keys())

        for circuit_id in circuit_ids:
            maintenance_actions["analyzed"] += 1

            # Analyze stability
            stability = self.stability_analyzer.analyze_circuit_stability(circuit_id, current_epoch)
            recommendation = stability["lifecycle_recommendation"]

            if recommendation == "remove":
                self._remove_circuit(circuit_id, stability, current_epoch, logger)
                maintenance_actions["removed"] += 1
                maintenance_actions["removal_details"].append({
                    "circuit_id": circuit_id,
                    "reason": stability["stability_class"]["composite_stability"],
                    "age": stability["age"],
                    "epochs_since_last_seen": stability["epochs_since_last_seen"]
                })

            elif recommendation == "downgrade":
                self._downgrade_circuit(circuit_id, stability, current_epoch, logger)
                maintenance_actions["downgraded"] += 1
                maintenance_actions["downgrade_details"].append({
                    "circuit_id": circuit_id,
                    "old_confidence": self.registry.circuit_metadata[circuit_id].detection_confidence,
                    "strength_decline": stability["strength_metrics"]["strength_decline"]
                })

            elif recommendation == "monitor":
                maintenance_actions["monitored"] += 1

            else:  # keep
                maintenance_actions["kept"] += 1

        # Log summary
        if logger:
            logger.info(f"  📊 Maintenance Summary:")
            logger.info(f"    Analyzed: {maintenance_actions['analyzed']} circuits")
            logger.info(f"    Kept: {maintenance_actions['kept']} circuits")
            logger.info(f"    Removed: {maintenance_actions['removed']} circuits")
            logger.info(f"    Downgraded: {maintenance_actions['downgraded']} circuits")
            logger.info(f"    Monitored: {maintenance_actions['monitored']} circuits")

        return maintenance_actions

    def _remove_circuit(self, circuit_id, stability_analysis, current_epoch, logger=None):
        """Remove a declining circuit from the registry"""

        if circuit_id in self.registry.circuits:
            circuit = self.registry.circuits[circuit_id]

            # Record removal
            self.removal_history.append({
                "circuit_id": circuit_id,
                "epoch_removed": current_epoch,
                "reason": stability_analysis["stability_class"]["composite_stability"],
                "final_strength": circuit.attribution,
                "age": stability_analysis["age"],
                "stability_analysis": stability_analysis
            })

            # Remove from registry
            del self.registry.circuits[circuit_id]

            # Remove metadata
            if circuit_id in self.registry.circuit_metadata:
                del self.registry.circuit_metadata[circuit_id]

            # Remove relationships
            if circuit_id in self.registry.relationship_graph:
                del self.registry.relationship_graph[circuit_id]

            # Remove references in other relationships
            for other_circuit, relationships in self.registry.relationship_graph.items():
                if circuit_id in relationships:
                    del relationships[circuit_id]

            if logger:
                logger.debug(
                    f"    ❌ Removed circuit {circuit_id}: {stability_analysis['stability_class']['composite_stability']}")

    def _downgrade_circuit(self, circuit_id, stability_analysis, current_epoch, logger=None):
        """Downgrade a declining circuit's confidence"""

        if circuit_id in self.registry.circuit_metadata:
            metadata = self.registry.circuit_metadata[circuit_id]

            # Calculate new confidence based on decline
            strength_decline = stability_analysis["strength_metrics"]["strength_decline"]
            old_confidence = metadata.detection_confidence
            new_confidence = old_confidence * (1.0 - strength_decline * 0.5)  # Reduce by 50% of decline

            # Update metadata
            metadata.detection_confidence = max(0.1, new_confidence)  # Minimum 0.1

            # Record downgrade
            self.downgrade_history.append({
                "circuit_id": circuit_id,
                "epoch_downgraded": current_epoch,
                "old_confidence": old_confidence,
                "new_confidence": metadata.detection_confidence,
                "strength_decline": strength_decline
            })

            if logger:
                logger.debug(
                    f"    📉 Downgraded circuit {circuit_id}: confidence {old_confidence:.3f} → {metadata.detection_confidence:.3f}")

    def get_lifecycle_summary(self):
        """Get summary of lifecycle management actions"""

        return {
            "total_removals": len(self.removal_history),
            "total_downgrades": len(self.downgrade_history),
            "recent_removals": [r for r in self.removal_history if r["epoch_removed"] >=
                                max(0, max([r["epoch_removed"] for r in self.removal_history], default=0) - 100)],
            "recent_downgrades": [d for d in self.downgrade_history if d["epoch_downgraded"] >=
                                  max(0, max([d["epoch_downgraded"] for d in self.downgrade_history], default=0) - 100)],
        }
