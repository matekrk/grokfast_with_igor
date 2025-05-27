import json
from pathlib import Path

import numpy as np

from analysis.utils.utils import CircuitJSONEncoder, get_current_callable_info


class CircuitLogger:
    """Enhanced circuit logger with robust JSON serialization"""

    def __init__(self, registry, save_dir):
        self.registry = registry
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.circuit_history = []
        self.registry_snapshots = {}

    def prepare_data_for_json(self, data, float_precision=4):
        """Prepare data for JSON serialization with reduced float precision"""
        if isinstance(data, dict):
            return {k: self.prepare_data_for_json(v, float_precision) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.prepare_data_for_json(item, float_precision) for item in data]
        elif isinstance(data, tuple):
            return [self.prepare_data_for_json(item, float_precision) for item in data]
        elif isinstance(data, float):
            return round(data, float_precision)
        elif isinstance(data, np.floating):
            return round(float(data), float_precision)
        else:
            return data

    def log_circuit_event(self, event_type, circuit, epoch, details=None):
        """Log circuit events with JSON-safe data preparation"""

        # Prepare circuit metadata for JSON
        clean_metadata = self.prepare_data_for_json(circuit.metadata.copy())

        # Prepare details for JSON
        clean_details = self.prepare_data_for_json(details.copy() if details else {})

        log_entry = {
            'timestamp': epoch,
            'event': event_type,
            'circuit_id': circuit.id,
            'circuit_type': circuit.type.value,
            'attribution': round(circuit.attribution, 4),
            'source': clean_details.get('source', 'unknown'),
            'metadata': clean_metadata,
            'details': clean_details,
            'registry_size': len(self.registry.circuits)
        }

        # Additional cleaning pass
        log_entry = self.prepare_data_for_json(log_entry)
        self.circuit_history.append(log_entry)

    def snapshot_registry_state(self, epoch):
        """Take JSON-safe registry snapshot"""
        snapshot = {
            'epoch': epoch,
            'total_circuits': len(self.registry.circuits),
            'circuits_by_type': {},
            'circuits_by_source': {},
            'top_circuits': [],
            'circuit_details': []
        }

        # Analyze circuits
        for circuit in self.registry.circuits.values():
            circuit_type = circuit.type.value
            source = self.registry.sources.get(circuit.id, 'unknown')

            # Count by type
            snapshot['circuits_by_type'][circuit_type] = snapshot['circuits_by_type'].get(circuit_type, 0) + 1

            # Count by source
            snapshot['circuits_by_source'][source] = snapshot['circuits_by_source'].get(source, 0) + 1

            # Store detailed circuit info (JSON-safe)
            circuit_info = {
                'id': circuit.id,
                'type': circuit_type,
                'attribution': round(circuit.attribution, 4),
                'discovered_at': circuit.discovered_at,
                'source': source,
                'element_count': len(circuit.elements),
                'connection_count': len(circuit.connections),
                'metadata': self.prepare_data_for_json(circuit.metadata.copy())
            }
            snapshot['circuit_details'].append(circuit_info)

        # Top circuits by attribution
        top_circuits = sorted(self.registry.circuits.values(),
                              key=lambda c: c.attribution, reverse=True)[:10]
        snapshot['top_circuits'] = [
            {
                'id': c.id,
                'type': c.type.value,
                'attribution': round(c.attribution, 4),
                'discovered_at': c.discovered_at
            }
            for c in top_circuits
        ]

        # Clean the entire snapshot
        snapshot = self.prepare_data_for_json(snapshot)
        self.registry_snapshots[epoch] = snapshot
        return snapshot

    def save_logs(self, epoch=None, float_precision=4):
        """Save circuit logs with robust JSON handling"""
        if not self.save_dir:
            return

        timestamp = epoch if epoch is not None else "final"

        try:
            # Prepare data for JSON
            prepared_data = {
                'circuit_events': self.prepare_data_for_json(self.circuit_history, float_precision),
                'registry_snapshots': self.prepare_data_for_json(self.registry_snapshots, float_precision),
                'metadata': {
                    'total_events': len(self.circuit_history),
                    'epochs_tracked': list(self.registry_snapshots.keys()),
                    'generated_at': timestamp,
                    'float_precision': float_precision
                }
            }

            # Save with custom encoder
            json_path = self.save_dir / f"circuit_logs_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(prepared_data, f, cls=CircuitJSONEncoder, indent=2)

            print(f"\t{get_current_callable_info()}:\t✅circuit logs saved to {json_path}")

        except Exception as e:
            print(f"\t{get_current_callable_info()}:\t⚠️error saving JSON logs: {e}")
            # Fallback: save with minimal data
            self._save_minimal_fallback(timestamp)

    def _save_minimal_fallback(self, timestamp):
        """Fallback saving with only essential data"""
        try:
            minimal_data = {
                'summary': {
                    'total_events': len(self.circuit_history),
                    'total_snapshots': len(self.registry_snapshots),
                    'timestamp': timestamp
                },
                'event_timeline': [
                    {
                        'epoch': event.get('timestamp', 0),
                        'event': event.get('event', 'unknown'),
                        'circuit_id': event.get('circuit_id', 'unknown'),
                        'attribution': round(event.get('attribution', 0), 4)
                    }
                    for event in self.circuit_history[-100:]  # Last 100 events
                ],
                'latest_snapshot': list(self.registry_snapshots.values())[-1] if self.registry_snapshots else {}
            }

            fallback_path = self.save_dir / f"circuit_logs_minimal_{timestamp}.json"
            with open(fallback_path, 'w') as f:
                json.dump(minimal_data, f, indent=2)

            print(f"📝 Minimal circuit logs saved to {fallback_path}")

        except Exception as e:
            print(f"❌failed to save even minimal logs: {e}")

    # ✅ KEEP these methods from the previous version if you want them:
    def get_circuit_timeline(self, circuit_id):
        """Get evolution timeline for specific circuit"""
        return [entry for entry in self.circuit_history if entry['circuit_id'] == circuit_id]

    def should_log_detailed(self, epoch, total_epochs):
        """Determine if detailed logging should occur"""
        if epoch < total_epochs * 0.1:
            return epoch % 5 == 0
        elif epoch < total_epochs * 0.8:
            return epoch % 20 == 0
        else:
            return epoch % 10 == 0

    def should_take_snapshot(self, epoch, total_epochs):
        """Determine if registry snapshot should be taken"""
        milestones = [0.1, 0.2, 0.5, 0.8, 0.9, 1.0]
        milestone_epochs = [int(total_epochs * m) for m in milestones]
        return epoch in milestone_epochs or epoch % 100 == 0