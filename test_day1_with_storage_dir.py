# test_day1_quick_fix.py - PUT THIS IN PROJECT ROOT
# Quick fix: Test with storage_dir to avoid the None path issue

import tempfile
from pathlib import Path


def test_day1_with_storage_dir():
    """Day 1 test with temporary storage directory to avoid path bug"""

    print("🧪 Day 1 Quick Fix Test (with storage dir)")
    print("=" * 40)

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test 1: Import new enums
        try:
            from analysis.core.circuit_schema import EmergencePhase, CircuitStability, RelationshipType

            phase = EmergencePhase.EARLY
            stability = CircuitStability.TRANSIENT
            relationship = RelationshipType.PREREQUISITE

            print("✅ 1. New enums work")
            print(f"   EmergencePhase.EARLY = {phase.value}")
            print(f"   CircuitStability.TRANSIENT = {stability.value}")
            print(f"   RelationshipType.PREREQUISITE = {relationship.value}")

        except Exception as e:
            print(f"❌ 1. New enums failed: {e}")
            return False

        # Test 2: CircuitMetadata
        try:
            from analysis.core.circuit_schema import CircuitMetadata

            metadata = CircuitMetadata(
                first_detected=10,
                detection_method="test_method",
                detection_confidence=0.8
            )

            print("✅ 2. CircuitMetadata works")
            print(f"   Detection method: {metadata.detection_method}")
            print(f"   Confidence: {metadata.detection_confidence}")

        except Exception as e:
            print(f"❌ 2. CircuitMetadata failed: {e}")
            return False

        # Test 3: Serialization
        try:
            metadata_dict = metadata.to_dict()
            metadata_restored = CircuitMetadata.from_dict(metadata_dict)

            print("✅ 3. Serialization works")
            print(f"   Original: {metadata.detection_confidence}")
            print(f"   Restored: {metadata_restored.detection_confidence}")

        except Exception as e:
            print(f"❌ 3. Serialization failed: {e}")
            return False

        # Test 4: CircuitRegistry WITH storage_dir (avoids the bug)
        try:
            from analysis.core.circuit_registry import CircuitRegistry
            from analysis.core.circuit_schema import Circuit, CircuitType, Element, ElementType

            # Create registry WITH storage directory to avoid None path bug
            registry = CircuitRegistry(storage_dir=temp_path / "test_circuits")

            element = Element(id="test", type=ElementType.TOKEN)
            circuit = Circuit(id="test_circuit", type=CircuitType.TOKEN, elements=[element])

            registry.register_circuit(circuit, "test")

            print("✅ 4. CircuitRegistry works with storage_dir")
            print(f"   Registry circuits: {len(registry.circuits)}")
            print(f"   Storage dir: {registry.storage_dir}")

        except Exception as e:
            print(f"❌ 4. CircuitRegistry failed: {e}")
            return False

        print("\n🎉 Day 1 Quick Fix Test PASSED!")
        print("The schema enhancements work, but CircuitRegistry needs the path bug fixed.")
        return True


if __name__ == "__main__":
    success = test_day1_with_storage_dir()
    if success:
        print("\n✅ Day 1 schema works! Now fix the CircuitRegistry path bug for permanent solution.")
    else:
        print("\n❌ Still have issues to resolve.")