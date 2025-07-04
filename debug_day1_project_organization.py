# ============================================================================
# PROJECT ORGANIZATION AND TESTING SETUP
# ============================================================================

"""
CURRENT PROJECT STRUCTURE (based on your uploaded file):

your_project_root/
├── analysis/                    # Your main analysis module
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── circuit_schema.py
│   │   ├── circuit_registry.py
│   │   └── ... (other files)
│   ├── analyzers/
│   ├── utils/
│   ├── models/
│   └── tests/                   # ← Your existing tests directory
│       ├── __init__.py
│       └── ... (existing test files)
├── other_project_files/
└── ... (your other project files)

QUESTION: Where should you run scripts and put tests?
"""

# ============================================================================
# 1. WHERE TO RUN SCRIPTS FROM
# ============================================================================

print("📁 WHERE TO RUN SCRIPTS FROM")
print("=" * 40)
print("""
ALWAYS run scripts from your PROJECT ROOT directory, not from inside analysis/

Example:
your_project_root/          ← RUN SCRIPTS FROM HERE
├── analysis/
├── debug_day1.py          ← Put debug scripts here temporarily  
├── test_day1_temp.py      ← Put temporary tests here
└── run_week1_tests.py     ← Put week tests here

COMMAND LINE:
cd /path/to/your/project/root
python debug_day1.py
python test_day1_temp.py
""")

# ============================================================================
# 2. WHERE TO PUT TEMPORARY DEBUG/TEST SCRIPTS
# ============================================================================

print("\n📁 TEMPORARY SCRIPTS PLACEMENT")
print("=" * 40)
print("""
For debugging and temporary testing during Week 1 setup:

your_project_root/
├── debug_day1.py           ← Debug script (temporary)
├── test_day1_temp.py       ← Day 1 test (temporary)  
├── test_day2_temp.py       ← Day 2 test (temporary)
├── test_week1_full.py      ← Full Week 1 test (temporary)
└── analysis/
    ├── core/
    └── tests/              ← Permanent test location

These temporary files can be deleted after Week 1 is working.
""")

# ============================================================================
# 3. WHERE TO PUT PERMANENT DAY TESTS
# ============================================================================

print("\n📁 PERMANENT TEST PLACEMENT")
print("=" * 40)
print("""
After each day is working, move tests to the permanent location:

analysis/tests/
├── __init__.py
├── test_day1_infrastructure.py    ← Day 1 permanent test
├── test_day2_infrastructure.py    ← Day 2 permanent test  
├── test_day3_infrastructure.py    ← Day 3 permanent test
├── test_week1_integration.py      ← Full Week 1 integration test
└── ... (your existing tests)

These become part of your project's test suite.
""")


# ============================================================================
# 4. SPECIFIC FILES TO CREATE NOW
# ============================================================================

def create_debug_script():
    """Create the debug script in project root"""

    debug_content = '''# debug_day1.py - PUT THIS IN PROJECT ROOT
# Run from project root with: python debug_day1.py

print("🔍 Debugging Day 1 Import Issues")
print("Current working directory:", __import__('os').getcwd())
print("=" * 50)

# Test 1: Check if basic analysis imports work
try:
    print("1. Testing basic analysis import...")
    import analysis
    print("   ✅ Basic analysis import successful")
except Exception as e:
    print(f"   ❌ Basic analysis import failed: {e}")

# Test 2: Check core module
try:
    print("2. Testing analysis.core import...")
    import analysis.core
    print("   ✅ analysis.core import successful")
except Exception as e:
    print(f"   ❌ analysis.core import failed: {e}")

# Test 3: Check circuit_schema specifically
try:
    print("3. Testing circuit_schema import...")
    from analysis.core import circuit_schema
    print("   ✅ circuit_schema import successful")

    # Check what's available
    available_classes = [name for name in dir(circuit_schema) if not name.startswith('_')]
    print(f"   Available classes: {available_classes}")

except Exception as e:
    print(f"   ❌ circuit_schema import failed: {e}")

# Test 4: Check for new Day 1 classes
try:
    print("4. Testing new Day 1 classes...")
    from analysis.core.circuit_schema import EmergencePhase, CircuitStability, RelationshipType, CircuitMetadata
    print("   ✅ All new Day 1 classes imported successfully")
except Exception as e:
    print(f"   ❌ New Day 1 classes import failed: {e}")

# Test 5: Check circuit_registry
try:
    print("5. Testing circuit_registry...")
    from analysis.core.circuit_registry import CircuitRegistry
    print("   ✅ CircuitRegistry imported successfully")

    # Check if EnhancedCircuitRegistry exists (shouldn't for Day 1)
    try:
        from analysis.core.circuit_registry import EnhancedCircuitRegistry
        print("   ⚠️ EnhancedCircuitRegistry found (this might be the problem!)")
    except ImportError:
        print("   ✅ EnhancedCircuitRegistry not found (correct for Day 1)")

except Exception as e:
    print(f"   ❌ circuit_registry import failed: {e}")

print("\\n🔍 Debug complete. Look for ❌ errors above.")
'''

    return debug_content


def create_temp_day1_test():
    """Create temporary Day 1 test in project root"""

    test_content = '''# test_day1_temp.py - PUT THIS IN PROJECT ROOT  
# Run from project root with: python test_day1_temp.py

def test_day1_schema_only():
    """Test ONLY Day 1 changes - schema enhancements"""

    print("🧪 Day 1 Temporary Test")
    print("=" * 30)

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

    # Test 4: Existing functionality
    try:
        from analysis.core.circuit_schema import Circuit, CircuitType, Element, ElementType
        from analysis.core.circuit_registry import CircuitRegistry

        element = Element(id="test", type=ElementType.TOKEN)
        circuit = Circuit(id="test_circuit", type=CircuitType.TOKEN, elements=[element])

        registry = CircuitRegistry()
        registry.register_circuit(circuit, "test")

        print("✅ 4. Existing functionality works")
        print(f"   Circuit: {circuit.id}")
        print(f"   Registry circuits: {len(registry.circuits)}")

    except Exception as e:
        print(f"❌ 4. Existing functionality failed: {e}")
        return False

    print("\\n🎉 Day 1 Test PASSED!")
    return True

if __name__ == "__main__":
    success = test_day1_schema_only()
    if success:
        print("\\n✅ Ready to proceed to Day 2")
    else:
        print("\\n❌ Fix Day 1 issues before proceeding")
'''

    return test_content


# ============================================================================
# 5. PERMANENT TEST STRUCTURE
# ============================================================================

def create_permanent_test_structure():
    """Show the permanent test structure for analysis/tests/"""

    permanent_test = '''# analysis/tests/test_day1_infrastructure.py
# This goes in your analysis/tests/ directory after Day 1 is working

from analysis.core.circuit_schema import (
    EmergencePhase, CircuitStability, RelationshipType, CircuitMetadata,
    Circuit, CircuitType, Element, ElementType
)
from analysis.core.circuit_registry import CircuitRegistry

def test_day1_schema_enhancements():
    """Permanent test for Day 1 schema enhancements"""

    # Test new enums
    phase = EmergencePhase.EARLY
    stability = CircuitStability.TRANSIENT
    relationship = RelationshipType.PREREQUISITE

    assert phase.value == "early"
    assert stability.value == "transient" 
    assert relationship.value == "prerequisite"

    # Test metadata
    metadata = CircuitMetadata(
        first_detected=10,
        detection_method="test_method",
        detection_confidence=0.8
    )

    assert metadata.detection_method == "test_method"
    assert metadata.detection_confidence == 0.8

    # Test serialization
    metadata_dict = metadata.to_dict()
    metadata_restored = CircuitMetadata.from_dict(metadata_dict)
    assert metadata_restored.detection_confidence == 0.8

    # Test backward compatibility
    element = Element(id="test", type=ElementType.TOKEN)
    circuit = Circuit(id="test_circuit", type=CircuitType.TOKEN, elements=[element])
    registry = CircuitRegistry()
    registry.register_circuit(circuit, "test")

    assert len(registry.circuits) == 1
    assert circuit.id in registry.circuits

    print("✅ Day 1 infrastructure test passed")

if __name__ == "__main__":
    test_day1_schema_enhancements()
'''

    return permanent_test


# ============================================================================
# 6. WORKFLOW SUMMARY
# ============================================================================

def show_workflow():
    """Show the complete workflow for Week 1 testing"""

    workflow = """
🔄 WEEK 1 TESTING WORKFLOW
========================

PHASE 1: DEBUG AND FIX (temporary files in project root)
---------------------------------------------------------
1. Create debug_day1.py in project root
2. Run: python debug_day1.py
3. Fix any import issues found
4. Create test_day1_temp.py in project root  
5. Run: python test_day1_temp.py
6. Repeat until Day 1 passes

PHASE 2: IMPLEMENT NEXT DAYS (temporary files in project root)
---------------------------------------------------------------
7. Create test_day2_temp.py in project root
8. Implement Day 2 features
9. Run: python test_day2_temp.py
10. Repeat for Days 3-5

PHASE 3: INTEGRATION (temporary file in project root)
------------------------------------------------------
11. Create test_week1_full.py in project root
12. Run full Week 1 integration test
13. Fix any integration issues

PHASE 4: PERMANENT PLACEMENT (move to analysis/tests/)
-------------------------------------------------------
14. Move working tests to analysis/tests/test_dayX_infrastructure.py
15. Delete temporary files from project root
16. Add tests to your permanent test suite

FILE LOCATIONS SUMMARY:
- Temporary scripts: project_root/*.py
- Permanent tests: analysis/tests/test_*_infrastructure.py
- Run location: Always from project root
"""

    print(workflow)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("📁 PROJECT ORGANIZATION GUIDE")
    print("=" * 50)

    # Show workflow
    show_workflow()

    print("\n📝 IMMEDIATE NEXT STEPS:")
    print("1. Create debug_day1.py in your project root")
    print("2. Create test_day1_temp.py in your project root")
    print("3. Run both from project root directory")
    print("4. Fix any issues found")
    print("5. Move to permanent location when working")

    print("\n📋 FILES TO CREATE NOW:")
    print("- PROJECT_ROOT/debug_day1.py (temporary)")
    print("- PROJECT_ROOT/test_day1_temp.py (temporary)")
    print("- Later: analysis/tests/test_day1_infrastructure.py (permanent)")

    # Offer to create the files
    print("\n💾 SCRIPT CONTENTS:")
    print("(Copy these to create your files)")

    print("\n--- debug_day1.py ---")
    print(create_debug_script())

    print("\n--- test_day1_temp.py ---")
    print(create_temp_day1_test())

    print("\n--- analysis/tests/test_day1_infrastructure.py (for later) ---")
    print(create_permanent_test_structure())