# ============================================================================
# FIX CIRCUIT REGISTRY PATH BUG
# ============================================================================

"""
PROBLEM: CircuitRegistry.__init__ fails when storage_dir is None because:
self.circuit_logger = CircuitLogger(self, save_dir=self.storage_dir / "circuit_logger")
tries to do: None / "circuit_logger" which fails

SOLUTION: Fix the CircuitRegistry.__init__ method to handle None storage_dir
"""


# ============================================================================
# FIX: Update your analysis/core/circuit_registry.py
# ============================================================================

def fix_circuit_registry_init():
    """
    The issue is in your analysis/core/circuit_registry.py file.

    FIND this line in CircuitRegistry.__init__:
    self.circuit_logger = circuit_logger if circuit_logger else CircuitLogger(self, save_dir=self.storage_dir / "circuit_logger")

    REPLACE with this corrected version:
    """

    corrected_init = '''
# In analysis/core/circuit_registry.py, in CircuitRegistry.__init__ method:

def __init__(self, storage_dir: Optional[Path] = None, circuit_logger=None):
    self.circuits: Dict[str, Circuit] = {}
    self.sources: Dict[str, str] = {}  # Circuit ID to source mapping
    self.related_circuits: Dict[str, Set[str]] = {}  # Circuit ID to related circuit IDs
    self.storage_dir = storage_dir
    if self.storage_dir:
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # FIX: Handle None storage_dir properly
    if circuit_logger:
        self.circuit_logger = circuit_logger
    elif self.storage_dir:
        # Only create CircuitLogger if we have a storage directory
        self.circuit_logger = CircuitLogger(self, save_dir=self.storage_dir / "circuit_logger")
    else:
        # Create a minimal circuit logger that doesn't save to disk
        self.circuit_logger = CircuitLogger(self, save_dir=None)
'''

    return corrected_init


# ============================================================================
# ALTERNATIVE FIX: Simpler patch for CircuitLogger
# ============================================================================

def fix_circuit_logger_none_handling():
    """
    Alternative: Fix CircuitLogger to handle None save_dir

    In analysis/core/circuit_logger.py, update CircuitLogger.__init__:
    """

    circuit_logger_fix = '''
# In analysis/core/circuit_logger.py, in CircuitLogger.__init__:

def __init__(self, registry, save_dir=None):
    self.registry = registry
    self.save_dir = Path(save_dir) if save_dir else None  # Handle None case

    if self.save_dir:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ... rest of the init method
'''

    return circuit_logger_fix


# ============================================================================
# QUICK TEMPORARY FIX: Test with storage_dir
# ============================================================================

def create_quick_fix_test():
    """
    Quick temporary fix: Test CircuitRegistry WITH a storage_dir
    """

    quick_fix_test = '''
# test_day1_quick_fix.py - PUT IN PROJECT ROOT
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

        except Exception as e:
            print(f"❌ 2. CircuitMetadata failed: {e}")
            return False

        # Test 3: Serialization
        try:
            metadata_dict = metadata.to_dict()
            metadata_restored = CircuitMetadata.from_dict(metadata_dict)

            print("✅ 3. Serialization works")

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

        except Exception as e:
            print(f"❌ 4. CircuitRegistry failed: {e}")
            return False

        print("\\n🎉 Day 1 Quick Fix Test PASSED!")
        print("The schema enhancements work, but CircuitRegistry needs the path bug fixed.")
        return True

if __name__ == "__main__":
    success = test_day1_with_storage_dir()
    if success:
        print("\\n✅ Day 1 schema works! Now fix the CircuitRegistry path bug.")
    else:
        print("\\n❌ Still have issues to resolve.")
'''

    return quick_fix_test


# ============================================================================
# PERMANENT FIX INSTRUCTIONS
# ============================================================================

def show_permanent_fix_instructions():
    """Show step-by-step instructions to permanently fix the bug"""

    instructions = '''
🔧 PERMANENT FIX INSTRUCTIONS
============================

STEP 1: Locate the bug
-----------------------
Open: analysis/core/circuit_registry.py
Find: CircuitRegistry.__init__ method
Look for this line:
    self.circuit_logger = circuit_logger if circuit_logger else CircuitLogger(self, save_dir=self.storage_dir / "circuit_logger")

STEP 2: Replace the problematic line
------------------------------------
REPLACE the problematic line with:

    # Fixed version that handles None storage_dir
    if circuit_logger:
        self.circuit_logger = circuit_logger
    elif self.storage_dir:
        self.circuit_logger = CircuitLogger(self, save_dir=self.storage_dir / "circuit_logger")
    else:
        # Handle case where storage_dir is None
        self.circuit_logger = CircuitLogger(self, save_dir=None)

STEP 3: Also fix CircuitLogger (if needed)
-------------------------------------------
Open: analysis/core/circuit_logger.py
In CircuitLogger.__init__, make sure it handles None save_dir:

    def __init__(self, registry, save_dir=None):
        self.registry = registry
        self.save_dir = Path(save_dir) if save_dir else None  # Handle None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # ... rest of init

STEP 4: Test the fix
--------------------
After making these changes, run:
    python test_day1_temp.py

It should now work without the path error.
'''

    return instructions


# ============================================================================
# DIAGNOSIS: Check what's actually in your CircuitRegistry
# ============================================================================

def create_circuit_registry_diagnosis():
    """Create a script to diagnose the exact issue in CircuitRegistry"""

    diagnosis_script = '''
# diagnose_circuit_registry.py - PUT IN PROJECT ROOT
# Check what's actually in your CircuitRegistry.__init__

def diagnose_circuit_registry():
    """Examine the CircuitRegistry class to find the exact issue"""

    print("🔍 Diagnosing CircuitRegistry Issue")
    print("=" * 40)

    try:
        import inspect
        from analysis.core.circuit_registry import CircuitRegistry

        # Get the source code of CircuitRegistry.__init__
        init_source = inspect.getsource(CircuitRegistry.__init__)

        print("📋 CircuitRegistry.__init__ source code:")
        print("-" * 40)
        print(init_source)
        print("-" * 40)

        # Try to identify the problematic line
        lines = init_source.split('\\n')
        for i, line in enumerate(lines):
            if 'circuit_logger' in line and '/' in line:
                print(f"🎯 PROBLEMATIC LINE {i+1}: {line.strip()}")

    except Exception as e:
        print(f"❌ Could not diagnose: {e}")

    # Try creating CircuitRegistry with different parameters
    print("\\n🧪 Testing CircuitRegistry creation:")

    # Test 1: No parameters (this should fail)
    try:
        print("1. Creating CircuitRegistry() with no parameters...")
        registry = CircuitRegistry()
        print("   ✅ Success (unexpected!)")
    except Exception as e:
        print(f"   ❌ Failed as expected: {e}")

    # Test 2: With storage_dir
    try:
        import tempfile
        from pathlib import Path

        print("2. Creating CircuitRegistry(storage_dir=temp_path)...")
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = CircuitRegistry(storage_dir=Path(temp_dir))
            print("   ✅ Success with storage_dir")
    except Exception as e:
        print(f"   ❌ Failed even with storage_dir: {e}")

if __name__ == "__main__":
    diagnose_circuit_registry()
'''

    return diagnosis_script


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🔧 FIXING CIRCUIT REGISTRY PATH BUG")
    print("=" * 50)

    print("The error occurs because CircuitRegistry tries to do:")
    print("None / 'circuit_logger' when storage_dir is None")
    print("")

    print("📋 SOLUTION OPTIONS:")
    print("1. QUICK FIX: Test with storage_dir (temporary)")
    print("2. PERMANENT FIX: Update CircuitRegistry.__init__ (recommended)")
    print("3. DIAGNOSIS: Check exact code causing the issue")

    print("\n💾 QUICK FIX TEST (option 1):")
    print("Create this test file to bypass the bug temporarily:")
    print(create_quick_fix_test())

    print("\n🔧 PERMANENT FIX (option 2):")
    print(show_permanent_fix_instructions())

    print("\n🔍 DIAGNOSIS SCRIPT (option 3):")
    print("Create this script to see the exact problematic code:")
    print(create_circuit_registry_diagnosis())

    print("\n📋 RECOMMENDED APPROACH:")
    print("1. First run the diagnosis script to see the exact issue")
    print("2. Apply the permanent fix to CircuitRegistry.__init__")
    print("3. Test with the original Day 1 test")
    print("4. Proceed to Day 2 once Day 1 passes")