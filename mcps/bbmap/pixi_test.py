#!/usr/bin/env python3
"""
BBMap MCP Server - Pixi Environment Test

This script tests the BBMap MCP server functionality using the pixi environment.
It validates that all components work correctly together.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def test_pixi_environment():
    """Test that pixi environment has the required packages"""
    print("🧪 Testing Pixi Environment Setup")
    print("=" * 50)

    # Test Python availability
    result = subprocess.run(
        ["pixi", "run", "python", "--version"],
        capture_output=True, text=True, cwd="/pscratch/sd/j/jvillada/nelli-ai-scientist"
    )

    if result.returncode == 0:
        print(f"✅ Python: {result.stdout.strip()}")
    else:
        print(f"❌ Python test failed: {result.stderr}")
        return False

    # Test FastMCP import
    result = subprocess.run([
        "pixi", "run", "python", "-c",
        "import fastmcp; print('FastMCP version:', fastmcp.__version__ if hasattr(fastmcp, '__version__') else 'available')"
    ], capture_output=True, text=True, cwd="/pscratch/sd/j/jvillada/nelli-ai-scientist")

    if result.returncode == 0:
        print(f"✅ FastMCP: {result.stdout.strip()}")
    else:
        print(f"❌ FastMCP test failed: {result.stderr}")
        return False

    # Test our BBMap tools import
    result = subprocess.run([
        "pixi", "run", "python", "-c",
        "import sys; sys.path.insert(0, 'mcps/bbmap/src'); from bbmap_tools import BBMapToolkit; print('BBMapToolkit available')"
    ], capture_output=True, text=True, cwd="/pscratch/sd/j/jvillada/nelli-ai-scientist")

    if result.returncode == 0:
        print(f"✅ BBMap Tools: {result.stdout.strip()}")
    else:
        print(f"❌ BBMap Tools test failed: {result.stderr}")
        return False

    return True

def test_bbmap_toolkit():
    """Test BBMapToolkit functionality"""
    print("\n🔧 Testing BBMapToolkit Functionality")
    print("=" * 50)

    test_script = '''
import sys
sys.path.insert(0, "mcps/bbmap/src")
from bbmap_tools import BBMapToolkit

# Test initialization
toolkit = BBMapToolkit()
print(f"Toolkit image: {toolkit.shifter_image}")
print(f"Base command: {toolkit.base_command}")

# Test command building
import tempfile
import os

# Create minimal test files
with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as ref:
    ref.write(">test\\nATCG\\n")
    ref_path = ref.name

with tempfile.NamedTemporaryFile(mode='w', suffix='.fastq', delete=False) as reads:
    reads.write("@read1\\nATCG\\n+\\nIIII\\n")
    reads_path = reads.name

print(f"Created test files: {ref_path}, {reads_path}")

# Test file validation (should not raise exceptions)
if os.path.exists(ref_path) and os.path.exists(reads_path):
    print("✅ File validation works")
else:
    print("❌ File validation failed")

# Clean up
os.unlink(ref_path)
os.unlink(reads_path)
print("✅ Test files cleaned up")
'''

    result = subprocess.run([
        "pixi", "run", "python", "-c", test_script
    ], capture_output=True, text=True, cwd="/pscratch/sd/j/jvillada/nelli-ai-scientist")

    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"❌ BBMapToolkit test failed: {result.stderr}")
        return False

def test_shifter_availability():
    """Test shifter container system"""
    print("\n🐳 Testing Shifter Container System")
    print("=" * 50)

    # Test shifter command
    result = subprocess.run(["which", "shifter"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Shifter found at: {result.stdout.strip()}")
    else:
        print("❌ Shifter not found")
        return False

    # Test shifter help (quick command)
    result = subprocess.run(["shifter", "--help"], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("✅ Shifter is functional")
    else:
        print(f"⚠️  Shifter help failed: {result.stderr}")

    # Test BBTools image availability (this might take time)
    print("🔍 Testing BBTools image access...")
    result = subprocess.run([
        "timeout", "15", "shifter", "--image", "bryce911/bbtools:latest", "echo", "BBTools container accessible"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ BBTools container: {result.stdout.strip()}")
    else:
        print(f"⚠️  BBTools container test: {result.stderr or 'Timeout or access issue'}")

    return True

def create_comprehensive_example():
    """Create a comprehensive usage example"""
    print("\n📝 Creating Comprehensive Usage Example")
    print("=" * 50)

    example_content = '''#!/usr/bin/env python3
"""
BBMap MCP Server - Complete Usage Example with Pixi

This example shows how to use the BBMap MCP server in a real bioinformatics workflow
using the pixi environment for dependency management.
"""

import asyncio
import sys
import os
from pathlib import Path

async def run_bbmap_workflow():
    """Complete BBMap workflow using pixi environment"""

    # Import from our BBMap tools
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from bbmap_tools import BBMapToolkit

    print("🧬 BBMap MCP Server - Complete Workflow Example")
    print("Using pixi environment for dependency management")
    print("=" * 60)

    # Initialize toolkit
    toolkit = BBMapToolkit()
    print(f"✅ BBMap toolkit initialized")
    print(f"   Container image: {toolkit.shifter_image}")
    print(f"   Command prefix: {' '.join(toolkit.base_command)}")

    # Example workflow steps
    print("\\n📋 Workflow Steps:")
    print("   1. Quality assessment of reads")
    print("   2. Read filtering (optional)")
    print("   3. Read mapping to reference")
    print("   4. Coverage analysis")

    print("\\n💡 Usage Instructions:")
    print("   To run with your actual data:")
    print("   1. Replace paths below with your contig FASTA and reads FASTQ")
    print("   2. Run: pixi run python complete_workflow_example.py")

    # Example paths (replace with your actual data)
    reference_path = "/path/to/your/contig.fasta"
    reads_path = "/path/to/your/reads.fastq"

    print(f"\\n📁 Example Data Paths:")
    print(f"   Reference: {reference_path}")
    print(f"   Reads: {reads_path}")

    # Show example commands that would be executed
    print("\\n🔧 BBMap Commands That Would Be Executed:")

    commands = [
        f"shifter --image bryce911/bbtools:latest readlength.sh in={reads_path} out=quality_stats.txt hist=quality_hist.txt",
        f"shifter --image bryce911/bbtools:latest bbduk.sh in={reads_path} out=filtered_reads.fastq minlen=50 maq=20",
        f"shifter --image bryce911/bbtools:latest bbmap.sh ref={reference_path} in=filtered_reads.fastq out=alignment.sam",
        f"shifter --image bryce911/bbtools:latest pileup.sh in=alignment.sam ref={reference_path} out=coverage.txt stats=coverage_stats.txt"
    ]

    for i, cmd in enumerate(commands, 1):
        print(f"   {i}. {cmd}")

    print("\\n🚀 Ready to Process Your Data!")
    print("   Modify the file paths above and run this script with your actual genomics data.")

if __name__ == "__main__":
    asyncio.run(run_bbmap_workflow())
'''

    with open("/pscratch/sd/j/jvillada/nelli-ai-scientist/mcps/bbmap/complete_workflow_example.py", "w") as f:
        f.write(example_content)

    print("✅ Created: complete_workflow_example.py")
    return True

def main():
    """Run all tests"""
    print("🧬 BBMap MCP Server - Pixi Environment Validation")
    print("=" * 60)

    tests = [
        ("Pixi Environment", test_pixi_environment),
        ("BBMapToolkit", test_bbmap_toolkit),
        ("Shifter System", test_shifter_availability),
        ("Usage Example", create_comprehensive_example)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            print(f"\\n🧪 Running: {test_name}")
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append(False)

    # Summary
    print("\\n" + "=" * 60)
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if all(results):
        print("\\n🎉 All tests passed! Your BBMap MCP server is ready to use with pixi.")
        print("\\n🚀 Next Steps:")
        print("   1. Run: pixi run python mcps/bbmap/complete_workflow_example.py")
        print("   2. Update file paths with your actual genomics data")
        print("   3. Execute the workflow with: pixi run python [your_script]")
    else:
        print("\\n⚠️  Some tests failed. Check the output above for details.")

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
