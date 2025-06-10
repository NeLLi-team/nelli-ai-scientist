#!/usr/bin/env python3
"""
BBMap MCP Server - Simple Pixi Test

A clean, simple test that validates the BBMap MCP server works with pixi.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_pixi_command(cmd, description):
    """Run a command through pixi and return success/failure"""
    print(f"🧪 {description}")

    result = subprocess.run(
        ["pixi", "run"] + cmd,
        capture_output=True,
        text=True,
        cwd="/pscratch/sd/j/jvillada/nelli-ai-scientist"
    )

    if result.returncode == 0:
        print(f"✅ {description}: SUCCESS")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    else:
        print(f"❌ {description}: FAILED")
        if result.stderr.strip():
            print(f"   Error: {result.stderr.strip()}")
        return False

def main():
    """Run basic validation tests"""
    print("🧬 BBMap MCP Server - Simple Pixi Validation")
    print("=" * 55)

    tests = [
        (["python", "--version"], "Python Version Check"),
        (["python", "-c", "import sys; print('Python executable:', sys.executable)"], "Python Executable"),
        (["python", "-c", "import fastmcp; print('FastMCP imported successfully')"], "FastMCP Import"),
        (["python", "-c", "import os; print('Current directory:', os.getcwd())"], "Environment Check"),
    ]

    results = []
    for cmd, description in tests:
        results.append(run_pixi_command(cmd, description))

    # Test our BBMap tools
    print(f"\n🔧 Testing BBMap Tools")
    bbmap_test_cmd = [
        "python", "-c",
        "import sys; sys.path.insert(0, 'mcps/bbmap/src'); from bbmap_tools import BBMapToolkit; toolkit = BBMapToolkit(); print(f'BBMapToolkit: {toolkit.shifter_image}')"
    ]
    results.append(run_pixi_command(bbmap_test_cmd, "BBMap Tools Import"))

    # Test shifter availability
    print(f"\n🐳 Testing Shifter")
    shifter_result = subprocess.run(["which", "shifter"], capture_output=True, text=True)
    if shifter_result.returncode == 0:
        print(f"✅ Shifter Available: {shifter_result.stdout.strip()}")
        results.append(True)
    else:
        print("❌ Shifter Not Found")
        results.append(False)

    # Summary
    print("\n" + "=" * 55)
    print("📊 Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if all(results):
        print("\n🎉 SUCCESS: Your BBMap MCP server is ready!")
        print("\n🚀 Next Steps:")
        print("   1. Create your genomics data files (contig.fasta, reads.fastq)")
        print("   2. Run: pixi run python mcps/bbmap/hands_on_tutorial.py")
        print("   3. Try: pixi run python mcps/bbmap/real_world_example.py")
    else:
        print("\n⚠️  Some components need attention. See errors above.")

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
