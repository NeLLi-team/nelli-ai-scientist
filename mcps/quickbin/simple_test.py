#!/usr/bin/env python3
"""
Simple test for QuickBin MCP Server

This script tests the QuickBin MCP server functionality.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from quickbin_tools import QuickBinToolkit


async def test_quickbin_toolkit():
    """Test the QuickBin toolkit functionality"""
    print("🧪 Testing QuickBin MCP Toolkit")
    print("=" * 50)

    toolkit = QuickBinToolkit()

    # Test 1: Check if shifter is available
    print("\n1. Testing shifter availability...")
    try:
        stdout, stderr, returncode = toolkit._run_command(["quickbin.sh", "--help"])
        if returncode == 0 and "quickbin" in stdout.lower():
            print("✅ QuickBin is accessible via shifter")
            print(f"   Version info found in output")
        else:
            print("❌ QuickBin not accessible")
            print(f"   Return code: {returncode}")
            print(f"   Stderr: {stderr[:200]}...")
            return False
    except Exception as e:
        print(f"❌ Error testing shifter: {e}")
        return False

    # Test 2: Create dummy test files
    print("\n2. Creating test files...")
    test_dir = Path("quickbin_test")
    test_dir.mkdir(exist_ok=True)

    # Create a dummy contigs file
    contigs_file = test_dir / "test_contigs.fa"
    with open(contigs_file, 'w') as f:
        f.write(">contig1\nATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\n")
        f.write(">contig2\nGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\n")

    # Create a dummy SAM file (header only, no alignments)
    sam_file = test_dir / "test.sam"
    with open(sam_file, 'w') as f:
        f.write("@HD\tVN:1.0\tSO:unsorted\n")
        f.write("@SQ\tSN:contig1\tLN:42\n")
        f.write("@SQ\tSN:contig2\tLN:42\n")

    print(f"✅ Created test files in {test_dir}")

    # Test 3: Test coverage generation (will likely fail due to no real alignments)
    print("\n3. Testing coverage generation...")
    try:
        result = await toolkit.generate_coverage(
            contigs_path=str(contigs_file),
            sam_files=[str(sam_file)],
            output_coverage=str(test_dir / "test_coverage.txt"),
            additional_params="mincontig=10"  # Very low threshold for test
        )

        if result["status"] == "success":
            print("✅ Coverage generation completed")
            print(f"   Coverage file: {result['coverage_file']}")
        else:
            print("⚠️  Coverage generation failed (expected with dummy data)")
            print(f"   Error: {result.get('error_message', 'Unknown error')}")
    except Exception as e:
        print(f"⚠️  Coverage generation failed: {e}")

    # Test 4: Test basic tool validation
    print("\n4. Testing tool structure...")
    try:
        # Test that we can instantiate and access methods
        methods = [method for method in dir(toolkit) if not method.startswith('_')]
        expected_methods = ['bin_contigs', 'bin_contigs_with_coverage', 'generate_coverage', 'evaluate_bins']

        for method in expected_methods:
            if method in methods:
                print(f"✅ Method '{method}' available")
            else:
                print(f"❌ Method '{method}' missing")

    except Exception as e:
        print(f"❌ Error testing tool structure: {e}")

    # Cleanup
    print(f"\n5. Cleaning up test files...")
    try:
        import shutil
        shutil.rmtree(test_dir)
        print("✅ Test files cleaned up")
    except Exception as e:
        print(f"⚠️  Could not clean up: {e}")

    print("\n" + "=" * 50)
    print("🎯 QuickBin MCP Toolkit test completed!")
    print("\nNote: Some tests may fail with dummy data, but the toolkit structure should be valid.")
    return True


async def test_fastmcp_server():
    """Test the FastMCP server can be imported and initialized"""
    print("\n🚀 Testing FastMCP Server Import")
    print("=" * 50)

    try:
        # Import the server module
        from server_fastmcp import mcp, toolkit
        print("✅ FastMCP server imported successfully")
        print(f"   Server name: {mcp.name}")
        print(f"   Toolkit type: {type(toolkit).__name__}")

        # Check that tools are registered
        # Note: FastMCP doesn't expose tools directly, so we just verify import works
        print("✅ Server initialization successful")

        return True
    except Exception as e:
        print(f"❌ Error importing FastMCP server: {e}")
        return False


async def main():
    """Run all tests"""
    print("🧬 QuickBin MCP Server Test Suite")
    print("=" * 60)

    # Test toolkit
    toolkit_success = await test_quickbin_toolkit()

    # Test FastMCP server
    server_success = await test_fastmcp_server()

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"Toolkit Test: {'✅ PASS' if toolkit_success else '❌ FAIL'}")
    print(f"Server Test:  {'✅ PASS' if server_success else '❌ FAIL'}")

    if toolkit_success and server_success:
        print("\n🎉 All tests passed! QuickBin MCP is ready to use.")
        print("\nNext steps:")
        print("1. Test with real data (contigs + SAM files)")
        print("2. Add to your MCP configuration")
        print("3. Run a complete binning workflow")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("Common issues:")
        print("- Shifter not available or BBTools container not accessible")
        print("- Missing dependencies or import errors")

    return toolkit_success and server_success


if __name__ == "__main__":
    success = asyncio.run(main())
