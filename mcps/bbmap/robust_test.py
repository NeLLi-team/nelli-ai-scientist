#!/usr/bin/env python3
"""
BBMap MCP Server - Robust Real Data Test

This script provides better error handling and debugging for testing
your BBMap MCP server with real microbiome data.
"""

import asyncio
import sys
import os
import time
import logging
import subprocess
from pathlib import Path

# Add our BBMap tools to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bbmap_tools import BBMapToolkit

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_shifter_access():
    """Test basic shifter access"""
    print("🐳 Testing Shifter Access...")

    try:
        # Test shifter command
        result = subprocess.run(["shifter", "--help"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Shifter is available")
        else:
            print(f"❌ Shifter help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Shifter test failed: {e}")
        return False

    try:
        # Test BBTools container access
        result = subprocess.run([
            "shifter", "--image", "bryce911/bbtools:latest", "echo", "Container accessible"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print(f"✅ BBTools container accessible: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ BBTools container access failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ BBTools container test failed: {e}")
        return False

def validate_input_files(contigs_file, reads_file):
    """Validate input files exist and get basic info"""
    print("🔍 Validating Input Files...")

    if not os.path.exists(contigs_file):
        print(f"❌ Contigs file not found: {contigs_file}")
        return False

    if not os.path.exists(reads_file):
        print(f"❌ Reads file not found: {reads_file}")
        return False

    # Get file sizes
    contigs_size = os.path.getsize(contigs_file) / (1024**2)  # MB
    reads_size = os.path.getsize(reads_file) / (1024**2)  # MB

    print(f"✅ Contigs file: {contigs_size:.1f} MB")
    print(f"✅ Reads file: {reads_size:.1f} MB")

    # Quick peek at contigs file
    try:
        with open(contigs_file, 'r') as f:
            first_lines = [f.readline().strip() for _ in range(3)]

        if first_lines[0].startswith('>'):
            print(f"✅ Contigs file appears to be valid FASTA")
            print(f"   First sequence: {first_lines[0]}")
        else:
            print(f"⚠️  Contigs file format unclear: {first_lines[0]}")
    except Exception as e:
        print(f"⚠️  Could not read contigs file: {e}")

    return True

async def test_simple_bbmap_command():
    """Test a simple BBMap command first"""
    print("\n🧪 Testing Simple BBMap Command...")

    # Your actual data files
    contigs_file = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
    reads_file = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads.fq.gz"
    output_dir = "bbmap_test_output"
    simple_sam = os.path.join(output_dir, "test_alignment.sam")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Initialize toolkit
        toolkit = BBMapToolkit()
        print(f"✅ BBMapToolkit initialized")

        # Test the command construction first
        command = [
            "bbmap.sh",
            f"ref={contigs_file}",
            f"in={reads_file}",
            f"out={simple_sam}",
            "stats=test_mapping_stats.txt",
            "scafstats=test_scaffold_stats.txt",
            "maxindel=5",
            "ambig=random",
            "minid=0.85"  # Slightly lower threshold for microbiome data
        ]

        full_command = toolkit.base_command + command
        print(f"\n🔧 Command to execute:")
        print(f"   {' '.join(full_command)}")

        # Execute with timeout
        print(f"\n⏳ Executing BBMap command...")
        print(f"   This may take 10-20 minutes for large microbiome datasets")

        start_time = time.time()

        stdout, stderr, returncode = toolkit._run_command(command)

        end_time = time.time()
        runtime = end_time - start_time

        print(f"\n⏱️  Execution completed in {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
        print(f"📊 Return code: {returncode}")

        if returncode == 0:
            print("✅ BBMap command executed successfully!")

            # Check output files
            if os.path.exists(simple_sam):
                sam_size = os.path.getsize(simple_sam) / (1024**2)  # MB
                print(f"✅ SAM file created: {sam_size:.1f} MB")

                # Quick SAM file validation
                with open(simple_sam, 'r') as f:
                    lines = [f.readline() for _ in range(10)]

                header_count = sum(1 for line in lines if line.startswith('@'))
                alignment_count = len([line for line in lines if line.strip() and not line.startswith('@')])

                print(f"📄 SAM file preview:")
                print(f"   Header lines in first 10: {header_count}")
                print(f"   Alignment lines in first 10: {alignment_count}")

                return True
            else:
                print(f"❌ SAM file was not created")
                return False
        else:
            print(f"❌ BBMap command failed!")
            print(f"📝 STDOUT: {stdout}")
            print(f"📝 STDERR: {stderr}")
            return False

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        logger.exception("BBMap test exception")
        return False

async def main():
    """Run comprehensive test"""
    print("🧬 BBMap MCP Server - Robust Real Data Test")
    print("=" * 55)

    # Your data files
    contigs_file = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
    reads_file = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads.fq.gz"

    print(f"📁 Testing with:")
    print(f"   Contigs: {contigs_file}")
    print(f"   Reads: {reads_file}")

    # Run tests step by step
    tests = [
        ("Shifter Access", test_shifter_access),
        ("Input File Validation", lambda: validate_input_files(contigs_file, reads_file)),
        ("BBMap Execution", test_simple_bbmap_command)
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                return 1

        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            logger.exception(f"{test_name} failed")
            return 1

    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"✅ Your BBMap MCP server is working with real microbiome data!")
    print(f"✅ SAM file successfully generated from your contigs and reads!")

    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
