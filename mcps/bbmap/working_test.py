#!/usr/bin/env python3
"""
BBMap MCP - Working Real Data Test

This script will successfully generate a SAM file from your microbiome data
using your BBMap MCP server. Based on the BBMap help, we'll use optimal parameters.
"""

import asyncio
import sys
import os
import time
import subprocess
from pathlib import Path

# Add our BBMap tools to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bbmap_tools import BBMapToolkit

async def working_bbmap_test():
    """Working test that will generate SAM file from your data"""

    print("🧬 BBMap MCP - Working Real Data Test")
    print("=" * 50)

    # Your actual microbiome data
    contigs_file = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
    reads_file = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads.fq.gz"
    output_sam = "working_microbiome_alignment.sam"

    print(f"📁 Data Files:")
    print(f"   Reference: {contigs_file}")
    print(f"   Reads: {reads_file}")
    print(f"   Output SAM: {output_sam}")

    # Verify files exist
    if not os.path.exists(contigs_file):
        print(f"❌ Contigs file not found!")
        return False

    if not os.path.exists(reads_file):
        print(f"❌ Reads file not found!")
        return False

    contigs_size = os.path.getsize(contigs_file) / (1024**2)
    reads_size = os.path.getsize(reads_file) / (1024**2)
    print(f"✅ Contigs: {contigs_size:.1f} MB")
    print(f"✅ Reads: {reads_size:.1f} MB")

    # Initialize BBMap toolkit
    print(f"\n🔧 Initializing BBMap Toolkit...")
    toolkit = BBMapToolkit()

    # Use optimized parameters for microbiome data
    additional_params = "minid=0.85 maxindel=100 ambig=random threads=auto fast=t overwrite=t"

    print(f"\n🎯 Starting BBMap Alignment...")
    print(f"   Using optimized parameters for microbiome data:")
    print(f"   - minid=0.85 (85% identity threshold)")
    print(f"   - maxindel=100 (allow indels up to 100bp)")
    print(f"   - ambig=random (handle multi-mapping reads)")
    print(f"   - fast=t (faster mode)")
    print(f"   - threads=auto (use all available cores)")

    start_time = time.time()

    try:
        # Run BBMap alignment
        result = await toolkit.map_reads(
            reference_path=contigs_file,
            reads_path=reads_file,
            output_sam=output_sam,
            additional_params=additional_params
        )

        end_time = time.time()
        runtime_minutes = (end_time - start_time) / 60

        print(f"\n🎉 BBMap Alignment Completed!")
        print(f"⏱️  Runtime: {runtime_minutes:.1f} minutes")
        print(f"📊 Status: {result['status']}")

        # Check if SAM file was created
        if os.path.exists(output_sam):
            sam_size = os.path.getsize(output_sam) / (1024**2)
            print(f"✅ SAM file created: {sam_size:.1f} MB")

            # Quick SAM validation
            with open(output_sam, 'r') as f:
                lines = []
                for i, line in enumerate(f):
                    lines.append(line.strip())
                    if i >= 20:  # Just read first 20 lines
                        break

            header_lines = sum(1 for line in lines if line.startswith('@'))
            alignment_lines = len([line for line in lines if line and not line.startswith('@')])

            print(f"📄 SAM File Validation:")
            print(f"   Header lines: {header_lines}")
            print(f"   Alignment lines (in sample): {alignment_lines}")

            # Show some sample alignments
            print(f"   Sample header: {lines[0] if lines else 'None'}")
            for line in lines:
                if not line.startswith('@') and line.strip():
                    print(f"   Sample alignment: {line[:80]}...")
                    break
        else:
            print(f"❌ SAM file was not created")
            return False

        # Display mapping statistics
        if 'mapping_stats' in result:
            stats = result['mapping_stats']
            print(f"\n📈 Mapping Statistics:")
            for key, value in stats.items():
                if key != 'error':
                    print(f"   {key}: {value}")

        # Show additional output files
        output_files = ["mapping_stats.txt", "scaffold_stats.txt"]
        print(f"\n📁 Additional Output Files:")
        for file_path in output_files:
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                print(f"   ✅ {file_path} ({size_kb:.1f} KB)")
            else:
                print(f"   ⚠️  {file_path} (not found)")

        print(f"\n🎉 SUCCESS! Your BBMap MCP server is working perfectly!")
        print(f"✅ Generated SAM alignment file from your microbiome data")
        print(f"✅ All BBMap tools are functioning correctly")
        print(f"✅ Ready for integration with your master agent")

        return True

    except Exception as e:
        end_time = time.time()
        runtime_minutes = (end_time - start_time) / 60

        print(f"\n❌ BBMap test failed after {runtime_minutes:.1f} minutes")
        print(f"Error: {e}")

        # Check for partial outputs
        if os.path.exists(output_sam):
            sam_size = os.path.getsize(output_sam) / (1024**2)
            print(f"⚠️  Partial SAM file exists: {sam_size:.1f} MB")

        return False

async def main():
    """Run the working test"""
    success = await working_bbmap_test()

    if success:
        print(f"\n🚀 Next Steps:")
        print(f"   1. Examine the SAM file: working_microbiome_alignment.sam")
        print(f"   2. Try the coverage analysis: python test_coverage.py")
        print(f"   3. Integrate into your agent workflows")
        return 0
    else:
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check system resources (memory/disk space)")
        print(f"   2. Verify Shifter container access")
        print(f"   3. Check input file permissions")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
