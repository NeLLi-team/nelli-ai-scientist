#!/usr/bin/env python3
"""
Quick BBMap Test - Direct Approach

Test your BBMap MCP server with your microbiome data using a direct approach.
"""

import asyncio
import sys
import os
import subprocess
from pathlib import Path

# Add our BBMap tools to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def quick_bbmap_test():
    """Quick test of BBMap with your real data"""

    print("🧬 Quick BBMap MCP Test with Your Microbiome Data")
    print("=" * 55)

    # Your data files
    contigs_file = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
    reads_file = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads.fq.gz"
    output_sam = "quick_test_alignment.sam"

    print(f"📁 Input Files:")
    print(f"   Contigs: {contigs_file}")
    print(f"   Reads: {reads_file}")
    print(f"   Output: {output_sam}")

    # Validate files exist
    if not os.path.exists(contigs_file):
        print(f"❌ Contigs file not found!")
        return False

    if not os.path.exists(reads_file):
        print(f"❌ Reads file not found!")
        return False

    # Get file sizes
    contigs_size = os.path.getsize(contigs_file) / (1024**2)  # MB
    reads_size = os.path.getsize(reads_file) / (1024**2)  # MB
    print(f"✅ Files found - Contigs: {contigs_size:.1f}MB, Reads: {reads_size:.1f}MB")

    # Import and test BBMap toolkit
    try:
        from bbmap_tools import BBMapToolkit
        toolkit = BBMapToolkit()
        print(f"✅ BBMapToolkit imported successfully")
        print(f"   Image: {toolkit.shifter_image}")
        print(f"   Command: {' '.join(toolkit.base_command)}")
    except Exception as e:
        print(f"❌ BBMapToolkit import failed: {e}")
        return False

    # Test the mapping function
    try:
        print(f"\n🎯 Starting BBMap read mapping...")
        print(f"   This will map {reads_size:.0f}MB of reads to {contigs_size:.0f}MB of contigs")
        print(f"   Expected runtime: 10-30 minutes for microbiome data")

        # Clean up any previous files
        for cleanup_file in [output_sam, "mapping_stats.txt", "scaffold_stats.txt"]:
            if os.path.exists(cleanup_file):
                os.remove(cleanup_file)

        result = await toolkit.map_reads(
            reference_path=contigs_file,
            reads_path=reads_file,
            output_sam=output_sam,
            additional_params="minid=0.85 maxindel=5 ambig=random threads=auto"
        )

        print(f"\n🎉 BBMap completed successfully!")
        print(f"   Status: {result['status']}")
        print(f"   Output SAM: {result['output_sam']}")

        # Validate SAM file
        if os.path.exists(output_sam):
            sam_size = os.path.getsize(output_sam) / (1024**2)  # MB
            print(f"✅ SAM file created: {sam_size:.1f} MB")

            # Count lines
            with open(output_sam, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"   Lines in SAM file: {line_count:,}")

            # Show mapping stats if available
            if 'mapping_stats' in result:
                print(f"📊 Mapping Statistics:")
                for key, value in result['mapping_stats'].items():
                    print(f"   {key}: {value}")

            print(f"\n🎉 SUCCESS: Your BBMap MCP server works perfectly!")
            print(f"✅ Generated SAM file from your microbiome data")
            print(f"✅ File size: {sam_size:.1f} MB with {line_count:,} lines")

            return True
        else:
            print(f"❌ SAM file was not created")
            return False

    except Exception as e:
        print(f"❌ BBMap execution failed: {e}")
        return False

async def main():
    """Run the quick test"""
    success = await quick_bbmap_test()

    if success:
        print(f"\n🚀 Next Steps:")
        print(f"   1. Your SAM file is ready for downstream analysis")
        print(f"   2. Integrate this into your master agent workflow")
        print(f"   3. Scale to process multiple samples")
        return 0
    else:
        print(f"\n❌ Test failed - check error messages above")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
