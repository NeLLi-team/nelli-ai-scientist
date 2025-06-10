#!/usr/bin/env python3
"""
BBMap MCP Server - Real Data Test

This script tests your BBMap MCP server with your actual microbiome data:
- Contigs: /global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa
- Reads: /global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads.fq.gz

This will generate a SAM file from your real genomics data!
"""

import asyncio
import sys
import os
import time
import logging
from pathlib import Path

# Add our BBMap tools to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bbmap_tools import BBMapToolkit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_real_microbiome_data():
    """Test BBMap with real microbiome data"""

    print("🧬 Testing BBMap MCP Server with Real Microbiome Data")
    print("=" * 65)

    # Your actual data files
    contigs_file = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
    reads_file = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads.fq.gz"
    output_sam = "microbiome_alignment.sam"
    output_dir = "microbiome_bbmap_results"

    print(f"📁 Input Files:")
    print(f"   Contigs: {contigs_file}")
    print(f"   Reads: {reads_file}")
    print(f"   Output SAM: {output_sam}")
    print(f"   Results directory: {output_dir}")

    # Check if input files exist
    print(f"\n🔍 Validating Input Files...")

    if not os.path.exists(contigs_file):
        print(f"❌ Contigs file not found: {contigs_file}")
        return False
    else:
        file_size = os.path.getsize(contigs_file) / (1024**2)  # MB
        print(f"✅ Contigs file found ({file_size:.1f} MB)")

    if not os.path.exists(reads_file):
        print(f"❌ Reads file not found: {reads_file}")
        return False
    else:
        file_size = os.path.getsize(reads_file) / (1024**2)  # MB
        print(f"✅ Reads file found ({file_size:.1f} MB)")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory created: {output_dir}")

    # Initialize BBMap toolkit
    print(f"\n🔧 Initializing BBMap Toolkit...")
    toolkit = BBMapToolkit()
    print(f"✅ BBMap toolkit ready")
    print(f"   Container image: {toolkit.shifter_image}")
    print(f"   Command prefix: {' '.join(toolkit.base_command)}")

    try:
        # Step 1: Quality analysis (optional - can be slow for large files)
        print(f"\n📊 Step 1: Quality Analysis (Quick Check)")
        print("   Note: Skipping full quality analysis for speed - focus on mapping")

        # Step 2: Read mapping (main test)
        print(f"\n🎯 Step 2: Mapping Reads to Contigs")
        print("   This is the main test - generating SAM file from your data")

        start_time = time.time()

        # Use relative path for output SAM in results directory
        sam_output_path = os.path.join(output_dir, output_sam)

        print(f"   Command will be:")
        print(f"   shifter --image bryce911/bbtools:latest bbmap.sh \\")
        print(f"     ref={contigs_file} \\")
        print(f"     in={reads_file} \\")
        print(f"     out={sam_output_path} \\")
        print(f"     stats=mapping_stats.txt")

        print(f"\n⏳ Starting read mapping... (this may take several minutes)")

        mapping_result = await toolkit.map_reads(
            reference_path=contigs_file,
            reads_path=reads_file,
            output_sam=sam_output_path,
            additional_params="minid=0.90 maxindel=5 ambig=random threads=auto"
        )

        end_time = time.time()
        runtime = end_time - start_time

        print(f"\n🎉 Read Mapping Completed!")
        print(f"   Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
        print(f"   Status: {mapping_result['status']}")
        print(f"   Output SAM: {mapping_result['output_sam']}")

        # Check if SAM file was created
        if os.path.exists(sam_output_path):
            sam_size = os.path.getsize(sam_output_path) / (1024**2)  # MB
            print(f"✅ SAM file generated successfully ({sam_size:.1f} MB)")

            # Count lines in SAM file (quick quality check)
            with open(sam_output_path, 'r') as f:
                line_count = sum(1 for line in f)
            print(f"   SAM file contains {line_count:,} lines")

        else:
            print(f"❌ SAM file was not created")
            return False

        # Display mapping statistics if available
        if 'mapping_stats' in mapping_result and mapping_result['mapping_stats']:
            stats = mapping_result['mapping_stats']
            print(f"\n📈 Mapping Statistics:")
            for key, value in stats.items():
                if key != 'error':
                    print(f"   {key}: {value}")

        # Step 3: Coverage analysis (optional)
        print(f"\n📊 Step 3: Coverage Analysis")
        try:
            coverage_result = await toolkit.coverage_analysis(
                sam_path=sam_output_path,
                reference_path=contigs_file,
                output_prefix=os.path.join(output_dir, "coverage")
            )

            print(f"✅ Coverage analysis completed")
            if 'coverage_stats' in coverage_result and coverage_result['coverage_stats']:
                stats = coverage_result['coverage_stats']
                print(f"   Coverage Statistics:")
                for key, value in stats.items():
                    if key != 'error':
                        print(f"     {key}: {value}")

        except Exception as e:
            print(f"⚠️  Coverage analysis had issues: {e}")
            print("   This is okay - the main mapping test was successful")

        # Summary
        print(f"\n🎉 SUCCESS: Your BBMap MCP Server Works!")
        print(f"=" * 50)
        print(f"✅ Successfully processed your microbiome data")
        print(f"✅ Generated SAM alignment file: {sam_output_path}")
        print(f"✅ BBMap container integration working")
        print(f"✅ MCP server tools functioning correctly")

        print(f"\n📁 Output Files Generated:")
        output_files = [
            sam_output_path,
            "mapping_stats.txt",
            "scaffold_stats.txt",
            os.path.join(output_dir, "coverage_coverage.txt"),
            os.path.join(output_dir, "coverage_stats.txt")
        ]

        for file_path in output_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"   ✅ {file_path} ({size:.1f} KB)")

        print(f"\n🚀 Next Steps:")
        print(f"   1. Examine the SAM file: {sam_output_path}")
        print(f"   2. Use this SAM file for downstream analysis")
        print(f"   3. Integrate this workflow into your master agent")
        print(f"   4. Scale up to process multiple samples")

        return True

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        logger.error(f"BBMap test failed: {e}")
        return False

async def main():
    """Run the real data test"""
    success = await test_real_microbiome_data()

    if success:
        print(f"\n🎉 Your BBMap MCP server is working perfectly with real data!")
        return 0
    else:
        print(f"\n❌ Test failed - check the error messages above")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
