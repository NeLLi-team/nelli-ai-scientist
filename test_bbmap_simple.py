#!/usr/bin/env python3
"""
Simple test for BBMap functionality
"""

import asyncio
import sys
import os
sys.path.append('/pscratch/sd/j/jvillada/nelli-ai-scientist/mcps/bbmap/src')

from bbmap_tools import BBMapToolkit

async def test_bbmap():
    """Test BBMap with the provided files"""

    print("Creating BBMapToolkit...")
    toolkit = BBMapToolkit()
    print("BBMapToolkit created successfully!")

    reference_path = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
    reads_path = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads_subsampled-100k.fq.gz"
    output_sam = "/tmp/test_mapping.sam"

    # Check if files exist
    print(f"Reference file exists: {os.path.exists(reference_path)}")
    print(f"Reads file exists: {os.path.exists(reads_path)}")

    try:
        print("Starting BBMap test...")
        result = await toolkit.map_reads(
            reference_path=reference_path,
            reads_path=reads_path,
            output_sam=output_sam
        )

        print("BBMap completed successfully!")
        print(f"Result: {result}")

        # Check if output SAM was created
        print(f"Output SAM exists: {os.path.exists(output_sam)}")
        if os.path.exists(output_sam):
            # Show first few lines
            with open(output_sam, 'r') as f:
                lines = f.readlines()[:10]
                print("First 10 lines of SAM file:")
                for line in lines:
                    print(f"  {line.rstrip()}")

    except Exception as e:
        print(f"BBMap test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting test...")
    try:
        asyncio.run(test_bbmap())
        print("Test completed!")
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
