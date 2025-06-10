#!/usr/bin/env python3
"""
Simple BBMap Test - Run a basic alignment
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add our BBMap tools to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bbmap_tools import BBMapToolkit

async def main():
    print("🧬 Simple BBMap Test")
    print("=" * 30)

    # Data files
    contigs = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
    reads = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads.fq.gz"
    output = "simple_test.sam"

    # Check files
    print(f"Contigs: {os.path.exists(contigs)} ({os.path.getsize(contigs)/(1024**2):.1f} MB)")
    print(f"Reads: {os.path.exists(reads)} ({os.path.getsize(reads)/(1024**2):.1f} MB)")

    # Initialize toolkit
    toolkit = BBMapToolkit()
    print("Toolkit initialized")

    # Run BBMap with optimized parameters
    print("\nRunning BBMap alignment...")
    start_time = time.time()

    try:
        result = await toolkit.map_reads(
            reference_path=contigs,
            reads_path=reads,
            output_sam=output,
            additional_params="minid=0.85 maxindel=100 fast=t threads=auto overwrite=t"
        )

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed/60:.1f} minutes")
        print(f"Status: {result['status']}")

        if os.path.exists(output):
            size = os.path.getsize(output) / (1024**2)
            print(f"SAM file created: {size:.1f} MB")

            # Quick peek at SAM file
            with open(output, 'r') as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"Line {i+1}: {line.strip()[:80]}...")
                    else:
                        break
        else:
            print("SAM file not created")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Error after {elapsed/60:.1f} minutes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
