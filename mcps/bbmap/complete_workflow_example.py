#!/usr/bin/env python3
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
    print("\n📋 Workflow Steps:")
    print("   1. Quality assessment of reads")
    print("   2. Read filtering (optional)")
    print("   3. Read mapping to reference")
    print("   4. Coverage analysis")

    print("\n💡 Usage Instructions:")
    print("   To run with your actual data:")
    print("   1. Replace paths below with your contig FASTA and reads FASTQ")
    print("   2. Run: pixi run python complete_workflow_example.py")

    # Example paths (replace with your actual data)
    reference_path = "/path/to/your/contig.fasta"
    reads_path = "/path/to/your/reads.fastq"

    print(f"\n📁 Example Data Paths:")
    print(f"   Reference: {reference_path}")
    print(f"   Reads: {reads_path}")

    # Show example commands that would be executed
    print("\n🔧 BBMap Commands That Would Be Executed:")

    commands = [
        f"shifter --image bryce911/bbtools:latest readlength.sh in={reads_path} out=quality_stats.txt hist=quality_hist.txt",
        f"shifter --image bryce911/bbtools:latest bbduk.sh in={reads_path} out=filtered_reads.fastq minlen=50 maq=20",
        f"shifter --image bryce911/bbtools:latest bbmap.sh ref={reference_path} in=filtered_reads.fastq out=alignment.sam",
        f"shifter --image bryce911/bbtools:latest pileup.sh in=alignment.sam ref={reference_path} out=coverage.txt stats=coverage_stats.txt"
    ]

    for i, cmd in enumerate(commands, 1):
        print(f"   {i}. {cmd}")

    print("\n🚀 Ready to Process Your Data!")
    print("   Modify the file paths above and run this script with your actual genomics data.")

if __name__ == "__main__":
    asyncio.run(run_bbmap_workflow())
