#!/usr/bin/env python3
"""
Direct test for BBMap map_reads functionality through MCP
"""

import asyncio
import sys
import os

# Add the agent path for importing
sys.path.append('/pscratch/sd/j/jvillada/nelli-ai-scientist/agents/template/src')
sys.path.append('/pscratch/sd/j/jvillada/nelli-ai-scientist/mcps/bbmap/src')

async def test_map_reads_direct():
    """Test map_reads tool directly"""

    # Import the BBMap toolkit directly
    from bbmap_tools import BBMapToolkit

    print("Creating BBMapToolkit...")
    toolkit = BBMapToolkit()

    reference_path = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/contigs.fa"
    reads_path = "/global/cfs/cdirs/nelli/juan/hackathon_microbiome_data/reads_subsampled-100k.fq.gz"
    output_sam = "/tmp/mapped_reads.sam"

    print(f"Reference file exists: {os.path.exists(reference_path)}")
    print(f"Reads file exists: {os.path.exists(reads_path)}")

    try:
        print("Starting BBMap mapping...")
        result = await toolkit.map_reads(
            reference_path=reference_path,
            reads_path=reads_path,
            output_sam=output_sam
        )

        print("✅ BBMap mapping completed successfully!")
        print(f"Result status: {result.get('status', 'unknown')}")
        print(f"Output SAM: {result.get('output_sam', 'not specified')}")
        print(f"Mapping stats: {result.get('mapping_stats', {})}")

        # Check if output SAM was created
        if os.path.exists(output_sam):
            file_size = os.path.getsize(output_sam)
            print(f"✅ Output SAM file created: {output_sam} ({file_size} bytes)")

            # Show first few lines of SAM file
            with open(output_sam, 'r') as f:
                lines = f.readlines()[:10]
                print("First 10 lines of SAM file:")
                for i, line in enumerate(lines, 1):
                    print(f"  {i:2}: {line.rstrip()}")
        else:
            print("❌ Output SAM file was not created")

    except Exception as e:
        print(f"❌ BBMap test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧬 Starting BBMap Direct Test...")
    try:
        asyncio.run(test_map_reads_direct())
        print("🎉 Test completed!")
    except Exception as e:
        print(f"💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
