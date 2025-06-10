#!/usr/bin/env python3
"""
BBMap MCP Server - Hands-On Tutorial

This script demonstrates how to use the BBMap MCP server with real bioinformatics data.
We'll walk through a complete workflow: quality assessment → filtering → mapping → coverage analysis.
"""

import asyncio
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bbmap_tools import BBMapToolkit

def create_sample_data():
    """Create sample bioinformatics data for the tutorial"""
    print("📁 Creating sample data files...")

    # Create sample reference genome (contig)
    reference_content = """>contig_1 Sample bacterial genome contig
ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCCAGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCAGCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAAATAAAACATGTCCTGCATGGCATTAGTTTGTTGGGGCAGTGCCCGGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAAATGTCGATCGCCATTATGGCCGGCGTATTAGAAGCGCGCGGTCACAACGTTACTGTTATCGATCCGGTCGAAAAACTGCTGGCAGTGGGGCATTACCTCGAATCTACCGTCGATATTGCTGAGTCCACCCGCCGTATTGCGGCAAGCCGCATTCCGGCTGATCACATGGTGCTGATGGCAGGTTTCACCGCCGGTAATGAAAAAGGCGAACTGGTGGTGCTTGGACGCAACGGTTCCGACTACTCTGCTGCGGTGCTGGCTGCCTGTTTACGCGCCGATTGTTGCGAGATTTGGACGGACGTTGACGGGGTCTATACCTGCGACCCGCGTCAGGTGCCCGATGCGAGGTTGTTGAAGTCGATGTCCTACCAGGAAGCGATGGAGCTTTCCTACTTCGGCGCTAAAGTTCTTCACCCCCGCACCATTACCCCCATCGCCCAGTTCCAGATCCCTTGCCTGATTAAAAATACCGGAAATCCTCAAGCACCAGGTACGCTCATTGGTGCCAGCCGTGATGAAGACGAATTACCGGTCAAGGGCATTTCCAATCTGAATAACATGGCAATGTTCAGCGTTTCTGGTCCGGGGATGAAAGGGATGGTCGGCATGGCGGCGCGCGTCTTTGCAGCGATGTCACGCGCCCGTATTTCCGTGGTGCTGATTACGCAATCATCTTCCGAATACAGCATCAGTTTCTGCGTTCCACAAAGCGACTGTGTGCGAGCTGAACGGGCAATGCAGGAAGAGTTCTACCTGGAACTGAAAGAAGGCTTACTGGAGCCGCTGGCAGTGACGGAACGGCTGGCCATTATCTCGGTGGTAGGTGATGGTATGCGCACCTTGCGTGGGATCTCGGCACCAATCTCATGACCAAAATCCCTTAACGTGAGTTACGCGCGCGTGTTCCGTGATATGGGCATTCACCATGGCAAGCTGGTGGCAATGGCGAAGAATCACGGAGTGGAACCCGACGTGGGTAGGAGGAGGCTGAATCCCTGTATCACCGTGGCAAATGGACAAAGGACGAGCCTTTTGACGAGACGACCGACGACTACGCCTCCCGCCACGAAGACTGA
"""

    # Create sample reads (FASTQ)
    reads_content = """@read_1 Simulated read from contig_1 position 1-75
ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAG
+
IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
@read_2 Simulated read from contig_1 position 50-125
ACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGA
+
IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
@read_3 Simulated read from contig_1 position 100-175
CAGGAAACACAGAAAAAAGCCCGCACCTGACAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACC
+
IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
@read_4 Low quality read (should be filtered)
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
+
################################################++++++++++++++++++++++++++++++
@read_5 Short read (should be filtered)
ATCGATCG
+
########
@read_6 Good read from middle of contig
ATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAA
+
IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
"""

    # Write files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(reference_content)
        reference_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fastq', delete=False) as f:
        f.write(reads_content)
        reads_path = f.name

    print(f"✅ Reference genome: {reference_path}")
    print(f"✅ Reads file: {reads_path}")

    return reference_path, reads_path

async def tutorial_workflow():
    """Complete BBMap workflow tutorial"""
    print("🧬 BBMap MCP Server - Hands-On Tutorial")
    print("=" * 60)

    # Initialize toolkit
    print("\n🔧 Step 1: Initialize BBMap Toolkit")
    toolkit = BBMapToolkit()
    print(f"✅ Using BBTools image: {toolkit.shifter_image}")
    print(f"✅ Command prefix: {' '.join(toolkit.base_command)}")

    # Create sample data
    print("\n📁 Step 2: Prepare Sample Data")
    reference_path, reads_path = create_sample_data()

    try:
        # Step 3: Quality Assessment
        print("\n📊 Step 3: Assess Read Quality")
        print("This step analyzes the quality statistics of your FASTQ file.")
        print("We'll look at read length distribution, quality scores, and base composition.")

        try:
            quality_result = await toolkit.quality_stats(
                fastq_path=reads_path,
                output_prefix="tutorial_quality"
            )

            print("✅ Quality analysis completed!")
            print(f"   Status: {quality_result['status']}")
            print(f"   Command used: {quality_result['command_used']}")
            if quality_result.get('quality_stats'):
                stats = quality_result['quality_stats']
                print(f"   Stats: {stats}")

        except Exception as e:
            print(f"⚠️  Quality analysis encountered an issue: {e}")
            print("   This is normal in tutorial mode - BBTools needs actual execution environment")

        # Step 4: Read Filtering
        print("\n🔍 Step 4: Filter Low-Quality Reads")
        print("This step removes reads that are too short or have poor quality scores.")
        print("We'll filter out reads shorter than 30bp and with average quality < 20.")

        filtered_reads_path = "tutorial_filtered_reads.fastq"

        try:
            filter_result = await toolkit.filter_reads(
                input_fastq=reads_path,
                output_fastq=filtered_reads_path,
                min_length=30,
                min_quality=20.0,
                additional_params="qtrim=rl trimq=20"
            )

            print("✅ Read filtering completed!")
            print(f"   Status: {filter_result['status']}")
            print(f"   Input file: {filter_result['input_file']}")
            print(f"   Output file: {filter_result['output_file']}")
            print(f"   Filter parameters: {filter_result['filter_params']}")

        except Exception as e:
            print(f"⚠️  Read filtering encountered an issue: {e}")
            print("   Using original reads for next step...")
            filtered_reads_path = reads_path

        # Step 5: Read Mapping
        print("\n🎯 Step 5: Map Reads to Reference Genome")
        print("This is the core step - aligning sequencing reads to the reference genome.")
        print("BBMap will find the best matching positions for each read.")

        output_sam = "tutorial_alignment.sam"

        try:
            mapping_result = await toolkit.map_reads(
                reference_path=reference_path,
                reads_path=filtered_reads_path,
                output_sam=output_sam,
                additional_params="minid=0.95 maxindel=3 ambig=random"
            )

            print("✅ Read mapping completed!")
            print(f"   Status: {mapping_result['status']}")
            print(f"   Output SAM: {mapping_result['output_sam']}")
            if mapping_result.get('mapping_stats'):
                stats = mapping_result['mapping_stats']
                print(f"   Mapping statistics: {stats}")

        except Exception as e:
            print(f"⚠️  Read mapping encountered an issue: {e}")
            print("   Creating mock SAM file for next step...")

            # Create a simple SAM file for the tutorial
            with open(output_sam, 'w') as sam_file:
                sam_file.write("@HD\tVN:1.0\tSO:coordinate\n")
                sam_file.write("@SQ\tSN:contig_1\tLN:1000\n")
                sam_file.write("read_1\t0\tcontig_1\t1\t60\t75M\t*\t0\t0\t*\t*\n")
                sam_file.write("read_2\t0\tcontig_1\t50\t60\t75M\t*\t0\t0\t*\t*\n")

        # Step 6: Coverage Analysis
        print("\n📈 Step 6: Analyze Coverage")
        print("This step analyzes how well the reads cover the reference genome.")
        print("We'll look at coverage depth, evenness, and identify any gaps.")

        try:
            coverage_result = await toolkit.coverage_analysis(
                sam_path=output_sam,
                reference_path=reference_path,
                output_prefix="tutorial_coverage"
            )

            print("✅ Coverage analysis completed!")
            print(f"   Status: {coverage_result['status']}")
            print(f"   SAM file: {coverage_result['sam_file']}")
            if coverage_result.get('coverage_stats'):
                stats = coverage_result['coverage_stats']
                print(f"   Coverage statistics: {stats}")

        except Exception as e:
            print(f"⚠️  Coverage analysis encountered an issue: {e}")
            print("   This is normal in tutorial mode")

        # Summary
        print("\n🎉 Tutorial Complete!")
        print("=" * 60)
        print("📚 What You've Learned:")
        print("   1. ✅ How to initialize the BBMapToolkit")
        print("   2. ✅ Quality assessment of sequencing reads")
        print("   3. ✅ Filtering low-quality reads")
        print("   4. ✅ Mapping reads to a reference genome")
        print("   5. ✅ Analyzing coverage statistics")

        print("\n🔧 Key BBMap Tools Used:")
        print("   • readlength.sh - Quality statistics")
        print("   • bbduk.sh - Read filtering")
        print("   • bbmap.sh - Read mapping")
        print("   • pileup.sh - Coverage analysis")

        print("\n📊 Typical Workflow Results:")
        print("   • Good mapping rates: >85% for clean data")
        print("   • Quality filtering removes ~5-15% of reads")
        print("   • Coverage should be relatively uniform")
        print("   • Identity scores >95% indicate good alignment")

        print("\n🚀 Next Steps:")
        print("   1. Try with your own data files")
        print("   2. Experiment with different parameters")
        print("   3. Integrate with other MCP servers")
        print("   4. Build agent workflows that use these tools")

    finally:
        # Cleanup
        print("\n🧹 Cleaning up temporary files...")
        for filepath in [reference_path, reads_path]:
            if os.path.exists(filepath):
                os.unlink(filepath)
                print(f"   Removed: {filepath}")

        # Clean up output files (in a real scenario, you'd keep these)
        output_files = [
            "tutorial_alignment.sam",
            "tutorial_filtered_reads.fastq",
            "tutorial_quality.txt",
            "tutorial_quality_hist.txt",
            "tutorial_coverage_coverage.txt",
            "tutorial_coverage_stats.txt",
            "mapping_stats.txt",
            "scaffold_stats.txt",
            "filter_stats.txt",
            "coverage_histogram.txt"
        ]

        for filepath in output_files:
            if os.path.exists(filepath):
                os.unlink(filepath)
                print(f"   Removed: {filepath}")

if __name__ == "__main__":
    asyncio.run(tutorial_workflow())
