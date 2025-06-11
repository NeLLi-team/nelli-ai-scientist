#!/usr/bin/env python3
"""
Complete Metagenome Binning Workflow Example

This example demonstrates how to use the QuickBin MCP for a complete
metagenomics binning workflow, from contigs and SAM files to evaluated bins.
"""

import asyncio
import json
import os
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from quickbin_tools import QuickBinToolkit


async def complete_binning_workflow():
    """
    Demonstrate a complete metagenome binning workflow

    This workflow assumes you have:
    1. Assembled contigs from metagenomic sequencing
    2. SAM files from mapping reads back to contigs (e.g., using bbmap)
    """

    print("🧬 Complete Metagenome Binning Workflow")
    print("=" * 60)

    # Initialize toolkit
    toolkit = QuickBinToolkit()

    # Setup paths (modify these for your data)
    print("📁 Setting up file paths...")

    # Input files (you would replace these with your actual files)
    contigs_file = "data/assembled_contigs.fasta"  # Your assembled contigs
    sam_files = [
        "data/sample1_mapped.sam",  # SAM file from sample 1
        "data/sample2_mapped.sam",  # SAM file from sample 2
        "data/sample3_mapped.sam",  # SAM file from sample 3
    ]

    # Output directories
    output_dir = Path("quickbin_results")
    output_dir.mkdir(exist_ok=True)

    coverage_file = output_dir / "coverage_stats.txt"
    bins_dir = output_dir / "bins"
    bins_dir.mkdir(exist_ok=True)

    print(f"   Contigs: {contigs_file}")
    print(f"   SAM files: {len(sam_files)} files")
    print(f"   Output directory: {output_dir}")

    # Check if input files exist (for demo purposes, we'll create dummy files)
    if not os.path.exists(contigs_file):
        print("\n⚠️  Input files not found. Creating demo files...")
        await create_demo_files(contigs_file, sam_files)

    # Workflow Step 1: Generate Coverage Statistics
    print(f"\n🔍 Step 1: Generating coverage statistics...")
    print("-" * 40)

    try:
        coverage_result = await toolkit.generate_coverage(
            contigs_path=contigs_file,
            sam_files=sam_files,
            output_coverage=str(coverage_file),
            additional_params="mincontig=500"  # Only consider contigs >500bp
        )

        if coverage_result["status"] == "success":
            print("✅ Coverage generation successful")
            print(f"   Coverage file: {coverage_result['coverage_file']}")
            print(f"   Total contigs: {coverage_result['coverage_stats']['total_contigs']}")
            print(f"   Average coverage: {coverage_result['coverage_stats']['average_coverage']:.2f}x")
            print(f"   Samples: {len(coverage_result['coverage_stats']['samples'])}")
        else:
            print("❌ Coverage generation failed")
            print(f"   Error: {coverage_result['error_message']}")
            return False

    except Exception as e:
        print(f"❌ Error in coverage generation: {e}")
        return False

    # Workflow Step 2: Bin Contigs with Different Stringencies
    print(f"\n🗂️  Step 2: Binning contigs with different stringencies...")
    print("-" * 40)

    stringencies = ["strict", "normal", "loose"]
    binning_results = {}

    for stringency in stringencies:
        print(f"\n   Binning with '{stringency}' stringency...")

        try:
            result = await toolkit.bin_contigs_with_coverage(
                contigs_path=contigs_file,
                coverage_file=str(coverage_file),
                output_pattern=str(bins_dir / f"{stringency}_bin%.fa"),
                stringency=stringency,
                additional_params="mincluster=50k minseed=2000"
            )

            if result["status"] == "success":
                binning_results[stringency] = result
                print(f"   ✅ {stringency}: {result['bin_stats']['total_bins']} bins generated")
                print(f"      Largest bin: {result['bin_stats']['largest_bin_size']} bytes")
            else:
                print(f"   ❌ {stringency}: Failed - {result['error_message']}")

        except Exception as e:
            print(f"   ❌ {stringency}: Error - {e}")

    # Workflow Step 3: Evaluate Binning Quality
    print(f"\n📊 Step 3: Evaluating binning quality...")
    print("-" * 40)

    evaluation_results = {}

    for stringency in stringencies:
        if stringency in binning_results:
            print(f"\n   Evaluating '{stringency}' bins...")

            try:
                # Create a temporary directory for this stringency's bins
                stringency_bins_dir = bins_dir / f"{stringency}_bins"
                stringency_bins_dir.mkdir(exist_ok=True)

                # Move bins to separate directory for evaluation
                # (In practice, they might already be organized this way)

                eval_result = await toolkit.evaluate_bins(
                    bin_directory=str(stringency_bins_dir),
                    additional_params="validate=f"  # No reference validation for demo
                )

                if eval_result["status"] == "success":
                    evaluation_results[stringency] = eval_result
                    stats = eval_result["quality_stats"]
                    print(f"   ✅ {stringency}: Evaluation complete")
                    print(f"      Total bins: {eval_result['total_bins']}")
                    print(f"      Total length: {stats['total_length']:,} bp")
                    print(f"      Average N50: {stats['average_n50']:,.0f} bp")
                    print(f"      Average GC: {stats['average_gc_content']:.1f}%")
                else:
                    print(f"   ❌ {stringency}: Evaluation failed")

            except Exception as e:
                print(f"   ❌ {stringency}: Evaluation error - {e}")

    # Workflow Step 4: Compare Results and Recommendations
    print(f"\n🎯 Step 4: Results comparison and recommendations...")
    print("-" * 40)

    if evaluation_results:
        print("\n   📋 BINNING COMPARISON:")
        print("   " + "-" * 50)
        print("   Stringency | Bins | Total Length | Avg N50")
        print("   " + "-" * 50)

        for stringency in stringencies:
            if stringency in evaluation_results:
                result = evaluation_results[stringency]
                stats = result["quality_stats"]
                print(f"   {stringency:10} | {result['total_bins']:4} | {stats['total_length']:11,} | {stats['average_n50']:8,.0f}")

        print("   " + "-" * 50)

        # Recommendations
        print("\n   💡 RECOMMENDATIONS:")

        # Find best balance
        best_stringency = None
        best_score = 0

        for stringency in stringencies:
            if stringency in evaluation_results:
                result = evaluation_results[stringency]
                stats = result["quality_stats"]

                # Simple scoring: balance between number of bins and total length
                score = result['total_bins'] * (stats['total_length'] / 1000000)  # Bins * Mb

                if score > best_score:
                    best_score = score
                    best_stringency = stringency

        if best_stringency:
            print(f"   🏆 Best overall result: '{best_stringency}' stringency")
            print(f"      Good balance of bin count and genome recovery")

        print(f"   📈 For maximum completeness: Use 'loose' stringency")
        print(f"   🎯 For maximum purity: Use 'strict' stringency")
        print(f"   ⚖️  For balanced approach: Use 'normal' stringency")

    # Workflow Step 5: Save Summary Report
    print(f"\n📄 Step 5: Generating summary report...")
    print("-" * 40)

    try:
        summary_report = {
            "workflow": "QuickBin Metagenome Binning",
            "input_files": {
                "contigs": contigs_file,
                "sam_files": sam_files,
                "total_samples": len(sam_files)
            },
            "coverage_analysis": coverage_result if 'coverage_result' in locals() else None,
            "binning_results": binning_results,
            "evaluation_results": evaluation_results,
            "recommendations": {
                "best_stringency": best_stringency if 'best_stringency' in locals() else None,
                "for_completeness": "loose",
                "for_purity": "strict",
                "for_balance": "normal"
            }
        }

        report_file = output_dir / "binning_workflow_report.json"
        with open(report_file, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)

        print(f"✅ Summary report saved: {report_file}")

    except Exception as e:
        print(f"⚠️  Could not save report: {e}")

    print(f"\n🎉 Workflow completed successfully!")
    print(f"   Results saved in: {output_dir}")
    print(f"   Next steps:")
    print(f"   1. Review bins in chosen stringency directory")
    print(f"   2. Perform taxonomic classification (e.g., with CheckM)")
    print(f"   3. Annotate genomes (e.g., with Prokka)")
    print(f"   4. Comparative genomics analysis")

    return True


async def create_demo_files(contigs_file: str, sam_files: list):
    """Create demo files for testing the workflow"""
    print("   Creating demo metagenome data...")

    # Create directory structure
    os.makedirs(os.path.dirname(contigs_file), exist_ok=True)

    # Create demo contigs file
    with open(contigs_file, 'w') as f:
        # Simulate contigs from different organisms
        f.write(">NODE_1_length_5423_cov_12.5\n")
        f.write("ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG" * 50 + "\n")
        f.write(">NODE_2_length_8932_cov_8.3\n")
        f.write("GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA" * 70 + "\n")
        f.write(">NODE_3_length_3421_cov_15.7\n")
        f.write("TTAACCGGTTAACCGGTTAACCGGTTAACCGGTTAACCGGTTAA" * 30 + "\n")
        f.write(">NODE_4_length_6789_cov_6.2\n")
        f.write("CCCCGGGGAAAATTTTCCCCGGGGAAAATTTTCCCCGGGGAAAA" * 55 + "\n")
        f.write(">NODE_5_length_4567_cov_11.4\n")
        f.write("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT" * 40 + "\n")

    # Create demo SAM files
    for sam_file in sam_files:
        os.makedirs(os.path.dirname(sam_file), exist_ok=True)
        with open(sam_file, 'w') as f:
            # SAM header
            f.write("@HD\tVN:1.0\tSO:unsorted\n")
            f.write("@SQ\tSN:NODE_1_length_5423_cov_12.5\tLN:2150\n")
            f.write("@SQ\tSN:NODE_2_length_8932_cov_8.3\tLN:2940\n")
            f.write("@SQ\tSN:NODE_3_length_3421_cov_15.7\tLN:1290\n")
            f.write("@SQ\tSN:NODE_4_length_6789_cov_6.2\tLN:2365\n")
            f.write("@SQ\tSN:NODE_5_length_4567_cov_11.4\tLN:1720\n")

            # Some dummy alignments (minimal for coverage calculation)
            f.write("read1\t0\tNODE_1_length_5423_cov_12.5\t100\t60\t50M\t*\t0\t0\tATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT\t*\n")
            f.write("read2\t0\tNODE_2_length_8932_cov_8.3\t200\t60\t50M\t*\t0\t0\tGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\t*\n")

    print(f"   ✅ Demo files created:")
    print(f"      Contigs: {contigs_file} (5 contigs)")
    print(f"      SAM files: {len(sam_files)} files with minimal alignments")


async def main():
    """Run the complete workflow example"""
    try:
        success = await complete_binning_workflow()
        if success:
            print("\n✨ Workflow example completed successfully!")
        else:
            print("\n⚠️  Workflow example encountered issues.")
            return 1
    except KeyboardInterrupt:
        print("\n⚠️  Workflow interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Workflow failed with error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
