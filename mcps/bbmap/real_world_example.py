#!/usr/bin/env python3
"""
Real-World BBMap MCP Usage Example

This example demonstrates how to use your BBMap MCP server with actual
genomics data files that you mentioned (contig FASTA and reads FASTQ).
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bbmap_tools import BBMapToolkit


async def real_world_example():
    """
    Example using real genomics data files
    """
    print("🧬 Real-World BBMap MCP Usage Example")
    print("=" * 60)

    # Initialize toolkit
    toolkit = BBMapToolkit()
    print(f"✅ BBMap toolkit initialized")
    print(f"   Container: {toolkit.shifter_image}")
    print(f"   Command: {' '.join(toolkit.base_command)}")

    # Example file paths (you would replace these with your actual files)
    print("\n📁 Setting up file paths...")
    print("   Replace these paths with your actual data files:")

    example_files = {
        "reference_contig": "/path/to/your/contig.fasta",
        "sequencing_reads": "/path/to/your/reads.fastq",
        "output_directory": "/path/to/output/"
    }

    for file_type, path in example_files.items():
        print(f"   {file_type}: {path}")

    print("\n🔧 Example Workflow Commands:")
    print("   Here's how you would run each step with your actual data:")

    # Step 1: Quality Analysis
    print("\n1️⃣ Quality Analysis:")
    print("   Purpose: Assess the quality of your sequencing reads")

    quality_command = f"""
await toolkit.quality_stats(
    fastq_path="{example_files['sequencing_reads']}",
    output_prefix="my_quality_analysis"
)
"""
    print(f"   Code:{quality_command}")
    print("   Output: my_quality_analysis.txt, my_quality_analysis_hist.txt")

    # Step 2: Read Filtering (if needed)
    print("\n2️⃣ Read Filtering (Optional):")
    print("   Purpose: Remove low-quality or short reads")

    filter_command = f"""
await toolkit.filter_reads(
    input_fastq="{example_files['sequencing_reads']}",
    output_fastq="{example_files['output_directory']}filtered_reads.fastq",
    min_length=50,
    min_quality=25.0,
    additional_params="qtrim=rl trimq=20"
)
"""
    print(f"   Code:{filter_command}")
    print("   Output: filtered_reads.fastq, filter_stats.txt")

    # Step 3: Read Mapping
    print("\n3️⃣ Read Mapping:")
    print("   Purpose: Align reads to your reference contig")

    mapping_command = f"""
await toolkit.map_reads(
    reference_path="{example_files['reference_contig']}",
    reads_path="{example_files['sequencing_reads']}",
    output_sam="{example_files['output_directory']}alignment.sam",
    additional_params="minid=0.95 maxindel=3 ambig=random"
)
"""
    print(f"   Code:{mapping_command}")
    print("   Output: alignment.sam, mapping_stats.txt, scaffold_stats.txt")

    # Step 4: Coverage Analysis
    print("\n4️⃣ Coverage Analysis:")
    print("   Purpose: Analyze how well reads cover your contig")

    coverage_command = f"""
await toolkit.coverage_analysis(
    sam_path="{example_files['output_directory']}alignment.sam",
    reference_path="{example_files['reference_contig']}",
    output_prefix="{example_files['output_directory']}coverage_analysis"
)
"""
    print(f"   Code:{coverage_command}")
    print("   Output: coverage_analysis_coverage.txt, coverage_analysis_stats.txt")

    print("\n📊 Interpreting Results:")
    print("=" * 60)

    interpretation_guide = {
        "Quality Stats": [
            "• total_reads: Number of sequencing reads",
            "• average_length: Average read length in base pairs",
            "• Check if reads are long enough for your analysis"
        ],
        "Filter Stats": [
            "• filtering_rate: Percentage of reads removed",
            "• <15% filtering is typically good",
            "• High filtering may indicate poor sequencing quality"
        ],
        "Mapping Stats": [
            "• mapping_rate: Percentage of reads that aligned",
            "• >80% is excellent, >60% is acceptable",
            "• average_identity: How well reads match the reference"
        ],
        "Coverage Stats": [
            "• average_coverage: Mean depth of coverage",
            "• percent_covered: Percentage of reference covered",
            "• Look for even coverage distribution"
        ]
    }

    for category, metrics in interpretation_guide.items():
        print(f"\n📈 {category}:")
        for metric in metrics:
            print(f"   {metric}")

    print("\n🚨 Common Issues & Solutions:")
    print("=" * 60)

    troubleshooting = {
        "Low mapping rate (<50%)": [
            "Check if reference and reads are from same organism",
            "Try reducing minid parameter (e.g., minid=0.90)",
            "Verify file formats are correct"
        ],
        "High filtering rate (>30%)": [
            "Raw data may be low quality",
            "Adjust quality thresholds (lower min_quality)",
            "Check sequencing technology specifications"
        ],
        "Uneven coverage": [
            "May indicate PCR bias or repetitive regions",
            "Check for adapter contamination",
            "Consider different library preparation methods"
        ],
        "File not found errors": [
            "Verify all file paths are correct",
            "Check file permissions",
            "Ensure files are in expected format"
        ]
    }

    for issue, solutions in troubleshooting.items():
        print(f"\n⚠️  {issue}:")
        for solution in solutions:
            print(f"   • {solution}")


def create_sample_script():
    """Create a sample script for users to customize"""

    sample_script = '''#!/usr/bin/env python3
"""
Your BBMap Analysis Script

Customize this script with your actual file paths and parameters.
"""

import asyncio
import sys
from pathlib import Path

# Add BBMap MCP source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bbmap_tools import BBMapToolkit

async def my_bbmap_analysis():
    """Your custom BBMap workflow"""

    # Initialize toolkit
    toolkit = BBMapToolkit()

    # 🔧 CUSTOMIZE THESE PATHS FOR YOUR DATA
    reference_path = "path/to/your/contig.fasta"
    reads_path = "path/to/your/reads.fastq"
    output_dir = "results/"

    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Quality assessment
        print("🔍 Analyzing read quality...")
        quality_result = await toolkit.quality_stats(
            fastq_path=reads_path,
            output_prefix=f"{output_dir}quality"
        )
        print(f"✅ Quality analysis complete: {quality_result['status']}")

        # Step 2: Read mapping
        print("🎯 Mapping reads to reference...")
        mapping_result = await toolkit.map_reads(
            reference_path=reference_path,
            reads_path=reads_path,
            output_sam=f"{output_dir}alignment.sam",
            additional_params="minid=0.95"  # Adjust as needed
        )
        print(f"✅ Mapping complete: {mapping_result['status']}")

        # Step 3: Coverage analysis
        print("📊 Analyzing coverage...")
        coverage_result = await toolkit.coverage_analysis(
            sam_path=f"{output_dir}alignment.sam",
            reference_path=reference_path,
            output_prefix=f"{output_dir}coverage"
        )
        print(f"✅ Coverage analysis complete: {coverage_result['status']}")

        # Print summary
        print("\\n📋 Analysis Summary:")
        if 'mapping_stats' in mapping_result:
            stats = mapping_result['mapping_stats']
            print(f"  Mapping rate: {stats.get('mapping_rate', 'N/A')}%")
            print(f"  Average identity: {stats.get('average_identity', 'N/A')}%")

        if 'coverage_stats' in coverage_result:
            stats = coverage_result['coverage_stats']
            print(f"  Average coverage: {stats.get('average_coverage', 'N/A')}x")
            print(f"  Genome covered: {stats.get('percent_covered', 'N/A')}%")

    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        print("Check file paths and BBMap installation")

if __name__ == "__main__":
    asyncio.run(my_bbmap_analysis())
'''

    with open("my_bbmap_script.py", "w") as f:
        f.write(sample_script)

    print(f"\n📝 Created sample script: my_bbmap_script.py")
    print("   Customize the file paths and run with: python my_bbmap_script.py")


if __name__ == "__main__":
    asyncio.run(real_world_example())
    create_sample_script()

    print("\n🎯 Summary: Building Your BBMap Agent/MCP")
    print("=" * 80)
    print("✅ You've successfully built a comprehensive BBMap MCP server!")
    print("✅ Learned the difference between Agents and MCP servers")
    print("✅ Implemented 4 core BBMap tools with proper error handling")
    print("✅ Created comprehensive tests and documentation")
    print("✅ Demonstrated agent integration patterns")
    print("✅ Built reusable components for your team")

    print("\n🚀 Next Steps for Your Bioinformatics AI Framework:")
    print("1. 🧬 Test with your actual contig and reads files")
    print("2. 🔧 Build additional MCP servers (BLAST, SAMtools, etc.)")
    print("3. 🤖 Create specialized agents for different workflows")
    print("4. 🎼 Implement master orchestration agent")
    print("5. 📊 Add monitoring and logging capabilities")
    print("6. 🔄 Scale to handle large genomics datasets")

    print("\n💡 Key Architecture Insights:")
    print("• MCP servers = Specialized tool providers")
    print("• Agents = Intelligent workflow orchestrators")
    print("• Master agent = Coordinates multiple specialized agents")
    print("• Each component is independently testable and reusable")
    print("• Pattern scales to any bioinformatics pipeline")
