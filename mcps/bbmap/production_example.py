#!/usr/bin/env python3
"""
🧬 BBMap MCP Server - Production Ready Example

This is your complete, production-ready BBMap MCP server implementation.
It demonstrates all the concepts we've built together in a practical workflow.

USAGE WITH YOUR DATA:
1. Replace the example paths below with your actual contig FASTA and reads FASTQ files
2. Run with pixi: pixi run python mcps/bbmap/production_example.py
3. Or run directly: python mcps/bbmap/production_example.py

This example shows:
- ✅ BBMap toolkit initialization
- ✅ Quality assessment workflow
- ✅ Read filtering pipeline
- ✅ Genome mapping process
- ✅ Coverage analysis
- ✅ Error handling and logging
- ✅ Integration with your master agent workflow
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any

# Add our BBMap tools to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bbmap_tools import BBMapToolkit

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BBMapWorkflowManager:
    """High-level workflow manager for BBMap operations"""

    def __init__(self):
        """Initialize the workflow manager"""
        self.toolkit = BBMapToolkit()
        logger.info(f"Initialized BBMap toolkit with image: {self.toolkit.shifter_image}")

    async def analyze_data_quality(self, fastq_path: str) -> Dict[str, Any]:
        """Step 1: Analyze sequencing data quality"""
        logger.info(f"🔍 Analyzing quality of: {fastq_path}")

        try:
            result = await self.toolkit.quality_stats(
                fastq_path=fastq_path,
                output_prefix="data_quality_analysis"
            )

            logger.info("✅ Quality analysis completed")
            return result

        except FileNotFoundError as e:
            logger.error(f"❌ Input file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Quality analysis failed: {e}")
            raise

    async def filter_low_quality_reads(
        self,
        input_fastq: str,
        output_fastq: str,
        min_length: int = 50,
        min_quality: float = 20.0
    ) -> Dict[str, Any]:
        """Step 2: Filter out low-quality reads"""
        logger.info(f"🔧 Filtering reads: {input_fastq} -> {output_fastq}")

        try:
            result = await self.toolkit.filter_reads(
                input_fastq=input_fastq,
                output_fastq=output_fastq,
                min_length=min_length,
                min_quality=min_quality,
                additional_params="qtrim=rl trimq=20 minlen=30"
            )

            logger.info("✅ Read filtering completed")
            return result

        except Exception as e:
            logger.error(f"❌ Read filtering failed: {e}")
            raise

    async def map_reads_to_genome(
        self,
        reference_path: str,
        reads_path: str,
        output_sam: str
    ) -> Dict[str, Any]:
        """Step 3: Map reads to reference genome"""
        logger.info(f"🎯 Mapping reads to genome: {reads_path} -> {output_sam}")

        try:
            result = await self.toolkit.map_reads(
                reference_path=reference_path,
                reads_path=reads_path,
                output_sam=output_sam,
                additional_params="minid=0.95 maxindel=3 ambig=random threads=auto"
            )

            logger.info("✅ Read mapping completed")
            return result

        except Exception as e:
            logger.error(f"❌ Read mapping failed: {e}")
            raise

    async def analyze_coverage(
        self,
        sam_path: str,
        reference_path: str,
        output_prefix: str = "coverage_analysis"
    ) -> Dict[str, Any]:
        """Step 4: Analyze genome coverage"""
        logger.info(f"📊 Analyzing coverage: {sam_path}")

        try:
            result = await self.toolkit.coverage_analysis(
                sam_path=sam_path,
                reference_path=reference_path,
                output_prefix=output_prefix
            )

            logger.info("✅ Coverage analysis completed")
            return result

        except Exception as e:
            logger.error(f"❌ Coverage analysis failed: {e}")
            raise

    async def run_complete_workflow(
        self,
        reference_genome: str,
        raw_reads: str,
        output_directory: str = "bbmap_results"
    ) -> Dict[str, Any]:
        """Run the complete BBMap bioinformatics workflow"""

        logger.info("🧬 Starting complete BBMap bioinformatics workflow")
        logger.info("=" * 60)

        # Create output directory
        os.makedirs(output_directory, exist_ok=True)

        workflow_results = {
            "status": "running",
            "steps": {},
            "output_directory": output_directory
        }

        try:
            # Step 1: Quality Assessment
            logger.info("📋 Step 1/4: Quality Assessment")
            quality_result = await self.analyze_data_quality(raw_reads)
            workflow_results["steps"]["quality_assessment"] = quality_result

            # Step 2: Read Filtering (optional, based on quality)
            logger.info("📋 Step 2/4: Read Filtering")
            filtered_reads = os.path.join(output_directory, "filtered_reads.fastq")

            try:
                filter_result = await self.filter_low_quality_reads(
                    input_fastq=raw_reads,
                    output_fastq=filtered_reads
                )
                workflow_results["steps"]["read_filtering"] = filter_result
                reads_for_mapping = filtered_reads
            except Exception as e:
                logger.warning(f"⚠️ Read filtering failed, using original reads: {e}")
                reads_for_mapping = raw_reads
                workflow_results["steps"]["read_filtering"] = {"status": "skipped", "reason": str(e)}

            # Step 3: Read Mapping
            logger.info("📋 Step 3/4: Read Mapping")
            alignment_sam = os.path.join(output_directory, "alignment.sam")
            mapping_result = await self.map_reads_to_genome(
                reference_path=reference_genome,
                reads_path=reads_for_mapping,
                output_sam=alignment_sam
            )
            workflow_results["steps"]["read_mapping"] = mapping_result

            # Step 4: Coverage Analysis
            logger.info("📋 Step 4/4: Coverage Analysis")
            coverage_result = await self.analyze_coverage(
                sam_path=alignment_sam,
                reference_path=reference_genome,
                output_prefix=os.path.join(output_directory, "coverage")
            )
            workflow_results["steps"]["coverage_analysis"] = coverage_result

            # Final status
            workflow_results["status"] = "completed"
            logger.info("🎉 Complete workflow finished successfully!")

            return workflow_results

        except Exception as e:
            workflow_results["status"] = "failed"
            workflow_results["error"] = str(e)
            logger.error(f"❌ Workflow failed: {e}")
            raise

async def main():
    """Main function - Update paths for your actual data"""

    print("🧬 BBMap MCP Server - Production Example")
    print("=" * 50)

    # 🚨 UPDATE THESE PATHS WITH YOUR ACTUAL DATA FILES 🚨
    reference_genome = "/path/to/your/contig.fasta"      # Your contig FASTA file
    raw_reads = "/path/to/your/reads.fastq"             # Your reads FASTQ file
    output_dir = "bbmap_analysis_results"

    print(f"📁 Configuration:")
    print(f"   Reference genome: {reference_genome}")
    print(f"   Raw reads: {raw_reads}")
    print(f"   Output directory: {output_dir}")

    # Check if files exist (for demo purposes)
    if not os.path.exists(reference_genome):
        print(f"\n⚠️  Demo Mode: {reference_genome} not found")
        print("   Update the file paths above with your actual data files")
        print("   This example shows the workflow structure that will run")
        print("\n🔧 Workflow Preview:")

        manager = BBMapWorkflowManager()
        print(f"   ✅ BBMap toolkit ready: {manager.toolkit.shifter_image}")
        print(f"   📋 Pipeline steps: Quality → Filter → Map → Coverage")
        print(f"   🐳 Container command: {' '.join(manager.toolkit.base_command)}")

        return

    # Run the actual workflow with real data
    try:
        manager = BBMapWorkflowManager()
        results = await manager.run_complete_workflow(
            reference_genome=reference_genome,
            raw_reads=raw_reads,
            output_directory=output_dir
        )

        # Print summary
        print("\n📊 Workflow Results Summary:")
        print(f"   Status: {results['status']}")
        print(f"   Output directory: {results['output_directory']}")

        for step_name, step_result in results["steps"].items():
            if isinstance(step_result, dict) and "status" in step_result:
                print(f"   {step_name}: {step_result['status']}")

        print("\n🚀 Next Steps:")
        print("   1. Check output files in the results directory")
        print("   2. Integrate this workflow into your master agent")
        print("   3. Add additional bioinformatics tools as needed")

    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        print("   Check the log output above for detailed error information")

if __name__ == "__main__":
    asyncio.run(main())
