"""
BBMap MCP Server using FastMCP

This server provides BBMap bioinformatics tools through the Model Context Protocol (MCP).
BBMap is a fast and accurate read mapper for DNA sequencing data.
"""

from fastmcp import FastMCP
import json
import logging
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bbmap_tools import BBMapToolkit
# Import tool schemas (handle both relative and direct imports)
try:
    from .tool_schema import get_tool_schemas, get_resource_schemas
except ImportError:
    from tool_schema import get_tool_schemas, get_resource_schemas

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Reduce FastMCP logging noise during startup
logging.getLogger('fastmcp').setLevel(logging.WARNING)
logging.getLogger('mcp').setLevel(logging.WARNING)

# Create FastMCP server
mcp = FastMCP("BBMap Tools 🧬")
toolkit = BBMapToolkit()

@mcp.tool
async def map_reads(
    reference_path: str,
    reads_path: str,
    output_sam: str,
    additional_params: str | None = None
) -> dict:
    """Map sequencing reads to a reference genome using BBMap

    Args:
        reference_path: Path to reference genome file (FASTA format)
        reads_path: Path to reads file (FASTQ format)
        output_sam: Output path for SAM alignment file
        additional_params: Additional BBMap parameters (optional)
    """
    logger.info(f"Mapping reads from {reads_path} to {reference_path}")
    return await toolkit.map_reads(
        reference_path=reference_path,
        reads_path=reads_path,
        output_sam=output_sam,
        additional_params=additional_params
    )

@mcp.tool
async def quality_stats(fastq_path: str, output_prefix: str = "quality_stats") -> dict:
    """Generate comprehensive quality statistics for FASTQ files

    Args:
        fastq_path: Path to FASTQ file to analyze
        output_prefix: Prefix for output statistics files
    """
    logger.info(f"Analyzing quality statistics for {fastq_path}")
    return await toolkit.quality_stats(
        fastq_path=fastq_path,
        output_prefix=output_prefix
    )

@mcp.tool
async def coverage_analysis(
    sam_path: str,
    reference_path: str,
    output_prefix: str = "coverage"
) -> dict:
    """Analyze read coverage from SAM/BAM alignment files

    Args:
        sam_path: Path to SAM/BAM alignment file
        reference_path: Path to reference genome file (FASTA format)
        output_prefix: Prefix for coverage output files
    """
    logger.info(f"Analyzing coverage for {sam_path}")
    return await toolkit.coverage_analysis(
        sam_path=sam_path,
        reference_path=reference_path,
        output_prefix=output_prefix
    )

@mcp.tool
async def filter_reads(
    input_fastq: str,
    output_fastq: str,
    min_length: int = 50,
    min_quality: float = 20.0,
    additional_params: str | None = None
) -> dict:
    """Filter reads based on quality and length criteria

    Args:
        input_fastq: Input FASTQ file to filter
        output_fastq: Output path for filtered FASTQ file
        min_length: Minimum read length threshold
        min_quality: Minimum average quality score
        additional_params: Additional filtering parameters (optional)
    """
    logger.info(f"Filtering reads from {input_fastq}")
    return await toolkit.filter_reads(
        input_fastq=input_fastq,
        output_fastq=output_fastq,
        min_length=min_length,
        min_quality=min_quality,
        additional_params=additional_params
    )

@mcp.resource("bbmap://docs/user-guide")
async def get_user_guide() -> str:
    """Get BBMap user guide documentation"""
    return """# BBMap User Guide

## Overview
BBMap is a fast, accurate splice-aware aligner for RNA and DNA. It can align reads from all major sequencing platforms.

## Basic Usage

### 1. Read Mapping
Map reads to a reference genome:
```
map_reads(
    reference_path="/path/to/reference.fasta",
    reads_path="/path/to/reads.fastq",
    output_sam="/path/to/output.sam"
)
```

### 2. Quality Analysis
Analyze read quality statistics:
```
quality_stats(
    fastq_path="/path/to/reads.fastq",
    output_prefix="quality_analysis"
)
```

### 3. Coverage Analysis
Analyze alignment coverage:
```
coverage_analysis(
    sam_path="/path/to/alignment.sam",
    reference_path="/path/to/reference.fasta",
    output_prefix="coverage_results"
)
```

### 4. Read Filtering
Filter reads by quality and length:
```
filter_reads(
    input_fastq="/path/to/input.fastq",
    output_fastq="/path/to/filtered.fastq",
    min_length=50,
    min_quality=20.0
)
```

## Common Parameters
- `additional_params`: Pass extra parameters like "minid=0.95 maxindel=3"
- Quality scores are typically in Phred scale (0-40)
- Length thresholds help remove low-quality short reads

## Output Files
- SAM files: Standard alignment format
- Stats files: Tab-delimited statistics
- Histogram files: Quality/length distributions
"""

@mcp.resource("bbmap://examples/basic-workflow")
async def get_basic_workflow() -> str:
    """Get example BBMap workflow"""
    return """# Basic BBMap Workflow Example

This workflow demonstrates a complete read mapping and analysis pipeline:

## Step 1: Quality Assessment (Pre-mapping)
```python
# Analyze raw read quality
quality_result = await quality_stats(
    fastq_path="raw_reads.fastq",
    output_prefix="raw_quality"
)
print(f"Raw reads: {quality_result['quality_stats']['total_reads']}")
```

## Step 2: Read Filtering (Optional)
```python
# Filter low-quality reads
filter_result = await filter_reads(
    input_fastq="raw_reads.fastq",
    output_fastq="filtered_reads.fastq",
    min_length=50,
    min_quality=25.0
)
print(f"Filtered reads: {filter_result['filter_stats']['output_reads']}")
```

## Step 3: Read Mapping
```python
# Map filtered reads to reference
mapping_result = await map_reads(
    reference_path="reference_genome.fasta",
    reads_path="filtered_reads.fastq",
    output_sam="alignment.sam",
    additional_params="minid=0.95 maxindel=3"
)
print(f"Mapping rate: {mapping_result['mapping_stats']['mapping_rate']}%")
```

## Step 4: Coverage Analysis
```python
# Analyze alignment coverage
coverage_result = await coverage_analysis(
    sam_path="alignment.sam",
    reference_path="reference_genome.fasta",
    output_prefix="final_coverage"
)
print(f"Average coverage: {coverage_result['coverage_stats']['average_coverage']}x")
```

## Expected Results
- Quality stats show read length/quality distributions
- Filtering typically removes 5-15% of reads
- Good mapping rates are usually >80% for clean data
- Coverage should be relatively uniform across the genome
"""

@mcp.resource("bbmap://tools/available")
async def get_available_tools() -> str:
    """Get list of available BBMap tools"""
    tools_info = {
        "core_tools": [
            {
                "name": "map_reads",
                "description": "Map reads to reference genome",
                "primary_use": "Alignment/mapping"
            },
            {
                "name": "quality_stats",
                "description": "Analyze FASTQ quality statistics",
                "primary_use": "Quality control"
            },
            {
                "name": "coverage_analysis",
                "description": "Analyze alignment coverage",
                "primary_use": "Coverage assessment"
            },
            {
                "name": "filter_reads",
                "description": "Filter reads by quality/length",
                "primary_use": "Quality filtering"
            }
        ],
        "underlying_bbtools": [
            "bbmap.sh - Main read mapper",
            "readlength.sh - Read statistics",
            "pileup.sh - Coverage analysis",
            "bbduk.sh - Read filtering/trimming"
        ],
        "container_info": {
            "image": "bryce911/bbtools:39.27",
            "runtime": "shifter",
            "command_prefix": "shifter --image bryce911/bbtools:39.27"
        }
    }

    return json.dumps(tools_info, indent=2)

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()