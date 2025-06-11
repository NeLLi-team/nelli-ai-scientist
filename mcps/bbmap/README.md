# BBMap MCP Server 🧬

A **Model Context Protocol (MCP)** server that provides BBMap bioinformatics tools for read mapping, quality analysis, and sequence processing.

## Overview

BBMap is a fast, accurate aligner for RNA and DNA sequencing data. This MCP server wraps BBMap functionality to make it accessible through the Model Context Protocol, enabling integration with AI agents and other bioinformatics workflows.

## Features

### 🔧 **Core Tools**
- **`map_reads`**: Map sequencing reads to reference genomes
- **`quality_stats`**: Analyze FASTQ file quality statistics
- **`coverage_analysis`**: Assess alignment coverage and depth
- **`filter_reads`**: Filter reads by quality and length criteria

### 📊 **Key Capabilities**
- **High-performance mapping** using BBMap's optimized algorithms
- **Comprehensive quality metrics** for sequencing data assessment
- **Coverage analysis** with detailed statistics and visualizations
- **Smart read filtering** to improve downstream analysis quality
- **Container-based execution** through Shifter for reproducibility

## Quick Start

### Installation

```bash
# Navigate to BBMap MCP directory
cd /pscratch/sd/j/jvillada/nelli-ai-scientist/mcps/bbmap

# Install dependencies (if needed)
pip install fastmcp

# Test the server
python -m src.server_fastmcp
```

### Basic Usage

#### 1. Map Reads to Reference
```python
result = await map_reads(
    reference_path="reference_genome.fasta",
    reads_path="sequencing_reads.fastq",
    output_sam="alignment.sam"
)
print(f"Mapping rate: {result['mapping_stats']['mapping_rate']}%")
```

#### 2. Analyze Read Quality
```python
result = await quality_stats(
    fastq_path="sequencing_reads.fastq",
    output_prefix="quality_analysis"
)
print(f"Total reads: {result['quality_stats']['total_reads']}")
```

#### 3. Assess Coverage
```python
result = await coverage_analysis(
    sam_path="alignment.sam",
    reference_path="reference_genome.fasta",
    output_prefix="coverage_stats"
)
print(f"Average coverage: {result['coverage_stats']['average_coverage']}x")
```

#### 4. Filter Low-Quality Reads
```python
result = await filter_reads(
    input_fastq="raw_reads.fastq",
    output_fastq="filtered_reads.fastq",
    min_length=50,
    min_quality=25.0
)
print(f"Kept {result['filter_stats']['output_reads']} high-quality reads")
```

## Technical Details

### Container Execution
This MCP server uses **Shifter** to run BBTools in a containerized environment:
```bash
shifter --image bryce911/bbtools:39.27 [bbtools_command]
```

### Supported File Formats
- **Input**: FASTA (reference), FASTQ (reads), SAM/BAM (alignments)
- **Output**: SAM (alignments), TXT (statistics), TSV (coverage data)

### BBTools Programs Used
- `bbmap.sh` - Main read mapping tool
- `readlength.sh` - Quality statistics generation
- `pileup.sh` - Coverage analysis
- `bbduk.sh` - Read filtering and trimming

## Example Workflow

Here's a complete bioinformatics workflow using the BBMap MCP:

```python
# 1. Assess raw read quality
quality_result = await quality_stats(
    fastq_path="raw_reads.fastq",
    output_prefix="raw_quality"
)

# 2. Filter low-quality reads (optional)
if quality_result['quality_stats']['average_length'] < 100:
    filter_result = await filter_reads(
        input_fastq="raw_reads.fastq",
        output_fastq="filtered_reads.fastq",
        min_length=50,
        min_quality=20.0
    )
    reads_file = "filtered_reads.fastq"
else:
    reads_file = "raw_reads.fastq"

# 3. Map reads to reference genome
mapping_result = await map_reads(
    reference_path="reference.fasta",
    reads_path=reads_file,
    output_sam="alignment.sam",
    additional_params="minid=0.95"
)

# 4. Analyze mapping coverage
coverage_result = await coverage_analysis(
    sam_path="alignment.sam",
    reference_path="reference.fasta",
    output_prefix="final_coverage"
)

print(f"Pipeline complete!")
print(f"Mapping rate: {mapping_result['mapping_stats']['mapping_rate']}%")
print(f"Average coverage: {coverage_result['coverage_stats']['average_coverage']}x")
```

## Integration with Agents

This MCP server is designed to work seamlessly with AI agents for automated bioinformatics workflows:

```yaml
# Agent configuration example
tools:
  - name: bbmap_server
    type: mcp_server
    config:
      command: python
      args: ["-m", "src.server_fastmcp"]
      cwd: "/path/to/bbmap"
```

## Performance Tips

1. **File Paths**: Use absolute paths for better reliability
2. **Quality Thresholds**: Adjust `min_quality` based on your sequencing technology
3. **Memory Usage**: BBMap can use significant memory for large genomes
4. **Parallel Processing**: BBMap automatically uses available CPU cores

## Troubleshooting

### Common Issues

**Container Access**
```bash
# Test shifter access
shifter --image bryce911/bbtools:39.27 bbmap.sh --help
```

**File Permissions**
```bash
# Ensure read/write access to input/output directories
chmod 755 /path/to/data/directory
```

**Memory Issues**
```bash
# Monitor memory usage during large mappings
htop
```

## Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_bbmap_tools.py::test_map_reads_success -v
```

### Adding New Tools
1. Add the tool method to `BBMapToolkit` class
2. Add the corresponding schema to `tool_schema.py`
3. Add the MCP tool decorator in `server_fastmcp.py`
4. Write comprehensive tests

## Contributing

This MCP server follows the established patterns in the nelli-ai-scientist framework. See the main project documentation for contribution guidelines.

## License

This project is part of the nelli-ai-scientist framework and follows the same licensing terms.
