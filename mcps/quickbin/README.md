# QuickBin MCP Server

A Model Context Protocol (MCP) server providing QuickBin metagenomics binning tools. QuickBin bins contigs using coverage and kmer frequencies to group assembled sequences into taxonomically coherent clusters (bins) representing individual genomes.

## Overview

QuickBin is part of the BBTools suite and is designed for metagenomics applications where you have:
- Assembled contigs from metagenomic sequencing
- SAM/BAM files from mapping reads back to contigs
- Need to group contigs into bins representing individual genomes

## Features

- **Contig Binning**: Group contigs using coverage and kmer frequencies
- **Multiple Stringency Levels**: From extra strict to extra loose
- **Coverage Analysis**: Generate coverage statistics from SAM files
- **Quality Evaluation**: Assess binning completeness and contamination
- **Flexible Output**: Pattern-based or directory-based output formats

## Tools

### Binning Tools
- `bin_contigs(contigs_path, sam_files, output_pattern, stringency)` - Main binning with SAM files
- `bin_contigs_with_coverage(contigs_path, coverage_file, output_pattern, stringency)` - Fast re-binning with coverage file
- `generate_coverage(contigs_path, sam_files, output_coverage)` - Generate coverage statistics
- `evaluate_bins(bin_directory, reference_taxonomy)` - Evaluate binning quality

### Stringency Levels
- **xstrict**: Maximum purity, <0.1% contamination
- **strict**: High purity, <0.5% contamination
- **normal**: Balanced approach, <1% contamination (default)
- **loose**: Higher completeness, 1-5% contamination
- **xloose**: Maximum completeness, 2-10% contamination

## Installation & Setup

### Prerequisites
- Python 3.8+
- Access to shifter containers
- BBTools container image: `bryce911/bbtools:latest`

### Container Setup
This MCP uses shifter to run QuickBin from the BBTools container:
```bash
# Test container access
shifter --image bryce911/bbtools:39.27 quickbin.sh --help
```

### MCP Configuration
Add to your MCP configuration:
```json
{
  "mcpServers": {
    "quickbin": {
      "command": "python",
      "args": ["-m", "src.server_fastmcp"],
      "cwd": "/path/to/mcps/quickbin"
    }
  }
}
```

## Usage Examples

### Basic Binning Workflow
```python
# 1. Bin contigs using multiple SAM files
result = await bin_contigs(
    contigs_path="assembly/contigs.fasta",
    sam_files=["mapping/sample1.sam", "mapping/sample2.sam", "mapping/sample3.sam"],
    output_pattern="bins/bin%.fa",
    stringency="normal"
)

print(f"Generated {result['bin_stats']['total_bins']} bins")
```

### Two-Step Process (Recommended for Multiple Runs)
```python
# 1. Generate coverage file (do once)
coverage_result = await generate_coverage(
    contigs_path="assembly/contigs.fasta",
    sam_files=["mapping/sample1.sam", "mapping/sample2.sam"],
    output_coverage="coverage_stats.txt"
)

# 2. Bin using coverage file (fast, can repeat with different parameters)
binning_result = await bin_contigs_with_coverage(
    contigs_path="assembly/contigs.fasta",
    coverage_file="coverage_stats.txt",
    output_pattern="bins/",
    stringency="normal"
)
```

### Quality Evaluation
```python
# Evaluate binning quality
evaluation = await evaluate_bins(
    bin_directory="bins/",
    reference_taxonomy="reference.txt"  # optional
)

print(f"Average N50: {evaluation['quality_stats']['average_n50']}")
print(f"Total bins: {evaluation['total_bins']}")
```

## Input Requirements

### Contigs File
- **Format**: FASTA (.fa, .fasta, .fna)
- **Source**: Assembled contigs from SPAdes, metaSPAdes, MEGAHIT, etc.
- **Quality**: Remove short contigs (<1000 bp) for better binning

### SAM/BAM Files
- **Format**: Standard SAM or BAM alignment files
- **Generation**: Use bbmap with recommended parameters:
  ```bash
  bbmap.sh ref=contigs.fa in=reads.fq ambig=random mateqtag minid=0.9 maxindel=10 out=mapped.sam
  ```
- **Multiple samples**: Use 2-5+ SAM files from different samples for best results

### Coverage File (Optional)
- **Format**: Tab-delimited coverage statistics
- **Generation**: Created by QuickBin's `generate_coverage` function
- **Reuse**: Allows fast re-binning with different parameters

## Output Formats

### Pattern-Based Output
```python
output_pattern="bin%.fa"  # Creates: bin1.fa, bin2.fa, bin3.fa, etc.
```

### Directory-Based Output
```python
output_pattern="bins/"  # Creates files in bins/ directory
```

### Single File Output
```python
output_pattern="all_bins.fa"  # All bins in one file with annotations
```

## Performance Tips

1. **Use multiple SAM files**: 3-5 samples significantly improve binning accuracy
2. **Pre-filter contigs**: Remove contigs <1000 bp to reduce memory usage
3. **Generate coverage once**: Use `generate_coverage` then `bin_contigs_with_coverage` for multiple runs
4. **Start with normal stringency**: Adjust based on downstream requirements
5. **Memory requirements**: Large metagenomes may need 32GB+ RAM

## Common Workflows

### Typical Metagenome Pipeline
1. **Assembly**: SPAdes/metaSPAdes → contigs.fasta
2. **Mapping**: bbmap → multiple SAM files
3. **Binning**: QuickBin → genome bins
4. **Evaluation**: Assess quality and completeness
5. **Downstream**: Annotation, phylogenetics, comparative analysis

### Paired with BBMap MCP
```python
# Complete pipeline using both MCPs
# 1. Map reads (BBMap MCP)
mapping_result = await map_reads(
    reference_path="contigs.fasta",
    reads_path="sample1.fastq",
    output_sam="sample1.sam"
)

# 2. Bin contigs (QuickBin MCP)
binning_result = await bin_contigs(
    contigs_path="contigs.fasta",
    sam_files=["sample1.sam", "sample2.sam"],
    output_pattern="bins/bin%.fa"
)
```

## Resources

- `quickbin://docs/user-guide` - Comprehensive user guide
- `quickbin://examples/metagenome-workflow` - Complete workflow examples
- `quickbin://tools/available` - Available tools and descriptions
- `quickbin://parameters/stringency` - Stringency selection guide

## Troubleshooting

### Common Issues

**"Container not found"**
- Ensure shifter is available and BBTools container is accessible
- Test: `shifter --image bryce911/bbtools:39.27 quickbin.sh --help`

**"Out of memory"**
- Reduce dataset size or increase available RAM
- Filter short contigs: `mincontig=1000`
- Reduce quantization: `gcwidth=0.05 depthwidth=1.0`

**"Few/no bins generated"**
- Check SAM file quality and mapping rates
- Try looser stringency: `stringency="loose"`
- Ensure multiple samples for coverage diversity

**"High contamination"**
- Use stricter stringency: `stringency="strict"`
- Improve mapping quality parameters
- Check for chimeric contigs in assembly

### Performance Optimization

```python
# For large metagenomes
additional_params = "mincluster=100k minseed=5000 gcwidth=0.05 depthwidth=1.0"

# For high memory efficiency
additional_params = "mincontig=2000 minpentamersize=5k"

# For high sensitivity
additional_params = "minseed=1000 cutoff=0.45"
```

## Citation

If you use QuickBin in your research, please cite:
- BBTools suite by Brian Bushnell (DOE Joint Genome Institute)
- QuickBin publication (when available)

## Support

For issues with:
- **QuickBin tool**: Contact Brian Bushnell at bbushnell@lbl.gov
- **MCP server**: Create an issue in this repository
- **Integration**: Check the troubleshooting section above
