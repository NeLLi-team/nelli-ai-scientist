# Nucleic Acid Analysis MCP Server

A Model Context Protocol (MCP) server focused exclusively on DNA and RNA sequence analysis using BioPython library. This server provides specialized tools for nucleic acid analysis including assembly statistics, promoter detection, GC skew analysis, and specialized giant virus analysis.

## Features

- **Assembly Statistics**: N50, L50, size distributions, GC content
- **Promoter Detection**: Prokaryotic, eukaryotic, and giant virus-specific motifs
- **GC Skew Analysis**: Replication origin identification
- **CpG Island Detection**: Methylation site analysis
- **Repeat Analysis**: Tandem repeats, SSRs, direct/inverted repeats
- **Gene Prediction**: Pyrodigal-based gene finding and coding density
- **K-mer Analysis**: Composition analysis with diversity metrics
- **Giant Virus Analysis**: NCLDV-specific promoter motifs and AT-rich regions

## Tools

### Core Sequence Analysis
- `sequence_stats(sequence, sequence_type)` - Calculate comprehensive sequence statistics
- `translate_sequence(sequence, genetic_code=1)` - Translate DNA/RNA to protein
- `analyze_fasta_file(file_path, sequence_type="dna")` - Comprehensive FASTA file analysis

### Assembly & Quality Metrics
- `assembly_stats(sequences)` - Calculate N50, L50, size distributions
- `repeat_detection(sequences, min_repeat_length=10, max_repeat_length=100)` - Find repeats and SSRs

### Regulatory Element Detection
- `promoter_identification(sequences, upstream_length=100)` - Standard promoter motifs
- `giant_virus_promoter_search(sequences, upstream_length=150)` - Giant virus-specific motifs
- `cpg_island_detection(sequences, min_length=200, gc_threshold=50.0, oe_ratio_threshold=0.6)` - CpG islands

### Genome Architecture Analysis
- `gc_skew_analysis(sequences, window_size=10000, step_size=5000)` - Replication origin detection
- `gene_prediction_and_coding_stats(sequences, genetic_code=11, meta_mode=True)` - Gene finding with Pyrodigal
- `kmer_analysis(sequences, k_values=[3,4,5,6], per_sequence=False)` - K-mer composition analysis

### File Operations
- `read_fasta_file(file_path)` - Read sequences from FASTA file
- `write_json_report(data, output_path)` - Write analysis results to JSON

## Resources

- `sequences://examples` - Example sequences for testing
- `analysis://help` - Tool documentation and specialties

## Architecture

This server uses a simplified single-class architecture:
- **Single NucleicAcidAnalyzer class** in `tools.py` - All nucleic acid analysis methods
- **Direct method calls** in `server.py` - No wrapper layer
- **Focus on nucleic acid analysis only** - Removed external tool placeholders
- **FastMCP auto-generates schemas** - No manual schema definitions needed

## Usage

```python
# Start the server
python src/server.py

# Test the simplified analyzer
python test_simplified.py
```

## Files

- `src/server.py` - FastMCP server exposing nucleic acid analysis tools
- `src/tools.py` - Single NucleicAcidAnalyzer class with all methods
- `test_simplified.py` - Test script for the simplified architecture

## Dependencies

- `biopython` - Bioinformatics library
- `fastmcp` - FastMCP framework  
- `mcp` - Model Context Protocol library

## Running with Pixi

```bash
# Start the MCP server
pixi run mcp-run

# Run the test client
pixi run mcp-client

# Run tests
pixi run mcp-test
```