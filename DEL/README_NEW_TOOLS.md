# New Bioinformatics Tools in BioPython MCP Server

This document describes the new specialized bioinformatics analysis tools that have been added to the BioPython MCP server.

## Overview

The following new tools have been added for advanced genomic analysis:

1. **Giant Virus Promoter Search** - Detect giant virus-specific promoter motifs
2. **GC Skew Analysis** - Identify replication origins and strand bias
3. **CpG Island Detection** - Find CpG-rich regions in sequences
4. **Assembly Statistics** - Calculate comprehensive assembly metrics (N50, L50, etc.)
5. **Repeat Detection** - Identify tandem repeats and simple sequence repeats
6. **Gene Prediction and Coding Stats** - Predict genes using Pyrodigal
7. **K-mer Analysis** - Analyze k-mer frequencies and diversity
8. **Promoter Identification** - Find prokaryotic and eukaryotic promoter motifs

## Tool Descriptions

### 1. Giant Virus Promoter Search

**Function**: `giant_virus_promoter_search`

Searches for giant virus-specific promoter motifs based on NCLDV (Nucleocytoplasmic Large DNA Viruses) research literature.

**Motifs Detected**:
- **Mimivirus early**: AAAATTGA, AAAATTGG, GAAATTGA (45% of genes)
- **Mimivirus late**: AT-rich regions with specific patterns
- **MEGA-box**: TATATAAAATTGA, TATATAAAATTGG (proposed ancestral motif)
- **Asfarviridae**: TATTT, TATATA box motifs
- **Marseillevirus**: AAATATTT motifs
- **CroV late**: TCTA core promoters
- **Phycodnavirus**: AAAAATTGA early promoters

**Parameters**:
- `sequences`: List of sequences to analyze
- `upstream_length`: Length of upstream region (default: 150bp)

**Returns**: Motif locations, types, densities, and A/T-rich region analysis

### 2. GC Skew Analysis

**Function**: `gc_skew_analysis`

Calculates GC skew [(G-C)/(G+C)] using sliding windows to identify replication origins and strand bias patterns.

**Parameters**:
- `sequences`: List of sequences to analyze
- `window_size`: Sliding window size (default: 10,000bp)
- `step_size`: Step size for windows (default: 5,000bp)

**Returns**: 
- Per-sequence GC skew profiles
- Potential replication origin candidates
- Maximum/minimum skew regions
- Overall skew statistics

### 3. CpG Island Detection

**Function**: `cpg_island_detection`

Detects CpG islands using standard criteria: length ≥200bp, GC content ≥50%, observed/expected CpG ratio ≥0.6.

**Parameters**:
- `sequences`: List of sequences to analyze
- `min_length`: Minimum island length (default: 200bp)
- `gc_threshold`: Minimum GC content % (default: 50.0)
- `oe_ratio_threshold`: Minimum O/E CpG ratio (default: 0.6)

**Returns**:
- Island locations and properties
- Coverage statistics
- Merged overlapping islands

### 4. Assembly Statistics

**Function**: `assembly_stats`

Calculates comprehensive assembly quality metrics commonly used in genome assembly evaluation.

**Parameters**:
- `sequences`: List of contigs/scaffolds

**Returns**:
- **N50/L50**: Assembly contiguity metrics
- **N90/L90**: Additional contiguity measures
- **Size distribution**: Contig length categories
- **GC content**: Overall nucleotide composition
- **Gap content**: N base statistics

### 5. Repeat Detection

**Function**: `repeat_detection`

Identifies various types of repetitive elements in sequences.

**Parameters**:
- `sequences`: List of sequences to analyze
- `min_repeat_length`: Minimum repeat unit size (default: 10bp)
- `max_repeat_length`: Maximum repeat unit size (default: 100bp)

**Returns**:
- **Tandem repeats**: Perfect and near-perfect repeats
- **Simple sequence repeats (SSRs)**: Microsatellites (1-6bp motifs)
- **Repeat density**: Per-sequence and overall statistics
- **Size distribution**: Repeat length categories

### 6. Gene Prediction and Coding Stats

**Function**: `gene_prediction_and_coding_stats`

Predicts genes using Pyrodigal and calculates coding density statistics.

**Parameters**:
- `sequences`: List of sequences to analyze
- `genetic_code`: NCBI genetic code table (default: 11 for bacteria)
- `meta_mode`: Use metagenomic mode (default: true)

**Returns**:
- **Gene predictions**: Coordinates, strand, partial genes
- **Coding density**: Percentage of genome coding
- **Codon usage**: RSCU values, GC3 content
- **Gene length distribution**: Size categories

### 7. K-mer Analysis

**Function**: `kmer_analysis`

Analyzes k-mer frequencies for sequence composition and complexity assessment.

**Parameters**:
- `sequences`: List of sequences to analyze
- `k_values`: List of k-mer sizes (default: [3,4,5,6])
- `per_sequence`: Analyze each sequence separately (default: false)

**Returns**:
- **K-mer frequencies**: Top abundant k-mers
- **Shannon entropy**: Sequence complexity measure
- **Expected frequencies**: Comparison to random sequences
- **Per-sequence analysis**: Optional individual breakdowns

### 8. Promoter Identification

**Function**: `promoter_identification`

Identifies potential promoter regions using known motif patterns.

**Parameters**:
- `sequences`: List of sequences to analyze
- `upstream_length`: Upstream region length (default: 100bp)

**Motifs Detected**:
- **Prokaryotic**: Pribnow box (TATAAT), -35 box (TTGACA), Shine-Dalgarno
- **Eukaryotic**: TATA box, CAAT box, GC box

**Returns**:
- Motif locations and types
- Promoter density statistics
- Sequence context around motifs

## Usage Examples

### Basic Usage

```python
from biotools import BioToolkit

# Create toolkit
toolkit = BioToolkit()

# Prepare sequences
sequences = [
    {"id": "contig1", "sequence": "ATCGATCG..."},
    {"id": "contig2", "sequence": "GCTAGCTA..."}
]

# Run analyses
assembly_stats = await toolkit.assembly_stats(sequences)
giant_virus_motifs = await toolkit.giant_virus_promoter_search(sequences)
gc_skew = await toolkit.gc_skew_analysis(sequences)
```

### MCP Server Integration

All tools are automatically exposed through the FastMCP server and can be called via the MCP protocol:

```json
{
  "method": "tools/call",
  "params": {
    "name": "giant_virus_promoter_search",
    "arguments": {
      "sequences": [{"id": "seq1", "sequence": "ATCG..."}],
      "upstream_length": 150
    }
  }
}
```

## Research Applications

These tools are particularly useful for:

- **Viral genomics**: Giant virus promoter analysis, genome organization
- **Comparative genomics**: GC skew patterns, codon usage bias
- **Genome assembly**: Quality assessment, repeat content
- **Metagenomics**: Gene prediction in mixed communities
- **Evolutionary biology**: CpG island evolution, promoter conservation

## Dependencies

- **BioPython**: Core sequence analysis
- **NumPy**: Numerical computations
- **Pyrodigal**: Gene prediction
- **Pandas**: Data manipulation (optional)
- **Collections**: Built-in Python module

## Performance Notes

- **Memory usage**: Proportional to sequence length and window sizes
- **Processing time**: GC skew analysis is most compute-intensive for large genomes
- **Parallelization**: Some methods support multi-threading (configurable in NucleicAcidAnalyzer)
- **Optimization**: CpG island detection uses sliding windows for efficiency

## References

1. Yutin N, Koonin EV. Giant viruses: giants of evolution. PMID: 23223451
2. Koonin EV, Yutin N. Origin and evolution of eukaryotic large nucleo-cytoplasmic DNA viruses. PMID: 20452388
3. Lobry JR. Asymmetric substitution patterns in the two DNA strands of bacteria. PMID: 8849441
4. Gardiner-Garden M, Frommer M. CpG islands in vertebrate genomes. PMID: 3656447