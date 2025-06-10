# BioPython MCP Server

A Model Context Protocol (MCP) server that provides bioinformatics tools using BioPython library.

## Features

- **Sequence Analysis**: Calculate statistics, GC content, ORFs, composition
- **BLAST Search**: Local BLAST database searches
- **Multiple Alignment**: ClustalW, MUSCLE, MAFFT algorithms
- **Phylogenetics**: Neighbor-joining, UPGMA, maximum likelihood trees
- **Translation**: DNA/RNA to protein translation with genetic code tables
- **File I/O**: Read/write FASTA files, generate JSON reports

## Tools

### Sequence Analysis
- `sequence_stats(sequence, sequence_type)` - Calculate comprehensive sequence statistics
- `translate_sequence(sequence, genetic_code=1)` - Translate DNA/RNA to protein
- `analyze_fasta_file(file_path, sequence_type="dna")` - Comprehensive FASTA file analysis

### Alignment & Phylogenetics  
- `multiple_alignment(sequences, algorithm="clustalw")` - Multiple sequence alignment
- `phylogenetic_tree(alignment, method="nj")` - Build phylogenetic trees

### Database Search
- `blast_local(sequence, database, program, e_value=0.001)` - Local BLAST search

### File Operations
- `read_fasta_file(file_path)` - Read sequences from FASTA file
- `write_json_report(data, output_path)` - Write analysis results to JSON

## Resources

- `sequences://examples` - Example sequences for testing
- `databases://blast/list` - Available BLAST databases

## Usage

```python
# Start the server
python src/server.py

# Or with FastMCP
from fastmcp import FastMCP
from src.server_fastmcp import mcp
mcp.run()
```

## Files

- `src/server.py` - Original MCP server implementation
- `src/server_fastmcp.py` - FastMCP server implementation  
- `src/biotools.py` - BioPython wrapper functions
- `src/tool_schema.py` - Tool and resource schema definitions
- `src/client.py` - MCP client for testing
- `tests/` - Test suite
- `README.md` - This documentation

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