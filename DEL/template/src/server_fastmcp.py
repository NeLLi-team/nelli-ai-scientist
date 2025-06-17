"""
BioPython MCP Server using FastMCP
"""

from fastmcp import FastMCP
import json
import logging
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from biotools import BioToolkit
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
mcp = FastMCP("BioPython Tools 🧬")
toolkit = BioToolkit()

@mcp.tool
async def sequence_stats(sequence: str, sequence_type: str) -> dict:
    """Calculate comprehensive sequence statistics
    
    Args:
        sequence: DNA, RNA, or protein sequence
        sequence_type: Type of sequence (dna, rna, protein)
    """
    return await toolkit.sequence_stats(sequence=sequence, sequence_type=sequence_type)

@mcp.tool  
async def blast_local(sequence: str, database: str, program: str, e_value: float = 0.001) -> dict:
    """Run local BLAST search against a database
    
    Args:
        sequence: Query sequence
        database: Database name or path
        program: BLAST program to use (blastn, blastp, blastx, tblastn, tblastx)
        e_value: E-value threshold
    """
    return await toolkit.blast_local(
        sequence=sequence, 
        database=database, 
        program=program, 
        e_value=e_value
    )

@mcp.tool
async def multiple_alignment(sequences: list, algorithm: str = "clustalw") -> dict:
    """Perform multiple sequence alignment
    
    Args:
        sequences: List of sequences to align (each with 'id' and 'sequence')
        algorithm: Alignment algorithm (clustalw, muscle, mafft)
    """
    return await toolkit.multiple_alignment(sequences=sequences, algorithm=algorithm)

@mcp.tool
async def phylogenetic_tree(alignment: str, method: str = "nj") -> dict:
    """Build phylogenetic tree from sequences
    
    Args:
        alignment: Multiple sequence alignment in FASTA format
        method: Tree building method (nj, upgma, maximum_likelihood)
    """
    return await toolkit.phylogenetic_tree(alignment=alignment, method=method)

@mcp.tool
async def translate_sequence(sequence: str, genetic_code: int = 1) -> dict:
    """Translate DNA/RNA sequence to protein
    
    Args:
        sequence: DNA or RNA sequence
        genetic_code: NCBI genetic code table number
    """
    return await toolkit.translate_sequence(sequence=sequence, genetic_code=genetic_code)

@mcp.tool
async def read_fasta_file(file_path: str) -> dict:
    """Read sequences from a FASTA file
    
    Args:
        file_path: Path to the FASTA file
    """
    return await toolkit.read_fasta_file(file_path=file_path)

@mcp.tool
async def write_json_report(data: dict, output_path: str) -> dict:
    """Write analysis results to a JSON report file
    
    Args:
        data: Analysis data to write
        output_path: Path for the output JSON file
    """
    return await toolkit.write_json_report(data=data, output_path=output_path)

@mcp.tool
async def analyze_fasta_file(file_path: str, sequence_type: str = "dna") -> dict:
    """Comprehensive analysis of sequences in a FASTA file
    
    Args:
        file_path: Path to the FASTA file
        sequence_type: Type of sequences in the file (dna, rna, protein)
    """
    return await toolkit.analyze_fasta_file(file_path=file_path, sequence_type=sequence_type)

# Add some example resources
@mcp.resource("sequences://examples")
def get_example_sequences():
    """Get example sequences for testing"""
    return {
        "dna_example": "ATCGATCGATCGATCGATCG",
        "protein_example": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG"
    }

@mcp.resource("databases://blast/list")
def list_blast_databases():
    """List available BLAST databases"""
    return {
        "databases": ["nt", "nr", "refseq_protein", "refseq_rna"],
        "local_databases": []
    }

if __name__ == "__main__":
    # Run with stdio transport by default
    mcp.run()