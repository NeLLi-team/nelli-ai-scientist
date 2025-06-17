"""
Nucleic Acid Analysis MCP Server using FastMCP
Focused on DNA/RNA sequence analysis including assembly stats, 
promoter detection, GC skew analysis, and specialized giant virus analysis
"""

from fastmcp import FastMCP
import json
import logging
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tools import NucleicAcidAnalyzer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Reduce FastMCP logging noise during startup
logging.getLogger('fastmcp').setLevel(logging.WARNING)
logging.getLogger('mcp').setLevel(logging.WARNING)

# Create FastMCP server
mcp = FastMCP("Nucleic Acid Analysis Tools 🧬")
analyzer = NucleicAcidAnalyzer()

@mcp.tool
async def sequence_stats(sequence: str, sequence_type: str = "dna") -> dict:
    """Calculate comprehensive sequence statistics
    
    Args:
        sequence: DNA or RNA sequence
        sequence_type: Type of sequence (dna, rna)
    """
    return await analyzer.sequence_stats(sequence=sequence, sequence_type=sequence_type)

@mcp.tool
async def validate_nucleic_acid(sequence: str) -> dict:
    """Validate if input is a valid nucleic acid sequence
    
    Args:
        sequence: DNA or RNA sequence to validate
    """
    return analyzer.validate_nucleic_acid(sequence=sequence)

@mcp.tool
async def translate_sequence(sequence: str, genetic_code: int = 1) -> dict:
    """Translate DNA/RNA sequence to protein
    
    Args:
        sequence: DNA or RNA sequence
        genetic_code: NCBI genetic code table number
    """
    return await analyzer.translate_sequence(sequence=sequence, genetic_code=genetic_code)

@mcp.tool
async def read_fasta_file(file_path: str, sample_large_files: bool = True) -> dict:
    """Smart FASTA file reader that handles large files intelligently
    
    Args:
        file_path: Path to the FASTA file
        sample_large_files: Whether to use smart sampling for large files (>50MB)
    """
    return await analyzer.read_fasta_file(file_path=file_path, sample_large_files=sample_large_files)

@mcp.tool
async def write_json_report(data: dict, output_path: str) -> dict:
    """Write analysis results to a JSON report file
    
    Args:
        data: Analysis data to write
        output_path: Path for the output JSON file
    """
    return await analyzer.write_json_report(data=data, output_path=output_path)

@mcp.tool
async def execute_python_analysis(code: str, context_data: dict = None, output_file: str = None) -> dict:
    """Execute Python code for custom analysis with access to analysis results in a sandboxed environment
    
    Args:
        code: Python code to execute (use 'data' variable to access context_data)
        context_data: Optional analysis results or data to make available as 'data' variable
        output_file: Optional path to save code output and plots
    """
    import tempfile
    import sys
    import io
    import os
    from pathlib import Path
    
    try:
        # Create sandbox directory
        sandbox_dir = Path(__file__).parent.parent.parent.parent / "sandbox" / "analysis"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # Import required libraries
        import json
        import numpy as np
        import pandas as pd
        from collections import Counter, defaultdict
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for plots
        import matplotlib.pyplot as plt
        import seaborn as sns
        from Bio.Seq import Seq
        from Bio.SeqUtils import gc_fraction
        import re
        from datetime import datetime
        
        # Set up plot saving directory
        plots_dir = sandbox_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Capture stdout for print statements
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Create safe execution environment with more scientific libraries
        safe_globals = {
            '__builtins__': {
                'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'max': max, 'min': min, 'sum': sum, 'abs': abs, 'round': round,
                'sorted': sorted, 'reversed': reversed, 'any': any, 'all': all,
                'print': print, 'type': type, 'isinstance': isinstance,
            },
            'json': json,
            'np': np,
            'numpy': np,
            'pd': pd,
            'pandas': pd,
            'Counter': Counter,
            'defaultdict': defaultdict,
            'plt': plt,
            'matplotlib': matplotlib,
            'sns': sns,
            'seaborn': sns,
            'Seq': Seq,
            'gc_fraction': gc_fraction,
            're': re,
            'datetime': datetime,
            'data': context_data,  # Make context data available as 'data' variable
            'sandbox_dir': str(sandbox_dir),
            'plots_dir': str(plots_dir),
        }
        
        # Create a local namespace for execution
        local_namespace = {}
        
        # Execute the code
        exec(code, safe_globals, local_namespace)
        
        # Restore stdout and capture output
        sys.stdout = old_stdout
        stdout_content = captured_output.getvalue()
        
        # Save plots if any were created
        saved_plots = []
        if plt.get_fignums():
            for i, fig_num in enumerate(plt.get_fignums()):
                plot_file = plots_dir / f"analysis_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                plt.figure(fig_num)
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                saved_plots.append(str(plot_file))
            plt.close('all')  # Clean up figures
        
        # Save output to file if requested
        if output_file:
            output_path = sandbox_dir / output_file
            with open(output_path, 'w') as f:
                f.write(f"# Python Analysis Output\\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\\n\\n")
                f.write(f"# Code Executed:\\n")
                f.write(f"```python\\n{code}\\n```\\n\\n")
                f.write(f"# Output:\\n{stdout_content}\\n")
                if local_namespace:
                    f.write(f"\\n# Variables Created:\\n")
                    for k, v in local_namespace.items():
                        if not k.startswith('_'):
                            f.write(f"{k}: {str(v)[:200]}{'...' if len(str(v)) > 200 else ''}\\n")
        
        # Extract meaningful variables (not internal ones)
        meaningful_vars = {}
        for k, v in local_namespace.items():
            if not k.startswith('_'):
                try:
                    # Try to serialize to check if it's JSON-compatible
                    json.dumps(v, default=str)
                    meaningful_vars[k] = v
                except:
                    # For non-serializable objects, store string representation
                    meaningful_vars[k] = str(v)[:500] + ("..." if len(str(v)) > 500 else "")
        
        return {
            "success": True,
            "executed_code": code,
            "stdout": stdout_content,
            "local_variables": meaningful_vars,
            "saved_plots": saved_plots,
            "sandbox_dir": str(sandbox_dir),
            "output_file": str(sandbox_dir / output_file) if output_file else None,
            "execution_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Restore stdout in case of error
        sys.stdout = old_stdout
        plt.close('all')  # Clean up any figures
        
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "executed_code": code,
            "sandbox_dir": str(sandbox_dir) if 'sandbox_dir' in locals() else None
        }

@mcp.tool
async def read_analysis_results(file_path: str) -> dict:
    """Read and parse analysis results from a JSON file for detailed interpretation
    
    Args:
        file_path: Path to the JSON analysis results file
    """
    try:
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return {
            "success": True,
            "file_path": file_path,
            "analysis_data": data,
            "data_type": type(data).__name__,
            "keys_available": list(data.keys()) if isinstance(data, dict) else "Not a dictionary"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read analysis results: {str(e)}",
            "file_path": file_path
        }

@mcp.tool
async def analyze_fasta_file(file_path: str, sequence_type: str = "dna") -> dict:
    """Comprehensive analysis of sequences in a FASTA file
    
    Args:
        file_path: Path to the FASTA file
        sequence_type: Type of sequences in the file (dna, rna)
    """
    return await analyzer.analyze_fasta_file(file_path=file_path, sequence_type=sequence_type)

@mcp.tool
async def assembly_stats(sequences) -> dict:
    """Calculate comprehensive assembly statistics including N50, L50, etc.
    
    Args:
        sequences: Either a file path (str) or list of sequences (each with 'id' and 'sequence')
                  For large files (>50MB), automatic sampling will be used
    """
    return await analyzer.assembly_stats(sequences=sequences)

@mcp.tool
async def repeat_detection(sequences, min_repeat_length: int = 10, max_repeat_length: int = 100) -> dict:
    """Detect various types of repeats in sequences
    
    Args:
        sequences: Either a file path (str) or list of sequences (each with 'id' and 'sequence')
                  For large files (>50MB), automatic sampling will be used
        min_repeat_length: Minimum length for tandem repeats
        max_repeat_length: Maximum length for tandem repeats
    """
    return await analyzer.repeat_detection(sequences=sequences, min_repeat_length=min_repeat_length, max_repeat_length=max_repeat_length)

@mcp.tool
async def gene_prediction_and_coding_stats(sequences, genetic_code: int = 11, meta_mode: bool = True) -> dict:
    """Predict genes and calculate coding density using Pyrodigal
    
    Args:
        sequences: Either a file path (str) or list of sequences (each with 'id' and 'sequence')
                  For large files (>50MB), automatic sampling will be used
        genetic_code: NCBI genetic code table number
        meta_mode: Use meta mode for gene prediction (good for mixed genomes)
    """
    return await analyzer.gene_prediction_and_coding_stats(sequences=sequences, genetic_code=genetic_code, meta_mode=meta_mode)

@mcp.tool
async def kmer_analysis(sequences: list, k_values: list = [3, 4, 5, 6], per_sequence: bool = False) -> dict:
    """Analyze k-mer frequencies
    
    Args:
        sequences: List of sequences (each with 'id' and 'sequence')
        k_values: List of k-mer sizes to analyze
        per_sequence: Whether to analyze each sequence separately
    """
    return await analyzer.kmer_analysis(sequences=sequences, k_values=k_values, per_sequence=per_sequence)

@mcp.tool
async def promoter_identification(sequences: list, upstream_length: int = 100) -> dict:
    """Identify potential promoter regions using motif patterns
    
    Args:
        sequences: List of sequences (each with 'id' and 'sequence')
        upstream_length: Length of upstream region to analyze
    """
    return await analyzer.promoter_identification(sequences=sequences, upstream_length=upstream_length)

@mcp.tool
async def giant_virus_promoter_search(sequences: list, upstream_length: int = 150) -> dict:
    """Search for giant virus-specific promoter motifs based on NCLDV research
    
    Args:
        sequences: List of sequences (each with 'id' and 'sequence')
        upstream_length: Length of upstream region to analyze
    """
    return analyzer.giant_virus_promoter_search(sequences=sequences, upstream_length=upstream_length)

@mcp.tool
async def gc_skew_analysis(sequences: list, window_size: int = 10000, step_size: int = 5000) -> dict:
    """Calculate GC skew to identify replication origins and strand bias
    
    Args:
        sequences: List of sequences (each with 'id' and 'sequence')
        window_size: Size of sliding window for GC skew calculation
        step_size: Step size for sliding window
    """
    return await analyzer.gc_skew_analysis(sequences=sequences, window_size=window_size, step_size=step_size)

@mcp.tool
async def cpg_island_detection(sequences: list, min_length: int = 200, gc_threshold: float = 50.0, oe_ratio_threshold: float = 0.6) -> dict:
    """Detect CpG islands in sequences
    
    Args:
        sequences: List of sequences (each with 'id' and 'sequence')
        min_length: Minimum length for CpG islands
        gc_threshold: Minimum GC content percentage
        oe_ratio_threshold: Minimum observed/expected CpG ratio
    """
    return await analyzer.cpg_island_detection(sequences=sequences, min_length=min_length, gc_threshold=gc_threshold, oe_ratio_threshold=oe_ratio_threshold)

# Add some example resources
@mcp.resource("sequences://examples")
def get_example_sequences():
    """Get example sequences for testing"""
    return {
        "dna_example": "ATCGATCGATCGATCGATCG",
        "rna_example": "AUCGAUCGAUCGAUCGAUCG"
    }

@mcp.resource("analysis://help")
def get_analysis_help():
    """Get help information for nucleic acid analysis"""
    return {
        "tools_available": [
            "sequence_stats", "validate_nucleic_acid", "translate_sequence", 
            "assembly_stats", "repeat_detection", "gene_prediction_and_coding_stats",
            "kmer_analysis", "promoter_identification", 
            "giant_virus_promoter_search", "gc_skew_analysis", 
            "cpg_island_detection", "analyze_fasta_file", "read_fasta_file",
            "execute_python_analysis", "read_analysis_results", "write_json_report"
        ],
        "focus": "DNA and RNA sequence analysis",
        "specialties": [
            "Assembly quality metrics (N50, L50)",
            "Giant virus promoter detection",
            "GC skew analysis for replication origins",
            "CpG island identification",
            "Repeat element detection",
            "Gene prediction with Pyrodigal",
            "Custom Python analysis with sandboxed execution",
            "Scientific visualization and plotting",
            "Advanced statistical analysis and reporting"
        ]
    }

if __name__ == "__main__":
    # Run with stdio transport by default
    mcp.run()