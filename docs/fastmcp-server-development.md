# FastMCP Server Development Guide

Build powerful, async MCP servers that integrate seamlessly with the NeLLi AI Scientist Agent.

## 🚀 FastMCP vs Traditional MCP

### Why FastMCP?

**Traditional MCP**:
- JSON-RPC protocol overhead
- Complex transport configuration
- Synchronous tool execution
- Manual parameter validation

**FastMCP**:
- **Python-native**: Direct function calls, no JSON-RPC overhead
- **Async-first**: Built on asyncio for concurrent operations
- **Type-safe**: Pydantic parameter validation
- **Simple setup**: No transport configuration needed
- **Better debugging**: Rich error messages and stack traces

## 📦 Basic FastMCP Server

### Minimal Example

```python
# mcps/my_tools/src/server.py
from fastmcp import FastMCP
import asyncio

# Create the MCP server
mcp = FastMCP("My Tools 🛠️")

@mcp.tool
async def hello_world(name: str) -> dict:
    """Say hello to someone
    
    Args:
        name: Name of the person to greet
    """
    return {
        "greeting": f"Hello, {name}!",
        "timestamp": "2024-01-01T12:00:00Z"
    }

@mcp.tool
async def calculate(expression: str) -> dict:
    """Safely evaluate mathematical expressions
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2")
    """
    try:
        # Safe evaluation (restricted to basic math)
        result = eval(expression, {"__builtins__": {}})
        return {
            "expression": expression,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }

if __name__ == "__main__":
    mcp.run()
```

### Register with Agent

```json
// agents/template/mcp_config.json
{
  "mcp_servers": {
    "my_tools": {
      "name": "My Tools",
      "description": "Custom utility tools",
      "fastmcp_script": "../../mcps/my_tools/src/server.py",
      "enabled": true,
      "use_cases": ["utilities", "calculations"]
    }
  }
}
```

## 🧬 Scientific Computing Server

### Bioinformatics Example

```python
# mcps/genomics/src/server.py
from fastmcp import FastMCP
import asyncio
import aiofiles
from Bio.Seq import Seq
from Bio.SeqUtils import GC, molecular_weight
import json

mcp = FastMCP("Genomics Tools 🧬")

@mcp.tool
async def gc_content(sequence: str) -> dict:
    """Calculate GC content of DNA sequence
    
    Args:
        sequence: DNA sequence string
    """
    try:
        seq = Seq(sequence.upper())
        gc_percent = GC(seq)
        
        return {
            "sequence_length": len(sequence),
            "gc_content_percent": round(gc_percent, 2),
            "at_content_percent": round(100 - gc_percent, 2),
            "sequence_type": "DNA"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
async def translate_dna(sequence: str, genetic_code: int = 1) -> dict:
    """Translate DNA sequence to protein
    
    Args:
        sequence: DNA sequence to translate
        genetic_code: NCBI genetic code table number (default: 1 = standard)
    """
    try:
        dna_seq = Seq(sequence.upper())
        
        # Get all 6 reading frames
        frames = []
        for frame in range(3):
            # Forward frames
            protein = dna_seq[frame:].translate(table=genetic_code)
            frames.append({
                "frame": f"+{frame + 1}",
                "protein": str(protein),
                "length": len(protein)
            })
            
            # Reverse frames
            rev_protein = dna_seq.reverse_complement()[frame:].translate(table=genetic_code)
            frames.append({
                "frame": f"-{frame + 1}",
                "protein": str(rev_protein),
                "length": len(rev_protein)
            })
        
        return {
            "dna_sequence": sequence,
            "genetic_code": genetic_code,
            "reading_frames": frames,
            "longest_orf": max(frames, key=lambda x: x["length"])
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
async def analyze_fasta_batch(file_paths: list, analysis_type: str = "basic") -> dict:
    """Analyze multiple FASTA files in parallel
    
    Args:
        file_paths: List of FASTA file paths to analyze
        analysis_type: Type of analysis (basic, detailed, comparative)
    """
    async def analyze_single_file(file_path: str):
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
            
            # Parse FASTA (simplified)
            sequences = []
            current_seq = ""
            current_id = ""
            
            for line in content.split('\\n'):
                if line.startswith('>'):
                    if current_seq:
                        sequences.append({"id": current_id, "sequence": current_seq})
                    current_id = line[1:].strip()
                    current_seq = ""
                else:
                    current_seq += line.strip()
            
            if current_seq:
                sequences.append({"id": current_id, "sequence": current_seq})
            
            # Analyze sequences
            results = []
            for seq_record in sequences:
                seq = seq_record["sequence"]
                analysis = {
                    "id": seq_record["id"],
                    "length": len(seq),
                    "gc_content": GC(Seq(seq)) if seq else 0
                }
                
                if analysis_type == "detailed":
                    analysis.update({
                        "molecular_weight": molecular_weight(Seq(seq)),
                        "a_count": seq.count('A'),
                        "t_count": seq.count('T'),
                        "g_count": seq.count('G'),
                        "c_count": seq.count('C')
                    })
                
                results.append(analysis)
            
            return {
                "file_path": file_path,
                "sequence_count": len(sequences),
                "results": results,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "file_path": file_path,
                "error": str(e),
                "status": "failed"
            }
    
    # Process files in parallel
    tasks = [analyze_single_file(path) for path in file_paths]
    file_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Aggregate results
    successful = [r for r in file_results if isinstance(r, dict) and r.get("status") == "success"]
    failed = [r for r in file_results if isinstance(r, dict) and r.get("status") == "failed"]
    
    return {
        "total_files": len(file_paths),
        "successful_files": len(successful),
        "failed_files": len(failed),
        "results": file_results,
        "summary": {
            "total_sequences": sum(r.get("sequence_count", 0) for r in successful),
            "analysis_type": analysis_type
        }
    }

if __name__ == "__main__":
    mcp.run()
```

## 🔧 Advanced FastMCP Patterns

### Error Handling & Validation

```python
from pydantic import BaseModel, validator
from typing import Optional, List

class SequenceAnalysisParams(BaseModel):
    sequence: str
    analysis_type: str = "basic"
    
    @validator('sequence')
    def validate_dna_sequence(cls, v):
        # Remove whitespace and convert to uppercase
        clean_seq = ''.join(v.split()).upper()
        
        # Check for valid DNA characters
        valid_chars = set('ATCG')
        if not all(c in valid_chars for c in clean_seq):
            invalid_chars = set(clean_seq) - valid_chars
            raise ValueError(f"Invalid DNA characters: {invalid_chars}")
        
        return clean_seq
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = ['basic', 'detailed', 'comparative']
        if v not in valid_types:
            raise ValueError(f"Analysis type must be one of: {valid_types}")
        return v

@mcp.tool
async def robust_sequence_analysis(params: SequenceAnalysisParams) -> dict:
    """Robust sequence analysis with validation
    
    Args:
        params: Validated sequence analysis parameters
    """
    try:
        sequence = params.sequence
        analysis_type = params.analysis_type
        
        # Your analysis logic here
        result = perform_analysis(sequence, analysis_type)
        
        return {
            "success": True,
            "data": result,
            "metadata": {
                "sequence_length": len(sequence),
                "analysis_type": analysis_type
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
```

### Async Database Operations

```python
import aiosqlite
import asyncio

@mcp.tool
async def query_protein_database(protein_id: str, database_path: str = "proteins.db") -> dict:
    """Query protein database asynchronously
    
    Args:
        protein_id: Protein identifier to search for
        database_path: Path to SQLite database file
    """
    try:
        async with aiosqlite.connect(database_path) as db:
            cursor = await db.execute(
                "SELECT * FROM proteins WHERE id = ?", 
                (protein_id,)
            )
            result = await cursor.fetchone()
            
            if result:
                columns = [description[0] for description in cursor.description]
                protein_data = dict(zip(columns, result))
                return {
                    "found": True,
                    "protein": protein_data
                }
            else:
                return {
                    "found": False,
                    "message": f"Protein {protein_id} not found"
                }
                
    except Exception as e:
        return {
            "error": str(e),
            "database_path": database_path
        }

@mcp.tool
async def batch_protein_lookup(protein_ids: List[str], database_path: str = "proteins.db") -> dict:
    """Look up multiple proteins concurrently
    
    Args:
        protein_ids: List of protein IDs to look up
        database_path: Path to SQLite database file
    """
    async def lookup_single(protein_id: str):
        result = await query_protein_database(protein_id, database_path)
        return {
            "protein_id": protein_id,
            **result
        }
    
    # Execute lookups in parallel
    tasks = [lookup_single(pid) for pid in protein_ids]
    results = await asyncio.gather(*tasks)
    
    found_count = sum(1 for r in results if r.get("found", False))
    
    return {
        "total_queries": len(protein_ids),
        "found_count": found_count,
        "not_found_count": len(protein_ids) - found_count,
        "results": results
    }
```

### Resource Management

```python
@mcp.resource("data://sequences/examples")
def get_example_sequences():
    """Provide example sequences for testing"""
    return {
        "dna_examples": {
            "short_sequence": "ATCGATCGATCG",
            "gene_fragment": "ATGAAACGTATTGCATCAGTGGCCAATAA",
            "promoter_region": "TATAATGCGAATTCGAGCTC"
        },
        "protein_examples": {
            "short_peptide": "MVLSPADKTNVKAAW",
            "enzyme_active_site": "DTLHDSFHGFLGPV"
        }
    }

@mcp.resource("databases://reference/info")  
def get_database_info():
    """Information about available databases"""
    return {
        "blast_databases": ["nt", "nr", "refseq_protein"],
        "local_databases": ["custom_genomes.db", "proteins.db"],
        "last_updated": "2024-01-01"
    }
```

## 🔄 Integration Patterns

### Tool Chaining

```python
@mcp.tool
async def comprehensive_gene_analysis(gene_sequence: str, gene_id: str) -> dict:
    """Complete gene analysis pipeline
    
    Args:
        gene_sequence: DNA sequence of the gene
        gene_id: Identifier for the gene
    """
    # Step 1: Basic sequence analysis
    basic_stats = await gc_content(gene_sequence)
    
    # Step 2: Translation analysis
    translation_result = await translate_dna(gene_sequence)
    
    # Step 3: Database lookup (if available)
    try:
        db_result = await query_protein_database(gene_id)
    except:
        db_result = {"found": False, "note": "Database unavailable"}
    
    # Step 4: Compile comprehensive report
    return {
        "gene_id": gene_id,
        "sequence_analysis": basic_stats,
        "translation_analysis": translation_result,
        "database_match": db_result,
        "analysis_timestamp": "2024-01-01T12:00:00Z",
        "pipeline_version": "1.0"
    }
```

### External API Integration

```python
import aiohttp

@mcp.tool
async def fetch_uniprot_data(protein_id: str) -> dict:
    """Fetch protein data from UniProt API
    
    Args:
        protein_id: UniProt protein identifier
    """
    url = f"https://www.uniprot.org/uniprot/{protein_id}.json"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "protein_data": data,
                        "source": "UniProt"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}",
                        "protein_id": protein_id
                    }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "protein_id": protein_id
        }
```

## 📊 Tool Schemas & Documentation

### Comprehensive Schema Definition

```python
# mcps/my_domain/src/tool_schema.py
def get_tool_schemas():
    return {
        "comprehensive_gene_analysis": {
            "description": "Complete gene analysis including sequence stats, translation, and database lookup",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "gene_sequence": {
                        "type": "string",
                        "description": "DNA sequence of the gene (ATCG format)",
                        "pattern": "^[ATCG]+$",
                        "minLength": 1,
                        "maxLength": 50000
                    },
                    "gene_id": {
                        "type": "string", 
                        "description": "Unique identifier for the gene",
                        "pattern": "^[A-Za-z0-9_-]+$"
                    }
                },
                "required": ["gene_sequence", "gene_id"]
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "gene_id": {"type": "string"},
                    "sequence_analysis": {
                        "type": "object",
                        "properties": {
                            "length": {"type": "integer"},
                            "gc_content_percent": {"type": "number"}
                        }
                    },
                    "translation_analysis": {
                        "type": "object",
                        "properties": {
                            "reading_frames": {"type": "array"},
                            "longest_orf": {"type": "object"}
                        }
                    }
                }
            }
        }
    }
```

## 🧪 Testing FastMCP Servers

### Unit Testing

```python
# tests/test_genomics_server.py
import pytest
import asyncio
from mcps.genomics.src.server import mcp

@pytest.mark.asyncio
async def test_gc_content():
    """Test GC content calculation"""
    # Mock the FastMCP client behavior
    result = await mcp.tool_registry["gc_content"]("ATCGATCG")
    
    assert result["sequence_length"] == 8
    assert result["gc_content_percent"] == 50.0
    assert result["at_content_percent"] == 50.0

@pytest.mark.asyncio  
async def test_translate_dna():
    """Test DNA translation"""
    result = await mcp.tool_registry["translate_dna"]("ATGAAATAG")
    
    assert len(result["reading_frames"]) == 6
    assert result["genetic_code"] == 1
    assert "protein" in result["reading_frames"][0]

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for invalid sequences"""
    result = await mcp.tool_registry["gc_content"]("INVALID")
    
    assert "error" in result
```

### Integration Testing

```python
# tests/test_integration.py  
import pytest
from fastmcp import Client

@pytest.mark.asyncio
async def test_server_integration():
    """Test full server integration"""
    script_path = "mcps/genomics/src/server.py"
    
    async with Client(script_path) as client:
        # Test tool discovery
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]
        
        assert "gc_content" in tool_names
        assert "translate_dna" in tool_names
        
        # Test tool execution
        result = await client.call_tool("gc_content", {"sequence": "ATCG"})
        assert result.text  # FastMCP returns TextContent
```

## 🚀 Performance Optimization

### Async Patterns

```python
# Good: Concurrent processing
@mcp.tool
async def process_multiple_sequences(sequences: List[str]) -> dict:
    async def process_one(seq):
        return await gc_content(seq)
    
    tasks = [process_one(seq) for seq in sequences]
    results = await asyncio.gather(*tasks)
    return {"results": results}

# Bad: Sequential processing  
@mcp.tool
async def process_multiple_sequences_slow(sequences: List[str]) -> dict:
    results = []
    for seq in sequences:
        result = await gc_content(seq)  # Blocks other sequences
        results.append(result)
    return {"results": results}
```

### Caching

```python
from functools import lru_cache
import asyncio

# Cache for expensive computations
@lru_cache(maxsize=1000)
def _compute_gc_content(sequence: str) -> float:
    """Cached GC content computation"""
    return GC(Seq(sequence))

@mcp.tool
async def cached_gc_content(sequence: str) -> dict:
    """GC content with caching"""
    # Offload CPU-intensive work to thread pool
    loop = asyncio.get_event_loop()
    gc_percent = await loop.run_in_executor(
        None, _compute_gc_content, sequence
    )
    
    return {
        "sequence_length": len(sequence),
        "gc_content_percent": round(gc_percent, 2)
    }
```

## 💡 Best Practices

### 1. Tool Design
- **Single responsibility**: Each tool should do one thing well
- **Async by default**: Use async/await for all I/O operations
- **Comprehensive errors**: Return detailed error information
- **Type hints**: Use proper type annotations

### 2. Parameter Validation
- **Use Pydantic models**: For complex parameter validation
- **Provide defaults**: Make tools easy to use
- **Clear descriptions**: Help the LLM understand tool purpose

### 3. Return Values
- **Consistent structure**: Use similar return format across tools
- **Rich metadata**: Include context and provenance information
- **Success indicators**: Clear success/failure status

### 4. Performance
- **Batch operations**: Provide batch variants for better efficiency
- **Concurrent execution**: Use asyncio.gather() for parallel processing
- **Resource management**: Use async context managers properly

### 5. Documentation
- **Tool schemas**: Define input/output schemas
- **Usage examples**: Provide clear examples in docstrings
- **Error scenarios**: Document expected error conditions

FastMCP enables you to build powerful, efficient MCP servers that integrate seamlessly with the NeLLi AI Scientist Agent's universal architecture. The async-first design and Python-native approach make it ideal for scientific computing workloads.