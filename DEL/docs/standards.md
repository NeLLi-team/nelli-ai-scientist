# Standards and Protocols

## Communication Standards

### FIPA-ACL Message Format
All agents must use FIPA-ACL protocol with the following message structure:

```json
{
  "performative": "request|inform|query|confirm|failure|not_understood",
  "sender": "agent-id",
  "receiver": "target-agent-id", 
  "content": {},
  "conversation_id": "uuid",
  "reply_with": "message-id",
  "in_reply_to": "original-message-id",
  "language": "json"
}
```

#### Performative Types
- **REQUEST**: Ask agent to perform action
- **QUERY**: Ask for information
- **INFORM**: Provide information
- **CONFIRM**: Acknowledge receipt
- **FAILURE**: Report execution failure
- **NOT_UNDERSTOOD**: Cannot process message

### Agent Tool Registration
All agent tools must be registered with proper type hints:

```python
@self.tools.register("tool_name")
async def tool_function(param1: str, param2: int) -> Dict[str, Any]:
    """Clear description of what the tool does"""
    # Implementation
    return {"result": "data"}
```

## MCP Server Standards

### Tool Definition Schema
All MCP tools must follow this schema:

```json
{
  "name": "tool_name",
  "description": "Clear, actionable description",
  "inputSchema": {
    "type": "object",
    "properties": {
      "param_name": {
        "type": "string|number|boolean|object|array",
        "description": "Parameter description",
        "enum": ["option1", "option2"]  // Optional
      }
    },
    "required": ["param1", "param2"],
    "additionalProperties": false
  }
}
```

### Server Implementation Standards
```python
@self.server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a tool with proper error handling"""
    try:
        if name == "tool_name":
            result = await self.toolkit.tool_function(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
            
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return [TextContent(
            type="text", 
            text=json.dumps({"error": str(e), "tool": name})
        )]
```

## File Processing Standards

### FASTA File Handling
```python
# Standard FASTA reading function signature
async def read_fasta_file(file_path: str) -> Dict[str, Any]:
    """Standard FASTA file reader"""
    return {
        "file_path": file_path,
        "num_sequences": int,
        "sequences": [
            {
                "id": str,
                "description": str, 
                "sequence": str,
                "length": int
            }
        ]
    }
```

### JSON Report Format
All analysis reports must include metadata:

```json
{
  "metadata": {
    "generated_by": "NeLLi AI Scientist [Agent|MCP Server]",
    "agent_id": "agent-identifier",
    "timestamp": "2025-06-09T22:05:37.525887",
    "version": "1.0.0"
  },
  "analysis_results": {
    // Actual analysis data
  }
}
```

## Code Standards

### Python Requirements
- **Python 3.11+** (managed by pixi)
- **Type hints required** for all functions
- **Black formatting** (line length 88)
- **Ruff linting** with default settings
- **80% test coverage minimum**
- **Docstrings** for all public functions

### Import Organization
```python
# Standard library imports
import asyncio
import json
from typing import Dict, Any, List

# Third-party imports  
from Bio import SeqIO
from pydantic import BaseModel

# Local imports
from .llm_interface import LLMInterface
from .tools import ToolRegistry
```

### Error Handling
```python
try:
    result = await operation()
    return {"status": "success", "data": result}
except FileNotFoundError:
    return {"error": f"File not found: {file_path}"}
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return {"error": f"Operation failed: {str(e)}"}
```

## Integration Standards

### Health Check Implementation
All components must implement health checks:

```python
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "dependencies": {
            "llm_api": "connected",
            "database": "connected"
        }
    }
```

### Logging Standards
Use structured JSON logging:

```python
import logging
import json

logger = logging.getLogger(__name__)

# Log structured data
logger.info("Tool executed", extra={
    "tool_name": tool_name,
    "execution_time": execution_time,
    "success": True
})
```

### Retry Logic
Implement exponential backoff for external APIs:

```python
import asyncio
from typing import TypeVar, Callable

T = TypeVar('T')

async def retry_with_backoff(
    func: Callable[[], T], 
    max_retries: int = 3,
    base_delay: float = 1.0
) -> T:
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)
```

## Bioinformatics Data Standards

### Supported Formats
- **Sequences**: FASTA, FASTQ, GenBank
- **Alignments**: Clustal, FASTA, PHYLIP
- **Trees**: Newick, Nexus
- **Variants**: VCF, BCF
- **Annotations**: GFF3, GTF
- **Structures**: PDB, mmCIF

### Sequence Type Detection
```python
def detect_sequence_type(sequence: str) -> str:
    """Detect if sequence is DNA, RNA, or protein"""
    sequence = sequence.upper().replace('U', 'T')
    dna_chars = set('ATCG')
    protein_chars = set('ACDEFGHIKLMNPQRSTVWY')
    
    if set(sequence) <= dna_chars:
        return "dna"
    elif set(sequence) <= protein_chars:
        return "protein"
    else:
        return "unknown"
```

### Quality Score Handling
```python
def parse_quality_scores(quality_string: str, format: str = "phred33") -> List[int]:
    """Parse FASTQ quality scores"""
    if format == "phred33":
        return [ord(char) - 33 for char in quality_string]
    elif format == "phred64":
        return [ord(char) - 64 for char in quality_string]
    else:
        raise ValueError(f"Unknown quality format: {format}")
```

## Development Workflow with Pixi

### Dependency Management
```bash
# Add new dependencies
pixi add biopython requests pandas

# Add development dependencies  
pixi add --feature dev pytest black ruff mypy

# Update dependencies
pixi update

# Install all dependencies
pixi install
```

### Available Tasks
```bash
# Testing
pixi run test              # Run all tests with coverage
pixi run agent-test        # Test agent template
pixi run mcp-test         # Test MCP template

# Code Quality
pixi run format           # Format with black and ruff
pixi run lint            # Check style and types
pixi run typecheck       # Type checking only

# Development
pixi run agent-run       # Run agent template
pixi run mcp-run        # Run MCP server
```

### Environment Configuration
All configuration through environment variables:

```bash
# .env file structure
CBORG_API_KEY="your-api-key"
CBORG_BASE_URL="https://api.cborg.lbl.gov"
CBORG_MODEL="google/gemini-flash-lite"

# Optional: Override for specific components
AGENT_LOG_LEVEL="INFO"
MCP_SERVER_PORT="8000"
```

## Performance Standards

### Response Time Requirements
- **Agent tool execution**: < 5 seconds
- **MCP server tool calls**: < 10 seconds  
- **File processing**: < 30 seconds for files < 100MB
- **Health checks**: < 1 second

### Memory Usage
- **Agent instances**: < 512MB RAM
- **MCP servers**: < 1GB RAM
- **Large file processing**: Stream processing for files > 100MB

### Concurrent Operations
- **Agent**: Handle 10 concurrent requests
- **MCP Server**: Handle 50 concurrent tool calls
- **File I/O**: Maximum 5 concurrent file operations

## Security Standards

### API Key Management
- Store API keys in environment variables only
- Never commit API keys to version control
- Rotate API keys regularly
- Use different keys for development/production

### Input Validation
```python
from pydantic import BaseModel, validator

class SequenceInput(BaseModel):
    sequence: str
    sequence_type: str
    
    @validator('sequence')
    def validate_sequence(cls, v):
        if len(v) > 10_000_000:  # 10MB limit
            raise ValueError("Sequence too long")
        return v.upper()
    
    @validator('sequence_type') 
    def validate_type(cls, v):
        if v not in ['dna', 'rna', 'protein']:
            raise ValueError("Invalid sequence type")
        return v
```

### File Access
- Validate all file paths
- Restrict access to designated directories
- Check file sizes before processing
- Use temporary directories for intermediate files

```python
import os
from pathlib import Path

def validate_file_path(file_path: str, allowed_dirs: List[str]) -> bool:
    """Validate file path is within allowed directories"""
    abs_path = Path(file_path).resolve()
    return any(
        abs_path.is_relative_to(Path(allowed_dir).resolve()) 
        for allowed_dir in allowed_dirs
    )
```