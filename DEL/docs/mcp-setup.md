# MCP Server Setup Guide

This guide covers how to set up multiple MCP (Model Context Protocol) servers, configure LLM clients to discover them, and implement proper tool schemas.

## Table of Contents
- [MCP Overview](#mcp-overview)
- [Single MCP Server Setup](#single-mcp-server-setup)
- [Multiple MCP Servers](#multiple-mcp-servers)
- [Claude Desktop Configuration](#claude-desktop-configuration)
- [Custom LLM Client Configuration](#custom-llm-client-configuration)
- [Tool Schema Design](#tool-schema-design)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## MCP Overview

The Model Context Protocol (MCP) enables LLMs to securely connect to external tools and data sources. Key concepts:

- **MCP Server**: Exposes tools/resources to LLMs
- **MCP Client**: LLM application that connects to servers
- **Transport**: Communication method (stdio, HTTP, WebSocket)
- **Tools**: Functions the LLM can call
- **Resources**: Data sources the LLM can read

## Single MCP Server Setup

### 1. Basic Server Implementation

```python
# mcps/my-server/src/server.py
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent
import asyncio

class MyMCPServer:
    def __init__(self, name: str = "my-mcp-server"):
        self.server = Server(name)
        self._register_handlers()
    
    def _register_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="my_tool",
                    description="Description of what this tool does",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "input_param": {
                                "type": "string",
                                "description": "Parameter description"
                            }
                        },
                        "required": ["input_param"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "my_tool":
                result = await self.my_tool_function(**arguments)
                return [TextContent(type="text", text=json.dumps(result))]
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def run(self, transport: str = "stdio"):
        if transport == "stdio":
            from mcp.server.stdio import stdio_server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="my-mcp-server",
                        server_version="1.0.0"
                    )
                )
```

### 2. Package Structure

```
mcps/my-server/
├── pyproject.toml          # Package configuration
├── src/
│   ├── __init__.py
│   ├── server.py          # Main server implementation
│   ├── tools.py           # Tool implementations
│   └── models.py          # Data models
├── tests/
│   └── test_server.py
└── README.md
```

### 3. Package Configuration (pyproject.toml)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-mcp-server"
version = "0.1.0"
description = "My custom MCP server"
dependencies = [
    "mcp>=0.1.0",
    "biopython>=1.80",
]

[project.scripts]
my-mcp-server = "src.server:main"
```

## Multiple MCP Servers

### Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   LLM Client    │    │   LLM Client     │    │   LLM Client     │
│  (Claude App)   │    │  (Custom App)    │    │   (Terminal)     │
└─────────┬───────┘    └────────┬─────────┘    └────────┬─────────┘
          │                     │                       │
          └──────────┬──────────┴─────────┬─────────────┘
                     │                    │
         ┌───────────▼──────────┐    ┌────▼────────┐
         │    MCP Server 1      │    │ MCP Server 2 │
         │   (BioPython)        │    │ (Literature) │
         └──────────────────────┘    └──────────────┘
```

### Configuration for Multiple Servers

The LLM client needs to know about all available MCP servers through configuration files.

## Claude Desktop Configuration

### Location
Place the configuration file at:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

## Adding Existing MCP Servers

### Quick Setup with Context7 (Vector Database)

Context7 is a ready-to-use MCP server that provides vector database capabilities for AI applications. Perfect for semantic search and knowledge retrieval.

#### 1. Add Context7 to Claude Desktop

Edit your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "Context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```

#### 2. Test Context7 Integration

After restarting Claude Desktop, you should see Context7 tools available. Try these commands in Claude:

**Test vector storage:**
```
Can you help me store some text in the vector database? Store this information: "Bioinformatics is the application of computational techniques to analyze biological data"
```

**Test semantic search:**
```
Search the vector database for content related to "computational biology"
```

**Test data management:**
```
List all stored vectors and show me what's in the database
```

### Adding Multiple MCP Servers

Combine our bioinformatics server with Context7 and other servers:

```json
{
  "mcpServers": {
    "biopython-server": {
      "command": "pixi",
      "args": ["run", "python", "-m", "mcps.template.src.server"],
      "cwd": "/path/to/nelli-ai-scientist"
    },
    "Context7": {
      "command": "npx", 
      "args": ["-y", "@upstash/context7-mcp"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-brave-api-key"
      }
    }
  }
}
```

### Popular Existing MCP Servers

| Server | Package | Purpose |
|--------|---------|---------|
| **Context7** | `@upstash/context7-mcp` | Vector database for semantic search |
| **Filesystem** | `@modelcontextprotocol/server-filesystem` | File system operations |
| **Brave Search** | `@modelcontextprotocol/server-brave-search` | Web search capabilities |
| **GitHub** | `@modelcontextprotocol/server-github` | GitHub repository access |
| **PostgreSQL** | `@modelcontextprotocol/server-postgres` | Database operations |
| **Puppeteer** | `@modelcontextprotocol/server-puppeteer` | Web automation |
| **SQLite** | `@modelcontextprotocol/server-sqlite` | SQLite database access |

### Installing Node.js MCP Servers

Most existing MCP servers are Node.js packages. Ensure you have Node.js installed:

```bash
# Check if Node.js is installed
node --version
npm --version

# If not installed, install Node.js
# On macOS with Homebrew:
brew install node

# On Ubuntu/Debian:
sudo apt update && sudo apt install nodejs npm

# On Windows: Download from nodejs.org
```

### Configuration Format

```json
{
  "mcpServers": {
    "biopython-server": {
      "command": "python",
      "args": ["-m", "mcps.template.src.server"],
      "cwd": "/path/to/nelli-ai-scientist",
      "env": {
        "PYTHONPATH": "/path/to/nelli-ai-scientist"
      }
    },
    "literature-server": {
      "command": "python", 
      "args": ["-m", "mcps.literature.src.server"],
      "cwd": "/path/to/nelli-ai-scientist",
      "env": {
        "PUBMED_API_KEY": "your-api-key"
      }
    },
    "genomics-tools": {
      "command": "pixi",
      "args": ["run", "python", "-m", "mcps.genomics.src.server"],
      "cwd": "/path/to/nelli-ai-scientist"
    },
    "web-search": {
      "command": "node",
      "args": ["mcps/web-search/index.js"],
      "cwd": "/path/to/nelli-ai-scientist"
    },
    "file-operations": {
      "command": "python",
      "args": ["-m", "mcps.file_ops.src.server"],
      "cwd": "/path/to/nelli-ai-scientist",
      "env": {
        "ALLOWED_PATHS": "/home/user/data:/tmp"
      }
    }
  },
  "globalShortcut": "Ctrl+Shift+M"
}
```

### Server-Specific Configurations

#### BioPython Server
```json
"biopython-server": {
  "command": "pixi",
  "args": ["run", "python", "-m", "mcps.template.src.server"],
  "cwd": "/path/to/nelli-ai-scientist",
  "env": {
    "BLAST_DB_PATH": "/data/blast",
    "MAX_SEQUENCE_LENGTH": "1000000"
  }
}
```

#### Literature Search Server
```json
"literature-server": {
  "command": "python",
  "args": ["-m", "mcps.literature.src.server"],
  "cwd": "/path/to/nelli-ai-scientist", 
  "env": {
    "PUBMED_API_KEY": "your-pubmed-key",
    "ARXIV_API_KEY": "your-arxiv-key",
    "RATE_LIMIT": "10"
  }
}
```

#### Database Server
```json
"database-server": {
  "command": "python",
  "args": ["-m", "mcps.database.src.server"],
  "cwd": "/path/to/nelli-ai-scientist",
  "env": {
    "DATABASE_URL": "postgresql://user:pass@localhost/biodata",
    "REDIS_URL": "redis://localhost:6379"
  }
}
```

## Custom LLM Client Configuration

### Python Client Setup

```python
# config/mcp_config.py
from typing import Dict, List
import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

class MCPClientManager:
    def __init__(self):
        self.servers: Dict[str, ClientSession] = {}
        self.server_configs = self._load_server_configs()
    
    def _load_server_configs(self) -> Dict[str, Dict]:
        """Load MCP server configurations"""
        return {
            "biopython": {
                "command": ["python", "-m", "mcps.template.src.server"],
                "cwd": "/path/to/nelli-ai-scientist"
            },
            "literature": {
                "command": ["python", "-m", "mcps.literature.src.server"],
                "cwd": "/path/to/nelli-ai-scientist"
            }
        }
    
    async def connect_to_servers(self):
        """Connect to all configured MCP servers"""
        for server_name, config in self.server_configs.items():
            try:
                read, write = await stdio_client(
                    command=config["command"],
                    cwd=config.get("cwd")
                )
                session = ClientSession(read, write)
                await session.initialize()
                self.servers[server_name] = session
                print(f"Connected to {server_name}")
            except Exception as e:
                print(f"Failed to connect to {server_name}: {e}")
    
    async def list_all_tools(self) -> Dict[str, List]:
        """Get tools from all connected servers"""
        all_tools = {}
        for server_name, session in self.servers.items():
            try:
                tools = await session.list_tools()
                all_tools[server_name] = tools
            except Exception as e:
                print(f"Error listing tools from {server_name}: {e}")
        return all_tools
    
    async def call_tool(self, server_name: str, tool_name: str, **kwargs):
        """Call a specific tool on a specific server"""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not connected")
        
        session = self.servers[server_name]
        return await session.call_tool(tool_name, kwargs)

# Usage example
async def main():
    manager = MCPClientManager()
    await manager.connect_to_servers()
    
    # List all available tools
    tools = await manager.list_all_tools()
    print("Available tools:", tools)
    
    # Call a specific tool
    result = await manager.call_tool(
        "biopython", 
        "sequence_stats", 
        sequence="ATCGATCG",
        sequence_type="dna"
    )
    print("Result:", result)
```

### Agent Integration

```python
# agents/your-agent/src/mcp_integration.py
from typing import Dict, Any, List
import asyncio
from .mcp_client import MCPClientManager

class MCPIntegratedAgent:
    def __init__(self, config):
        self.config = config
        self.mcp_manager = MCPClientManager()
        self.available_tools = {}
    
    async def initialize(self):
        """Initialize agent and connect to MCP servers"""
        await self.mcp_manager.connect_to_servers()
        self.available_tools = await self.mcp_manager.list_all_tools()
        self._register_mcp_tools()
    
    def _register_mcp_tools(self):
        """Register MCP tools as agent tools"""
        for server_name, tools in self.available_tools.items():
            for tool in tools:
                tool_id = f"{server_name}_{tool.name}"
                
                # Create a wrapper function for each MCP tool
                async def mcp_tool_wrapper(**kwargs):
                    return await self.mcp_manager.call_tool(
                        server_name, tool.name, **kwargs
                    )
                
                # Register with agent's tool registry
                self.tools.register(tool_id)(mcp_tool_wrapper)
```

## Tool Schema Design

### Schema Validation

```python
# mcps/template/src/schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Literal
from enum import Enum

class SequenceType(str, Enum):
    DNA = "dna"
    RNA = "rna" 
    PROTEIN = "protein"

class SequenceStatsInput(BaseModel):
    sequence: str = Field(..., description="Biological sequence")
    sequence_type: SequenceType = Field(..., description="Type of sequence")
    
    @validator('sequence')
    def validate_sequence(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Sequence cannot be empty")
        if len(v) > 1_000_000:
            raise ValueError("Sequence too long (max 1M characters)")
        return v.upper().strip()

class BlastSearchInput(BaseModel):
    sequence: str = Field(..., description="Query sequence") 
    database: str = Field(..., description="Database to search")
    program: Literal["blastn", "blastp", "blastx", "tblastn", "tblastx"]
    e_value: float = Field(0.001, description="E-value threshold", ge=0, le=1)
    max_hits: int = Field(100, description="Maximum hits to return", ge=1, le=1000)

class FileProcessingInput(BaseModel):
    file_path: str = Field(..., description="Path to input file")
    file_type: Optional[str] = Field(None, description="File format override")
    
    @validator('file_path')
    def validate_path(cls, v):
        # Security: validate file path
        allowed_extensions = {'.fasta', '.fastq', '.fa', '.fq', '.gb', '.gbk'}
        if not any(v.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(f"Unsupported file type. Allowed: {allowed_extensions}")
        return v
```

### Tool Registration with Schemas

```python
# mcps/template/src/server.py
from .schemas import SequenceStatsInput, BlastSearchInput

def _register_handlers(self):
    @self.server.list_tools()
    async def handle_list_tools() -> List[Tool]:
        tools = []
        
        # Tool with Pydantic schema
        tools.append(Tool(
            name="sequence_stats",
            description="Calculate comprehensive sequence statistics including GC content, composition, and ORF analysis",
            inputSchema=SequenceStatsInput.model_json_schema()
        ))
        
        # Tool with manual schema
        tools.append(Tool(
            name="read_fasta_file",
            description="Read and parse sequences from a FASTA format file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the FASTA file",
                        "pattern": r"^/.*\.(fasta|fa|fas)$"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        ))
        
        return tools
    
    @self.server.call_tool()
    async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            # Validate inputs using Pydantic
            if name == "sequence_stats":
                validated_input = SequenceStatsInput(**arguments)
                result = await self.toolkit.sequence_stats(
                    validated_input.sequence,
                    validated_input.sequence_type
                )
            elif name == "blast_search":
                validated_input = BlastSearchInput(**arguments)
                result = await self.toolkit.blast_local(
                    sequence=validated_input.sequence,
                    database=validated_input.database,
                    program=validated_input.program,
                    e_value=validated_input.e_value
                )
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except ValidationError as e:
            error_details = {
                "error": "Input validation failed",
                "details": e.errors(),
                "tool": name
            }
            return [TextContent(type="text", text=json.dumps(error_details))]
```

## Advanced Features

### Resource Management

```python
# mcps/template/src/resources.py
@self.server.list_resources()
async def handle_list_resources() -> List[Resource]:
    return [
        Resource(
            uri="sequences://examples",
            name="Example Sequences",
            description="Collection of example biological sequences",
            mimeType="application/json"
        ),
        Resource(
            uri="databases://blast/nr",
            name="NCBI Non-Redundant Database",
            description="NCBI nr protein database",
            mimeType="application/json"
        )
    ]

@self.server.read_resource()
async def handle_read_resource(uri: str) -> str:
    if uri == "sequences://examples":
        examples = {
            "dna_examples": [
                {
                    "id": "example_1",
                    "sequence": "ATCGATCGATCGATCG",
                    "description": "Simple DNA sequence"
                }
            ],
            "protein_examples": [
                {
                    "id": "insulin",
                    "sequence": "GIVEQCCTSICSLYQLENYCN",
                    "description": "Human insulin B chain"
                }
            ]
        }
        return json.dumps(examples, indent=2)
    
    elif uri.startswith("databases://blast/"):
        db_name = uri.split("/")[-1]
        db_info = await self.get_database_info(db_name)
        return json.dumps(db_info)
    
    else:
        raise ValueError(f"Unknown resource: {uri}")
```

### Progress Reporting

```python
# mcps/template/src/progress.py
from mcp.types import Progress

@self.server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    if name == "large_analysis":
        # Report progress for long-running operations
        progress_token = "analysis_" + str(uuid.uuid4())
        
        # Start progress
        await self.server.send_progress_notification(
            progress_token, 
            Progress(progress=0, total=100)
        )
        
        try:
            result = await self.run_large_analysis(
                arguments,
                progress_callback=lambda p: self.server.send_progress_notification(
                    progress_token,
                    Progress(progress=p, total=100)
                )
            )
            
            # Complete progress
            await self.server.send_progress_notification(
                progress_token,
                Progress(progress=100, total=100)
            )
            
            return [TextContent(type="text", text=json.dumps(result))]
            
        except Exception as e:
            # Error in progress
            await self.server.send_progress_notification(
                progress_token,
                Progress(progress=-1, total=100)  # -1 indicates error
            )
            raise
```

### Server-to-Server Communication

```python
# mcps/orchestrator/src/server.py
class OrchestrationServer:
    def __init__(self):
        self.downstream_servers = {}
    
    async def initialize(self):
        """Connect to downstream MCP servers"""
        configs = {
            "biopython": ["python", "-m", "mcps.template.src.server"],
            "literature": ["python", "-m", "mcps.literature.src.server"]
        }
        
        for name, command in configs.items():
            read, write = await stdio_client(command)
            session = ClientSession(read, write)
            await session.initialize()
            self.downstream_servers[name] = session
    
    async def complex_workflow(self, sequence: str) -> Dict[str, Any]:
        """Orchestrate complex workflow across multiple servers"""
        results = {}
        
        # Step 1: Analyze sequence
        bio_result = await self.downstream_servers["biopython"].call_tool(
            "sequence_stats", 
            {"sequence": sequence, "sequence_type": "dna"}
        )
        results["sequence_analysis"] = bio_result
        
        # Step 2: Search literature based on sequence characteristics
        if bio_result["gc_content"] > 60:
            lit_result = await self.downstream_servers["literature"].call_tool(
                "search_papers",
                {"query": "high GC content bacteria genome"}
            )
            results["literature"] = lit_result
        
        return results
```

## Troubleshooting

### Common Issues

#### 1. Server Not Found
```bash
# Check if server executable is in PATH
which python
which pixi

# Test server manually
python -m mcps.template.src.server

# Check working directory
pwd
ls mcps/template/src/server.py
```

#### 2. Import Errors
```python
# Add to server startup
import sys
sys.path.append("/path/to/project")

# Or use PYTHONPATH in config
"env": {
    "PYTHONPATH": "/path/to/nelli-ai-scientist"
}
```

#### 3. Tool Schema Validation
```python
# Test schema manually
from mcps.template.src.schemas import SequenceStatsInput

try:
    input_data = SequenceStatsInput(sequence="ATCG", sequence_type="dna")
    print("Schema valid")
except ValidationError as e:
    print("Schema errors:", e.errors())
```

#### 4. Server Connectivity
```python
# Test MCP server connection
import asyncio
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

async def test_connection():
    try:
        read, write = await stdio_client([
            "python", "-m", "mcps.template.src.server"
        ])
        session = ClientSession(read, write)
        await session.initialize()
        
        tools = await session.list_tools()
        print(f"Connected! Available tools: {[t.name for t in tools]}")
        
    except Exception as e:
        print(f"Connection failed: {e}")

asyncio.run(test_connection())
```

### Debugging Configuration

```json
{
  "mcpServers": {
    "debug-server": {
      "command": "python",
      "args": ["-u", "-m", "mcps.template.src.server"],
      "cwd": "/path/to/nelli-ai-scientist",
      "env": {
        "PYTHONPATH": "/path/to/nelli-ai-scientist",
        "MCP_DEBUG": "1",
        "LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

### Logging Setup

```python
# mcps/template/src/server.py
import logging
import sys

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("MCP_DEBUG") else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Important: use stderr for MCP
)

logger = logging.getLogger(__name__)

class BioMCPServer:
    def __init__(self, name: str = "biopython-mcp"):
        logger.info(f"Initializing MCP server: {name}")
        self.server = Server(name)
        # ... rest of initialization
```

This comprehensive guide covers everything needed to set up and configure multiple MCP servers that can be discovered by LLM clients, with proper tool schemas and error handling.