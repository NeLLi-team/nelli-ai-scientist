# MCP Server Reorganization

This document describes the reorganized structure of MCP (Model Context Protocol) servers in the project.

## New Structure

```
mcps/
├── template/                    # BioPython MCP Server  
│   ├── README.md               # Server documentation
│   ├── src/
│   │   ├── __init__.py
│   │   ├── server.py           # Original MCP server
│   │   ├── server_fastmcp.py   # FastMCP implementation
│   │   ├── biotools.py         # BioPython wrapper functions
│   │   ├── tool_schema.py      # Tool schema definitions
│   │   └── client.py           # Test client
│   └── tests/
│       ├── __init__.py
│       └── test_biotools.py
└── filesystem/                 # Filesystem MCP Server
    ├── README.md               # Server documentation  
    ├── src/
    │   ├── __init__.py
    │   ├── server.py           # FastMCP filesystem server
    │   └── tool_schema.py      # Tool schema definitions
    └── __init__.py
```

## Key Improvements

### 1. **Organized Structure**
- Each MCP server now has its own subdirectory
- Consistent file naming: `server.py` for main implementation
- Dedicated `tool_schema.py` files for schema definitions
- Comprehensive `README.md` documentation for each server

### 2. **Tool Schema Files**
Tool schema files define the structure and parameters for each tool:

```python
# Example from tool_schema.py
{
    "sequence_stats": {
        "name": "sequence_stats",
        "description": "Calculate comprehensive sequence statistics",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "DNA, RNA, or protein sequence"},
                "sequence_type": {"type": "string", "enum": ["dna", "rna", "protein"]}
            },
            "required": ["sequence", "sequence_type"]
        }
    }
}
```

### 3. **Updated Configuration**
The `mcp_config.json` has been updated with new paths:

```json
{
  "biopython": {
    "fastmcp_script": "/home/fschulz/dev/nelli-ai-scientist/mcps/template/src/server_fastmcp.py"
  },
  "filesystem": {
    "fastmcp_script": "/home/fschulz/dev/nelli-ai-scientist/mcps/filesystem/src/server.py"
  }
}
```

## Server Details

### BioPython MCP Server (`mcps/template/`)

**Purpose**: Bioinformatics tools using BioPython library

**Tools**:
- `sequence_stats` - Calculate sequence statistics
- `analyze_fasta_file` - Comprehensive FASTA analysis
- `blast_local` - Local BLAST searches
- `multiple_alignment` - Multiple sequence alignment
- `phylogenetic_tree` - Phylogenetic tree construction
- `translate_sequence` - DNA/RNA to protein translation
- `read_fasta_file` - FASTA file reading
- `write_json_report` - JSON report generation

**Files**:
- `server_fastmcp.py` - Main FastMCP implementation
- `biotools.py` - BioPython wrapper functions
- `tool_schema.py` - Tool schema definitions

### Filesystem MCP Server (`mcps/filesystem/`)

**Purpose**: Safe filesystem operations with security restrictions

**Tools**:
- `read_file` - Read file contents
- `write_file` - Write content to files
- `list_directory` - List directory contents
- `create_directory` - Create directories
- `delete_file` - Delete files
- `file_exists` - Check file/directory existence
- `explore_directory_tree` - Recursive directory exploration

**Security**: Operations restricted to `/tmp` and project directory

## Tool Schema Usage

Tool schemas serve multiple purposes:

1. **Documentation**: Clear specification of tool parameters and types
2. **Validation**: Input validation for tool calls
3. **IDE Support**: Auto-completion and type checking
4. **API Generation**: Automatic API documentation generation

### Loading Schemas

```python
from mcps.template.src.tool_schema import get_tool_schemas
from mcps.filesystem.src.tool_schema import get_tool_schemas as get_fs_schemas

bio_schemas = get_tool_schemas()
fs_schemas = get_fs_schemas()
```

## Migration Benefits

1. **Better Organization**: Clear separation of different MCP servers
2. **Easier Maintenance**: Each server is self-contained with documentation
3. **Schema Documentation**: Tool schemas provide clear API definitions
4. **Consistent Structure**: All servers follow the same organization pattern
5. **Improved Discovery**: README files make it easy to understand each server's capabilities

## Future Additions

To add a new MCP server:

1. Create `mcps/new_server/` directory
2. Add `src/server.py` with FastMCP implementation
3. Create `src/tool_schema.py` with tool definitions
4. Write `README.md` with documentation
5. Update `mcp_config.json` with server configuration
6. Add agent prompt guidance for new tools

This structure makes the project more professional, maintainable, and easier to extend with new MCP servers.