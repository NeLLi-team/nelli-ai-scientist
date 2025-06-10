# Filesystem MCP Server

A Model Context Protocol (MCP) server that provides safe filesystem operations with security restrictions.

## Features

- **File Operations**: Read, write, delete files
- **Directory Operations**: List, create, explore directory trees  
- **Security**: Restricted to allowed directories for safety
- **Recursive Exploration**: Multi-level directory tree traversal

## Tools

### File Operations
- `read_file(path)` - Read file contents
- `write_file(path, content)` - Write content to file
- `delete_file(path)` - Delete a file
- `file_exists(path)` - Check if file/directory exists

### Directory Operations  
- `list_directory(path="/tmp")` - List directory contents
- `create_directory(path)` - Create a directory
- `explore_directory_tree(path=None, max_depth=3, include_files=True)` - Recursively explore directory structure

## Resources

- `filesystem://allowed-dirs` - List of allowed directories
- `filesystem://examples` - Example operations

## Security

File operations are restricted to:
- `/tmp` - Temporary files
- `/home/fschulz/dev/nelli-ai-scientist` - Project directory

## Usage

```python
# Start the server
python src/server.py

# Or with FastMCP
from fastmcp import FastMCP
from src.server import mcp
mcp.run()
```

## Configuration

The server uses FastMCP framework and communicates via stdio transport by default.

## Files

- `src/server.py` - Main server implementation
- `src/tool_schema.py` - Tool and resource schema definitions  
- `README.md` - This documentation