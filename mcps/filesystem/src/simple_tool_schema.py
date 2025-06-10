"""
Simple Tool Schema for the Simple Filesystem MCP Server
"""

from typing import Dict, Any

def get_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Get the tool schemas for simple filesystem operations"""
    
    return {
        "tree_view": {
            "name": "tree_view",
            "description": "Display directory structure in a tree format with file details",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to explore",
                        "default": "."
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files/directories",
                        "default": False
                    },
                    "file_extensions": {
                        "type": "string",
                        "description": "Only show files with these extensions (e.g., 'py,txt,fna,fasta')",
                        "default": ""
                    }
                },
                "required": []
            }
        },
        
        "find_files": {
            "name": "find_files",
            "description": "Find files by pattern or extension recursively",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Starting directory",
                        "default": "."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Filename pattern using wildcards (e.g., '*contigs*', '*.py')",
                        "default": "*"
                    },
                    "extensions": {
                        "type": "string",
                        "description": "File extensions to search for (e.g., 'fna,fasta,fastq,py,txt')",
                        "default": ""
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum search depth",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": []
            }
        },
        
        "read_file": {
            "name": "read_file",
            "description": "Read file contents safely with optional line limit",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (0 = read all)",
                        "default": 0,
                        "minimum": 0
                    }
                },
                "required": ["path"]
            }
        },
        
        "write_file": {
            "name": "write_file",
            "description": "Write content to a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        },
        
        "find_file_by_name": {
            "name": "find_file_by_name",
            "description": "Find a file by name in the directory tree",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to find (e.g., 'contigs100k.fna')"
                    },
                    "search_path": {
                        "type": "string",
                        "description": "Directory to search in",
                        "default": "."
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum search depth",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["filename"]
            }
        },
        
        "file_info": {
            "name": "file_info",
            "description": "Get detailed information about a file or directory",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to examine"
                    }
                },
                "required": ["path"]
            }
        }
    }

def get_resource_schemas() -> Dict[str, Dict[str, Any]]:
    """Get the resource schemas"""
    
    return {
        "simple-filesystem://help": {
            "name": "simple-filesystem://help",
            "description": "Get help information for simple filesystem tools",
            "mimeType": "application/json"
        }
    }