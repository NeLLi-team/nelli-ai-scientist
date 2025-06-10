"""
Tool Schema for Filesystem MCP Server
Defines the schema for filesystem operations tools
"""

from typing import Dict, Any

def get_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Get the tool schemas for filesystem operations"""
    
    return {
        "read_file": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
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
        
        "list_directory": {
            "name": "list_directory",
            "description": "List contents of a directory",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list",
                        "default": "/tmp"
                    }
                },
                "required": []
            }
        },
        
        "create_directory": {
            "name": "create_directory",
            "description": "Create a directory",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to create"
                    }
                },
                "required": ["path"]
            }
        },
        
        "delete_file": {
            "name": "delete_file",
            "description": "Delete a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to delete"
                    }
                },
                "required": ["path"]
            }
        },
        
        "file_exists": {
            "name": "file_exists",
            "description": "Check if a file or directory exists",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to check"
                    }
                },
                "required": ["path"]
            }
        },
        
        "find_files_by_pattern": {
            "name": "find_files_by_pattern",
            "description": "Find files matching a pattern or extensions in a directory tree",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string", 
                        "description": "Starting directory path (defaults to current working directory)",
                        "default": None
                    },
                    "pattern": {
                        "type": "string",
                        "description": "File name pattern (e.g., '*.txt', 'seq*')",
                        "default": "*"
                    },
                    "extensions": {
                        "type": "string",
                        "description": "Comma-separated file extensions (e.g., 'fna,fasta,fastq,fa')"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to search",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": []
            }
        },
        
        "explore_directory_tree": {
            "name": "explore_directory_tree",
            "description": "Explore directory structure recursively from current working directory or specified path",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Starting directory path (defaults to current working directory)",
                        "default": None
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    },
                    "include_files": {
                        "type": "boolean",
                        "description": "Whether to include files in the output",
                        "default": True
                    }
                },
                "required": []
            }
        }
    }


def get_resource_schemas() -> Dict[str, Dict[str, Any]]:
    """Get the resource schemas for filesystem operations"""
    
    return {
        "filesystem://allowed-dirs": {
            "name": "filesystem://allowed-dirs",
            "description": "Get list of allowed directories for file operations",
            "mimeType": "application/json"
        },
        
        "filesystem://examples": {
            "name": "filesystem://examples", 
            "description": "Get example file operations",
            "mimeType": "application/json"
        }
    }