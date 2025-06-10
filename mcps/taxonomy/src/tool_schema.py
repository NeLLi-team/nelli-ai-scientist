"""
Tool Schema for Filesystem MCP Server
Defines the schema for filesystem operations tools
"""

from typing import Dict, Any


def get_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Get the tool schemas for filesystem operations"""

    return {
        "get_rank_lineage": {
            "name": "get_rank_lineage",
            "description": "Get the rank lineage for a given taxon ID",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "taxon_id": {
                        "type": "integer",
                        "description": "The NCBI taxon ID to get the lineage for",
                    }
                },
                "required": ["taxon_id"],
            },
        },
    }


def get_resource_schemas() -> Dict[str, Dict[str, Any]]:
    """Get the resource schemas for filesystem operations"""

    return {
        "filesystem://allowed-dirs": {
            "name": "filesystem://allowed-dirs",
            "description": "Get list of allowed directories for file operations",
            "mimeType": "application/json",
        },
        "filesystem://examples": {
            "name": "filesystem://examples",
            "description": "Get example file operations",
            "mimeType": "application/json",
        },
    }
