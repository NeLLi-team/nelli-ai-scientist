"""
Tool schema definitions for BBMap MCP server
"""

from typing import Dict, Any, List


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Get all tool schemas for BBMap tools"""

    return [
        {
            "name": "map_reads",
            "description": "Map sequencing reads to a reference genome using BBMap",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference_path": {
                        "type": "string",
                        "description": "Path to reference genome file (FASTA format)"
                    },
                    "reads_path": {
                        "type": "string",
                        "description": "Path to reads file (FASTQ format)"
                    },
                    "output_sam": {
                        "type": "string",
                        "description": "Output path for SAM alignment file"
                    },
                    "additional_params": {
                        "type": "string",
                        "description": "Additional BBMap parameters (optional)",
                        "default": None
                    }
                },
                "required": ["reference_path", "reads_path", "output_sam"]
            }
        },
        {
            "name": "quality_stats",
            "description": "Generate comprehensive quality statistics for FASTQ files",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "fastq_path": {
                        "type": "string",
                        "description": "Path to FASTQ file to analyze"
                    },
                    "output_prefix": {
                        "type": "string",
                        "description": "Prefix for output statistics files",
                        "default": "quality_stats"
                    }
                },
                "required": ["fastq_path"]
            }
        },
        {
            "name": "coverage_analysis",
            "description": "Analyze read coverage from SAM/BAM alignment files",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sam_path": {
                        "type": "string",
                        "description": "Path to SAM/BAM alignment file"
                    },
                    "reference_path": {
                        "type": "string",
                        "description": "Path to reference genome file (FASTA format)"
                    },
                    "output_prefix": {
                        "type": "string",
                        "description": "Prefix for coverage output files",
                        "default": "coverage"
                    }
                },
                "required": ["sam_path", "reference_path"]
            }
        },
        {
            "name": "filter_reads",
            "description": "Filter reads based on quality and length criteria",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_fastq": {
                        "type": "string",
                        "description": "Input FASTQ file to filter"
                    },
                    "output_fastq": {
                        "type": "string",
                        "description": "Output path for filtered FASTQ file"
                    },
                    "min_length": {
                        "type": "integer",
                        "description": "Minimum read length threshold",
                        "default": 50
                    },
                    "min_quality": {
                        "type": "number",
                        "description": "Minimum average quality score",
                        "default": 20.0
                    },
                    "additional_params": {
                        "type": "string",
                        "description": "Additional filtering parameters (optional)",
                        "default": None
                    }
                },
                "required": ["input_fastq", "output_fastq"]
            }
        }
    ]


def get_resource_schemas() -> List[Dict[str, Any]]:
    """Get resource schemas for BBMap MCP server"""

    return [
        {
            "uri": "bbmap://docs/user-guide",
            "name": "BBMap User Guide",
            "description": "Comprehensive guide for using BBMap tools",
            "mimeType": "text/markdown"
        },
        {
            "uri": "bbmap://examples/basic-workflow",
            "name": "Basic BBMap Workflow",
            "description": "Example workflow for read mapping and analysis",
            "mimeType": "text/markdown"
        },
        {
            "uri": "bbmap://tools/available",
            "name": "Available BBMap Tools",
            "description": "List of all available BBMap/BBTools programs",
            "mimeType": "application/json"
        }
    ]