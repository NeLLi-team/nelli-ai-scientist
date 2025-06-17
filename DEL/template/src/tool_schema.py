"""
Tool Schema for BioPython MCP Server
Defines the schema for bioinformatics tools using BioPython
"""

from typing import Dict, Any

def get_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Get the tool schemas for bioinformatics operations"""
    
    return {
        "sequence_stats": {
            "name": "sequence_stats",
            "description": "Calculate comprehensive sequence statistics",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "DNA, RNA, or protein sequence"
                    },
                    "sequence_type": {
                        "type": "string",
                        "description": "Type of sequence",
                        "enum": ["dna", "rna", "protein"],
                        "default": "dna"
                    }
                },
                "required": ["sequence", "sequence_type"]
            }
        },
        
        "blast_local": {
            "name": "blast_local",
            "description": "Run local BLAST search against a database",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Query sequence"
                    },
                    "database": {
                        "type": "string",
                        "description": "Database name or path"
                    },
                    "program": {
                        "type": "string",
                        "description": "BLAST program to use",
                        "enum": ["blastn", "blastp", "blastx", "tblastn", "tblastx"]
                    },
                    "e_value": {
                        "type": "number",
                        "description": "E-value threshold",
                        "default": 0.001,
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["sequence", "database", "program"]
            }
        },
        
        "multiple_alignment": {
            "name": "multiple_alignment",
            "description": "Perform multiple sequence alignment",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sequences": {
                        "type": "array",
                        "description": "List of sequences to align",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "sequence": {"type": "string"}
                            },
                            "required": ["id", "sequence"]
                        }
                    },
                    "algorithm": {
                        "type": "string", 
                        "description": "Alignment algorithm",
                        "enum": ["clustalw", "muscle", "mafft"],
                        "default": "clustalw"
                    }
                },
                "required": ["sequences"]
            }
        },
        
        "phylogenetic_tree": {
            "name": "phylogenetic_tree",
            "description": "Build phylogenetic tree from sequences",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "alignment": {
                        "type": "string",
                        "description": "Multiple sequence alignment in FASTA format"
                    },
                    "method": {
                        "type": "string",
                        "description": "Tree building method",
                        "enum": ["nj", "upgma", "maximum_likelihood"],
                        "default": "nj"
                    }
                },
                "required": ["alignment"]
            }
        },
        
        "translate_sequence": {
            "name": "translate_sequence", 
            "description": "Translate DNA/RNA sequence to protein",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "DNA or RNA sequence"
                    },
                    "genetic_code": {
                        "type": "integer",
                        "description": "NCBI genetic code table number",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 31
                    }
                },
                "required": ["sequence"]
            }
        },
        
        "read_fasta_file": {
            "name": "read_fasta_file",
            "description": "Read sequences from a FASTA file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the FASTA file"
                    }
                },
                "required": ["file_path"]
            }
        },
        
        "write_json_report": {
            "name": "write_json_report",
            "description": "Write analysis results to a JSON report file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Analysis data to write"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path for the output JSON file"
                    }
                },
                "required": ["data", "output_path"]
            }
        },
        
        "analyze_fasta_file": {
            "name": "analyze_fasta_file",
            "description": "Comprehensive analysis of sequences in a FASTA file", 
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the FASTA file"
                    },
                    "sequence_type": {
                        "type": "string",
                        "description": "Type of sequences in the file",
                        "enum": ["dna", "rna", "protein"],
                        "default": "dna"
                    }
                },
                "required": ["file_path"]
            }
        }
    }


def get_resource_schemas() -> Dict[str, Dict[str, Any]]:
    """Get the resource schemas for bioinformatics operations"""
    
    return {
        "sequences://examples": {
            "name": "sequences://examples",
            "description": "Get example sequences for testing",
            "mimeType": "application/json"
        },
        
        "databases://blast/list": {
            "name": "databases://blast/list",
            "description": "List available BLAST databases", 
            "mimeType": "application/json"
        }
    }