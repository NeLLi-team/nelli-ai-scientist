"""
BioPython MCP Server Implementation
Provides bioinformatics tools via Model Context Protocol
"""

import asyncio
import json
from typing import Dict, Any, List
import logging

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent

from .biotools import BioToolkit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioMCPServer:
    """MCP Server for BioPython tools"""

    def __init__(self, name: str = "biopython-mcp"):
        self.server = Server(name)
        self.toolkit = BioToolkit()

        # Register handlers
        self._register_handlers()

        logger.info(f"Initialized BioMCP server: {name}")

    def _register_handlers(self):
        """Register all MCP handlers"""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List all available BioPython tools"""
            tools = []

            # Sequence statistics tool
            tools.append(
                Tool(
                    name="sequence_stats",
                    description="Calculate comprehensive sequence statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sequence": {
                                "type": "string",
                                "description": "DNA, RNA, or protein sequence",
                            },
                            "sequence_type": {
                                "type": "string",
                                "enum": ["dna", "rna", "protein"],
                                "description": "Type of sequence",
                            },
                        },
                        "required": ["sequence", "sequence_type"],
                    },
                )
            )

            # BLAST search tool
            tools.append(
                Tool(
                    name="blast_local",
                    description="Run local BLAST search against a database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sequence": {
                                "type": "string",
                                "description": "Query sequence",
                            },
                            "database": {
                                "type": "string",
                                "description": "Database name or path",
                            },
                            "program": {
                                "type": "string",
                                "enum": [
                                    "blastn",
                                    "blastp",
                                    "blastx",
                                    "tblastn",
                                    "tblastx",
                                ],
                                "description": "BLAST program to use",
                            },
                            "e_value": {
                                "type": "number",
                                "description": "E-value threshold",
                                "default": 0.001,
                            },
                        },
                        "required": ["sequence", "database", "program"],
                    },
                )
            )

            # Multiple sequence alignment
            tools.append(
                Tool(
                    name="multiple_alignment",
                    description="Perform multiple sequence alignment",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sequences": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "sequence": {"type": "string"},
                                    },
                                },
                                "description": "List of sequences to align",
                            },
                            "algorithm": {
                                "type": "string",
                                "enum": ["clustalw", "muscle", "mafft"],
                                "description": "Alignment algorithm",
                                "default": "clustalw",
                            },
                        },
                        "required": ["sequences"],
                    },
                )
            )

            # Phylogenetic tree
            tools.append(
                Tool(
                    name="phylogenetic_tree",
                    description="Build phylogenetic tree from sequences",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "alignment": {
                                "type": "string",
                                "description": "Multiple sequence alignment in FASTA format",
                            },
                            "method": {
                                "type": "string",
                                "enum": ["nj", "upgma", "maximum_likelihood"],
                                "description": "Tree building method",
                                "default": "nj",
                            },
                        },
                        "required": ["alignment"],
                    },
                )
            )

            # Sequence translation
            tools.append(
                Tool(
                    name="translate_sequence",
                    description="Translate DNA/RNA sequence to protein",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sequence": {
                                "type": "string",
                                "description": "DNA or RNA sequence",
                            },
                            "genetic_code": {
                                "type": "integer",
                                "description": "NCBI genetic code table number",
                                "default": 1,
                            },
                        },
                        "required": ["sequence"],
                    },
                )
            )

            # FASTA file reading
            tools.append(
                Tool(
                    name="read_fasta_file",
                    description="Read sequences from a FASTA file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the FASTA file",
                            },
                        },
                        "required": ["file_path"],
                    },
                )
            )

            # JSON report writing
            tools.append(
                Tool(
                    name="write_json_report",
                    description="Write analysis results to a JSON report file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "description": "Analysis data to write",
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Path for the output JSON file",
                            },
                        },
                        "required": ["data", "output_path"],
                    },
                )
            )

            # Comprehensive FASTA file analysis
            tools.append(
                Tool(
                    name="analyze_fasta_file",
                    description="Comprehensive analysis of sequences in a FASTA file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the FASTA file",
                            },
                            "sequence_type": {
                                "type": "string",
                                "enum": ["dna", "rna", "protein"],
                                "description": "Type of sequences in the file",
                                "default": "dna",
                            },
                        },
                        "required": ["file_path"],
                    },
                )
            )

            return tools

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> List[TextContent]:
            """Execute a BioPython tool"""

            logger.info(f"Executing tool: {name} with args: {arguments}")

            try:
                if name == "sequence_stats":
                    result = await self.toolkit.sequence_stats(**arguments)

                elif name == "blast_local":
                    result = await self.toolkit.blast_local(**arguments)

                elif name == "multiple_alignment":
                    result = await self.toolkit.multiple_alignment(**arguments)

                elif name == "phylogenetic_tree":
                    result = await self.toolkit.phylogenetic_tree(**arguments)

                elif name == "translate_sequence":
                    result = await self.toolkit.translate_sequence(**arguments)

                elif name == "read_fasta_file":
                    result = await self.toolkit.read_fasta_file(**arguments)

                elif name == "write_json_report":
                    result = await self.toolkit.write_json_report(**arguments)

                elif name == "analyze_fasta_file":
                    result = await self.toolkit.analyze_fasta_file(**arguments)

                else:
                    raise ValueError(f"Unknown tool: {name}")

                # Format result as TextContent
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": str(e), "tool": name, "arguments": arguments}
                        ),
                    )
                ]

        @self.server.list_resources()
        async def handle_list_resources() -> List[str]:
            """List available resources"""
            return [
                "sequences://examples",
                "databases://blast/nt",
                "databases://blast/nr",
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a resource"""
            if uri == "sequences://examples":
                return json.dumps(
                    {
                        "dna_example": "ATCGATCGATCGATCGATCG",
                        "protein_example": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG",
                    }
                )

            return json.dumps({"error": "Resource not found"})

    async def run(self, transport: str = "stdio"):
        """Run the MCP server"""
        if transport == "stdio":
            # Run with stdio transport
            from mcp.server.stdio import stdio_server

            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="biopython-mcp", 
                        server_version="1.0.0",
                        capabilities={}
                    ),
                )

        elif transport == "http":
            # Run with HTTP transport
            from mcp.server.fastapi import create_mcp_app
            import uvicorn

            app = create_mcp_app(self.server)

            config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")

            server = uvicorn.Server(config)
            await server.serve()


# Main entry point
if __name__ == "__main__":
    import sys

    async def main():
        transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"

        server = BioMCPServer("biopython-mcp-template")
        await server.run(transport)

    asyncio.run(main())
