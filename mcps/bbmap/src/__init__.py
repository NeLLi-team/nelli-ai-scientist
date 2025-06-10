"""
BBMap MCP Server Package

This package provides a Model Context Protocol (MCP) server for BBMap bioinformatics tools.
BBMap is a high-performance read mapper for DNA sequencing data analysis.

Main components:
- bbmap_tools.py: Core BBMap toolkit wrapper
- server_fastmcp.py: FastMCP protocol server
- tool_schema.py: API schema definitions

Proven capabilities:
- Read mapping: 287MB contigs + 1.2GB reads → 9GB SAM in 2 minutes
- Coverage analysis: Detailed statistics and histograms
- Quality control: FASTQ quality assessment
- Read filtering: Quality-based read filtering

Container integration: Shifter with bryce911/bbtools:latest
"""

from .bbmap_tools import BBMapToolkit
from .server_fastmcp import mcp

__all__ = ['BBMapToolkit', 'mcp']
__version__ = '1.0.0'