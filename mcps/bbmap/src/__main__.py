"""
BBMap MCP Server - Main Entry Point

Run the BBMap MCP server using:
    python -m mcps.bbmap.src.server_fastmcp

Or:
    pixi run python -m mcps.bbmap.src.server_fastmcp
"""

from .server_fastmcp import mcp

if __name__ == "__main__":
    # Run the FastMCP server
    mcp.run()
