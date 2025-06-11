"""
QuickBin MCP Server Entry Point

Run the QuickBin MCP server directly using:
python -m mcps.quickbin.src.server_fastmcp
"""

from .server_fastmcp import mcp

if __name__ == "__main__":
    mcp.run()
