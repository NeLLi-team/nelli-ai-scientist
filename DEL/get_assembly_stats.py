#!/usr/bin/env python3
"""Get assembly stats using the remote bioseq MCP server"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.sophisticated_agent.src.mcp_websocket_client import MCPWebSocketClient

async def get_assembly_stats():
    """Get assembly statistics from the remote bioseq server"""
    
    # Use the Cloudflare tunnel URL (same as in mcp_config.json)
    server_config = {
        "uri": "wss://exempt-club-java-iron.trycloudflare.com"
    }
    
    client = MCPWebSocketClient("bioseq-remote", server_config)
    
    print("Getting assembly statistics from remote bioseq server...")
    
    try:
        result = await client.call_tool(
            tool_name="assembly_stats",
            arguments={"sequences": "/home/fschulz/dev/nelli-ai-scientist/example/AC3300027503___Ga0255182_1000024.fna"}
        )
        
        print("\n" + "="*50)
        print("ASSEMBLY STATISTICS")
        print("="*50)
        print(result)
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(get_assembly_stats())