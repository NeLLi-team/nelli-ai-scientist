#!/usr/bin/env python3
"""Test WebSocket MCP client directly"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.sophisticated_agent.src.mcp_websocket_client import MCPWebSocketClient

async def test_websocket_client():
    """Test the WebSocket MCP client directly"""
    
    server_config = {
        "uri": "ws://localhost:8765"
    }
    
    client = MCPWebSocketClient("test-bioseq", server_config)
    
    print("Testing WebSocket client...")
    
    try:
        print("Listing tools...")
        tools = await client.list_tools()
        print(f"Found {len(tools)} tools")
        
        if tools:
            print(f"First tool: {tools[0]['name']}")
            
            print("Testing assembly_stats tool call...")
            result = await client.call_tool(
                tool_name="assembly_stats",
                arguments={"sequences": "/home/fschulz/dev/nelli-ai-scientist/example/AC3300027503___Ga0255182_1000024.fna"}
            )
            print(f"Assembly stats result: {result}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket_client())