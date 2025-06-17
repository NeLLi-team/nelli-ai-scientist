#!/usr/bin/env python3
"""Debug tool schema for remote WebSocket tools"""

import asyncio
import json
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.sophisticated_agent.src.mcp_websocket_client import MCPWebSocketClient

async def debug_schema():
    """Debug the schema of remote tools"""
    
    server_config = {
        "uri": "wss://exempt-club-java-iron.trycloudflare.com"
    }
    
    client = MCPWebSocketClient("bioseq-remote", server_config)
    
    print("=== REMOTE TOOLS SCHEMA DEBUG ===")
    
    try:
        tools = await client.list_tools()
        
        for tool in tools:
            if tool["name"] == "analyze_fasta_file":
                print(f"Tool: {tool['name']}")
                print(f"Description: {tool['description']}")
                print(f"Schema: {json.dumps(tool['input_schema'], indent=2)}")
                
                # Test what properties are available
                properties = tool['input_schema'].get('properties', {})
                required = tool['input_schema'].get('required', [])
                
                print(f"Available properties: {list(properties.keys())}")
                print(f"Required properties: {required}")
                break
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_schema())