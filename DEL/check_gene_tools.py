#!/usr/bin/env python3
"""Check what gene-related tools are available"""

import asyncio
import json
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.sophisticated_agent.src.mcp_websocket_client import MCPWebSocketClient

async def check_gene_tools():
    """Check gene-related tools and their descriptions"""
    
    server_config = {
        "uri": "wss://exempt-club-java-iron.trycloudflare.com"
    }
    
    client = MCPWebSocketClient("bioseq-remote", server_config)
    
    print("=== GENE-RELATED TOOLS ===")
    
    try:
        tools = await client.list_tools()
        
        gene_keywords = ['gene', 'coding', 'predict', 'orf', 'protein']
        
        for tool in tools:
            tool_name = tool["name"].lower()
            tool_desc = tool["description"].lower()
            
            if any(keyword in tool_name or keyword in tool_desc for keyword in gene_keywords):
                print(f"\nTool: {tool['name']}")
                print(f"Description: {tool['description']}")
                print(f"Required parameters: {tool['input_schema'].get('required', [])}")
                print(f"All parameters: {list(tool['input_schema'].get('properties', {}).keys())}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_gene_tools())