#!/usr/bin/env python3
"""
Test script for remote bioseq MCP server connection
"""

import asyncio
import json
from src.mcp_websocket_client import MCPWebSocketClient

async def test_remote_bioseq():
    """Test connection to remote bioseq server"""
    
    # Configuration for remote server
    server_config = {
        "name": "Remote Bioseq MCP",
        "transport": "websocket",
        "uri": "wss://wear-budapest-valve-sublime.trycloudflare.com",
        "enabled": True
    }
    
    print("🔧 Testing remote bioseq MCP server connection...")
    print(f"URI: {server_config['uri']}")
    
    try:
        # Create client
        client = MCPWebSocketClient("bioseq-remote", server_config)
        
        # List available tools
        print("\n📋 Discovering available tools...")
        tools = await client.list_tools()
        
        print(f"\n✅ Successfully connected! Found {len(tools)} tools:")
        for tool in tools[:5]:  # Show first 5 tools
            print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        
        if len(tools) > 5:
            print(f"  ... and {len(tools) - 5} more tools")
            
        # Test a simple tool call
        print("\n🧪 Testing sequence_stats tool...")
        test_sequence = "ATCGATCGATCGATCGATCG"
        
        result = await client.call_tool("sequence_stats", {
            "sequence": test_sequence
        })
        
        print("📊 Result:")
        print(result)
        
        print("\n✅ Remote connection test successful!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_remote_bioseq())