#!/usr/bin/env python3
"""
Direct test of filesystem tools to verify they work
"""
import asyncio
import sys
from pathlib import Path

# Add the mcps/filesystem src to the path
sys.path.insert(0, str(Path(__file__).parent / "mcps" / "filesystem" / "src"))

async def test_filesystem_tools():
    """Test filesystem tools directly"""
    
    try:
        # Import the filesystem tools
        from simple_server import mcp
        print("✅ Filesystem server imports successfully")
        
        # Try to find tools that include tree_view
        tools = []
        for name, tool in mcp._tools.items():
            tools.append(name)
            
        print(f"✅ Found {len(tools)} tools: {tools}")
        
        # Test if we can call tree_view directly
        if 'tree_view' in tools:
            print("✅ tree_view tool is available")
            
            # Try to get the function directly
            tree_view_tool = mcp._tools['tree_view']
            print(f"Tree view tool: {tree_view_tool}")
            
            # The tool should be callable - let's try to inspect it
            print(f"Tool type: {type(tree_view_tool)}")
            
        else:
            print("❌ tree_view tool not found")
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_filesystem_tools())