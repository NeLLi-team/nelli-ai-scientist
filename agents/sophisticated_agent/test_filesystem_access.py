#!/usr/bin/env python3
"""
Test filesystem MCP access to verify file paths work correctly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.enhanced_agent import EnhancedUniversalMCPAgent, EnhancedAgentConfig
from src.config_loader import ConfigLoader

async def test_filesystem_access():
    """Test filesystem access from the enhanced agent"""
    
    print("🧪 Testing Filesystem MCP Access...")
    print("=" * 60)
    
    # Get current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Expected file paths (both relative and absolute)
    test_files = [
        "data/nelli_hackathon/contigs100k.fna",  # relative path
        "/home/fschulz/nelli-ai-scientist/data/nelli_hackathon/contigs100k.fna",  # absolute path
        "../../data/nelli_hackathon/contigs100k.fna"  # relative from agent dir
    ]
    
    print(f"\nTesting file existence from current directory ({cwd}):")
    for file_path in test_files:
        exists = os.path.exists(file_path)
        abs_path = os.path.abspath(file_path)
        print(f"  {file_path} -> {abs_path} (exists: {exists})")
    
    # Create temporary config loader
    config_loader = ConfigLoader("config/agent_config.yaml")
    
    # Create enhanced configuration
    config = EnhancedAgentConfig(
        name=config_loader.get_agent_name(),
        mcp_config_path="../../mcp_config.json",  # Path from agent dir to root
        config_file="config/agent_config.yaml"
    )
    
    # Create agent but skip initialization that requires API key
    print(f"\n🤖 Creating enhanced agent...")
    agent = EnhancedUniversalMCPAgent(config, config_loader)
    
    try:
        # Initialize to discover tools
        print("🔍 Initializing agent and discovering MCP tools...")
        await agent.initialize()
        
        # Test filesystem tools
        print(f"\n📁 Testing filesystem tool access...")
        
        # Test with different path formats
        for file_path in test_files:
            if os.path.exists(file_path):
                print(f"\n🧪 Testing path: {file_path}")
                try:
                    # Test file_exists tool
                    result = await agent._call_mcp_tool("file_exists", path=file_path)
                    print(f"   file_exists result: {result}")
                    
                    if result.get("exists"):
                        # Test read_file tool  
                        read_result = await agent._call_mcp_tool("read_file", path=file_path)
                        if read_result.get("success"):
                            content_length = len(read_result.get("content", ""))
                            print(f"   read_file: ✅ Success (content length: {content_length})")
                        else:
                            print(f"   read_file: ❌ Error - {read_result.get('error')}")
                    else:
                        print(f"   file_exists: ❌ File not found via MCP")
                        
                except Exception as e:
                    print(f"   ❌ MCP tool error: {e}")
                
                break  # Test only the first existing file
        
        print(f"\n📂 Testing list_directory tool...")
        try:
            # Test listing the data directory
            list_result = await agent._call_mcp_tool("list_directory", path="data/nelli_hackathon")
            print(f"   list_directory result: {list_result}")
        except Exception as e:
            print(f"   ❌ list_directory error: {e}")
            
    except Exception as e:
        print(f"❌ Agent initialization error: {e}")
        print("💡 This might be due to missing API keys - checking filesystem MCP config directly...")
        
        # Direct filesystem MCP check
        print(f"\n🔧 Direct filesystem MCP path check...")
        from mcps.filesystem.src.server import REPO_BASE, ALLOWED_DIRS, _check_path_security
        
        print(f"   Repository base: {REPO_BASE}")
        print(f"   Allowed directories: {ALLOWED_DIRS}")
        
        for file_path in test_files:
            if os.path.exists(file_path):
                abs_path = os.path.abspath(file_path)
                is_allowed = _check_path_security(file_path)
                print(f"   Path: {file_path}")
                print(f"     Absolute: {abs_path}")
                print(f"     Security check: {is_allowed}")
                break

if __name__ == "__main__":
    # Set environment variables if needed
    if not os.getenv("CBORG_API_KEY"):
        os.environ["CBORG_API_KEY"] = "test-key"
        os.environ["CBORG_BASE_URL"] = "https://api.cborg.lbl.gov" 
        os.environ["CBORG_MODEL"] = "google/gemini-flash-lite"
    
    asyncio.run(test_filesystem_access())