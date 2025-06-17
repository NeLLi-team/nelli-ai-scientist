#!/usr/bin/env python3
"""Debug tool argument passing in sophisticated agent"""

import asyncio
import json
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.sophisticated_agent.src.agent_stdio import UniversalMCPAgentStdio, AgentConfig

async def debug_tool_args():
    """Debug what happens to tool arguments"""
    
    config = AgentConfig(
        name="debug-agent",
        description="Debug agent for tool arguments",
        llm_provider="openai",
        max_conversation_turns=10,
        enable_biological_intelligence=True,
        enable_adaptive_analysis=True,
        use_stdio_connections=True
    )
    
    agent = UniversalMCPAgentStdio(config)
    await agent.initialize()
    
    # Check what tools are discovered
    print("=== DISCOVERED TOOLS ===")
    for tool_name, tool_info in agent.discovered_tools.items():
        if "Remote" in tool_info.get("server_name", ""):
            print(f"Tool: {tool_name}")
            print(f"  Server: {tool_info.get('server_name')}")
            print(f"  Schema: {tool_info.get('schema')}")
            print()
    
    # Test the parameter validation for a specific tool
    if 'analyze_fasta_file' in agent.discovered_tools:
        print("=== TESTING analyze_fasta_file PARAMETERS ===")
        tool_info = agent.discovered_tools['analyze_fasta_file']
        print(f"Tool info: {tool_info}")
        
        # Test parameters
        test_params = {"file_path": "example/AC3300027503___Ga0255182_1000024.fna"}
        print(f"Input parameters: {test_params}")
        
        # Validate parameters
        cleaned = agent._validate_parameters('analyze_fasta_file', test_params)
        print(f"Cleaned parameters: {cleaned}")
        
        # Check schema
        schema = tool_info.get('schema', {})
        properties = schema.get('properties', {})
        print(f"Schema properties: {list(properties.keys())}")

if __name__ == "__main__":
    asyncio.run(debug_tool_args())