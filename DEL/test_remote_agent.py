#!/usr/bin/env python3
"""Test sophisticated agent with remote WebSocket connection"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.sophisticated_agent.src.agent_stdio import UniversalMCPAgentStdio

async def test_remote_agent():
    """Test the agent with remote WebSocket connection"""
    
    agent = UniversalMCPAgentStdio()
    
    print("Initializing agent...")
    await agent.initialize()
    
    print("Listing available tools...")
    tools = await agent.list_available_tools()
    
    # Find remote bioseq tools
    remote_tools = [tool for tool in tools if 'Remote Nucleic Acid' in tool.get('server_name', '')]
    print(f"Found {len(remote_tools)} remote tools")
    
    if remote_tools:
        print("Testing remote tool call...")
        try:
            result = await agent.call_tool(
                tool_name="sequence_stats",
                arguments={"sequence": "ATCGATCG"},
                server_id="bioseq-remote"
            )
            print(f"Tool call result: {result}")
        except Exception as e:
            print(f"Tool call failed: {e}")
            import traceback
            traceback.print_exc()
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(test_remote_agent())