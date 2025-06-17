#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "agents" / "sophisticated_agent" / "src"))

async def check_tools():
    from agent_stdio import UniversalMCPAgentStdio, AgentConfig
    from llm_interface import LLMProvider
    
    config = AgentConfig(
        name='tool-check',
        role='test', 
        description='Check tools',
        llm_provider=LLMProvider.CBORG,
        mcp_config_path='agents/sophisticated_agent/mcp_config.json'
    )
    
    agent = UniversalMCPAgentStdio(config)
    await agent.initialize()
    
    # Look for code execution tools
    code_tools = [name for name in agent.discovered_tools.keys() if 'code' in name.lower() or 'execute' in name.lower() or 'python' in name.lower()]
    print('Code execution tools found:')
    for tool in code_tools:
        print(f'  - {tool}: {agent.discovered_tools[tool].get("description", "")}')
    
    print('\nAll available tools:')
    for name in sorted(agent.discovered_tools.keys()):
        print(f'  - {name}')
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(check_tools())