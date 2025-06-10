#!/usr/bin/env python3
import asyncio
from src.agent import BioinformaticsAgent, AgentConfig
from src.llm_interface import LLMProvider

async def test():
    print("🧪 Testing Chat Agent Functions")
    
    config = AgentConfig(name='chat-test', llm_provider=LLMProvider.CBORG)
    agent = BioinformaticsAgent(config)
    
    # Test parsing user input
    message = agent._parse_user_input('analyze ATCGATCG')
    print('✅ Parsed message action:', message.content.get('action'))
    print('✅ Parsed message data:', message.content.get('data'))
    
    # Test help display  
    print('\n📖 Testing help display:')
    agent._show_help()
    
    print('\n🔧 Available tools:')
    for tool in agent.tools.list_tools():
        print(f'   • {tool}')

asyncio.run(test())