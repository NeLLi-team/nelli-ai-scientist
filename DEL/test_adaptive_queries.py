#!/usr/bin/env python3
"""
Test script to verify the agent properly uses adaptive reasoning
for analytical queries like "what's the average gene length?"
"""

import asyncio
import json
from pathlib import Path

# Add the agent directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent / "agents/sophisticated_agent/src"))

from agent_stdio import UniversalMCPAgentStdio, AgentConfig
from llm_interface import LLMProvider
from prompt_manager import PromptManager


async def test_analytical_query():
    """Test that the agent uses adaptive workflow for analytical queries"""
    
    print("🧪 Testing Agent's Adaptive Analysis Capabilities\n")
    
    # Create agent config
    config = AgentConfig(
        name="test-agent",
        description="Test agent for analytical queries",
        mcp_config_path="agents/sophisticated_agent/mcp_config.json",
        llm_provider=LLMProvider.CBORG
    )
    
    # Create agent
    agent = UniversalMCPAgentStdio(config)
    
    # Initialize agent
    print("🚀 Initializing agent...")
    await agent.initialize()
    
    # Test scenario: Previous analysis exists
    test_context = """
    The user previously analyzed a FASTA file:
    - AC3300027503___Ga0255182_1000024.fna: /home/fschulz/dev/nelli-ai-scientist/example/AC3300027503___Ga0255182_1000024.fna
    
    Analysis results were saved to: /home/fschulz/dev/nelli-ai-scientist/reports/AC3300027503___Ga0255182_1000024_analysis.json
    
    The analysis included gene prediction with 221 genes identified.
    """
    
    # Test query that requires calculation
    test_query = "what's the average gene length?"
    
    print(f"\n📝 Test Context:\n{test_context}")
    print(f"\n❓ Test Query: {test_query}")
    print("\n" + "="*60)
    
    # Process the query
    print("\n🤔 Processing query through tool selection...")
    
    # Get the prompt manager
    prompt_manager = agent.prompt_manager
    
    # Format the tool selection prompt
    tools_context = json.dumps(agent.discovered_tools, indent=2)
    
    prompt = prompt_manager.format_prompt(
        "tool_selection",
        user_input=test_query,
        tools_context=tools_context,
        conversation_context=test_context
    )
    
    # Get LLM response
    response = await agent.llm.generate(prompt, temperature=0.1)
    
    print("\n🤖 Agent's Tool Selection Response:")
    print("="*60)
    
    try:
        result = json.loads(response)
        print(json.dumps(result, indent=2))
        
        # Check if agent selected adaptive workflow
        if result.get("response_type") == "use_tools":
            tools = result.get("suggested_tools", [])
            tool_names = [t["tool_name"] for t in tools]
            
            print(f"\n✅ Agent selected tools: {tool_names}")
            
            # Check for adaptive workflow
            if "read_file" in tool_names and ("create_analysis_code" in tool_names or "execute_code" in tool_names):
                print("🎉 SUCCESS: Agent correctly chose adaptive analysis workflow!")
                print("\nThe agent will:")
                for i, tool in enumerate(tools, 1):
                    print(f"  {i}. {tool['reason']}")
            else:
                print("❌ FAIL: Agent did not select adaptive workflow")
                print("Expected tools: read_file → create_analysis_code → execute_code")
        else:
            print(f"❌ FAIL: Agent chose {result.get('response_type')} instead of using tools")
            
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Failed to parse agent response as JSON: {e}")
        print(f"Raw response: {response}")
    
    print("\n" + "="*60)
    print("\n🏁 Test Complete!")


if __name__ == "__main__":
    asyncio.run(test_analytical_query())