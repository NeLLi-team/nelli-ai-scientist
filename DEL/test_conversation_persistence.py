#!/usr/bin/env python3
"""
Test script for conversation persistence and parameter resolution
"""

import asyncio
import sys
import json
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "agents" / "sophisticated_agent" / "src"))

# Import modules directly
import llm_interface
import agent_stdio

UniversalMCPAgentStdio = agent_stdio.UniversalMCPAgentStdio
AgentConfig = agent_stdio.AgentConfig
LLMProvider = llm_interface.LLMProvider

async def test_conversation_persistence():
    """Test that conversation history is saved and parameter resolution works"""
    
    print("🧪 Testing Conversation Persistence and Parameter Resolution")
    print("=" * 60)
    
    # Create agent config
    config = AgentConfig(
        name="test-agent",
        role="Test Agent",
        description="Testing conversation persistence",
        llm_provider=LLMProvider.CBORG,
        temperature=0.7,
        max_tokens=4096,
        mcp_config_path="agents/sophisticated_agent/mcp_config.json",
        use_stdio_connections=True
    )
    
    # Initialize agent
    print("🚀 Initializing agent...")
    agent = UniversalMCPAgentStdio(config)
    await agent.initialize()
    
    # Test 1: Parameter resolution
    print("\n📋 Test 1: Parameter Resolution")
    
    # Mock previous results with find_file_by_name format
    mock_previous_results = [
        {
            "tool": "find_file_by_name",
            "result": {
                "found_files": [
                    {
                        "name": "contigs100k.fna",
                        "path": "/home/fschulz/dev/nelli-ai-scientist/data/nelli_hackathon/contigs100k.fna",
                        "size": 12191117
                    }
                ]
            }
        }
    ]
    
    # Test parameter resolution
    test_params = {"file_path": "USE_PATH_FROM_PREVIOUS_RESULT", "sequence_type": "dna"}
    resolved_params = agent._resolve_chained_parameters(test_params, mock_previous_results)
    
    print(f"Original params: {test_params}")
    print(f"Resolved params: {resolved_params}")
    
    expected_path = "/home/fschulz/dev/nelli-ai-scientist/data/nelli_hackathon/contigs100k.fna"
    if resolved_params.get("file_path") == expected_path:
        print("✅ Parameter resolution working correctly!")
    else:
        print("❌ Parameter resolution failed!")
        print(f"Expected: {expected_path}")
        print(f"Got: {resolved_params.get('file_path')}")
    
    # Test 2: Conversation persistence
    print("\n📋 Test 2: Conversation Persistence")
    
    # Add some conversation entries
    user_entry = {
        "role": "user",
        "content": "list sequence files in data directory",
        "timestamp": "2025-06-11T21:50:00"
    }
    
    assistant_entry = {
        "role": "assistant", 
        "content": "I found several sequence files in the data directory.",
        "tool_calls": [
            {
                "tool": "find_files",
                "result": {
                    "found_files": [
                        {"name": "contigs100k.fna", "path": "/data/nelli_hackathon/contigs100k.fna"}
                    ]
                }
            }
        ],
        "timestamp": "2025-06-11T21:50:05"
    }
    
    # Add to conversation history and save
    agent.conversation_history.append(user_entry)
    agent._save_conversation_entry(user_entry)
    
    agent.conversation_history.append(assistant_entry)
    agent._save_conversation_entry(assistant_entry)
    
    # Check if chat log file exists and has content
    if agent.chat_log_file.exists():
        print(f"✅ Chat log file created: {agent.chat_log_file}")
        
        # Read log file
        with open(agent.chat_log_file, 'r') as f:
            lines = f.readlines()
            print(f"📝 Log file has {len(lines)} entries")
            
            # Check first entry
            if lines:
                first_entry = json.loads(lines[0])
                print(f"📄 First entry: {first_entry['role']} - {first_entry['content'][:50]}...")
    else:
        print("❌ Chat log file not created!")
    
    # Test 3: Context summary
    print("\n📋 Test 3: Context Summary")
    
    context_summary = agent._get_recent_context_summary()
    print(f"📋 Context summary:\n{context_summary}")
    
    if "contigs100k.fna" in context_summary:
        print("✅ Context summary includes file information!")
    else:
        print("❌ Context summary missing file information")
    
    # Cleanup
    await agent.cleanup()
    
    print("\n🏁 Testing completed!")

if __name__ == "__main__":
    asyncio.run(test_conversation_persistence())