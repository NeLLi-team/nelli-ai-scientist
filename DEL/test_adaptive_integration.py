#!/usr/bin/env python3
"""
Test the adaptive integration in the sophisticated agent
"""

import asyncio
import json
from pathlib import Path

# Add the agent source to path
import sys
sys.path.append('/home/fschulz/dev/nelli-ai-scientist/agents/sophisticated_agent/src')

from agent_stdio import UniversalMCPAgentStdio, AgentConfig

async def test_adaptive_integration():
    """Test that the adaptive code solver is properly integrated"""
    
    # Create agent config
    config = AgentConfig(
        name="test_adaptive_agent",
        description="Testing adaptive code integration",
        mcp_config_path="/home/fschulz/dev/nelli-ai-scientist/agents/sophisticated_agent/mcp_config.json"
    )
    
    # Create agent
    agent = UniversalMCPAgentStdio(config)
    
    # Initialize
    await agent.initialize()
    
    # Test adaptive solver detection
    test_requests = [
        "provide more details on the most frequently found tandem repeats",
        "analyze the gene length distribution",
        "show me the detailed statistics of the assembly",
        "calculate the frequency distribution of repeat units",
        "visualize the GC content distribution"
    ]
    
    print("🧪 Testing Adaptive Solver Integration")
    print("=" * 60)
    
    for request in test_requests:
        should_use_adaptive = agent._should_use_adaptive_solver(request)
        print(f"Request: {request[:50]}...")
        print(f"Use Adaptive Solver: {should_use_adaptive}")
        print("-" * 40)
    
    # Test with sample data
    sample_data = {
        "repeat_analysis": {
            "tandem_repeats": [
                {
                    "start": 1000,
                    "end": 1020,
                    "repeat_unit": "ATGC",
                    "copy_number": 5,
                    "total_length": 20
                }
            ]
        }
    }
    
    print("\n🔧 Testing Adaptive Code Solver Components")
    print("=" * 60)
    
    # Test adaptive solver directly
    try:
        solver = agent.adaptive_solver
        print(f"✅ Adaptive solver initialized: {type(solver).__name__}")
        
        # Test if it can analyze a request
        analysis = solver._analyze_user_request(
            "provide detailed analysis of tandem repeats", 
            sample_data
        )
        print(f"✅ Request analysis working: {analysis['solvable_with_code']}")
        
        print(f"✅ Request type: {analysis['request_type']}")
        print(f"✅ Key terms: {analysis['key_terms']}")
        
    except Exception as e:
        print(f"❌ Error testing adaptive solver: {e}")
    
    # Test biological engine
    try:
        bio_engine = agent.bio_engine
        print(f"✅ Biological engine initialized: {type(bio_engine).__name__}")
        
        # Test biological analysis
        bio_analysis = bio_engine.analyze_biological_data(sample_data, "tandem_repeats")
        print(f"✅ Biological analysis working: {bio_analysis['data_type']}")
        
    except Exception as e:
        print(f"❌ Error testing biological engine: {e}")
    
    # Cleanup
    await agent.cleanup()
    
    print("\n✅ Integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_adaptive_integration())