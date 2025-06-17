#!/usr/bin/env python3
"""
Test the reasoning model integration in the sophisticated agent
"""

import asyncio
import json
from pathlib import Path

# Add the agent source to path  
import sys
sys.path.append('/home/fschulz/dev/nelli-ai-scientist/agents/sophisticated_agent/src')

async def test_reasoning_model():
    """Test that the reasoning model is properly integrated"""
    
    print("🧪 Testing Reasoning Model Integration")
    print("=" * 60)
    
    try:
        # Test imports
        from agent_stdio import UniversalMCPAgentStdio, AgentConfig
        print("✅ Agent imports successful")
        
        from task_planner import TaskPlanner
        from execution_models import TaskComplexity, ExecutionPlan
        print("✅ Reasoning components available")
        
        # Create test agent config
        config = AgentConfig(
            name="test_reasoning_agent",
            description="Testing reasoning model integration",
            mcp_config_path="/home/fschulz/dev/nelli-ai-scientist/agents/sophisticated_agent/mcp_config.json"
        )
        
        # Create agent 
        agent = UniversalMCPAgentStdio(config)
        print("✅ Agent created successfully")
        
        # Test reasoning model method
        test_requests = [
            "What percentage of the genome consists of tandem repeats?",
            "Show me detailed statistics about the most frequent repeats",
            "Calculate the GC content distribution",
            "Analyze the assembly quality metrics"
        ]
        
        # Load sample data
        sample_file = Path("/home/fschulz/dev/nelli-ai-scientist/reports/AC3300027503___Ga0255182_1000024_analysis.json")
        if sample_file.exists():
            with open(sample_file, 'r') as f:
                sample_data = json.load(f)
            print("✅ Sample data loaded")
        else:
            sample_data = {}
            print("⚠️ No sample data available")
        
        print("\n" + "🧠 Testing Reasoning Model Responses" + "\n" + "="*60)
        
        for i, request in enumerate(test_requests, 1):
            print(f"\n**Test {i}:** {request}")
            print("-" * 40)
            
            try:
                # Test the reasoning model method directly
                response = await agent._solve_with_reasoning_model(request, sample_data)
                
                if response:
                    print("✅ Reasoning model responded:")
                    print(response[:200] + "..." if len(response) > 200 else response)
                else:
                    print("❌ Reasoning model returned None")
                    
                    # Test fallback mechanisms
                    if sample_data:
                        bio_response = agent._provide_biological_intelligence(request, sample_data)
                        if bio_response:
                            print("✅ Biological fallback worked:")
                            print(bio_response[:200] + "..." if len(bio_response) > 200 else bio_response)
                        else:
                            print("❌ Biological fallback also failed")
                            
            except Exception as e:
                print(f"❌ Error testing request: {e}")
        
        print("\n" + "🔧 Testing Component Integration" + "\n" + "="*60)
        
        # Test task planner directly
        try:
            reasoning_llm = agent.llm_interface
            task_planner = TaskPlanner(reasoning_llm, agent.prompt_manager)
            print("✅ Task planner created")
            
            # Test reasoning about a simple task
            reasoning_result = await task_planner.reason_about_task(
                "Calculate the percentage of tandem repeats", 
                agent.discovered_tools
            )
            print(f"✅ Reasoning result: {reasoning_result.complexity_assessment}")
            
        except Exception as e:
            print(f"❌ Task planner error: {e}")
        
        # Test adaptive solver
        try:
            solver_result = await agent.adaptive_solver.solve_user_request(
                "Calculate the percentage of genome in tandem repeats",
                sample_data
            )
            
            if solver_result.get("success"):
                print("✅ Adaptive solver working")
            else:
                print(f"❌ Adaptive solver failed: {solver_result.get('reason', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Adaptive solver error: {e}")
        
        print("\n✅ Reasoning integration test completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Enhanced reasoning components may not be available")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    asyncio.run(test_reasoning_model())