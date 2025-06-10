#!/usr/bin/env python3
"""
Test script for the Enhanced Universal MCP Agent
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.enhanced_agent import EnhancedUniversalMCPAgent, EnhancedAgentConfig
from src.llm_interface import LLMProvider


async def test_enhanced_agent():
    """Test the enhanced agent functionality"""
    
    print("🧪 Testing Enhanced Universal MCP Agent")
    print("="*50)
    
    # Create test configuration
    config = EnhancedAgentConfig(
        name="test-enhanced-agent",
        description="Test instance of Enhanced Universal MCP Agent",
        llm_provider=LLMProvider.CBORG,
        mcp_config_path="mcp_config.json",
        enable_reasoning=True,
        enable_planning=True,
        enable_progress_tracking=True,
        reports_directory="../reports/sophisticated_agent"
    )
    
    # Create agent
    try:
        agent = EnhancedUniversalMCPAgent(config)
        print("✅ Agent created successfully")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return
    
    # Test initialization
    try:
        await agent.initialize()
        print("✅ Agent initialized successfully")
        print(f"   📊 Discovered {len(agent.discovered_tools)} tools")
        print(f"   🔧 From {len(agent.mcp_servers)} servers")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Test simple reasoning without execution
    test_request = "analyze the mimivirus genome file and create a summary report"
    
    try:
        print(f"\n🧠 Testing reasoning for: '{test_request}'")
        
        reasoning_result = await agent.task_planner.reason_about_task(
            test_request, agent.discovered_tools
        )
        
        print("✅ Reasoning completed successfully")
        print(f"   🎯 Understanding: {reasoning_result.understanding[:100]}...")
        print(f"   📊 Complexity: {reasoning_result.complexity_assessment}")
        print(f"   🔧 Estimated Steps: {reasoning_result.estimated_steps}")
        
    except Exception as e:
        print(f"❌ Reasoning test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test planning
    try:
        print(f"\n📋 Testing planning...")
        
        from src.execution_models import PlanningContext
        
        planning_context = PlanningContext(
            reasoning_result=reasoning_result,
            available_tools=agent.discovered_tools,
            tool_categories=agent.tool_categories
        )
        
        plan = await agent.task_planner.create_execution_plan(
            reasoning_result, planning_context
        )
        
        print("✅ Planning completed successfully")
        print(f"   📋 Plan Goal: {plan.goal}")
        print(f"   📊 Complexity: {plan.complexity}")
        print(f"   🔧 Steps: {len(plan.steps)}")
        
        for i, step in enumerate(plan.steps[:3], 1):  # Show first 3 steps
            print(f"   {i}. {step.name} ({step.tool_name})")
        
        if len(plan.steps) > 3:
            print(f"   ... and {len(plan.steps) - 3} more steps")
            
    except Exception as e:
        print(f"❌ Planning test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test progress tracking
    try:
        print(f"\n📊 Testing progress tracking...")
        
        plan_id = agent.progress_tracker.start_plan(plan)
        print("✅ Progress tracking started")
        print(f"   📋 Plan ID: {plan_id}")
        
        # Test progress display
        progress_display = agent.progress_tracker.display_colored_progress(plan_id)
        print("✅ Progress display generated")
        
        # Test checklist
        checklist = agent.progress_tracker.get_checklist_display(plan_id)
        print(f"✅ Checklist generated with {len(checklist)} items")
        
    except Exception as e:
        print(f"❌ Progress tracking test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Enhanced Agent testing completed!")
    print("💡 To use the agent interactively, run:")
    print("   python src/enhanced_agent.py")


if __name__ == "__main__":
    asyncio.run(test_enhanced_agent())