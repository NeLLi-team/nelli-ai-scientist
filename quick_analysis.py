#!/usr/bin/env python3
"""
Quick sequence analysis script for the enhanced agent
This provides a simple way to run the analysis that was requested
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the agent src directory to the path
sys.path.insert(0, "agents/sophisticated_agent/src")

from enhanced_agent import EnhancedUniversalMCPAgent, EnhancedAgentConfig
from config_loader import ConfigLoader

async def run_sequence_analysis():
    """Run the sequence analysis that was requested"""
    
    print("🧬 Running Sequence Analysis of contigs100k.fna")
    print("=" * 60)
    
    # Create config loader
    agent_dir = "agents/sophisticated_agent"
    config_file_path = os.path.join(agent_dir, "config/agent_config.yaml")
    
    config_loader = ConfigLoader(config_file_path)
    config = EnhancedAgentConfig(
        name=config_loader.get_agent_name(),
        mcp_config_path="mcp_config.json",
        config_file=config_file_path
    )
    
    print("🤖 Initializing enhanced agent...")
    agent = EnhancedUniversalMCPAgent(config, config_loader)
    
    try:
        # Initialize agent
        await agent.initialize()
        print(f"✅ Agent initialized with {len(agent.discovered_tools)} tools")
        
        # Test 1: Analyze FASTA file directly
        print("\n🔬 Step 1: Analyzing FASTA file...")
        fasta_path = "data/nelli_hackathon/contigs100k.fna"
        
        try:
            result = await agent._call_mcp_tool("analyze_fasta_file", path=fasta_path)
            if result.get("success"):
                print("✅ FASTA analysis completed!")
                analysis_data = result.get("result", {})
                
                print("\n📊 Analysis Results:")
                if isinstance(analysis_data, dict):
                    for key, value in analysis_data.items():
                        if isinstance(value, (int, float)):
                            if isinstance(value, float):
                                print(f"  • {key}: {value:.2f}")
                            else:
                                print(f"  • {key}: {value:,}")
                        else:
                            print(f"  • {key}: {value}")
                else:
                    print(f"  Raw result: {analysis_data}")
                
                # Test 2: Write report
                print(f"\n📝 Step 2: Writing analysis report...")
                report_result = await agent._call_mcp_tool(
                    "write_json_report", 
                    data=analysis_data,
                    filename="reports/contigs100k_analysis.json"
                )
                
                if report_result.get("success"):
                    print("✅ Report written successfully!")
                    print(f"📄 Report saved to: reports/contigs100k_analysis.json")
                else:
                    print(f"❌ Report failed: {report_result.get('error')}")
            else:
                print(f"❌ Analysis failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Tool execution error: {e}")
            
        # Test 3: Get sequence statistics  
        print(f"\n📈 Step 3: Getting sequence statistics...")
        try:
            stats_result = await agent._call_mcp_tool("sequence_stats", path=fasta_path)
            if stats_result.get("success"):
                print("✅ Sequence statistics completed!")
                stats_data = stats_result.get("result", {})
                
                print("\n📈 Sequence Statistics:")
                if isinstance(stats_data, dict):
                    for key, value in stats_data.items():
                        if isinstance(value, (int, float)):
                            if isinstance(value, float):
                                print(f"  • {key}: {value:.2f}")
                            else:
                                print(f"  • {key}: {value:,}")
                        else:
                            print(f"  • {key}: {value}")
            else:
                print(f"❌ Statistics failed: {stats_result.get('error')}")
                
        except Exception as e:
            print(f"❌ Statistics error: {e}")
            
    except Exception as e:
        print(f"❌ Agent error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure we have API key for testing
    if not os.getenv("CBORG_API_KEY"):
        # Load from .env file
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("CBORG_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"')
                        os.environ["CBORG_API_KEY"] = key
                        break
    
    asyncio.run(run_sequence_analysis())