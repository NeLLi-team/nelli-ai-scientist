#!/usr/bin/env python3
"""
BBMap Agent Integration Example

This script demonstrates how to integrate the BBMap MCP server with an AI agent
for automated bioinformatics workflows. This is a key part of building your
master orchestration agent.
"""

import asyncio
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List

# Mock MCP client for demonstration
class MockMCPClient:
    """
    Mock MCP client to demonstrate agent-MCP integration
    In a real scenario, this would be your actual MCP client
    """

    def __init__(self, server_config: Dict[str, Any]):
        self.server_config = server_config
        self.available_tools = [
            "map_reads",
            "quality_stats",
            "coverage_analysis",
            "filter_reads"
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate calling an MCP tool"""
        print(f"🔧 MCP Call: {tool_name}({arguments})")

        # Simulate different tool responses
        if tool_name == "quality_stats":
            return {
                "status": "success",
                "fastq_file": arguments["fastq_path"],
                "quality_stats": {
                    "total_reads": 1000,
                    "total_bases": 75000,
                    "average_length": 75.0,
                    "median_length": 75,
                    "mode_length": 75
                },
                "output_files": {
                    "stats": "quality_stats.txt",
                    "histogram": "quality_stats_hist.txt"
                }
            }

        elif tool_name == "filter_reads":
            return {
                "status": "success",
                "input_file": arguments["input_fastq"],
                "output_file": arguments["output_fastq"],
                "filter_params": {
                    "min_length": arguments.get("min_length", 50),
                    "min_quality": arguments.get("min_quality", 20.0)
                },
                "filter_stats": {
                    "input_reads": 1000,
                    "output_reads": 850,
                    "filtered_reads": 150,
                    "filtering_rate": 15.0
                }
            }

        elif tool_name == "map_reads":
            return {
                "status": "success",
                "output_sam": arguments["output_sam"],
                "mapping_stats": {
                    "reads_used": 850,
                    "mapped_reads": 765,
                    "mapping_rate": 90.0,
                    "average_identity": 96.5,
                    "average_coverage": 12.3
                }
            }

        elif tool_name == "coverage_analysis":
            return {
                "status": "success",
                "sam_file": arguments["sam_path"],
                "coverage_stats": {
                    "average_coverage": 12.3,
                    "coverage_std": 2.1,
                    "percent_covered": 88.5,
                    "plus_reads": 380,
                    "minus_reads": 385
                }
            }

        return {"status": "error", "message": "Unknown tool"}


class BioinformaticsAgent:
    """
    Example AI Agent that uses BBMap MCP server for bioinformatics workflows
    """

    def __init__(self, mcp_client: MockMCPClient):
        self.mcp_client = mcp_client
        self.workflow_history = []
        self.current_files = {}

    async def analyze_sequencing_data(self, reference_path: str, reads_path: str) -> Dict[str, Any]:
        """
        Complete sequencing data analysis workflow
        This demonstrates how an agent orchestrates multiple MCP tools
        """
        print("🤖 Agent: Starting comprehensive sequencing analysis...")
        print("=" * 60)

        workflow_results = {
            "input_files": {
                "reference": reference_path,
                "reads": reads_path
            },
            "steps": [],
            "final_summary": {}
        }

        # Step 1: Quality Assessment
        print("\n🔍 Agent Decision: First, I'll assess the quality of the raw reads")
        quality_result = await self.mcp_client.call_tool("quality_stats", {
            "fastq_path": reads_path,
            "output_prefix": "initial_quality"
        })

        workflow_results["steps"].append({
            "step": "quality_assessment",
            "result": quality_result
        })

        # Agent reasoning based on results
        total_reads = quality_result["quality_stats"]["total_reads"]
        avg_length = quality_result["quality_stats"]["average_length"]

        print(f"🧠 Agent Analysis: Found {total_reads} reads with average length {avg_length}bp")

        # Step 2: Conditional filtering
        if avg_length < 100:
            print("🧠 Agent Decision: Reads are relatively short, applying quality filtering")
            filter_result = await self.mcp_client.call_tool("filter_reads", {
                "input_fastq": reads_path,
                "output_fastq": "filtered_reads.fastq",
                "min_length": 30,
                "min_quality": 20.0,
                "additional_params": "qtrim=rl trimq=20"
            })

            workflow_results["steps"].append({
                "step": "read_filtering",
                "result": filter_result
            })

            filtered_reads = filter_result["filter_stats"]["output_reads"]
            filtering_rate = filter_result["filter_stats"]["filtering_rate"]

            print(f"🧠 Agent Analysis: Filtered {filtering_rate}% of reads, {filtered_reads} remain")

            # Update file path for next step
            reads_for_mapping = filter_result["output_file"]
        else:
            print("🧠 Agent Decision: Read quality is good, skipping filtering")
            reads_for_mapping = reads_path

        # Step 3: Read Mapping
        print("🧠 Agent Decision: Now mapping reads to reference genome")
        mapping_result = await self.mcp_client.call_tool("map_reads", {
            "reference_path": reference_path,
            "reads_path": reads_for_mapping,
            "output_sam": "alignment.sam",
            "additional_params": "minid=0.95 maxindel=3"
        })

        workflow_results["steps"].append({
            "step": "read_mapping",
            "result": mapping_result
        })

        mapping_rate = mapping_result["mapping_stats"]["mapping_rate"]
        avg_identity = mapping_result["mapping_stats"]["average_identity"]

        print(f"🧠 Agent Analysis: Achieved {mapping_rate}% mapping rate with {avg_identity}% identity")

        # Step 4: Coverage Analysis
        print("🧠 Agent Decision: Analyzing genome coverage from alignments")
        coverage_result = await self.mcp_client.call_tool("coverage_analysis", {
            "sam_path": mapping_result["output_sam"],
            "reference_path": reference_path,
            "output_prefix": "final_coverage"
        })

        workflow_results["steps"].append({
            "step": "coverage_analysis",
            "result": coverage_result
        })

        avg_coverage = coverage_result["coverage_stats"]["average_coverage"]
        percent_covered = coverage_result["coverage_stats"]["percent_covered"]

        print(f"🧠 Agent Analysis: Average coverage {avg_coverage}x, {percent_covered}% genome covered")

        # Agent's final assessment
        workflow_results["final_summary"] = await self._generate_final_assessment(workflow_results)

        return workflow_results

    async def _generate_final_assessment(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Agent generates final assessment of the analysis"""
        print("\n🎯 Agent: Generating final assessment...")

        # Extract key metrics
        steps = workflow_results["steps"]
        quality_step = next(s for s in steps if s["step"] == "quality_assessment")
        mapping_step = next(s for s in steps if s["step"] == "read_mapping")
        coverage_step = next(s for s in steps if s["step"] == "coverage_analysis")

        # Agent's assessment logic
        mapping_rate = mapping_step["result"]["mapping_stats"]["mapping_rate"]
        coverage = coverage_step["result"]["coverage_stats"]["average_coverage"]
        percent_covered = coverage_step["result"]["coverage_stats"]["percent_covered"]

        # Quality assessment
        if mapping_rate >= 85:
            mapping_quality = "excellent"
        elif mapping_rate >= 70:
            mapping_quality = "good"
        elif mapping_rate >= 50:
            mapping_quality = "acceptable"
        else:
            mapping_quality = "poor"

        if coverage >= 20:
            coverage_quality = "high"
        elif coverage >= 10:
            coverage_quality = "moderate"
        elif coverage >= 5:
            coverage_quality = "low"
        else:
            coverage_quality = "very_low"

        assessment = {
            "overall_quality": "good" if mapping_quality in ["excellent", "good"] and coverage_quality in ["high", "moderate"] else "needs_improvement",
            "mapping_assessment": {
                "rate": mapping_rate,
                "quality": mapping_quality,
                "recommendation": "Good mapping quality" if mapping_rate >= 70 else "Consider parameter optimization"
            },
            "coverage_assessment": {
                "depth": coverage,
                "breadth": percent_covered,
                "quality": coverage_quality,
                "recommendation": "Sufficient coverage" if coverage >= 10 else "Consider increasing sequencing depth"
            },
            "next_steps": self._suggest_next_steps(mapping_rate, coverage, percent_covered)
        }

        return assessment

    def _suggest_next_steps(self, mapping_rate: float, coverage: float, percent_covered: float) -> List[str]:
        """Agent suggests next steps based on analysis results"""
        suggestions = []

        if mapping_rate < 70:
            suggestions.append("Consider adjusting mapping parameters or checking reference quality")

        if coverage < 10:
            suggestions.append("Consider increasing sequencing depth or improving library preparation")

        if percent_covered < 80:
            suggestions.append("Investigate regions with low coverage - may indicate repetitive elements")

        if mapping_rate >= 85 and coverage >= 15:
            suggestions.append("Data quality is excellent - proceed with downstream analysis")
            suggestions.append("Consider variant calling or comparative genomics")

        return suggestions


async def demonstrate_agent_mcp_integration():
    """
    Demonstrate how an agent integrates with BBMap MCP server
    """
    print("🚀 BBMap Agent-MCP Integration Demo")
    print("=" * 60)

    # Initialize MCP client (would be real MCP client in practice)
    mcp_config = {
        "server_name": "bbmap",
        "command": "python -m src.server_fastmcp",
        "cwd": "/pscratch/sd/j/jvillada/nelli-ai-scientist/mcps/bbmap"
    }
    mcp_client = MockMCPClient(mcp_config)

    # Initialize agent
    agent = BioinformaticsAgent(mcp_client)

    # Create sample data paths (would be real data in practice)
    reference_path = "/data/reference_genome.fasta"
    reads_path = "/data/sequencing_reads.fastq"

    print(f"📁 Input Data:")
    print(f"   Reference: {reference_path}")
    print(f"   Reads: {reads_path}")

    # Run agent workflow
    results = await agent.analyze_sequencing_data(reference_path, reads_path)

    # Display results
    print("\n📊 Final Results Summary:")
    print("=" * 60)

    summary = results["final_summary"]
    print(f"🎯 Overall Quality: {summary['overall_quality']}")

    print(f"\n📈 Mapping Assessment:")
    mapping = summary["mapping_assessment"]
    print(f"   Rate: {mapping['rate']}%")
    print(f"   Quality: {mapping['quality']}")
    print(f"   Recommendation: {mapping['recommendation']}")

    print(f"\n📊 Coverage Assessment:")
    coverage = summary["coverage_assessment"]
    print(f"   Depth: {coverage['depth']}x")
    print(f"   Breadth: {coverage['breadth']}%")
    print(f"   Quality: {coverage['quality']}")
    print(f"   Recommendation: {coverage['recommendation']}")

    print(f"\n🔮 Agent Suggestions:")
    for i, suggestion in enumerate(summary["next_steps"], 1):
        print(f"   {i}. {suggestion}")

    print("\n✨ Integration Benefits Demonstrated:")
    print("   ✅ Agent makes intelligent decisions based on data")
    print("   ✅ Conditional workflow execution")
    print("   ✅ Automated quality assessment")
    print("   ✅ Actionable recommendations")
    print("   ✅ Orchestration of multiple MCP tools")


async def demonstrate_multi_agent_orchestration():
    """
    Demonstrate how multiple agents could work together using BBMap MCP
    """
    print("\n🌐 Multi-Agent Orchestration Example")
    print("=" * 60)

    print("🤖 Master Agent: Coordinating specialized agents...")

    agents = {
        "quality_agent": "Specializes in read quality assessment",
        "mapping_agent": "Specializes in read mapping optimization",
        "coverage_agent": "Specializes in coverage analysis",
        "report_agent": "Specializes in generating comprehensive reports"
    }

    print("\n🎭 Agent Roles:")
    for agent_name, role in agents.items():
        print(f"   • {agent_name}: {role}")

    print("\n🔄 Orchestration Flow:")
    print("   1. Master Agent receives analysis request")
    print("   2. Delegates quality assessment to Quality Agent")
    print("   3. Based on quality results, consults Mapping Agent")
    print("   4. Mapping Agent optimizes parameters and runs mapping")
    print("   5. Coverage Agent analyzes results and identifies issues")
    print("   6. Report Agent compiles comprehensive analysis")
    print("   7. Master Agent coordinates final recommendations")

    print("\n🏗️ Architecture Benefits:")
    print("   ✅ Specialized expertise in each agent")
    print("   ✅ Scalable and maintainable")
    print("   ✅ Reusable components across projects")
    print("   ✅ Easy to add new agents (e.g., variant calling agent)")


if __name__ == "__main__":
    print("🧬 BBMap MCP Server - Agent Integration Tutorial")
    print("=" * 80)

    # Run the demonstrations
    asyncio.run(demonstrate_agent_mcp_integration())
    asyncio.run(demonstrate_multi_agent_orchestration())

    print("\n🎯 Key Takeaways:")
    print("=" * 80)
    print("1. 🔧 MCP servers provide specialized tools (BBMap functionality)")
    print("2. 🤖 Agents make intelligent decisions about when/how to use tools")
    print("3. 🎼 Orchestration allows complex workflows with multiple agents")
    print("4. 📊 Each component is testable and reusable")
    print("5. 🔄 The pattern scales to any bioinformatics workflow")

    print("\n🚀 Your Next Steps:")
    print("1. Build additional MCP servers for other bioinformatics tools")
    print("2. Create specialized agents for different analysis types")
    print("3. Implement a master orchestration agent")
    print("4. Add real MCP client integration")
    print("5. Scale to handle large-scale genomics projects")
