#!/usr/bin/env python3
"""
Test script for agent file handling functionality
"""

import asyncio
from src.agent import BioinformaticsAgent, AgentConfig
from src.llm_interface import LLMProvider


async def test_agent_tools():
    config = AgentConfig(
        name="test-agent",
        capabilities=["sequence_analysis", "file_handling"],
        llm_provider=LLMProvider.CBORG,
    )
    agent = BioinformaticsAgent(config)

    # Test FASTA file reading tool
    print("=== Testing agent FASTA file reading ===")
    result = await agent.tools.execute(
        "read_fasta_file",
        file_path="/home/fschulz/dev/nelli-ai-scientist/test_sequences.fasta",
    )
    print(f"Agent read {result['num_sequences']} sequences")
    for seq in result["sequences"][:2]:
        print(f"- {seq['id']}: {len(seq['sequence'])} bp")

    # Test JSON report writing tool
    print("\n=== Testing agent JSON report writing ===")
    test_data = {
        "analysis_type": "FASTA file analysis",
        "sequences_analyzed": result["num_sequences"],
        "total_length": sum(len(seq["sequence"]) for seq in result["sequences"]),
    }

    report_result = await agent.tools.execute(
        "write_json_report",
        data=test_data,
        output_path="/home/fschulz/dev/nelli-ai-scientist/agent_test_report.json",
    )
    print(f"Agent report written: {report_result}")

    print("\n=== All agent tools working! ===")


if __name__ == "__main__":
    asyncio.run(test_agent_tools())
