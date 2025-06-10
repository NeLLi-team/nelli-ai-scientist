"""
Test client for BioPython MCP Server
"""

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_biopython_mcp():
    """Test the BioPython MCP server"""

    # Create server parameters
    server_params = StdioServerParameters(
        command="python", args=["-m", "src.server"], env=None
    )

    # Connect to server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")

            # Test sequence statistics
            print("\n1. Testing sequence statistics:")
            result = await session.call_tool(
                "sequence_stats",
                {"sequence": "ATCGATCGATCGATCGATCGATCG", "sequence_type": "dna"},
            )
            print(json.dumps(json.loads(result[0].text), indent=2))

            # Test translation
            print("\n2. Testing sequence translation:")
            result = await session.call_tool(
                "translate_sequence",
                {
                    "sequence": "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG",
                    "genetic_code": 1,
                },
            )
            print(json.dumps(json.loads(result[0].text), indent=2))

            # Test multiple alignment
            print("\n3. Testing multiple alignment:")
            result = await session.call_tool(
                "multiple_alignment",
                {
                    "sequences": [
                        {"id": "seq1", "sequence": "ATCGATCGATCG"},
                        {"id": "seq2", "sequence": "ATCGATGGATCG"},
                        {"id": "seq3", "sequence": "ATCGATTGATCG"},
                    ],
                    "algorithm": "clustalw",
                },
            )
            print(json.dumps(json.loads(result[0].text), indent=2))


if __name__ == "__main__":
    asyncio.run(test_biopython_mcp())
