#!/usr/bin/env python3
"""
Scientific MCP Client
Production-ready stdio client for bioinformatics workflows using independent MCPs
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import traceback

from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession
import mcp.types as types

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScientificMCPClient:
    """
    Production client for scientific MCP servers
    Handles connection lifecycle properly and provides simple interface
    """
    
    def __init__(self):
        self.available_servers = {
            "bioseq": {
                "name": "Nucleic Acid Analysis",
                "command": "pixi",
                "args": ["run", "--manifest-path", "mcps/bioseq", "run"],
                "description": "DNA/RNA sequence analysis, assembly stats, promoter detection"
            },
            "filesystem": {
                "name": "File Operations", 
                "command": "pixi",
                "args": ["run", "--manifest-path", "mcps/filesystem", "run-simple"],
                "description": "Safe file and directory operations"
            },
            "json_tools": {
                "name": "JSON Processing",
                "command": "npx",
                "args": ["--yes", "@gongrzhe/server-json-mcp@1.0.3"],
                "description": "JSON querying and manipulation"
            }
        }
        
    async def connect_and_call(self, server_id: str, tool_name: str, **kwargs) -> Any:
        """
        Connect to server, call tool, and disconnect cleanly
        This is the main method for scientific workflows
        """
        if server_id not in self.available_servers:
            raise ValueError(f"Unknown server: {server_id}. Available: {list(self.available_servers.keys())}")
        
        server_config = self.available_servers[server_id]
        logger.info(f"🔌 Connecting to {server_config['name']} for {tool_name}")
        
        # Create server parameters
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"]
        )
        
        try:
            # Connect and call within context managers
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Call the tool
                    result = await session.call_tool(tool_name, kwargs)
                    
                    # Extract content from MCP result
                    if hasattr(result, 'content') and result.content:
                        if len(result.content) == 1 and hasattr(result.content[0], 'text'):
                            # Single text content
                            content = result.content[0].text
                            try:
                                # Try to parse as JSON
                                return json.loads(content)
                            except:
                                # Return as string if not JSON
                                return content
                        else:
                            # Multiple content items
                            return [item.text if hasattr(item, 'text') else str(item) for item in result.content]
                    else:
                        return str(result)
                        
        except Exception as e:
            logger.error(f"❌ Error calling {tool_name} on {server_id}: {e}")
            raise
    
    async def list_tools(self, server_id: str) -> List[Dict[str, Any]]:
        """List available tools from a server"""
        if server_id not in self.available_servers:
            raise ValueError(f"Unknown server: {server_id}")
        
        server_config = self.available_servers[server_id]
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"]
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    tools_result = await session.list_tools()
                    tools = tools_result.tools if hasattr(tools_result, 'tools') else tools_result
                    
                    return [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema.get("properties", {}) if hasattr(tool, 'inputSchema') else {}
                        }
                        for tool in tools
                    ]
        except Exception as e:
            logger.error(f"❌ Error listing tools from {server_id}: {e}")
            raise
    
    def print_available_servers(self):
        """Print information about available servers"""
        print("\n🔬 Available Scientific MCP Servers:")
        print("=" * 60)
        for server_id, config in self.available_servers.items():
            print(f"\n📡 {config['name']} (ID: {server_id})")
            print(f"   Description: {config['description']}")
            print(f"   Command: {config['command']} {' '.join(config['args'])}")
    
    async def print_server_tools(self, server_id: str):
        """Print available tools for a server"""
        try:
            tools = await self.list_tools(server_id)
            server_name = self.available_servers[server_id]['name']
            
            print(f"\n🔧 {server_name} Tools ({len(tools)} available):")
            print("=" * 60)
            
            for tool in tools:
                print(f"\n• {tool['name']}")
                print(f"  Description: {tool['description']}")
                if tool['parameters']:
                    print(f"  Parameters: {', '.join(tool['parameters'].keys())}")
                    
        except Exception as e:
            print(f"❌ Error getting tools: {e}")


class BioinformaticsWorkflows:
    """
    High-level bioinformatics workflows using the MCP client
    """
    
    def __init__(self):
        self.client = ScientificMCPClient()
    
    async def analyze_sequence(self, sequence: str, sequence_type: str = "dna") -> Dict[str, Any]:
        """Simple sequence analysis"""
        logger.info(f"🧬 Analyzing {sequence_type.upper()} sequence ({len(sequence)} bp)")
        
        result = await self.client.connect_and_call(
            "bioseq", 
            "sequence_stats",
            sequence=sequence,
            sequence_type=sequence_type
        )
        return result
    
    async def analyze_fasta_file(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive FASTA file analysis"""
        logger.info(f"📁 Analyzing FASTA file: {file_path}")
        
        result = await self.client.connect_and_call(
            "bioseq",
            "analyze_fasta_file", 
            file_path=file_path,
            sequence_type="dna"
        )
        return result
    
    async def find_giant_virus_promoters(self, sequences: List[Dict[str, str]]) -> Dict[str, Any]:
        """Search for giant virus promoter motifs"""
        logger.info(f"🦠 Searching for giant virus promoters in {len(sequences)} sequences")
        
        result = await self.client.connect_and_call(
            "bioseq",
            "giant_virus_promoter_search",
            sequences=sequences,
            upstream_length=150
        )
        return result
    
    async def calculate_assembly_stats(self, sequences: List[Dict[str, str]]) -> Dict[str, Any]:
        """Calculate assembly statistics (N50, L50, etc.)"""
        logger.info(f"📊 Calculating assembly stats for {len(sequences)} sequences")
        
        result = await self.client.connect_and_call(
            "bioseq",
            "assembly_stats",
            sequences=sequences
        )
        return result
    
    async def gc_skew_analysis(self, sequences: List[Dict[str, str]], window_size: int = 10000) -> Dict[str, Any]:
        """GC skew analysis for replication origin detection"""
        logger.info(f"📈 Performing GC skew analysis (window: {window_size} bp)")
        
        result = await self.client.connect_and_call(
            "bioseq",
            "gc_skew_analysis",
            sequences=sequences,
            window_size=window_size,
            step_size=window_size//2
        )
        return result
    
    async def save_results(self, data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Save analysis results to JSON file"""
        logger.info(f"💾 Saving results to: {output_path}")
        
        result = await self.client.connect_and_call(
            "bioseq",
            "write_json_report",
            data=data,
            output_path=output_path
        )
        return result


async def interactive_demo():
    """Interactive demonstration of the scientific MCP client"""
    client = ScientificMCPClient()
    workflows = BioinformaticsWorkflows()
    
    print("\n🧪 Scientific MCP Client - Interactive Demo")
    print("=" * 60)
    
    # Show available servers
    client.print_available_servers()
    
    while True:
        print("\n💡 What would you like to do?")
        print("1. List tools from a server")
        print("2. Analyze a DNA sequence")
        print("3. Analyze FASTA file")
        print("4. Test with example sequences")
        print("5. Quit")
        
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                server_id = input("Enter server ID (bioseq/filesystem/json_tools): ").strip()
                if server_id in client.available_servers:
                    await client.print_server_tools(server_id)
                else:
                    print("❌ Invalid server ID")
            
            elif choice == "2":
                sequence = input("Enter DNA sequence: ").strip().upper()
                if sequence and all(c in "ATCGN" for c in sequence):
                    result = await workflows.analyze_sequence(sequence)
                    print(f"\n📊 Analysis Results:")
                    print(json.dumps(result, indent=2))
                else:
                    print("❌ Invalid DNA sequence")
            
            elif choice == "3":
                file_path = input("Enter FASTA file path: ").strip()
                if Path(file_path).exists():
                    try:
                        result = await workflows.analyze_fasta_file(file_path)
                        print(f"\n📊 FASTA Analysis Results:")
                        print(json.dumps(result, indent=2))
                    except Exception as e:
                        print(f"❌ Error analyzing file: {e}")
                else:
                    print("❌ File not found")
            
            elif choice == "4":
                print("🧬 Testing with example sequences...")
                
                # Example sequences
                test_sequences = [
                    {
                        "id": "test_seq_1",
                        "sequence": "ATCGATCGATCGATCGTATATAAAATTGAATCGATCGATCGAAAAAATTGGATCGATCGTATATAAATCGATCGATCG"
                    },
                    {
                        "id": "test_seq_2", 
                        "sequence": "CGATCGATCGATCGCGCGCGCGCGCGATCGATCGAAAATTGAATCGATCGATATATATATATATATATCGATCGATCG"
                    }
                ]
                
                # Run multiple analyses
                print("\n1. Assembly Statistics:")
                assembly_stats = await workflows.calculate_assembly_stats(test_sequences)
                print(json.dumps(assembly_stats, indent=2))
                
                print("\n2. Giant Virus Promoter Search:")
                promoters = await workflows.find_giant_virus_promoters(test_sequences)
                print(f"Found {promoters.get('summary', {}).get('total_motifs_found', 0)} motifs")
                
                print("\n3. Individual Sequence Analysis:")
                seq_result = await workflows.analyze_sequence(test_sequences[0]["sequence"])
                print(f"Sequence 1: {seq_result['length']} bp, GC: {seq_result['gc_content']}%")
                
            elif choice == "5":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()


async def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        await interactive_demo()
    else:
        print("🧪 Scientific MCP Client")
        print("=" * 40)
        print("Usage:")
        print("  python scientific_mcp_client.py demo    # Interactive demo")
        print("")
        print("Example programmatic usage:")
        print("""
import asyncio
from scientific_mcp_client import BioinformaticsWorkflows

async def example():
    workflows = BioinformaticsWorkflows()
    
    # Analyze a sequence
    result = await workflows.analyze_sequence("ATCGATCGATCG")
    print(f"GC content: {result['gc_content']}%")
    
    # Analyze FASTA file
    result = await workflows.analyze_fasta_file("example.fna")
    print(f"N50: {result['assembly_stats']['n50']} bp")

asyncio.run(example())
        """)


if __name__ == "__main__":
    asyncio.run(main())