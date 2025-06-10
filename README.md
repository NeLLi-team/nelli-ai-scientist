# NeLLi AI Scientist Agent Template

[![Version](https://img.shields.io/github/v/release/fschulz/nelli-ai-scientist?style=flat-square&color=blue)](https://github.com/fschulz/nelli-ai-scientist/releases)
[![License](https://img.shields.io/github/license/fschulz/nelli-ai-scientist?style=flat-square&color=green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](https://python.org)
[![FastMCP](https://img.shields.io/badge/FastMCP-1.0%2B-purple?style=flat-square)](https://github.com/phdowling/fastmcp)

A sophisticated Universal MCP Agent system designed for scientific AI applications with FastMCP integration and dynamic tool discovery.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository>
cd nelli-ai-scientist
pixi install

# Run the agent
pixi run agent-run

# Test with example
echo "analyze the DNA sequence ATCGATCGATCG" | pixi run agent-run
```

## 🧬 What You Get

The NeLLi AI Scientist Agent Template provides:

- **Universal MCP Agent**: Automatically discovers and uses tools from any MCP server
- **15 Pre-built Tools**: Bioinformatics + filesystem operations ready to use
- **Async Architecture**: Built on FastMCP for efficient concurrent operations
- **Intelligent Reflection**: Agent analyzes and interprets tool results
- **External Prompts**: Easily customize agent behavior without code changes

## 🏗️ System Architecture

```
You ──► Universal Agent ──► LLM Service (CBORG/Claude/OpenAI)
                │
                ▼
        FastMCP Clients
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
BioPython   Filesystem   Your Custom
 Tools       Tools        MCP Server
```

## 🛠️ Available Tools

**BioPython Tools (8 tools)**
- `sequence_stats` - Calculate sequence statistics (length, GC content, etc.)
- `analyze_fasta_file` - Comprehensive FASTA file analysis
- `blast_local` - Local BLAST sequence searches
- `multiple_alignment` - Multiple sequence alignment (ClustalW, MUSCLE)
- `phylogenetic_tree` - Build phylogenetic trees
- `translate_sequence` - DNA/RNA to protein translation
- `read_fasta_file` - Parse FASTA files
- `write_json_report` - Generate structured analysis reports

**Filesystem Tools (7 tools)**
- `read_file` - Read file contents
- `write_file` - Write to files  
- `list_directory` - List directory contents
- `create_directory` - Create directories
- `delete_file` - Delete files safely
- `file_exists` - Check file existence
- `explore_directory_tree` - Recursive directory exploration

## 🎯 Example Usage

### Natural Language Interface
```bash
pixi run agent-run
> analyze the DNA sequence ATCGATCGATCG and tell me about it

# Agent automatically:
# 1. Recognizes this needs sequence analysis
# 2. Uses sequence_stats tool
# 3. Reflects on results with biological context
# 4. Provides comprehensive analysis
```

### File Operations
```bash
> read the file example/mimivirus_genome.fna and provide sequence statistics

# Agent automatically:
# 1. Uses read_fasta_file to load sequences
# 2. Uses sequence_stats for analysis
# 3. Reflects on the biological significance
# 4. Provides detailed genomic analysis
```

### Multi-Tool Workflows
```bash
> analyze all FASTA files in the example directory and create a summary report

# Agent automatically:
# 1. Uses explore_directory_tree to find FASTA files
# 2. Uses analyze_fasta_file for each file
# 3. Uses write_json_report to create summary
# 4. Reflects on comparative genomic insights
```

## 🔧 Building Custom MCP Servers

Create domain-specific tools with FastMCP:

```python
# mcps/my_domain/src/server.py
from fastmcp import FastMCP

mcp = FastMCP("My Domain Tools 🧪")

@mcp.tool
async def my_analysis_tool(data: str, method: str = "standard") -> dict:
    """Analyze data using my custom method
    
    Args:
        data: Input data to analyze
        method: Analysis method (standard, advanced, custom)
    """
    # Your custom analysis logic here
    result = perform_analysis(data, method)
    
    return {
        "analysis_result": result,
        "method_used": method,
        "confidence": 0.95
    }

if __name__ == "__main__":
    mcp.run()
```

Register with the agent in `agents/template/mcp_config.json`:

```json
{
  "mcp_servers": {
    "my_domain": {
      "name": "My Domain Tools",
      "fastmcp_script": "../../mcps/my_domain/src/server.py",
      "enabled": true
    }
  }
}
```

## 📁 Repository Structure

```
nelli-ai-scientist/
├── agents/template/              # Universal MCP Agent
│   ├── src/
│   │   ├── agent.py             # UniversalMCPAgent implementation
│   │   ├── llm_interface.py     # LLM integration (CBORG/Claude/OpenAI)
│   │   └── prompt_manager.py    # External prompt management
│   ├── prompts/                 # External prompt files
│   └── mcp_config.json         # MCP server configuration
├── mcps/                        # FastMCP Servers
│   ├── template/src/           # BioPython tools
│   │   ├── server_fastmcp.py   # FastMCP server implementation
│   │   ├── biotools.py         # Tool implementations
│   │   └── tool_schema.py      # API documentation
│   └── filesystem/src/         # File operations
├── docs/                       # Comprehensive documentation
├── pixi.toml                   # Environment and tasks
└── README.md                   # This file
```

## 🎓 Learning Path

**Beginner (30 minutes)**
1. Run the agent and try example queries
2. Explore available tools with `tools` command
3. Modify prompts to change agent personality

**Intermediate (2 hours)**  
1. Create a simple FastMCP server with 1-2 tools
2. Integrate with agent configuration
3. Test multi-tool workflows

**Advanced (Half day)**
1. Build domain-specific MCP server with async tools
2. Implement custom reflection logic
3. Create sophisticated analysis pipelines

## 📚 Documentation

- **[Hackathon Quick Start](docs/hackathon-quick-start.md)** - Essential 15-minute guide
- **[Architecture Overview](docs/architecture-overview.md)** - System design and concepts
- **[FastMCP Server Development](docs/fastmcp-server-development.md)** - Building custom tools
- **[Agent Customization](docs/agent-customization.md)** - Domain-specific modifications
- **[Advanced Agent Concepts](docs/advanced-agent-concepts.md)** - Self-evolving systems
- **[Standards & Best Practices](docs/standards-and-best-practices.md)** - Code quality guidelines
- **[Troubleshooting Guide](docs/troubleshooting.md)** - Common issues and solutions
- **[Pixi Environment Setup](docs/pixi-setup.md)** - Environment management

## 🔧 Configuration

### Environment Variables (.env)
```bash
CBORG_API_KEY="your-key-here"           # Required: Your CBORG API key
CBORG_BASE_URL="https://api.cborg.lbl.gov"  # CBORG API endpoint
CBORG_MODEL="google/gemini-flash-lite"  # Default model
```

### Available Commands
```bash
# Core Usage
pixi run agent-run        # Run the NeLLi AI Agent
pixi run agent-test       # Quick functionality test

# Development
pixi run lint            # Check code quality
pixi run format          # Format code
pixi run test            # Run test suite

# MCP Server Testing
pixi run test-biopython  # Test BioPython MCP server
pixi run test-filesystem # Test filesystem MCP server
```

## 🎯 Key Features

- **Async-First**: Built on asyncio for efficient concurrent operations
- **Universal Tool Discovery**: Works with any MCP server without modification
- **Reflective Analysis**: Intelligent interpretation of tool results
- **Multi-Server Support**: Integrate multiple specialized MCP servers
- **External Prompt System**: Customize behavior without code changes
- **Scientific Focus**: Pre-configured with bioinformatics and data analysis tools

## 💡 Pro Tips

1. **Start Simple**: Begin with basic tool calls, then build complexity
2. **Use Reflection**: The agent's analysis of results is often more valuable than raw tool output
3. **Async Everything**: FastMCP's async nature enables powerful concurrent processing  
4. **External Prompts**: Customize behavior without code changes
5. **Tool Schemas**: Well-documented tools get better usage from the LLM

## 🚀 Ready to Build!

The NeLLi AI Scientist Agent Template provides everything you need to build sophisticated AI scientist agents:

- ✅ Universal MCP Agent with 15 pre-built tools
- ✅ FastMCP integration for efficient async operations
- ✅ Intelligent reflection and analysis capabilities
- ✅ External prompt management for easy customization
- ✅ Comprehensive documentation and examples
- ✅ Scientific computing focus with bioinformatics tools

**Start building the future of AI-assisted scientific research!** 🧬🚀