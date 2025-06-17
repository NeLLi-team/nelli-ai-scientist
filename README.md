# NeLLi AI Scientist Agent Template

[![Version](https://img.shields.io/github/v/release/fschulz/nelli-ai-scientist?style=flat-square&color=blue)](https://github.com/fschulz/nelli-ai-scientist/releases)
[![License](https://img.shields.io/github/license/fschulz/nelli-ai-scientist?style=flat-square&color=green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](https://python.org)
[![FastMCP](https://img.shields.io/badge/FastMCP-1.0%2B-purple?style=flat-square)](https://github.com/phdowling/fastmcp)

A sophisticated Universal MCP Agent system with **enhanced reasoning, planning, and progress tracking** designed for scientific AI applications with FastMCP integration and dynamic tool discovery.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository>
cd nelli-ai-scientist
pixi install

# Run the Enhanced Sophisticated Agent (RECOMMENDED)
pixi run sophisticated-agent

# Or run the basic template agent
pixi run agent-run

# Test with example
echo "analyze the DNA sequence ATCGATCGATCG" | pixi run sophisticated-agent
```

## 🧠 Enhanced Features (New!)

The **Sophisticated Agent** (`pixi run sophisticated-agent`) includes:

- **🧠 Initial Reasoning**: Deep task analysis using advanced models
- **📋 Execution Planning**: Step-by-step workflow planning
- **📊 Progress Tracking**: Real-time progress visualization
- **🔍 Self-Reflection**: Adaptive learning and plan optimization
- **🎯 Smart Tool Selection**: Intelligent tool discovery and usage

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

### Enhanced Natural Language Interface
```bash
pixi run sophisticated-agent
> analyze the DNA sequence ATCGATCGATCG and tell me about it

# Sophisticated Agent automatically:
# 🧠 Reasoning: Analyzes task complexity and requirements
# 📋 Planning: Creates step-by-step execution plan
# 🔧 Execution: Uses sequence_stats tool with proper parameters
# 📊 Progress: Shows real-time progress tracking
# 🔍 Reflection: Provides comprehensive biological analysis
```

### File-Based Sequence Analysis
```bash
> generate sequence stats for contigs100k.fna

# Sophisticated Agent automatically:
# 🧠 Reasoning: Understands file-based analysis requirements
# 📋 Planning: Plans analyze_fasta_file → write_json_report workflow
# 🔧 Execution: Executes tools with proper error handling
# 📊 Progress: Shows analysis progress with formatted statistics
# 📄 Output: Generates detailed report with sequence statistics
```

### Multi-Tool Scientific Workflows
```bash
> list all sequence files recursively and analyze the largest one

# Sophisticated Agent automatically:
# 🧠 Reasoning: Breaks down complex multi-step request
# 📋 Planning: Creates find_files → analyze_fasta_file → report workflow
# 🔧 Execution: Handles file discovery, size comparison, and analysis
# 📊 Progress: Tracks each step with clear status updates
# 🔍 Reflection: Provides scientific insights on the analysis
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
├── agents/
│   ├── sophisticated_agent/     # Enhanced Agent (RECOMMENDED)
│   │   ├── src/
│   │   │   ├── enhanced_agent.py    # Enhanced agent with reasoning/planning
│   │   │   ├── task_planner.py      # Execution planning system
│   │   │   ├── progress_tracker.py  # Real-time progress tracking
│   │   │   └── llm_interface.py     # Multi-model LLM integration
│   │   ├── config/
│   │   │   └── agent_config.yaml    # Model and feature configuration
│   │   ├── prompts/                 # Reasoning and planning prompts
│   │   └── mcp_config.json         # MCP server configuration
│   └── template/                # Basic Universal MCP Agent
│       ├── src/agent.py         # UniversalMCPAgent implementation
│       └── mcp_config.json      # Basic MCP configuration
├── mcps/                        # FastMCP Servers
│   ├── template/src/           # BioPython tools
│   │   ├── server_fastmcp.py   # FastMCP server implementation
│   │   ├── biotools.py         # Enhanced tool implementations
│   │   └── tool_schema.py      # API documentation
│   └── filesystem/src/         # File operations
│       ├── simple_server.py    # Enhanced filesystem MCP server
│       └── server.py           # Original filesystem server
├── docs/                       # Comprehensive documentation
├── pixi.toml                   # Environment and tasks
└── README.md                   # This file
```

## 🎓 Learning Path

**Beginner (30 minutes)**
1. Run the sophisticated agent: `pixi run sophisticated-agent`
2. Try sequence analysis: "generate sequence stats for contigs100k.fna"
3. Explore available tools with `tools` command
4. Watch the reasoning and planning phases in action

**Intermediate (2 hours)**  
1. Configure different models in `agents/sophisticated_agent/config/agent_config.yaml`
2. Customize reasoning and planning prompts
3. Test complex multi-tool workflows
4. Create a simple FastMCP server with 1-2 tools

**Advanced (Half day)**
1. Build domain-specific MCP server with async tools
2. Implement custom execution planning logic
3. Create sophisticated multi-step analysis pipelines
4. Customize the task complexity assessment system

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
pixi run sophisticated-agent  # Enhanced agent with reasoning & planning (RECOMMENDED)
pixi run agent-run           # Basic template agent
pixi run agent-test          # Quick functionality test

# Development
pixi run lint               # Check code quality
pixi run format             # Format code
pixi run test               # Run test suite

# MCP Server Testing
pixi run test-biopython     # Test BioPython MCP server
pixi run test-filesystem    # Test filesystem MCP server

# Remote MCP Server (Bioseq)
cd mcps/bioseq
pixi run websocket          # Start WebSocket bridge on port 8765
pixi run cf-run             # Start Cloudflare tunnel (production domain)
# Or for quick testing:
pixi run tunnel             # Start temporary tunnel (generates .trycloudflare.com URL)
```

### 🌐 Remote Bioseq MCP Server

The **bioseq** MCP server provides 32 specialized nucleic acid analysis tools and can be accessed remotely via WebSocket:

#### Starting the Bioseq MCP Server

1. **Navigate to bioseq directory:**
   ```bash
   cd mcps/bioseq
   ```

2. **Start the WebSocket bridge:**
   ```bash
   pixi run websocket
   # Starts on ws://localhost:8765
   ```

3. **For remote access, start Cloudflare tunnel:**
   
   **Option A - Production (with custom domain):**
   ```bash
   pixi run cf-run
   # Uses wss://mcp.newlineages.com (or your configured domain)
   ```
   
   **Option B - Testing (temporary URL):**
   ```bash
   pixi run tunnel
   # Generates temporary URL like wss://abc123.trycloudflare.com
   ```

4. **Configure agent to use remote server:**
   
   Edit `agents/sophisticated_agent/mcp_config.json`:
   ```json
   {
     "bioseq-remote": {
       "transport": "websocket",
       "uri": "wss://mcp.newlineages.com",  // or your tunnel URL
       "enabled": true
     }
   }
   ```

The bioseq server includes tools for:
- Assembly statistics and genome analysis
- Gene prediction and promoter detection
- GC skew and CpG island analysis
- Tandem repeat and motif detection
- K-mer analysis and much more

See [Remote MCP Architecture Guide](docs/REMOTE_MCP_ARCHITECTURE.md) and [MCP Remote Hosting Guide](docs/MCP_REMOTE_HOSTING_GUIDE.md) for detailed setup instructions.

## 🤖 Model Configuration

Configure the reasoning and execution models in `agents/sophisticated_agent/config/agent_config.yaml`:

```yaml
# Enhanced Features Configuration
enhanced_features:
  reasoning:
    enabled: true
    model: "google/gemini-pro"           # Advanced model for deep reasoning
    temperature: 0.3                     # Lower temperature for focused analysis
    max_tokens: 4000
    
  planning:
    enabled: true  
    model: "google/gemini-flash-lite"    # Efficient model for planning
    temperature: 0.2                     # Low temperature for structured planning
    max_tokens: 2000
    
  execution:
    model: "google/gemini-flash-lite"    # Fast model for tool coordination
    temperature: 0.1                     # Very low temperature for precise calls
    max_tokens: 1000
```

### Available Models (CBORG API)

**Reasoning Models (for complex analysis):**
- `google/gemini-pro` - Most capable Google model
- `anthropic/claude-opus` - Very capable Claude model  
- `openai/gpt-4o` - Capable OpenAI model

**Planning/Execution Models (for efficiency):**
- `google/gemini-flash-lite` - Fast and efficient Google model
- `anthropic/claude-sonnet` - Fast Claude model
- `openai/gpt-4o-mini` - Fast OpenAI model

## 🎯 Key Features

### Enhanced Agent Features (NEW!)
- **🧠 Multi-Model Reasoning**: Uses advanced models for deep task analysis
- **📋 Intelligent Planning**: Creates step-by-step execution workflows
- **📊 Real-Time Progress**: Visual progress tracking with colored output
- **🔍 Adaptive Learning**: Self-reflection and plan optimization
- **🚀 Smart Execution**: Enhanced error handling and parameter validation

### Core Universal Agent Features
- **Async-First**: Built on asyncio for efficient concurrent operations
- **Universal Tool Discovery**: Works with any MCP server without modification
- **Reflective Analysis**: Intelligent interpretation of tool results
- **Multi-Server Support**: Integrate multiple specialized MCP servers
- **External Prompt System**: Customize behavior without code changes
- **Scientific Focus**: Pre-configured with bioinformatics and data analysis tools

## 💡 Pro Tips

### For the Sophisticated Agent
1. **Use `pixi run sophisticated-agent`**: The enhanced agent provides much better results
2. **Watch the Reasoning**: The initial reasoning phase provides valuable insights
3. **Configure Models**: Use powerful models for reasoning, efficient ones for execution
4. **Complex Requests**: The agent excels at multi-step scientific workflows
5. **Progress Tracking**: Monitor real-time progress for long-running analyses

### General Tips
1. **Start Simple**: Begin with basic tool calls, then build complexity
2. **Use Reflection**: The agent's analysis of results is often more valuable than raw tool output
3. **Async Everything**: FastMCP's async nature enables powerful concurrent processing  
4. **External Prompts**: Customize behavior without code changes
5. **Tool Schemas**: Well-documented tools get better usage from the LLM

## 🚀 Ready to Build!

The NeLLi AI Scientist Agent Template provides everything you need to build sophisticated AI scientist agents:

### Enhanced Features (NEW!)
- ✅ **Sophisticated Agent** with reasoning, planning, and progress tracking
- ✅ **Multi-Model Support** for optimal performance at each stage
- ✅ **Real-Time Progress** visualization and adaptive learning
- ✅ **Enhanced Error Handling** for robust sequence analysis
- ✅ **Intelligent Tool Selection** and parameter validation

### Core Features
- ✅ Universal MCP Agent with 15 pre-built tools
- ✅ FastMCP integration for efficient async operations
- ✅ Intelligent reflection and analysis capabilities
- ✅ External prompt management for easy customization
- ✅ Comprehensive documentation and examples
- ✅ Scientific computing focus with bioinformatics tools

**Start with `pixi run sophisticated-agent` and build the future of AI-assisted scientific research!** 🧬🚀