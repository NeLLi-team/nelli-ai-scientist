# NeLLi AI Scientist Agent - Hackathon Quick Start

Welcome to the NeLLi AI Scientist Agent Template! This guide gets you building sophisticated AI agents with FastMCP integration in minutes.

## 🚀 Quick Setup

```bash
# 1. Clone and setup environment
git clone <repository>
cd nelli-ai-scientist
pixi install

# 2. Run the agent
pixi run agent-run

# 3. Test with example
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

## 📁 Key Files & Directories

### Core Agent Files
```
agents/template/
├── mcp_config.json         # MCP server configuration
├── src/
│   ├── agent.py            # UniversalMCPAgent implementation  
│   ├── llm_interface.py    # LLM integration (CBORG/Claude/OpenAI)
│   └── prompt_manager.py   # External prompt management
└── prompts/
    ├── tool_selection.txt  # How agent chooses tools
    ├── reflection.txt      # How agent analyzes results
    └── general_response.txt # General conversation handling
```

### MCP Servers
```
mcps/
├── template/src/           # BioPython tools (sequence analysis)
│   ├── server_fastmcp.py   # FastMCP server implementation
│   ├── biotools.py         # Tool implementations  
│   └── tool_schema.py      # API documentation
└── filesystem/src/         # File operations
    └── server.py           # FastMCP filesystem server
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

## 🎯 Agent Interaction Examples

### Natural Language Processing
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

## 🔧 Building Your Own MCP Server

### 1. Create FastMCP Server

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

@mcp.tool  
async def process_batch(files: list, parallel: bool = True) -> dict:
    """Process multiple files in batch
    
    Args:
        files: List of file paths to process
        parallel: Whether to process in parallel
    """
    results = []
    
    if parallel:
        # Async batch processing
        import asyncio
        tasks = [process_single_file(f) for f in files]
        results = await asyncio.gather(*tasks)
    else:
        # Sequential processing
        for file_path in files:
            result = await process_single_file(file_path)
            results.append(result)
    
    return {
        "processed_files": len(files),
        "results": results,
        "processing_mode": "parallel" if parallel else "sequential"
    }

if __name__ == "__main__":
    mcp.run()
```

### 2. Add Tool Schemas (Optional but Recommended)

```python
# mcps/my_domain/src/tool_schema.py
def get_tool_schemas():
    return {
        "my_analysis_tool": {
            "description": "Advanced data analysis with custom algorithms",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Input data"},
                    "method": {"type": "string", "enum": ["standard", "advanced", "custom"]}
                },
                "required": ["data"]
            }
        }
    }
```

### 3. Register with Agent

```json
// agents/template/mcp_config.json
{
  "mcp_servers": {
    "my_domain": {
      "name": "My Domain Tools",
      "description": "Custom analysis tools for my domain",
      "fastmcp_script": "../../mcps/my_domain/src/server.py",
      "enabled": true,
      "use_cases": ["custom_analysis", "batch_processing"]
    }
  }
}
```

### 4. Test Your Server

```bash
# Test server directly
cd mcps/my_domain/src && python server.py

# Test with agent
pixi run agent-run
> use my analysis tool to process this data: "sample data here"
```

## 🧠 Customizing Agent Behavior

### Modify Prompts (No Code Changes!)

**Tool Selection** (`agents/template/prompts/tool_selection.txt`):
```
You are a universal AI agent with access to various tools...

KEY PRINCIPLES:
- For biological sequences, prefer BioPython tools
- For file operations, use filesystem tools  
- For custom domain tasks, use domain-specific tools

RESPONSE FORMAT:
{
  "response_type": "direct_answer" OR "use_tools",
  "direct_answer": "your answer (if response_type is direct_answer)",
  "intent": "what the user wants to accomplish",
  "suggested_tools": [
    {
      "tool_name": "exact_tool_name",
      "parameters": {"param1": "value1"},
      "reason": "why this tool helps achieve the intent"
    }
  ]
}
```

**Reflection** (`agents/template/prompts/reflection.txt`):
```
Analyze the results from tool execution and provide insights...

Consider:
- Scientific significance of the results
- Potential next steps or follow-up analyses
- Limitations or caveats
- Connections to broader scientific context
```

### Advanced Agent Patterns

**Iterative Analysis Loop**:
```python
# The agent naturally supports iterative analysis
# 1. Initial tool execution
# 2. Reflection on results  
# 3. Follow-up questions or deeper analysis
# 4. Additional tool execution based on insights
```

**Multi-Modal Processing**:
```python
# Agent can combine different tool types
# 1. File operations to gather data
# 2. Domain-specific analysis tools
# 3. Reporting and visualization tools
# 4. External integrations
```

## 🚀 Advanced Features

### Async Tool Execution

The agent uses asyncio for efficient concurrent operations:

```python
# Multiple tools can run concurrently
async def execute_parallel_analysis():
    tasks = [
        agent.execute_tool("sequence_stats", sequence=seq1),
        agent.execute_tool("sequence_stats", sequence=seq2),
        agent.execute_tool("blast_local", sequence=seq3)
    ]
    results = await asyncio.gather(*tasks)
```

### Intelligent Reflection

The agent doesn't just execute tools - it thinks about results:

```python
# After tool execution, agent reflects:
# - What do these results mean scientifically?
# - What patterns or insights emerge?
# - What follow-up analysis might be valuable?
# - How do results connect to user's broader goals?
```

### External Prompt Management

Easily customize agent behavior without touching code:

```python
# Prompts are loaded from external files
self.prompt_manager.format_prompt(
    "tool_selection",
    tools_context=available_tools,
    user_input=user_request
)
```

## 🎓 Learning Path

### Beginner (30 minutes)
1. Run the agent and try example queries
2. Explore available tools with `tools` command
3. Modify prompts to change agent personality

### Intermediate (2 hours)  
1. Create a simple FastMCP server with 1-2 tools
2. Integrate with agent configuration
3. Test multi-tool workflows

### Advanced (Half day)
1. Build domain-specific MCP server with async tools
2. Implement custom reflection logic
3. Create sophisticated analysis pipelines

## 🐛 Troubleshooting

**Agent not finding tools?**
```bash
# Check MCP configuration
cat agents/template/mcp_config.json

# Verify server paths exist
ls -la mcps/*/src/server*.py

# Test server directly
cd mcps/template/src && python server_fastmcp.py
```

**Import errors?**
```bash
# Ensure pixi environment is activated
pixi shell

# Check Python path
pixi run python -c "import sys; print('\\n'.join(sys.path))"
```

**Tool execution fails?**
```bash
# Check agent logs (they include detailed error info)
pixi run agent-run --help  # See logging options
```

## 📚 Next Steps

- **[Architecture Overview](architecture-overview.md)** - Understand the system design
- **[FastMCP Server Development](fastmcp-server-development.md)** - Build sophisticated MCP servers  
- **[Advanced Agent Concepts](advanced-agent-concepts.md)** - Self-evolving AI patterns
- **[Pixi Environment Setup](pixi-setup.md)** - Master the development environment

## 💡 Pro Tips

1. **Start Simple**: Begin with basic tool calls, then build complexity
2. **Use Reflection**: The agent's analysis of results is often more valuable than raw tool output
3. **Async Everything**: FastMCP's async nature enables powerful concurrent processing  
4. **External Prompts**: Customize behavior without code changes
5. **Tool Schemas**: Well-documented tools get better usage from the LLM

Ready to build the future of AI-assisted scientific research! 🧬🚀