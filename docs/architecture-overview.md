# Architecture Overview: Universal MCP Agent System

## 🏗️ System Architecture

The NeLLi AI Scientist Agent Template is built on a modern, flexible architecture that combines Universal Agent capabilities with FastMCP protocol implementation.

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    NeLLi AI Scientist Agent                     │
├─────────────────────────────────────────────────────────────────┤
│  Universal MCP Agent (UniversalMCPAgent)                       │
│  ├── Dynamic Tool Discovery                                    │
│  ├── External Prompt Management (PromptManager)               │
│  ├── LLM Interface (CBORG/Claude/OpenAI)                      │
│  ├── FIPA-ACL Communication Protocol                          │
│  └── Reflection & Analysis Engine                             │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastMCP Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  FastMCP Client Library                                        │
│  ├── Async Connection Management                               │
│  ├── Tool Schema Discovery                                     │
│  ├── Parameter Validation                                      │
│  └── Error Handling                                            │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ BioPython    │ │ Filesystem   │ │   Custom     │
│ MCP Server   │ │ MCP Server   │ │ MCP Servers  │
│              │ │              │ │              │
│ • Sequence   │ │ • File Ops   │ │ • Your Tools │
│   Analysis   │ │ • Directory  │ │ • Domain     │
│ • FASTA      │ │   Listing    │ │   Specific   │
│   Processing │ │ • File I/O   │ │   Logic      │
│ • BLAST      │ │ • Tree       │ │              │
│ • Phylogeny  │ │   Explorer   │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
```

## 🔧 Universal Agent Design

### Key Principles

1. **Tool Agnostic**: The agent doesn't know about specific tools beforehand
2. **Dynamic Discovery**: Tools are discovered at runtime from MCP servers
3. **Schema-Driven**: Tool parameters and behavior defined by schemas
4. **Async-First**: Built on asyncio for efficient concurrent operations
5. **Reflective**: Analyzes and interprets tool results intelligently

### Agent Lifecycle

```python
async def agent_lifecycle():
    # 1. Initialization
    agent = UniversalMCPAgent(config)
    await agent.initialize()
    
    # 2. Discovery Phase
    mcp_servers = agent._load_mcp_config()
    tools = await agent._discover_all_tools()
    
    # 3. Chat Loop
    while True:
        user_input = get_user_input()
        
        # 4. Intent Analysis
        analysis = await agent.process_natural_language(user_input)
        
        if analysis.get("response_type") == "direct_answer":
            # 5a. Direct Response
            response = analysis.get("direct_answer")
            
        elif analysis.get("suggested_tools"):
            # 5b. Tool Execution
            results = []
            for suggestion in analysis["suggested_tools"]:
                result = await agent.execute_tool_suggestion(suggestion)
                results.append(result)
            
            # 6. Reflection
            reflection = await agent._reflect_on_tool_results(
                user_input, results
            )
```

## 🚀 FastMCP Integration

### Why FastMCP?

FastMCP provides significant advantages over the original MCP protocol:

- **Python-Native**: No JSON-RPC overhead
- **Async Support**: Built for asyncio from the ground up
- **Type Safety**: Pydantic-based parameter validation
- **Simpler Setup**: No complex transport configuration
- **Better Error Handling**: Rich exception information

### FastMCP Server Pattern

```python
from fastmcp import FastMCP

# Create server
mcp = FastMCP("My Scientific Tools 🧪")

@mcp.tool
async def analyze_data(data: str, method: str = "standard") -> dict:
    """Analyze scientific data using specified method
    
    Args:
        data: Input data to analyze
        method: Analysis method (standard, advanced, custom)
    """
    # Your analysis logic here
    return {"result": "analysis complete", "method": method}

if __name__ == "__main__":
    mcp.run()
```

## 🧠 Prompt Management System

### External Prompts

The agent uses external prompt files for easy customization:

```
agents/template/prompts/
├── tool_selection.txt       # How to choose tools
├── reflection.txt           # How to analyze results  
├── general_response.txt     # General conversations
└── error_handling.txt       # Error situations
```

### Prompt Manager

```python
class PromptManager:
    def load_prompt(self, prompt_name: str) -> str:
        """Load prompt from file with caching"""
        
    def format_prompt(self, prompt_name: str, **kwargs) -> str:
        """Format prompt with variables"""
        prompt = self.load_prompt(prompt_name)
        return prompt.format(**kwargs)

# Usage in agent
prompt = self.prompt_manager.format_prompt(
    "tool_selection",
    tools_context=tools_context,
    user_input=user_input
)
```

## 🔄 Reflection Engine

### Intelligent Analysis

The agent doesn't just execute tools - it reflects on results:

```python
async def _reflect_on_tool_results(
    self, 
    user_request: str, 
    tool_results: List[Dict[str, Any]]
) -> str:
    """Reflect on and interpret tool execution results"""
    
    # Build context about what tools were used
    results_context = []
    for result in tool_results:
        tool_name = result.get('tool', 'unknown')
        tool_result = result.get('result', {})
        results_context.append(f"Tool '{tool_name}' result: {tool_result}")
    
    # Use LLM to analyze and interpret
    prompt = self.prompt_manager.format_prompt(
        "reflection",
        user_request=user_request,
        results_summary="\\n".join(results_context)
    )
    
    return await self.llm.generate(prompt)
```

## 🔧 Tool Discovery Process

### Dynamic Discovery

```python
async def _discover_all_tools(self) -> Dict[str, Dict[str, Any]]:
    """Discover all tools from all configured MCP servers"""
    all_tools = {}
    
    for server_id, server_config in self.mcp_servers.items():
        try:
            from fastmcp import Client
            
            # Connect to MCP server
            script_path = self._get_fastmcp_script_path(server_id, server_config)
            async with Client(script_path) as client:
                
                # Discover tools
                tools = await client.list_tools()
                
                for tool in tools:
                    tool_name = tool.name
                    all_tools[tool_name] = {
                        "server_id": server_id,
                        "description": tool.description,
                        "schema": tool.inputSchema,
                        "server_config": server_config
                    }
                    
        except Exception as e:
            logger.error(f"Failed to discover tools from {server_id}: {e}")
    
    return all_tools
```

## 🎯 Extensibility Points

### Adding New MCP Servers

1. **Create FastMCP Server**:
   ```python
   # mcps/my_domain/src/server.py
   from fastmcp import FastMCP
   
   mcp = FastMCP("My Domain Tools")
   
   @mcp.tool
   async def my_tool(param: str) -> dict:
       return {"result": f"processed {param}"}
   ```

2. **Update Configuration**:
   ```json
   // agents/template/mcp_config.json
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

3. **Agent Automatically Discovers**: No code changes needed!

### Customizing Agent Behavior

- **Modify Prompts**: Edit files in `agents/template/prompts/`
- **Add LLM Providers**: Extend `LLMInterface` in `llm_interface.py`
- **Custom Analysis**: Override `_reflect_on_tool_results()` method
- **New Communication**: Extend FIPA-ACL handlers

## 🔒 Security & Best Practices

- **Async-Safe**: All operations use proper async patterns
- **Parameter Validation**: FastMCP handles type checking
- **Error Isolation**: Server failures don't crash the agent
- **Secure File Ops**: Filesystem tools include safety checks
- **LLM Safety**: Prompts designed to prevent injection attacks

This architecture provides a solid foundation for building sophisticated AI scientist agents while maintaining flexibility for customization and extension.