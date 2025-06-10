# Agent-to-MCP Server Integration Guide

This guide shows how to connect your AI agents directly to existing MCP servers (like Context7) without requiring Claude Desktop. This is useful for integrating any web-hosted or third-party MCP servers into your custom agents.

## Overview

Instead of using Claude Desktop as an intermediary, your agents can connect directly to MCP servers:

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Your Agent    │◄──►│   MCP Server     │◄──►│  External Service│
│   (Template)    │    │   (Context7)     │    │  (Vector DB)     │
└─────────────────┘    └──────────────────┘    └──────────────────┘
```

This allows agents to:
- **Use hosted MCP servers** (Context7, web search, databases)
- **Access external tools** without manual integration
- **Leverage existing MCP ecosystem** 
- **Scale with multiple MCP servers** per agent

## Setting Up Context7 Integration

### 1. Install Dependencies

Context7 is already available through our pixi environment:

```bash
# Ensure Node.js is available via pixi
pixi install

# Test Context7 server
pixi run context7-test
```

### 2. Update Agent to Connect to MCP Servers

Modify your agent to include MCP client capabilities:

```python
# agents/your-agent/src/mcp_client.py
import asyncio
import json
from typing import Dict, Any, List, Optional
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

class MCPClientManager:
    def __init__(self):
        self.servers: Dict[str, ClientSession] = {}
        self.available_tools: Dict[str, List] = {}
    
    async def connect_to_server(self, server_name: str, command: List[str]) -> bool:
        """Connect to an MCP server"""
        try:
            read, write = await stdio_client(command)
            session = ClientSession(read, write)
            await session.initialize()
            
            self.servers[server_name] = session
            
            # Get available tools
            tools = await session.list_tools()
            self.available_tools[server_name] = tools
            
            print(f"✅ Connected to {server_name}: {[t.name for t in tools]}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to {server_name}: {e}")
            return False
    
    async def call_tool(self, server_name: str, tool_name: str, **kwargs) -> Any:
        """Call a tool on a specific MCP server"""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not connected")
        
        session = self.servers[server_name]
        result = await session.call_tool(tool_name, kwargs)
        return result
    
    async def list_all_tools(self) -> Dict[str, List[str]]:
        """List all available tools across all connected servers"""
        all_tools = {}
        for server_name, tools in self.available_tools.items():
            all_tools[server_name] = [tool.name for tool in tools]
        return all_tools
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        for server_name, session in self.servers.items():
            try:
                # Close the session gracefully
                await session.close()
            except:
                pass
        self.servers.clear()
        self.available_tools.clear()
```

### 3. Integrate MCP Client into Your Agent

Update your agent to use MCP servers:

```python
# agents/your-agent/src/agent.py
from .mcp_client import MCPClientManager

class BioinformaticsAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMInterface(provider=config.llm_provider)
        self.tools = ToolRegistry()
        self.mcp_client = MCPClientManager()
        
        # Initialize MCP connections
        asyncio.create_task(self._initialize_mcp_servers())
        
        # Register default tools (existing code)
        self._register_default_tools()
    
    async def _initialize_mcp_servers(self):
        """Initialize connections to MCP servers"""
        mcp_servers = {
            "context7": ["pixi", "run", "npx", "-y", "@upstash/context7-mcp"],
            # Add more MCP servers here
        }
        
        for server_name, command in mcp_servers.items():
            await self.mcp_client.connect_to_server(server_name, command)
        
        # Register MCP tools as agent tools
        await self._register_mcp_tools()
    
    async def _register_mcp_tools(self):
        """Register MCP server tools as agent tools"""
        all_tools = await self.mcp_client.list_all_tools()
        
        for server_name, tool_names in all_tools.items():
            for tool_name in tool_names:
                # Create a unique tool name
                agent_tool_name = f"{server_name}_{tool_name}"
                
                # Create wrapper function
                async def mcp_tool_wrapper(**kwargs):
                    return await self.mcp_client.call_tool(
                        server_name, tool_name, **kwargs
                    )
                
                # Register with agent's tool registry
                self.tools.register(agent_tool_name)(mcp_tool_wrapper)
                print(f"📋 Registered MCP tool: {agent_tool_name}")
    
    async def shutdown(self):
        """Shutdown agent and disconnect from MCP servers"""
        await self.mcp_client.disconnect_all()
```

### 4. Test Agent-MCP Integration

Create a test script to verify the integration:

```python
# test_agent_mcp_integration.py
import asyncio
from agents.template.src.agent import BioinformaticsAgent, AgentConfig
from agents.template.src.llm_interface import LLMProvider

async def test_agent_mcp_integration():
    """Test agent's ability to use MCP servers"""
    
    # Create agent
    config = AgentConfig(
        name="mcp-test-agent",
        capabilities=["sequence_analysis", "vector_storage"],
        llm_provider=LLMProvider.CBORG,
    )
    
    agent = BioinformaticsAgent(config)
    
    # Wait for MCP connections to establish
    await asyncio.sleep(5)
    
    print("🔍 Testing MCP integration...")
    
    # Test 1: List available tools
    print("\n1. Available tools from MCP servers:")
    mcp_tools = await agent.mcp_client.list_all_tools()
    for server, tools in mcp_tools.items():
        print(f"   {server}: {tools}")
    
    # Test 2: Use Context7 to store information
    if "context7" in mcp_tools:
        print("\n2. Testing Context7 vector storage...")
        try:
            store_result = await agent.mcp_client.call_tool(
                "context7", 
                "store_vectors",
                id="bio_test_1",
                text="Bioinformatics combines computer science and biology",
                metadata={"category": "definition"}
            )
            print(f"   ✅ Stored: {store_result}")
        except Exception as e:
            print(f"   ❌ Store failed: {e}")
        
        # Test 3: Search Context7
        print("\n3. Testing Context7 semantic search...")
        try:
            search_result = await agent.mcp_client.call_tool(
                "context7",
                "search_vectors", 
                query="computational biology",
                limit=5
            )
            print(f"   ✅ Search results: {search_result}")
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    # Test 4: Use agent's registered MCP tools
    print("\n4. Testing agent's registered MCP tools:")
    agent_tools = agent.tools.list_tools()
    mcp_agent_tools = [name for name in agent_tools.keys() if "_" in name]
    print(f"   Registered MCP tools: {mcp_agent_tools}")
    
    # Test using a tool through agent interface
    if "context7_list_vectors" in agent_tools:
        try:
            list_result = await agent.tools.execute("context7_list_vectors")
            print(f"   ✅ List vectors: {list_result}")
        except Exception as e:
            print(f"   ❌ List failed: {e}")
    
    # Cleanup
    await agent.shutdown()
    print("\n✅ MCP integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_agent_mcp_integration())
```

## Adding Other MCP Servers

### Common MCP Servers for Agents

| Server | Command | Use Case |
|--------|---------|----------|
| **Context7** | `["pixi", "run", "npx", "-y", "@upstash/context7-mcp"]` | Vector storage, semantic search |
| **Filesystem** | `["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path"]` | File operations |
| **Web Search** | `["npx", "-y", "@modelcontextprotocol/server-brave-search"]` | Internet search |
| **SQLite** | `["npx", "-y", "@modelcontextprotocol/server-sqlite", "/path/to/db"]` | Database queries |
| **GitHub** | `["npx", "-y", "@modelcontextprotocol/server-github"]` | Repository access |

### Example: Adding Multiple MCP Servers

```python
async def _initialize_mcp_servers(self):
    """Initialize connections to multiple MCP servers"""
    mcp_servers = {
        "context7": ["pixi", "run", "npx", "-y", "@upstash/context7-mcp"],
        "filesystem": ["pixi", "run", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "web_search": ["pixi", "run", "npx", "-y", "@modelcontextprotocol/server-brave-search"],
    }
    
    for server_name, command in mcp_servers.items():
        success = await self.mcp_client.connect_to_server(server_name, command)
        if success:
            print(f"✅ {server_name} connected")
        else:
            print(f"⚠️ {server_name} failed to connect")
    
    # Register all MCP tools
    await self._register_mcp_tools()
```

### Environment Variables for MCP Servers

Some MCP servers need API keys or configuration:

```python
# In your agent initialization
import os

async def _initialize_mcp_servers(self):
    """Initialize MCP servers with environment variables"""
    
    # Context7 (no config needed for basic use)
    await self.mcp_client.connect_to_server(
        "context7", 
        ["pixi", "run", "npx", "-y", "@upstash/context7-mcp"]
    )
    
    # Web search (requires API key)
    if os.getenv("BRAVE_API_KEY"):
        await self.mcp_client.connect_to_server(
            "web_search",
            ["pixi", "run", "npx", "-y", "@modelcontextprotocol/server-brave-search"]
        )
    
    # Filesystem (restrict to safe directories)
    safe_dir = os.getenv("AGENT_WORKSPACE_DIR", "/tmp")
    await self.mcp_client.connect_to_server(
        "filesystem",
        ["pixi", "run", "npx", "-y", "@modelcontextprotocol/server-filesystem", safe_dir]
    )
```

## Advanced Usage Patterns

### 1. Conditional Tool Usage

```python
async def analyze_with_context(self, sequence: str) -> Dict[str, Any]:
    """Analyze sequence and store/retrieve context"""
    
    # First, do the analysis
    analysis = await self.tools.execute("sequence_stats", sequence=sequence)
    
    # Store in vector database if available
    if "context7" in self.mcp_client.servers:
        await self.mcp_client.call_tool(
            "context7", 
            "store_vectors",
            id=f"seq_analysis_{hash(sequence)}",
            text=f"Sequence analysis: {json.dumps(analysis)}",
            metadata={"type": "sequence_analysis", "length": len(sequence)}
        )
    
    # Search for related analyses
    if "context7" in self.mcp_client.servers:
        related = await self.mcp_client.call_tool(
            "context7",
            "search_vectors",
            query=f"sequence analysis length {len(sequence)}",
            limit=3
        )
        analysis["related_analyses"] = related
    
    return analysis
```

### 2. MCP Server Health Checks

```python
async def check_mcp_health(self) -> Dict[str, bool]:
    """Check health of connected MCP servers"""
    health = {}
    
    for server_name, session in self.mcp_client.servers.items():
        try:
            # Try a simple operation
            tools = await session.list_tools()
            health[server_name] = len(tools) > 0
        except:
            health[server_name] = False
    
    return health
```

### 3. Dynamic MCP Server Discovery

```python
async def discover_and_connect_mcp_servers(self):
    """Dynamically discover and connect to MCP servers"""
    
    # Define potential servers to try
    potential_servers = [
        ("context7", ["pixi", "run", "npx", "-y", "@upstash/context7-mcp"]),
        ("filesystem", ["pixi", "run", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
        ("sqlite", ["pixi", "run", "npx", "-y", "@modelcontextprotocol/server-sqlite", ":memory:"]),
    ]
    
    connected_servers = []
    
    for server_name, command in potential_servers:
        try:
            success = await self.mcp_client.connect_to_server(server_name, command)
            if success:
                connected_servers.append(server_name)
        except Exception as e:
            print(f"Could not connect to {server_name}: {e}")
    
    print(f"Connected to MCP servers: {connected_servers}")
    return connected_servers
```

## Troubleshooting

### Common Issues

#### 1. MCP Server Won't Start
```python
# Add timeout and better error handling
async def connect_to_server(self, server_name: str, command: List[str], timeout: int = 30) -> bool:
    try:
        read, write = await asyncio.wait_for(
            stdio_client(command), 
            timeout=timeout
        )
        # ... rest of connection logic
    except asyncio.TimeoutError:
        print(f"❌ {server_name} connection timed out")
        return False
```

#### 2. Tool Not Found
```python
# Verify tool exists before calling
async def safe_call_tool(self, server_name: str, tool_name: str, **kwargs):
    if server_name not in self.servers:
        raise ValueError(f"Server {server_name} not connected")
    
    available_tools = [t.name for t in self.available_tools[server_name]]
    if tool_name not in available_tools:
        raise ValueError(f"Tool {tool_name} not available. Available: {available_tools}")
    
    return await self.call_tool(server_name, tool_name, **kwargs)
```

#### 3. Connection Drops
```python
# Add reconnection logic
async def ensure_connection(self, server_name: str) -> bool:
    if server_name not in self.servers:
        return False
    
    try:
        # Test connection with a simple operation
        await self.servers[server_name].list_tools()
        return True
    except:
        # Reconnect
        print(f"🔄 Reconnecting to {server_name}")
        return await self.reconnect_server(server_name)
```

## Benefits for Hackathon Participants

### 1. **Rapid Integration**
- No need to implement external APIs manually
- Leverage existing MCP ecosystem
- Focus on agent logic, not integration

### 2. **Scalable Architecture**
- Add new capabilities by connecting to MCP servers
- Mix and match different services
- Easy to swap or upgrade components

### 3. **Real-world Services**
- Vector databases (Context7)
- Web search (Brave)
- File systems
- Databases (SQLite, PostgreSQL)
- Version control (GitHub)

### 4. **Development Efficiency**
- Test integrations independently
- Mock services for development
- Easy deployment across environments

This pattern allows participants to build sophisticated agents that can leverage the growing ecosystem of MCP servers without vendor lock-in or complex integration work!