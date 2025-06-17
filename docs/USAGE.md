# nelli-ai-scientist Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pixi install
```

### 2. Run the Sophisticated Agent (Main Interface)
```bash
pixi run sophisticated-agent
```
This runs the stdio-based agent that can connect to local and remote MCP servers.

## MCP Servers

### Start Individual MCP Servers

#### bioseq MCP (DNA/RNA Analysis)
```bash
# Local server
pixi run bioseq-mcp

# For remote access via WebSocket
pixi run bioseq-websocket     # Terminal 1
pixi run bioseq-tunnel        # Terminal 2 - provides public URL
```

#### Other MCP Servers
```bash
pixi run biocoding-mcp       # Code generation and execution
pixi run filesystem-mcp      # File operations
```

## Development Tasks

### Code Quality
```bash
pixi run format     # Format code with black and fix with ruff
pixi run lint       # Check code style
pixi run clean      # Remove Python cache files
```

### Testing
```bash
pixi run test-agent     # Test agent imports
pixi run test-mcp       # Test MCP server imports
```

## Remote Access Setup

1. Start the WebSocket wrapper:
   ```bash
   pixi run bioseq-websocket
   ```

2. Start Cloudflare tunnel (in another terminal):
   ```bash
   pixi run bioseq-tunnel
   ```

3. Note the tunnel URL from the output (e.g., `https://random-name.trycloudflare.com`)

4. Update `agents/sophisticated_agent/mcp_config.json` with your tunnel URL:
   ```json
   "bioseq-remote": {
     "uri": "wss://your-tunnel-url.trycloudflare.com"
   }
   ```

5. From another machine, run the agent and it will connect to your remote MCP server

## File Structure

```
nelli-ai-scientist/
├── agents/sophisticated_agent/    # Main agent implementation
│   ├── run_stdio_agent.py        # Entry point
│   ├── mcp_config.json           # MCP server configuration
│   ├── config/                   # Agent configuration
│   ├── prompts/                  # System prompts
│   └── src/                      # Agent source code
├── mcps/                         # MCP servers
│   ├── bioseq/                   # DNA/RNA analysis tools
│   ├── biocoding/                # Code generation
│   └── filesystem/               # File operations
└── pixi.toml                     # Dependencies and tasks
```

## Configuration

- Agent config: `agents/sophisticated_agent/config/agent_config.yaml`
- MCP servers: `agents/sophisticated_agent/mcp_config.json`
- Environment: `.env` file in project root