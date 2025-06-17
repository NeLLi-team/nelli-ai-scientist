# Complete Guide: Hosting MCP Servers Remotely via Cloudflare Tunnel

## Overview

This guide provides comprehensive instructions for making stdio-based MCP servers accessible remotely through Cloudflare tunnel, enabling AI agents to access specialized tools from anywhere on the internet.

## Architecture

```
[AI Agent] <--WebSocket--> [Cloudflare Tunnel] <--WebSocket--> [WebSocket Bridge] <--stdio--> [MCP Server]
```

### Components:
1. **MCP Server**: Original stdio-based FastMCP server
2. **WebSocket Bridge**: Protocol converter (stdio ↔ WebSocket)
3. **Cloudflare Tunnel**: Secure internet exposure
4. **WebSocket Client**: Agent-side connection handler

## Step-by-Step Implementation

### 1. Prepare Your MCP Server

#### Create MCP Server Structure
```python
# Example: src/server.py
from fastmcp import FastMCP
import asyncio

mcp = FastMCP("Your Tool Server")

@mcp.tool
async def your_tool(param: str) -> dict:
    """Your tool implementation"""
    return {"result": f"Processed: {param}"}

if __name__ == "__main__":
    mcp.run(stdio=True)
```

#### Set Up Pixi Configuration
```toml
# File: pixi.toml
[project]
name = "your-mcp-server"
version = "0.1.0"
description = "Your MCP Server"
channels = ["conda-forge", "bioconda"]
platforms = ["linux-64", "osx-64", "osx-arm64", "win-64"]

[dependencies]
python = ">=3.9,<3.12"
pip = "*"

[pypi-dependencies]
mcp = ">=0.1.0"
fastmcp = ">=1.0"
websockets = ">=12.0"

[tasks]
# Server tasks
run = "python -m src.server"

# WebSocket and tunnel tasks  
websocket = "python mcp_websocket_bridge.py"
tunnel = "./cloudflared tunnel --url http://localhost:8765"

# Production tunnel (requires setup)
tunnel-prod = "./cloudflared tunnel --config cloudflared-config.yml run my-mcp-server"
```

### 2. Create WebSocket Bridge

**File: `mcp_websocket_bridge.py`**

```python
#!/usr/bin/env python3
"""
MCP WebSocket Bridge - Converts stdio MCP server to WebSocket protocol
"""

import asyncio
import json
import logging
import subprocess
import os
from pathlib import Path
import websockets

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MCPWebSocketBridge:
    def __init__(self):
        self.connections = {}
        
    async def handle_connection(self, websocket):
        """Handle a WebSocket connection."""
        connection_id = id(websocket)
        logger.info(f"New connection {connection_id} from {websocket.remote_address}")
        
        # Start MCP server process
        process = None
        try:
            # Start the MCP server in your project directory
            project_dir = Path(__file__).parent
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
            
            # Adjust this command for your specific setup
            process = await asyncio.create_subprocess_exec(
                "python", "-u", "-m", "src.server",  # or your server startup command
                cwd=project_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            logger.info(f"Started MCP server process for connection {connection_id}")
            
            # Store connection info
            self.connections[connection_id] = {
                'websocket': websocket,
                'process': process
            }
            
            # Create tasks for bidirectional communication
            tasks = [
                asyncio.create_task(self.websocket_to_process(connection_id)),
                asyncio.create_task(self.process_to_websocket(connection_id)),
                asyncio.create_task(self.monitor_stderr(connection_id))
            ]
            
            # Wait for any task to complete
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in connection {connection_id}: {e}", exc_info=True)
        finally:
            # Clean up process
            if process:
                logger.info(f"Terminating MCP server for connection {connection_id}")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Process didn't terminate, killing it")
                    process.kill()
                    await process.wait()
                    
            if connection_id in self.connections:
                del self.connections[connection_id]
                
            logger.info(f"Connection {connection_id} closed")
            
    async def websocket_to_process(self, connection_id):
        """Forward messages from WebSocket to MCP process."""
        conn = self.connections.get(connection_id)
        if not conn:
            return
            
        websocket = conn['websocket']
        process = conn['process']
        
        try:
            async for message in websocket:
                if process.stdin:
                    logger.debug(f"WS->MCP [{connection_id}]: {message}")
                    process.stdin.write(message.encode() + b'\\n')
                    await process.stdin.drain()
                else:
                    logger.warning(f"Process stdin closed for {connection_id}")
                    break
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket closed for {connection_id}")
        except Exception as e:
            logger.error(f"Error forwarding to process {connection_id}: {e}")
            
    async def process_to_websocket(self, connection_id):
        """Forward messages from MCP process to WebSocket."""
        conn = self.connections.get(connection_id)
        if not conn:
            return
            
        websocket = conn['websocket']
        process = conn['process']
        
        try:
            while True:
                if not process.stdout:
                    logger.warning(f"Process stdout closed for {connection_id}")
                    break
                    
                line = await process.stdout.readline()
                if not line:
                    logger.info(f"Process ended for {connection_id}")
                    break
                    
                message = line.decode().strip()
                if message:
                    logger.debug(f"MCP->WS [{connection_id}]: {message}")
                    await websocket.send(message)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket closed for {connection_id}")
        except Exception as e:
            logger.error(f"Error forwarding to websocket {connection_id}: {e}")
            
    async def monitor_stderr(self, connection_id):
        """Monitor stderr output from the process."""
        conn = self.connections.get(connection_id)
        if not conn:
            return
            
        process = conn['process']
        
        try:
            while True:
                if not process.stderr:
                    break
                    
                line = await process.stderr.readline()
                if not line:
                    break
                    
                error_msg = line.decode().strip()
                if error_msg:
                    logger.warning(f"MCP stderr [{connection_id}]: {error_msg}")
        except Exception as e:
            logger.error(f"Error monitoring stderr {connection_id}: {e}")

async def main():
    """Main entry point."""
    host = "0.0.0.0"
    port = 8765
    
    bridge = MCPWebSocketBridge()
    
    logger.info(f"Starting MCP WebSocket Bridge on ws://{host}:{port}")
    
    # Create WebSocket server
    async with websockets.serve(bridge.handle_connection, host, port):
        logger.info("MCP WebSocket Bridge is running...")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down MCP WebSocket Bridge")
```

### 3. Set Up Cloudflare Tunnel

#### Install Cloudflared (Local Binary)
```bash
# Download cloudflared binary directly to your project
curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared

# Verify installation
./cloudflared version
```

#### Quick Setup (Testing) - Using Pixi Tasks
```bash
# Option 1: Use pixi tasks (recommended)
pixi run websocket &    # Start WebSocket bridge
pixi run tunnel        # Create tunnel (generates temporary URL)

# Option 2: Manual commands
python mcp_websocket_bridge.py &
./cloudflared tunnel --url http://localhost:8765
```

#### Production Setup (Dedicated Domain)
```bash
# 1. Authenticate with Cloudflare (using local binary)
./cloudflared tunnel login

# 2. Create named tunnel
./cloudflared tunnel create my-mcp-server

# 3. Create configuration file (in your project directory)
# File: cloudflared-config.yml
tunnel: my-mcp-server
credentials-file: ./tunnel-credentials.json  # Local to project

ingress:
  - hostname: mcp.yourdomain.com
    service: ws://localhost:8765
  - service: http_status:404

# 4. Add DNS record
./cloudflared tunnel route dns my-mcp-server mcp.yourdomain.com

# 5. Run tunnel with local config
./cloudflared tunnel --config cloudflared-config.yml run my-mcp-server
```

### 4. Create WebSocket MCP Client

**File: `mcp_websocket_client.py`**

```python
"""
WebSocket MCP Client - Connects to MCP servers over WebSocket
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import websockets
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class MCPWebSocketClient:
    """Client for connecting to MCP servers via WebSocket"""
    
    def __init__(self, server_id: str, server_config: Dict[str, Any]):
        self.server_id = server_id
        self.server_config = server_config
        self.uri = server_config.get("uri")
        if not self.uri:
            raise ValueError(f"WebSocket MCP server {server_id} requires 'uri' in config")
        
        self._request_id = 0
        
    def _next_request_id(self) -> int:
        """Generate next request ID"""
        self._request_id += 1
        return self._request_id
        
    @asynccontextmanager
    async def connect(self):
        """Connect to WebSocket MCP server"""
        try:
            async with websockets.connect(self.uri) as websocket:
                # Initialize the connection
                init_request = {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "0.1.0",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "your-agent",
                            "version": "1.0.0"
                        }
                    },
                    "id": self._next_request_id()
                }
                
                await websocket.send(json.dumps(init_request))
                response = await websocket.recv()
                init_response = json.loads(response)
                
                if "error" in init_response:
                    raise Exception(f"Failed to initialize: {init_response['error']}")
                
                # Send initialized notification to complete handshake
                initialized_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }
                await websocket.send(json.dumps(initialized_notification))
                    
                yield MCPWebSocketSession(websocket, self)
                
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket MCP server {self.server_id}: {e}")
            raise
            
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server"""
        async with self.connect() as session:
            return await session.list_tools()
            
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the server"""
        async with self.connect() as session:
            return await session.call_tool(tool_name, arguments)

class MCPWebSocketSession:
    """Active WebSocket session with an MCP server"""
    
    def __init__(self, websocket, client: MCPWebSocketClient):
        self.websocket = websocket
        self.client = client
        
    async def _send_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Send request and wait for response"""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.client._next_request_id()
        }
        
        await self.websocket.send(json.dumps(request))
        
        # Wait for response with matching ID
        while True:
            response_text = await self.websocket.recv()
            response = json.loads(response_text)
            
            if response.get("id") == request["id"]:
                if "error" in response:
                    raise Exception(f"MCP error: {response['error']}")
                return response.get("result", {})
                
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        result = await self._send_request("tools/list", {})
        tools = result.get("tools", [])
        
        # Convert to expected format
        formatted_tools = []
        for tool in tools:
            formatted_tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {})
            })
            
        return formatted_tools
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool"""
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        
        # Extract content from response
        content = result.get("content", [])
        if content and isinstance(content, list):
            # Return first text content
            for item in content:
                if item.get("type") == "text":
                    return item.get("text", "")
                    
        return result
```

### 5. Integration in AI Agents

#### Configuration
```json
{
  "mcp_servers": {
    "your-remote-server": {
      "name": "Your Remote Tools",
      "description": "Description of your tools",
      "transport": "websocket",
      "uri": "wss://your-tunnel-url.trycloudflare.com",
      "enabled": true,
      "use_cases": ["your", "use", "cases"]
    }
  }
}
```

#### Multi-Transport Support
```python
class MCPConnectionManager:
    async def call_tool(self, tool_name: str, tool_info: Dict[str, Any], arguments: Dict[str, Any]) -> Any:
        """Call a tool on the appropriate server"""
        server_id = tool_info.get("server_id")
        server_config = self.server_configs[server_id]
        
        # Check transport type
        if server_config.get("transport") == "websocket":
            client = MCPWebSocketClient(server_id, server_config)
            return await client.call_tool(tool_name, arguments)
        else:
            # Regular stdio server
            client = MCPStdioClient(server_id, server_config)
            return await client.call_tool(tool_name, arguments)
```

### 6. Testing Your Setup

#### Test WebSocket Bridge
```python
# test_websocket.py
import asyncio
import json
import websockets

async def test_connection():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Test initialization
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            },
            "id": 1
        }
        
        await websocket.send(json.dumps(init_request))
        response = await websocket.recv()
        print(f"Response: {response}")

asyncio.run(test_connection())
```

#### Test Remote Connection
```python
# test_remote.py
import asyncio
from mcp_websocket_client import MCPWebSocketClient

async def test_remote():
    config = {"uri": "wss://your-tunnel-url.trycloudflare.com"}
    client = MCPWebSocketClient("test", config)
    
    tools = await client.list_tools()
    print(f"Found {len(tools)} tools")
    
    if tools:
        result = await client.call_tool(tools[0]["name"], {})
        print(f"Result: {result}")

asyncio.run(test_remote())
```

## Common Issues and Solutions

### 1. WebSocket Connection Issues
- **Check port availability**: Ensure port 8765 is not in use
- **Firewall settings**: Local firewall might block connections
- **SSL/TLS**: Use `wss://` for Cloudflare tunnel URLs

### 2. Process Management
- **Orphaned processes**: Implement proper cleanup in bridge
- **Resource leaks**: Monitor memory usage with multiple connections
- **Startup delays**: Add startup verification before forwarding

### 3. Protocol Issues
- **Handshake failures**: Ensure proper MCP initialization sequence
- **Message framing**: Use line-based JSON messages for stdio
- **Error propagation**: Handle MCP errors properly

### 4. Performance Considerations
- **Connection pooling**: Reuse connections when possible
- **Process lifecycle**: Balance resource usage vs connection latency
- **Logging overhead**: Adjust log levels for production

## Custom Domain Setup - Production Ready

### Why You Need a Custom Domain
Cloudflare's temporary URLs (`.trycloudflare.com`):
- **Change on restart**: New URL generated each time
- **Not persistent**: Cannot rely on URL for production use
- **Limited branding**: Generic Cloudflare subdomain

### Complete Custom Domain Implementation

#### 1. Prerequisites
- **Owned domain**: Domain registered with any provider
- **Cloudflare account**: Free tier sufficient
- **Domain on Cloudflare**: Add domain to Cloudflare and update nameservers

#### 2. Step-by-Step Custom Domain Setup

```bash
# 1. Login to Cloudflare (one-time setup)
./cloudflared tunnel login

# 2. Create named tunnel (use your project name)
./cloudflared tunnel create bioseq-mcp
# This creates tunnel ID and credentials file

# 3. Create tunnel configuration file
# File: cloudflared-config.yml (in your project directory)
```

**cloudflared-config.yml**:
```yaml
tunnel: ea5eba81-a8cd-4d55-8b10-1b14fd3ae646  # Your tunnel ID
credentials-file: /home/user/.cloudflared/YOUR_TUNNEL_ID.json

ingress:
  # Your custom domain for MCP WebSocket server
  - hostname: mcp.newlineages.com
    service: ws://localhost:8765
    originRequest:
      connectTimeout: 30s
      tlsTimeout: 30s
      keepAliveTimeout: 90s
      # Important for WebSocket support
      noTLSVerify: false
      
  # Catch-all (returns 404 for all other requests)
  - service: http_status:404
```

```bash
# 4. Add DNS record (creates subdomain pointing to tunnel)
./cloudflared tunnel route dns bioseq-mcp mcp.newlineages.com

# 5. Add pixi task for production tunnel
# Add to pixi.toml [tasks] section:
cf-run = "./cloudflared tunnel --config cloudflared-config.yml run bioseq-mcp"

# 6. Start production tunnel
pixi run cf-run
```

#### 3. Update Agent Configuration

**mcp_config.json**:
```json
{
  "mcp_servers": {
    "bioseq-remote": {
      "name": "Remote Nucleic Acid Analysis Tools",
      "description": "Remote bioseq MCP server accessed via Cloudflare Tunnel",
      "transport": "websocket",
      "uri": "wss://mcp.newlineages.com",
      "enabled": true,
      "use_cases": ["nucleic_acid_analysis", "assembly_statistics", "gene_prediction"],
      "note": "Remote MCP server accessed via custom domain through Cloudflare Tunnel."
    }
  }
}
```

#### 4. Complete Startup Process

```bash
# Terminal 1: Start WebSocket bridge
pixi run websocket

# Terminal 2: Start production tunnel
pixi run cf-run

# Test connection
# Your agent can now connect to wss://mcp.newlineages.com
```

#### 5. Benefits of Custom Domain
- **Persistent URL**: `wss://mcp.newlineages.com` never changes
- **Professional branding**: Your domain, your control
- **SSL included**: Cloudflare provides automatic SSL/TLS
- **Custom configuration**: Advanced routing and security options
- **Multiple subdomains**: Can host multiple MCP servers

#### 6. Adding Additional MCP Servers to Same Domain

### Subdomain Naming Pattern

Use clear, functional subdomain names for different MCP server types:

| **MCP Server Type** | **Subdomain** | **Port** | **Purpose** |
|---|---|---|---|
| Bioseq/DNA Analysis | `mcp.newlineages.com` | 8765 | Nucleic acid analysis (primary) |
| ML/Code Generation | `ml.newlineages.com` | 8766 | Code generation & ML pipelines |
| Filesystem Ops | `fs.newlineages.com` | 8767 | File operations & data management |
| Memory/Knowledge | `memory.newlineages.com` | 8768 | Context & knowledge management |
| Database/SQL | `db.newlineages.com` | 8769 | Database operations |
| Web/Scraping | `web.newlineages.com` | 8770 | Web scraping & automation |

### Step-by-Step Multi-Server Setup

**1. Add DNS Records for New Subdomains:**
```bash
# Add DNS routing for each new subdomain
./cloudflared tunnel route dns bioseq-mcp ml.newlineages.com
./cloudflared tunnel route dns bioseq-mcp fs.newlineages.com
./cloudflared tunnel route dns bioseq-mcp memory.newlineages.com
./cloudflared tunnel route dns bioseq-mcp db.newlineages.com
```

**2. Update cloudflared-config.yml:**
```yaml
tunnel: ea5eba81-a8cd-4d55-8b10-1b14fd3ae646
credentials-file: /home/user/.cloudflared/ea5eba81-a8cd-4d55-8b10-1b14fd3ae646.json

ingress:
  # Bioseq MCP server (nucleic acid analysis)
  - hostname: mcp.newlineages.com
    service: ws://localhost:8765
    originRequest:
      connectTimeout: 30s
      tlsTimeout: 30s
      keepAliveTimeout: 90s
      noTLSVerify: false
      
  # ML/BioCoding MCP server (code generation & execution)
  - hostname: ml.newlineages.com
    service: ws://localhost:8766
    originRequest:
      connectTimeout: 30s
      tlsTimeout: 30s
      keepAliveTimeout: 90s
      noTLSVerify: false
      
  # Filesystem MCP server (file operations)
  - hostname: fs.newlineages.com
    service: ws://localhost:8767
    originRequest:
      connectTimeout: 30s
      tlsTimeout: 30s
      keepAliveTimeout: 90s
      noTLSVerify: false
      
  # Memory/Context MCP server (knowledge management)
  - hostname: memory.newlineages.com
    service: ws://localhost:8768
    originRequest:
      connectTimeout: 30s
      tlsTimeout: 30s
      keepAliveTimeout: 90s
      noTLSVerify: false
      
  # Catch-all (returns 404 for all other requests)
  - service: http_status:404
```

**3. Update Agent Configuration (mcp_config.json):**
```json
{
  "mcp_servers": {
    "bioseq-remote": {
      "name": "Remote Nucleic Acid Analysis Tools",
      "description": "Remote bioseq MCP server - DNA/RNA sequence analysis",
      "transport": "websocket",
      "uri": "wss://mcp.newlineages.com",
      "enabled": true,
      "use_cases": ["nucleic_acid_analysis", "assembly_statistics", "gene_prediction"]
    },
    "ml-remote": {
      "name": "Remote ML/BioCoding Tools",
      "description": "Remote ML and code generation MCP server",
      "transport": "websocket", 
      "uri": "wss://ml.newlineages.com",
      "enabled": false,
      "use_cases": ["code_generation", "code_execution", "ml_pipelines"]
    },
    "filesystem-remote": {
      "name": "Remote Filesystem Operations",
      "description": "Remote filesystem MCP server for file operations",
      "transport": "websocket",
      "uri": "wss://fs.newlineages.com", 
      "enabled": false,
      "use_cases": ["file_operations", "data_management"]
    },
    "memory-remote": {
      "name": "Remote Memory/Context Management", 
      "description": "Remote memory and context management MCP server",
      "transport": "websocket",
      "uri": "wss://memory.newlineages.com",
      "enabled": false,
      "use_cases": ["memory_storage", "semantic_search", "knowledge_retrieval"]
    }
  }
}
```

**4. Create WebSocket Bridges for Each MCP Server:**

For each new MCP server, copy the WebSocket bridge pattern and modify the port:

```python
# ml_websocket_bridge.py (port 8766)
# fs_websocket_bridge.py (port 8767) 
# memory_websocket_bridge.py (port 8768)

# In each bridge, change:
port = 8766  # or 8767, 8768, etc.

# And update the MCP server startup command:
process = await asyncio.create_subprocess_exec(
    "pixi", "run", "python", "-u", "-m", "src.server",  # Your specific server
    cwd=your_mcp_project_dir,
    # ... rest of config
)
```

**5. Add Pixi Tasks for Each Server:**

Add to each MCP server's `pixi.toml`:
```toml
[tasks]
# WebSocket bridge for this specific server
websocket = "python websocket_bridge.py"

# Use same tunnel config but different startup
tunnel-run = "../bioseq/cloudflared tunnel --config ../bioseq/cloudflared-config.yml run bioseq-mcp"
```

### Quick Setup for New MCP Server

To add a new MCP server type:

1. **Create the MCP server** in its own directory with `pixi.toml`
2. **Copy WebSocket bridge** and change port number
3. **Add DNS record**: `./cloudflared tunnel route dns bioseq-mcp newtype.newlineages.com`
4. **Add hostname to cloudflared-config.yml** with new port
5. **Add to agent mcp_config.json** with new URI
6. **Restart tunnel**: `pixi run cf-run`
7. **Start new WebSocket bridge**: `pixi run websocket` (in new server dir)

This approach allows unlimited MCP servers under one domain with clear, functional naming.

#### 4. Production Configuration
```yaml
# ~/.cloudflared/config.yml
tunnel: production-mcp
credentials-file: /home/user/.cloudflared/tunnel-credentials.json

ingress:
  # Main MCP server
  - hostname: mcp.yourdomain.com
    service: ws://localhost:8765
    originRequest:
      connectTimeout: 30s
      keepAliveTimeout: 90s
      
  # Additional MCP servers
  - hostname: bioseq.yourdomain.com
    service: ws://localhost:8766
    
  - hostname: ml.yourdomain.com
    service: ws://localhost:8767
    
  # Catch-all
  - service: http_status:404

warp-routing:
  enabled: true

metrics: localhost:9090
```

#### 5. DNS Configuration
```bash
# Add multiple subdomains
cloudflared tunnel route dns production-mcp mcp.yourdomain.com
cloudflared tunnel route dns production-mcp bioseq.yourdomain.com
cloudflared tunnel route dns production-mcp ml.yourdomain.com
```

### Production Deployment Considerations

#### 1. Service Management
```bash
# Install as system service
sudo cloudflared service install

# Configure service to start on boot
sudo systemctl enable cloudflared
sudo systemctl start cloudflared

# Monitor service
sudo systemctl status cloudflared
sudo journalctl -u cloudflared -f
```

#### 2. Security Enhancements
- **Access control**: Cloudflare Access for authentication
- **Rate limiting**: Cloudflare rules for DDoS protection
- **IP filtering**: Restrict access to specific IP ranges
- **Authentication tokens**: Custom authentication in MCP bridge

#### 3. Monitoring and Alerting
- **Health checks**: Monitor tunnel and MCP server health
- **Metrics collection**: Cloudflare analytics and custom metrics
- **Alerting**: Setup alerts for service failures
- **Logging**: Centralized logging for debugging

## Summary

This guide provides a complete implementation for hosting MCP servers remotely via Cloudflare tunnel. The key components are:

1. **WebSocket Bridge**: Converts stdio MCP to WebSocket protocol
2. **Cloudflare Tunnel**: Secure internet exposure without infrastructure changes
3. **WebSocket Client**: Agent-side connection handling
4. **Multi-transport Support**: Seamless integration with existing stdio servers

The main limitation is the lack of a dedicated domain for persistent URLs, which can be resolved by setting up a custom domain with Cloudflare DNS management.

This approach enables AI agents to access specialized MCP tools from anywhere on the internet while maintaining security and reliability.