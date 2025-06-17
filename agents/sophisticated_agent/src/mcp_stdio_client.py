"""
Fixed MCP Stdio Client - Uses proper connection-per-operation pattern
Now with WebSocket support for remote MCP servers
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession
import mcp.types as types

try:
    from .mcp_websocket_client import MCPWebSocketClient
except ImportError:
    MCPWebSocketClient = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # Only show errors

# Suppress MCP and FastMCP logging during tool discovery
logging.getLogger('mcp').setLevel(logging.ERROR)
logging.getLogger('FastMCP').setLevel(logging.ERROR)
logging.getLogger('fastmcp').setLevel(logging.ERROR)


class MCPStdioClient:
    """Client for connecting to MCP servers via stdio subprocess"""
    
    def __init__(self, server_id: str, server_config: Dict[str, Any]):
        self.server_id = server_id
        self.server_config = server_config
        
    def _build_server_params(self) -> StdioServerParameters:
        """Build server parameters from config"""
        command = self.server_config.get("command", "python")
        args = self.server_config.get("args", [])
        cwd = self.server_config.get("cwd")
        env = self.server_config.get("env", {})
        
        # Handle special pixi case
        if command == "pixi" and "--manifest-path" in args:
            # For pixi, we need to ensure proper path resolution
            manifest_idx = args.index("--manifest-path")
            if manifest_idx + 1 < len(args):
                manifest_path = args[manifest_idx + 1]
                # Make path absolute if relative
                if not Path(manifest_path).is_absolute():
                    manifest_path = str(Path.cwd() / manifest_path)
                    args[manifest_idx + 1] = manifest_path
        
        # Suppress pixi output by redirecting to null
        import os
        if env is None:
            env = {}
        
        # Add environment variables to suppress pixi and MCP server output
        env.update({
            "PIXI_QUIET": "true",
            "PIXI_NO_PROGRESS": "true", 
            "PIXI_LOG_LEVEL": "error",
            # Suppress FastMCP and Python logging
            "PYTHONUNBUFFERED": "0",
            "FASTMCP_LOG_LEVEL": "ERROR",
            "MCP_LOG_LEVEL": "ERROR",
            "LOGGING_LEVEL": "ERROR",
            # Suppress general Python warnings and info
            "PYTHONWARNINGS": "ignore",
            # Try to suppress Rich console output from FastMCP
            "TERM": "dumb",
            "NO_COLOR": "1",
            # Suppress pixi task output
            "PIXI_SILENT": "true"
        })
        
        return StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=cwd
        )
        
    async def discover_tools(self) -> Dict[str, Dict[str, Any]]:
        """Discover available tools from the server"""
        try:
            server_params = self._build_server_params()
            # Removed verbose logging during tool discovery
            
            # Temporarily redirect stderr to suppress subprocess output
            import sys
            import os
            from contextlib import redirect_stderr
            
            # Use devnull to suppress stderr output from subprocess
            with open(os.devnull, 'w') as devnull:
                # The stdio_client will still work but subprocess messages are suppressed
                async with stdio_client(server_params) as (read_stream, write_stream):
                    async with ClientSession(
                        read_stream,
                        write_stream,
                        client_info=types.Implementation(
                            name="nelli-agent",
                            version="1.0.0"
                        )
                    ) as session:
                        await session.initialize()
                        
                        # List tools
                        tools_result = await session.list_tools()
                        discovered_tools = {}
                        
                        # Handle the tools response properly
                        if hasattr(tools_result, 'tools'):
                            tools = tools_result.tools
                        else:
                            tools = tools_result if isinstance(tools_result, list) else []
                        
                        for tool in tools:
                            try:
                                tool_info = {
                                    "server_id": self.server_id,
                                    "server_name": self.server_config.get("name", self.server_id),
                                    "description": getattr(tool, 'description', ""),
                                    "schema": getattr(tool, 'inputSchema', {}),
                                    "server_config": self.server_config
                                }
                                tool_name = getattr(tool, 'name', str(tool))
                                discovered_tools[tool_name] = tool_info
                            except Exception as tool_error:
                                logger.warning(f"Skipping tool due to error: {tool_error}")
                                continue
                        
                        # Reduced verbosity - only log if there are issues
                        if len(discovered_tools) == 0:
                            logger.warning(f"No tools discovered from {self.server_id}")
                        return discovered_tools
                    
        except Exception as e:
            import traceback
            logger.error(f"Failed to discover tools from {self.server_id}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {}
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the server"""
        try:
            server_params = self._build_server_params()
            # Reduced verbosity during tool calls
            
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    client_info=types.Implementation(
                        name="nelli-agent",
                        version="1.0.0"
                    )
                ) as session:
                    await session.initialize()
                    
                    # Call the tool
                    result = await session.call_tool(tool_name, arguments)
                    return result
                    
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on {self.server_id}: {e}")
            raise


class MCPConnectionManager:
    """Manages multiple MCP stdio connections"""
    
    def __init__(self):
        self.server_configs: Dict[str, Dict[str, Any]] = {}
    
    async def connect_server(self, server_id: str, server_config: Dict[str, Any]) -> bool:
        """Connect to a server (really just store config for later use)"""
        self.server_configs[server_id] = server_config
        return True  # Always succeeds since we connect per-operation
    
    def add_server(self, server_id: str, server_config: Dict[str, Any]):
        """Add a server configuration"""
        self.server_configs[server_id] = server_config
    
    async def discover_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Discover tools from all configured servers"""
        all_tools = {}
        
        for server_id, server_config in self.server_configs.items():
            if server_config.get("enabled", True):
                # Check if this is a WebSocket server
                if server_config.get("transport") == "websocket":
                    if MCPWebSocketClient:
                        try:
                            client = MCPWebSocketClient(server_id, server_config)
                            tools_list = await client.list_tools()
                            # Convert to expected format
                            for tool in tools_list:
                                tool_name = tool["name"]
                                all_tools[tool_name] = {
                                    "name": tool_name,
                                    "description": tool.get("description", ""),
                                    "schema": tool.get("input_schema", {}),  # Fix: use "schema" key to match stdio format
                                    "server_id": server_id,
                                    "server_name": server_config.get("name", server_id)
                                }
                        except Exception as e:
                            logger.error(f"Failed to discover tools from WebSocket server {server_id}: {e}")
                else:
                    # Regular stdio server
                    client = MCPStdioClient(server_id, server_config)
                    tools = await client.discover_tools()
                    all_tools.update(tools)
        
        return all_tools
    
    async def call_tool(self, tool_name: str, tool_info: Dict[str, Any], arguments: Dict[str, Any]) -> Any:
        """Call a tool on the appropriate server"""
        server_id = tool_info.get("server_id")
        if server_id not in self.server_configs:
            raise RuntimeError(f"Server {server_id} not configured")
        
        server_config = self.server_configs[server_id]
        
        # Check if this is a WebSocket server
        if server_config.get("transport") == "websocket":
            if not MCPWebSocketClient:
                raise RuntimeError("WebSocket support not available")
            client = MCPWebSocketClient(server_id, server_config)
            return await client.call_tool(tool_name, arguments)
        else:
            # Regular stdio server
            client = MCPStdioClient(server_id, server_config)
            return await client.call_tool(tool_name, arguments)
    
    async def disconnect_all(self):
        """Disconnect all servers (no-op since we use connection-per-operation)"""
        pass