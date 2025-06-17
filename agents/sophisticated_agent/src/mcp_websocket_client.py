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
                            "name": "nelli-agent",
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
                
                # Send initialized notification to complete the handshake
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