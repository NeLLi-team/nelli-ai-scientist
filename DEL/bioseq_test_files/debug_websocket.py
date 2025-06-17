#!/usr/bin/env python3
"""Debug WebSocket argument passing"""

import asyncio
import json
import websockets

async def test_tool_call():
    uri = "ws://localhost:8766"
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {},
                "clientInfo": {
                    "name": "debug-client",
                    "version": "1.0.0"
                }
            },
            "id": 1
        }
        
        await websocket.send(json.dumps(init_request))
        response = await websocket.recv()
        print(f"Init response: {response}")
        
        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        await websocket.send(json.dumps(initialized_notification))
        
        # List tools
        tools_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        
        await websocket.send(json.dumps(tools_request))
        response = await websocket.recv()
        print(f"Tools response: {response}")
        
        # Test tool call with arguments
        tool_call_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "sequence_stats",
                "arguments": {
                    "sequence": "ATCGATCG"
                }
            },
            "id": 3
        }
        
        print(f"Sending tool call: {json.dumps(tool_call_request, indent=2)}")
        await websocket.send(json.dumps(tool_call_request))
        response = await websocket.recv()
        print(f"Tool call response: {response}")

if __name__ == "__main__":
    try:
        asyncio.run(test_tool_call())
    except Exception as e:
        print(f"Error: {e}")