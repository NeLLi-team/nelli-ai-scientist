#!/usr/bin/env python3
"""Test WebSocket connection to bioseq MCP server"""

import asyncio
import json
import websockets

async def test_connection():
    uri = "ws://localhost:8765"
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        # Send initialize request
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            },
            "id": 1
        }
        
        print("Sending initialize request...")
        await websocket.send(json.dumps(request))
        
        print("Waiting for response...")
        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        print(f"Response: {response}")
        
        # Parse and display
        data = json.loads(response)
        print(f"Parsed response: {json.dumps(data, indent=2)}")

if __name__ == "__main__":
    try:
        asyncio.run(test_connection())
    except Exception as e:
        print(f"Error: {e}")