#!/usr/bin/env python3
"""Test MCP server directly with stdio"""

import asyncio
import json
import subprocess
import sys

async def test_mcp_server():
    # Start the MCP server
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-m", "src.server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, 'PYTHONUNBUFFERED': '1'}
    )
    
    # Send initialize request
    request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "0.1.0",
            "capabilities": {}
        },
        "id": 1
    }
    
    print("Sending initialize request...")
    request_str = json.dumps(request) + '\n'
    process.stdin.write(request_str.encode())
    await process.stdin.drain()
    
    print("Waiting for response...")
    
    # Read response with timeout
    try:
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
        if response_line:
            response = response_line.decode().strip()
            print(f"Response: {response}")
            data = json.loads(response)
            print(f"Parsed: {json.dumps(data, indent=2)}")
        else:
            print("No response received")
            
        # Check stderr
        stderr_data = await process.stderr.read()
        if stderr_data:
            print(f"Stderr: {stderr_data.decode()}")
            
    except asyncio.TimeoutError:
        print("Timeout waiting for response")
        
        # Check what's in stderr
        try:
            stderr_data = await asyncio.wait_for(process.stderr.read(), timeout=1.0)
            if stderr_data:
                print(f"Stderr: {stderr_data.decode()}")
        except:
            pass
    
    # Clean up
    process.terminate()
    await process.wait()

import os
if __name__ == "__main__":
    asyncio.run(test_mcp_server())