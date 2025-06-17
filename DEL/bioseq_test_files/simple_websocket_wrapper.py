#!/usr/bin/env python3
"""
Simple WebSocket wrapper for MCP servers
"""

import asyncio
import json
import logging
import subprocess
import sys
import os
from pathlib import Path

import websockets

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SimpleMCPWrapper:
    def __init__(self):
        self.process = None
        
    async def start_mcp_server(self):
        """Start the MCP server process."""
        bioseq_path = Path(__file__).parent
        
        # Set up environment to use the bioseq pixi environment
        env = os.environ.copy()
        
        # Start the server process in the bioseq directory
        logger.info("Starting MCP server process...")
        self.process = await asyncio.create_subprocess_exec(
            "pixi", "run", "run",
            cwd=bioseq_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Give it a moment to start
        await asyncio.sleep(1)
        
        if self.process.returncode is not None:
            stderr = await self.process.stderr.read()
            raise Exception(f"MCP server failed to start: {stderr.decode()}")
            
        logger.info("MCP server started successfully")
        
    async def stop_mcp_server(self):
        """Stop the MCP server process."""
        if self.process:
            logger.info("Stopping MCP server...")
            self.process.terminate()
            await self.process.wait()
            self.process = None
            
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connection."""
        logger.info(f"New WebSocket connection from {websocket.remote_address}")
        
        try:
            # Start MCP server for this connection
            await self.start_mcp_server()
            
            # Create tasks for bidirectional communication
            ws_to_mcp_task = asyncio.create_task(self.ws_to_mcp(websocket))
            mcp_to_ws_task = asyncio.create_task(self.mcp_to_ws(websocket))
            
            # Wait for either task to complete (which means connection closed)
            done, pending = await asyncio.wait(
                [ws_to_mcp_task, mcp_to_ws_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
        finally:
            await self.stop_mcp_server()
            logger.info("WebSocket connection handler finished")
            
    async def ws_to_mcp(self, websocket):
        """Forward messages from WebSocket to MCP server."""
        try:
            async for message in websocket:
                logger.debug(f"WS -> MCP: {message}")
                if self.process and self.process.stdin:
                    self.process.stdin.write((message + '\n').encode())
                    await self.process.stdin.drain()
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed (ws_to_mcp)")
        except Exception as e:
            logger.error(f"Error in ws_to_mcp: {e}")
            
    async def mcp_to_ws(self, websocket):
        """Forward messages from MCP server to WebSocket."""
        try:
            while self.process:
                # Check if websocket is still open
                if websocket.closed:
                    break
                    
                line = await self.process.stdout.readline()
                if not line:
                    break
                    
                message = line.decode().strip()
                if message:
                    logger.debug(f"MCP -> WS: {message}")
                    await websocket.send(message)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed (mcp_to_ws)")
        except Exception as e:
            logger.error(f"Error in mcp_to_ws: {e}")

async def websocket_handler(websocket):
    """Handle incoming WebSocket connections."""
    wrapper = SimpleMCPWrapper()
    await wrapper.handle_websocket(websocket, "/")

async def main():
    """Main entry point."""
    host = "0.0.0.0"
    port = 8765
    
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    
    async with websockets.serve(websocket_handler, host, port):
        logger.info("WebSocket server is running...")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())