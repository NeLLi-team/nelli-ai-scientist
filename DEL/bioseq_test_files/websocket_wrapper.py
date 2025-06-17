#!/usr/bin/env python3
"""
WebSocket wrapper for stdio-based MCP servers.
Allows exposing stdio MCP servers over WebSocket for remote access.
"""

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import websockets
import websockets.server

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StdioMCPWrapper:
    """Wraps a stdio-based MCP server to expose it over WebSocket."""
    
    def __init__(self, command: list[str], cwd: Optional[Path] = None):
        self.command = command
        self.cwd = cwd or Path.cwd()
        self.process: Optional[subprocess.Popen] = None
        self.websocket: Optional[websockets.server.WebSocketServerProtocol] = None
        
    async def start_process(self):
        """Start the stdio MCP server process."""
        logger.info(f"Starting MCP server: {' '.join(self.command)}")
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd
        )
        
    async def stop_process(self):
        """Stop the MCP server process."""
        if self.process:
            logger.info("Stopping MCP server process")
            self.process.terminate()
            await self.process.wait()
            self.process = None
            
    async def read_from_process(self):
        """Read messages from the MCP server stdout and send to WebSocket."""
        while self.process and self.websocket:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break
                    
                # MCP messages are JSON-RPC over stdio
                message = line.decode('utf-8').strip()
                if message:
                    logger.debug(f"MCP -> WS: {message}")
                    await self.websocket.send(message)
                    
            except Exception as e:
                logger.error(f"Error reading from process: {e}")
                break
                
    async def write_to_process(self, message: str):
        """Write message to the MCP server stdin."""
        if self.process and self.process.stdin:
            try:
                logger.debug(f"WS -> MCP: {message}")
                self.process.stdin.write((message + '\n').encode('utf-8'))
                await self.process.stdin.drain()
            except Exception as e:
                logger.error(f"Error writing to process: {e}")
                
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connection."""
        logger.info(f"New WebSocket connection from {websocket.remote_address}")
        self.websocket = websocket
        
        try:
            # Start the MCP server process
            await self.start_process()
            
            # Start reading from process in background
            read_task = asyncio.create_task(self.read_from_process())
            
            # Handle incoming WebSocket messages
            async for message in websocket:
                logger.debug(f"Received message: {message}")
                await self.write_to_process(message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")
        finally:
            # Clean up
            self.websocket = None
            read_task.cancel()
            await self.stop_process()
            logger.info("Connection handler finished")

async def websocket_handler(websocket, path):
    """WebSocket handler function."""
    # Path to the bioseq MCP server
    bioseq_path = Path(__file__).parent
    
    # Command to run the MCP server
    command = [sys.executable, "-m", "src.server"]
    
    # Create wrapper for this connection
    wrapper = StdioMCPWrapper(command, cwd=bioseq_path)
    
    # Handle the connection
    await wrapper.handle_websocket(websocket, path)

async def main():
    """Main entry point."""
    # Configuration
    host = "0.0.0.0"
    port = 8765
    
    # Start WebSocket server
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    async with websockets.serve(websocket_handler, host, port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())