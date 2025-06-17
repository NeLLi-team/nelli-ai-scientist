#!/usr/bin/env python3
"""
MCP WebSocket Bridge - Properly bridges stdio MCP server to WebSocket
"""

import asyncio
import json
import logging
import subprocess
import sys
import os
from pathlib import Path

import websockets

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
            # Start the MCP server in the bioseq directory with its pixi environment
            bioseq_dir = Path(__file__).parent
            env = os.environ.copy()
            # Force unbuffered output for Python
            env['PYTHONUNBUFFERED'] = '1'
            
            process = await asyncio.create_subprocess_exec(
                "pixi", "run", "python", "-u", "-m", "src.server",
                cwd=bioseq_dir,
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
            
            # Create task to monitor stderr
            stderr_task = asyncio.create_task(self.monitor_stderr(connection_id))
            
            # Create tasks for bidirectional communication
            tasks = [
                asyncio.create_task(self.websocket_to_process(connection_id)),
                asyncio.create_task(self.process_to_websocket(connection_id)),
                stderr_task
            ]
            
            # Wait for either task to complete
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
            # Clean up
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
                    process.stdin.write(message.encode() + b'\n')
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
    
    # Create server with the bridge's handle_connection method
    async with websockets.serve(bridge.handle_connection, host, port):
        logger.info("MCP WebSocket Bridge is running...")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down MCP WebSocket Bridge")