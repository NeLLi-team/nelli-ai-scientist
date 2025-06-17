#!/bin/bash
# Quick test tunnel without authentication

echo "Starting bioseq MCP with quick Cloudflare tunnel..."
echo ""

# Start WebSocket wrapper in background
echo "Starting WebSocket wrapper..."
python websocket_wrapper.py &
WRAPPER_PID=$!

# Wait for it to start
sleep 3

# Start quick tunnel
echo "Starting Cloudflare quick tunnel..."
echo "This will give you a temporary URL for testing..."
echo ""

if [ -f "./cloudflared" ]; then
    ./cloudflared tunnel --url http://localhost:8765
else
    cloudflared tunnel --url http://localhost:8765
fi