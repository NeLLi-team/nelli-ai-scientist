#!/bin/bash
# Setup Cloudflare Tunnel for bioseq MCP server

# Configuration
TUNNEL_NAME="bioseq-mcp"
SUBDOMAIN="bioseq-mcp"  # You can change this
WEBSOCKET_PORT=8765

echo "=== Cloudflare Tunnel Setup for bioseq MCP ==="
echo ""

# Check if cloudflared is installed
if [ -f "./cloudflared" ]; then
    CLOUDFLARED="./cloudflared"
elif command -v cloudflared &> /dev/null; then
    CLOUDFLARED="cloudflared"
else
    echo "Error: cloudflared is not installed."
    echo "Install it from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation"
    exit 1
fi

# Step 1: Login to Cloudflare (if not already)
echo "Step 1: Checking Cloudflare authentication..."
if [ ! -f ~/.cloudflared/cert.pem ]; then
    echo "Please login to Cloudflare:"
    $CLOUDFLARED tunnel login
else
    echo "Already authenticated with Cloudflare."
fi

# Step 2: Create tunnel (if not exists)
echo ""
echo "Step 2: Creating tunnel..."
if $CLOUDFLARED tunnel list | grep -q "$TUNNEL_NAME"; then
    echo "Tunnel '$TUNNEL_NAME' already exists."
    TUNNEL_ID=$($CLOUDFLARED tunnel list --name "$TUNNEL_NAME" --output json | jq -r '.[0].id')
else
    $CLOUDFLARED tunnel create "$TUNNEL_NAME"
    TUNNEL_ID=$($CLOUDFLARED tunnel list --name "$TUNNEL_NAME" --output json | jq -r '.[0].id')
fi

echo "Tunnel ID: $TUNNEL_ID"

# Step 3: Create configuration file
echo ""
echo "Step 3: Creating tunnel configuration..."
cat > ~/.cloudflared/config.yml << EOF
tunnel: $TUNNEL_NAME
credentials-file: $HOME/.cloudflared/$TUNNEL_ID.json

ingress:
  - hostname: $SUBDOMAIN.\${DOMAIN}
    service: ws://localhost:$WEBSOCKET_PORT
  - service: http_status:404
EOF

echo "Configuration created at ~/.cloudflared/config.yml"

# Step 4: Instructions for DNS routing
echo ""
echo "Step 4: DNS Routing"
echo "================================"
echo "You need to run ONE of these commands (replace yourdomain.com with your domain):"
echo ""
echo "Option 1 - If you own a domain:"
echo "  $CLOUDFLARED tunnel route dns $TUNNEL_NAME $SUBDOMAIN.yourdomain.com"
echo ""
echo "Option 2 - For quick testing without a domain:"
echo "  $CLOUDFLARED tunnel --url ws://localhost:$WEBSOCKET_PORT"
echo "  (This gives you a temporary URL like https://random-name.trycloudflare.com)"
echo ""
echo "================================"

# Step 5: Create start script
cat > start_tunnel.sh << 'EOF'
#!/bin/bash
# Start the bioseq MCP server with WebSocket wrapper and Cloudflare Tunnel

echo "Starting bioseq MCP WebSocket wrapper..."
python websocket_wrapper.py &
WRAPPER_PID=$!

echo "Waiting for WebSocket server to start..."
sleep 3

echo "Starting Cloudflare Tunnel..."
$CLOUDFLARED tunnel run bioseq-mcp &
TUNNEL_PID=$!

echo "Services started:"
echo "  WebSocket wrapper PID: $WRAPPER_PID"
echo "  Cloudflare Tunnel PID: $TUNNEL_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo 'Stopping services...'; kill $WRAPPER_PID $TUNNEL_PID; exit" INT TERM
wait
EOF

chmod +x start_tunnel.sh

# Step 6: Create test script
cat > test_tunnel.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify the WebSocket connection through Cloudflare Tunnel."""

import asyncio
import json
import sys
import websockets

async def test_connection(uri):
    """Test the WebSocket MCP connection."""
    print(f"Testing connection to: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send initialize request (MCP protocol)
            initialize_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "0.1.0",
                    "capabilities": {}
                },
                "id": 1
            }
            
            print("Sending initialize request...")
            await websocket.send(json.dumps(initialize_request))
            
            # Wait for response
            response = await websocket.recv()
            response_data = json.loads(response)
            
            print("Response received:")
            print(json.dumps(response_data, indent=2))
            
            if "result" in response_data:
                print("\n✅ Connection successful! Server responded with capabilities.")
                return True
            else:
                print("\n❌ Unexpected response format")
                return False
                
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_tunnel.py wss://your-tunnel-url.com")
        sys.exit(1)
        
    uri = sys.argv[1]
    asyncio.run(test_connection(uri))
EOF

chmod +x test_tunnel.py

# Final instructions
echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Set up DNS routing (see Step 4 above)"
echo "2. Run './start_tunnel.sh' to start all services"
echo "3. Test with: python test_tunnel.py wss://your-tunnel-url.com"
echo ""
echo "For Claude Desktop configuration, add to config.json:"
echo '{'
echo '  "mcpServers": {'
echo '    "bioseq": {'
echo '      "uri": "wss://your-tunnel-url.com",'
echo '      "transport": "websocket"'
echo '    }'
echo '  }'
echo '}'