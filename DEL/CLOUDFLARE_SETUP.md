# Cloudflare Tunnel Setup for bioseq MCP Server

## Quick Start (Temporary URL)

The bioseq MCP server has been successfully configured to work with Cloudflare Tunnels. Here's how to use it:

### 1. Start the services:
```bash
# In terminal 1 - Start WebSocket wrapper
pixi run python websocket_wrapper.py

# In terminal 2 - Start Cloudflare tunnel
./cloudflared tunnel --url http://localhost:8765
```

### 2. Get your tunnel URL
When you run the cloudflared command, you'll see output like:
```
Your quick Tunnel has been created! Visit it at:
https://your-random-subdomain.trycloudflare.com
```

### 3. Configure Claude Desktop
Add this to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "bioseq-remote": {
      "uri": "wss://your-random-subdomain.trycloudflare.com",
      "transport": "websocket"
    }
  }
}
```

**Important**: Replace `your-random-subdomain` with the actual URL from step 2. Change `https://` to `wss://`.

## Using Your Cloudflare Token (Permanent Setup)

Since you have a Cloudflare token, you can create a permanent tunnel:

### 1. Set up authentication
```bash
# Export your token
export CLOUDFLARE_API_TOKEN="225a75a7-5ce1-4dd0-b5e4-8ee8cf03068d-0.0.2-integration-token"

# Or login via browser
./cloudflared tunnel login
```

### 2. Create a named tunnel
```bash
./cloudflared tunnel create bioseq-mcp
```

### 3. Route to your domain
```bash
# Replace 'yourdomain.com' with your actual domain
./cloudflared tunnel route dns bioseq-mcp bioseq.yourdomain.com
```

### 4. Create config file
Edit `~/.cloudflared/config.yml`:
```yaml
tunnel: bioseq-mcp
credentials-file: /home/yourusername/.cloudflared/[TUNNEL-ID].json

ingress:
  - hostname: bioseq.yourdomain.com
    service: http://localhost:8765
  - service: http_status:404
```

### 5. Run the tunnel
```bash
# Start WebSocket wrapper
pixi run python websocket_wrapper.py &

# Start named tunnel
./cloudflared tunnel run bioseq-mcp
```

## Architecture

The setup consists of:
1. **bioseq MCP server** - stdio-based MCP server with bioinformatics tools
2. **WebSocket wrapper** - Converts stdio to WebSocket protocol
3. **Cloudflare Tunnel** - Exposes the WebSocket server to the internet

```
[Claude Desktop] <--wss--> [Cloudflare] <--http--> [WebSocket Wrapper] <--stdio--> [bioseq MCP]
```

## Testing

Use the provided test script:
```bash
python test_tunnel.py wss://your-tunnel-url.trycloudflare.com
```

## Notes

- Quick tunnels (using `tunnel --url`) are temporary and for testing only
- For production use, create a named tunnel with your Cloudflare account
- The WebSocket wrapper handles the stdio-to-WebSocket conversion
- All bioseq MCP tools are available through the WebSocket connection