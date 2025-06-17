# Connecting to Remote bioseq MCP Server

This guide explains how to connect the sophisticated agent to a remote bioseq MCP server hosted via Cloudflare Tunnel.

## Prerequisites

1. The remote bioseq MCP server must be running with:
   - WebSocket wrapper (`websocket_wrapper.py`)
   - Cloudflare Tunnel active
   
2. You need the tunnel URL (e.g., `https://your-subdomain.trycloudflare.com`)

## Configuration Steps

### 1. Update mcp_config.json

The configuration has already been added to `mcp_config.json`:

```json
"bioseq-remote": {
  "name": "Remote Nucleic Acid Analysis Tools",
  "description": "Remote bioseq MCP server accessed via Cloudflare Tunnel - Specialized DNA/RNA sequence analysis",
  "transport": "websocket",
  "uri": "wss://wear-budapest-valve-sublime.trycloudflare.com",
  "enabled": true,
  "use_cases": ["nucleic_acid_analysis", "assembly_statistics", "promoter_detection", "gc_skew_analysis", "cpg_islands", "giant_virus_motifs", "repeat_detection", "gene_prediction", "kmer_analysis"],
  "note": "Remote MCP server accessed via WebSocket through Cloudflare Tunnel. Replace URI with your actual tunnel URL."
}
```

**Important**: Replace the `uri` value with your actual tunnel URL, changing `https://` to `wss://`.

### 2. Install Dependencies

Make sure websockets is installed:

```bash
cd /path/to/nelli-ai-scientist
pixi install
```

### 3. Test the Connection

Run the test script to verify connectivity:

```bash
cd agents/sophisticated_agent
python test_remote_bioseq.py
```

You should see:
- List of available tools
- Successful test of sequence_stats tool

### 4. Use with Sophisticated Agent

The agent will automatically discover and use the remote bioseq tools:

```bash
# From the project root
pixi run sophisticated-agent-stdio
```

## How It Works

1. **Remote Server**: The bioseq MCP server runs on the remote machine with a WebSocket wrapper
2. **Cloudflare Tunnel**: Exposes the WebSocket server to the internet with a public URL
3. **WebSocket Client**: The sophisticated agent connects to the remote server via WebSocket
4. **MCP Protocol**: All tool calls are transmitted using the MCP protocol over WebSocket

## Troubleshooting

### Connection Failed
- Verify the tunnel URL is correct
- Ensure the remote server is running (both WebSocket wrapper and Cloudflare tunnel)
- Check if the URL uses `wss://` (not `https://`)

### Tools Not Discovered
- Check if `enabled: true` in the configuration
- Verify the WebSocket wrapper is running on port 8765
- Look at agent logs for connection errors

### Tool Calls Fail
- Ensure the remote bioseq server has all required dependencies
- Check if the data files are available on the remote server
- Verify the MCP protocol version compatibility

## Security Notes

- Quick tunnels (using `cloudflared tunnel --url`) are temporary and for testing
- For production, use authenticated Cloudflare tunnels with access controls
- Consider implementing API keys or authentication in the WebSocket wrapper
- Monitor tunnel access logs for suspicious activity

## Example Usage

Once configured, you can use remote bioseq tools as if they were local:

```python
# In your agent queries
"Analyze the GC content of this sequence: ATCGATCGATCG"
"Find promoters in the file /path/to/sequence.fasta"
"Calculate assembly statistics for the genome"
```

The agent will automatically route these requests to the remote bioseq server.