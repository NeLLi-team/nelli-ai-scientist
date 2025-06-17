# Integration Layer

This directory contains the orchestration and integration components for connecting all agents and MCP servers.

## Components

- `orchestrator.py` - Main orchestration engine
- `registry.yaml` - Agent and MCP registry
- `docker-compose.yml` - Container orchestration
- `tests/` - Integration tests

## Usage

1. Register your agent/MCP in `registry.yaml`
2. Run integration tests: `pixi run integration-test`
3. Launch full system: `docker-compose up`

## Registry Format

### Adding an Agent
```yaml
agents:
  - name: your-agent-name
    path: agents/your-name
    branch: agent/your-name-purpose
    capabilities:
      - capability1
      - capability2
    mcp_dependencies:
      - mcp-server-name
    status: development  # or ready
```

### Adding an MCP Server
```yaml
mcps:
  - name: your-mcp-name
    path: mcps/your-tool
    port: 8002
    tools:
      - tool1
      - tool2
    status: ready
```