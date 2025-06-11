# AI Agent Template

This template provides a starting point for building AI agents that integrate with CBORG API (and legacy Claude/OpenAI APIs).

## Setup

1. Copy this template to your own directory:
   ```bash
   cp -r agents/template agents/<your-name>
   cd agents/<your-name>
   ```

2. Configure your API keys:
   ```bash
   cp config/secrets.yaml.example config/secrets.yaml
   # Edit config/secrets.yaml with your CBORG API key
   ```

3. Run tests:
   ```bash
   pixi run agent-test
   ```

## Structure

- `src/agent.py` - Main agent implementation
- `src/llm_interface.py` - LLM abstraction layer
- `src/tools.py` - Agent tools and capabilities
- `src/communication.py` - FIPA-ACL protocol implementation
- `config/` - Configuration files
- `tests/` - Unit and integration tests

## Usage

```python
from src.agent import BioinformaticsAgent, AgentConfig
from src.llm_interface import LLMProvider

# Create agent
config = AgentConfig(
    name="my-agent",
    capabilities=["sequence_analysis", "literature_search"],
    llm_provider=LLMProvider.CBORG  # Default, or use CLAUDE/OPENAI
)
agent = BioinformaticsAgent(config)

# Process message
response = await agent.process_message({
    "action": "analyze",
    "data": "ATCGATCG..."
})
```

## Running with Pixi

```bash
# Run the agent
pixi run agent-run

# Run tests
pixi run agent-test
```