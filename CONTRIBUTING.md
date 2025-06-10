# Contributing to NeLLi AI Scientist

## Hackathon Workflow

1. Create your branch following the naming convention:
   - Agents: `agent/<your-name>-<purpose>`
   - MCPs: `mcp/<tool-name>`

2. Work in your own directory:
   - Agents: `agents/<your-name>/`
   - MCPs: `mcps/<tool-name>/`

3. Follow the standards in `docs/standards.md`

4. Submit PR when ready for integration

## Development Setup

1. **Install pixi:**
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. **Install dependencies:**
   ```bash
   pixi install
   ```

3. **Run tests:**
   ```bash
   pixi run test
   ```

## Code Standards

- Python 3.9+
- Black for formatting
- Type hints required
- 80% test coverage minimum
- Documentation for all public functions

## Commit Messages

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Test additions
- `refactor:` Code restructuring

## Testing

All code must include tests:
```bash
# Run all tests
pixi run test

# Run specific component tests
pixi run agent-test
pixi run mcp-test

# Check code style
pixi run lint
```

## Pixi Commands

We use pixi for dependency management. Key commands:
- `pixi install` - Install all dependencies
- `pixi run <task>` - Run a task defined in pixi.toml
- `pixi add <package>` - Add a new dependency
- `pixi shell` - Activate the environment