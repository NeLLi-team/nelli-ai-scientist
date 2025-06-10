# Pixi Environment Setup Guide

## 🐾 What is Pixi?

Pixi is a modern package manager for Python that provides:
- **Fast dependency resolution** using conda-forge packages
- **Isolated environments** per project
- **Cross-platform consistency** (Linux, macOS, Windows)
- **Lock files** for reproducible builds
- **Task automation** with built-in scripts

## 📦 Understanding pixi.toml

The `pixi.toml` file defines your project's environment and automation:

```toml
[project]
name = "nelli-ai-scientist"
version = "0.1.0"
description = "NeLLi AI Scientist Agent Template"
channels = ["conda-forge", "anaconda"]
platforms = ["linux-64", "osx-64", "osx-arm64", "win-64"]

[dependencies]
# Core Python environment
python = ">=3.11"
pip = "*"

# Scientific computing
numpy = "*"
pandas = "*"
matplotlib = "*"
seaborn = "*"

# Bioinformatics (for BioPython MCP server)
biopython = "*"
blast = "*"  # Local BLAST for sequence analysis

# AI/ML
openai = "*"           # OpenAI API support
anthropic = "*"        # Claude API support
python-dotenv = "*"    # Environment variable management

# FastMCP and async
fastmcp = "*"          # Modern MCP implementation
asyncio = "*"          # Async programming
aiofiles = "*"         # Async file operations

# Development tools
pytest = "*"           # Testing framework
black = "*"            # Code formatting
ruff = "*"             # Fast linting
mypy = "*"             # Type checking

[tasks]
# Primary agent execution
agent-run = "cd agents/template && python -m src.agent"
agent-test = "cd agents/template && python -m src.agent --name test-agent"

# Development tasks
lint = "ruff check ."
format = "black ."
typecheck = "mypy agents/ mcps/"
test = "pytest -v"

# MCP server testing
test-biopython = "cd mcps/template/src && python server_fastmcp.py"
test-filesystem = "cd mcps/filesystem/src && python server.py"

# Clean tasks
clean = "find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true"
clean-pyc = "find . -name '*.pyc' -delete"
```

## 🚀 Quick Setup

### 1. Install Pixi

```bash
# Using curl (recommended)
curl -fsSL https://pixi.sh/install.sh | bash

# Using conda
conda install -c conda-forge pixi

# Using pip
pip install pixi
```

### 2. Initialize Project

```bash
# Clone the repository
git clone <your-nelli-repo>
cd nelli-ai-scientist

# Install dependencies (this reads pixi.toml)
pixi install

# Verify installation
pixi info
```

### 3. Activate Environment

```bash
# Activate the pixi environment
pixi shell

# Or run commands directly
pixi run agent-run
```

## 🎯 Key Pixi Commands

### Environment Management

```bash
# Install all dependencies
pixi install

# Add a new dependency
pixi add requests numpy
pixi add --dev pytest black  # Development dependencies

# Remove a dependency
pixi remove requests

# Update dependencies
pixi update

# Show environment info
pixi info

# List installed packages
pixi list
```

### Task Execution

```bash
# Run the NeLLi agent
pixi run agent-run

# Run with custom settings
pixi run python agents/template/src/agent.py --name my-agent

# Development tasks
pixi run lint        # Check code quality
pixi run format      # Format code
pixi run test        # Run tests
pixi run typecheck   # Type checking

# Clean up
pixi run clean       # Remove __pycache__ directories
```

### Environment Activation

```bash
# Activate shell (like conda activate)
pixi shell

# Run single command in environment
pixi run python --version

# Execute complex commands
pixi run "cd agents/template && python -m src.agent --config custom.json"
```

## 🔧 Customizing for Your Project

### Adding Scientific Packages

```bash
# Bioinformatics
pixi add biopython blast clustalw muscle

# Machine learning
pixi add scikit-learn pytorch tensorflow

# Data analysis
pixi add jupyter notebook plotly

# Specific versions
pixi add "numpy>=1.24,<2.0"
```

### Adding Custom Tasks

Edit `pixi.toml`:

```toml
[tasks]
# Your custom agent configurations
my-agent = "cd agents/template && python -m src.agent --name my-custom-agent"

# Data processing pipeline
process-data = """
cd data && \
python preprocess.py && \
pixi run agent-run
"""

# Development workflow
dev-setup = ["pixi run format", "pixi run lint", "pixi run test"]

# Custom MCP server testing
test-my-mcp = "cd mcps/my_server/src && python server.py"
```

### Environment Variables

Create `.env` file:

```bash
# .env (git-ignored)
CBORG_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_claude_key

# Custom settings
AGENT_LOG_LEVEL=DEBUG
MCP_SERVER_TIMEOUT=30
```

Reference in code:

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("CBORG_API_KEY")
```

## 🏗️ Development Workflow

### 1. Setup New Feature

```bash
# Create feature branch
git checkout -b feature/my-new-agent

# Ensure environment is up to date
pixi install

# Start development
pixi shell
```

### 2. Development Cycle

```bash
# Make changes to code
# ...

# Check code quality
pixi run lint
pixi run typecheck

# Format code
pixi run format

# Test changes
pixi run test
pixi run agent-test
```

### 3. Testing MCP Servers

```bash
# Test individual MCP servers
pixi run test-biopython
pixi run test-filesystem

# Test with agent
pixi run agent-run
```

## 🐛 Troubleshooting

### Common Issues

**Problem**: `pixi command not found`
```bash
# Solution: Add pixi to PATH
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Problem**: Dependencies not resolving
```bash
# Solution: Clear cache and reinstall
pixi clean
pixi install
```

**Problem**: Platform compatibility
```bash
# Solution: Add your platform to pixi.toml
[project]
platforms = ["linux-64", "osx-64", "osx-arm64", "win-64"]
```

### Debugging Environment

```bash
# Check what's installed
pixi list

# Verify Python environment
pixi run python -c "import sys; print(sys.executable)"

# Check package versions
pixi run python -c "import fastmcp; print(fastmcp.__version__)"

# Environment variables
pixi run env | grep -E "(CONDA|PIXI|PATH)"
```

## 🚀 Advanced Usage

### Multiple Environments

```toml
# pixi.toml
[environments]
default = ["base"]
dev = ["base", "dev"]
gpu = ["base", "gpu"]

[dependencies]
# Base environment
python = ">=3.11"
fastmcp = "*"

[feature.dev.dependencies]
pytest = "*"
black = "*"

[feature.gpu.dependencies]
pytorch = { version = "*", channel = "pytorch" }
```

```bash
# Use specific environment
pixi run -e dev test
pixi run -e gpu train-model
```

### Cross-Platform Scripts

```toml
[tasks]
# Works on all platforms
clean = { cmd = "python -c \"import shutil; shutil.rmtree('__pycache__', ignore_errors=True)\"" }

# Platform-specific
[tasks.windows]
clean = "rmdir /s /q __pycache__ 2>nul || echo 'Nothing to clean'"

[tasks.unix]
clean = "rm -rf __pycache__"
```

## 💡 Best Practices

1. **Pin Important Versions**: For reproducibility
   ```toml
   python = "3.11.*"
   fastmcp = ">=1.0,<2.0"
   ```

2. **Use Feature Groups**: Organize dependencies
   ```toml
   [feature.ml.dependencies]
   scikit-learn = "*"
   pytorch = "*"
   ```

3. **Document Tasks**: Clear descriptions
   ```toml
   [tasks]
   agent-run = { cmd = "cd agents/template && python -m src.agent", description = "Run the NeLLi AI Agent" }
   ```

4. **Environment Variables**: Use `.env` for secrets

5. **Lock File**: Commit `pixi.lock` for reproducibility

Pixi provides a robust foundation for managing the NeLLi AI Scientist Agent Template's complex dependency requirements while maintaining reproducibility across different development environments.