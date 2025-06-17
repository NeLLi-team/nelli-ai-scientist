#!/usr/bin/env python3
"""
Launcher for the Enhanced Universal MCP Agent
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the agent directory (where this script is located)
AGENT_DIR = Path(__file__).parent
REPO_ROOT = AGENT_DIR.parent.parent

# Add the src directory to the path
sys.path.insert(0, str(AGENT_DIR / "src"))

from src.enhanced_agent import main

if __name__ == "__main__":
    # Change to repository root for proper file access
    original_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    
    # Load environment variables from .env file
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"🔐 Loaded environment variables from {env_file}")
    else:
        print(f"⚠️  No .env file found at {env_file}")
    
    print(f"🏠 Changed working directory to: {os.getcwd()}")
    print(f"📁 Agent config path: {AGENT_DIR / 'config/agent_config.yaml'}")
    
    try:
        # Set default paths relative to agent directory
        if len(sys.argv) == 1:  # No arguments provided
            # Point to the agent's MCP config file
            agent_mcp_config = str(AGENT_DIR / "mcp_config.json")
            sys.argv.extend([
                "--config", agent_mcp_config,
            ])
        
        # Store agent directory for config loading
        os.environ["NELLI_AGENT_DIR"] = str(AGENT_DIR)
        
        # Run the enhanced agent
        asyncio.run(main())
    finally:
        # Restore original working directory
        os.chdir(original_cwd)