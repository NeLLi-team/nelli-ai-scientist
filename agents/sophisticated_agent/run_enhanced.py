#!/usr/bin/env python3
"""
Launcher for the Enhanced Universal MCP Agent
"""

import asyncio
import sys
import os
from pathlib import Path

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
    
    print(f"🏠 Changed working directory to: {os.getcwd()}")
    print(f"📁 Agent config path: {AGENT_DIR / 'config/agent_config.yaml'}")
    
    try:
        # Set default paths relative to repo root
        if len(sys.argv) == 1:  # No arguments provided
            sys.argv.extend([
                "--config", "mcp_config.json",
                # We'll handle the config file path in the enhanced_agent.py
            ])
        
        # Store agent directory for config loading
        os.environ["NELLI_AGENT_DIR"] = str(AGENT_DIR)
        
        # Run the enhanced agent
        asyncio.run(main())
    finally:
        # Restore original working directory
        os.chdir(original_cwd)