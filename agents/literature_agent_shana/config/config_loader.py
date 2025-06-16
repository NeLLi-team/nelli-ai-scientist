import json
import os
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AgentConfig:
    """Configuration for the results analyzer agent."""
    name: str
    version: str
    description: str
    capabilities: list[str]
    llm_provider: str
    llm_providers: Dict[str, Dict[str, str]]
    paper_summarization: Dict[str, Any]
    logging: Dict[str, str]

def load_config(config_path: str = None) -> AgentConfig:
    """Load and validate the agent configuration.
    
    Args:
        config_path: Path to the config file. If None, uses default location.
        
    Returns:
        AgentConfig: Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
        ValueError: If config is missing required fields
    """
    if config_path is None:
        # Use default config path relative to this file
        config_path = Path(__file__).parent / "agent_config.json"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Validate required sections
    required_sections = ['agent', 'paper_summarization', 'logging']
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate agent section
    agent = config_dict['agent']
    required_agent_fields = ['name', 'version', 'capabilities', 'llm_provider']
    for field in required_agent_fields:
        if field not in agent:
            raise ValueError(f"Missing required agent field: {field}")
    
    # Create and return config object
    return AgentConfig(
        name=agent['name'],
        version=agent['version'],
        description=agent.get('description', ''),
        capabilities=agent['capabilities'],
        llm_provider=agent['llm_provider'],
        llm_providers=agent.get('llm_providers', {}),
        paper_summarization=config_dict['paper_summarization'],
        logging=config_dict['logging']
    )

# Example usage:
if __name__ == "__main__":
    try:
        config = load_config()
        print(f"Loaded configuration for {config.name} v{config.version}")
        print(f"Capabilities: {', '.join(config.capabilities)}")
    except Exception as e:
        print(f"Error loading configuration: {e}") 