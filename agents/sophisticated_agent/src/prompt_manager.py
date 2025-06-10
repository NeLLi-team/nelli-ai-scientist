"""
Prompt Manager - Handles loading and formatting system prompts
"""

import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages system prompts for the agent"""
    
    def __init__(self, prompts_dir: str = None):
        """Initialize the prompt manager
        
        Args:
            prompts_dir: Directory containing prompt files (defaults to ../prompts)
        """
        if prompts_dir is None:
            # Default to prompts directory relative to this file
            prompts_dir = Path(__file__).parent.parent / "prompts"
        
        self.prompts_dir = Path(prompts_dir)
        self._prompt_cache = {}
        
        # Ensure prompts directory exists
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            
    def load_prompt(self, prompt_name: str) -> str:
        """Load a prompt from file
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            
        Returns:
            The prompt template as a string
        """
        # Check cache first
        if prompt_name in self._prompt_cache:
            return self._prompt_cache[prompt_name]
        
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            logger.error(f"Prompt file not found: {prompt_file}")
            return f"Error: Prompt '{prompt_name}' not found"
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
            
            # Cache the prompt
            self._prompt_cache[prompt_name] = prompt_content
            return prompt_content
            
        except Exception as e:
            logger.error(f"Failed to load prompt '{prompt_name}': {e}")
            return f"Error loading prompt: {e}"
    
    def format_prompt(self, prompt_name: str, **kwargs) -> str:
        """Load and format a prompt with provided variables
        
        Args:
            prompt_name: Name of the prompt to load
            **kwargs: Variables to substitute in the prompt
            
        Returns:
            The formatted prompt
        """
        prompt_template = self.load_prompt(prompt_name)
        
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable for prompt '{prompt_name}': {e}")
            return f"Error: Missing variable {e} in prompt '{prompt_name}'"
        except Exception as e:
            logger.error(f"Failed to format prompt '{prompt_name}': {e}")
            return f"Error formatting prompt: {e}"
    
    def list_prompts(self) -> list:
        """List all available prompts
        
        Returns:
            List of available prompt names (without .txt extension)
        """
        if not self.prompts_dir.exists():
            return []
        
        prompt_files = []
        for file in self.prompts_dir.glob("*.txt"):
            prompt_files.append(file.stem)
        
        return sorted(prompt_files)
    
    def reload_prompts(self):
        """Clear the prompt cache to force reloading from files"""
        self._prompt_cache.clear()
        logger.info("Prompt cache cleared - prompts will be reloaded on next access")