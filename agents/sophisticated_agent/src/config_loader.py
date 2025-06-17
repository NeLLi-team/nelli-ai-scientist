"""
Configuration Loader for Enhanced Agent
Loads configuration from YAML files and command line arguments
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages agent configuration from YAML files"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to config directory relative to this file
            config_path = Path(__file__).parent.parent / "config" / "agent_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is missing"""
        return {
            "agent": {
                "name": "nelli-enhanced-agent",
                "description": "Enhanced Universal MCP Agent",
                "enhanced_features": {
                    "reasoning": {
                        "enabled": True,
                        "model": "google/gemini-pro",
                        "temperature": 0.3,
                        "max_tokens": 2000
                    },
                    "planning": {
                        "enabled": True,
                        "model": "google/gemini-flash-lite",
                        "temperature": 0.2,
                        "max_tokens": 1500,
                        "max_iterations": 5
                    },
                    "execution": {
                        "model": "google/gemini-flash-lite",
                        "temperature": 0.1,
                        "max_tokens": 1000
                    },
                    "progress_tracking": {
                        "enabled": True,
                        "reports_directory": "../../reports/sophisticated_agent",
                        "save_plans": True,
                        "save_progress": True,
                        "colored_output": True
                    }
                },
                "llm": {
                    "provider": "cborg",
                    "temperature": 0.7,
                    "max_tokens": 4096
                }
            }
        }
    
    def get_agent_name(self) -> str:
        """Get agent name from config"""
        return self.config.get("agent", {}).get("name", "nelli-enhanced-agent")
    
    def get_agent_description(self) -> str:
        """Get agent description from config"""
        return self.config.get("agent", {}).get("description", "Enhanced Universal MCP Agent")
    
    def get_llm_provider(self) -> str:
        """Get LLM provider from config"""
        return self.config.get("agent", {}).get("llm", {}).get("provider", "cborg")
    
    def get_reasoning_and_planning_config(self) -> Dict[str, Any]:
        """Get reasoning and planning phase configuration"""
        return self.config.get("agent", {}).get("enhanced_features", {}).get("reasoning_and_planning", {
            "enabled": True,
            "model": "google/gemini-pro",
            "temperature": 0.3,
            "max_tokens": 4000
        })
    
    # Backward compatibility methods
    def get_reasoning_config(self) -> Dict[str, Any]:
        """Get reasoning phase configuration (backward compatibility)"""
        return self.get_reasoning_and_planning_config()
    
    def get_planning_config(self) -> Dict[str, Any]:
        """Get planning phase configuration (backward compatibility)"""
        config = self.get_reasoning_and_planning_config()
        # Return planning-specific defaults if needed
        return {
            "enabled": config.get("enabled", True),
            "model": config.get("model", "google/gemini-pro"),
            "temperature": 0.2,  # Slightly more focused for planning
            "max_tokens": config.get("max_tokens", 4000),
            "max_iterations": 10
        }
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution phase configuration"""
        return self.config.get("agent", {}).get("enhanced_features", {}).get("execution", {
            "model": "google/gemini-flash-lite",
            "temperature": 0.1,
            "max_tokens": 1000
        })
    
    def get_progress_tracking_config(self) -> Dict[str, Any]:
        """Get progress tracking configuration"""
        return self.config.get("agent", {}).get("enhanced_features", {}).get("progress_tracking", {
            "enabled": True,
            "reports_directory": "../../reports/sophisticated_agent",
            "save_plans": True,
            "save_progress": True,
            "colored_output": True
        })
    
    def is_reasoning_enabled(self) -> bool:
        """Check if reasoning is enabled"""
        return self.get_reasoning_config().get("enabled", True)
    
    def is_planning_enabled(self) -> bool:
        """Check if planning is enabled"""
        return self.get_planning_config().get("enabled", True)
    
    def is_progress_tracking_enabled(self) -> bool:
        """Check if progress tracking is enabled"""
        return self.get_progress_tracking_config().get("enabled", True)
    
    def get_reasoning_and_planning_model(self) -> str:
        """Get model for reasoning and planning phases"""
        return self.get_reasoning_and_planning_config().get("model", "google/gemini-pro")
    
    def get_execution_model(self) -> str:
        """Get model for execution phase"""
        return self.get_execution_config().get("model", "google/gemini-flash-lite")
    
    # Backward compatibility methods
    def get_reasoning_model(self) -> str:
        """Get model for reasoning phase (backward compatibility)"""
        return self.get_reasoning_and_planning_model()
    
    def get_planning_model(self) -> str:
        """Get model for planning phase (backward compatibility)"""
        return self.get_reasoning_and_planning_model()
    
    def get_reports_directory(self) -> str:
        """Get reports directory path"""
        return self.get_progress_tracking_config().get("reports_directory", "../../reports/sophisticated_agent")
    
    def get_max_planning_iterations(self) -> int:
        """Get maximum planning iterations"""
        return self.get_execution_config().get("max_iterations", 10)
    
    def override_with_args(self, args) -> None:
        """Override configuration with command line arguments"""
        if hasattr(args, 'name') and args.name:
            self.config.setdefault("agent", {})["name"] = args.name
        
        if hasattr(args, 'reasoning_model') and args.reasoning_model:
            reasoning_config = self.config.setdefault("agent", {}).setdefault("enhanced_features", {}).setdefault("reasoning", {})
            reasoning_config["model"] = args.reasoning_model
        
        if hasattr(args, 'planning_model') and args.planning_model:
            planning_config = self.config.setdefault("agent", {}).setdefault("enhanced_features", {}).setdefault("planning", {})
            planning_config["model"] = args.planning_model
            
            # Also update execution model if not separately specified
            execution_config = self.config.setdefault("agent", {}).setdefault("enhanced_features", {}).setdefault("execution", {})
            if "model" not in execution_config:
                execution_config["model"] = args.planning_model
        
        if hasattr(args, 'disable_reasoning') and args.disable_reasoning:
            reasoning_config = self.config.setdefault("agent", {}).setdefault("enhanced_features", {}).setdefault("reasoning", {})
            reasoning_config["enabled"] = False
        
        if hasattr(args, 'disable_planning') and args.disable_planning:
            planning_config = self.config.setdefault("agent", {}).setdefault("enhanced_features", {}).setdefault("planning", {})
            planning_config["enabled"] = False
        
        if hasattr(args, 'disable_progress') and args.disable_progress:
            progress_config = self.config.setdefault("agent", {}).setdefault("enhanced_features", {}).setdefault("progress_tracking", {})
            progress_config["enabled"] = False
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        return self.config.get("agent", {}).get("available_models", {})
    
    def display_config_summary(self) -> None:
        """Display configuration summary"""
        print(f"\n📋 Configuration Summary:")
        print(f"   Agent: {self.get_agent_name()}")
        print(f"   Provider: {self.get_llm_provider()}")
        print(f"   Reasoning: {'✅' if self.is_reasoning_enabled() else '❌'} ({self.get_reasoning_model()})")
        print(f"   Planning: {'✅' if self.is_planning_enabled() else '❌'} ({self.get_planning_model()})")
        print(f"   Progress: {'✅' if self.is_progress_tracking_enabled() else '❌'}")
        print(f"   Reports: {self.get_reports_directory()}")