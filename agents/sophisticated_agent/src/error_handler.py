"""
Error handling module for robust agent execution
Provides validation, recovery, and user-friendly error messages
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import traceback

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur during execution"""
    VALIDATION_ERROR = "validation_error"
    TOOL_NOT_FOUND = "tool_not_found"
    PARAMETER_MISSING = "parameter_missing"
    EXECUTION_ERROR = "execution_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorRecoveryStrategy(Enum):
    """Strategies for recovering from errors"""
    RETRY = "retry"
    USE_DEFAULT = "use_default"
    SKIP_STEP = "skip_step"
    ALTERNATIVE_TOOL = "alternative_tool"
    FAIL_GRACEFULLY = "fail_gracefully"


class ErrorHandler:
    """Handles errors with validation, recovery, and user-friendly messaging"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorType, List[ErrorRecoveryStrategy]] = {
            ErrorType.VALIDATION_ERROR: [ErrorRecoveryStrategy.USE_DEFAULT, ErrorRecoveryStrategy.SKIP_STEP],
            ErrorType.TOOL_NOT_FOUND: [ErrorRecoveryStrategy.ALTERNATIVE_TOOL, ErrorRecoveryStrategy.SKIP_STEP],
            ErrorType.PARAMETER_MISSING: [ErrorRecoveryStrategy.USE_DEFAULT, ErrorRecoveryStrategy.SKIP_STEP],
            ErrorType.EXECUTION_ERROR: [ErrorRecoveryStrategy.RETRY, ErrorRecoveryStrategy.FAIL_GRACEFULLY],
            ErrorType.CONNECTION_ERROR: [ErrorRecoveryStrategy.RETRY, ErrorRecoveryStrategy.FAIL_GRACEFULLY],
            ErrorType.TIMEOUT_ERROR: [ErrorRecoveryStrategy.RETRY, ErrorRecoveryStrategy.SKIP_STEP],
            ErrorType.UNKNOWN_ERROR: [ErrorRecoveryStrategy.FAIL_GRACEFULLY]
        }
        
        # Default parameter values for common tools
        self.default_parameters = {
            "tree_view": {
                "path": ".",
                "file_extensions": "",  # Empty string instead of None
                "ignore_hidden": True,
                "max_depth": 3
            },
            "find_files": {
                "path": ".",
                "pattern": "*",
                "recursive": True
            },
            "read_file": {
                "path": None  # Required - no default
            }
        }
    
    def validate_tool_parameters(self, tool_name: str, parameters: Dict[str, Any], 
                               tool_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate and fix tool parameters before execution
        
        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate
            tool_schema: Optional schema from tool discovery
            
        Returns:
            Validated and fixed parameters
        """
        validated_params = parameters.copy()
        
        # Apply defaults for known tools
        if tool_name in self.default_parameters:
            defaults = self.default_parameters[tool_name]
            for param, default_value in defaults.items():
                if param not in validated_params or validated_params[param] is None:
                    if default_value is not None:
                        validated_params[param] = default_value
                        logger.info(f"Applied default value for {tool_name}.{param}: {default_value}")
        
        # Validate against schema if provided
        if tool_schema:
            properties = tool_schema.get("properties", {})
            required = tool_schema.get("required", [])
            
            # Check required parameters
            for req_param in required:
                if req_param not in validated_params or validated_params[req_param] is None:
                    # Try to infer or use a sensible default
                    param_type = properties.get(req_param, {}).get("type", "string")
                    if param_type == "string":
                        validated_params[req_param] = ""
                    elif param_type == "boolean":
                        validated_params[req_param] = False
                    elif param_type == "number" or param_type == "integer":
                        validated_params[req_param] = 0
                    elif param_type == "array":
                        validated_params[req_param] = []
                    elif param_type == "object":
                        validated_params[req_param] = {}
                    
                    logger.warning(f"Missing required parameter {req_param} for {tool_name}, using default: {validated_params[req_param]}")
            
            # Remove None values and convert them to appropriate defaults
            for param, value in list(validated_params.items()):
                if value is None:
                    param_info = properties.get(param, {})
                    param_type = param_info.get("type", "string")
                    
                    if param_type == "string":
                        validated_params[param] = ""
                    elif param_type == "boolean":
                        validated_params[param] = param_info.get("default", False)
                    elif param_type in ["number", "integer"]:
                        validated_params[param] = param_info.get("default", 0)
                    else:
                        # Remove the parameter if we can't determine a good default
                        del validated_params[param]
        
        return validated_params
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorType:
        """Classify an error to determine the appropriate recovery strategy"""
        error_str = str(error).lower()
        
        if "validation error" in error_str or "invalid" in error_str:
            return ErrorType.VALIDATION_ERROR
        elif "tool" in error_str and "not found" in error_str:
            return ErrorType.TOOL_NOT_FOUND
        elif "missing" in error_str and "parameter" in error_str:
            return ErrorType.PARAMETER_MISSING
        elif "timeout" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "connection" in error_str or "network" in error_str:
            return ErrorType.CONNECTION_ERROR
        elif isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorType.VALIDATION_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def get_user_friendly_message(self, error: Exception, error_type: ErrorType, 
                                context: Dict[str, Any] = None) -> str:
        """Convert technical errors into user-friendly messages"""
        tool_name = context.get("tool_name", "unknown") if context else "unknown"
        
        messages = {
            ErrorType.VALIDATION_ERROR: f"The parameters provided for '{tool_name}' need adjustment. I'll fix this and try again.",
            ErrorType.TOOL_NOT_FOUND: f"The tool '{tool_name}' isn't available. I'll try an alternative approach.",
            ErrorType.PARAMETER_MISSING: f"Some required information for '{tool_name}' is missing. I'll use default values.",
            ErrorType.EXECUTION_ERROR: f"There was an issue executing '{tool_name}'. Retrying with adjusted parameters.",
            ErrorType.CONNECTION_ERROR: "There's a connection issue. I'll retry in a moment.",
            ErrorType.TIMEOUT_ERROR: f"The operation '{tool_name}' is taking longer than expected. Trying a different approach.",
            ErrorType.UNKNOWN_ERROR: "An unexpected issue occurred. I'll try a different approach."
        }
        
        return messages.get(error_type, "An error occurred. Attempting recovery...")
    
    def suggest_recovery_action(self, error_type: ErrorType, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Suggest a recovery action based on error type and context"""
        strategies = self.recovery_strategies.get(error_type, [ErrorRecoveryStrategy.FAIL_GRACEFULLY])
        
        # Track error frequency
        error_key = f"{error_type.value}:{context.get('tool_name', 'unknown')}" if context else error_type.value
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Choose strategy based on error count and available strategies
        if self.error_counts[error_key] > 3:
            # Too many errors, fail gracefully
            return {
                "strategy": ErrorRecoveryStrategy.FAIL_GRACEFULLY,
                "action": "skip_and_continue",
                "message": "Multiple attempts failed. Moving to next step."
            }
        
        # Select primary strategy
        strategy = strategies[0] if strategies else ErrorRecoveryStrategy.FAIL_GRACEFULLY
        
        recovery_actions = {
            ErrorRecoveryStrategy.RETRY: {
                "strategy": strategy,
                "action": "retry_with_fixes",
                "max_retries": 3,
                "message": "Retrying with corrected parameters..."
            },
            ErrorRecoveryStrategy.USE_DEFAULT: {
                "strategy": strategy,
                "action": "use_defaults",
                "message": "Using default values for missing parameters..."
            },
            ErrorRecoveryStrategy.SKIP_STEP: {
                "strategy": strategy,
                "action": "skip",
                "message": "Skipping this step and continuing..."
            },
            ErrorRecoveryStrategy.ALTERNATIVE_TOOL: {
                "strategy": strategy,
                "action": "find_alternative",
                "message": "Looking for an alternative tool..."
            },
            ErrorRecoveryStrategy.FAIL_GRACEFULLY: {
                "strategy": strategy,
                "action": "graceful_failure",
                "message": "This step couldn't be completed, but continuing with workflow..."
            }
        }
        
        return recovery_actions.get(strategy, recovery_actions[ErrorRecoveryStrategy.FAIL_GRACEFULLY])
    
    def create_error_report(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a structured error report for logging and analysis"""
        error_type = self.classify_error(error, context)
        
        # Get sanitized traceback (last 3 frames only)
        tb_lines = traceback.format_exc().split('\n')
        sanitized_traceback = '\n'.join(tb_lines[-10:]) if len(tb_lines) > 10 else traceback.format_exc()
        
        report = {
            "error_type": error_type.value,
            "error_message": str(error),
            "user_message": self.get_user_friendly_message(error, error_type, context),
            "recovery_suggestion": self.suggest_recovery_action(error_type, context),
            "context": context or {},
            "technical_details": {
                "exception_type": type(error).__name__,
                "traceback_summary": sanitized_traceback
            }
        }
        
        return report


class ParameterResolver:
    """Resolves and validates parameters with intelligent defaults"""
    
    @staticmethod
    def resolve_path_parameter(path: Any, default: str = ".") -> str:
        """Resolve path parameters with validation"""
        if path is None or path == "None":
            return default
        
        path_str = str(path).strip()
        if not path_str:
            return default
            
        # Handle common path issues
        if path_str == "~":
            import os
            return os.path.expanduser("~")
        
        return path_str
    
    @staticmethod
    def resolve_pattern_parameter(pattern: Any, default: str = "*") -> str:
        """Resolve pattern parameters with validation"""
        if pattern is None or pattern == "None":
            return default
        
        pattern_str = str(pattern).strip()
        if not pattern_str:
            return default
            
        return pattern_str
    
    @staticmethod
    def resolve_boolean_parameter(value: Any, default: bool = False) -> bool:
        """Resolve boolean parameters with validation"""
        if value is None:
            return default
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            return value.lower() in ["true", "yes", "1", "on"]
        
        return bool(value)