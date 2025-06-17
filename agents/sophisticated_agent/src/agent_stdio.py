"""
Enhanced Universal MCP Agent with Stdio Support
Supports both FastMCP and standard MCP stdio connections
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pathlib import Path

from pydantic import BaseModel
try:
    from .llm_interface import LLMInterface, LLMProvider
    from .communication import FIPAMessage, Performative
    from .prompt_manager import PromptManager
    from .error_handler import ErrorHandler, ParameterResolver
    from .mcp_stdio_client import MCPConnectionManager
    from .biological_analysis_engine import BiologicalAnalysisEngine
    from .adaptive_code_solver import AdaptiveCodeSolver
except ImportError:
    # Fallback for direct execution
    from llm_interface import LLMInterface, LLMProvider
    from communication import FIPAMessage, Performative
    from prompt_manager import PromptManager
    from error_handler import ErrorHandler, ParameterResolver
    from mcp_stdio_client import MCPConnectionManager
    from biological_analysis_engine import BiologicalAnalysisEngine
    from adaptive_code_solver import AdaptiveCodeSolver

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Reduce noise from external libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('fastmcp').setLevel(logging.WARNING)
logging.getLogger('mcp').setLevel(logging.WARNING)


class AgentConfig(BaseModel):
    """Agent configuration model"""
    name: str
    role: str = "universal_assistant"
    description: str = "A universal agent that can work with any MCP tools"
    llm_provider: LLMProvider = LLMProvider.CBORG
    temperature: float = 0.7
    max_tokens: int = 4096
    mcp_config_path: str = "mcp_config.json"
    use_stdio_connections: bool = True  # Enable stdio connections by default


class UniversalMCPAgentStdio:
    """
    Universal AI Agent with stdio support for truly independent MCP servers
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMInterface(provider=config.llm_provider)
        self.conversation_history: List[Dict[str, Any]] = []
        self.agent_id = f"{config.name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # MCP servers and tools
        self.mcp_servers = {}
        self.discovered_tools = {}
        self.tool_categories = {}
        
        # Connection managers
        self.stdio_manager = MCPConnectionManager() if config.use_stdio_connections else None
        self.fastmcp_tools = {}  # For backward compatibility
        
        # Initialize prompt manager and error handler
        self.prompt_manager = PromptManager()
        self.error_handler = ErrorHandler()
        self.bio_engine = BiologicalAnalysisEngine()  # Enhanced biological analysis
        self.adaptive_solver = AdaptiveCodeSolver(mcp_tool_caller=self._call_mcp_tool)  # Adaptive code generation for any task
        
        # Set up chat session persistence
        self.chat_session_id = f"chat-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.chat_log_dir = Path(__file__).parent.parent.parent.parent / "logs" / "chat_sessions"
        self.chat_log_dir.mkdir(parents=True, exist_ok=True)
        self.chat_log_file = self.chat_log_dir / f"{self.chat_session_id}.jsonl"
        
        # Reduced initialization verbosity
        pass
        
        # Load previous conversation history if exists
        self._load_conversation_history()

    async def initialize(self):
        """Initialize the agent by discovering all available MCP servers and tools"""
        # Suppress verbose initialization messages
        
        # Load MCP configuration
        self._load_mcp_config()
        
        # Connect to servers and discover tools
        if self.mcp_servers:
            await self._connect_all_servers()
            self.discovered_tools = await self._discover_all_tools()
            # Only show final summary, not verbose discovery process
        else:
            logger.warning("⚠️  No MCP servers configured")

    def _load_mcp_config(self):
        """Load MCP server configuration from JSON file"""
        config_path = Path(self.config.mcp_config_path)
        
        if not config_path.exists():
            logger.warning(f"MCP config file not found at: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                mcp_config = json.load(f)
            
            # Load all enabled MCP servers
            for server_id, server_config in mcp_config.get("mcp_servers", {}).items():
                enabled = server_config.get("enabled", False)
                if enabled:
                    self.mcp_servers[server_id] = server_config
                    # Reduced verbosity during server loading
                    
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")

    async def _connect_all_servers(self):
        """Connect to all configured MCP servers"""
        for server_id, server_config in self.mcp_servers.items():
            try:
                # Determine connection type
                use_stdio = server_config.get("use_stdio", True)
                
                if use_stdio and self.stdio_manager:
                    # Use stdio connection for true isolation
                    success = await self.stdio_manager.connect_server(server_id, server_config)
                    # Reduced connection verbosity
                else:
                    # Fallback to FastMCP if available
                    pass  # Reduced verbosity
                    
            except Exception as e:
                server_name = server_config.get("name", server_id)
                logger.error(f"Failed to connect to {server_name}: {e}")

    async def _discover_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Discover all tools from all connected servers"""
        all_tools = {}
        
        # Get tools from stdio connections
        if self.stdio_manager:
            stdio_tools = await self.stdio_manager.discover_all_tools()
            all_tools.update(stdio_tools)
            
            # Organize tools by category
            for tool_name, tool_info in stdio_tools.items():
                server_id = tool_info.get("server_id")
                if server_id not in self.tool_categories:
                    self.tool_categories[server_id] = {
                        "name": tool_info.get("server_name", server_id),
                        "description": self.mcp_servers[server_id].get("description", ""),
                        "tools": []
                    }
                self.tool_categories[server_id]["tools"].append(tool_name)
        
        # Add any FastMCP tools (backward compatibility)
        all_tools.update(self.fastmcp_tools)
        
        return all_tools

    async def _call_mcp_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool with the given arguments"""
        if tool_name not in self.discovered_tools:
            # Suggest alternative tools for common mistakes
            suggestions = []
            if tool_name == "find_genes":
                suggestions.append("gene_prediction_and_coding_stats")
            elif tool_name == "detect_promoters":
                suggestions.append("promoter_identification")
            elif tool_name == "calculate_gc_skew":
                suggestions.append("gc_skew_analysis")
            elif tool_name == "calculate_sequence_stats":
                suggestions.append("sequence_stats")
            
            error_msg = f"Unknown tool: {tool_name}"
            if suggestions:
                error_msg += f". Did you mean: {', '.join(suggestions)}?"
            else:
                # Find similar tool names
                available_tools = list(self.discovered_tools.keys())
                similar = [t for t in available_tools if tool_name.lower() in t.lower() or t.lower() in tool_name.lower()]
                if similar:
                    error_msg += f". Similar tools available: {', '.join(similar[:3])}"
            
            return {"error": error_msg}
        
        tool_info = self.discovered_tools[tool_name]
        server_id = tool_info.get("server_id")
        
        logger.info(f"🔧 Calling tool '{tool_name}' on server '{server_id}'")
        logger.debug(f"   Arguments: {tool_args}")
        
        # Special validation for common mistakes
        if tool_name == "sequence_stats" and "sequence" in tool_args:
            sequence = tool_args["sequence"]
            # Check if user passed a filename instead of sequence
            if isinstance(sequence, str) and ('.' in sequence or '/' in sequence or len(sequence) < 20):
                return {
                    "error": f"sequence_stats requires actual DNA/RNA sequence, not filename. Use 'analyze_fasta_file' for FASTA files instead.",
                    "suggestion": "For FASTA file analysis, use 'analyze_fasta_file' tool with the file path."
                }

        try:
            # Use stdio connection if available
            if self.stdio_manager and server_id in self.stdio_manager.server_configs:
                result = await self.stdio_manager.call_tool(tool_name, tool_info, tool_args)
                # Always format the result to extract from TextContent
                formatted = self._format_tool_result(result)
                logger.debug(f"Formatted result for {tool_name}: {type(formatted)}")
                return formatted
            else:
                # Fallback to FastMCP (if implemented)
                return {"error": f"No connection available for server {server_id}"}
                
        except Exception as e:
            logger.error(f"Tool execution failed: {type(e).__name__}: {e}")
            
            # Special handling for sequence validation errors
            if "Invalid characters found" in str(e) and tool_name == "sequence_stats":
                return {
                    "error": "sequence_stats received invalid characters (likely a filename instead of sequence). Use 'analyze_fasta_file' for FASTA files.",
                    "suggestion": "For FASTA file analysis, use 'analyze_fasta_file' tool with the file path."
                }
            
            error_info = self.error_handler.create_error_report(e, {"tool_name": tool_name, "arguments": tool_args})
            return {"error": str(e), "recovery_suggestions": error_info.get("recovery_suggestion", {})}

    def _format_tool_result(self, result: Any) -> Dict[str, Any]:
        """Format tool result consistently - handles FastMCP and MCP results"""
        try:
            # Use the comprehensive TextContent extraction
            cleaned_result = self._extract_text_content_recursively(result)
            
            # Ensure we always return a dict
            if isinstance(cleaned_result, dict):
                return cleaned_result
            else:
                return {"result": cleaned_result}
                
        except Exception as e:
            logger.warning(f"Error in _format_tool_result: {e}")
            # Fallback to original logic if extraction fails
            if isinstance(result, dict):
                return result
            else:
                return {"result": str(result)}

    async def process_message(self, content: str) -> str:
        """Process a user message and return a response"""
        # Create FIPA message
        message = FIPAMessage(
            performative=Performative.REQUEST,
            sender="user",
            receiver=self.agent_id,
            content={"message": content},
            conversation_id=f"conv-{datetime.now().timestamp()}"
        )
        
        # Add to conversation history
        user_entry = {
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(user_entry)
        self._save_conversation_entry(user_entry)
        
        # Get system prompts
        system_prompt = self.prompt_manager.format_prompt("general_response", 
            user_input=content,
            agent_role=self.config.role,
            tools=self._get_tool_descriptions()
        )
        
        # Load cached data from recent analysis files
        cached_data = None
        try:
            analysis_files = list(Path("reports").glob("*analysis*.json"))
            if analysis_files:
                latest_file = max(analysis_files, key=lambda f: f.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    cached_data = json.load(f)
        except Exception as e:
            logger.debug(f"Error loading cached analysis: {e}")
        
        # Priority 1: Check for quick biological intelligence responses first when cached data is available
        if cached_data:
            try:
                biological_response = self._provide_biological_intelligence(content, cached_data)
                if biological_response:
                    logger.info("🧬 Using biological intelligence for cached data query...")
                    assistant_entry = {
                        "role": "assistant", 
                        "content": biological_response,
                        "tool_calls": [],
                        "timestamp": datetime.now().isoformat(),
                        "intelligence_type": "biological_intelligence"
                    }
                    self.conversation_history.append(assistant_entry)
                    self._save_conversation_entry(assistant_entry)
                    return biological_response
            except Exception as e:
                logger.debug(f"Biological intelligence failed: {e}")
        
        # Priority 2: Use reasoning model for complex analysis when biological intelligence doesn't apply
        try:
            logger.info("🧠 Activating reasoning model for dynamic problem solving...")
            reasoning_response = await self._solve_with_reasoning_model(content, cached_data)
            
            if reasoning_response:
                assistant_entry = {
                    "role": "assistant",
                    "content": reasoning_response,
                    "tool_calls": [],
                    "timestamp": datetime.now().isoformat(),
                    "intelligence_type": "reasoning_model_solution"
                }
                self.conversation_history.append(assistant_entry)
                self._save_conversation_entry(assistant_entry)
                return reasoning_response
                
        except Exception as e:
            logger.debug(f"Reasoning model failed: {e}")
        
        # Fallback: Direct adaptive code solver for complex requests that need new analysis
        try:
            solver_result = await self.adaptive_solver.solve_user_request(content, cached_data)
            if solver_result.get("success"):
                solution = solver_result["solution"]
                execution_result = solution.get("execution_result", {})
                
                adaptive_response = f"## 🔧 Adaptive Code Solution\n\n"
                adaptive_response += f"**Request:** {content}\n\n"
                
                if execution_result.get("stdout"):
                    adaptive_response += f"**Result:**\n```\n{execution_result['stdout']}\n```\n"
                
                assistant_entry = {
                    "role": "assistant",
                    "content": adaptive_response,
                    "tool_calls": [],
                    "timestamp": datetime.now().isoformat(),
                    "intelligence_type": "adaptive_fallback"
                }
                self.conversation_history.append(assistant_entry)
                self._save_conversation_entry(assistant_entry)
                return adaptive_response
        except Exception as e:
            logger.debug(f"Adaptive fallback failed: {e}")
        
        # Final fallback: Basic LLM analysis
        # Process with LLM (simplified approach like original agent)
        # Since our LLMInterface doesn't support tools directly, 
        # we'll use the planning approach from the original agent
        analysis = await self.process_natural_language(content)
        
        # Handle the analysis result
        tool_results = []
        file_warnings = []
        if analysis.get("suggested_tools"):
            for suggestion in analysis["suggested_tools"]:
                # Check for large files before executing tools
                parameters = suggestion.get("parameters", {})
                for param_name, param_value in parameters.items():
                    if isinstance(param_value, str) and param_name in ['file_path', 'path', 'filename']:
                        file_check = self._check_file_size_and_suggest_tools(param_value)
                        if file_check and file_check.get('severity') == 'critical':
                            tool_name = suggestion["tool_name"]
                            if tool_name not in ['read_fasta_file', 'assembly_stats', 'repeat_detection', 'gene_prediction_and_coding_stats']:
                                file_warnings.append(file_check)
                
                # Use execute_tool_suggestion for proper parameter chaining
                result = await self.execute_tool_suggestion(suggestion, tool_results)
                tool_results.append(result)
        
        # Create response
        if tool_results:
            # Format tool results
            response_content = f"I understood you want to: {analysis.get('intent', 'process your request')}\n\n"
            
            # Add file warnings if any
            if file_warnings:
                response_content += "⚠️  **Large File Warning**\n"
                for warning in file_warnings:
                    response_content += f"• {warning['message']}\n"
                    response_content += f"  File: {warning['file_path']} ({warning['file_size_mb']}MB)\n"
                    response_content += f"  Recommended tools: {', '.join(warning['recommended_tools'])}\n"
                response_content += "\n"
            for result in tool_results:
                response_content += f"Tool: {result['tool']}\n"
                if result['reason']:
                    response_content += f"Reason: {result['reason']}\n"
                if isinstance(result['result'], dict) and result['result'].get("error"):
                    response_content += f"Result: Error - {result['result']['error']}\n"
                else:
                    # Format result based on tool type
                    try:
                        tool_name = result.get('tool', 'unknown')
                        result_data = result.get('result', {})
                        formatted_result = self._format_tool_result_for_display(tool_name, result_data)
                        response_content += f"Result:\n{formatted_result}\n"
                    except Exception as e:
                        logger.error(f"Error formatting result: {e}")
                        logger.error(f"Tool: {result.get('tool', 'unknown')}")
                        logger.error(f"Result data type: {type(result.get('result'))}")
                        logger.error(f"Result data: {result.get('result')}")
                        # Fallback to simple formatting
                        response_content += f"Result: {result.get('result', 'No result data')}\n"
                response_content += "\n"
        elif analysis.get("response_type") == "direct_answer":
            response_content = analysis.get("direct_answer", "I understand your request.")
        elif analysis.get("needs_clarification"):
            response_content = "I need some clarification:\n"
            for question in analysis.get("clarification_questions", []):
                response_content += f"• {question}\n"
        else:
            # Fallback response
            response_content = "I'm not sure how to help with that. Please try rephrasing your request."
        
        # Save analysis results for future reference
        self._save_analysis_results(tool_results)
        
        # Add to history
        assistant_entry = {
            "role": "assistant",
            "content": response_content,
            "tool_calls": tool_results,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(assistant_entry)
        self._save_conversation_entry(assistant_entry)
        
        return response_content

    def _extract_text_content_recursively(self, data: Any) -> Any:
        """Recursively extract TextContent and CallToolResult objects from any data structure"""
        # Handle CallToolResult objects (from MCP)
        if hasattr(data, 'content') and hasattr(data, 'meta'):
            # This is a CallToolResult object
            content = data.content
            if isinstance(content, list) and len(content) > 0:
                # Extract from first content item
                first_content = content[0]
                return self._extract_text_content_recursively(first_content)
            else:
                return content
        elif hasattr(data, 'text'):
            # It's a TextContent object
            try:
                return json.loads(data.text)
            except (json.JSONDecodeError, TypeError):
                return data.text
        elif isinstance(data, dict):
            # Recursively process dictionary values
            return {key: self._extract_text_content_recursively(value) for key, value in data.items()}
        elif isinstance(data, list):
            # Recursively process list items
            return [self._extract_text_content_recursively(item) for item in data]
        else:
            # Return as-is for primitive types
            return data

    def _format_tool_result_for_display(self, tool_name: str, result_data: Dict[str, Any]) -> str:
        """Format tool results in a user-friendly way"""
        try:
            # First, recursively extract all TextContent objects
            cleaned_data = self._extract_text_content_recursively(result_data)
            
            # Handle nested result structure
            if isinstance(cleaned_data, dict) and 'result' in cleaned_data:
                actual_result = cleaned_data['result']
            else:
                actual_result = cleaned_data
            
            # Format based on tool type
            if tool_name == "tree_view":
                return self._format_tree_view_result(actual_result)
            elif tool_name == "find_files":
                return self._format_find_files_result(actual_result)
            elif tool_name in ["analyze_fasta_file", "calculate_sequence_stats"]:
                return self._format_sequence_analysis_result(actual_result)
            else:
                # Generic formatting for other tools
                return self._format_generic_result(actual_result)
        except Exception as e:
            # Since we have comprehensive TextContent extraction, this should rarely happen
            logger.warning(f"Error formatting result for {tool_name}: {e}")
            
            # Fallback to simple string representation of cleaned data
            try:
                cleaned_data = self._extract_text_content_recursively(result_data)
                if isinstance(cleaned_data, dict) and 'result' in cleaned_data:
                    return str(cleaned_data['result'])
                else:
                    return str(cleaned_data)
            except:
                return f"[Unable to format result: {type(result_data)}]"

    def _format_tree_view_result(self, result: Dict[str, Any]) -> str:
        """Format tree_view results nicely"""
        if isinstance(result, dict) and "tree_display" in result:
            tree_display = result["tree_display"]
            summary = result.get("summary", {})
            
            # More concise output
            formatted = f"{tree_display}\n"
            if summary.get('files', 0) > 0 or summary.get('directories', 0) > 0:
                formatted += f"\n📊 {summary.get('files', 0)} file(s), {summary.get('directories', 0)} folder(s)"
            
            return formatted
        else:
            return json.dumps(result, indent=2)

    def _format_find_files_result(self, result: Dict[str, Any]) -> str:
        """Format find_files results nicely"""
        if isinstance(result, dict) and "found_files" in result:
            found_files = result["found_files"]
            total_files = result.get("total_files", len(found_files))
            
            if total_files == 0:
                return f"🔍 No files found matching the criteria"
            
            formatted = f"🔍 Found {total_files} files:\n\n"
            for file_info in found_files[:20]:  # Show first 20
                name = file_info.get("name", "Unknown")
                size = file_info.get("size_formatted", "Unknown size")
                path = file_info.get("path", "Unknown path")
                formatted += f"  📄 {name} ({size})\n      {path}\n\n"
            
            if len(found_files) > 20:
                formatted += f"  ... and {len(found_files) - 20} more files\n"
            
            return formatted.rstrip()
        else:
            return json.dumps(result, indent=2)

    def _format_sequence_analysis_result(self, result: Dict[str, Any]) -> str:
        """Format sequence analysis results nicely"""
        if isinstance(result, dict):
            formatted = "🧬 Sequence Analysis Results:\n\n"
            
            # Basic stats
            if "sequence_count" in result:
                formatted += f"  Sequences: {result['sequence_count']}\n"
            if "total_length" in result:
                formatted += f"  Total length: {result['total_length']:,} bp\n"
            if "gc_content" in result:
                formatted += f"  GC content: {result['gc_content']:.2f}%\n"
            if "n50" in result:
                formatted += f"  N50: {result['n50']:,} bp\n"
            
            # Additional details if present
            if "longest_sequence" in result:
                formatted += f"  Longest sequence: {result['longest_sequence']:,} bp\n"
            if "shortest_sequence" in result:
                formatted += f"  Shortest sequence: {result['shortest_sequence']:,} bp\n"
            
            return formatted
        else:
            return json.dumps(result, indent=2)

    def _format_generic_result(self, result: Any) -> str:
        """Generic result formatting"""
        # First check if this is a TextContent object
        if hasattr(result, 'text'):
            try:
                parsed = json.loads(result.text)
                return self._format_generic_result(parsed)  # Recursive call with parsed content
            except:
                return str(result.text)
        
        if isinstance(result, list):
            if len(result) == 0:
                return "No results found"
            # Check if list contains TextContent objects
            if len(result) > 0 and hasattr(result[0], 'text'):
                # Extract text from all TextContent objects
                extracted = []
                for item in result:
                    if hasattr(item, 'text'):
                        extracted.append(item.text)
                    else:
                        extracted.append(str(item))
                return "\n".join([f"  • {item}" for item in extracted[:10]])
            elif len(result) <= 10:
                return "\n".join([f"  • {str(item)}" for item in result])
            else:
                formatted = "\n".join([f"  • {str(item)}" for item in result[:10]])
                formatted += f"\n  ... and {len(result) - 10} more items"
                return formatted
        elif isinstance(result, dict):
            # For small dicts, format nicely
            if len(result) <= 5:
                formatted = ""
                for key, value in result.items():
                    # Handle TextContent in dict values
                    if hasattr(value, 'text'):
                        formatted += f"  {key}: {value.text}\n"
                    else:
                        formatted += f"  {key}: {value}\n"
                return formatted.rstrip()
            else:
                return json.dumps(result, indent=2)
        else:
            return str(result)

    def _get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Get formatted tool descriptions for LLM context"""
        descriptions = []
        for tool_name, tool_info in self.discovered_tools.items():
            descriptions.append({
                "name": tool_name,
                "description": tool_info.get("description", ""),
                "parameters": tool_info.get("schema", {}),
                "server": tool_info.get("server_name", "")
            })
        return descriptions

    async def _get_llm_response_with_tools(self, user_message: str, system_prompt: str) -> Any:
        """Get LLM response with tool support"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Add recent conversation history for context
        for hist in self.conversation_history[-5:]:  # Last 5 messages
            messages.append({
                "role": hist["role"],
                "content": hist["content"]
            })
        
        # Define tools for LLM
        tools = []
        for tool_name, tool_info in self.discovered_tools.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info.get("description", ""),
                    "parameters": tool_info.get("schema", {})
                }
            })
        
        return await self.llm.generate(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            tools=tools if tools else None
        )

    async def cleanup(self):
        """Clean up resources"""
        if self.stdio_manager:
            await self.stdio_manager.disconnect_all()
        logger.info(f"Agent {self.agent_id} cleaned up")
    
    def _check_file_size_and_suggest_tools(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Check file size and suggest appropriate tools for large biological files"""
        try:
            if not os.path.exists(file_path):
                return None
                
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # Define size thresholds
            LARGE_FILE_THRESHOLD = 50  # MB
            VERY_LARGE_FILE_THRESHOLD = 500  # MB
            
            if file_size_mb < LARGE_FILE_THRESHOLD:
                return None  # File is small enough for normal processing
            
            # Check if it's a biological sequence file
            file_extension = Path(file_path).suffix.lower()
            is_bio_file = file_extension in ['.fasta', '.fa', '.fna', '.ffn', '.faa', '.fastq', '.fq']
            
            if not is_bio_file:
                # Check content for FASTA-like format
                try:
                    with open(file_path, 'r') as f:
                        first_lines = [f.readline().strip() for _ in range(5)]
                    is_bio_file = any(line.startswith('>') for line in first_lines)
                except:
                    pass
            
            if not is_bio_file:
                return None  # Not a biological file, proceed normally
            
            # Suggest appropriate tools for large biological files
            suggestions = {
                "large_file_detected": True,
                "file_size_mb": round(file_size_mb, 2),
                "file_path": file_path,
                "warning": f"Large biological file detected ({file_size_mb:.1f}MB)",
                "recommended_approach": "Use specialized analysis tools instead of reading entire file"
            }
            
            if file_size_mb > VERY_LARGE_FILE_THRESHOLD:
                suggestions["severity"] = "critical"
                suggestions["message"] = f"Very large file ({file_size_mb:.1f}MB) - direct reading may cause token limit errors"
                suggestions["recommended_tools"] = [
                    "assembly_stats(file_path) - for assembly statistics with smart sampling",
                    "read_fasta_file(file_path) - with automatic large file handling",
                    "gene_prediction_and_coding_stats(file_path) - for gene analysis with sampling"
                ]
            else:
                suggestions["severity"] = "warning"
                suggestions["message"] = f"Large file ({file_size_mb:.1f}MB) - consider using specialized tools"
                suggestions["recommended_tools"] = [
                    "assembly_stats(file_path) - comprehensive assembly analysis",
                    "repeat_detection(file_path) - repeat sequence analysis", 
                    "promoter_identification(file_path) - promoter motif search"
                ]
            
            return suggestions
            
        except Exception as e:
            logger.debug(f"Error checking file size for {file_path}: {e}")
            return None

    def _load_conversation_history(self):
        """Load conversation history from chat log file"""
        if not self.chat_log_file.exists():
            return
        
        try:
            with open(self.chat_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            self.conversation_history.append(entry)
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping malformed log entry: {line}")
            
            # Reduced verbosity - only show if there are many entries
            if len(self.conversation_history) > 10:
                logger.info(f"Loaded {len(self.conversation_history)} conversation entries from previous session")
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
    
    def _save_conversation_entry(self, entry: Dict[str, Any]):
        """Save a single conversation entry to the chat log file"""
        try:
            with open(self.chat_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to save conversation entry: {e}")
    
    def _save_analysis_results(self, tool_results: List[Dict[str, Any]]):
        """Save analysis results to persistent storage for future reference"""
        try:
            # Create analysis results directory
            results_dir = Path("reports") / "analysis_cache"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each significant analysis result
            for tool_result in tool_results:
                tool_name = tool_result.get("tool", "")
                result_data = tool_result.get("result", {})
                
                # Only save substantial analysis results, not simple file operations
                if tool_name in [
                    "analyze_fasta_file", "assembly_stats", "gene_prediction_and_coding_stats",
                    "promoter_identification", "repeat_detection", "kmer_analysis",
                    "read_analysis_results", "adaptive_code_generation", "execute_code"
                ]:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{tool_name}_{timestamp}.json"
                    filepath = results_dir / filename
                    
                    # Extract and save meaningful data
                    if isinstance(result_data, dict):
                        # Handle TextContent extraction if needed
                        clean_data = self._extract_text_content_recursively(result_data)
                        
                        with open(filepath, 'w') as f:
                            json.dump({
                                "tool": tool_name,
                                "timestamp": datetime.now().isoformat(),
                                "result": clean_data,
                                "session_id": self.agent_id
                            }, f, indent=2, default=str)
                        
                        logger.info(f"Saved analysis result: {filename}")
                        
                        # Also save to latest symlink for easy access
                        latest_link = results_dir / f"latest_{tool_name}.json"
                        if latest_link.exists():
                            latest_link.unlink()
                        latest_link.symlink_to(filename)
                            
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
    
    def _get_recent_context_summary(self, max_entries: int = 10) -> str:
        """Get a summary of recent conversation for context"""
        if not self.conversation_history:
            return ""
        
        # Limit to recent entries and truncate if too long
        recent_entries = self.conversation_history[-min(max_entries, 5):]  # Even more conservative
        context_lines = []
        
        for entry in recent_entries:
            timestamp = entry.get("timestamp", "")
            role = entry.get("role", "")
            content = entry.get("content", "")
            
            if role == "user":
                context_lines.append(f"User: {content}")
            elif role == "assistant":
                # Truncate long responses more aggressively
                truncated_content = content[:100] + "..." if len(content) > 100 else content
                context_lines.append(f"Assistant: {truncated_content}")
                
                # Include only essential tool results 
                tool_calls = entry.get("tool_calls", [])
                for tool_call in tool_calls[-2:]:  # Only last 2 tool calls
                    tool_name = tool_call.get("tool", "")
                    result = tool_call.get("result", {})
                    
                    # Extract key information from tool results
                    if isinstance(result, dict):
                        if "found_files" in result and result["found_files"]:
                            # File search results
                            file_info = result["found_files"][0]
                            context_lines.append(f"Found file: {file_info.get('name', '')} at {file_info.get('path', '')}")
                        elif "tree_display" in result:
                            # Tree view results  
                            context_lines.append(f"Directory listing completed")
                        elif "sequence_count" in result:
                            # Sequence analysis results
                            context_lines.append(f"Analyzed {result.get('sequence_count', 0)} sequences")
                        elif "gene_analysis" in result or "promoter_analysis" in result:
                            # Gene/promoter analysis results
                            context_lines.append(f"Gene/promoter analysis completed - results saved to analysis cache")
                        elif "assembly_stats" in result:
                            # Assembly analysis results
                            context_lines.append(f"Assembly analysis completed - results available in cache")
        
        context_text = "\n".join(context_lines[-10:])  # Even fewer context lines
        
        # Rough token count estimation (1 token ≈ 4 characters)
        estimated_tokens = len(context_text) // 4
        if estimated_tokens > 5000:  # Conservative limit for context
            # Truncate further if still too long
            context_text = context_text[:20000] + "...[truncated for token limit]"
        
        return context_text

    def _get_analysis_cache_info(self) -> str:
        """Get information about available cached analysis results"""
        try:
            results_dir = Path("reports") / "analysis_cache"
            if not results_dir.exists():
                return ""
            
            cache_info = []
            
            # Look for recent analysis files
            for json_file in results_dir.glob("*.json"):
                if json_file.name.startswith("latest_"):
                    continue  # Skip symlinks
                    
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    tool_name = data.get("tool", "")
                    timestamp = data.get("timestamp", "")
                    
                    # Extract key information about what's available
                    result = data.get("result", {})
                    if isinstance(result, dict):
                        if "gene_analysis" in result:
                            cache_info.append(f"• Gene analysis results available (genes, coding stats)")
                        elif "promoter_analysis" in result:
                            cache_info.append(f"• Promoter analysis results available (motifs, positions)")
                        elif "assembly_stats" in result:
                            cache_info.append(f"• Assembly statistics available (N50, GC content)")
                        elif "repeat_analysis" in result:
                            cache_info.append(f"• Repeat analysis results available")
                        elif tool_name == "analyze_fasta_file":
                            cache_info.append(f"• Complete genome analysis available")
                            
                except Exception:
                    continue
            
            if cache_info:
                return "\n".join(cache_info) + "\n• Use read_analysis_results('/path/to/latest/file.json') to access cached data"
            
            return ""
            
        except Exception as e:
            logger.debug(f"Error getting analysis cache info: {e}")
            return ""

    def _provide_biological_intelligence(self, user_request: str, cached_data: Dict[str, Any]) -> Optional[str]:
        """Provide intelligent biological analysis when available data matches user request"""
        try:
            # Detect what type of biological analysis is being requested
            user_lower = user_request.lower()
            
            # Handle gene-related queries first
            if any(keyword in user_lower for keyword in ['gene', 'genes', 'how many', 'count', 'number of', 'encode', 'predicted']):
                return self._handle_gene_queries(user_request, cached_data)
            
            # Map request patterns to data analysis
            if any(keyword in user_lower for keyword in ['tandem repeat', 'repeat', 'repetitive', 'percentage', 'percent', 'genome']):
                # User is asking about repeats - provide biological analysis
                if isinstance(cached_data, dict) and ('repeat_analysis' in cached_data or 'tandem_repeats' in str(cached_data)):
                    
                    # Extract repeat focus area from request
                    focus_area = None
                    if 'most frequent' in user_lower or 'frequently found' in user_lower:
                        focus_area = 'tandem_repeats'
                    elif 'percentage' in user_lower or 'percent' in user_lower:
                        focus_area = 'genome_statistics'
                    
                    # Use biological engine for deep analysis
                    bio_analysis = self.bio_engine.analyze_biological_data(cached_data, focus_area)
                    
                    # Format the analysis into a human-readable response
                    if 'biological_interpretation' in bio_analysis:
                        interpretation = bio_analysis['biological_interpretation']
                        if 'tandem_repeats' in interpretation:
                            repeat_analysis = interpretation['tandem_repeats']
                            return self._format_tandem_repeat_analysis(repeat_analysis)
                    
                    # Also check if we have direct repeat data
                    repeat_data = None
                    if 'repeat_analysis' in cached_data:
                        repeat_data = cached_data['repeat_analysis']
                    elif 'tandem_repeats' in cached_data:
                        repeat_data = cached_data
                    
                    if repeat_data:
                        # Check if this is specifically asking for percentage
                        if 'percentage' in user_lower or 'percent' in user_lower:
                            return self._calculate_repeat_percentage(repeat_data, cached_data)
                        else:
                            return self._format_tandem_repeat_analysis(repeat_data)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in biological intelligence: {e}")
            return None
    
    def _handle_gene_queries(self, user_request: str, cached_data: Dict[str, Any]) -> Optional[str]:
        """Handle gene-related queries using cached analysis data"""
        try:
            user_lower = user_request.lower()
            
            # Look for gene prediction data in cached results
            gene_data = None
            
            # Check different possible locations for gene data
            if isinstance(cached_data, dict):
                # Check for gene_analysis (modern format)
                if 'gene_analysis' in cached_data:
                    gene_data = cached_data['gene_analysis']
                # Direct gene prediction results (legacy format)
                elif 'gene_prediction_and_coding_stats' in cached_data:
                    gene_data = cached_data['gene_prediction_and_coding_stats']
                elif 'gene_prediction' in cached_data:
                    gene_data = cached_data['gene_prediction']
                # Check if the whole cached_data is gene prediction result
                elif 'predicted_genes' in cached_data:
                    gene_data = cached_data
                # Check nested structure
                elif any('gene' in str(key).lower() for key in cached_data.keys()):
                    for key, value in cached_data.items():
                        if 'gene' in str(key).lower() and isinstance(value, dict):
                            if 'predicted_genes' in value or 'summary' in value:
                                gene_data = value
                                break
            
            if not gene_data:
                logger.debug("No gene prediction data found in cached results")
                return None
            
            # Extract gene information from different formats
            gene_count = None
            avg_gene_length = None
            
            # Modern format: gene_analysis.summary
            if 'summary' in gene_data:
                summary = gene_data['summary']
                if 'total_genes' in summary:
                    gene_count = summary['total_genes']
                if 'mean_gene_length' in summary:
                    avg_gene_length = summary['mean_gene_length']
            
            # Legacy format: predicted_genes array
            elif 'predicted_genes' in gene_data:
                genes = gene_data['predicted_genes']
                gene_count = len(genes)
                
                # Calculate average gene length from individual genes
                if genes and len(genes) > 0:
                    gene_lengths = []
                    for gene in genes:
                        start = gene.get('start', 0)
                        end = gene.get('end', 0)
                        length = abs(end - start)
                        gene_lengths.append(length)
                    
                    if gene_lengths:
                        avg_gene_length = sum(gene_lengths) / len(gene_lengths)
            
            # Direct gene_count field
            elif 'gene_count' in gene_data:
                gene_count = gene_data['gene_count']
            
            # Handle different types of gene queries
            if gene_count is not None:
                if any(term in user_lower for term in ['how many', 'number of', 'count']):
                    if any(term in user_lower for term in ['gene', 'genes']):
                        return f"Based on the previous analysis, the genome encodes **{gene_count:,} genes**."
                
                elif 'encode' in user_lower:
                    return f"According to the gene prediction analysis, this genome encodes **{gene_count:,} genes**."
            
            if avg_gene_length is not None and any(term in user_lower for term in ['average', 'mean']) and 'length' in user_lower:
                return f"Based on the analysis of {gene_count:,} predicted genes, the **average gene length is {avg_gene_length:,.0f} base pairs**."
            
            return None
            
        except Exception as e:
            logger.debug(f"Error handling gene queries: {e}")
            return None
    
    async def _solve_with_reasoning_model(self, user_request: str, cached_data: Dict[str, Any] = None) -> Optional[str]:
        """Use reasoning model to analyze and solve ANY user request dynamically"""
        try:
            from .task_planner import TaskPlanner
            from .execution_models import TaskComplexity
            
            # Create a reasoning instance
            reasoning_llm = self.llm_interface  # Use the same LLM interface
            task_planner = TaskPlanner(reasoning_llm, self.prompt_manager)
            
            logger.info(f"🧠 Using reasoning model to solve: {user_request[:50]}...")
            
            # Step 1: Reason about the task
            reasoning_result = await task_planner.reason_about_task(user_request, self.discovered_tools)
            
            # Check if the reasoning model thinks this can be solved
            if reasoning_result.complexity_assessment == TaskComplexity.SIMPLE:
                # For simple tasks, try direct approach
                if cached_data and any(keyword in user_request.lower() for keyword in ['repeat', 'percentage', 'tandem', 'frequent']):
                    # Use biological intelligence for known biological patterns
                    return self._provide_biological_intelligence(user_request, cached_data)
            
            # Step 2: For complex tasks, create execution plan
            planning_context = {
                "user_request": user_request,
                "available_data": cached_data,
                "available_tools": self.discovered_tools,
                "reasoning_result": reasoning_result
            }
            
            execution_plan = await task_planner.create_execution_plan(reasoning_result, planning_context)
            
            # Step 3: Execute the plan iteratively
            final_result = await self._execute_reasoning_plan(execution_plan, cached_data)
            
            return final_result
            
        except ImportError:
            logger.debug("Enhanced reasoning components not available")
            return None
        except Exception as e:
            logger.debug(f"Error in reasoning model: {e}")
            return None
    
    async def _execute_reasoning_plan(self, execution_plan, available_data: Dict[str, Any] = None) -> Optional[str]:
        """Execute a reasoning-based plan with multiple steps and iterations"""
        try:
            results = []
            
            for step in execution_plan.steps:
                step_result = await self._execute_reasoning_step(step, available_data, results)
                results.append(step_result)
                
                # If step failed, try adaptive approach
                if not step_result.get("success", False):
                    adaptive_result = await self._try_adaptive_code_solution(step, available_data)
                    if adaptive_result:
                        results[-1] = adaptive_result
            
            # Combine results into final response
            return self._format_reasoning_results(execution_plan, results)
            
        except Exception as e:
            logger.error(f"Error executing reasoning plan: {e}")
            return None
    
    async def _execute_reasoning_step(self, step, available_data: Dict[str, Any], previous_results: List[Dict]) -> Dict[str, Any]:
        """Execute a single reasoning step"""
        try:
            # Check if this step involves tool usage
            if step.tool_name and step.tool_name in self.discovered_tools:
                # Execute the tool
                result = await self._call_mcp_tool(step.tool_name, step.parameters or {})
                return {
                    "step_type": "tool_execution",
                    "tool_name": step.tool_name, 
                    "result": result,
                    "success": "error" not in result
                }
            
            # Check if this step involves code generation
            elif "code" in step.description.lower() or "calculate" in step.description.lower():
                # Use adaptive code solver
                code_result = await self.adaptive_solver.solve_user_request(step.description, available_data)
                return {
                    "step_type": "code_generation",
                    "description": step.description,
                    "result": code_result,
                    "success": code_result.get("success", False)
                }
            
            # Default: analyze with available data
            else:
                analysis_result = self._analyze_step_with_data(step.description, available_data, previous_results)
                return {
                    "step_type": "data_analysis",
                    "description": step.description,
                    "result": analysis_result,
                    "success": analysis_result is not None
                }
                
        except Exception as e:
            return {
                "step_type": "error",
                "description": step.description,
                "error": str(e),
                "success": False
            }
    
    async def _try_adaptive_code_solution(self, step, available_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to solve a failed step using adaptive code generation"""
        try:
            code_result = await self.adaptive_solver.solve_user_request(step.description, available_data)
            if code_result.get("success"):
                return {
                    "step_type": "adaptive_code_recovery",
                    "description": step.description,
                    "result": code_result,
                    "success": True
                }
        except Exception as e:
            logger.debug(f"Adaptive code recovery failed: {e}")
        return None
    
    def _analyze_step_with_data(self, description: str, available_data: Dict[str, Any], previous_results: List[Dict]) -> Optional[str]:
        """Analyze a step using available data and previous results"""
        try:
            # Simple analysis based on available data
            if available_data and any(keyword in description.lower() for keyword in ['repeat', 'percentage', 'tandem']):
                return self._provide_biological_intelligence(description, available_data)
            
            # Combine previous results if relevant
            if previous_results:
                combined_info = f"Based on previous analysis: "
                for result in previous_results[-2:]:  # Use last 2 results
                    if result.get("success") and "result" in result:
                        combined_info += f"{result['result']}. "
                return combined_info
            
            return f"Analysis for: {description}"
            
        except Exception as e:
            logger.debug(f"Error in step analysis: {e}")
            return None
    
    def _format_reasoning_results(self, execution_plan, results: List[Dict[str, Any]]) -> str:
        """Format the reasoning plan results into a comprehensive response"""
        response_parts = []
        response_parts.append(f"## 🧠 Reasoning Model Solution\n")
        response_parts.append(f"**Task:** {execution_plan.goal}\n")
        
        successful_steps = [r for r in results if r.get("success", False)]
        response_parts.append(f"**Completed steps:** {len(successful_steps)}/{len(results)}\n")
        
        for i, result in enumerate(results, 1):
            if result.get("success", False):
                step_type = result.get("step_type", "unknown")
                if step_type == "tool_execution":
                    response_parts.append(f"**Step {i}:** Used {result['tool_name']} tool")
                elif step_type == "code_generation":
                    response_parts.append(f"**Step {i}:** Generated and executed custom code")
                    if "result" in result and "solution" in result["result"]:
                        exec_result = result["result"]["solution"].get("execution_result", {})
                        if exec_result.get("stdout"):
                            response_parts.append(f"```\n{exec_result['stdout'][:300]}...\n```")
                elif step_type == "data_analysis":
                    response_parts.append(f"**Step {i}:** {result['result']}")
        
        return "\n".join(response_parts)
    
    def _calculate_repeat_percentage(self, repeat_data: Dict[str, Any], full_data: Dict[str, Any] = None) -> str:
        """Calculate and return just the tandem repeat percentage of the genome"""
        if not repeat_data or 'tandem_repeats' not in repeat_data:
            return "No tandem repeat data available to calculate percentage"
        
        tandem_repeats = repeat_data['tandem_repeats']
        if not tandem_repeats:
            return "No tandem repeats found in the sequences"
        
        # Calculate genome percentage - try multiple sources for total length
        total_genome_length = repeat_data.get('total_length', 0)
        
        # Try to get total length from assembly stats if available
        if total_genome_length == 0 and full_data and 'assembly_stats' in full_data:
            total_genome_length = full_data['assembly_stats'].get('total_length', 0)
        
        if total_genome_length == 0:
            # Try alternative ways to get total length
            if 'per_sequence_stats' in repeat_data:
                total_genome_length = sum(stat.get('length', 0) for stat in repeat_data['per_sequence_stats'])
        
        # Also check if it's in summary
        if total_genome_length == 0 and 'summary' in repeat_data:
            total_genome_length = repeat_data['summary'].get('total_length', 0)
        
        total_repeat_bases = sum(repeat.get('total_length', 0) for repeat in tandem_repeats)
        
        if total_genome_length > 0:
            repeat_percentage = (total_repeat_bases / total_genome_length) * 100
            
            response = f"## 📊 Tandem Repeat Genome Coverage\n\n"
            response += f"**Total genome analyzed:** {total_genome_length:,} bp\n"
            response += f"**Total bases in tandem repeats:** {total_repeat_bases:,} bp\n"
            response += f"**Percentage of genome consisting of tandem repeats:** {repeat_percentage:.2f}%\n\n"
            
            # Add some context
            if repeat_percentage < 1:
                response += "*This is a relatively low percentage, suggesting the genome has few repetitive regions.*"
            elif repeat_percentage < 5:
                response += "*This is a moderate percentage, typical for many bacterial genomes.*"
            elif repeat_percentage < 15:
                response += "*This is a high percentage, indicating significant repetitive content.*"
            else:
                response += "*This is a very high percentage, suggesting extensive repetitive regions.*"
            
            return response
        else:
            return "Unable to calculate percentage - total genome length not available in the data"

    def _format_tandem_repeat_analysis(self, repeat_data: Dict[str, Any]) -> str:
        """Format tandem repeat data into a detailed analysis"""
        if not repeat_data or 'tandem_repeats' not in repeat_data:
            return "No tandem repeat data available"
        
        tandem_repeats = repeat_data['tandem_repeats']
        if not tandem_repeats:
            return "No tandem repeats found in the sequences"
        
        # Calculate genome percentage if total length is available
        total_genome_length = repeat_data.get('total_length', 0)
        
        # Try multiple sources for genome length (similar to _calculate_repeat_percentage)
        if total_genome_length == 0:
            # Try to find it in other places
            if 'per_sequence_stats' in repeat_data:
                total_genome_length = sum(stat.get('length', 0) for stat in repeat_data['per_sequence_stats'])
        
        # Also check if it's in summary
        if total_genome_length == 0 and 'summary' in repeat_data:
            total_genome_length = repeat_data['summary'].get('total_length', 0)
        
        total_repeat_bases = sum(repeat.get('total_length', 0) for repeat in tandem_repeats)
        
        if total_genome_length > 0:
            repeat_percentage = (total_repeat_bases / total_genome_length) * 100
            genome_stats = f"\n## 📊 Genome-wide Tandem Repeat Statistics\n"
            genome_stats += f"**Total genome length:** {total_genome_length:,} bp\n"
            genome_stats += f"**Total tandem repeat bases:** {total_repeat_bases:,} bp\n"
            genome_stats += f"**Percentage of genome in tandem repeats:** {repeat_percentage:.2f}%\n"
        else:
            genome_stats = ""
        
        # Collect statistics
        from collections import defaultdict
        repeat_unit_stats = defaultdict(lambda: {
            'count': 0,
            'total_copies': 0,
            'total_length': 0,
            'positions': [],
            'copy_numbers': []
        })
        
        for repeat in tandem_repeats:
            unit = repeat['repeat_unit']
            stats = repeat_unit_stats[unit]
            stats['count'] += 1
            stats['total_copies'] += repeat['copy_number']
            stats['total_length'] += repeat['total_length']
            stats['positions'].append(repeat['start'])
            stats['copy_numbers'].append(repeat['copy_number'])
        
        # Sort by frequency
        sorted_units = sorted(repeat_unit_stats.items(), 
                             key=lambda x: x[1]['count'], 
                             reverse=True)
        
        # Format results
        results = []
        
        # Add genome statistics first if available
        if genome_stats:
            results.append(genome_stats)
        
        results.append("## 📊 Most Frequently Found Tandem Repeats\n")
        results.append(f"**Total unique repeat units:** {len(repeat_unit_stats)}")
        results.append(f"**Total tandem repeat instances:** {len(tandem_repeats)}\n")
        
        results.append("### Top 10 Most Frequent Tandem Repeats:\n")
        results.append("| Repeat Unit | Occurrences | Avg Copy Number | Total Bases | Category |")
        results.append("|-------------|-------------|-----------------|-------------|----------|")
        
        for i, (unit, stats) in enumerate(sorted_units[:10]):
            avg_copies = stats['total_copies'] / stats['count']
            
            # Categorize by length
            unit_len = len(unit)
            if unit_len <= 2:
                category = "Dinucleotide"
            elif unit_len <= 3:
                category = "Trinucleotide"  
            elif unit_len <= 6:
                category = "Microsatellite"
            else:
                category = "Minisatellite"
            
            results.append(f"| {unit} | {stats['count']} | {avg_copies:.1f} | {stats['total_length']} | {category} |")
        
        # Additional analysis
        results.append("\n### Repeat Unit Composition Analysis:\n")
        
        # AT-rich vs GC-rich
        at_rich = []
        gc_rich = []
        palindromic = []
        
        for unit, stats in repeat_unit_stats.items():
            at_content = (unit.count('A') + unit.count('T')) / len(unit)
            
            if at_content >= 0.7:
                at_rich.append((unit, stats['count']))
            elif at_content <= 0.3:
                gc_rich.append((unit, stats['count']))
            
            # Check if palindromic
            if unit == unit[::-1]:
                palindromic.append((unit, stats['count']))
        
        if at_rich:
            results.append(f"**AT-rich repeats ({len(at_rich)} types):** " + 
                          ", ".join([f"{u} ({c}x)" for u, c in sorted(at_rich, key=lambda x: x[1], reverse=True)[:5]]))
        
        if gc_rich:
            results.append(f"**GC-rich repeats ({len(gc_rich)} types):** " + 
                          ", ".join([f"{u} ({c}x)" for u, c in sorted(gc_rich, key=lambda x: x[1], reverse=True)[:5]]))
        
        if palindromic:
            results.append(f"**Palindromic repeats ({len(palindromic)} types):** " + 
                          ", ".join([f"{u} ({c}x)" for u, c in sorted(palindromic, key=lambda x: x[1], reverse=True)[:5]]))
        
        return "\n".join(results)
    
    def _should_use_adaptive_solver(self, user_input: str) -> bool:
        """Determine if the adaptive code solver should be used for this request"""
        user_lower = user_input.lower()
        
        # Keywords that indicate need for custom code generation
        adaptive_indicators = [
            'detailed analysis', 'custom analysis', 'statistical analysis', 'specific analysis',
            'generate code', 'write script', 'calculate', 'compute', 'analyze the',
            'visualiz', 'plot', 'graph', 'chart', 'show distribution', 'frequency',
            'most frequent', 'top', 'detailed', 'specific', 'extract', 'identify the',
            'provide sequences', 'show me', 'find the', 'list the', 'count the',
            'compare', 'correlation', 'pattern', 'trend', 'summary statistics'
        ]
        
        # Biological analysis requests that need custom code
        biological_indicators = [
            'tandem repeat', 'promoter', 'gene length', 'coding density', 'gc content',
            'sequence composition', 'motif', 'palindromic', 'at-rich', 'gc-rich',
            'copy number', 'clustering', 'terminal repeat', 'regulatory element',
            'biological significance', 'genomic distribution', 'conservation',
            'structural element', 'functional annotation', 'repeat analysis',
            'gene analysis', 'assembly stats'
        ]
        
        # Requests that suggest iterative analysis
        iterative_indicators = [
            'detailed', 'comprehensive', 'in-depth', 'thorough', 'complete analysis',
            'break down', 'step by step', 'provide insights', 'explain the',
            'what does this mean', 'significance of', 'interpret'
        ]
        
        return (any(indicator in user_lower for indicator in adaptive_indicators) or
                any(indicator in user_lower for indicator in biological_indicators) or
                any(indicator in user_lower for indicator in iterative_indicators))
    
    def _detect_adaptive_analysis_need(self, user_input: str) -> bool:
        """Detect if user request needs adaptive code generation with biological intelligence"""
        # This method is now deprecated in favor of _should_use_adaptive_solver
        return self._should_use_adaptive_solver(user_input)

    async def process_natural_language(self, user_input: str) -> Dict[str, Any]:
        """Process natural language input using LLM to determine actions"""
        
        # Check if this needs adaptive analysis
        needs_adaptive = self._detect_adaptive_analysis_need(user_input)
        
        # Build context about available tools
        tools_context = self._build_tools_context()
        
        # Build conversation context to help find file paths and cached analysis
        conversation_context = self._build_conversation_context()
        analysis_cache_info = self._get_analysis_cache_info()
        
        if analysis_cache_info:
            conversation_context += f"\n\nAVAILABLE ANALYSIS CACHE:\n{analysis_cache_info}"
        
        # Enhanced prompt for adaptive analysis if needed
        if needs_adaptive:
            tools_context += "\n\n🧠 ADAPTIVE ANALYSIS TOOLS AVAILABLE:\n"
            tools_context += "- introspect_data_structure: Analyze any data structure intelligently\n"
            tools_context += "- adaptive_code_generation: Generate custom analysis code for any goal\n"
            tools_context += "- execute_code: Run generated code safely with visualizations\n"
            tools_context += "- read_analysis_results: Load previous analysis data\n"
            tools_context += "\nFor detailed analysis requests, consider using these adaptive tools in sequence."
        
        # Use prompt manager to get the tool selection prompt
        prompt = self.prompt_manager.format_prompt(
            "tool_selection",
            tools_context=tools_context,
            user_input=user_input,
            conversation_context=conversation_context
        )

        try:
            response = await self.llm.generate(prompt)
            logger.debug(f"LLM Response: {response}")
            
            # Check if response is None
            if response is None:
                logger.error("LLM returned None response")
                # Try to handle simple requests with fallback logic
                return self._handle_simple_request_fallback(user_input)
            
            # Clean the response - sometimes models add markdown formatting
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Raw response: {response}")
            # Fallback to simple query
            return {
                "intent": user_input,
                "suggested_tools": [],
                "needs_clarification": True,
                "clarification_questions": ["I had trouble understanding your request. Could you please rephrase it?"]
            }
        except Exception as e:
            logger.error(f"Unexpected error in process_natural_language: {e}")
            logger.error(f"Error type: {type(e)}")
            # Fallback for any other errors
            return {
                "intent": user_input,
                "suggested_tools": [],
                "needs_clarification": True,
                "clarification_questions": [f"I encountered an error processing your request. Please try again."]
            }
    
    def _build_tools_context(self) -> str:
        """Build a formatted string describing all available tools"""
        if not self.discovered_tools:
            return "No tools available."
        
        context_parts = []
        
        for category_info in self.tool_categories.values():
            context_parts.append(f"\n{category_info['name']} Server:")
            if category_info['description']:
                context_parts.append(f"  {category_info['description']}")
            
            for tool_name in category_info['tools']:
                tool_info = self.discovered_tools[tool_name]
                schema = tool_info.get('schema', {})
                required_params = schema.get('required', [])
                all_params = list(schema.get('properties', {}).keys())
                
                context_parts.append(f"  - {tool_name}: {tool_info['description']}")
                if all_params:
                    params_str = ", ".join([f"{p}{'*' if p in required_params else ''}" for p in all_params])
                    context_parts.append(f"    Parameters: {params_str} (* = required)")
        
        return "\n".join(context_parts)

    def _handle_simple_request_fallback(self, user_input: str) -> Dict[str, Any]:
        """Handle simple requests when LLM fails"""
        user_lower = user_input.lower()
        
        # Handle file listing requests
        if any(keyword in user_lower for keyword in ['list', 'files', 'directory', 'dir', 'ls']):
            # Determine path from the request
            repo_root = "/home/fschulz/dev/nelli-ai-scientist"
            
            if 'example' in user_lower:
                path = f"{repo_root}/example"
                return {
                    "intent": f"List files in example directory",
                    "response_type": "use_tools",
                    "suggested_tools": [
                        {
                            "tool_name": "tree_view",
                            "reason": "Display directory structure with file details",
                            "parameters": {"path": path}
                        }
                    ]
                }
            elif 'data' in user_lower:
                path = f"{repo_root}/data"
                return {
                    "intent": f"List files in data directory",
                    "response_type": "use_tools", 
                    "suggested_tools": [
                        {
                            "tool_name": "tree_view",
                            "reason": "Display directory structure with file details",
                            "parameters": {"path": path}
                        }
                    ]
                }
            else:
                # Default to project root
                return {
                    "intent": f"List files in project root",
                    "response_type": "use_tools",
                    "suggested_tools": [
                        {
                            "tool_name": "tree_view", 
                            "reason": "Display directory structure with file details",
                            "parameters": {"path": repo_root}
                        }
                    ]
                }
        
        # Handle help requests
        if any(keyword in user_lower for keyword in ['help', 'tools', 'commands']):
            return {
                "intent": "Show available tools and commands",
                "response_type": "direct_answer",
                "direct_answer": "Available commands: help, tools, clear, quit. You can also ask me to analyze files, generate code, or perform scientific analysis."
            }
        
        # Handle analytical requests with adaptive workflow
        analytical_keywords = [
            'stats', 'statistics', 'analysis', 'analyze', 'calculate', 'compute', 
            'assembly', 'genome', 'sequence', 'gene', 'genes', 'average', 'distribution',
            'quality', 'metrics', 'length', 'count', 'percentage', 'ratio', 'how many', 'encode'
        ]
        
        if any(keyword in user_lower for keyword in analytical_keywords):
            # Extract filename if mentioned
            import re
            filename_match = re.search(r'([a-zA-Z0-9_\-\.]+\.(?:fna|fasta|fa|fastq|fq))', user_input)
            
            repo_root = "/home/fschulz/dev/nelli-ai-scientist"
            
            if filename_match:
                filename = filename_match.group(1)
                # Check if we should load existing analysis results first
                if any(term in user_lower for term in ['average', 'mean', 'genes', 'length', 'predicted', 'analysis']):
                    # This request likely refers to previous analysis results
                    analysis_file = f"{repo_root}/reports/{filename.replace('.fna', '').replace('.fasta', '')}_analysis.json"
                    return {
                        "intent": f"Calculate statistics from previous analysis of {filename}",
                        "response_type": "use_tools",
                        "suggested_tools": [
                            {
                                "tool_name": "read_file",
                                "reason": "Load the previous analysis results",
                                "parameters": {"path": analysis_file}
                            },
                            {
                                "tool_name": "create_analysis_code",
                                "reason": f"Generate code to calculate: {user_input}",
                                "parameters": {
                                    "task_description": user_input,
                                    "data_context": "PREVIOUS_TOOL_RESULT"
                                }
                            },
                            {
                                "tool_name": "execute_code",
                                "reason": "Execute the generated code to get the answer",
                                "parameters": {
                                    "code": "PREVIOUS_TOOL_RESULT",
                                    "context_data": "ANALYSIS_DATA"
                                }
                            }
                        ]
                    }
                else:
                    # New analysis request
                    return {
                        "intent": f"Analyze {filename} using adaptive workflow",
                        "response_type": "use_tools",
                        "suggested_tools": [
                            {
                                "tool_name": "find_file_by_name",
                                "reason": "First locate the file since its full path is not known",
                                "parameters": {"filename": filename, "search_path": repo_root, "max_depth": 5}
                            },
                            {
                                "tool_name": "read_file",
                                "reason": "Read the file for analysis",
                                "parameters": {"path": "USE_PATH_FROM_PREVIOUS_RESULT"}
                            },
                            {
                                "tool_name": "create_analysis_code",
                                "reason": f"Generate code to handle the request: {user_input}",
                                "parameters": {
                                    "task_description": user_input,
                                    "data_context": "ANALYSIS_RESULTS"
                                }
                            },
                            {
                                "tool_name": "execute_code",
                                "reason": "Execute the generated code to get the answer",
                                "parameters": {
                                    "code": "CODE_FROM_PREVIOUS_STEP",
                                    "context_data": "ANALYSIS_RESULTS"
                                }
                            }
                        ]
                    }
            else:
                # Check if this refers to a previous analysis without mentioning filename
                if any(term in user_lower for term in ['genes', 'predicted', 'analysis', '221']):
                    return {
                        "intent": f"Calculate statistics from most recent analysis",
                        "response_type": "use_tools",
                        "suggested_tools": [
                            {
                                "tool_name": "read_file",
                                "reason": "Load the most recent analysis results",
                                "parameters": {"path": f"{repo_root}/reports/AC3300027503___Ga0255182_1000024_analysis.json"}
                            },
                            {
                                "tool_name": "create_analysis_code",
                                "reason": f"Generate code to calculate: {user_input}",
                                "parameters": {
                                    "task_description": user_input,
                                    "data_context": "PREVIOUS_TOOL_RESULT"
                                }
                            },
                            {
                                "tool_name": "execute_code",
                                "reason": "Execute the generated code to get the answer",
                                "parameters": {
                                    "code": "PREVIOUS_TOOL_RESULT",
                                    "context_data": "ANALYSIS_DATA"
                                }
                            }
                        ]
                    }
                else:
                    # General analytical request without specific file
                    return {
                        "intent": f"Handle analytical request: {user_input}",
                        "response_type": "use_tools",
                        "suggested_tools": [
                            {
                                "tool_name": "create_analysis_code",
                                "reason": f"Generate code to handle the request: {user_input}",
                                "parameters": {
                                    "task_description": user_input
                                }
                            },
                            {
                                "tool_name": "execute_code",
                                "reason": "Execute the generated code to get the answer",
                                "parameters": {
                                    "code": "CODE_FROM_PREVIOUS_STEP"
                                }
                            }
                        ]
                    }
        
        # Default fallback for non-analytical requests
        return {
            "intent": user_input,
            "response_type": "direct_answer",
            "direct_answer": "I can help you analyze files, generate code, list directories, or answer questions. What would you like me to do?"
        }

    def _build_conversation_context(self) -> str:
        """Build context from recent conversation history to help find file paths"""
        if not self.conversation_history:
            return ""
        
        # Get recent context summary
        recent_context = self._get_recent_context_summary()
        if recent_context:
            context = f"\nRecent conversation context:\n{recent_context}\n"
        else:
            context = ""
        
        # Look at recent conversation for file listings (keep existing logic for now)
        file_paths_found = []
        
        # Also check the formatted output in conversation content
        for entry in self.conversation_history[-5:]:  # Look at last 5 entries
            content = entry.get("content", "")
            
            # Parse tree structure from formatted output
            if "└──" in content or "├──" in content:
                lines = content.split('\n')
                current_dir = ""
                
                # Extract directory from tree_view context - look for directory name at start
                for line in lines:
                    if line.strip() and not line.strip().startswith('└──') and not line.strip().startswith('├──') and line.endswith('/'):
                        # This is a directory name line
                        current_dir = line.strip().rstrip('/')
                        break
                
                for line in lines:
                    # Extract file names and build paths from tree structure  
                    if '.fna' in line or '.fasta' in line or '.fa' in line or '.fq' in line:
                        # Extract filename
                        import re
                        match = re.search(r'([^\s├└─]+\.(fna|fasta|fa|fq|gz))', line)
                        if match:
                            filename = match.group(1)
                            # Build dynamic path using discovered directory
                            if current_dir:
                                file_paths_found.append({
                                    "name": filename,
                                    "path": f"{{repo_root}}/{current_dir}/{filename}"
                                })
                            else:
                                # Fallback - try to extract from filename itself
                                file_paths_found.append({
                                    "name": filename,
                                    "path": f"{{repo_root}}/{filename}"  # Assume root level
                                })
            
            # Parse formatted find_files output (e.g., "📄 filename.fna (size)\n    /full/path")
            elif "📄" in content and ("fna" in content or "fasta" in content or "fa" in content):
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "📄" in line and ('.fna' in line or '.fasta' in line or '.fa' in line or '.fq' in line):
                        # Extract filename from current line
                        import re
                        filename_match = re.search(r'📄\s+([^\s(]+\.(fna|fasta|fa|fq|gz))', line)
                        if filename_match:
                            filename = filename_match.group(1)
                            # Look for path in next line
                            if i + 1 < len(lines):
                                next_line = lines[i + 1].strip()
                                if next_line.startswith('/') and filename in next_line:
                                    # This is the full absolute path
                                    file_paths_found.append({
                                        "name": filename,
                                        "path": next_line
                                    })
            
            # Also check tool results if they exist
            if entry.get("role") == "assistant" and entry.get("tool_calls"):
                for tool_call in entry["tool_calls"]:
                    result = tool_call.get("result", {})
                    
                    # Handle wrapped results
                    if isinstance(result, dict) and "result" in result:
                        inner_result = result["result"]
                        # Try to extract from TextContent
                        if isinstance(inner_result, list) and len(inner_result) > 0:
                            try:
                                # Attempt to parse TextContent
                                if hasattr(inner_result[0], 'text'):
                                    parsed = json.loads(inner_result[0].text)
                                    result = parsed
                            except:
                                pass
                    
                    # Check for tree_view results
                    if isinstance(result, dict) and "tree_structure" in result:
                        for item in result.get("tree_structure", []):
                            if item.get("type") == "file":
                                file_paths_found.append({
                                    "name": item.get("name", ""),
                                    "path": item.get("path", "")
                                })
                    
                    # Check for find_files results
                    elif isinstance(result, dict) and "found_files" in result:
                        for file_info in result.get("found_files", []):
                            file_paths_found.append({
                                "name": file_info.get("name", ""),
                                "path": file_info.get("path", "")
                            })
        
        if file_paths_found:
            context += "\nRecent file listings from conversation history:\n"
            seen = set()  # Avoid duplicates
            for fp in file_paths_found:
                key = f"{fp['name']}:{fp['path']}"
                if key not in seen:
                    seen.add(key)
                    context += f"- {fp['name']}: {fp['path']}\n"
        
        return context

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        connected_servers = []
        if self.stdio_manager:
            connected_servers = list(self.stdio_manager.server_configs.keys())
            
        return {
            "agent_id": self.agent_id,
            "role": self.config.role,
            "connected_servers": len(connected_servers),
            "discovered_tools": len(self.discovered_tools),
            "conversation_length": len(self.conversation_history),
            "tool_categories": self.tool_categories
        }

    async def terminal_chat(self):
        """Interactive terminal chat interface - ported from original agent.py"""
        # Initialize first
        await self.initialize()
        
        # Display welcome message
        self._display_welcome()
        
        # Check if we're in an interactive terminal
        import sys
        if not sys.stdin.isatty():
            print("\n⚠️  Not running in interactive terminal. Exiting.")
            print("💡 To use the chat interface, run this directly in a terminal.")
            return
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() in ['help', 'h', '?']:
                    self._display_help()
                    continue
                
                if user_input.lower() in ['tools', 'list']:
                    self._display_tools()
                    continue
                
                if user_input.lower() == 'clear':
                    self._clear_screen()
                    self._display_welcome()
                    continue
                
                # Process natural language input
                print("\n🤔 Thinking...")
                analysis = await self.process_natural_language(user_input)
                
                # Handle clarification needs
                if analysis.get("needs_clarification"):
                    print("\n🤖 Assistant: I need some clarification:")
                    for question in analysis.get("clarification_questions", []):
                        print(f"   • {question}")
                    continue
                
                # Handle direct answers 
                if analysis.get("response_type") == "direct_answer":
                    direct_answer = analysis.get("direct_answer", "")
                    if direct_answer:
                        print(f"\n🤖 Assistant: {direct_answer}")
                    else:
                        # Fallback to generate response
                        response = await self._generate_response(user_input)
                        print(f"\n🤖 Assistant: {response}")
                
                # Execute suggested tools
                elif analysis.get("suggested_tools"):
                    print(f"\n🤖 Assistant: I understand you want to: {analysis['intent']}")
                    
                    tool_results = []
                    for i, suggestion in enumerate(analysis["suggested_tools"]):
                        print(f"\n🔧 Using tool: {suggestion['tool_name']}")
                        print(f"   Reason: {suggestion['reason']}")
                        
                        # Handle chained tools - pass previous results
                        result = await self.execute_tool_suggestion(suggestion, tool_results)
                        self._display_tool_result(result)
                        tool_results.append(result)
                    
                    # Reflect on the results
                    if tool_results:
                        print(f"\n🤔 Let me analyze these results...")
                        reflection = await self._reflect_on_tool_results(user_input, tool_results)
                        print(f"\n💡 Analysis: {reflection}")
                        
                else:
                    # No tools needed and no direct answer, just respond
                    response = await self._generate_response(user_input)
                    print(f"\n🤖 Assistant: {response}")
                
                # Store in history
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "user": user_input,
                    "analysis": analysis,
                    "type": "chat"
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.exception("Chat error")

    def _display_welcome(self):
        """Display welcome message - ported from original agent.py"""
        # ASCII art with colors
        print("\033[36m")  # Cyan color
        print("\n" + "="*70)
        print(r"  ███╗   ██╗███████╗██╗     ██╗     ██╗")
        print(r"  ████╗  ██║██╔════╝██║     ██║     ██║")
        print(r"  ██╔██╗ ██║█████╗  ██║     ██║     ██║")
        print(r"  ██║╚██╗██║██╔══╝  ██║     ██║     ██║")
        print(r"  ██║ ╚████║███████╗███████╗███████╗██║")
        print(r"  ╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝╚═╝")
        print("                🧪 AI Scientist Agent Template 🔬")
        print("\033[0m")  # Reset color
        print("="*70)
        print(f"\033[32mAgent:\033[0m {self.config.name}")
        print(f"\033[32mRole:\033[0m {self.config.description}")
        print(f"\033[32mID:\033[0m {self.agent_id}")
        
        if self.discovered_tools:
            print(f"\n\033[33m📊 Loaded {len(self.discovered_tools)} tools from {len(self.tool_categories)} servers\033[0m")
            
            # Show summary by server
            for category in self.tool_categories.values():
                print(f"  \033[36m•\033[0m {category['name']}: \033[33m{len(category['tools'])}\033[0m tools")
        else:
            print("\n\033[31m⚠️  No tools loaded. Check your MCP configuration.\033[0m")
        
        print("\n\033[35m💡 Commands:\033[0m help, tools, clear, quit")
        print("\033[35m💬 Or just type naturally to interact with available tools\033[0m")
        print("\033[36m" + "="*70 + "\033[0m")

    def _display_help(self):
        """Display help information - ported from original agent.py"""
        print("\n📚 Universal MCP Agent Help")
        print("="*50)
        print("\n🎯 Natural Language:")
        print("  Just type what you want to do, and I'll find the right tools!")
        print("\n⌨️  Commands:")
        print("  help, h, ?     - Show this help")
        print("  tools, list    - List all available tools")
        print("  clear          - Clear screen")
        print("  quit, exit, q  - Exit the chat")
        print("\n💡 Tips:")
        print("  • Describe what you want in plain English")
        print("  • I'll suggest appropriate tools and parameters")
        print("  • Ask for clarification if you're unsure")

    def _display_tools(self):
        """Display all available tools organized by server - ported from original agent.py"""
        print("\n🔧 Available MCP Tools")
        print("="*60)
        
        if not self.discovered_tools:
            print("No tools available. Check your MCP configuration.")
            return
        
        for category in self.tool_categories.values():
            print(f"\n📦 {category['name']}")
            if category['description']:
                print(f"   {category['description']}")
            print(f"   {'-'*50}")
            
            for tool_name in sorted(category['tools']):
                tool_info = self.discovered_tools[tool_name]
                print(f"   • {tool_name}")
                print(f"     {tool_info['description']}")
                
                # Show parameters
                schema = tool_info.get('schema', {})
                if schema.get('properties'):
                    params = []
                    required = schema.get('required', [])
                    for param, details in schema['properties'].items():
                        param_str = param
                        if param in required:
                            param_str += "*"
                        if 'type' in details:
                            param_str += f" ({details['type']})"
                        params.append(param_str)
                    print(f"     Parameters: {', '.join(params)}")

    def _display_tool_result(self, result: Dict[str, Any]):
        """Display tool execution result - ported from original agent.py"""
        result_data = result.get("result", {})
        
        # Check for errors
        if isinstance(result_data, dict) and "error" in result_data:
            print(f"\n❌ Error: {result_data['error']}")
            return
        
        print(f"\n✅ Result:")
        
        # Use the enhanced formatting if available
        tool_name = result.get("tool", "")
        if tool_name:
            try:
                formatted_output = self._format_tool_result_for_display(tool_name, result_data)
                print(formatted_output)
                return
            except Exception as e:
                logger.debug(f"Enhanced formatting failed: {e}")
        
        # Fallback to simple display
        if isinstance(result_data, dict):
            for key, value in result_data.items():
                # Handle TextContent objects
                if hasattr(value, 'text'):
                    print(f"  {key}: {value.text}")
                elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'text'):
                    print(f"  {key}:")
                    for item in value[:5]:  # Show first 5
                        if hasattr(item, 'text'):
                            print(f"    • {item.text}")
                        else:
                            print(f"    • {item}")
                    if len(value) > 5:
                        print(f"    ... and {len(value) - 5} more items")
                elif isinstance(value, (dict, list)):
                    print(f"  {key}:")
                    try:
                        print(f"    {json.dumps(value, indent=4)}")
                    except TypeError:
                        # Handle non-serializable objects
                        print(f"    {str(value)}")
                else:
                    print(f"  {key}: {value}")
        else:
            # Handle single TextContent
            if hasattr(result_data, 'text'):
                print(f"  {result_data.text}")
            else:
                print(f"  {result_data}")

    def _clear_screen(self):
        """Clear the terminal screen - ported from original agent.py"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')

    async def execute_tool_suggestion(self, suggestion: Dict[str, Any], previous_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a tool based on LLM suggestion - ported from original agent.py"""
        tool_name = suggestion.get("tool_name")
        parameters = suggestion.get("parameters", {})
        
        if not tool_name:
            return {"error": "No tool specified"}
        
        # Handle chained tool parameters - replace placeholders with actual data
        if previous_results:
            parameters = self._resolve_chained_parameters(parameters, previous_results)
            logger.debug(f"Resolved parameters for {tool_name}: {parameters}")
        
        # Validate and clean parameters based on tool schema
        cleaned_params = self._validate_parameters(tool_name, parameters)
        
        # Execute the tool
        result = await self._call_mcp_tool(tool_name, cleaned_params)
        
        return {
            "tool": tool_name,
            "parameters": cleaned_params,
            "result": result,
            "reason": suggestion.get("reason", "")
        }

    def _validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean parameters based on tool schema with smart file handling"""
        if tool_name not in self.discovered_tools:
            return parameters
        
        tool_info = self.discovered_tools[tool_name]
        schema = tool_info.get('schema', {})
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        cleaned = {}
        
        # Include all provided parameters that are in the schema
        for param_name, param_value in parameters.items():
            if param_name in properties:
                # Check for file paths that might be too large
                if isinstance(param_value, str) and param_name in ['file_path', 'path', 'filename']:
                    file_check = self._check_file_size_and_suggest_tools(param_value)
                    if file_check:
                        # Large biological file detected
                        severity = file_check.get('severity', 'warning')
                        if severity == 'critical' and tool_name not in ['read_fasta_file', 'assembly_stats', 'repeat_detection', 'gene_prediction_and_coding_stats']:
                            # For critical size files, suggest using specialized tools
                            logger.warning(f"Large file detected ({file_check['file_size_mb']}MB): {file_check['message']}")
                            logger.info(f"Recommended tools: {', '.join(file_check['recommended_tools'])}")
                            
                            # Add a warning parameter that tools can use
                            cleaned['_large_file_warning'] = file_check
                        elif severity == 'warning':
                            logger.info(f"Large file detected ({file_check['file_size_mb']}MB): Consider using specialized analysis tools")
                            cleaned['_large_file_info'] = file_check
                
                cleaned[param_name] = param_value
        
        # Check for missing required parameters
        missing = [r for r in required if r not in cleaned]
        if missing:
            logger.warning(f"Missing required parameters for {tool_name}: {missing}")
        
        return cleaned
    
    def _resolve_chained_parameters(self, parameters: Dict[str, Any], previous_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve parameter placeholders with data from previous tool executions - ported from original agent.py"""
        resolved_params = {}
        
        for param_name, param_value in parameters.items():
            if param_value == "DATA_FROM_PREVIOUS_TOOL" and previous_results:
                # Use the result data from the most recent tool execution (for adaptive analysis)
                last_result = previous_results[-1]
                if "result" in last_result and last_result["result"]:
                    result_data = last_result["result"]
                    
                    # Extract the actual data (for biocoding tools)
                    if isinstance(result_data, dict):
                        # For adaptive_code_generation, we want the actual data that was analyzed
                        if "data" in result_data:
                            resolved_params[param_name] = result_data["data"]
                        elif "analysis_data" in result_data:
                            resolved_params[param_name] = result_data["analysis_data"]
                        else:
                            resolved_params[param_name] = result_data
                    else:
                        resolved_params[param_name] = result_data
                else:
                    resolved_params[param_name] = {}
            elif param_value == "ANALYSIS_RESULTS" and previous_results:
                # Use the result data from the most recent tool execution
                last_result = previous_results[-1]
                if "result" in last_result and last_result["result"]:
                    result_data = last_result["result"]
                    
                    # Extract data from TextContent if needed
                    if isinstance(result_data, dict) and 'result' in result_data:
                        inner_result = result_data['result']
                        if isinstance(inner_result, list) and len(inner_result) > 0:
                            first_item = inner_result[0]
                            if hasattr(first_item, 'text'):
                                try:
                                    # Parse the JSON from TextContent
                                    resolved_params[param_name] = json.loads(first_item.text)
                                    continue
                                except:
                                    pass
                    
                    # If not TextContent or parsing failed, use as-is
                    resolved_params[param_name] = result_data
                else:
                    resolved_params[param_name] = last_result
            elif param_value == "USE_PATH_FROM_PREVIOUS_RESULT" and previous_results:
                # Extract file path from previous find_file_by_name or similar results
                for result in reversed(previous_results):  # Check most recent first
                    result_data = result.get("result", {})
                    
                    # Handle TextContent extraction first
                    if isinstance(result_data, dict) and 'result' in result_data:
                        inner_result = result_data['result']
                        if isinstance(inner_result, list) and len(inner_result) > 0:
                            first_item = inner_result[0]
                            if hasattr(first_item, 'text'):
                                try:
                                    result_data = json.loads(first_item.text)
                                except:
                                    pass
                    
                    # Look for file path in various result formats
                    if isinstance(result_data, dict):
                        # Check for find_file_by_name format
                        if "found_files" in result_data and result_data["found_files"]:
                            file_path = result_data["found_files"][0].get("path")
                            if file_path:
                                resolved_params[param_name] = file_path
                                logger.info(f"Resolved {param_name} to path: {file_path}")
                                break
                        # Check for direct path field
                        elif "path" in result_data:
                            resolved_params[param_name] = result_data["path"]
                            logger.info(f"Resolved {param_name} to path: {result_data['path']}")
                            break
                        # Check for file_path field
                        elif "file_path" in result_data:
                            resolved_params[param_name] = result_data["file_path"]
                            logger.info(f"Resolved {param_name} to path: {result_data['file_path']}")
                            break
                
                # If no path found, keep original value but log warning
                if param_name not in resolved_params:
                    logger.warning(f"Could not resolve path from previous results for {param_name}")
                    resolved_params[param_name] = param_value
            else:
                resolved_params[param_name] = param_value
        
        return resolved_params

    async def _generate_response(self, user_input: str) -> str:
        """Generate a response when no tools are needed - ported from original agent.py"""
        # Use prompt manager for general responses
        prompt = self.prompt_manager.format_prompt(
            "general_response",
            user_input=user_input,
            agent_role=self.config.role
        )

        try:
            return await self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I understand your request, but I'm having trouble generating a response. Could you try rephrasing?"

    async def _reflect_on_tool_results(self, user_request: str, tool_results: List[Dict[str, Any]]) -> str:
        """Reflect on and interpret tool execution results - ported from original agent.py"""
        
        # Build context about what tools were used and their results
        results_context = []
        for result in tool_results:
            tool_name = result.get('tool', 'unknown')
            tool_result = result.get('result', {})
            
            if isinstance(tool_result, dict) and 'error' not in tool_result:
                try:
                    results_context.append(f"Tool '{tool_name}' executed successfully with result: {json.dumps(tool_result, indent=2)}")
                except TypeError:
                    # Handle non-serializable objects
                    results_context.append(f"Tool '{tool_name}' executed successfully with result: {str(tool_result)}")
            elif isinstance(tool_result, dict) and 'error' in tool_result:
                results_context.append(f"Tool '{tool_name}' encountered an error: {tool_result['error']}")
            else:
                results_context.append(f"Tool '{tool_name}' returned: {str(tool_result)}")
        
        results_summary = "\n".join(results_context)
        
        # Use prompt manager for reflection
        prompt = self.prompt_manager.format_prompt(
            "reflection",
            user_request=user_request,
            results_summary=results_summary
        )

        try:
            return await self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Failed to generate reflection: {e}")
            return "I've executed the requested tools, but I'm having trouble interpreting the results. Please see the raw output above."