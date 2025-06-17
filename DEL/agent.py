"""
Universal MCP Agent - Domain-agnostic AI agent that works with any MCP server
Dynamically discovers and uses tools from any configured MCP servers
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
import logging
from pathlib import Path

from pydantic import BaseModel
from .llm_interface import LLMInterface, LLMProvider
from .communication import FIPAMessage, Performative
from .prompt_manager import PromptManager
from .error_handler import ErrorHandler, ParameterResolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Agent configuration model"""
    name: str
    role: str = "universal_assistant"
    description: str = "A universal agent that can work with any MCP tools"
    llm_provider: LLMProvider = LLMProvider.CBORG
    temperature: float = 0.7
    max_tokens: int = 4096
    mcp_config_path: str = "mcp_config.json"


class UniversalMCPAgent:
    """
    Universal AI Agent that discovers and uses tools from any MCP server
    Completely domain-agnostic - works with any tools exposed via MCP
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMInterface(provider=config.llm_provider)
        self.conversation_history: List[Dict[str, Any]] = []
        self.agent_id = f"{config.name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # MCP servers and tools will be discovered dynamically
        self.mcp_servers = {}
        self.discovered_tools = {}
        self.tool_categories = {}  # Group tools by server/category
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
        
        # Initialize error handler
        self.error_handler = ErrorHandler()
        
        logger.info(f"Initialized Universal MCP Agent: {self.agent_id}")

    async def initialize(self):
        """Initialize the agent by discovering all available MCP servers and tools"""
        logger.info("🔍 Initializing Universal MCP Agent...")
        
        # Load MCP configuration
        self._load_mcp_config()
        
        # Discover all available tools
        if self.mcp_servers:
            self.discovered_tools = await self._discover_all_tools()
            logger.info(f"✅ Discovered {len(self.discovered_tools)} tools from {len(self.mcp_servers)} servers")
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
                logger.info(f"  • Server {server_id}: enabled={enabled}")
                if enabled:
                    self.mcp_servers[server_id] = server_config
                    logger.info(f"    ✓ Loaded MCP server: {server_config.get('name', server_id)}")
                    
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")

    async def _discover_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Discover all tools from all configured MCP servers"""
        all_tools = {}
        
        for server_id, server_config in self.mcp_servers.items():
            server_name = server_config.get("name", server_id)
            logger.info(f"\n🔧 Discovering tools from: {server_name}")
            
            try:
                tools = await self._discover_server_tools(server_id, server_config)
                
                # Organize tools by category
                if server_id not in self.tool_categories:
                    self.tool_categories[server_id] = {
                        "name": server_name,
                        "description": server_config.get("description", ""),
                        "tools": []
                    }
                
                for tool_name, tool_info in tools.items():
                    all_tools[tool_name] = tool_info
                    self.tool_categories[server_id]["tools"].append(tool_name)
                    
            except Exception as e:
                logger.error(f"  ✗ Failed to discover tools from {server_name}: {e}")
        
        return all_tools

    async def _discover_server_tools(self, server_id: str, server_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Discover tools from a specific MCP server using FastMCP"""
        server_tools = {}
        
        try:
            from fastmcp import Client
            logger.debug(f"  Using FastMCP client for {server_id}")
            
            # Build the connection string/command for FastMCP
            connection_params = None
            
            # For FastMCP, we need to provide the script path directly
            connection_params = self._get_fastmcp_script_path(server_id, server_config)
            if not connection_params:
                logger.warning(f"Cannot determine FastMCP script path for {server_id}")
                return server_tools
            
            logger.debug(f"  Connection params: {connection_params}")
            
            # Connect using FastMCP client
            async with Client(connection_params) as client:
                logger.debug(f"  Connected to {server_id}")
                
                # List available tools
                tools = await client.list_tools()
                logger.debug(f"  Found {len(tools)} tools for {server_id}")
                
                for tool in tools:
                    # FastMCP tools have different structure
                    tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                    tool_desc = tool.description if hasattr(tool, 'description') else ""
                    tool_schema = tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                    
                    server_tools[tool_name] = {
                        "server_id": server_id,
                        "server_name": server_config.get("name", server_id),
                        "description": tool_desc,
                        "schema": tool_schema,
                        "server_config": server_config
                    }
                
                logger.info(f"  ✅ Loaded {len(tools)} tools: {', '.join(server_tools.keys())}")
                
        except ImportError:
            logger.warning(f"FastMCP client not available. Cannot discover tools from {server_id}")
        except Exception as e:
            logger.error(f"Failed to discover tools from {server_id}: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return server_tools

    def _get_fastmcp_script_path(self, server_id: str, server_config: Dict[str, Any]) -> str:
        """Determine the FastMCP script path for a server configuration"""
        import os
        from pathlib import Path
        
        # Check for fastmcp_script in server config
        if "fastmcp_script" in server_config:
            script_path = server_config["fastmcp_script"]
            if os.path.exists(script_path):
                logger.info(f"    Using configured FastMCP script: {script_path}")
                return script_path
        
        # Check if command points to a python script
        args = server_config.get("args", [])
        if server_config.get("command") == "pixi" and "python" in args:
            # Look for python script in args
            for i, arg in enumerate(args):
                if arg.endswith(".py"):
                    script_path = arg
                    # Make path absolute if relative
                    if not os.path.isabs(script_path) and server_config.get("cwd"):
                        script_path = os.path.join(server_config["cwd"], script_path)
                    
                    if os.path.exists(script_path):
                        logger.info(f"    Found Python script: {script_path}")
                        return script_path
                
                # Check for module path (-m module.path)
                if arg == "-m" and i + 1 < len(args):
                    module_path = args[i + 1]
                    # Convert module path to file path
                    script_path = module_path.replace(".", "/") + ".py"
                    if server_config.get("cwd"):
                        script_path = os.path.join(server_config["cwd"], script_path)
                    
                    # Check for FastMCP version first
                    fastmcp_path = script_path.replace(".py", "_fastmcp.py")
                    if os.path.exists(fastmcp_path):
                        logger.info(f"    Using FastMCP version: {fastmcp_path}")
                        return fastmcp_path
                    elif os.path.exists(script_path):
                        logger.info(f"    Using original script: {script_path}")
                        return script_path
        
        # Check if it's a direct script path
        elif server_config.get("command", "").endswith(".py"):
            script_path = server_config["command"]
            if os.path.exists(script_path):
                logger.info(f"    Direct script path: {script_path}")
                return script_path
        
        logger.warning(f"    Could not determine script path for {server_id}")
        return None

    async def _call_mcp_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool on the appropriate MCP server using FastMCP"""
        if tool_name not in self.discovered_tools:
            return {"error": f"Tool '{tool_name}' not found. Available tools: {list(self.discovered_tools.keys())}"}
        
        tool_info = self.discovered_tools[tool_name]
        server_config = tool_info["server_config"]
        server_id = tool_info["server_id"]
        
        logger.info(f"🔄 Calling tool '{tool_name}' on server '{tool_info['server_name']}'")
        logger.debug(f"   Original parameters: {kwargs}")
        
        # Validate and fix parameters using error handler
        tool_schema = tool_info.get("schema", {})
        validated_params = self.error_handler.validate_tool_parameters(tool_name, kwargs, tool_schema)
        logger.debug(f"   Validated parameters: {validated_params}")
        
        try:
            from fastmcp import Client
            
            # Build connection params (same as in discovery)
            connection_params = self._get_fastmcp_script_path(server_id, server_config)
            if not connection_params:
                logger.error(f"Cannot determine FastMCP script path for {server_id}")
                return {"error": "FastMCP script path not found"}
            
            logger.info(f"    Using configured FastMCP script: {connection_params}")
            
            # Connect and call the tool
            async with Client(connection_params) as client:
                result = await client.call_tool(tool_name, validated_params)
                
                # FastMCP returns result differently
                if hasattr(result, 'text'):
                    # If it has a text attribute, use that
                    try:
                        return json.loads(result.text)
                    except (json.JSONDecodeError, AttributeError):
                        return {"result": result.text}
                elif isinstance(result, list) and len(result) > 0:
                    # Handle list of TextContent objects
                    first_result = result[0]
                    if hasattr(first_result, 'text'):
                        try:
                            return json.loads(first_result.text)
                        except (json.JSONDecodeError, AttributeError):
                            return {"result": first_result.text}
                    else:
                        return {"result": str(first_result)}
                elif isinstance(result, dict):
                    return result
                else:
                    return {"result": str(result)}
                    
        except ImportError:
            return {"error": "FastMCP client not available"}
        except Exception as e:
            # Create error report with context
            error_context = {
                "tool_name": tool_name,
                "server_id": server_id,
                "parameters": validated_params
            }
            error_report = self.error_handler.create_error_report(e, error_context)
            
            logger.error(f"Tool call failed: {error_report['user_message']}")
            logger.debug(f"Technical details: {error_report['technical_details']}")
            
            return {"error": error_report['user_message']}


    async def process_natural_language(self, user_input: str) -> Dict[str, Any]:
        """Process natural language input using LLM to determine actions"""
        
        # Build context about available tools
        tools_context = self._build_tools_context()
        
        # Use prompt manager to get the tool selection prompt
        prompt = self.prompt_manager.format_prompt(
            "tool_selection",
            tools_context=tools_context,
            user_input=user_input
        )

        try:
            response = await self.llm.generate(prompt)
            logger.debug(f"LLM Response: {response}")
            
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

    async def execute_tool_suggestion(self, suggestion: Dict[str, Any], previous_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a tool based on LLM suggestion"""
        tool_name = suggestion.get("tool_name")
        parameters = suggestion.get("parameters", {})
        
        if not tool_name:
            return {"error": "No tool specified"}
        
        # Handle chained tool parameters - replace placeholders with actual data
        if previous_results:
            parameters = self._resolve_chained_parameters(parameters, previous_results)
        
        # Validate and clean parameters based on tool schema
        cleaned_params = self._validate_parameters(tool_name, parameters)
        
        # Execute the tool
        result = await self._call_mcp_tool(tool_name, **cleaned_params)
        
        return {
            "tool": tool_name,
            "parameters": cleaned_params,
            "result": result,
            "reason": suggestion.get("reason", "")
        }

    def _validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean parameters based on tool schema"""
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
                cleaned[param_name] = param_value
        
        # Check for missing required parameters
        missing = [r for r in required if r not in cleaned]
        if missing:
            logger.warning(f"Missing required parameters for {tool_name}: {missing}")
        
        return cleaned
    
    def _resolve_chained_parameters(self, parameters: Dict[str, Any], previous_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve parameter placeholders with data from previous tool executions"""
        resolved_params = {}
        
        for param_name, param_value in parameters.items():
            if param_value == "ANALYSIS_RESULTS" and previous_results:
                # Use the result data from the most recent tool execution
                last_result = previous_results[-1]
                if "result" in last_result and last_result["result"]:
                    resolved_params[param_name] = last_result["result"]
                else:
                    resolved_params[param_name] = last_result
            else:
                resolved_params[param_name] = param_value
        
        return resolved_params

    async def terminal_chat(self):
        """Interactive terminal chat interface"""
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
        """Display welcome message"""
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
            print(f"\n\033[33m📊 Loaded {len(self.discovered_tools)} tools from {len(self.mcp_servers)} servers\033[0m")
            
            # Show summary by server
            for category in self.tool_categories.values():
                print(f"  \033[36m•\033[0m {category['name']}: \033[33m{len(category['tools'])}\033[0m tools")
        else:
            print("\n\033[31m⚠️  No tools loaded. Check your MCP configuration.\033[0m")
        
        print("\n\033[35m💡 Commands:\033[0m help, tools, clear, quit")
        print("\033[35m💬 Or just type naturally to interact with available tools\033[0m")
        print("\033[36m" + "="*70 + "\033[0m")

    def _display_help(self):
        """Display help information"""
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
        """Display all available tools organized by server"""
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
        """Display tool execution result"""
        if "error" in result.get("result", {}):
            print(f"\n❌ Error: {result['result']['error']}")
            return
        
        print(f"\n✅ Result:")
        
        # Pretty print the result
        result_data = result.get("result", {})
        if isinstance(result_data, dict):
            for key, value in result_data.items():
                if isinstance(value, (dict, list)):
                    print(f"  {key}:")
                    print(f"    {json.dumps(value, indent=4)}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {result_data}")

    async def _generate_response(self, user_input: str) -> str:
        """Generate a response when no tools are needed"""
        # Use prompt manager for general responses
        prompt = self.prompt_manager.format_prompt(
            "general_response",
            user_input=user_input
        )

        try:
            return await self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I understand your request, but I'm having trouble generating a response. Could you try rephrasing?"

    async def _reflect_on_tool_results(self, user_request: str, tool_results: List[Dict[str, Any]]) -> str:
        """Reflect on and interpret tool execution results"""
        
        # Build context about what tools were used and their results
        results_context = []
        for result in tool_results:
            tool_name = result.get('tool', 'unknown')
            tool_result = result.get('result', {})
            
            if isinstance(tool_result, dict) and 'error' not in tool_result:
                results_context.append(f"Tool '{tool_name}' executed successfully with result: {json.dumps(tool_result, indent=2)}")
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

    def _clear_screen(self):
        """Clear the terminal screen"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')

    # FIPA-ACL compatibility methods (for integration with other agents)
    
    async def process_message(self, message: FIPAMessage) -> FIPAMessage:
        """Process FIPA-ACL messages for agent integration"""
        logger.info(f"Processing FIPA message: {message.performative}")
        
        try:
            if message.performative == Performative.REQUEST:
                result = await self._handle_request(message.content)
            elif message.performative == Performative.QUERY:
                result = await self._handle_query(message.content)
            else:
                result = {"error": f"Unsupported performative: {message.performative}"}
            
            return FIPAMessage(
                performative=Performative.INFORM,
                sender=self.agent_id,
                receiver=message.sender,
                content=result,
                conversation_id=message.conversation_id,
                in_reply_to=message.reply_with
            )
            
        except Exception as e:
            logger.error(f"Error processing FIPA message: {e}")
            return FIPAMessage(
                performative=Performative.FAILURE,
                sender=self.agent_id,
                receiver=message.sender,
                content={"error": str(e)},
                conversation_id=message.conversation_id,
                in_reply_to=message.reply_with
            )

    async def _handle_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle FIPA REQUEST messages"""
        # Natural language request
        if "query" in content:
            analysis = await self.process_natural_language(content["query"])
            results = []
            
            for suggestion in analysis.get("suggested_tools", []):
                result = await self.execute_tool_suggestion(suggestion)
                results.append(result)
            
            # Add reflection if tools were used
            response = {"analysis": analysis, "results": results}
            if results:
                reflection = await self._reflect_on_tool_results(content["query"], results)
                response["reflection"] = reflection
            
            return response
        
        # Direct tool call
        elif "tool" in content:
            tool_name = content["tool"]
            parameters = content.get("parameters", {})
            result = await self._call_mcp_tool(tool_name, **parameters)
            return {"tool": tool_name, "result": result}
        
        else:
            return {"error": "Request must include 'query' or 'tool'"}

    async def _handle_query(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle FIPA QUERY messages"""
        query_type = content.get("type", "")
        
        if query_type == "capabilities":
            return {
                "agent_id": self.agent_id,
                "name": self.config.name,
                "role": self.config.role,
                "description": self.config.description,
                "mcp_servers": len(self.mcp_servers),
                "total_tools": len(self.discovered_tools),
                "tools_by_server": {
                    cat["name"]: len(cat["tools"]) 
                    for cat in self.tool_categories.values()
                }
            }
        
        elif query_type == "tools":
            return {
                "tools": {
                    name: {
                        "description": info["description"],
                        "server": info["server_name"],
                        "schema": info["schema"]
                    }
                    for name, info in self.discovered_tools.items()
                }
            }
        
        else:
            return {"error": f"Unknown query type: {query_type}"}


# Main execution
if __name__ == "__main__":
    import argparse
    
    async def main():
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Universal MCP Agent")
        parser.add_argument(
            "--config", 
            default="mcp_config.json",
            help="Path to MCP configuration file (default: mcp_config.json)"
        )
        parser.add_argument(
            "--name",
            default="nelli-agent-template", 
            help="Agent name (default: nelli-agent-template)"
        )
        parser.add_argument(
            "--llm-provider",
            default="cborg",
            choices=["cborg", "claude", "openai"],
            help="LLM provider to use (default: cborg)"
        )
        args = parser.parse_args()
        
        # Create configuration
        config = AgentConfig(
            name=args.name,
            description="NeLLi AI Scientist Agent Template - Universal MCP Assistant",
            llm_provider=LLMProvider(args.llm_provider.lower()),
            mcp_config_path=args.config
        )
        
        # Create and run agent
        agent = UniversalMCPAgent(config)
        
        # Start terminal chat
        await agent.terminal_chat()

    asyncio.run(main())