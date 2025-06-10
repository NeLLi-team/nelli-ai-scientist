"""
Tool Registry for Agent Capabilities
"""

from typing import Dict, Any, Callable, List, Optional
import inspect
import asyncio
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class Tool:
    """Represents a single tool/capability"""

    def __init__(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.func = func
        self.description = description or func.__doc__ or "No description"
        self.parameters = parameters or self._extract_parameters()

    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract parameters from function signature"""
        sig = inspect.signature(self.func)
        params = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            param_info = {
                "type": (
                    str(param.annotation) if param.annotation != param.empty else "Any"
                ),
                "required": param.default == param.empty,
            }

            if param.default != param.empty:
                param_info["default"] = param.default

            params[name] = param_info

        return params

    async def execute(self, **kwargs) -> Any:
        """Execute the tool"""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        else:
            return self.func(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registry for agent tools and capabilities"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, name: Optional[str] = None, description: Optional[str] = None):
        """Decorator to register a tool"""

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool = Tool(tool_name, func, description)
            self.tools[tool_name] = tool

            @wraps(func)
            async def wrapper(**kwargs):
                logger.info(f"Executing tool: {tool_name}")
                try:
                    result = await tool.execute(**kwargs)
                    logger.info(f"Tool {tool_name} completed successfully")
                    return result
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    raise

            return wrapper

        return decorator

    def add_tool(self, tool: Tool):
        """Add a tool directly"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all available tools"""
        return {name: tool.to_dict() for name, tool in self.tools.items()}

    async def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        return await tool.execute(**kwargs)

    def create_mcp_connector(self, mcp_endpoint: str):
        """Create a connector for MCP tools"""

        @self.register(f"mcp_{mcp_endpoint}", f"Connect to MCP server: {mcp_endpoint}")
        async def mcp_connector(tool: str, params: Dict[str, Any]) -> Any:
            """Execute tool on MCP server"""
            # This would implement actual MCP protocol
            # For now, it's a placeholder
            import aiohttp

            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": tool, "arguments": params},
                    "id": 1,
                }

                # This would be the actual MCP server URL
                url = f"http://{mcp_endpoint}/mcp"

                try:
                    async with session.post(url, json=payload) as response:
                        result = await response.json()
                        return result.get("result")
                except Exception as e:
                    logger.error(f"MCP call failed: {e}")
                    return {"error": str(e)}

        return mcp_connector


# Example bioinformatics tools
class BioinformaticsTools:
    """Collection of basic bioinformatics tools"""

    @staticmethod
    def create_basic_tools(registry: ToolRegistry):
        """Add basic bioinformatics tools to registry"""

        @registry.register("gc_content")
        async def gc_content(sequence: str) -> float:
            """Calculate GC content of a sequence"""
            sequence = sequence.upper()
            gc_count = sequence.count("G") + sequence.count("C")
            return (gc_count / len(sequence)) * 100 if sequence else 0

        @registry.register("reverse_complement")
        async def reverse_complement(sequence: str) -> str:
            """Get reverse complement of DNA sequence"""
            complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
            sequence = sequence.upper()
            return "".join(complement.get(base, base) for base in reversed(sequence))

        @registry.register("find_orfs")
        async def find_orfs(
            sequence: str, min_length: int = 100
        ) -> List[Dict[str, Any]]:
            """Find open reading frames in sequence"""
            orfs = []
            start_codon = "ATG"
            stop_codons = ["TAA", "TAG", "TGA"]

            sequence = sequence.upper()

            for frame in range(3):
                for i in range(frame, len(sequence) - 2, 3):
                    codon = sequence[i : i + 3]
                    if codon == start_codon:
                        # Look for stop codon
                        for j in range(i + 3, len(sequence) - 2, 3):
                            codon = sequence[j : j + 3]
                            if codon in stop_codons:
                                orf_length = j - i + 3
                                if orf_length >= min_length:
                                    orfs.append(
                                        {
                                            "start": i,
                                            "end": j + 3,
                                            "length": orf_length,
                                            "frame": frame,
                                            "sequence": sequence[i : j + 3],
                                        }
                                    )
                                break

            return orfs
