#!/usr/bin/env python3
"""
Enhanced Agent Runner with Stdio MCP Support
Uses the new stdio-based MCP connections for truly independent servers
"""

import asyncio
import os
import sys
from pathlib import Path
import signal
import logging

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Environment setup
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Change to project root
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)
print(f"🏠 Changed working directory to: {project_root}")

# Load environment variables if dotenv is available
env_path = project_root / ".env"
if env_path.exists() and HAS_DOTENV:
    load_dotenv(env_path)
    print(f"🔐 Loaded environment variables from {env_path}")
elif env_path.exists():
    print(f"📄 Found .env file but python-dotenv not installed")

# Import the stdio-enabled agent
from src.agent_stdio import UniversalMCPAgentStdio, AgentConfig
from src.llm_interface import LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StdioAgentRunner:
    """Runner for the stdio-enabled agent"""
    
    def __init__(self):
        self.agent = None
        self.running = True
        
    async def initialize_agent(self):
        """Initialize the agent with stdio MCP support"""
        # Load configuration
        config_path = project_root / "agents" / "sophisticated_agent" / "config" / "agent_config.yaml"
        mcp_config_path = project_root / "agents" / "sophisticated_agent" / "mcp_config.json"
        
        print(f"📁 Agent config path: {config_path}")
        
        # Create agent configuration
        agent_config = AgentConfig(
            name="nelli-enhanced-stdio-agent",
            role="Enhanced Universal MCP Agent with Stdio Support",
            description="AI agent with truly independent MCP servers via stdio",
            llm_provider=LLMProvider.CBORG,
            temperature=0.7,
            max_tokens=4096,
            mcp_config_path=str(mcp_config_path),
            use_stdio_connections=True
        )
        
        # Initialize agent
        self.agent = UniversalMCPAgentStdio(agent_config)
        await self.agent.initialize()
        
        return self.agent
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def print_banner(self):
        """Print the agent banner"""
        status = self.agent.get_status() if self.agent else {}
        
        print("\n" + "=" * 70)
        print("  ███╗   ██╗███████╗██╗     ██╗     ██╗    ✨ STDIO ENHANCED")
        print("  ████╗  ██║██╔════╝██║     ██║     ██║")
        print("  ██╔██╗ ██║█████╗  ██║     ██║     ██║    🧠 Reasoning")
        print("  ██║╚██╗██║██╔══╝  ██║     ██║     ██║    📋 Planning") 
        print("  ██║ ╚████║███████╗███████╗███████╗██║    📊 Progress Tracking")
        print("  ╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝╚═╝    🔌 Stdio MCPs")
        print("              🧪 Enhanced AI Scientist Agent 🔬")
        print("=" * 70)
        print(f"Agent: {status.get('agent_id', 'nelli-stdio-agent')}")
        print(f"Role: {status.get('role', 'Enhanced Universal MCP Agent')}")
        print(f"Connected Servers: {status.get('connected_servers', 0)}")
        print(f"Available Tools: {status.get('discovered_tools', 0)}")
        
        if status.get('tool_categories'):
            print(f"\n📊 Tools by Server:")
            for server_id, category in status['tool_categories'].items():
                tools_count = len(category.get('tools', []))
                print(f"  • {category.get('name', server_id)}: {tools_count} tools")
        
        print(f"\n✨ Features: 🧠 Reasoning | 📋 Planning | 📊 Progress | 🔌 Stdio MCPs")
        print(f"💡 Commands: help, tools, status, clear, quit")
        print(f"💬 Or just type naturally - I'll reason, plan, and execute!")
        print("=" * 70)
        
    async def interactive_loop(self):
        """Main interactive loop"""
        
        # Check if running in interactive terminal
        if not sys.stdin.isatty():
            print("⚠️  Not running in interactive terminal. Exiting.")
            print("💡 To use the chat interface, run this directly in a terminal.")
            return
            
        self.print_banner()
        
        while self.running:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    self.running = False
                    break
                elif user_input.lower() == 'help':
                    self.print_help()
                    continue
                elif user_input.lower() == 'tools':
                    self.show_tools()
                    continue
                elif user_input.lower() == 'status':
                    self.show_status()
                    continue
                elif user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    self.print_banner()
                    continue
                
                # Process with agent
                if self.agent:
                    print("\n🤔 Processing...")
                    response = await self.agent.process_message(user_input)
                    print(f"\n🤖 Agent: {response}")
                else:
                    print("❌ Agent not initialized")
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Interrupted. Use 'quit' to exit gracefully.")
            except EOFError:
                print(f"\n👋 Goodbye!")
                self.running = False
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.exception("Error in interactive loop")
                
    def print_help(self):
        """Print help information"""
        print("\n📚 Help - Available Commands:")
        print("  help     - Show this help message")
        print("  tools    - List all available tools")
        print("  status   - Show agent status")  
        print("  clear    - Clear the screen")
        print("  quit     - Exit the agent")
        print("\n💬 Natural Language:")
        print("  You can also just type naturally and the agent will:")
        print("  • Reason about your request")
        print("  • Plan the steps needed")
        print("  • Execute using available tools")
        print("  • Track progress throughout")
        
    def show_tools(self):
        """Show available tools"""
        if not self.agent:
            print("❌ Agent not initialized")
            return
            
        tools = self.agent.discovered_tools
        if not tools:
            print("⚠️  No tools available")
            return
            
        print(f"\n🔧 Available Tools ({len(tools)}):")
        for tool_name, tool_info in tools.items():
            server_name = tool_info.get('server_name', 'Unknown')
            description = tool_info.get('description', 'No description')
            print(f"  • {tool_name} ({server_name})")
            print(f"    {description}")
            
    def show_status(self):
        """Show detailed agent status"""
        if not self.agent:
            print("❌ Agent not initialized")
            return
            
        status = self.agent.get_status()
        print(f"\n📊 Agent Status:")
        print(f"  ID: {status.get('agent_id')}")
        print(f"  Role: {status.get('role')}")
        print(f"  Connected Servers: {status.get('connected_servers')}")
        print(f"  Available Tools: {status.get('discovered_tools')}")
        print(f"  Conversation Length: {status.get('conversation_length')}")
        
        if status.get('tool_categories'):
            print(f"\n🔌 Connected Servers:")
            for server_id, category in status['tool_categories'].items():
                tools = category.get('tools', [])
                print(f"  • {category.get('name', server_id)}: {len(tools)} tools")
                if tools:
                    print(f"    Tools: {', '.join(tools[:3])}{'...' if len(tools) > 3 else ''}")
                    
    async def cleanup(self):
        """Cleanup resources"""
        if self.agent:
            await self.agent.cleanup()


async def main():
    """Main entry point"""
    runner = StdioAgentRunner()
    
    try:
        # Setup signal handlers
        runner.setup_signal_handlers()
        
        # Initialize agent
        print("🚀 Initializing Enhanced Agent with Stdio MCP Support...")
        await runner.initialize_agent()
        
        # Start interactive loop - use the agent's built-in terminal chat
        await runner.agent.terminal_chat()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.exception("Fatal error in main")
    finally:
        # Cleanup
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())