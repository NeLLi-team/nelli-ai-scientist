"""
Enhanced Universal MCP Agent with Planning and Progress Tracking
Adds initial reasoning, execution planning, and step-by-step progress tracking
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pathlib import Path

# Reduce noise from HTTP requests and OpenAI client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from pydantic import BaseModel
from .llm_interface import LLMInterface, LLMProvider
from .communication import FIPAMessage, Performative
from .prompt_manager import PromptManager
from .task_planner import TaskPlanner
from .progress_tracker import ProgressTracker
from .config_loader import ConfigLoader
from .execution_models import (
    ExecutionPlan, ExecutionStep, ReasoningResult, PlanningContext,
    StepStatus, TaskComplexity
)

# Import the original agent for base functionality
from .agent import UniversalMCPAgent, AgentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedAgentConfig(AgentConfig):
    """Enhanced agent configuration with planning features"""
    config_file: str = "config/agent_config.yaml"


class EnhancedUniversalMCPAgent(UniversalMCPAgent):
    """
    Enhanced Universal AI Agent with planning and progress tracking capabilities
    
    New Features:
    1. Initial reasoning phase using advanced model
    2. Detailed execution planning
    3. Step-by-step progress tracking with colored display
    4. Progress reports saved to markdown files
    5. Self-reflection and plan adaptation
    """

    def __init__(self, config: EnhancedAgentConfig, config_loader: ConfigLoader = None):
        # Use provided config loader or create new one
        self.config_loader = config_loader if config_loader else ConfigLoader(config.config_file)
        
        # Update the base config with loaded values
        config.name = self.config_loader.get_agent_name()
        config.description = self.config_loader.get_agent_description()
        config.llm_provider = LLMProvider(self.config_loader.get_llm_provider())
        
        super().__init__(config)
        self.enhanced_config = config
        
        # Create separate LLM interface for reasoning with more capable model
        self.reasoning_llm = LLMInterface(provider=config.llm_provider)
        # Override the model for reasoning
        if hasattr(self.reasoning_llm.llm, 'model'):
            self.reasoning_llm.llm.model = self.config_loader.get_reasoning_model()
        
        # Initialize planning components
        self.task_planner = TaskPlanner(self.reasoning_llm, self.prompt_manager)
        self.progress_tracker = ProgressTracker(self.config_loader.get_reports_directory())
        
        # Enhanced state tracking
        self.current_plan: Optional[ExecutionPlan] = None
        self.current_plan_id: Optional[str] = None
        self.execution_results: List[Dict[str, Any]] = []
        self.iteration_count: int = 0
        
        logger.info(f"✨ Initialized Enhanced Universal MCP Agent: {self.agent_id}")

    async def process_natural_language_with_planning(self, user_input: str) -> Dict[str, Any]:
        """
        Enhanced natural language processing with reasoning and planning phases
        """
        self.iteration_count += 1
        
        # Step 1: Initial Reasoning Phase (using advanced model)
        if self.config_loader.is_reasoning_enabled():
            print("\n🧠 \033[36mInitial Reasoning Phase...\033[0m")
            reasoning_result = await self.task_planner.reason_about_task(user_input, self.discovered_tools)
            print(f"📊 Complexity Assessment: \033[33m{reasoning_result.complexity_assessment.value}\033[0m")
            print(f"🎯 Goal: {reasoning_result.goal_analysis}")
        else:
            # Fallback reasoning
            reasoning_result = ReasoningResult(
                user_request=user_input,
                understanding="Direct processing without reasoning",
                goal_analysis=user_input,
                complexity_assessment=TaskComplexity.MODERATE,
                available_tools_analysis="Standard tools",
                approach_strategy="Direct execution"
            )

        # Step 2: Planning Phase (if complex enough)
        if (self.config_loader.is_planning_enabled() and 
            reasoning_result.complexity_assessment != TaskComplexity.SIMPLE):
            
            print("\n📋 \033[36mExecution Planning Phase...\033[0m")
            
            planning_context = PlanningContext(
                reasoning_result=reasoning_result,
                available_tools=self.discovered_tools,
                tool_categories=self.tool_categories
            )
            
            self.current_plan = await self.task_planner.create_execution_plan(
                reasoning_result, planning_context
            )
            
            # Start progress tracking
            if self.config_loader.is_progress_tracking_enabled():
                self.current_plan_id = self.progress_tracker.start_plan(self.current_plan)
                print(f"📊 Created plan with \033[33m{len(self.current_plan.steps)}\033[0m steps")
                
                # Display initial plan
                self._display_plan_overview()
            
            # Execute the plan step by step
            return await self._execute_planned_workflow()
        
        else:
            # Simple direct execution for basic tasks
            print("\n🚀 \033[36mDirect Execution (Simple Task)...\033[0m")
            return await self.process_natural_language(user_input)

    async def _execute_planned_workflow(self) -> Dict[str, Any]:
        """Execute the planned workflow with progress tracking"""
        if not self.current_plan:
            return {"error": "No execution plan available"}
        
        plan = self.current_plan
        all_results = []
        
        print(f"\n🚀 \033[36mExecuting Planned Workflow...\033[0m")
        
        # Execute steps in dependency order
        while True:
            # Get next executable steps
            pending_steps = plan.get_pending_steps()
            
            if not pending_steps:
                # Check if we're done or stuck
                incomplete_steps = [s for s in plan.steps if s.status == StepStatus.PENDING]
                if not incomplete_steps:
                    print("\n🎉 \033[32mAll steps completed!\033[0m")
                    break
                else:
                    print(f"\n⚠️  \033[31mBlocked: {len(incomplete_steps)} steps waiting for dependencies\033[0m")
                    # Debug: Show what dependencies are blocking
                    completed_ids = {s.id for s in plan.steps if s.status == StepStatus.COMPLETED}
                    completed_names = {s.name for s in plan.steps if s.status == StepStatus.COMPLETED}
                    for step in incomplete_steps:
                        unmet_deps = [dep for dep in step.dependencies 
                                    if dep not in completed_ids and dep not in completed_names]
                        if unmet_deps:
                            print(f"   • {step.name} waiting for: {unmet_deps}")
                        else:
                            print(f"   • {step.name} has no dependencies but isn't marked as pending")
                    break
            
            # Execute next step
            step = pending_steps[0]  # Execute one step at a time for better progress tracking
            
            print(f"\n🔧 \033[33mExecuting:\033[0m {step.name}")
            
            # Update progress tracker
            if self.config_loader.is_progress_tracking_enabled():
                self.progress_tracker.update_step_progress(
                    self.current_plan_id, step.id, StepStatus.IN_PROGRESS
                )
            
            # Execute the step
            result = await self._execute_planned_step(step)
            all_results.append(result)
            
            # Update step status based on result
            # Check both the execution error and tool result for errors
            execution_error = result.get("error")
            tool_result = result.get("result", {})
            tool_error = None
            
            if isinstance(tool_result, dict):
                if "error" in tool_result:
                    tool_error = tool_result["error"]
                elif tool_result == {} or str(tool_result).strip() == "":
                    tool_error = "Tool returned empty result"
            elif tool_result is None:
                tool_error = "Tool returned no result"
            
            if execution_error or tool_error:
                error_msg = execution_error or tool_error
                print(f"❌ \033[31mStep failed:\033[0m {error_msg}")
                step.retry_count += 1
                
                if step.retry_count <= step.max_retries:
                    print(f"🔄 \033[33mRetrying step\033[0m ({step.retry_count}/{step.max_retries})")
                    # Reset status for retry
                    plan.update_step_status(step.id, StepStatus.PENDING)
                    continue
                else:
                    plan.update_step_status(step.id, StepStatus.FAILED, error=result["error"])
                    if self.config_loader.is_progress_tracking_enabled():
                        self.progress_tracker.update_step_progress(
                            self.current_plan_id, step.id, StepStatus.FAILED, error=result["error"]
                        )
            else:
                print(f"✅ \033[32mStep completed successfully\033[0m")
                
                # Display the step result immediately
                self._display_step_result(step, result)
                
                plan.update_step_status(step.id, StepStatus.COMPLETED, result=result)
                if self.config_loader.is_progress_tracking_enabled():
                    self.progress_tracker.update_step_progress(
                        self.current_plan_id, step.id, StepStatus.COMPLETED, result=result
                    )
            
            # Display updated progress
            if self.config_loader.is_progress_tracking_enabled():
                progress_display = self.progress_tracker.display_colored_progress(self.current_plan_id)
                print(progress_display)
                
                # Skip verbose step reflection to reduce output
                # Only generate reflection for complex tasks if needed
                if False:  # Disabled to reduce verbosity
                    try:
                        reflection_notes = await self._generate_step_reflection(step, result)
                        progress_report = self.progress_tracker.generate_progress_report(
                            self.current_plan_id, self.iteration_count, reflection_notes
                        )
                    except Exception as e:
                        # Skip reflection if rate limited
                        reflection_notes = f"Step reflection skipped due to API limitation: {e}"
            
            # Check if we need to adapt the plan based on results
            if step.status == StepStatus.FAILED and len(all_results) > 1:
                print("\n🔄 \033[33mAdapting plan based on results...\033[0m")
                adapted_plan = await self.task_planner.adapt_plan(
                    plan, all_results, self.discovered_tools
                )
                if adapted_plan != plan:
                    self.current_plan = adapted_plan
                    plan = adapted_plan
                    print("📋 \033[32mPlan adapted successfully\033[0m")
        
        # Complete the plan
        if self.config_loader.is_progress_tracking_enabled():
            self.progress_tracker.complete_plan(self.current_plan_id)
        
        # Skip final reflection to reduce verbose output
        final_reflection = "Workflow completed successfully" if all_results else "No steps executed"
        
        return {
            "plan_executed": True,
            "steps_completed": len([r for r in all_results if not r.get("error")]),
            "steps_failed": len([r for r in all_results if r.get("error")]),
            "results": all_results,
            "reflection": final_reflection
        }

    async def _execute_planned_step(self, step: ExecutionStep) -> Dict[str, Any]:
        """Execute a single planned step"""
        # Validate tool name
        if not step.tool_name or step.tool_name == "None":
            return {
                "step_id": step.id,
                "step_name": step.name,
                "tool": step.tool_name,
                "parameters": {},
                "result": {},
                "error": f"Invalid tool name: '{step.tool_name}'. This appears to be a manual step that cannot be executed."
            }
        
        # Check if tool exists
        if step.tool_name not in self.discovered_tools:
            available_tools = list(self.discovered_tools.keys())
            return {
                "step_id": step.id,
                "step_name": step.name,
                "tool": step.tool_name,
                "parameters": {},
                "result": {},
                "error": f"Tool '{step.tool_name}' not found. Available tools: {available_tools}"
            }
        
        # Resolve parameters (handle chaining)
        resolved_params = await self._resolve_step_parameters(step)
        
        # Call the tool
        result = await self._call_mcp_tool(step.tool_name, **resolved_params)
        
        return {
            "step_id": step.id,
            "step_name": step.name,
            "tool": step.tool_name,
            "parameters": resolved_params,
            "result": result,
            "error": result.get("error") if isinstance(result, dict) else None
        }

    async def _resolve_step_parameters(self, step: ExecutionStep) -> Dict[str, Any]:
        """Resolve step parameters, handling result chaining"""
        resolved_params = {}
        
        for param_name, param_value in step.parameters.items():
            if param_value == "ANALYSIS_RESULTS":
                if self.execution_results:
                    # Use the most recent successful result's actual result data
                    for result in reversed(self.execution_results):
                        if not result.get("error") and result.get("result"):
                            # Extract the actual tool result, not the wrapper
                            tool_result = result["result"]
                            if isinstance(tool_result, dict) and "result" in tool_result:
                                resolved_params[param_name] = tool_result["result"]
                            else:
                                resolved_params[param_name] = tool_result
                            break
                    else:
                        # Fallback if no successful results found
                        resolved_params[param_name] = {}
                else:
                    # No previous results available - skip this parameter or use empty dict
                    resolved_params[param_name] = {}
            else:
                resolved_params[param_name] = param_value
        
        return resolved_params

    async def _generate_step_reflection(self, step: ExecutionStep, result: Dict[str, Any]) -> str:
        """Generate reflection notes for a completed step"""
        prompt = f"""
Reflect on the execution of this step:

Step: {step.name}
Tool: {step.tool_name}
Expected: {step.expected_output}
Result: {json.dumps(result, indent=2)}

Provide a brief reflection (1-2 sentences) on:
- Did the step achieve its intended purpose?
- Any insights or issues discovered?
- Impact on overall progress?
"""
        
        try:
            return await self.llm.generate(prompt, temperature=0.3, max_tokens=200)
        except Exception as e:
            return f"Step reflection failed: {e}"

    async def _generate_final_reflection(self, all_results: List[Dict[str, Any]]) -> str:
        """Generate final reflection on the entire workflow"""
        successful_steps = [r for r in all_results if not r.get("error")]
        failed_steps = [r for r in all_results if r.get("error")]
        
        context = f"""
Workflow completed with {len(successful_steps)} successful steps and {len(failed_steps)} failed steps.

Original goal: {self.current_plan.goal if self.current_plan else 'Unknown'}

Results summary:
{json.dumps(all_results, indent=2)}

Provide a comprehensive reflection on:
1. Was the original goal achieved?
2. What insights were discovered?
3. What could be improved for similar tasks?
4. Overall assessment of the workflow effectiveness?
"""
        
        try:
            return await self.llm.generate(context, temperature=0.4, max_tokens=500)
        except Exception as e:
            return f"Final reflection failed: {e}"

    def _display_step_result(self, step: ExecutionStep, result: Dict[str, Any]):
        """Display the result of a completed step"""
        tool_name = step.tool_name
        tool_result = result.get("result")
        
        print(f"\n🛠️ \033[36mStep Result - {step.name}:\033[0m")
        
        # Special handling for tree_view tool
        if tool_name == "tree_view" and isinstance(tool_result, dict):
            if tool_result.get("success") and tool_result.get("tree_display"):
                print(f"\n📁 \033[32mDirectory Tree:\033[0m")
                print(tool_result["tree_display"])
                
                summary = tool_result.get("summary", {})
                if summary:
                    print(f"\n📊 \033[33mSummary:\033[0m {summary.get('directories', 0)} directories, {summary.get('files', 0)} files, {summary.get('total_size_formatted', '0 B')}")
            else:
                print(f"❌ Tree view failed: {tool_result.get('error', 'Unknown error')}")
        
        # Special handling for find_files tool
        elif tool_name == "find_files" and isinstance(tool_result, dict):
            if tool_result.get("success"):
                files = tool_result.get("found_files", [])
                print(f"\n🔍 \033[32mFound {len(files)} files:\033[0m")
                for file_info in files[:10]:  # Show first 10 files
                    print(f"  • {file_info['name']} ({file_info['size_formatted']}) - {file_info['relative_path']}")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more files")
                print(f"\n📊 Total size: {tool_result.get('total_size_formatted', '0 B')}")
            else:
                print(f"❌ File search failed: {tool_result.get('error', 'Unknown error')}")
        
        # Special handling for read_file tool
        elif tool_name == "read_file" and isinstance(tool_result, dict):
            if tool_result.get("success"):
                content = tool_result.get("content", "")
                lines = content.split('\n')
                total_lines = tool_result.get('lines', len(lines))
                size_formatted = tool_result.get('size_formatted', '0 B')
                
                print(f"\n📄 \033[32mFile Read Successfully:\033[0m")
                print(f"  📊 Size: {size_formatted}")
                print(f"  📝 Lines: {total_lines:,}")
                
                # Only show content preview for small files or if explicitly requested
                if total_lines <= 50:
                    print(f"\n📋 \033[33mFile Content:\033[0m")
                    for i, line in enumerate(lines[:10], 1):
                        print(f"  {i:3}: {line}")
                    if len(lines) > 10:
                        print(f"  ... and {len(lines) - 10} more lines")
                else:
                    print(f"\n📋 \033[33mContent Preview (first 3 lines):\033[0m")
                    for i, line in enumerate(lines[:3], 1):
                        print(f"  {i:3}: {line[:100]}{'...' if len(line) > 100 else ''}")
                    print(f"  📄 File is large ({total_lines:,} lines) - content preview limited")
            else:
                print(f"❌ File read failed: {tool_result.get('error', 'Unknown error')}")
        
        # Special handling for find_file_by_name tool
        elif tool_name == "find_file_by_name" and isinstance(tool_result, dict):
            if tool_result.get("success"):
                files = tool_result.get("found_files", [])
                filename = tool_result.get("filename", "unknown")
                print(f"\n🔍 \033[32mSearching for '{filename}':\033[0m")
                if files:
                    print(f"Found {len(files)} file(s):")
                    for file_info in files:
                        print(f"  📄 {file_info['path']} ({file_info['size_formatted']})")
                else:
                    print(f"❌ No files named '{filename}' found in {tool_result.get('search_path', '.')}")
            else:
                print(f"❌ File search failed: {tool_result.get('error', 'Unknown error')}")
        
        # Special handling for BioPython tools
        elif tool_name in ["analyze_fasta_file", "sequence_stats", "translate_sequence", "multiple_alignment", "phylogenetic_tree"]:
            if isinstance(tool_result, dict):
                if "error" in tool_result:
                    print(f"❌ BioPython tool failed: {tool_result['error']}")
                elif "summary_statistics" in tool_result:
                    # Handle analyze_fasta_file result
                    print(f"\n🧬 \033[32mSequence Analysis Results:\033[0m")
                    stats = tool_result["summary_statistics"]
                    print(f"  📁 File: {tool_result.get('file_info', {}).get('file_path', 'Unknown')}")
                    print(f"  🔢 Total sequences: {stats.get('total_sequences', 0):,}")
                    print(f"  📏 Total length: {stats.get('total_length', 0):,} bp")
                    print(f"  📊 Average length: {stats.get('average_length', 0):,.0f} bp")
                    print(f"  🧪 Average GC content: {stats.get('average_gc_content', 0):.1f}%")
                    print(f"  📐 Longest sequence: {stats.get('longest_sequence', 0):,} bp")
                    print(f"  📏 Shortest sequence: {stats.get('shortest_sequence', 0):,} bp")
                elif "length" in tool_result and "type" in tool_result:
                    # Handle individual sequence_stats result
                    print(f"\n🧬 \033[32mSequence Statistics:\033[0m")
                    print(f"  📏 Length: {tool_result.get('length', 0):,} bp")
                    print(f"  🧪 Type: {tool_result.get('type', 'unknown')}")
                    if "gc_content" in tool_result:
                        print(f"  🧪 GC content: {tool_result['gc_content']:.1f}%")
                    if "orfs" in tool_result:
                        orfs = tool_result["orfs"]
                        print(f"  🔬 ORFs found: {orfs.get('count', 0)}")
                        print(f"  📐 Longest ORF: {orfs.get('longest', 0)} bp")
                else:
                    print(f"✅ BioPython tool completed successfully")
                    # Show available data keys
                    keys = list(tool_result.keys())
                    print(f"  📋 Data available: {', '.join(keys)}")
            else:
                print(f"📄 BioPython result: {tool_result}")
        
        # Special handling for write_json_report
        elif tool_name == "write_json_report" and isinstance(tool_result, dict):
            if "error" in tool_result:
                print(f"❌ Report writing failed: {tool_result['error']}")
            elif "output_path" in tool_result:
                print(f"\n📄 \033[32mReport Generated:\033[0m")
                print(f"  📁 File: {tool_result['output_path']}")
                print(f"  📊 Size: {tool_result.get('file_size', 0):,} bytes")
            else:
                print(f"✅ Report writing completed")
        
        # Generic handling for other tools
        else:
            if isinstance(tool_result, dict):
                if tool_result.get("success"):
                    print(f"✅ Tool executed successfully")
                    # Display key information if available
                    for key in ["result", "output", "data", "content"]:
                        if key in tool_result and tool_result[key]:
                            value = tool_result[key]
                            if isinstance(value, str) and len(value) > 200:
                                print(f"  {key}: {value[:200]}...")
                            else:
                                print(f"  {key}: {value}")
                elif "error" in tool_result:
                    print(f"❌ Tool failed: {tool_result['error']}")
                else:
                    print(f"❌ Tool failed: Unknown error")
            else:
                print(f"📄 Result: {tool_result}")

    def _display_plan_overview(self):
        """Display the execution plan overview"""
        if not self.current_plan:
            return
        
        plan = self.current_plan
        print(f"\n📋 \033[1mExecution Plan Overview\033[0m")
        print(f"\033[36m{'='*50}\033[0m")
        print(f"🎯 \033[1mGoal:\033[0m {plan.goal}")
        print(f"📊 \033[1mComplexity:\033[0m {plan.complexity.value}")
        print(f"⏱️  \033[1mEstimated Duration:\033[0m {plan.estimated_duration_minutes} minutes")
        print(f"📝 \033[1mSteps:\033[0m {len(plan.steps)}")
        
        print(f"\n\033[1m📋 Step Checklist:\033[0m")
        for i, step in enumerate(plan.steps, 1):
            print(f"  \033[90m{i}. ⏳ {step.name}\033[0m")
        
        print(f"\033[36m{'='*50}\033[0m")

    async def terminal_chat(self):
        """Enhanced terminal chat interface with planning"""
        # Initialize first
        await self.initialize()
        
        # Display enhanced welcome message
        self._display_enhanced_welcome()
        
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
                    self._display_enhanced_help()
                    continue
                
                if user_input.lower() in ['tools', 'list']:
                    self._display_tools()
                    continue
                
                if user_input.lower() == 'clear':
                    self._clear_screen()
                    self._display_enhanced_welcome()
                    continue
                
                if user_input.lower() == 'progress' and self.current_plan_id:
                    progress_display = self.progress_tracker.display_colored_progress(self.current_plan_id)
                    print(progress_display)
                    continue
                
                # Process with enhanced planning
                print("\n🤔 \033[36mAnalyzing request...\033[0m")
                result = await self.process_natural_language_with_planning(user_input)
                
                # Display final result
                if result.get("plan_executed"):
                    print(f"\n💡 \033[32mWorkflow Summary:\033[0m")
                    print(f"   ✅ {result['steps_completed']} steps completed")
                    if result['steps_failed'] > 0:
                        print(f"   ❌ {result['steps_failed']} steps failed")
                    
                    # Only show very brief reflection if present
                    if result.get("reflection") and len(result['reflection']) < 100:
                        print(f"\n💡 \033[36mWorkflow Summary:\033[0m")
                        print(f"   {result['reflection']}")
                else:
                    # Handle direct responses - multiple formats
                    if isinstance(result, dict):
                        if result.get("response_type") == "direct_answer":
                            print(f"\n🤖 Assistant: {result.get('direct_answer', '')}")
                        elif result.get("tool_results"):
                            # Display tool results from direct execution
                            print(f"\n🛠️ \033[36mTool Results:\033[0m")
                            for tool_result in result["tool_results"]:
                                tool_name = tool_result.get("tool", "Unknown")
                                if tool_result.get("success"):
                                    print(f"\n✅ \033[32m{tool_name}:\033[0m")
                                    # Display tree view results specially
                                    if tool_name == "tree_view" and tool_result.get("result", {}).get("tree_display"):
                                        print(tool_result["result"]["tree_display"])
                                        summary = tool_result["result"].get("summary", {})
                                        print(f"\n📊 Summary: {summary.get('directories', 0)} dirs, {summary.get('files', 0)} files, {summary.get('total_size_formatted', '0 B')}")
                                    else:
                                        # Display other tool results
                                        print(f"   {tool_result.get('result', 'No result')}")
                                else:
                                    print(f"\n❌ \033[31m{tool_name} failed:\033[0m {tool_result.get('error', 'Unknown error')}")
                        elif result.get("direct_response"):
                            print(f"\n🤖 Assistant: {result['direct_response']}")
                        else:
                            # Fallback - display the raw result
                            print(f"\n📄 \033[36mResult:\033[0m")
                            print(f"   {result}")
                    else:
                        # Handle string responses
                        print(f"\n🤖 Assistant: {result}")
                
                # Store in history
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "user": user_input,
                    "result": result,
                    "type": "enhanced_chat"
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.exception("Enhanced chat error")

    def _display_enhanced_welcome(self):
        """Display enhanced welcome message"""
        print("\033[36m")  # Cyan color
        print("\n" + "="*70)
        print(r"  ███╗   ██╗███████╗██╗     ██╗     ██╗    ✨ ENHANCED")
        print(r"  ████╗  ██║██╔════╝██║     ██║     ██║")
        print(r"  ██╔██╗ ██║█████╗  ██║     ██║     ██║    🧠 Reasoning")
        print(r"  ██║╚██╗██║██╔══╝  ██║     ██║     ██║    📋 Planning")
        print(r"  ██║ ╚████║███████╗███████╗███████╗██║    📊 Progress Tracking")
        print(r"  ╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝╚═╝")
        print("              🧪 Enhanced AI Scientist Agent 🔬")
        print("\033[0m")  # Reset color
        print("="*70)
        print(f"\033[32mAgent:\033[0m {self.config.name}")
        print(f"\033[32mRole:\033[0m {self.config.description}")
        print(f"\033[32mID:\033[0m {self.agent_id}")
        print(f"\033[32mReasoning Model:\033[0m {self.config_loader.get_reasoning_model()}")
        print(f"\033[32mPlanning Model:\033[0m {self.config_loader.get_planning_model()}")
        
        if self.discovered_tools:
            print(f"\n\033[33m📊 Loaded {len(self.discovered_tools)} tools from {len(self.mcp_servers)} servers\033[0m")
            
            # Show summary by server
            for category in self.tool_categories.values():
                print(f"  \033[36m•\033[0m {category['name']}: \033[33m{len(category['tools'])}\033[0m tools")
        else:
            print("\n\033[31m⚠️  No tools loaded. Check your MCP configuration.\033[0m")
        
        # Enhanced features status
        features = []
        if self.config_loader.is_reasoning_enabled():
            features.append("🧠 Reasoning")
        if self.config_loader.is_planning_enabled():
            features.append("📋 Planning")
        if self.config_loader.is_progress_tracking_enabled():
            features.append("📊 Progress Tracking")
        
        if features:
            print(f"\n\033[35m✨ Enhanced Features:\033[0m {' | '.join(features)}")
        
        print("\n\033[35m💡 Commands:\033[0m help, tools, progress, clear, quit")
        print("\033[35m💬 Or just type naturally - I'll reason, plan, and execute!\033[0m")
        print("\033[36m" + "="*70 + "\033[0m")

    def _display_enhanced_help(self):
        """Display enhanced help information"""
        print("\n📚 Enhanced Universal MCP Agent Help")
        print("="*60)
        print("\n🎯 Natural Language with Intelligence:")
        print("  Tell me what you want to accomplish, and I'll:")
        print("  🧠 Reason about your request and analyze complexity")
        print("  📋 Create a detailed execution plan")
        print("  🚀 Execute steps with real-time progress tracking")
        print("  🔍 Reflect on results and adapt as needed")
        print("\n⌨️  Commands:")
        print("  help, h, ?     - Show this help")
        print("  tools, list    - List all available tools")
        print("  progress       - Show current execution progress")
        print("  clear          - Clear screen")
        print("  quit, exit, q  - Exit the chat")
        print("\n✨ Enhanced Features:")
        print("  • Initial reasoning with advanced model")
        print("  • Automatic task complexity assessment")
        print("  • Step-by-step execution planning")
        print("  • Real-time progress tracking with colors")
        print("  • Automatic progress reports saved to markdown")
        print("  • Plan adaptation based on execution results")
        print("  • Self-reflection and improvement")


# Main execution for enhanced agent
async def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Universal MCP Agent")
    parser.add_argument(
        "--config", 
        default="mcp_config.json",
        help="Path to MCP configuration file (default: mcp_config.json)"
    )
    parser.add_argument(
        "--name",
        default="nelli-enhanced-agent", 
        help="Agent name (default: nelli-enhanced-agent)"
    )
    parser.add_argument(
        "--llm-provider",
        default="cborg",
        choices=["cborg", "claude", "openai"],
        help="LLM provider to use (default: cborg)"
    )
    parser.add_argument(
        "--reasoning-model",
        default="google/gemini-pro",
        help="Model to use for initial reasoning phase (default: google/gemini-pro)"
    )
    parser.add_argument(
        "--planning-model", 
        default="google/gemini-flash-lite",
        help="Model to use for planning and execution (default: google/gemini-flash-lite)"
    )
    parser.add_argument(
        "--disable-reasoning",
        action="store_true",
        help="Disable initial reasoning phase"
    )
    parser.add_argument(
        "--disable-planning",
        action="store_true",
        help="Disable execution planning"
    )
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        help="Disable progress tracking"
    )
    args = parser.parse_args()
    
    # Get agent directory from environment or use relative path
    import os
    agent_dir = os.getenv("NELLI_AGENT_DIR", "agents/sophisticated_agent")
    config_file_path = os.path.join(agent_dir, "config/agent_config.yaml")
    
    # Create temporary config loader to apply command line overrides
    temp_loader = ConfigLoader(config_file_path)
    temp_loader.override_with_args(args)
    
    # Create enhanced configuration with values from config loader
    config = EnhancedAgentConfig(
        name=temp_loader.get_agent_name(),
        mcp_config_path=args.config,
        config_file=config_file_path
    )
    
    # Create and run enhanced agent
    agent = EnhancedUniversalMCPAgent(config, temp_loader)
    
    # Start enhanced terminal chat
    await agent.terminal_chat()


if __name__ == "__main__":
    asyncio.run(main())