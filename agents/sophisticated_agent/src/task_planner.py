"""
Task Planner - Creates execution plans using reasoning and tool analysis
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .execution_models import (
    ExecutionPlan, ExecutionStep, ReasoningResult, PlanningContext,
    TaskComplexity, TaskPriority, StepStatus
)
from .llm_interface import LLMInterface
from .prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class TaskPlanner:
    """Plans complex tasks by reasoning about requirements and available tools"""
    
    def __init__(self, llm: LLMInterface, prompt_manager: PromptManager):
        self.llm = llm
        self.prompt_manager = prompt_manager
        
    async def reason_about_task(self, user_request: str, available_tools: Dict[str, Any]) -> ReasoningResult:
        """
        Initial reasoning phase - deeply understand the task using a different model
        This uses the reasoning model to analyze the request in context of available tools
        """
        logger.info("🧠 Starting initial reasoning phase...")
        
        # Build tools context for reasoning
        tools_summary = self._build_tools_summary(available_tools)
        
        # Use reasoning prompt
        prompt = self.prompt_manager.format_prompt(
            "reasoning",
            user_request=user_request,
            tools_summary=tools_summary,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        try:
            # Use more capable model for reasoning (switch temporarily if needed)
            response = await self.llm.generate(prompt, temperature=0.3)
            
            # Parse JSON response
            response = self._clean_json_response(response)
            reasoning_data = json.loads(response)
            
            reasoning_result = ReasoningResult(**reasoning_data)
            logger.info(f"✅ Reasoning complete: {reasoning_result.complexity_assessment} complexity task")
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"❌ Reasoning failed: {e}")
            # Fallback reasoning
            return ReasoningResult(
                user_request=user_request,
                understanding="Basic task understanding (reasoning failed)",
                goal_analysis="Attempting to fulfill user request as understood",
                complexity_assessment=TaskComplexity.MODERATE,
                available_tools_analysis="Standard tools available",
                approach_strategy="Direct tool application",
                estimated_steps=2,
                success_definition=["Task completed without errors"]
            )
    
    async def create_execution_plan(self, reasoning_result: ReasoningResult, context: PlanningContext) -> ExecutionPlan:
        """
        Create detailed execution plan based on reasoning results
        """
        logger.info("📋 Creating execution plan...")
        
        # Determine complexity and priority
        complexity = reasoning_result.complexity_assessment
        priority = self._determine_priority(reasoning_result)
        
        # Build planning context
        tools_context = self._build_detailed_tools_context(context.available_tools)
        
        # Use planning prompt
        prompt = self.prompt_manager.format_prompt(
            "planning",
            reasoning_result=reasoning_result.model_dump(),
            tools_context=tools_context,
            complexity=complexity.value,
            estimated_steps=reasoning_result.estimated_steps
        )
        
        try:
            response = await self.llm.generate(prompt, temperature=0.2)
            response = self._clean_json_response(response)
            plan_data = json.loads(response)
            
            # Create execution plan
            plan = ExecutionPlan(
                goal=reasoning_result.user_request,
                description=plan_data.get("description", reasoning_result.goal_analysis),
                complexity=complexity,
                priority=priority,
                estimated_duration_minutes=plan_data.get("estimated_duration_minutes", self._estimate_duration(complexity)),
                success_criteria=reasoning_result.success_definition
            )
            
            # Add steps
            for step_data in plan_data.get("steps", []):
                step = ExecutionStep(
                    name=step_data["name"],
                    description=step_data["description"],
                    tool_name=step_data["tool_name"],
                    parameters=step_data.get("parameters", {}),
                    dependencies=step_data.get("dependencies", []),
                    expected_output=step_data.get("expected_output", "")
                )
                plan.steps.append(step)
            
            logger.info(f"✅ Plan created with {len(plan.steps)} steps")
            return plan
            
        except Exception as e:
            logger.error(f"❌ Planning failed: {e}")
            # Fallback to simple plan
            return self._create_fallback_plan(reasoning_result, context)
    
    def _create_fallback_plan(self, reasoning_result: ReasoningResult, context: PlanningContext) -> ExecutionPlan:
        """Create a simple fallback plan if detailed planning fails"""
        logger.info("🔄 Creating fallback plan...")
        
        user_request = reasoning_result.user_request.lower()
        
        # Determine appropriate fallback based on user request
        if any(keyword in user_request for keyword in ["list", "find", "files", "directory", "tree", "recursively"]):
            # File listing fallback
            plan = ExecutionPlan(
                goal=reasoning_result.user_request,
                description="File exploration and listing (fallback)",
                complexity=TaskComplexity.SIMPLE,
                priority=TaskPriority.MEDIUM,
                success_criteria=["Files listed successfully"]
            )
            
            # Use available tools for file exploration
            if "sequence" in user_request:
                step = ExecutionStep(
                    name="Find Sequence Files",
                    description="Find sequence files recursively",
                    tool_name="find_files",
                    parameters={"path": ".", "extensions": ["fasta", "fa", "fna", "fastq", "gb", "genbank"]}
                )
            else:
                step = ExecutionStep(
                    name="Show Directory Tree",
                    description="Show directory structure",
                    tool_name="tree_view", 
                    parameters={"path": ".", "max_depth": 3}
                )
            plan.steps.append(step)
            
        elif any(keyword in user_request for keyword in ["analysis", "analyze", "stats", "statistics"]):
            # Sequence analysis fallback
            plan = ExecutionPlan(
                goal=reasoning_result.user_request,
                description="Sequence analysis workflow (fallback)",
                complexity=TaskComplexity.MODERATE,
                priority=TaskPriority.MEDIUM,
                success_criteria=["FASTA file analyzed", "Report generated"]
            )
            
            # Step 1: Analyze FASTA file (this provides comprehensive stats)
            step1 = ExecutionStep(
                name="Analyze FASTA File",
                description="Perform comprehensive analysis of the FASTA file",
                tool_name="analyze_fasta_file",
                parameters={"file_path": "data/nelli_hackathon/contigs100k.fna", "sequence_type": "dna"}
            )
            plan.steps.append(step1)
            
            # Step 2: Write comprehensive report
            step2 = ExecutionStep(
                name="Write Analysis Report",
                description="Write comprehensive analysis results to JSON report",
                tool_name="write_json_report",
                parameters={"data": "ANALYSIS_RESULTS", "output_path": "data/nelli_hackathon/contigs100k_analysis_report.json"},
                dependencies=["Analyze FASTA File"]
            )
            plan.steps.append(step2)
        else:
            # Generic fallback - minimal execution
            plan = ExecutionPlan(
                goal=reasoning_result.user_request,
                description="Generic task execution (fallback)",
                complexity=TaskComplexity.SIMPLE,
                priority=TaskPriority.MEDIUM,
                success_criteria=["Task completed"]
            )
            
            # Just show directory tree as a safe fallback
            step = ExecutionStep(
                name="Explore Current Directory", 
                description="Show current directory structure",
                tool_name="tree_view",
                parameters={"path": ".", "max_depth": 2}
            )
            plan.steps.append(step)
        
        return plan
    
    def _build_tools_summary(self, available_tools: Dict[str, Any]) -> str:
        """Build a concise summary of available tools for reasoning"""
        if not available_tools:
            return "No tools available"
        
        categories = {}
        for tool_name, tool_info in available_tools.items():
            server_name = tool_info.get("server_name", "unknown")
            if server_name not in categories:
                categories[server_name] = []
            categories[server_name].append(f"{tool_name}: {tool_info.get('description', '')}")
        
        summary_parts = []
        for server_name, tools in categories.items():
            summary_parts.append(f"\n{server_name} Server:")
            for tool in tools[:5]:  # Limit to first 5 tools per server for reasoning
                summary_parts.append(f"  - {tool}")
            if len(tools) > 5:
                summary_parts.append(f"  ... and {len(tools) - 5} more tools")
        
        return "\n".join(summary_parts)
    
    def _build_detailed_tools_context(self, available_tools: Dict[str, Any]) -> str:
        """Build detailed tools context for planning"""
        if not available_tools:
            return "No tools available"
        
        context_parts = []
        for tool_name, tool_info in available_tools.items():
            schema = tool_info.get('schema', {})
            required_params = schema.get('required', [])
            all_params = list(schema.get('properties', {}).keys())
            
            context_parts.append(f"\n{tool_name}:")
            context_parts.append(f"  Description: {tool_info.get('description', 'No description')}")
            context_parts.append(f"  Server: {tool_info.get('server_name', 'unknown')}")
            
            if all_params:
                params_str = ", ".join([f"{p}{'*' if p in required_params else ''}" for p in all_params])
                context_parts.append(f"  Parameters: {params_str} (* = required)")
        
        return "\n".join(context_parts)
    
    def _determine_priority(self, reasoning_result: ReasoningResult) -> TaskPriority:
        """Determine task priority based on reasoning results"""
        complexity = reasoning_result.complexity_assessment
        
        if complexity == TaskComplexity.ADVANCED:
            return TaskPriority.HIGH
        elif complexity == TaskComplexity.COMPLEX:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.MEDIUM
    
    def _estimate_duration(self, complexity: TaskComplexity) -> int:
        """Estimate duration in minutes based on complexity"""
        duration_map = {
            TaskComplexity.SIMPLE: 2,
            TaskComplexity.MODERATE: 5,
            TaskComplexity.COMPLEX: 10,
            TaskComplexity.ADVANCED: 15
        }
        return duration_map.get(complexity, 5)
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from LLM"""
        response = response.strip()
        
        # Remove markdown formatting
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        
        # Find JSON boundaries - extract just the JSON object
        start_idx = response.find('{')
        if start_idx != -1:
            # Find the matching closing brace
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx != -1:
                response = response[start_idx:end_idx]
        
        return response
    
    async def adapt_plan(self, plan: ExecutionPlan, execution_results: List[Dict[str, Any]], 
                        available_tools: Dict[str, Any]) -> ExecutionPlan:
        """
        Adapt existing plan based on execution results and changing circumstances
        """
        logger.info("🔄 Adapting execution plan based on results...")
        
        # Build context about what happened
        results_summary = []
        for result in execution_results:
            if result.get("error"):
                results_summary.append(f"❌ {result['tool']}: {result['error']}")
            else:
                results_summary.append(f"✅ {result['tool']}: Success")
        
        # Use adaptation prompt
        prompt = self.prompt_manager.format_prompt(
            "plan_adaptation",
            original_plan=plan.model_dump(),
            execution_results="\n".join(results_summary),
            available_tools=self._build_detailed_tools_context(available_tools)
        )
        
        try:
            response = await self.llm.generate(prompt, temperature=0.3)
            response = self._clean_json_response(response)
            adaptation_data = json.loads(response)
            
            # Apply adaptations
            if adaptation_data.get("add_steps"):
                for step_data in adaptation_data["add_steps"]:
                    step = ExecutionStep(**step_data)
                    plan.steps.append(step)
            
            if adaptation_data.get("modify_steps"):
                for modification in adaptation_data["modify_steps"]:
                    step = plan.get_step_by_id(modification["step_id"])
                    if step:
                        for key, value in modification.get("changes", {}).items():
                            setattr(step, key, value)
            
            logger.info("✅ Plan adapted successfully")
            return plan
            
        except Exception as e:
            logger.error(f"❌ Plan adaptation failed: {e}")
            return plan