"""
Execution Models - Data structures for planning and progress tracking
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import uuid


class StepStatus(str, Enum):
    """Status of an execution step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(str, Enum):
    """Priority levels for tasks"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class TaskComplexity(str, Enum):
    """Complexity levels for task estimation"""
    SIMPLE = "simple"      # Single tool call
    MODERATE = "moderate"  # 2-3 tool calls, no dependencies
    COMPLEX = "complex"    # Multiple tool calls with dependencies
    ADVANCED = "advanced"  # Multi-step analysis with complex logic


class ExecutionStep(BaseModel):
    """Single step in an execution plan"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)  # Step IDs this depends on
    expected_output: str = ""
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2


class ExecutionPlan(BaseModel):
    """Complete execution plan for a user request"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    description: str
    complexity: TaskComplexity
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_duration_minutes: int = 5
    steps: List[ExecutionStep] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: StepStatus = StepStatus.PENDING
    
    def get_pending_steps(self) -> List[ExecutionStep]:
        """Get all pending steps that can be executed (dependencies met)"""
        pending_steps = []
        completed_step_ids = {step.id for step in self.steps if step.status == StepStatus.COMPLETED}
        completed_step_names = {step.name for step in self.steps if step.status == StepStatus.COMPLETED}
        
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                # Check if all dependencies are completed
                # Dependencies can be either step IDs or step names
                dependencies_met = True
                for dep in step.dependencies:
                    if dep not in completed_step_ids and dep not in completed_step_names:
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    pending_steps.append(step)
        
        return pending_steps
    
    def get_step_by_id(self, step_id: str) -> Optional[ExecutionStep]:
        """Get step by ID"""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def update_step_status(self, step_id: str, status: StepStatus, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """Update step status and result"""
        step = self.get_step_by_id(step_id)
        if step:
            step.status = status
            if status == StepStatus.IN_PROGRESS:
                step.started_at = datetime.now()
            elif status in [StepStatus.COMPLETED, StepStatus.FAILED]:
                step.completed_at = datetime.now()
            
            if result:
                step.result = result
            if error:
                step.error_message = error
    
    def calculate_progress(self) -> Dict[str, Any]:
        """Calculate overall progress statistics"""
        total_steps = len(self.steps)
        if total_steps == 0:
            return {"percentage": 0, "completed": 0, "total": 0, "pending": 0, "failed": 0}
        
        completed = sum(1 for step in self.steps if step.status == StepStatus.COMPLETED)
        failed = sum(1 for step in self.steps if step.status == StepStatus.FAILED)
        pending = sum(1 for step in self.steps if step.status == StepStatus.PENDING)
        in_progress = sum(1 for step in self.steps if step.status == StepStatus.IN_PROGRESS)
        
        percentage = (completed / total_steps) * 100
        
        return {
            "percentage": round(percentage, 1),
            "completed": completed,
            "total": total_steps,
            "pending": pending,
            "in_progress": in_progress,
            "failed": failed,
            "is_complete": completed == total_steps,
            "has_failures": failed > 0
        }


class ReasoningResult(BaseModel):
    """Result of initial reasoning phase"""
    user_request: str
    understanding: str
    goal_analysis: str
    complexity_assessment: TaskComplexity
    available_tools_analysis: str
    approach_strategy: str
    potential_challenges: List[str] = Field(default_factory=list)
    success_definition: List[str] = Field(default_factory=list)
    estimated_steps: int = 1
    recommended_model: str = "standard"  # standard, advanced, or expert


class PlanningContext(BaseModel):
    """Context for planning phase"""
    reasoning_result: ReasoningResult
    available_tools: Dict[str, Dict[str, Any]]
    tool_categories: Dict[str, Dict[str, Any]]
    previous_plans: List[ExecutionPlan] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)


class ProgressReport(BaseModel):
    """Progress report for a specific iteration"""
    iteration: int
    plan_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    progress_summary: Dict[str, Any]
    completed_steps: List[str] = Field(default_factory=list)
    current_step: Optional[str] = None
    next_steps: List[str] = Field(default_factory=list)
    issues_encountered: List[str] = Field(default_factory=list)
    adaptations_made: List[str] = Field(default_factory=list)
    reflection_notes: str = ""
    estimated_completion: Optional[datetime] = None