"""
Progress Tracker - Tracks execution progress and generates reports
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .execution_models import (
    ExecutionPlan, ExecutionStep, StepStatus, ProgressReport
)

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks task execution progress and generates detailed reports"""
    
    def __init__(self, reports_dir: str = "../../reports/sophisticated_agent"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.current_plans: Dict[str, ExecutionPlan] = {}
        self.progress_history: List[ProgressReport] = []
        
    def start_plan(self, plan: ExecutionPlan) -> str:
        """Start tracking a new execution plan"""
        plan.started_at = datetime.now()
        plan.status = StepStatus.IN_PROGRESS
        self.current_plans[plan.id] = plan
        
        logger.info(f"📊 Started tracking plan: {plan.goal}")
        
        # Create initial plan document
        self._save_plan_document(plan)
        
        return plan.id
    
    def update_step_progress(self, plan_id: str, step_id: str, status: StepStatus, 
                           result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """Update the status of a specific step"""
        if plan_id not in self.current_plans:
            logger.error(f"Plan {plan_id} not found")
            return
        
        plan = self.current_plans[plan_id]
        plan.update_step_status(step_id, status, result, error)
        
        # Log progress update
        step = plan.get_step_by_id(step_id)
        if step:
            if status == StepStatus.COMPLETED:
                logger.info(f"✅ Completed step: {step.name}")
            elif status == StepStatus.FAILED:
                logger.error(f"❌ Failed step: {step.name} - {error}")
            elif status == StepStatus.IN_PROGRESS:
                logger.info(f"🔄 Started step: {step.name}")
        
        # Update plan document
        self._save_plan_document(plan)
    
    def generate_progress_report(self, plan_id: str, iteration: int, reflection_notes: str = "") -> ProgressReport:
        """Generate a detailed progress report"""
        if plan_id not in self.current_plans:
            raise ValueError(f"Plan {plan_id} not found")
        
        plan = self.current_plans[plan_id]
        progress_stats = plan.calculate_progress()
        
        # Get completed steps
        completed_steps = [step.name for step in plan.steps if step.status == StepStatus.COMPLETED]
        
        # Get current step
        current_step = None
        for step in plan.steps:
            if step.status == StepStatus.IN_PROGRESS:
                current_step = step.name
                break
        
        # Get next steps
        pending_steps = plan.get_pending_steps()
        next_steps = [step.name for step in pending_steps[:3]]  # Next 3 steps
        
        # Identify issues
        issues = []
        adaptations = []
        for step in plan.steps:
            if step.status == StepStatus.FAILED:
                issues.append(f"Step '{step.name}' failed: {step.error_message}")
            if step.retry_count > 0:
                adaptations.append(f"Retried step '{step.name}' {step.retry_count} times")
        
        # Estimate completion
        estimated_completion = None
        if progress_stats["percentage"] > 0:
            elapsed = datetime.now() - plan.started_at if plan.started_at else timedelta(0)
            if elapsed.total_seconds() > 0:
                total_estimated = elapsed / (progress_stats["percentage"] / 100)
                estimated_completion = plan.started_at + total_estimated
        
        report = ProgressReport(
            iteration=iteration,
            plan_id=plan_id,
            progress_summary=progress_stats,
            completed_steps=completed_steps,
            current_step=current_step,
            next_steps=next_steps,
            issues_encountered=issues,
            adaptations_made=adaptations,
            reflection_notes=reflection_notes,
            estimated_completion=estimated_completion
        )
        
        self.progress_history.append(report)
        
        # Save progress report
        self._save_progress_report(report)
        
        return report
    
    def get_checklist_display(self, plan_id: str) -> List[Dict[str, Any]]:
        """Get checklist format for display in chat interface"""
        if plan_id not in self.current_plans:
            return []
        
        plan = self.current_plans[plan_id]
        checklist = []
        
        for step in plan.steps:
            item = {
                "name": step.name,
                "description": step.description,
                "status": step.status.value,
                "is_completed": step.status == StepStatus.COMPLETED,
                "is_failed": step.status == StepStatus.FAILED,
                "is_in_progress": step.status == StepStatus.IN_PROGRESS,
                "expected_output": step.expected_output
            }
            checklist.append(item)
        
        return checklist
    
    def display_colored_progress(self, plan_id: str) -> str:
        """Generate colored terminal output for progress display"""
        if plan_id not in self.current_plans:
            return "❌ Plan not found"
        
        plan = self.current_plans[plan_id]
        progress_stats = plan.calculate_progress()
        
        # Build colored display
        lines = []
        
        # Header with progress bar
        lines.append(f"\n\033[1m📋 Task Progress: {plan.goal}\033[0m")
        lines.append(f"\033[36m{'='*60}\033[0m")
        
        # Progress bar
        completed = progress_stats["completed"]
        total = progress_stats["total"]
        percentage = progress_stats["percentage"]
        
        bar_width = 30
        filled = int((completed / total) * bar_width) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        lines.append(f"\033[36mProgress: [{bar}] {percentage}% ({completed}/{total})\033[0m")
        
        # Step checklist
        lines.append(f"\n\033[1m📝 Step Checklist:\033[0m")
        
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED:
                lines.append(f"  \033[32m✅ {step.name}\033[0m")
            elif step.status == StepStatus.FAILED:
                lines.append(f"  \033[31m❌ {step.name}\033[0m \033[90m({step.error_message})\033[0m")
            elif step.status == StepStatus.IN_PROGRESS:
                lines.append(f"  \033[33m🔄 {step.name}\033[0m \033[90m(in progress...)\033[0m")
            else:
                lines.append(f"  \033[90m⏳ {step.name}\033[0m")
        
        # Summary
        if progress_stats["has_failures"]:
            lines.append(f"\n\033[31m⚠️  {progress_stats['failed']} step(s) failed\033[0m")
        
        if progress_stats["is_complete"]:
            lines.append(f"\n\033[32m🎉 All steps completed successfully!\033[0m")
        
        lines.append(f"\033[36m{'='*60}\033[0m")
        
        return "\n".join(lines)
    
    def complete_plan(self, plan_id: str):
        """Mark a plan as completed"""
        if plan_id not in self.current_plans:
            return
        
        plan = self.current_plans[plan_id]
        plan.completed_at = datetime.now()
        plan.status = StepStatus.COMPLETED
        
        logger.info(f"🎉 Plan completed: {plan.goal}")
        
        # Save final plan document
        self._save_plan_document(plan)
        
        # Generate final report
        final_report = self.generate_progress_report(plan_id, 999, "Final completion report")
        
    def _save_plan_document(self, plan: ExecutionPlan):
        """Save plan as markdown document"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plan_{timestamp}.md"
        filepath = self.reports_dir / filename
        
        # Create markdown content
        content = self._generate_plan_markdown(plan)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Also save as latest
        latest_path = self.reports_dir / "plan.md"
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _save_progress_report(self, report: ProgressReport):
        """Save progress report as markdown"""
        filename = f"progress_{report.iteration}.md"
        filepath = self.reports_dir / filename
        
        content = self._generate_progress_markdown(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _generate_plan_markdown(self, plan: ExecutionPlan) -> str:
        """Generate markdown content for execution plan"""
        progress_stats = plan.calculate_progress()
        
        content = f"""# Execution Plan: {plan.goal}

**Plan ID:** {plan.id}
**Complexity:** {plan.complexity.value}
**Priority:** {plan.priority.value}
**Status:** {plan.status.value}

## Description
{plan.description}

## Progress Overview
- **Overall Progress:** {progress_stats['percentage']}% ({progress_stats['completed']}/{progress_stats['total']})
- **Completed Steps:** {progress_stats['completed']}
- **Failed Steps:** {progress_stats['failed']}
- **Remaining Steps:** {progress_stats['pending']}

## Success Criteria
"""
        
        for criterion in plan.success_criteria:
            content += f"- {criterion}\n"
        
        content += "\n## Step Details\n\n"
        
        for i, step in enumerate(plan.steps, 1):
            status_emoji = {
                StepStatus.PENDING: "⏳",
                StepStatus.IN_PROGRESS: "🔄", 
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️"
            }.get(step.status, "❓")
            
            content += f"### {i}. {status_emoji} {step.name}\n\n"
            content += f"**Status:** {step.status.value}\n"
            content += f"**Tool:** {step.tool_name}\n"
            content += f"**Description:** {step.description}\n"
            
            if step.parameters:
                content += f"**Parameters:** `{json.dumps(step.parameters, indent=2)}`\n"
            
            if step.dependencies:
                content += f"**Dependencies:** {', '.join(step.dependencies)}\n"
            
            if step.expected_output:
                content += f"**Expected Output:** {step.expected_output}\n"
            
            if step.started_at:
                content += f"**Started:** {step.started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if step.completed_at:
                content += f"**Completed:** {step.completed_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if step.error_message:
                content += f"**Error:** {step.error_message}\n"
            
            if step.result:
                content += f"**Result:** `{json.dumps(step.result, indent=2)}`\n"
            
            content += "\n"
        
        # Add timeline
        content += "\n## Timeline\n\n"
        if plan.created_at:
            content += f"- **Created:** {plan.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        if plan.started_at:
            content += f"- **Started:** {plan.started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        if plan.completed_at:
            content += f"- **Completed:** {plan.completed_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return content
    
    def _generate_progress_markdown(self, report: ProgressReport) -> str:
        """Generate markdown content for progress report"""
        content = f"""# Progress Report - Iteration {report.iteration}

**Timestamp:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Plan ID:** {report.plan_id}

## Progress Summary
- **Overall Progress:** {report.progress_summary['percentage']}%
- **Completed Steps:** {report.progress_summary['completed']}/{report.progress_summary['total']}
- **Failed Steps:** {report.progress_summary['failed']}
- **Steps In Progress:** {report.progress_summary['in_progress']}

## Completed Steps
"""
        
        for step in report.completed_steps:
            content += f"- ✅ {step}\n"
        
        if report.current_step:
            content += f"\n## Current Step\n🔄 {report.current_step}\n"
        
        if report.next_steps:
            content += "\n## Next Steps\n"
            for step in report.next_steps:
                content += f"- ⏳ {step}\n"
        
        if report.issues_encountered:
            content += "\n## Issues Encountered\n"
            for issue in report.issues_encountered:
                content += f"- ❌ {issue}\n"
        
        if report.adaptations_made:
            content += "\n## Adaptations Made\n"
            for adaptation in report.adaptations_made:
                content += f"- 🔄 {adaptation}\n"
        
        if report.reflection_notes:
            content += f"\n## Reflection Notes\n{report.reflection_notes}\n"
        
        if report.estimated_completion:
            content += f"\n## Estimated Completion\n{report.estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return content