# Enhanced Universal MCP Agent Features

The Enhanced Universal MCP Agent extends the original agent with sophisticated planning and progress tracking capabilities.

## 🆕 New Features Overview

### 1. 🧠 Initial Reasoning Phase
- **Different Model**: Uses advanced reasoning to deeply understand user requests
- **Context Analysis**: Analyzes request in context of available tools and capabilities
- **Complexity Assessment**: Automatically determines task complexity (simple/moderate/complex/advanced)
- **Strategic Planning**: Identifies potential challenges and success criteria

### 2. 📋 Execution Planning
- **Detailed Plans**: Creates step-by-step execution plans with dependencies
- **Tool Orchestration**: Intelligently sequences tool calls for optimal results
- **Parameter Chaining**: Automatically passes results between tools using `ANALYSIS_RESULTS`
- **Adaptive Planning**: Plans adapt based on execution results and failures

### 3. 📊 Progress Tracking & Reporting
- **Real-time Progress**: Visual progress tracking with colored terminal output
- **Step Checklist**: Live checklist showing completed (✅), failed (❌), and pending (⏳) steps
- **Markdown Reports**: Automatic generation of detailed progress reports
- **Plan Documentation**: Complete execution plans saved to `reports/sophisticated_agent/plan.md`
- **Progress Reports**: Step-by-step progress saved to `reports/sophisticated_agent/progress_N.md`

### 4. 🔄 Self-Reflection & Adaptation
- **Step Reflection**: AI-generated insights after each step completion
- **Plan Adaptation**: Automatically adapts plans when steps fail or conditions change
- **Final Analysis**: Comprehensive reflection on overall workflow effectiveness
- **Continuous Learning**: Learns from execution patterns to improve future plans

## 🔧 Architecture Components

### Core Modules

1. **`execution_models.py`** - Data structures for plans, steps, and progress
2. **`task_planner.py`** - Creates and adapts execution plans using AI reasoning
3. **`progress_tracker.py`** - Tracks progress and generates reports
4. **`enhanced_agent.py`** - Main enhanced agent with integrated planning

### New Prompts

- **`reasoning.txt`** - Deep analysis and understanding of user requests
- **`planning.txt`** - Detailed execution plan creation
- **`plan_adaptation.txt`** - Plan modification based on results

## 🚀 Usage Examples

### Basic Usage
```bash
# Run the enhanced agent with default models
python run_enhanced.py

# Or with pixi
pixi run sophisticated-agent

# With specific configuration
python run_enhanced.py --name "my-agent" --llm-provider cborg

# With custom models for reasoning and planning
python run_enhanced.py --reasoning-model "google/gemini-pro" --planning-model "google/gemini-flash-lite"

# Use more powerful models for complex tasks
python run_enhanced.py --reasoning-model "anthropic/claude-3-5-sonnet" --planning-model "google/gemini-pro"
```

### Disable Features (if needed)
```bash
# Disable reasoning (uses simple processing)
python run_enhanced.py --disable-reasoning

# Disable planning (direct tool execution)
python run_enhanced.py --disable-planning

# Disable progress tracking
python run_enhanced.py --disable-progress
```

## 📋 Chat Interface Commands

Enhanced commands in the terminal chat interface:

- **`help`** - Show enhanced help with planning features
- **`tools`** - List available MCP tools
- **`progress`** - Show current execution progress (during active plan)
- **`clear`** - Clear screen and show welcome
- **`quit`** - Exit the agent

## 🎯 Workflow Example

When you ask: *"analyze the mimivirus genome file and create a summary report"*

### 1. 🧠 Reasoning Phase
```
🧠 Initial Reasoning Phase...
📊 Complexity Assessment: complex
🎯 Goal: Comprehensive genomic analysis with structured reporting
```

### 2. 📋 Planning Phase
```
📋 Execution Planning Phase...
📊 Created plan with 4 steps

📋 Execution Plan Overview
==================================================
🎯 Goal: analyze the mimivirus genome file and create a summary report
📊 Complexity: complex
⏱️ Estimated Duration: 8 minutes
📝 Steps: 4

📋 Step Checklist:
  1. ⏳ Create reports directory
  2. ⏳ Analyze FASTA file for genomic features
  3. ⏳ Generate comprehensive analysis report
  4. ⏳ Save results as JSON report
==================================================
```

### 3. 🚀 Execution Phase
```
🚀 Executing Planned Workflow...

🔧 Executing: Create reports directory
✅ Step completed successfully

🔧 Executing: Analyze FASTA file for genomic features
✅ Step completed successfully

📋 Task Progress: analyze the mimivirus genome file and create a summary report
============================================================
Progress: [████████████████████████████░░] 75.0% (3/4)

📝 Step Checklist:
  ✅ Create reports directory
  ✅ Analyze FASTA file for genomic features
  ✅ Generate comprehensive analysis report
  ⏳ Save results as JSON report
============================================================
```

### 4. 🔍 Final Analysis
```
💡 Workflow Summary:
   ✅ 4 steps completed
   ❌ 0 steps failed

🔍 Final Analysis:
   The genomic analysis workflow completed successfully, providing comprehensive
   insights into the mimivirus genome structure, including gene count, GC content,
   and sequence statistics. All results have been saved to structured reports.
```

## 📁 Generated Reports

### Execution Plan (`plan.md`)
```markdown
# Execution Plan: analyze the mimivirus genome file and create a summary report

**Plan ID:** plan_20250110_143022
**Complexity:** complex
**Status:** completed

## Progress Overview
- **Overall Progress:** 100% (4/4)
- **Completed Steps:** 4
- **Failed Steps:** 0

## Step Details
### 1. ✅ Create reports directory
**Status:** completed
**Tool:** create_directory
**Started:** 2025-01-10 14:30:25
**Completed:** 2025-01-10 14:30:25
```

### Progress Report (`progress_1.md`)
```markdown
# Progress Report - Iteration 1

**Timestamp:** 2025-01-10 14:30:30
**Plan ID:** plan_20250110_143022

## Progress Summary
- **Overall Progress:** 50%
- **Completed Steps:** 2/4

## Completed Steps
- ✅ Create reports directory
- ✅ Analyze FASTA file for genomic features

## Current Step
🔄 Generate comprehensive analysis report
```

## 🎛️ Configuration Options

### YAML Configuration File
**Primary configuration:** `agents/sophisticated_agent/config/agent_config.yaml`

```yaml
agent:
  name: "nelli-enhanced-agent"
  description: "Enhanced Universal MCP Agent with Reasoning, Planning, and Progress Tracking"
  
  enhanced_features:
    reasoning:
      enabled: true
      model: "google/gemini-pro"           # More capable model for reasoning
      temperature: 0.3
      max_tokens: 2000
      
    planning:
      enabled: true  
      model: "google/gemini-flash-lite"    # Efficient model for planning
      temperature: 0.2
      max_tokens: 1500
      max_iterations: 5
      
    execution:
      model: "google/gemini-flash-lite"    # Fast model for tool coordination
      temperature: 0.1
      max_tokens: 1000
      
    progress_tracking:
      enabled: true
      reports_directory: "../../reports/sophisticated_agent"
      save_plans: true
      save_progress: true
      colored_output: true
```

### Available Models (CBORG API)
- **google/gemini-pro** - Most capable, best for reasoning
- **google/gemini-flash-lite** - Fast and efficient, good for planning
- **anthropic/claude-3-5-sonnet** - Very capable Claude model
- **anthropic/claude-3-5-haiku** - Fast Claude model
- **openai/gpt-4o** - Capable OpenAI model  
- **openai/gpt-4o-mini** - Fast OpenAI model

### Command Line Overrides
Command line arguments override YAML configuration:

```bash
# Use default models from config file
pixi run sophisticated-agent

# Override reasoning model for complex tasks
pixi run sophisticated-agent \
  --reasoning-model "anthropic/claude-3-5-sonnet"

# Use powerful models for both phases
pixi run sophisticated-agent \
  --reasoning-model "anthropic/claude-3-5-sonnet" \
  --planning-model "google/gemini-pro"

# Disable features
pixi run sophisticated-agent \
  --disable-reasoning \
  --disable-planning

# Custom agent name
pixi run sophisticated-agent \
  --name "my-custom-agent" \
  --reasoning-model "openai/gpt-4o"
```

### Configuration Priority
1. **Command line arguments** (highest priority)
2. **YAML configuration file** (default)
3. **Environment variables** (fallback)
4. **Hard-coded defaults** (lowest priority)

## 🔗 Integration with Bio Agent

The enhanced agent can be used as a drop-in replacement for the original agent, or the planning components can be integrated into domain-specific agents like the bio agent you showed earlier.

Key integration points:
- Use `TaskPlanner` for complex biological workflows
- Use `ProgressTracker` for long-running analyses
- Use `ExecutionPlan` models for structured biological pipelines

This enhanced architecture provides the sophisticated reasoning, planning, and tracking capabilities you requested while maintaining compatibility with the existing MCP ecosystem.