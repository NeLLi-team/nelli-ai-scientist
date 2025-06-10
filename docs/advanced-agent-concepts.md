# Advanced Agent Concepts for NeLLi AI Scientists

Sophisticated patterns for building self-evolving, iterative AI agents using the Universal MCP Agent architecture with FastMCP integration.

## 🧠 Self-Evolving Agent Systems

### Reflective Learning Architecture

The NeLLi AI Scientist Agent Template provides a foundation for building agents that learn and adapt from their interactions:

```python
# agents/advanced/src/evolving_agent.py
from agents.template.src.agent import UniversalMCPAgent
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List

class EvolvingScientistAgent(UniversalMCPAgent):
    """Self-evolving agent that learns from scientific workflows"""
    
    def __init__(self, config):
        super().__init__(config)
        self.learning_memory = []
        self.success_patterns = {}
        self.failure_patterns = {}
        self.hypothesis_history = []
    
    async def evolve_from_interaction(self, 
                                    user_request: str, 
                                    tool_results: List[Dict], 
                                    user_feedback: str = None):
        """Learn from each interaction to improve future performance"""
        
        # Analyze interaction patterns
        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "request_type": self._classify_request(user_request),
            "tools_used": [r.get("tool") for r in tool_results],
            "success_indicators": self._extract_success_metrics(tool_results),
            "user_feedback": user_feedback
        }
        
        # Store in learning memory
        self.learning_memory.append(interaction_data)
        
        # Update success/failure patterns
        await self._update_patterns(interaction_data)
        
        # Generate adaptive prompts
        await self._adapt_prompts_from_experience()
    
    async def _update_patterns(self, interaction_data: Dict[str, Any]):
        """Update learned patterns based on interaction success"""
        request_type = interaction_data["request_type"]
        tools_used = interaction_data["tools_used"]
        success_score = self._calculate_success_score(interaction_data)
        
        # Track successful tool combinations
        if success_score > 0.7:
            if request_type not in self.success_patterns:
                self.success_patterns[request_type] = {}
            
            tool_combination = tuple(sorted(tools_used))
            if tool_combination not in self.success_patterns[request_type]:
                self.success_patterns[request_type][tool_combination] = []
            
            self.success_patterns[request_type][tool_combination].append({
                "success_score": success_score,
                "timestamp": interaction_data["timestamp"]
            })
    
    async def suggest_optimal_tools(self, user_request: str) -> List[str]:
        """Suggest tools based on learned patterns"""
        request_type = self._classify_request(user_request)
        
        if request_type in self.success_patterns:
            # Find most successful tool combinations
            best_combinations = []
            for tool_combo, results in self.success_patterns[request_type].items():
                avg_success = sum(r["success_score"] for r in results) / len(results)
                best_combinations.append((tool_combo, avg_success))
            
            # Sort by success rate
            best_combinations.sort(key=lambda x: x[1], reverse=True)
            
            if best_combinations:
                return list(best_combinations[0][0])
        
        # Fallback to standard tool selection
        analysis = await self.process_natural_language(user_request)
        return [tool["tool_name"] for tool in analysis.get("suggested_tools", [])]
```

### Dynamic Prompt Evolution

```python
class AdaptivePromptManager:
    """Prompts that evolve based on agent performance"""
    
    def __init__(self, base_prompt_manager):
        self.base_manager = base_prompt_manager
        self.prompt_variants = {}
        self.performance_history = {}
    
    async def evolve_prompt(self, prompt_name: str, performance_data: Dict[str, Any]):
        """Evolve prompts based on performance feedback"""
        
        if prompt_name not in self.prompt_variants:
            self.prompt_variants[prompt_name] = []
        
        # Generate prompt variations using LLM
        current_prompt = self.base_manager.load_prompt(prompt_name)
        
        evolution_request = f'''
        Improve this agent prompt based on performance data:
        
        Current Prompt:
        {current_prompt}
        
        Performance Issues:
        - Low success rate: {performance_data.get("success_rate", 0)}
        - Common failures: {performance_data.get("failure_modes", [])}
        - User feedback: {performance_data.get("feedback", "")}
        
        Generate an improved version that addresses these issues.
        '''
        
        # Use LLM to generate improved prompt
        improved_prompt = await self._generate_improved_prompt(evolution_request)
        
        # Test new prompt variant
        variant_id = len(self.prompt_variants[prompt_name])
        self.prompt_variants[prompt_name].append({
            "id": variant_id,
            "prompt": improved_prompt,
            "created": datetime.now().isoformat(),
            "performance": {}
        })
        
        return variant_id
    
    async def select_best_prompt(self, prompt_name: str) -> str:
        """Select the best-performing prompt variant"""
        if prompt_name not in self.prompt_variants:
            return self.base_manager.load_prompt(prompt_name)
        
        # Find variant with best performance
        best_variant = max(
            self.prompt_variants[prompt_name],
            key=lambda v: v["performance"].get("success_rate", 0)
        )
        
        return best_variant["prompt"]
```

## 🔬 Iterative Research Workflows

### Hypothesis-Driven Investigation

```python
class ResearchAgent(EvolvingScientistAgent):
    """Agent that conducts iterative scientific research"""
    
    async def conduct_research_cycle(self, research_question: str) -> Dict[str, Any]:
        """Complete research cycle: hypothesis → experiment → analysis → iteration"""
        
        research_session = {
            "question": research_question,
            "hypotheses": [],
            "experiments": [],
            "findings": [],
            "iterations": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Initial hypothesis generation
        initial_hypotheses = await self._generate_hypotheses(research_question)
        research_session["hypotheses"].extend(initial_hypotheses)
        
        # Iterative research loop
        max_iterations = 5
        while research_session["iterations"] < max_iterations:
            
            # Design experiment for current hypothesis
            current_hypothesis = research_session["hypotheses"][-1]
            experiment = await self._design_experiment(current_hypothesis)
            
            # Execute experiment using available tools
            results = await self._execute_experiment(experiment)
            experiment["results"] = results
            research_session["experiments"].append(experiment)
            
            # Analyze results and generate insights
            analysis = await self._analyze_experimental_results(results, current_hypothesis)
            research_session["findings"].append(analysis)
            
            # Determine if further investigation is needed
            should_continue = await self._evaluate_research_progress(research_session)
            
            if not should_continue:
                break
            
            # Generate new hypotheses based on findings
            new_hypotheses = await self._generate_follow_up_hypotheses(
                research_session["findings"]
            )
            research_session["hypotheses"].extend(new_hypotheses)
            research_session["iterations"] += 1
        
        # Generate final research report
        final_report = await self._generate_research_report(research_session)
        research_session["final_report"] = final_report
        
        return research_session
    
    async def _generate_hypotheses(self, research_question: str) -> List[Dict[str, Any]]:
        """Generate testable hypotheses for a research question"""
        
        prompt = f'''
        Generate 3-5 testable hypotheses for this research question: {research_question}
        
        Each hypothesis should:
        - Be specific and measurable
        - Suggest experimental approaches
        - Build on existing scientific knowledge
        - Be feasible with available bioinformatics tools
        
        Return as JSON list of hypothesis objects with fields:
        - hypothesis: Clear statement
        - rationale: Scientific reasoning
        - predicted_outcome: Expected result
        - experimental_approach: How to test it
        '''
        
        response = await self.llm.generate(prompt)
        return json.loads(response)
    
    async def _design_experiment(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Design experiment to test a hypothesis"""
        
        # Determine which tools are needed
        tools_context = self._build_tools_context()
        
        design_prompt = f'''
        Design an experiment to test this hypothesis:
        {json.dumps(hypothesis, indent=2)}
        
        Available tools:
        {tools_context}
        
        Return experiment design as JSON with:
        - objective: Clear experimental objective
        - methodology: Step-by-step approach
        - tools_needed: List of required tools
        - parameters: Tool parameters and settings
        - expected_outputs: What results to expect
        - success_criteria: How to measure success
        '''
        
        response = await self.llm.generate(design_prompt)
        return json.loads(response)
    
    async def _execute_experiment(self, experiment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the experimental design using available tools"""
        
        results = []
        
        for step in experiment.get("methodology", []):
            if "tool_name" in step:
                # Execute tool with specified parameters
                result = await self._call_mcp_tool(
                    step["tool_name"], 
                    **step.get("parameters", {})
                )
                results.append({
                    "step": step["description"],
                    "tool": step["tool_name"],
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    async def _analyze_experimental_results(self, 
                                          results: List[Dict[str, Any]], 
                                          hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experimental results in context of hypothesis"""
        
        analysis_prompt = f'''
        Analyze these experimental results in the context of the hypothesis:
        
        Hypothesis: {json.dumps(hypothesis, indent=2)}
        
        Results: {json.dumps(results, indent=2)}
        
        Provide analysis including:
        - hypothesis_supported: boolean
        - confidence_level: 0-1 scale
        - key_findings: List of important discoveries
        - statistical_significance: If applicable
        - limitations: Experimental limitations
        - implications: Scientific implications
        - future_directions: Suggested follow-up research
        '''
        
        response = await self.llm.generate(analysis_prompt)
        return json.loads(response)
```

## 🤖 Multi-Agent Collaboration

### Specialized Agent Orchestra

```python
class AgentOrchestrator:
    """Coordinate multiple specialized agents for complex research"""
    
    def __init__(self):
        self.agents = {
            "data_collector": DataCollectionAgent(),
            "sequence_analyst": SequenceAnalysisAgent(), 
            "statistician": StatisticalAnalysisAgent(),
            "literature_researcher": LiteratureAgent(),
            "hypothesis_generator": HypothesisAgent()
        }
        self.collaboration_history = []
    
    async def collaborative_research(self, research_objective: str) -> Dict[str, Any]:
        """Orchestrate collaborative research between specialized agents"""
        
        session = {
            "objective": research_objective,
            "phases": [],
            "start_time": datetime.now().isoformat()
        }
        
        # Phase 1: Data Collection
        data_phase = await self._execute_phase(
            "data_collection",
            research_objective,
            ["data_collector", "literature_researcher"]
        )
        session["phases"].append(data_phase)
        
        # Phase 2: Analysis
        analysis_phase = await self._execute_phase(
            "analysis", 
            research_objective,
            ["sequence_analyst", "statistician"],
            previous_results=data_phase["results"]
        )
        session["phases"].append(analysis_phase)
        
        # Phase 3: Hypothesis Generation
        hypothesis_phase = await self._execute_phase(
            "hypothesis_generation",
            research_objective, 
            ["hypothesis_generator"],
            previous_results=analysis_phase["results"]
        )
        session["phases"].append(hypothesis_phase)
        
        # Phase 4: Synthesis
        synthesis = await self._synthesize_results(session)
        session["synthesis"] = synthesis
        
        return session
    
    async def _execute_phase(self, 
                           phase_name: str,
                           objective: str, 
                           agent_names: List[str],
                           previous_results: Dict = None) -> Dict[str, Any]:
        """Execute a research phase with specified agents"""
        
        phase_results = {
            "phase": phase_name,
            "agents": agent_names,
            "start_time": datetime.now().isoformat(),
            "agent_results": {}
        }
        
        # Execute agents in parallel
        tasks = []
        for agent_name in agent_names:
            agent = self.agents[agent_name]
            task = agent.contribute_to_research(objective, previous_results)
            tasks.append((agent_name, task))
        
        # Gather results
        for agent_name, task in tasks:
            try:
                result = await task
                phase_results["agent_results"][agent_name] = result
            except Exception as e:
                phase_results["agent_results"][agent_name] = {"error": str(e)}
        
        phase_results["end_time"] = datetime.now().isoformat()
        return phase_results
```

## 🧬 Autonomous Hypothesis Testing

### Self-Directed Scientific Discovery

```python
class AutonomousDiscoveryAgent(ResearchAgent):
    """Agent that autonomously discovers scientific insights"""
    
    async def autonomous_discovery_loop(self, 
                                      domain: str, 
                                      duration_hours: int = 24) -> Dict[str, Any]:
        """Run autonomous discovery loop for specified duration"""
        
        discovery_session = {
            "domain": domain,
            "duration_hours": duration_hours,
            "discoveries": [],
            "hypotheses_tested": 0,
            "insights_generated": 0,
            "start_time": datetime.now().isoformat()
        }
        
        end_time = datetime.now().timestamp() + (duration_hours * 3600)
        
        while datetime.now().timestamp() < end_time:
            
            # Generate novel research question
            research_question = await self._generate_novel_question(domain)
            
            # Conduct mini research cycle
            research_results = await self.conduct_research_cycle(research_question)
            
            # Evaluate novelty and significance
            significance = await self._evaluate_significance(research_results)
            
            if significance["is_significant"]:
                discovery = {
                    "question": research_question,
                    "results": research_results,
                    "significance_score": significance["score"],
                    "discovery_type": significance["type"],
                    "timestamp": datetime.now().isoformat()
                }
                discovery_session["discoveries"].append(discovery)
                discovery_session["insights_generated"] += 1
            
            discovery_session["hypotheses_tested"] += len(research_results["hypotheses"])
            
            # Brief pause before next cycle
            await asyncio.sleep(300)  # 5 minutes
        
        # Generate final discovery report
        final_report = await self._generate_discovery_report(discovery_session)
        discovery_session["final_report"] = final_report
        
        return discovery_session
    
    async def _generate_novel_question(self, domain: str) -> str:
        """Generate novel research questions in the domain"""
        
        # Analyze current knowledge base
        current_knowledge = await self._analyze_knowledge_base(domain)
        
        # Identify knowledge gaps
        gaps = await self._identify_knowledge_gaps(current_knowledge)
        
        # Generate question targeting biggest gap
        question_prompt = f'''
        Generate a novel, answerable research question in {domain}.
        
        Current knowledge summary:
        {json.dumps(current_knowledge, indent=2)}
        
        Identified gaps:
        {json.dumps(gaps, indent=2)}
        
        Requirements:
        - Address a significant knowledge gap
        - Be answerable with available bioinformatics tools
        - Have potential for meaningful biological insights
        - Not duplicate previous research questions
        
        Return just the research question as a string.
        '''
        
        return await self.llm.generate(question_prompt)
```

## 🔄 Continuous Learning Integration

### Memory-Enhanced Learning

```python
class MemoryEnhancedAgent(AutonomousDiscoveryAgent):
    """Agent with sophisticated memory and learning capabilities"""
    
    def __init__(self, config):
        super().__init__(config)
        self.episodic_memory = []      # Specific experiences
        self.semantic_memory = {}      # General knowledge
        self.procedural_memory = {}    # Learned procedures
        self.meta_memory = {}          # Knowledge about knowledge
    
    async def learn_from_experience(self, experience: Dict[str, Any]):
        """Learn from each experience and update memory systems"""
        
        # Store episodic memory
        episode = {
            "timestamp": datetime.now().isoformat(),
            "experience": experience,
            "context": await self._extract_context(experience),
            "outcomes": await self._extract_outcomes(experience)
        }
        self.episodic_memory.append(episode)
        
        # Update semantic memory (general knowledge)
        await self._update_semantic_memory(experience)
        
        # Update procedural memory (learned procedures)
        await self._update_procedural_memory(experience)
        
        # Update meta-memory (knowledge about learning)
        await self._update_meta_memory(experience)
    
    async def retrieve_relevant_memories(self, current_context: str) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to current context"""
        
        relevant_memories = []
        
        # Semantic similarity search through episodic memory
        for episode in self.episodic_memory[-100:]:  # Recent episodes
            similarity = await self._calculate_similarity(
                current_context, 
                episode["context"]
            )
            
            if similarity > 0.7:
                relevant_memories.append({
                    "memory": episode,
                    "similarity": similarity,
                    "relevance_reason": await self._explain_relevance(
                        current_context, episode
                    )
                })
        
        # Sort by relevance
        relevant_memories.sort(key=lambda x: x["similarity"], reverse=True)
        
        return relevant_memories[:5]  # Top 5 most relevant
    
    async def apply_learned_procedures(self, task_context: str) -> Dict[str, Any]:
        """Apply previously learned procedures to new tasks"""
        
        # Find applicable procedures
        applicable_procedures = []
        
        for proc_name, procedure in self.procedural_memory.items():
            applicability = await self._assess_procedure_applicability(
                procedure, task_context
            )
            
            if applicability["is_applicable"]:
                applicable_procedures.append({
                    "name": proc_name,
                    "procedure": procedure,
                    "applicability_score": applicability["score"],
                    "adaptation_needed": applicability["adaptations"]
                })
        
        # Select best procedure
        if applicable_procedures:
            best_procedure = max(
                applicable_procedures, 
                key=lambda x: x["applicability_score"]
            )
            
            # Adapt procedure for current context
            adapted_procedure = await self._adapt_procedure(
                best_procedure["procedure"],
                task_context,
                best_procedure["adaptation_needed"]
            )
            
            return {
                "procedure_found": True,
                "original_procedure": best_procedure["procedure"],
                "adapted_procedure": adapted_procedure,
                "confidence": best_procedure["applicability_score"]
            }
        
        return {"procedure_found": False}
```

## 🚀 Implementation Guidelines

### Building Your Advanced Agent

1. **Start with the Universal Agent Base**:
   ```python
   from agents.template.src.agent import UniversalMCPAgent
   
   class MyAdvancedAgent(UniversalMCPAgent):
       def __init__(self, config):
           super().__init__(config)
           # Add your advanced capabilities
   ```

2. **Implement Incremental Learning**:
   - Start with simple pattern recognition
   - Add memory systems gradually
   - Implement feedback loops

3. **Use FastMCP for Tool Extension**:
   - Create specialized MCP servers for advanced capabilities
   - Implement async patterns for concurrent processing
   - Use tool chaining for complex workflows

4. **Leverage External Prompts**:
   - Store learning-enhanced prompts externally
   - Implement A/B testing for prompt variants
   - Use evolutionary prompt optimization

### Performance Considerations

- **Memory Management**: Implement memory pruning for long-running agents
- **Async Patterns**: Use concurrent processing for independent operations  
- **Resource Limits**: Set bounds on autonomous exploration
- **Evaluation Metrics**: Define clear success metrics for learning

### Ethical Considerations

- **Autonomous Boundaries**: Define clear limits for autonomous operation
- **Human Oversight**: Implement checkpoints for human review
- **Transparency**: Log all agent decisions and reasoning
- **Safety Measures**: Implement safeguards against harmful outputs

The NeLLi AI Scientist Agent Template provides the foundation for building these advanced capabilities. The Universal MCP Agent architecture, combined with FastMCP's async capabilities, enables sophisticated AI systems that can evolve, learn, and conduct autonomous scientific research while maintaining safety and transparency.

## 💡 Next Steps for Advanced Development

1. **Implement Memory Systems**: Start with episodic memory for experience storage
2. **Add Learning Loops**: Implement feedback-based prompt evolution
3. **Create Specialized MCPs**: Build domain-specific tools for your research area
4. **Design Evaluation Metrics**: Define success criteria for autonomous operation
5. **Build Safety Measures**: Implement oversight and control mechanisms

The combination of the Universal MCP Agent's flexibility with these advanced patterns enables building truly sophisticated AI scientist agents capable of autonomous discovery and continuous improvement.