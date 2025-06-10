# Agent Customization Guide

Learn how to customize the NeLLi AI Scientist Agent to fit your specific research needs and domain expertise.

## 🎨 Customization Overview

The Universal MCP Agent architecture is designed for easy customization through:

- **External Prompts**: Modify behavior without changing code
- **MCP Server Integration**: Add domain-specific tools
- **LLM Provider Selection**: Choose the best LLM for your needs
- **Reflection Logic**: Customize how the agent analyzes results
- **Memory Systems**: Add learning and adaptation capabilities

## 🧠 Prompt Customization

### Understanding the Prompt System

```
agents/template/prompts/
├── tool_selection.txt      # How agent chooses tools
├── reflection.txt          # How agent analyzes results
├── general_response.txt    # General conversation handling
└── error_handling.txt      # Error situation responses
```

### Customizing Tool Selection

Edit `agents/template/prompts/tool_selection.txt`:

```text
You are a specialized {DOMAIN} AI scientist with access to various tools.

DOMAIN EXPERTISE: {DOMAIN_KNOWLEDGE}
- Deep understanding of {SPECIFIC_FIELD} principles
- Experience with {COMMON_WORKFLOWS}
- Knowledge of {IMPORTANT_METHODS}

TOOL SELECTION PRINCIPLES:
- For {DOMAIN} data analysis, prefer {PREFERRED_TOOLS}
- When working with {DATA_TYPE}, always use {SPECIFIC_TOOL_PATTERN}
- For {ANALYSIS_TYPE}, combine tools in this order: {TOOL_SEQUENCE}

RESPONSE STRATEGY:
- If the request is about general {DOMAIN} knowledge, provide direct answers
- If the request requires data analysis, use appropriate tools
- Always consider {DOMAIN_SPECIFIC_CONSTRAINTS}

RESPONSE FORMAT:
{
  "response_type": "direct_answer" OR "use_tools",
  "direct_answer": "your expert response (if response_type is direct_answer)",
  "intent": "what the user wants to accomplish",
  "domain_context": "relevant {DOMAIN} background",
  "suggested_tools": [
    {
      "tool_name": "exact_tool_name",
      "parameters": {"param1": "value1"},
      "reason": "why this tool helps with {DOMAIN} analysis"
    }
  ]
}
```

### Customizing Reflection Analysis

Edit `agents/template/prompts/reflection.txt`:

```text
You are analyzing results from {DOMAIN} research tools. Provide expert interpretation.

ANALYSIS FRAMEWORK:
1. {DOMAIN} Significance
   - What do these results mean in {SPECIFIC_FIELD} context?
   - How do they relate to known {DOMAIN} principles?
   - What patterns are biologically/scientifically relevant?

2. Methodological Assessment
   - Are the methods appropriate for the research question?
   - What are the limitations of this analysis?
   - What controls or validations might be needed?

3. Research Implications
   - What hypotheses do these results support or refute?
   - What follow-up experiments would be valuable?
   - How do these findings advance {DOMAIN} knowledge?

4. Clinical/Applied Relevance (if applicable)
   - What are the practical implications?
   - How might this impact {APPLICATION_AREA}?

RESPONSE STRUCTURE:
Provide analysis in this format:
- **{DOMAIN} Interpretation**: [Expert analysis of biological/scientific meaning]
- **Methodological Notes**: [Assessment of approach and limitations]
- **Research Implications**: [Significance for advancing knowledge]
- **Recommended Next Steps**: [Suggested follow-up analyses]
- **Confidence Level**: [High/Medium/Low with reasoning]
```

### Domain-Specific Prompt Examples

**Genomics Agent**:
```text
You are a genomics AI scientist specializing in:
- Sequence analysis and annotation
- Comparative genomics and phylogenetics  
- Variant analysis and interpretation
- Gene expression and regulation

GENOMICS TOOL PREFERENCES:
- For sequence analysis: sequence_stats, analyze_fasta_file
- For evolutionary studies: multiple_alignment, phylogenetic_tree
- For functional analysis: translate_sequence, blast_local
```

**Proteomics Agent**:
```text
You are a proteomics AI scientist specializing in:
- Protein structure and function analysis
- Mass spectrometry data interpretation
- Protein-protein interactions
- Post-translational modifications

PROTEOMICS WORKFLOWS:
- Always consider protein folding context
- Interpret mass spec data with statistical validation
- Consider PTM effects on protein function
```

**Systems Biology Agent**:
```text
You are a systems biology AI scientist specializing in:
- Network analysis and pathway modeling
- Multi-omics data integration
- Dynamic systems modeling
- Biomarker discovery

SYSTEMS APPROACHES:
- Integrate multiple data types when available
- Consider pathway-level effects
- Use network topology for interpretation
```

## 🔧 Custom MCP Servers for Your Domain

### Creating Domain-Specific Tools

```python
# mcps/my_domain/src/server.py
from fastmcp import FastMCP
import asyncio
from typing import List, Dict, Any

mcp = FastMCP("My Domain Tools 🔬")

@mcp.tool
async def domain_specific_analysis(
    data: str, 
    method: str = "standard",
    sensitivity: float = 0.05
) -> dict:
    """Perform domain-specific analysis with custom algorithms
    
    Args:
        data: Input data in domain-specific format
        method: Analysis method (standard, enhanced, custom)
        sensitivity: Statistical sensitivity threshold
    """
    
    # Your domain-specific logic here
    analysis_result = await perform_domain_analysis(data, method, sensitivity)
    
    # Return structured results with domain context
    return {
        "analysis_method": method,
        "sensitivity_threshold": sensitivity,
        "results": analysis_result,
        "domain_interpretation": await interpret_results(analysis_result),
        "quality_metrics": await calculate_quality_metrics(analysis_result),
        "recommendations": await generate_recommendations(analysis_result)
    }

@mcp.tool
async def comparative_analysis(
    datasets: List[str], 
    comparison_type: str = "pairwise"
) -> dict:
    """Compare multiple datasets using domain-appropriate methods
    
    Args:
        datasets: List of datasets to compare
        comparison_type: Type of comparison (pairwise, hierarchical, network)
    """
    
    if comparison_type == "pairwise":
        results = await pairwise_comparison(datasets)
    elif comparison_type == "hierarchical":
        results = await hierarchical_comparison(datasets)
    elif comparison_type == "network":
        results = await network_comparison(datasets)
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type}")
    
    return {
        "comparison_type": comparison_type,
        "dataset_count": len(datasets),
        "comparison_results": results,
        "statistical_summary": await calculate_comparison_stats(results),
        "visualization_data": await prepare_visualization(results)
    }
```

### Integrating External APIs

```python
import aiohttp
import asyncio

@mcp.tool
async def fetch_external_data(
    query: str, 
    database: str = "pubmed",
    limit: int = 10
) -> dict:
    """Fetch data from external scientific databases
    
    Args:
        query: Search query
        database: Database to search (pubmed, uniprot, ensembl)
        limit: Maximum number of results
    """
    
    database_urls = {
        "pubmed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
        "uniprot": "https://rest.uniprot.org/uniprotkb/",
        "ensembl": "https://rest.ensembl.org/"
    }
    
    if database not in database_urls:
        return {"error": f"Unsupported database: {database}"}
    
    try:
        async with aiohttp.ClientSession() as session:
            # Build database-specific query
            url = build_query_url(database_urls[database], query, limit)
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Process and standardize results
                    processed_results = await process_external_data(data, database)
                    
                    return {
                        "success": True,
                        "database": database,
                        "query": query,
                        "result_count": len(processed_results),
                        "results": processed_results,
                        "metadata": {
                            "query_timestamp": datetime.now().isoformat(),
                            "database_version": await get_database_version(database)
                        }
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}",
                        "database": database
                    }
                    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "database": database
        }
```

## 🎯 Agent Behavior Customization

### Custom Reflection Logic

```python
# agents/my_domain/src/domain_agent.py
from agents.template.src.agent import UniversalMCPAgent
from typing import List, Dict, Any

class DomainSpecificAgent(UniversalMCPAgent):
    """Agent specialized for your research domain"""
    
    def __init__(self, config):
        super().__init__(config)
        self.domain_knowledge = self._load_domain_knowledge()
        self.analysis_history = []
    
    async def _reflect_on_tool_results(
        self, 
        user_request: str, 
        tool_results: List[Dict[str, Any]]
    ) -> str:
        """Domain-specific reflection on tool results"""
        
        # Extract domain-specific context
        domain_context = await self._extract_domain_context(user_request, tool_results)
        
        # Apply domain expertise
        domain_insights = await self._apply_domain_expertise(domain_context)
        
        # Generate specialized interpretation
        interpretation = await self._generate_domain_interpretation(
            user_request, tool_results, domain_insights
        )
        
        # Store for learning
        self.analysis_history.append({
            "request": user_request,
            "results": tool_results,
            "interpretation": interpretation,
            "timestamp": datetime.now().isoformat()
        })
        
        return interpretation
    
    async def _extract_domain_context(
        self, 
        user_request: str, 
        tool_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract domain-specific context from results"""
        
        context = {
            "data_types": [],
            "analysis_methods": [],
            "statistical_measures": [],
            "domain_concepts": []
        }
        
        # Analyze tool results for domain-specific patterns
        for result in tool_results:
            tool_name = result.get("tool", "")
            tool_result = result.get("result", {})
            
            # Extract data types
            if "sequence" in tool_result:
                context["data_types"].append("sequence_data")
            if "expression" in tool_result:
                context["data_types"].append("expression_data")
            
            # Extract analysis methods
            if "statistics" in tool_result:
                context["analysis_methods"].append("statistical_analysis")
            if "alignment" in tool_result:
                context["analysis_methods"].append("sequence_alignment")
            
            # Extract statistical measures
            if "p_value" in tool_result:
                context["statistical_measures"].append(tool_result["p_value"])
            if "confidence" in tool_result:
                context["statistical_measures"].append(tool_result["confidence"])
        
        return context
    
    async def _apply_domain_expertise(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply domain-specific expertise to context"""
        
        expertise_prompt = f"""
        As a {self.config.domain} expert, analyze this research context:
        
        Data Types: {context['data_types']}
        Analysis Methods: {context['analysis_methods']}
        Statistical Measures: {context['statistical_measures']}
        
        Domain Knowledge:
        {self.domain_knowledge}
        
        Provide expert insights including:
        - Biological/scientific significance
        - Methodological appropriateness
        - Potential limitations or biases
        - Connections to established {self.config.domain} principles
        - Recommendations for follow-up analyses
        
        Return as structured JSON.
        """
        
        insights = await self.llm.generate(expertise_prompt)
        return json.loads(insights)
```

### Learning from Interactions

```python
class LearningAgent(DomainSpecificAgent):
    """Agent that learns from user interactions"""
    
    def __init__(self, config):
        super().__init__(config)
        self.interaction_patterns = {}
        self.success_metrics = {}
    
    async def learn_from_feedback(
        self, 
        user_request: str,
        agent_response: str,
        user_feedback: str,
        satisfaction_score: int
    ):
        """Learn from user feedback to improve future responses"""
        
        # Classify the interaction
        interaction_type = await self._classify_interaction(user_request)
        
        # Extract patterns from successful interactions
        if satisfaction_score >= 4:  # High satisfaction (1-5 scale)
            await self._record_successful_pattern(
                interaction_type, user_request, agent_response
            )
        else:
            await self._record_improvement_opportunity(
                interaction_type, user_request, agent_response, user_feedback
            )
        
        # Update success metrics
        if interaction_type not in self.success_metrics:
            self.success_metrics[interaction_type] = []
        
        self.success_metrics[interaction_type].append(satisfaction_score)
        
        # Adapt prompts based on patterns
        await self._adapt_prompts_from_learning()
    
    async def _record_successful_pattern(
        self, 
        interaction_type: str,
        user_request: str, 
        agent_response: str
    ):
        """Record patterns from successful interactions"""
        
        if interaction_type not in self.interaction_patterns:
            self.interaction_patterns[interaction_type] = {
                "successful_approaches": [],
                "common_tools": [],
                "effective_language": []
            }
        
        # Extract tools used
        tools_used = await self._extract_tools_from_response(agent_response)
        self.interaction_patterns[interaction_type]["common_tools"].extend(tools_used)
        
        # Extract effective language patterns
        language_patterns = await self._extract_language_patterns(agent_response)
        self.interaction_patterns[interaction_type]["effective_language"].extend(language_patterns)
```

## 🔄 Custom Workflow Patterns

### Multi-Step Analysis Pipelines

```python
class PipelineAgent(LearningAgent):
    """Agent that executes complex multi-step analysis pipelines"""
    
    async def execute_analysis_pipeline(
        self, 
        pipeline_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a predefined analysis pipeline"""
        
        pipeline_config = await self._load_pipeline_config(pipeline_name)
        
        pipeline_results = {
            "pipeline_name": pipeline_name,
            "input_data": input_data,
            "steps": [],
            "final_results": {},
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            for step_config in pipeline_config["steps"]:
                step_result = await self._execute_pipeline_step(
                    step_config, pipeline_results
                )
                pipeline_results["steps"].append(step_result)
                
                # Check for early termination conditions
                if step_result.get("terminate_pipeline", False):
                    break
            
            # Generate final analysis
            pipeline_results["final_results"] = await self._synthesize_pipeline_results(
                pipeline_results["steps"]
            )
            
        except Exception as e:
            pipeline_results["error"] = str(e)
            pipeline_results["failed_at_step"] = len(pipeline_results["steps"])
        
        pipeline_results["execution_time"] = time.time() - start_time
        
        return pipeline_results
    
    async def _execute_pipeline_step(
        self, 
        step_config: Dict[str, Any],
        pipeline_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single step in the analysis pipeline"""
        
        step_result = {
            "step_name": step_config["name"],
            "step_type": step_config["type"],
            "start_time": time.time()
        }
        
        if step_config["type"] == "tool_execution":
            # Execute MCP tool
            tool_name = step_config["tool"]
            parameters = await self._prepare_tool_parameters(
                step_config["parameters"], pipeline_context
            )
            
            result = await self._call_mcp_tool(tool_name, **parameters)
            step_result["tool_result"] = result
            
        elif step_config["type"] == "data_transformation":
            # Transform data from previous steps
            transform_func = step_config["function"]
            input_data = await self._extract_pipeline_data(
                step_config["input_source"], pipeline_context
            )
            
            result = await self._apply_transformation(transform_func, input_data)
            step_result["transformation_result"] = result
            
        elif step_config["type"] == "quality_check":
            # Validate data quality
            check_func = step_config["check"]
            data_to_check = await self._extract_pipeline_data(
                step_config["data_source"], pipeline_context
            )
            
            quality_result = await self._perform_quality_check(check_func, data_to_check)
            step_result["quality_result"] = quality_result
            
            # Determine if pipeline should continue
            if not quality_result.get("passed", True):
                step_result["terminate_pipeline"] = True
        
        step_result["execution_time"] = time.time() - step_result["start_time"]
        
        return step_result
```

## 🎛️ Configuration Management

### Domain-Specific Configurations

```yaml
# agents/genomics/config/genomics_agent_config.yaml
agent:
  name: "genomics-specialist"
  domain: "genomics"
  specializations:
    - "sequence_analysis"
    - "comparative_genomics" 
    - "phylogenetics"
    - "variant_analysis"

domain_knowledge:
  key_concepts:
    - "genetic_code"
    - "codon_usage"
    - "splice_sites"
    - "regulatory_elements"
  
  analysis_preferences:
    sequence_analysis:
      default_window_size: 100
      overlap_percentage: 50
      statistical_threshold: 0.01
    
    alignment_analysis:
      default_algorithm: "clustalw"
      gap_penalty: -10
      match_score: 2

workflows:
  genome_annotation:
    steps:
      - name: "quality_check"
        type: "quality_check"
        check: "sequence_quality_validation"
        
      - name: "gene_prediction"
        type: "tool_execution"
        tool: "gene_prediction"
        parameters:
          algorithm: "genemark"
          
      - name: "functional_annotation"
        type: "tool_execution"
        tool: "blast_annotation"
        parameters:
          database: "nr"
          e_value: 0.001

prompt_customizations:
  tool_selection:
    domain_expertise: "genomics and molecular biology"
    preferred_methods: ["blast", "clustal", "phylogenetic_analysis"]
    
  reflection:
    analysis_framework: "genomics_interpretation"
    key_considerations:
      - "evolutionary_context"
      - "functional_implications"
      - "regulatory_significance"
```

### Loading Custom Configurations

```python
# agents/genomics/src/genomics_agent.py
import yaml
from pathlib import Path

class GenomicsAgent(PipelineAgent):
    """Genomics-specialized AI scientist agent"""
    
    def __init__(self, config_path: str = None):
        # Load domain-specific configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "genomics_agent_config.yaml"
        
        with open(config_path) as f:
            domain_config = yaml.safe_load(f)
        
        # Create base agent config
        base_config = AgentConfig(
            name=domain_config["agent"]["name"],
            description=f"Genomics AI Scientist specialized in {', '.join(domain_config['agent']['specializations'])}",
            domain=domain_config["agent"]["domain"]
        )
        
        super().__init__(base_config)
        
        # Store domain-specific settings
        self.domain_config = domain_config
        self.domain_knowledge = domain_config.get("domain_knowledge", {})
        self.workflows = domain_config.get("workflows", {})
        
        # Customize prompts with domain expertise
        self._customize_prompts_with_domain_config()
    
    def _customize_prompts_with_domain_config(self):
        """Customize prompts using domain configuration"""
        
        customizations = self.domain_config.get("prompt_customizations", {})
        
        # Update prompt manager with domain-specific templates
        for prompt_name, customization in customizations.items():
            self.prompt_manager.add_domain_customization(prompt_name, customization)
```

This comprehensive customization guide enables researchers to adapt the NeLLi AI Scientist Agent Template to their specific domains while maintaining the powerful Universal MCP Agent architecture and FastMCP integration capabilities.