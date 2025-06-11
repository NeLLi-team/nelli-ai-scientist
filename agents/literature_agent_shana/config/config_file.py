from src.agent import ResultsAnalyzerAgent, AgentConfig
from src.llm_interface import LLMProvider

# Create agent
config = AgentConfig(
    name="results_analyzer",
    capabilities=["statistical_analysis", "results_extraction", "quality_assessment"],
    llm_provider=LLMProvider.CBORG  # Default, or use CLAUDE/OPENAI
)
agent = ResultsAnalyzerAgent(config)