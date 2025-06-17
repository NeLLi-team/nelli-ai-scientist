#!/usr/bin/env python3
"""
Test script for paper results summarization functionality
"""
import asyncio
import logging
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.results_analyzer import ResultsAnalyzerAgent
from src.data_models import SummarizationRequest
from src.agent import AgentConfig
from config.config_loader import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_summarization():
    """Test the summarization functionality with a known paper."""
    
    # Load configuration
    config = load_config()
    
    # Create agent config
    agent_config = AgentConfig(
        name="results_analyzer",
        llm_provider=config.llm_provider,
        mcp_config_path="mcp_config.json"
    )
    
    # Initialize agent
    agent = ResultsAnalyzerAgent(agent_config)
    
    # Test cases
    test_cases = [
        {
            "paper_id": "2106.12345",
            "source": "arxiv",
            "description": "Test arXiv paper"
        },
        {
            "paper_id": "10.1101/2023.01.01.123456",
            "source": "biorxiv", 
            "description": "Test bioRxiv paper"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['description']}")
        print(f"Paper ID: {test_case['paper_id']}")
        print(f"Source: {test_case['source']}")
        print(f"{'='*60}")
        
        try:
            # Create request
            request = SummarizationRequest(
                paper_id=test_case["paper_id"],
                source=test_case["source"],
                max_summary_length=500
            )
            
            # Get summary
            response = agent.summarize_paper_results(request)
            
            if response.success:
                print("✅ SUCCESS!")
                print(f"Confidence: {response.results_summary.results_section.confidence_score}")
                print(f"Summary: {response.results_summary.summary_text}")
                print(f"Key findings:")
                for finding in response.results_summary.key_findings:
                    print(f"  • {finding}")
            else:
                print("❌ FAILED!")
                print(f"Error: {response.error}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_summarization()) 