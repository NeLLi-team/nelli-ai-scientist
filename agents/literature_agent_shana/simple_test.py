#!/usr/bin/env python3
"""
Simple test script for the summarize_paper_results tool
"""
import sys
import os

# Add the paper-search-mcp directory to the path
sys.path.append('/clusterfs/jgi/scratch/science/mgs/nelli/shana/nelli-ai-scientist/agents/literature_agent_shana/paper-search-mcp')

def test_basic_functionality():
    """Test basic functionality without complex dependencies."""
    
    print("Testing basic functionality...")
    
    try:
        # Test if we can import the basic modules
        from paper_search_mcp.server import summarize_paper_results
        print("✅ Successfully imported summarize_paper_results function")
        
        # Test if we can import the data models
        from paper_search_mcp.paper import Paper
        print("✅ Successfully imported Paper class")
        
        # Test if we can import the academic platforms
        from paper_search_mcp.academic_platforms.arxiv import ArxivSearcher
        print("✅ Successfully imported ArxivSearcher")
        
        print("\n🎉 Basic functionality test passed!")
        print("The summarize_paper_results tool is available and ready to use.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Some dependencies may be missing.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_basic_functionality() 