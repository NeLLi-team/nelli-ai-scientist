"""
Data models for results analysis
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class ResultsAnalysis:
    """Results section analysis data model."""
    paper_name: str
    primary_findings: List[str]
    statistical_results: Dict[str, List]
    quantitative_data: Dict[str, Any]
    summary: str
    full_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "paper_name": self.paper_name,
            "primary_findings": self.primary_findings,
            "statistical_results": self.statistical_results,
            "quantitative_data": self.quantitative_data,
            "summary": self.summary,
            "full_analysis": self.full_analysis
        }