"""
Data models for results analysis and paper summarization
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime

@dataclass
class PaperMetadata:
    """Metadata for a research paper."""
    paper_id: str
    title: str
    authors: List[str]
    source: str  # arxiv, pubmed, biorxiv, medrxiv
    published_date: Optional[datetime] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "source": self.source,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "doi": self.doi,
            "url": self.url,
            "pdf_url": self.pdf_url
        }

@dataclass
class ResultsSection:
    """Extracted results section from a paper."""
    paper_metadata: PaperMetadata
    raw_text: str
    extracted_at: datetime
    confidence_score: float  # Confidence in the extraction (0-1)
    section_boundaries: Dict[str, int]  # Start/end positions in the paper
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "paper_metadata": self.paper_metadata.to_dict(),
            "raw_text": self.raw_text,
            "extracted_at": self.extracted_at.isoformat(),
            "confidence_score": self.confidence_score,
            "section_boundaries": self.section_boundaries
        }

@dataclass
class ResultsSummary:
    """Summary of a paper's results section."""
    results_section: ResultsSection
    summary_text: str
    key_findings: List[str]
    generated_at: datetime
    model_used: str  # LLM or method used for summarization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "results_section": self.results_section.to_dict(),
            "summary_text": self.summary_text,
            "key_findings": self.key_findings,
            "generated_at": self.generated_at.isoformat(),
            "model_used": self.model_used
        }

@dataclass
class ResultsAnalysis:
    """Results section analysis data model."""
    paper_metadata: PaperMetadata
    results_summary: ResultsSummary
    primary_findings: List[str]
    statistical_results: Dict[str, List]
    quantitative_data: Dict[str, Any]
    full_analysis: Dict[str, Any]
    analysis_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "paper_metadata": self.paper_metadata.to_dict(),
            "results_summary": self.results_summary.to_dict(),
            "primary_findings": self.primary_findings,
            "statistical_results": self.statistical_results,
            "quantitative_data": self.quantitative_data,
            "full_analysis": self.full_analysis,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }

@dataclass
class SummarizationRequest:
    """Request for paper results summarization."""
    paper_id: str
    source: str
    save_path: Optional[str] = None
    max_summary_length: Optional[int] = None
    include_key_findings: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "paper_id": self.paper_id,
            "source": self.source,
            "save_path": self.save_path,
            "max_summary_length": self.max_summary_length,
            "include_key_findings": self.include_key_findings
        }

@dataclass
class SummarizationResponse:
    """Response from paper results summarization."""
    request: SummarizationRequest
    success: bool
    results_summary: Optional[ResultsSummary] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request": self.request.to_dict(),
            "success": self.success,
            "results_summary": self.results_summary.to_dict() if self.results_summary else None,
            "error": self.error
        }