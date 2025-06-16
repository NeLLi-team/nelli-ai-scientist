"""
AI Agent to analyze and summarize the results section of research papers
"""
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
import json
import re
from datetime import datetime
import os

from .base_agent import BaseAgent
from .data_models import (
    ResultsAnalysis, PaperMetadata, ResultsSection,
    ResultsSummary, SummarizationRequest, SummarizationResponse
)
from ..tools.filesystem_tools import FilesystemToolkit
from ..tools.biopython_tools import BioPythonToolkit
from ..llm_interface import LLMInterface

# Import paper source classes
from paper_search_mcp.academic_platforms.arxiv import ArxivSearcher
from paper_search_mcp.academic_platforms.pubmed import PubMedSearcher
from paper_search_mcp.academic_platforms.biorxiv import BioRxivSearcher
from paper_search_mcp.academic_platforms.medrxiv import MedRxivSearcher
from paper_search_mcp.paper import Paper

logger = logging.getLogger(__name__)

class ResultsAnalyzerAgent(BaseAgent):
    def __init__(self, config: 'AgentConfig'):
        super().__init__(config)
        self.config = config
        
        # Initialize tools
        self.fs_tools = FilesystemToolkit()
        self.bio_tools = BioPythonToolkit()
        self.llm = LLMInterface(config.llm_provider)
        
        # Initialize paper sources
        self.paper_sources = {
            'arxiv': ArxivSearcher(),
            'pubmed': PubMedSearcher(),
            'biorxiv': BioRxivSearcher(),
            'medrxiv': MedRxivSearcher()
        }
        
        # Load summarization config
        self.summarization_config = config.paper_summarization
        logger.info(f"ResultsAnalyzerAgent initialized with summarization capabilities")
    
    def summarize_paper_results(self, request: SummarizationRequest) -> SummarizationResponse:
        """
        Summarize the results section of a research paper.
        
        Args:
            request: SummarizationRequest containing paper details
            
        Returns:
            SummarizationResponse with the summary or error
        """
        try:
            # 1. Get paper metadata
            paper_metadata = self._get_paper_metadata(request)
            
            # 2. Load and extract results section
            results_section = self._extract_results_section_from_paper(request, paper_metadata)
            
            # 3. Generate summary
            summary = self._generate_results_summary(
                results_section,
                request.max_summary_length or self.summarization_config["max_summary_length"]
            )
            
            return SummarizationResponse(
                request=request,
                success=True,
                results_summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error summarizing paper results: {str(e)}")
            return SummarizationResponse(
                request=request,
                success=False,
                error=str(e)
            )
    
    def _get_paper_metadata(self, request: SummarizationRequest) -> PaperMetadata:
        """Get paper metadata from the source."""
        if request.source not in self.paper_sources:
            raise ValueError(f"Unsupported paper source: {request.source}")
        
        # Get the appropriate searcher
        searcher = self.paper_sources[request.source]
        
        try:
            # Search for the paper to get metadata
            # Note: This assumes the paper_id is unique enough for a search
            papers = searcher.search(f"id:{request.paper_id}", max_results=1)
            
            if not papers:
                raise ValueError(f"Paper not found: {request.paper_id} in {request.source}")
            
            paper = papers[0]  # Get the first (and should be only) result
            
            return PaperMetadata(
                paper_id=request.paper_id,
                title=paper.title,
                authors=paper.authors,
                source=request.source,
                published_date=paper.published_date,
                doi=paper.doi,
                url=paper.url,
                pdf_url=paper.pdf_url
            )
            
        except Exception as e:
            logger.error(f"Error getting paper metadata: {str(e)}")
            raise
    
    def _extract_results_section_from_paper(
        self, request: SummarizationRequest, paper_metadata: PaperMetadata
    ) -> ResultsSection:
        """Extract results section from paper content."""
        # Load paper content
        paper_content = self._load_paper_content(request)
        
        # Extract results section using patterns from config
        raw_text, confidence, boundaries = self._extract_results_section_with_confidence(
            paper_content,
            self.summarization_config["extraction_patterns"]
        )
        
        return ResultsSection(
            paper_metadata=paper_metadata,
            raw_text=raw_text,
            extracted_at=datetime.now(),
            confidence_score=confidence,
            section_boundaries=boundaries
        )
    
    def _extract_results_section_with_confidence(
        self, paper_content: str, patterns: List[str]
    ) -> tuple[str, float, Dict[str, int]]:
        """Extract results section with confidence score."""
        # Try pattern matching first
        for pattern in patterns:
            match = re.search(pattern, paper_content, re.DOTALL | re.IGNORECASE)
            if match:
                start, end = match.span()
                return (
                    paper_content[start:end].strip(),
                    0.9,  # High confidence for pattern match
                    {"start": start, "end": end}
                )
        
        # Fallback to LLM extraction
        llm_text = self._llm_extract_results_section(paper_content)
        return (
            llm_text,
            0.7,  # Lower confidence for LLM extraction
            {"start": 0, "end": len(llm_text)}  # Approximate boundaries
        )
    
    def _generate_results_summary(
        self, results_section: ResultsSection, max_length: int
    ) -> ResultsSummary:
        """Generate a summary of the results section."""
        prompt = f"""
        Summarize the key findings from this results section in {max_length} characters or less.
        Also extract the 3-5 most important findings as bullet points.
        
        Results section:
        {results_section.raw_text}
        
        Summary:
        """
        
        llm_response = self.llm.generate_response(
            prompt=prompt,
            max_tokens=max_length // 4,  # Approximate token count
            temperature=0.3
        )
        
        # Parse response to separate summary and key findings
        response_text = llm_response.get("text", "")
        summary_lines = response_text.split("\n")
        summary_text = summary_lines[0]
        key_findings = [
            line.strip("- *").strip()
            for line in summary_lines[1:]
            if line.strip().startswith(("-", "*"))
        ]
        
        return ResultsSummary(
            results_section=results_section,
            summary_text=summary_text,
            key_findings=key_findings,
            generated_at=datetime.now(),
            model_used=self.config.llm_provider
        )
    
    def _load_paper_content(self, request: SummarizationRequest) -> str:
        """Load paper content from the specified source."""
        if request.source not in self.paper_sources:
            raise ValueError(f"Unsupported paper source: {request.source}")
        
        # Get the appropriate searcher
        searcher = self.paper_sources[request.source]
        
        try:
            # Ensure save path exists
            save_path = request.save_path or self.summarization_config["save_path"]
            os.makedirs(save_path, exist_ok=True)
            
            # Read the paper content
            content = searcher.read_paper(request.paper_id, save_path)
            
            if not content:
                raise ValueError(f"Could not read paper content: {request.paper_id}")
            
            return content
            
        except Exception as e:
            logger.error(f"Error loading paper content: {str(e)}")
            raise
    
    def _get_source_searcher(self, source: str) -> Any:
        """Get the appropriate paper source searcher."""
        if source not in self.paper_sources:
            raise ValueError(f"Unsupported paper source: {source}")
        return self.paper_sources[source]
    
    def _validate_source(self, source: str) -> None:
        """Validate that the source is supported."""
        if source not in self.paper_sources:
            raise ValueError(
                f"Unsupported paper source: {source}. "
                f"Must be one of: {list(self.paper_sources.keys())}"
            )
    
    def _get_paper_from_source(self, paper_id: str, source: str) -> Paper:
        """Get a Paper object from the specified source."""
        searcher = self._get_source_searcher(source)
        papers = searcher.search(f"id:{paper_id}", max_results=1)
        
        if not papers:
            raise ValueError(f"Paper not found: {paper_id} in {source}")
        
        return papers[0]
    
    def _ensure_paper_downloaded(self, paper_id: str, source: str, save_path: str) -> str:
        """Ensure paper is downloaded and return the path."""
        searcher = self._get_source_searcher(source)
        pdf_path = os.path.join(save_path, f"{paper_id}.pdf")
        
        if not os.path.exists(pdf_path):
            try:
                pdf_path = searcher.download_pdf(paper_id, save_path)
            except Exception as e:
                raise ValueError(f"Failed to download paper: {str(e)}")
        
        return pdf_path
    
    def analyze_results_section(self, paper_path: Union[str, Path]) -> ResultsAnalysis:
        """
        Main method: Analyze the results section of a research paper.
        
        Args:
            paper_path: Path to the research paper PDF/text file
            
        Returns:
            ResultsAnalysis: Structured analysis of the results section
        """
        try:
            #1. Load paper safely
            paper_content = self._load_paper(paper_path)
            
            #2. Extract results section
            results_section = self._extract_results_section(paper_content)
            
            #3. Analyze results content
            analysis = self._analyze_results_content(results_section)
            
            #4. Generate structured report
            report = self._generate_results_report(analysis, paper_path)
            
            #5. Save analysis
            self._save_analysis(report, paper_path)
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing results section: {str(e)}")
            raise
    
    def _load_paper(self, paper_path: Union[str, Path]) -> str:
        """Load paper content using filesystem tools."""
        
        if not self.fs_tools.file_exists(str(paper_path)):
            raise FileNotFoundError(f"Paper not found: {paper_path}")
        
        return self.fs_tools.read_file(str(paper_path))
    
    def _extract_results_section(self, paper_content: str) -> str:
        """
        Extract the results section from the paper content.
        """
        # Look for results section headers
        results_patterns = [
            r'(?i)\n\s*(?:results?|findings?)\s*\n',
            r'(?i)\n\s*\d+\.?\s*(?:results?|findings?)\s*\n',
            r'(?i)\n\s*(?:results?|findings?)\s+and\s+discussion\s*\n'
        ]
        
        content_lower = paper_content.lower()
        
        # Find results section start
        start_pos = None
        for pattern in results_patterns:
            match = re.search(pattern, paper_content)
            if match:
                start_pos = match.end()
                break
        
        if not start_pos:
            # Fallback: use LLM to identify results section
            return self._llm_extract_results_section(paper_content)
        
        # Find section end (next major section)
        end_patterns = [
            r'(?i)\n\s*(?:discussion|conclusion|references|acknowledgments?)\s*\n',
            r'(?i)\n\s*\d+\.?\s*(?:discussion|conclusion|references)\s*\n'
        ]
        
        end_pos = len(paper_content)
        for pattern in end_patterns:
            match = re.search(pattern, paper_content[start_pos:])
            if match:
                end_pos = start_pos + match.start()
                break
        
        results_section = paper_content[start_pos:end_pos].strip()
        
        if len(results_section) < 100:  # Too short, likely extraction failed
            return self._llm_extract_results_section(paper_content)
        
        return results_section
    
    def _llm_extract_results_section(self, paper_content: str) -> str:
        """Use LLM to extract results section when pattern matching fails."""
        
        prompt = f"""
        Extract ONLY the results section from this research paper. 
        Return just the results section text, nothing else.
        
        Paper content (first 3000 chars):
        {paper_content[:3000]}...
        
        Results section:
        """
        
        response = self.llm.generate_response(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.1
        )
        
        return response.get("text", "Results section not found")
    
    def _analyze_results_content(self, results_section: str) -> Dict[str, Any]:
        """
        Analyze the extracted results section content.
        """
        analysis_prompt = f"""
        Analyze this results section and extract:
        
        1. PRIMARY FINDINGS: Main results and outcomes
        2. STATISTICAL RESULTS: P-values, confidence intervals, effect sizes, significance tests
        3. QUANTITATIVE DATA: Numbers, percentages, measurements, sample sizes
        4. COMPARISONS: Between groups, conditions, or treatments
        5. TABLES/FIGURES MENTIONED: References to data visualizations
        6. KEY PATTERNS: Trends, correlations, or relationships identified
        
        Provide structured output in JSON format.
        
        Results section:
        {results_section}
        
        Analysis:
        """
        
        llm_response = self.llm.generate_response(
            prompt=analysis_prompt,
            max_tokens=2000,
            temperature=0.2
        )
        
        # Extract statistical patterns using regex as backup
        statistical_patterns = self._extract_statistical_patterns(results_section)
        
        return {
            "llm_analysis": llm_response,
            "statistical_patterns": statistical_patterns,
            "section_length": len(results_section),
            "section_text": results_section[:500] + "..." if len(results_section) > 500 else results_section
        }
    
    def _extract_statistical_patterns(self, text: str) -> Dict[str, List]:
        """Extract statistical patterns using regex."""
        
        patterns = {
            "p_values": re.findall(r'p\s*[<>=]\s*0?\.\d+', text, re.IGNORECASE),
            "percentages": re.findall(r'\d+\.?\d*\s*%', text),
            "confidence_intervals": re.findall(r'95%\s*CI|confidence\s+interval', text, re.IGNORECASE),
            "sample_sizes": re.findall(r'n\s*=\s*\d+', text, re.IGNORECASE),
            "correlations": re.findall(r'r\s*=\s*0?\.\d+', text, re.IGNORECASE),
            "means_stds": re.findall(r'\d+\.?\d*\s*±\s*\d+\.?\d*', text)
        }
        
        return {key: list(set(values)) for key, values in patterns.items()}
    
    def _generate_results_report(self, analysis: Dict, paper_path: Union[str, Path]) -> ResultsAnalysis:
        """Generate structured results analysis report."""
        
        # Prepare report data
        report_data = {
            "paper_filename": Path(paper_path).name,
            "analysis_timestamp": self._get_timestamp(),
            "results_analysis": analysis,
            "summary": self._generate_summary(analysis),
            "agent_name": self.config.name
        }
        
        # Use BioPython's JSON report tool for structured output
        structured_report = self.bio_tools.write_json_report(report_data)
        
        return ResultsAnalysis(
            paper_name=Path(paper_path).name,
            primary_findings=analysis["llm_analysis"].get("primary_findings", []),
            statistical_results=analysis["statistical_patterns"],
            quantitative_data=analysis["llm_analysis"].get("quantitative_data", {}),
            summary=report_data["summary"],
            full_analysis=structured_report
        )
    
    def _save_analysis(self, report: ResultsAnalysis, original_path: Union[str, Path]):
        """Save analysis results using filesystem tools."""
        
        # Create results directory
        results_dir = Path("results_analysis")
        self.fs_tools.create_directory(str(results_dir))
        
        # Generate output filename
        paper_name = Path(original_path).stem
        output_file = results_dir / f"{paper_name}_results_analysis.json"
        
        # Save the report
        self.fs_tools.write_file(
            str(output_file),
            json.dumps(report.to_dict(), indent=2)
        )
        
        logger.info(f"Results analysis saved to: {output_file}")
    
    def _generate_summary(self, analysis: Dict) -> str:
        """Generate concise summary of results analysis."""
        
        stats = analysis["statistical_patterns"]
        summary_parts = []
        
        if stats["p_values"]:
            summary_parts.append(f"Found {len(stats['p_values'])} statistical significance tests")
        
        if stats["percentages"]:
            summary_parts.append(f"Identified {len(stats['percentages'])} percentage values")
        
        if stats["sample_sizes"]:
            summary_parts.append(f"Detected {len(stats['sample_sizes'])} sample size references")
        
        return "; ".join(summary_parts) if summary_parts else "Basic results section identified"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().isoformat()