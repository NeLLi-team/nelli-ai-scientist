"""
Biological Analysis Engine for Enhanced Scientific Interpretation
Provides domain-specific intelligence for biological data analysis
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import re
from collections import Counter, defaultdict


class BiologicalAnalysisEngine:
    """Enhanced analysis engine for biological data interpretation"""
    
    def __init__(self):
        self.analysis_patterns = {
            'tandem_repeats': self._analyze_tandem_repeats,
            'gene_analysis': self._analyze_gene_data,
            'promoter_analysis': self._analyze_promoter_data,
            'assembly_stats': self._analyze_assembly_data,
            'kmer_analysis': self._analyze_kmer_data,
            'repeat_analysis': self._analyze_repeat_patterns
        }
        
        # Biological significance thresholds
        self.significance_thresholds = {
            'high_gc_content': 60.0,
            'low_gc_content': 35.0,
            'high_coding_density': 90.0,
            'large_genome': 5000000,  # 5MB
            'high_repeat_density': 5.0,  # %
            'many_genes': 1000
        }
    
    def analyze_biological_data(self, data: Dict[str, Any], focus_area: str = None) -> Dict[str, Any]:
        """Perform comprehensive biological analysis with domain expertise"""
        
        if not isinstance(data, dict):
            return {"error": "Invalid data format - expected dictionary"}
        
        # Determine data type and structure
        data_type = self._identify_data_type(data)
        
        # Perform targeted analysis based on focus area or data content
        if focus_area and focus_area in self.analysis_patterns:
            analysis_func = self.analysis_patterns[focus_area]
            detailed_analysis = analysis_func(data)
        else:
            # Comprehensive analysis of all detected patterns
            detailed_analysis = self._perform_comprehensive_analysis(data)
        
        return {
            "data_type": data_type,
            "biological_interpretation": detailed_analysis,
            "key_findings": self._extract_key_findings(detailed_analysis),
            "recommended_followup": self._suggest_followup_analyses(data_type, detailed_analysis)
        }
    
    def _identify_data_type(self, data: Dict[str, Any]) -> str:
        """Identify the type of biological data"""
        keys = set(data.keys())
        
        if 'assembly_stats' in keys and 'gene_analysis' in keys:
            return "comprehensive_genome_analysis"
        elif 'tandem_repeats' in data or 'repeat_analysis' in keys:
            return "repeat_analysis"
        elif 'promoter_analysis' in keys:
            return "promoter_analysis" 
        elif 'gene_analysis' in keys:
            return "gene_analysis"
        elif 'assembly_stats' in keys:
            return "assembly_analysis"
        else:
            return "unknown_biological_data"
    
    def _analyze_tandem_repeats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deep analysis of tandem repeat patterns"""
        
        repeat_data = data.get('repeat_analysis', {})
        tandem_repeats = repeat_data.get('tandem_repeats', [])
        
        if not tandem_repeats:
            return {"message": "No tandem repeats found in data"}
        
        # Analyze repeat characteristics
        analysis = {
            "total_repeats": len(tandem_repeats),
            "repeat_length_distribution": self._categorize_repeat_lengths(tandem_repeats),
            "copy_number_analysis": self._analyze_copy_numbers(tandem_repeats),
            "sequence_composition": self._analyze_repeat_composition(tandem_repeats),
            "genomic_distribution": self._analyze_repeat_distribution(tandem_repeats),
            "most_significant_repeats": self._identify_significant_repeats(tandem_repeats),
            "biological_significance": self._interpret_repeat_biology(tandem_repeats, data)
        }
        
        return analysis
    
    def _categorize_repeat_lengths(self, repeats: List[Dict]) -> Dict[str, int]:
        """Categorize repeats by length with biological significance"""
        categories = {
            "microsatellites_10-20bp": 0,
            "short_tandem_20-50bp": 0, 
            "medium_tandem_50-100bp": 0,
            "long_tandem_100-500bp": 0,
            "very_long_tandem_500bp+": 0
        }
        
        for repeat in repeats:
            length = repeat.get('total_length', 0)
            if 10 <= length < 20:
                categories["microsatellites_10-20bp"] += 1
            elif 20 <= length < 50:
                categories["short_tandem_20-50bp"] += 1
            elif 50 <= length < 100:
                categories["medium_tandem_50-100bp"] += 1
            elif 100 <= length < 500:
                categories["long_tandem_100-500bp"] += 1
            else:
                categories["very_long_tandem_500bp+"] += 1
        
        return categories
    
    def _analyze_copy_numbers(self, repeats: List[Dict]) -> Dict[str, Any]:
        """Analyze copy number patterns"""
        copy_numbers = [r.get('copy_number', 0) for r in repeats]
        copy_counter = Counter(copy_numbers)
        
        return {
            "copy_number_distribution": dict(copy_counter),
            "most_common_copy_number": copy_counter.most_common(1)[0] if copy_counter else None,
            "high_copy_repeats": [r for r in repeats if r.get('copy_number', 0) >= 5],
            "average_copy_number": sum(copy_numbers) / len(copy_numbers) if copy_numbers else 0
        }
    
    def _analyze_repeat_composition(self, repeats: List[Dict]) -> Dict[str, Any]:
        """Analyze sequence composition of repeats"""
        compositions = []
        at_rich_count = 0
        gc_rich_count = 0
        palindromic_count = 0
        
        for repeat in repeats:
            unit = repeat.get('repeat_unit', '').upper()
            if unit:
                at_content = (unit.count('A') + unit.count('T')) / len(unit) * 100
                gc_content = (unit.count('G') + unit.count('C')) / len(unit) * 100
                
                compositions.append({
                    "unit": unit,
                    "at_content": at_content,
                    "gc_content": gc_content,
                    "is_palindromic": unit == unit[::-1].translate(str.maketrans('ATCG', 'TAGC'))
                })
                
                if at_content > 70:
                    at_rich_count += 1
                elif gc_content > 70:
                    gc_rich_count += 1
                
                if unit == unit[::-1].translate(str.maketrans('ATCG', 'TAGC')):
                    palindromic_count += 1
        
        return {
            "composition_details": compositions,
            "at_rich_repeats": at_rich_count,
            "gc_rich_repeats": gc_rich_count,
            "palindromic_repeats": palindromic_count,
            "dominant_composition": "AT-rich" if at_rich_count > gc_rich_count else "GC-rich" if gc_rich_count > at_rich_count else "Mixed"
        }
    
    def _analyze_repeat_distribution(self, repeats: List[Dict]) -> Dict[str, Any]:
        """Analyze genomic distribution of repeats"""
        positions = [r.get('start', 0) for r in repeats]
        if not positions:
            return {}
        
        # Calculate clustering
        sorted_positions = sorted(positions)
        gaps = [sorted_positions[i+1] - sorted_positions[i] for i in range(len(sorted_positions)-1)]
        
        return {
            "span_start": min(positions),
            "span_end": max(positions),
            "average_gap": sum(gaps) / len(gaps) if gaps else 0,
            "clustered_regions": self._identify_clusters(sorted_positions),
            "distribution_pattern": "clustered" if any(gap < 1000 for gap in gaps) else "dispersed"
        }
    
    def _identify_significant_repeats(self, repeats: List[Dict]) -> List[Dict]:
        """Identify the most biologically significant repeats"""
        significant = []
        
        # Sort by multiple criteria
        for repeat in repeats:
            significance_score = 0
            
            # High copy number is significant
            copy_num = repeat.get('copy_number', 0)
            if copy_num >= 5:
                significance_score += copy_num * 2
            
            # Long repeats are significant
            length = repeat.get('total_length', 0)
            if length >= 50:
                significance_score += length // 10
            
            # Position near genome ends
            start_pos = repeat.get('start', 0)
            if start_pos > 240000 or start_pos < 1000:  # Near ends
                significance_score += 5
            
            repeat['significance_score'] = significance_score
            
            if significance_score >= 10:
                significant.append(repeat)
        
        return sorted(significant, key=lambda x: x['significance_score'], reverse=True)[:5]
    
    def _interpret_repeat_biology(self, repeats: List[Dict], full_data: Dict) -> Dict[str, Any]:
        """Interpret biological significance of repeat patterns"""
        
        # Get genome context
        assembly_stats = full_data.get('assembly_stats', {})
        genome_length = assembly_stats.get('total_length', 0)
        gc_content = assembly_stats.get('gc_content', 0)
        
        interpretations = []
        
        # AT-rich bias interpretation
        at_rich_count = sum(1 for r in repeats 
                           if (r.get('repeat_unit', '').count('A') + r.get('repeat_unit', '').count('T')) / 
                           len(r.get('repeat_unit', 'X')) > 0.7)
        
        if at_rich_count > len(repeats) * 0.6:
            interpretations.append({
                "type": "compositional_bias",
                "finding": f"Strong AT-rich repeat bias ({at_rich_count}/{len(repeats)} repeats)",
                "significance": "Consistent with low GC genome organization, typical of some bacteria/viruses"
            })
        
        # Terminal repeat analysis
        terminal_repeats = [r for r in repeats if r.get('start', 0) > genome_length * 0.95]
        if terminal_repeats:
            interpretations.append({
                "type": "terminal_structure",
                "finding": f"{len(terminal_repeats)} repeats near genome terminus",
                "significance": "May represent terminal inverted repeats or packaging signals"
            })
        
        # High copy number significance
        high_copy = [r for r in repeats if r.get('copy_number', 0) >= 5]
        if high_copy:
            interpretations.append({
                "type": "high_copy_elements", 
                "finding": f"{len(high_copy)} repeats with ≥5 copies",
                "significance": "Potential regulatory elements or structural motifs"
            })
        
        return {
            "biological_interpretations": interpretations,
            "genome_context": {
                "repeat_density": len(repeats) / (genome_length / 1000) if genome_length else 0,
                "gc_consistency": "AT-rich repeats match low GC genome" if gc_content < 35 else "Compositional mismatch"
            }
        }
    
    def _identify_clusters(self, positions: List[int], max_gap: int = 5000) -> List[Dict]:
        """Identify clusters of repeats"""
        if len(positions) < 2:
            return []
        
        clusters = []
        current_cluster = [positions[0]]
        
        for i in range(1, len(positions)):
            if positions[i] - positions[i-1] <= max_gap:
                current_cluster.append(positions[i])
            else:
                if len(current_cluster) >= 2:
                    clusters.append({
                        "start": min(current_cluster),
                        "end": max(current_cluster),
                        "count": len(current_cluster)
                    })
                current_cluster = [positions[i]]
        
        # Check final cluster
        if len(current_cluster) >= 2:
            clusters.append({
                "start": min(current_cluster),
                "end": max(current_cluster), 
                "count": len(current_cluster)
            })
        
        return clusters
    
    def _perform_comprehensive_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of all available data"""
        analysis = {}
        
        for data_key, analysis_func in self.analysis_patterns.items():
            if data_key in data or any(data_key in str(k) for k in data.keys()):
                try:
                    analysis[data_key] = analysis_func(data)
                except Exception as e:
                    analysis[data_key] = {"error": f"Analysis failed: {str(e)}"}
        
        return analysis
    
    def _extract_key_findings(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis"""
        findings = []
        
        # Extract important patterns from each analysis type
        for analysis_type, results in analysis.items():
            if isinstance(results, dict):
                if "most_significant_repeats" in results:
                    sig_repeats = results["most_significant_repeats"]
                    if sig_repeats:
                        findings.append(f"Found {len(sig_repeats)} highly significant tandem repeats")
                
                if "biological_interpretations" in results:
                    interpretations = results["biological_interpretations"]
                    for interp in interpretations:
                        findings.append(interp.get("finding", ""))
        
        return findings[:5]  # Top 5 findings
    
    def _suggest_followup_analyses(self, data_type: str, analysis: Dict[str, Any]) -> List[str]:
        """Suggest follow-up analyses based on findings"""
        suggestions = []
        
        if data_type == "repeat_analysis":
            suggestions.extend([
                "Analyze repeat conservation across related genomes",
                "Investigate repeat-associated genes and regulatory elements",
                "Examine repeat expansion patterns and stability"
            ])
        
        return suggestions
    
    # Placeholder methods for other analysis types
    def _analyze_gene_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze gene prediction and coding statistics"""
        return {"status": "Gene analysis module - to be implemented"}
    
    def _analyze_promoter_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze promoter motif patterns"""
        return {"status": "Promoter analysis module - to be implemented"}
    
    def _analyze_assembly_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze assembly quality metrics"""
        return {"status": "Assembly analysis module - to be implemented"}
    
    def _analyze_kmer_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze k-mer frequency patterns"""
        return {"status": "K-mer analysis module - to be implemented"}
    
    def _analyze_repeat_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze general repeat patterns"""
        return self._analyze_tandem_repeats(data)  # Delegate to tandem repeat analysis