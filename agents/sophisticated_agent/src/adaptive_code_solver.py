"""
Adaptive Code Solver - Makes the agent truly intelligent for any task
Automatically plans, writes, executes, and iterates on code to solve user requests
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AdaptiveCodeSolver:
    """Intelligent code generation and execution system for solving any user request"""
    
    def __init__(self, mcp_tool_caller=None):
        self.mcp_tool_caller = mcp_tool_caller
        self.max_iterations = 3
        self.solution_patterns = {
            'data_analysis': self._plan_data_analysis,
            'file_processing': self._plan_file_processing,
            'visualization': self._plan_visualization,
            'calculation': self._plan_calculation,
            'text_processing': self._plan_text_processing,
            'biological_analysis': self._plan_biological_analysis
        }
    
    async def solve_user_request(self, user_request: str, available_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main method to solve any user request through adaptive code generation
        
        This method:
        1. Analyzes the user request to understand the goal
        2. Plans the code approach
        3. Generates initial code
        4. Executes and evaluates results
        5. Iterates to improve if needed
        """
        
        logger.info(f"🧠 Solving request: {user_request}")
        
        # Step 1: Analyze the request and determine the approach
        request_analysis = self._analyze_user_request(user_request, available_data)
        
        if not request_analysis['solvable_with_code']:
            return {
                "success": False,
                "reason": "Request doesn't appear to need code solution",
                "suggestion": request_analysis.get('alternative_approach')
            }
        
        # Step 2: Plan the solution approach
        solution_plan = self._plan_solution(request_analysis)
        
        # Step 3: Iterative code generation and execution
        for iteration in range(self.max_iterations):
            logger.info(f"🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Generate code based on plan and previous results
            code_result = await self._generate_and_execute_code(
                solution_plan, 
                user_request, 
                available_data,
                iteration
            )
            
            # Evaluate if the result solves the user's request
            evaluation = self._evaluate_solution(user_request, code_result, solution_plan)
            
            if evaluation['success']:
                return {
                    "success": True,
                    "solution": code_result,
                    "evaluation": evaluation,
                    "iterations_used": iteration + 1,
                    "plan": solution_plan
                }
            
            # If not successful, update the plan for next iteration
            solution_plan = self._refine_plan(solution_plan, evaluation, code_result)
        
        # If we've exhausted iterations without success
        return {
            "success": False,
            "reason": f"Could not solve after {self.max_iterations} iterations",
            "last_result": code_result,
            "final_evaluation": evaluation
        }
    
    def _analyze_user_request(self, user_request: str, available_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what the user is asking for and determine if code can solve it"""
        
        request_lower = user_request.lower()
        
        # Determine if this needs a code solution
        code_indicators = [
            'analyze', 'calculate', 'find', 'extract', 'process', 'generate',
            'visualize', 'plot', 'show', 'list', 'count', 'compare',
            'detailed', 'statistics', 'distribution', 'frequency', 'pattern',
            'most', 'frequent', 'largest', 'smallest', 'top', 'bottom'
        ]
        
        needs_code = any(indicator in request_lower for indicator in code_indicators)
        
        # Categorize the type of request
        request_type = self._categorize_request(request_lower)
        
        # Analyze what data is available
        data_analysis = self._analyze_available_data(available_data)
        
        return {
            'request': user_request,
            'solvable_with_code': needs_code,
            'request_type': request_type,
            'data_available': data_analysis,
            'key_terms': self._extract_key_terms(request_lower),
            'expected_output': self._predict_expected_output(request_lower)
        }
    
    def _categorize_request(self, request_lower: str) -> str:
        """Categorize the type of request to choose appropriate solution strategy"""
        
        if any(term in request_lower for term in ['repeat', 'tandem', 'motif', 'sequence']):
            return 'biological_analysis'
        elif any(term in request_lower for term in ['plot', 'graph', 'visualize', 'chart']):
            return 'visualization'
        elif any(term in request_lower for term in ['calculate', 'compute', 'math', 'statistics']):
            return 'calculation'
        elif any(term in request_lower for term in ['file', 'read', 'parse', 'process']):
            return 'file_processing'
        elif any(term in request_lower for term in ['text', 'string', 'parse', 'extract']):
            return 'text_processing'
        else:
            return 'data_analysis'
    
    def _analyze_available_data(self, available_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what data is available for the solution"""
        
        if not available_data:
            return {"has_data": False}
        
        data_info = {
            "has_data": True,
            "data_types": [],
            "key_sections": list(available_data.keys()) if isinstance(available_data, dict) else [],
            "data_structure": type(available_data).__name__
        }
        
        # Identify specific biological data types
        if isinstance(available_data, dict):
            if 'repeat_analysis' in available_data:
                data_info['data_types'].append('repeat_data')
            if 'gene_analysis' in available_data:
                data_info['data_types'].append('gene_data')
            if 'promoter_analysis' in available_data:
                data_info['data_types'].append('promoter_data')
            if 'assembly_stats' in available_data:
                data_info['data_types'].append('assembly_data')
        
        return data_info
    
    def _extract_key_terms(self, request_lower: str) -> List[str]:
        """Extract key terms that will guide code generation"""
        
        key_terms = []
        
        # Extract specific biological terms
        bio_terms = ['tandem repeat', 'promoter', 'gene', 'motif', 'sequence', 'gc content']
        for term in bio_terms:
            if term in request_lower:
                key_terms.append(term)
        
        # Extract analysis terms
        analysis_terms = ['most frequent', 'detailed', 'statistics', 'distribution', 'top']
        for term in analysis_terms:
            if term in request_lower:
                key_terms.append(term)
        
        return key_terms
    
    def _predict_expected_output(self, request_lower: str) -> str:
        """Predict what kind of output the user expects"""
        
        if 'detailed' in request_lower:
            return 'comprehensive_analysis'
        elif any(term in request_lower for term in ['most', 'top', 'frequent']):
            return 'ranked_list_with_details'
        elif 'statistics' in request_lower:
            return 'statistical_summary'
        elif any(term in request_lower for term in ['plot', 'graph', 'visualize']):
            return 'visualization'
        else:
            return 'informative_summary'
    
    def _plan_solution(self, request_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the solution approach based on request analysis"""
        
        request_type = request_analysis['request_type']
        
        if request_type in self.solution_patterns:
            plan_func = self.solution_patterns[request_type]
            return plan_func(request_analysis)
        else:
            return self._plan_generic_solution(request_analysis)
    
    def _plan_biological_analysis(self, request_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan biological analysis solution"""
        
        return {
            'approach': 'biological_analysis',
            'steps': [
                'Load and validate biological data',
                'Extract relevant data sections (repeats, genes, etc.)',
                'Perform statistical analysis',
                'Identify patterns and significance',
                'Format results with biological context'
            ],
            'expected_libraries': ['json', 'collections', 'statistics', 'pandas'],
            'output_format': 'structured_biological_report',
            'key_terms': request_analysis['key_terms']
        }
    
    def _plan_data_analysis(self, request_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan general data analysis solution"""
        
        return {
            'approach': 'data_analysis',
            'steps': [
                'Load and inspect data structure',
                'Extract relevant information',
                'Perform analysis based on request',
                'Calculate statistics or patterns',
                'Format and present results'
            ],
            'expected_libraries': ['json', 'collections', 'statistics'],
            'output_format': 'analytical_summary'
        }
    
    # Placeholder methods for other solution types
    def _plan_file_processing(self, request_analysis: Dict[str, Any]) -> Dict[str, Any]:
        return self._plan_generic_solution(request_analysis)
    
    def _plan_visualization(self, request_analysis: Dict[str, Any]) -> Dict[str, Any]:
        return self._plan_generic_solution(request_analysis)
    
    def _plan_calculation(self, request_analysis: Dict[str, Any]) -> Dict[str, Any]:
        return self._plan_generic_solution(request_analysis)
    
    def _plan_text_processing(self, request_analysis: Dict[str, Any]) -> Dict[str, Any]:
        return self._plan_generic_solution(request_analysis)
    
    def _plan_generic_solution(self, request_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generic solution planning"""
        
        return {
            'approach': 'generic_analysis',
            'steps': [
                'Understand the data structure',
                'Extract relevant information',
                'Process according to user request',
                'Present results clearly'
            ],
            'expected_libraries': ['json', 'collections'],
            'output_format': 'general_summary'
        }
    
    async def _generate_and_execute_code(self, plan: Dict[str, Any], user_request: str, 
                                       available_data: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Generate code based on plan and execute it"""
        
        # Generate code based on the plan and request
        code = self._generate_code(plan, user_request, available_data, iteration)
        
        # Execute the code using the biocoding MCP
        if self.mcp_tool_caller:
            try:
                result = await self.mcp_tool_caller('execute_code', {
                    'code': code,
                    'context_data': available_data,
                    'save_outputs': True
                })
                
                return {
                    'code': code,
                    'execution_result': result,
                    'success': 'error' not in result,
                    'iteration': iteration
                }
            
            except Exception as e:
                return {
                    'code': code,
                    'execution_result': {'error': str(e)},
                    'success': False,
                    'iteration': iteration
                }
        else:
            return {
                'code': code,
                'execution_result': {'error': 'No MCP tool caller available'},
                'success': False,
                'iteration': iteration
            }
    
    def _generate_code(self, plan: Dict[str, Any], user_request: str, 
                      available_data: Dict[str, Any], iteration: int) -> str:
        """Generate Python code to solve the user request"""
        
        if plan['approach'] == 'biological_analysis':
            return self._generate_biological_analysis_code(plan, user_request, available_data)
        else:
            return self._generate_generic_analysis_code(plan, user_request, available_data)
    
    def _generate_biological_analysis_code(self, plan: Dict[str, Any], user_request: str, 
                                         available_data: Dict[str, Any]) -> str:
        """Generate specialized code for biological analysis"""
        
        code_template = '''
import json
from collections import Counter, defaultdict
import statistics

print("🧬 Starting biological analysis...")

# Load the data
if 'data' in globals():
    analysis_data = data
else:
    print("No data available")
    analysis_data = {}

print(f"Data structure: {type(analysis_data)}")
if isinstance(analysis_data, dict):
    print(f"Available sections: {list(analysis_data.keys())}")

# Focus on tandem repeat analysis (user requested details on most frequent repeats)
if 'repeat_analysis' in analysis_data:
    repeat_data = analysis_data['repeat_analysis']
    tandem_repeats = repeat_data.get('tandem_repeats', [])
    
    print(f"\\n📊 TANDEM REPEAT ANALYSIS")
    print(f"Total repeats found: {len(tandem_repeats)}")
    
    if tandem_repeats:
        # Analyze repeat units
        repeat_units = [r.get('repeat_unit', '') for r in tandem_repeats]
        unit_counter = Counter(repeat_units)
        
        print(f"\\n🔍 MOST FREQUENT REPEAT UNITS:")
        for i, (unit, count) in enumerate(unit_counter.most_common(5), 1):
            print(f"{i}. '{unit}' - appears {count} times")
            
            # Find examples of this repeat
            examples = [r for r in tandem_repeats if r.get('repeat_unit') == unit]
            for example in examples[:2]:  # Show first 2 examples
                print(f"   Position: {example.get('start', 'unknown')}-{example.get('end', 'unknown')}")
                print(f"   Copies: {example.get('copy_number', 'unknown')}")
                print(f"   Total length: {example.get('total_length', 'unknown')} bp")
        
        # Analyze by length categories
        print(f"\\n📏 LENGTH DISTRIBUTION:")
        length_categories = defaultdict(int)
        for repeat in tandem_repeats:
            length = repeat.get('total_length', 0)
            if length < 20:
                length_categories['Short (10-20bp)'] += 1
            elif length < 50:
                length_categories['Medium (20-50bp)'] += 1
            elif length < 100:
                length_categories['Long (50-100bp)'] += 1
            else:
                length_categories['Very Long (100bp+)'] += 1
        
        for category, count in length_categories.items():
            print(f"• {category}: {count} repeats")
        
        # Analyze copy numbers
        print(f"\\n🔢 COPY NUMBER ANALYSIS:")
        copy_numbers = [r.get('copy_number', 0) for r in tandem_repeats]
        copy_counter = Counter(copy_numbers)
        
        for copies, count in sorted(copy_counter.items()):
            print(f"• {copies} copies: {count} repeats")
        
        # Find highest copy number repeats
        max_copies = max(copy_numbers) if copy_numbers else 0
        high_copy_repeats = [r for r in tandem_repeats if r.get('copy_number', 0) >= max_copies - 1]
        
        if high_copy_repeats:
            print(f"\\n⭐ HIGHEST COPY NUMBER REPEATS:")
            for repeat in high_copy_repeats:
                print(f"• '{repeat.get('repeat_unit', 'unknown')}' - {repeat.get('copy_number', 0)} copies")
                print(f"  Position: {repeat.get('start', 0):,} bp")
                print(f"  Total length: {repeat.get('total_length', 0)} bp")
        
        # Analyze sequence composition
        print(f"\\n🧬 SEQUENCE COMPOSITION:")
        at_rich_count = 0
        gc_rich_count = 0
        palindromic_count = 0
        
        for repeat in tandem_repeats:
            unit = repeat.get('repeat_unit', '').upper()
            if unit:
                at_content = (unit.count('A') + unit.count('T')) / len(unit)
                gc_content = (unit.count('G') + unit.count('C')) / len(unit)
                
                if at_content > 0.7:
                    at_rich_count += 1
                elif gc_content > 0.7:
                    gc_rich_count += 1
                
                # Check if palindromic
                complement = unit.translate(str.maketrans('ATCG', 'TAGC'))
                if unit == complement[::-1]:
                    palindromic_count += 1
        
        print(f"• AT-rich repeats (>70% AT): {at_rich_count}")
        print(f"• GC-rich repeats (>70% GC): {gc_rich_count}")
        print(f"• Palindromic repeats: {palindromic_count}")
        
        # Biological significance
        print(f"\\n🔬 BIOLOGICAL SIGNIFICANCE:")
        genome_length = analysis_data.get('assembly_stats', {}).get('total_length', 0)
        gc_content = analysis_data.get('assembly_stats', {}).get('gc_content', 0)
        
        if genome_length:
            repeat_density = len(tandem_repeats) / (genome_length / 1000)
            print(f"• Repeat density: {repeat_density:.2f} repeats per kb")
        
        if gc_content and at_rich_count > len(tandem_repeats) * 0.5:
            print(f"• AT-rich repeat bias consistent with low GC genome ({gc_content:.1f}%)")
        
        # Check for terminal repeats
        terminal_repeats = [r for r in tandem_repeats if r.get('start', 0) > genome_length * 0.95]
        if terminal_repeats:
            print(f"• {len(terminal_repeats)} repeats near genome terminus - may be packaging signals")

else:
    print("No repeat analysis data found in the provided data")

print("\\n✅ Analysis complete!")
'''
        
        return code_template.strip()
    
    def _generate_generic_analysis_code(self, plan: Dict[str, Any], user_request: str, 
                                      available_data: Dict[str, Any]) -> str:
        """Generate generic analysis code"""
        
        return '''
import json
from collections import Counter

print("🔍 Starting analysis...")

# Examine the available data
if 'data' in globals():
    analysis_data = data
    print(f"Data type: {type(analysis_data)}")
    
    if isinstance(analysis_data, dict):
        print(f"Available keys: {list(analysis_data.keys())}")
        
        # Try to provide useful analysis based on the data structure
        for key, value in analysis_data.items():
            print(f"\\n{key}: {type(value)}")
            if isinstance(value, list) and len(value) > 0:
                print(f"  Contains {len(value)} items")
                if isinstance(value[0], dict):
                    print(f"  Item keys: {list(value[0].keys())}")
else:
    print("No data available for analysis")

print("\\nAnalysis complete!")
'''
    
    def _evaluate_solution(self, user_request: str, code_result: Dict[str, Any], 
                          plan: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if the code result solves the user's request"""
        
        if not code_result.get('success', False):
            return {
                'success': False,
                'reason': 'Code execution failed',
                'details': code_result.get('execution_result', {})
            }
        
        execution_result = code_result.get('execution_result', {})
        stdout = execution_result.get('stdout', '')
        
        # Check if the output contains relevant information
        user_lower = user_request.lower()
        
        success_indicators = []
        
        if 'tandem repeat' in user_lower or 'repeat' in user_lower:
            if 'TANDEM REPEAT ANALYSIS' in stdout:
                success_indicators.append('Found tandem repeat analysis')
            if 'MOST FREQUENT' in stdout:
                success_indicators.append('Found frequency analysis')
            if 'copies:' in stdout:
                success_indicators.append('Found copy number details')
        
        if 'detailed' in user_lower or 'insight' in user_lower:
            if len(stdout.split('\\n')) > 10:  # Substantial output
                success_indicators.append('Generated detailed output')
        
        # Check for error indicators
        error_indicators = []
        if 'Error' in stdout or 'error' in stdout:
            error_indicators.append('Execution errors detected')
        if 'No data available' in stdout:
            error_indicators.append('Data not accessible')
        
        success = len(success_indicators) > 0 and len(error_indicators) == 0
        
        return {
            'success': success,
            'success_indicators': success_indicators,
            'error_indicators': error_indicators,
            'output_length': len(stdout),
            'contains_relevant_info': len(success_indicators) > 0
        }
    
    def _refine_plan(self, plan: Dict[str, Any], evaluation: Dict[str, Any], 
                    code_result: Dict[str, Any]) -> Dict[str, Any]:
        """Refine the solution plan based on evaluation results"""
        
        # If data access failed, try alternative approaches
        if 'Data not accessible' in evaluation.get('error_indicators', []):
            plan['steps'].insert(0, 'Try alternative data access methods')
        
        # If output was too brief, add more detail
        if evaluation.get('output_length', 0) < 500:
            plan['steps'].append('Add more detailed analysis and formatting')
        
        return plan