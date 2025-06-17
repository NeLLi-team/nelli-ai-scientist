"""
BioCoding MCP Server using FastMCP
Interactive code creation, execution, and analysis for biological data science
Empowers agents to write, test, iterate, and reflect on scientific code
"""

from fastmcp import FastMCP
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tools import BioCodingAnalyzer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Reduce FastMCP logging noise during startup
logging.getLogger('fastmcp').setLevel(logging.WARNING)
logging.getLogger('mcp').setLevel(logging.WARNING)

# Create FastMCP server
mcp = FastMCP("BioCoding - Interactive Scientific Code Generation 🧬💻")
analyzer = BioCodingAnalyzer()

@mcp.tool
async def create_analysis_code(task_description: str, 
                              data_context: dict = None,
                              libraries_hint: list = None) -> dict:
    """Generate Python code for a given analysis task
    
    This tool helps create well-structured scientific Python code based on
    natural language descriptions. It analyzes the task, determines required
    libraries, and generates appropriate code templates.
    
    Args:
        task_description: Natural language description of what the code should do
        data_context: Optional data that the code will work with
        libraries_hint: Optional list of libraries to prioritize
        
    Returns:
        Generated code, task analysis, and execution readiness status
    """
    return await analyzer.create_analysis_code(
        task_description=task_description,
        data_context=data_context,
        libraries_hint=libraries_hint
    )

@mcp.tool
async def execute_code(code: str, 
                      context_data: dict = None,
                      timeout: int = 30,
                      save_outputs: bool = True) -> dict:
    """Execute Python code in a sandboxed environment
    
    Safely executes Python code with access to scientific libraries like
    numpy, pandas, matplotlib, sklearn, and BioPython. Captures output,
    saves plots, and extracts results.
    
    Args:
        code: Python code to execute
        context_data: Optional data made available as 'data' variable
        timeout: Maximum execution time in seconds
        save_outputs: Whether to save outputs, plots, and reports
        
    Returns:
        Execution results including stdout, variables created, plots saved
    """
    return await analyzer.execute_code(
        code=code,
        context_data=context_data,
        timeout=timeout,
        save_outputs=save_outputs
    )

@mcp.tool
async def analyze_code_quality(code: str) -> dict:
    """Analyze code quality, complexity, and suggest improvements
    
    Performs static analysis on Python code to check syntax, measure
    complexity, identify security issues, and suggest best practices.
    
    Args:
        code: Python code to analyze
        
    Returns:
        Quality metrics, suggestions, and identified issues
    """
    return await analyzer.analyze_code_quality(code=code)

@mcp.tool
async def reflect_on_results(execution_result: dict, goal: str = None) -> dict:
    """Analyze execution results and suggest improvements
    
    Reflects on code execution results to provide insights, identify
    patterns in the data, assess goal achievement, and suggest next steps.
    
    Args:
        execution_result: Result from execute_code tool
        goal: Optional description of what the code aimed to achieve
        
    Returns:
        Reflection including insights, improvements, and next steps
    """
    return await analyzer.reflect_on_results(
        execution_result=execution_result,
        goal=goal
    )

@mcp.tool
async def iterate_on_code(original_code: str,
                         feedback: str,
                         previous_result: dict = None) -> dict:
    """Improve code based on feedback and previous results
    
    Takes existing code and improves it based on user feedback and
    analysis of previous execution results. Handles common improvements
    like adding visualizations, fixing errors, or enhancing analysis.
    
    Args:
        original_code: The code to improve
        feedback: Natural language feedback on what to change
        previous_result: Optional previous execution result to learn from
        
    Returns:
        Improved code with list of changes made
    """
    return await analyzer.iterate_on_code(
        original_code=original_code,
        feedback=feedback,
        previous_result=previous_result
    )

@mcp.tool
async def create_notebook(title: str,
                         cells: list,
                         save_path: str = None) -> dict:
    """Create a Jupyter notebook with the given cells
    
    Generates a complete Jupyter notebook with markdown and code cells,
    useful for creating reproducible analysis workflows.
    
    Args:
        title: Notebook title
        cells: List of cell dictionaries with 'type' and 'content'
        save_path: Optional custom save path
        
    Returns:
        Path to created notebook and summary
    """
    return await analyzer.create_notebook(
        title=title,
        cells=cells,
        save_path=save_path
    )

@mcp.tool
async def create_test_suite(function_code: str,
                           test_cases: list = None) -> dict:
    """Generate test suite for given function code
    
    Creates comprehensive unit tests for a Python function, including
    setup, teardown, and multiple test cases with different inputs.
    
    Args:
        function_code: The function code to test
        test_cases: Optional list of specific test cases
        
    Returns:
        Generated test code and test file path
    """
    return await analyzer.create_test_suite(
        function_code=function_code,
        test_cases=test_cases
    )

@mcp.tool
async def introspect_data_structure(data: dict, analysis_goal: str = None) -> dict:
    """Analyze data structure and suggest appropriate analysis approaches
    
    This tool examines any data structure and intelligently suggests
    analysis methods, visualizations, and code approaches based on
    the data's characteristics and patterns.
    
    Args:
        data: Data structure to analyze
        analysis_goal: Optional description of what you want to achieve
        
    Returns:
        Data insights, suggested analyses, and code recommendations
    """
    return await analyzer.introspect_data_structure(
        data=data,
        analysis_goal=analysis_goal
    )

@mcp.tool
async def adaptive_code_generation(data: dict, 
                                  goal: str,
                                  previous_attempts: list = None) -> dict:
    """Generate adaptive code that intelligently handles any data structure
    
    This is the core smart code generation tool that can analyze any
    data structure and generate appropriate analysis code automatically.
    It learns from previous attempts and adapts to handle new scenarios.
    
    Args:
        data: The data to analyze
        goal: What you want to achieve with the analysis
        previous_attempts: Optional list of previous code execution attempts to learn from
        
    Returns:
        Adaptive code that's tailored to your specific data and goal
    """
    return await analyzer.adaptive_code_generation(
        data=data,
        goal=goal,
        previous_attempts=previous_attempts
    )

@mcp.tool
async def get_code_examples(topic: str) -> dict:
    """Get code examples for common bioinformatics tasks
    
    Provides working code examples for various scientific computing
    and bioinformatics tasks.
    
    Args:
        topic: Topic like 'data_visualization', 'ml_classification', 
               'sequence_analysis', 'statistical_tests'
        
    Returns:
        Code examples with explanations
    """
    examples = {
        "data_visualization": {
            "description": "Creating scientific visualizations",
            "code": """
# Scientific Data Visualization Example
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'gene_expression': np.random.lognormal(3, 1.5, 1000),
    'protein_level': np.random.lognormal(2, 1, 1000),
    'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], 1000),
    'treatment': np.random.choice(['Control', 'Treated'], 1000)
})

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Distribution plot
sns.histplot(data=data, x='gene_expression', hue='treatment', 
             kde=True, ax=axes[0,0], bins=30)
axes[0,0].set_title('Gene Expression Distribution by Treatment')

# 2. Scatter plot with regression
sns.scatterplot(data=data, x='gene_expression', y='protein_level', 
                hue='cell_type', alpha=0.6, ax=axes[0,1])
axes[0,1].set_title('Gene Expression vs Protein Level')

# 3. Box plot
sns.boxplot(data=data, x='cell_type', y='gene_expression', 
            hue='treatment', ax=axes[1,0])
axes[1,0].set_title('Expression by Cell Type and Treatment')

# 4. Violin plot
sns.violinplot(data=data, x='treatment', y='protein_level', 
               split=True, ax=axes[1,1])
axes[1,1].set_title('Protein Level Distribution')

plt.tight_layout()
plt.show()

# Statistical summary
print(data.groupby(['cell_type', 'treatment']).agg({
    'gene_expression': ['mean', 'std'],
    'protein_level': ['mean', 'std']
}).round(2))
"""
        },
        "ml_classification": {
            "description": "Machine learning classification pipeline",
            "code": """
# Machine Learning Classification Example
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic biological data
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, n_redundant=5,
                          n_classes=3, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = rf_model.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"\\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': [f'Feature_{i}' for i in range(X.shape[1])],
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# Feature importance
feature_importance.head(10).plot(x='feature', y='importance', 
                                 kind='bar', ax=ax2)
ax2.set_title('Top 10 Feature Importances')
ax2.set_xlabel('Feature')
ax2.set_ylabel('Importance')

plt.tight_layout()
plt.show()
"""
        },
        "sequence_analysis": {
            "description": "Biological sequence analysis",
            "code": """
# Biological Sequence Analysis Example
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction, molecular_weight
import matplotlib.pyplot as plt
import numpy as np

# Create sample sequences
sequences = [
    Seq("ATCGATCGATCGTAGCTACGTACGATCG"),
    Seq("GCTAGCTAGCTAGCTAGCTAGCTAGCTA"),
    Seq("ATATATATATGCGCGCGCGCATATATA")
]

# Analyze sequences
results = []
for i, seq in enumerate(sequences):
    result = {
        'seq_id': f'Seq_{i+1}',
        'length': len(seq),
        'gc_content': gc_fraction(seq) * 100,
        'molecular_weight': molecular_weight(seq, 'DNA'),
        'complement': str(seq.complement()),
        'reverse_complement': str(seq.reverse_complement())
    }
    results.append(result)

# Display results
import pandas as pd
df_results = pd.DataFrame(results)
print("Sequence Analysis Results:")
print(df_results.to_string(index=False))

# Codon usage analysis
def analyze_codons(seq):
    codons = {}
    for i in range(0, len(seq) - 2, 3):
        codon = str(seq[i:i+3])
        codons[codon] = codons.get(codon, 0) + 1
    return codons

# K-mer analysis
def kmer_frequency(seq, k=3):
    kmers = {}
    for i in range(len(seq) - k + 1):
        kmer = str(seq[i:i+k])
        kmers[kmer] = kmers.get(kmer, 0) + 1
    return kmers

# Visualize GC content along sequence
window_size = 10
seq = sequences[0]
gc_values = []
positions = []

for i in range(0, len(seq) - window_size + 1):
    window = seq[i:i+window_size]
    gc_values.append(gc_fraction(window) * 100)
    positions.append(i + window_size/2)

plt.figure(figsize=(10, 6))
plt.plot(positions, gc_values, 'b-', linewidth=2)
plt.axhline(y=np.mean(gc_values), color='r', linestyle='--', 
            label=f'Mean GC: {np.mean(gc_values):.1f}%')
plt.xlabel('Position')
plt.ylabel('GC Content (%)')
plt.title('GC Content Along Sequence (10bp window)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
        },
        "statistical_tests": {
            "description": "Statistical analysis and hypothesis testing",
            "code": """
# Statistical Analysis Example
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample biological data
np.random.seed(42)
control_group = np.random.normal(100, 15, 50)  # Control expression levels
treatment_group = np.random.normal(120, 20, 50)  # Treatment expression levels

# Create DataFrame
data = pd.DataFrame({
    'expression_level': np.concatenate([control_group, treatment_group]),
    'group': ['Control']*50 + ['Treatment']*50
})

# 1. Descriptive statistics
print("Descriptive Statistics:")
print(data.groupby('group')['expression_level'].describe())

# 2. Normality tests
print("\\nNormality Tests:")
for group in ['Control', 'Treatment']:
    group_data = data[data['group'] == group]['expression_level']
    stat, p_value = stats.shapiro(group_data)
    print(f"{group}: Shapiro-Wilk p-value = {p_value:.4f}")

# 3. Hypothesis testing
# T-test (if normal)
t_stat, t_pvalue = stats.ttest_ind(control_group, treatment_group)
print(f"\\nT-test: t-statistic = {t_stat:.4f}, p-value = {t_pvalue:.4f}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, u_pvalue = stats.mannwhitneyu(control_group, treatment_group)
print(f"Mann-Whitney U: U-statistic = {u_stat:.4f}, p-value = {u_pvalue:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(control_group)-1)*np.std(control_group)**2 + 
                      (len(treatment_group)-1)*np.std(treatment_group)**2) / 
                     (len(control_group) + len(treatment_group) - 2))
cohens_d = (np.mean(treatment_group) - np.mean(control_group)) / pooled_std
print(f"\\nEffect size (Cohen's d): {cohens_d:.4f}")

# 4. Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Box plot
sns.boxplot(data=data, x='group', y='expression_level', ax=axes[0,0])
axes[0,0].set_title('Expression Levels by Group')

# Violin plot
sns.violinplot(data=data, x='group', y='expression_level', ax=axes[0,1])
axes[0,1].set_title('Distribution Comparison')

# Q-Q plots
stats.probplot(control_group, dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot: Control Group')

stats.probplot(treatment_group, dist="norm", plot=axes[1,1])
axes[1,1].set_title('Q-Q Plot: Treatment Group')

plt.tight_layout()
plt.show()

# Power analysis
from statsmodels.stats.power import tt_ind_solve_power
power = tt_ind_solve_power(effect_size=cohens_d, nobs1=50, 
                          alpha=0.05, alternative='two-sided')
print(f"\\nStatistical power: {power:.4f}")
"""
        }
    }
    
    if topic in examples:
        return {
            "success": True,
            "topic": topic,
            "example": examples[topic]
        }
    else:
        return {
            "success": True,
            "available_topics": list(examples.keys()),
            "message": f"Topic '{topic}' not found. Choose from available topics."
        }

# Add resources for code templates and documentation
@mcp.resource("code://templates")
def get_code_templates():
    """Get available code templates"""
    return {
        "analysis_pipeline": "Complete data analysis pipeline template",
        "visualization_suite": "Comprehensive visualization template",
        "ml_workflow": "Machine learning workflow template",
        "statistical_analysis": "Statistical testing template",
        "bioinformatics_toolkit": "Bioinformatics analysis template"
    }

@mcp.resource("code://help")
def get_coding_help():
    """Get help information for BioCoding tools"""
    return {
        "tools_available": [
            "create_analysis_code",
            "execute_code", 
            "analyze_code_quality",
            "reflect_on_results",
            "iterate_on_code",
            "create_notebook",
            "create_test_suite",
            "get_code_examples",
            "introspect_data_structure",
            "adaptive_code_generation"
        ],
        "capabilities": [
            "Generate scientific Python code from descriptions",
            "Execute code in sandboxed environment", 
            "Analyze code quality and complexity",
            "Reflect on results and suggest improvements",
            "Iterate on code based on feedback",
            "Create Jupyter notebooks",
            "Generate comprehensive test suites",
            "Access curated code examples",
            "Intelligently analyze any data structure",
            "Generate adaptive code for unpredefined tasks",
            "Learn from previous attempts and failures",
            "Auto-discover patterns in unknown data"
        ],
        "libraries_available": [
            "numpy, pandas - Data manipulation",
            "matplotlib, seaborn, plotly - Visualization",
            "scikit-learn - Machine learning",
            "scipy - Scientific computing",
            "BioPython - Biological sequence analysis",
            "black, autopep8 - Code formatting"
        ],
        "workflow": [
            "TRADITIONAL: Describe task → generate code → execute → reflect → iterate",
            "ADAPTIVE: Provide data + goal → introspect_data_structure → adaptive_code_generation → execute",
            "SMART: Any unknown data → agent automatically discovers patterns and generates appropriate analysis"
        ],
        "adaptive_features": [
            "🧠 Data structure introspection - understands any data format",
            "🔄 Learning from failures - improves with each attempt", 
            "🎯 Goal-oriented generation - code adapts to your specific objectives",
            "📊 Auto-pattern discovery - finds insights in unknown data structures",
            "🔧 Self-improving - gets smarter with each interaction"
        ]
    }

if __name__ == "__main__":
    # Run with stdio transport by default
    mcp.run()