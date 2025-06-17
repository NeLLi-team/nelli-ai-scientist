# BioCoding MCP Server

An interactive MCP server that empowers AI agents to write, execute, analyze, and iterate on scientific Python code. This server transforms agents into capable scientific programmers who can create data analysis pipelines, visualizations, machine learning models, and more.

## 🚀 Key Features

### Code Generation & Execution
- **Natural Language to Code**: Generate Python code from task descriptions
- **Safe Execution**: Run code in sandboxed environment with scientific libraries
- **Output Capture**: Capture stdout, plots, and created variables
- **Error Handling**: Comprehensive error reporting and recovery suggestions

### Code Intelligence
- **Quality Analysis**: Check syntax, complexity, and best practices
- **Reflection**: Analyze execution results and suggest improvements
- **Iteration**: Improve code based on feedback and previous results
- **Testing**: Generate comprehensive test suites for functions

### Scientific Computing
- **Data Science Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn with classification, regression, clustering
- **Bioinformatics**: BioPython for sequence analysis
- **Statistics**: SciPy for statistical tests and analysis

## 📦 Installation

```bash
cd mcps/biocoding
pixi install  # Install all dependencies including scikit-learn
```

## 🛠️ Available Tools

### 1. `create_analysis_code`
Generate Python code from natural language descriptions.

```python
result = await create_analysis_code(
    task_description="Create a function to analyze gene expression data and visualize the results",
    data_context={"genes": ["BRCA1", "TP53", "EGFR"]},
    libraries_hint=["pandas", "matplotlib"]
)
```

### 2. `execute_code`
Execute Python code in a sandboxed environment.

```python
result = await execute_code(
    code="import numpy as np\ndata = np.random.randn(100, 5)\nprint(data.mean())",
    context_data={"threshold": 0.5},
    save_outputs=True
)
```

### 3. `analyze_code_quality`
Analyze code for quality, complexity, and best practices.

```python
analysis = await analyze_code_quality(
    code="def process_data(x): return x*2"
)
# Returns syntax validation, complexity metrics, suggestions
```

### 4. `reflect_on_results`
Analyze execution results and suggest next steps.

```python
reflection = await reflect_on_results(
    execution_result=previous_result,
    goal="Identify outliers in the dataset"
)
```

### 5. `iterate_on_code`
Improve code based on feedback.

```python
improved = await iterate_on_code(
    original_code=code,
    feedback="Add error handling and create a visualization",
    previous_result=execution_result
)
```

### 6. `create_notebook`
Generate Jupyter notebooks.

```python
notebook = await create_notebook(
    title="Gene Expression Analysis",
    cells=[
        {"type": "markdown", "content": "# Data Loading"},
        {"type": "code", "content": "import pandas as pd\ndata = pd.read_csv('genes.csv')"}
    ]
)
```

### 7. `create_test_suite`
Generate unit tests for functions.

```python
tests = await create_test_suite(
    function_code="def add(a, b): return a + b",
    test_cases=[
        {"input": "(2, 3)", "expected": 5, "description": "Basic addition"}
    ]
)
```

### 8. `get_code_examples`
Access curated code examples.

```python
examples = await get_code_examples(
    topic="ml_classification"  # or "data_visualization", "sequence_analysis", "statistical_tests"
)
```

## 🔄 Typical Workflow

1. **Describe Task**: "I need to analyze RNA-seq data and identify differentially expressed genes"
2. **Generate Code**: Use `create_analysis_code` to generate initial code
3. **Execute**: Run with `execute_code` to see results
4. **Reflect**: Use `reflect_on_results` to understand output
5. **Iterate**: Improve with `iterate_on_code` based on feedback
6. **Test**: Create tests with `create_test_suite`
7. **Document**: Generate notebook with `create_notebook`

## 📁 Sandbox Structure

```
sandbox/
├── code/        # Generated code files
├── data/        # Data files
├── plots/       # Saved visualizations
├── models/      # Trained ML models
├── reports/     # Analysis reports
└── notebooks/   # Jupyter notebooks
```

## 🧬 Example: Complete Analysis Pipeline

```python
# 1. Generate analysis code
code_result = await create_analysis_code(
    task_description="Analyze gene expression data, perform PCA, and cluster samples"
)

# 2. Execute the code
exec_result = await execute_code(
    code=code_result["code"],
    context_data={"expression_matrix": data}
)

# 3. Reflect on results
reflection = await reflect_on_results(
    execution_result=exec_result,
    goal="Identify distinct sample clusters"
)

# 4. Iterate based on insights
improved = await iterate_on_code(
    original_code=code_result["code"],
    feedback="Add statistical validation and save cluster assignments"
)

# 5. Create tests
tests = await create_test_suite(
    function_code=improved["improved_code"]
)
```

## 🔐 Security Features

- **Sandboxed Execution**: Code runs in isolated environment
- **Resource Limits**: Timeout and memory constraints
- **Safe Imports**: Only scientific libraries available
- **Output Validation**: Results sanitized before return

## 🎯 Use Cases

- **Data Analysis**: Exploratory data analysis, statistical testing
- **Machine Learning**: Model training, evaluation, feature engineering
- **Bioinformatics**: Sequence analysis, genomic data processing
- **Visualization**: Scientific plots, interactive dashboards
- **Automation**: Batch processing, pipeline creation
- **Education**: Learning by generating and experimenting with code

## 🤝 Integration with AI Agents

This MCP server is designed to make AI agents powerful scientific programmers. Agents can:

1. **Understand Requirements**: Parse natural language analysis requests
2. **Write Code**: Generate appropriate Python code
3. **Test & Debug**: Execute and fix issues iteratively
4. **Optimize**: Improve code based on results
5. **Document**: Create notebooks and reports
6. **Learn**: Build on previous examples and results

The server encourages agents to think like scientists - hypothesize, experiment, analyze, and iterate.