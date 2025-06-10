# Standards and Best Practices for NeLLi AI Scientists

Guidelines for developing high-quality, secure, and maintainable AI scientist agents using the Universal MCP Agent architecture.

## 🏗️ Architecture Standards

### Universal Agent Design Principles

1. **Tool Agnostic Design**
   ```python
   # ✅ Good: Dynamic tool discovery
   tools = await agent.discover_all_tools()
   for tool in tools:
       if tool.matches_intent(user_request):
           await agent.execute_tool(tool.name, **parameters)
   
   # ❌ Bad: Hardcoded tool references  
   if "sequence" in user_request:
       await agent.call_biopython_tool(...)
   ```

2. **Async-First Development**
   ```python
   # ✅ Good: Concurrent tool execution
   async def analyze_multiple_files(file_paths):
       tasks = [analyze_single_file(path) for path in file_paths]
       results = await asyncio.gather(*tasks)
       return results
   
   # ❌ Bad: Sequential blocking
   def analyze_multiple_files_blocking(file_paths):
       results = []
       for path in file_paths:
           result = analyze_single_file(path)  # Blocks
           results.append(result)
       return results
   ```

3. **External Configuration**
   ```python
   # ✅ Good: External prompt management
   prompt = self.prompt_manager.format_prompt(
       "tool_selection",
       tools_context=tools_context,
       user_input=user_input
   )
   
   # ❌ Bad: Hardcoded prompts
   prompt = f"You are an agent with tools: {tools}. User asks: {user_input}"
   ```

### FastMCP Server Standards

1. **Tool Function Design**
   ```python
   # ✅ Good: Well-documented async tool
   @mcp.tool
   async def analyze_sequence(sequence: str, analysis_type: str = "basic") -> dict:
       """Analyze biological sequence with specified method
       
       Args:
           sequence: DNA, RNA, or protein sequence (required)
           analysis_type: Type of analysis - basic, detailed, or comprehensive
           
       Returns:
           dict: Analysis results with metadata
           
       Raises:
           ValueError: If sequence contains invalid characters
           TypeError: If analysis_type is not supported
       """
       # Validate inputs
       if not sequence or not isinstance(sequence, str):
           raise ValueError("Sequence must be a non-empty string")
       
       # Async processing
       result = await perform_analysis(sequence, analysis_type)
       
       # Structured return
       return {
           "success": True,
           "sequence_length": len(sequence),
           "analysis_type": analysis_type,
           "results": result,
           "timestamp": datetime.now().isoformat(),
           "version": "1.0"
       }
   ```

2. **Error Handling Standards**
   ```python
   @mcp.tool
   async def robust_tool(data: str) -> dict:
       """Tool with comprehensive error handling"""
       try:
           # Validate inputs
           if not data:
               return {
                   "success": False,
                   "error": "No data provided",
                   "error_type": "ValidationError"
               }
           
           # Process data
           result = await process_data(data)
           
           return {
               "success": True,
               "data": result,
               "metadata": {"processed_at": datetime.now().isoformat()}
           }
           
       except ValueError as e:
           return {
               "success": False,
               "error": str(e),
               "error_type": "ValidationError"
           }
       except Exception as e:
           logger.exception(f"Unexpected error in robust_tool: {e}")
           return {
               "success": False,
               "error": "Internal processing error",
               "error_type": "InternalError"
           }
   ```

## 🔒 Security Standards

### Input Validation

1. **Sanitize All Inputs**
   ```python
   from pathlib import Path
   import re
   
   def validate_file_path(file_path: str) -> str:
       """Safely validate and normalize file paths"""
       # Resolve path and check it's within allowed directories
       resolved_path = Path(file_path).resolve()
       allowed_roots = [Path("/data"), Path("/tmp"), Path("./workspace")]
       
       if not any(str(resolved_path).startswith(str(root)) for root in allowed_roots):
           raise ValueError(f"File path outside allowed directories: {file_path}")
       
       return str(resolved_path)
   
   def validate_sequence(sequence: str) -> str:
       """Validate biological sequence"""
       # Remove whitespace and convert to uppercase
       clean_seq = re.sub(r'\\s+', '', sequence.upper())
       
       # Check for valid characters (DNA example)
       if not re.match(r'^[ATCG]+$', clean_seq):
           raise ValueError("Invalid DNA sequence characters")
       
       # Check reasonable length limits
       if len(clean_seq) > 1000000:  # 1M bases
           raise ValueError("Sequence too long for processing")
       
       return clean_seq
   ```

2. **Prevent Code Injection**
   ```python
   # ✅ Good: Safe expression evaluation
   import ast
   import operator
   
   def safe_eval_math(expression: str) -> float:
       """Safely evaluate mathematical expressions"""
       allowed_operators = {
           ast.Add: operator.add,
           ast.Sub: operator.sub,
           ast.Mult: operator.mul,
           ast.Div: operator.truediv,
           ast.Pow: operator.pow,
           ast.USub: operator.neg,
       }
       
       def eval_node(node):
           if isinstance(node, ast.Num):
               return node.n
           elif isinstance(node, ast.BinOp):
               return allowed_operators[type(node.op)](
                   eval_node(node.left), 
                   eval_node(node.right)
               )
           elif isinstance(node, ast.UnaryOp):
               return allowed_operators[type(node.op)](eval_node(node.operand))
           else:
               raise ValueError(f"Unsupported operation: {type(node)}")
       
       tree = ast.parse(expression, mode='eval')
       return eval_node(tree.body)
   
   # ❌ Bad: Unsafe eval
   def unsafe_eval(expression: str):
       return eval(expression)  # Can execute arbitrary code!
   ```

### File Operations Security

```python
import os
from pathlib import Path

class SecureFileHandler:
    """Secure file operations with sandbox constraints"""
    
    def __init__(self, allowed_paths: List[str]):
        self.allowed_paths = [Path(p).resolve() for p in allowed_paths]
    
    def _validate_path(self, file_path: str) -> Path:
        """Validate path is within sandbox"""
        resolved_path = Path(file_path).resolve()
        
        # Check if path is within allowed directories
        for allowed_path in self.allowed_paths:
            try:
                resolved_path.relative_to(allowed_path)
                return resolved_path
            except ValueError:
                continue
        
        raise ValueError(f"Path outside sandbox: {file_path}")
    
    async def safe_read_file(self, file_path: str) -> str:
        """Safely read file with validation"""
        validated_path = self._validate_path(file_path)
        
        # Check file size before reading
        file_size = validated_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError("File too large to process")
        
        async with aiofiles.open(validated_path, 'r') as f:
            return await f.read()
    
    async def safe_write_file(self, file_path: str, content: str) -> bool:
        """Safely write file with validation"""
        validated_path = self._validate_path(file_path)
        
        # Check content size
        if len(content.encode('utf-8')) > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError("Content too large to write")
        
        # Ensure directory exists
        validated_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(validated_path, 'w') as f:
            await f.write(content)
        
        return True
```

## 📊 Code Quality Standards

### Type Hints and Documentation

```python
from typing import Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

class AnalysisType(Enum):
    BASIC = "basic"
    DETAILED = "detailed" 
    COMPREHENSIVE = "comprehensive"

@dataclass
class SequenceAnalysisResult:
    """Result of sequence analysis operation"""
    sequence_id: str
    sequence_length: int
    gc_content: float
    analysis_type: AnalysisType
    metadata: Dict[str, Union[str, int, float]]
    success: bool = True
    error_message: Optional[str] = None

async def analyze_sequences(
    sequences: List[str],
    analysis_type: AnalysisType = AnalysisType.BASIC,
    batch_size: int = 10
) -> AsyncGenerator[SequenceAnalysisResult, None]:
    """Analyze multiple sequences with specified analysis type
    
    Args:
        sequences: List of DNA/RNA/protein sequences to analyze
        analysis_type: Type of analysis to perform
        batch_size: Number of sequences to process concurrently
        
    Yields:
        SequenceAnalysisResult: Analysis result for each sequence
        
    Raises:
        ValueError: If sequences list is empty or contains invalid data
        RuntimeError: If analysis fails due to system error
        
    Example:
        >>> sequences = ["ATCGATCG", "GCTAGCTA"]
        >>> async for result in analyze_sequences(sequences):
        ...     print(f"Sequence {result.sequence_id}: GC={result.gc_content}%")
    """
    if not sequences:
        raise ValueError("Sequences list cannot be empty")
    
    # Process in batches for memory efficiency
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        
        # Process batch concurrently
        tasks = [
            analyze_single_sequence(seq, analysis_type, i + j)
            for j, seq in enumerate(batch)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                yield SequenceAnalysisResult(
                    sequence_id="unknown",
                    sequence_length=0,
                    gc_content=0.0,
                    analysis_type=analysis_type,
                    metadata={},
                    success=False,
                    error_message=str(result)
                )
            else:
                yield result
```

### Testing Standards

```python
# tests/test_sequence_analysis.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from mcps.genomics.src.server import mcp

class TestSequenceAnalysis:
    """Test suite for sequence analysis tools"""
    
    @pytest.mark.asyncio
    async def test_gc_content_calculation(self):
        """Test GC content calculation with known sequences"""
        # Test data with expected results
        test_cases = [
            ("ATCG", 50.0),  # 50% GC
            ("AAAA", 0.0),   # 0% GC  
            ("GCGC", 100.0), # 100% GC
            ("", 0.0)        # Empty sequence
        ]
        
        for sequence, expected_gc in test_cases:
            result = await mcp.tool_registry["gc_content"](sequence)
            
            if sequence:
                assert result["gc_content_percent"] == expected_gc
                assert result["sequence_length"] == len(sequence)
                assert "error" not in result
            else:
                assert "error" in result
    
    @pytest.mark.asyncio
    async def test_invalid_sequence_handling(self):
        """Test handling of invalid sequences"""
        invalid_sequences = [
            "ATCGX",     # Invalid character
            "123456",    # Numbers
            "atcg xyz"   # Mixed case with spaces
        ]
        
        for invalid_seq in invalid_sequences:
            result = await mcp.tool_registry["gc_content"](invalid_seq)
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent sequence processing"""
        sequences = [f"ATCG" * (i + 1) for i in range(10)]
        
        # Process concurrently
        tasks = [
            mcp.tool_registry["gc_content"](seq) 
            for seq in sequences
        ]
        results = await asyncio.gather(*tasks)
        
        # Verify all processed successfully
        assert len(results) == len(sequences)
        for i, result in enumerate(results):
            assert "error" not in result
            assert result["sequence_length"] == 4 * (i + 1)
    
    @pytest.mark.asyncio
    async def test_resource_limits(self):
        """Test handling of resource limits"""
        # Very long sequence (should be rejected)
        long_sequence = "A" * 2000000  # 2M bases
        
        result = await mcp.tool_registry["gc_content"](long_sequence)
        assert "error" in result
        assert "too long" in result["error"].lower()
    
    @pytest.fixture
    async def mock_database(self):
        """Mock database for testing"""
        with patch('mcps.genomics.src.server.database') as mock_db:
            mock_db.query.return_value = AsyncMock()
            yield mock_db
```

## 🔄 Development Workflow Standards

### Version Control Practices

1. **Commit Message Format**
   ```
   type(scope): description
   
   feat(agent): add reflective learning capabilities
   fix(mcp): resolve async timeout in sequence analysis
   docs(guides): update FastMCP server development guide
   test(integration): add end-to-end agent workflow tests
   refactor(prompts): extract hardcoded prompts to external files
   ```

2. **Branch Naming Convention**
   ```
   feature/agent-memory-system
   fix/mcp-connection-timeout
   docs/advanced-concepts-guide
   test/concurrent-tool-execution
   ```

### Code Review Checklist

- [ ] **Security**: All inputs validated and sanitized
- [ ] **Async**: Proper use of async/await patterns
- [ ] **Error Handling**: Comprehensive error handling with structured responses
- [ ] **Type Hints**: Complete type annotations
- [ ] **Documentation**: Docstrings for all public functions
- [ ] **Tests**: Unit tests for new functionality
- [ ] **Performance**: No blocking operations in async functions
- [ ] **Standards**: Follows project coding standards

## 📈 Performance Guidelines

### Memory Management

```python
import gc
import psutil
from typing import AsyncGenerator

class ResourceMonitor:
    """Monitor and manage resource usage"""
    
    def __init__(self, memory_limit_mb: int = 1024):
        self.memory_limit_mb = memory_limit_mb
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    
    async def memory_efficient_processing(
        self, 
        large_dataset: List[str]
    ) -> AsyncGenerator[Dict, None]:
        """Process large dataset with memory management"""
        
        chunk_size = 100
        for i in range(0, len(large_dataset), chunk_size):
            chunk = large_dataset[i:i + chunk_size]
            
            # Process chunk
            results = await process_chunk(chunk)
            
            # Yield results
            for result in results:
                yield result
            
            # Memory cleanup
            del results
            gc.collect()
            
            # Check memory usage
            memory_usage = self.check_memory_usage()
            if memory_usage["rss_mb"] > self.memory_limit_mb:
                raise RuntimeError(f"Memory limit exceeded: {memory_usage['rss_mb']:.1f}MB")
```

### Async Best Practices

```python
# ✅ Good: Proper async resource management
async def efficient_batch_processing(items: List[str]) -> List[Dict]:
    semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
    
    async def process_with_semaphore(item: str) -> Dict:
        async with semaphore:
            return await process_item(item)
    
    # Process with controlled concurrency
    tasks = [process_with_semaphore(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to process item {i}: {result}")
        else:
            successful_results.append(result)
    
    return successful_results

# ❌ Bad: Uncontrolled concurrency
async def inefficient_processing(items: List[str]) -> List[Dict]:
    # Could overwhelm system with thousands of concurrent operations
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)
```

## 🧪 Scientific Computing Standards

### Reproducibility

```python
import random
import numpy as np
from datetime import datetime

class ReproducibleAnalysis:
    """Ensure reproducible scientific analysis"""
    
    def __init__(self, seed: int = None):
        self.seed = seed or int(datetime.now().timestamp())
        self.set_random_seeds()
    
    def set_random_seeds(self):
        """Set all random seeds for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        # Set other library seeds as needed
    
    async def reproducible_analysis(self, data: List[str]) -> Dict[str, Any]:
        """Perform analysis with reproducible results"""
        # Reset seeds before analysis
        self.set_random_seeds()
        
        results = await perform_analysis(data)
        
        # Include reproducibility metadata
        results["metadata"] = {
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "environment": {
                "python_version": sys.version,
                "numpy_version": np.__version__
            }
        }
        
        return results
```

### Data Validation

```python
from pydantic import BaseModel, validator
from typing import List, Optional

class BiologicalSequence(BaseModel):
    """Validated biological sequence"""
    
    sequence: str
    sequence_type: str
    organism: Optional[str] = None
    description: Optional[str] = None
    
    @validator('sequence')
    def validate_sequence_content(cls, v, values):
        """Validate sequence contains only valid characters"""
        sequence_type = values.get('sequence_type', '').lower()
        
        if sequence_type == 'dna':
            valid_chars = set('ATCG')
        elif sequence_type == 'rna':
            valid_chars = set('AUCG')
        elif sequence_type == 'protein':
            valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
        else:
            raise ValueError(f"Unknown sequence type: {sequence_type}")
        
        clean_seq = v.upper().replace(' ', '')
        invalid_chars = set(clean_seq) - valid_chars
        
        if invalid_chars:
            raise ValueError(f"Invalid characters for {sequence_type}: {invalid_chars}")
        
        return clean_seq
    
    @validator('sequence_type')
    def validate_sequence_type(cls, v):
        """Validate sequence type"""
        valid_types = {'dna', 'rna', 'protein'}
        if v.lower() not in valid_types:
            raise ValueError(f"Sequence type must be one of: {valid_types}")
        return v.lower()
```

## 🛡️ Error Handling and Logging

### Structured Logging

```python
import logging
import json
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    """Structured logging for scientific workflows"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_analysis_start(self, analysis_id: str, parameters: Dict[str, Any]):
        """Log analysis start with parameters"""
        self.logger.info(json.dumps({
            "event": "analysis_start",
            "analysis_id": analysis_id,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat()
        }))
    
    def log_analysis_result(self, analysis_id: str, result: Dict[str, Any]):
        """Log analysis results"""
        self.logger.info(json.dumps({
            "event": "analysis_complete",
            "analysis_id": analysis_id,
            "result_summary": {
                "success": result.get("success", False),
                "result_count": len(result.get("results", [])),
                "processing_time": result.get("processing_time_seconds")
            },
            "timestamp": datetime.now().isoformat()
        }))
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log errors with context"""
        self.logger.error(json.dumps({
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }))
```

These standards ensure that NeLLi AI Scientist agents are built with quality, security, and maintainability in mind while leveraging the full capabilities of the Universal MCP Agent architecture.