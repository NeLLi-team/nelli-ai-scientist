# Troubleshooting Guide for NeLLi AI Scientists

Common issues and solutions when working with the Universal MCP Agent and FastMCP servers.

## 🚨 Common Issues

### Agent Startup Problems

#### Problem: Agent can't find MCP configuration
```bash
WARNING:__main__:MCP config file not found at: mcp_config.json
⚠️  No MCP servers configured
```

**Solution:**
```bash
# Check if config file exists in correct location
ls -la agents/template/mcp_config.json

# Verify you're running from the correct directory
cd agents/template
pixi run agent-run

# Check config file path in agent.py
grep "mcp_config_path" agents/template/src/agent.py
```

#### Problem: FastMCP import errors
```bash
ModuleNotFoundError: No module named 'fastmcp'
```

**Solution:**
```bash
# Ensure you're in the pixi environment
pixi shell

# Check if FastMCP is installed
pixi list | grep fastmcp

# Install if missing
pixi add fastmcp

# Verify installation
pixi run python -c "import fastmcp; print('FastMCP available')"
```

#### Problem: Agent finds no tools
```bash
INFO:__main__:✅ Discovered 0 tools from 0 servers
```

**Solution:**
1. **Check MCP server configuration:**
   ```bash
   # Verify servers are enabled
   cat agents/template/mcp_config.json | grep -A 5 "enabled.*true"
   ```

2. **Test MCP servers individually:**
   ```bash
   # Test BioPython server
   cd mcps/template/src && python server_fastmcp.py
   
   # Test filesystem server  
   cd mcps/filesystem/src && python server.py
   ```

3. **Check server paths:**
   ```bash
   # Verify FastMCP script paths exist
   ls -la ../../mcps/template/src/server_fastmcp.py
   ls -la ../../mcps/filesystem/src/server.py
   ```

### FastMCP Server Issues

#### Problem: Server fails to start
```bash
Traceback (most recent call last):
  File "server.py", line 5, in <module>
    from fastmcp import FastMCP
ImportError: No module named 'fastmcp'
```

**Solutions:**
1. **Environment check:**
   ```bash
   # Ensure pixi environment is active
   pixi shell
   which python  # Should show pixi environment path
   
   # Test FastMCP import
   python -c "from fastmcp import FastMCP; print('OK')"
   ```

2. **Path issues:**
   ```bash
   # Check current directory
   pwd
   
   # Ensure running from correct location
   cd mcps/template/src
   python server_fastmcp.py
   ```

#### Problem: Tool registration fails
```bash
TypeError: FastMCP.tool() missing 1 required positional argument: 'func'
```

**Solution:**
```python
# ✅ Correct: Use @mcp.tool decorator
@mcp.tool
async def my_tool(param: str) -> dict:
    return {"result": param}

# ❌ Incorrect: Missing decorator syntax
@mcp.tool()  # Extra parentheses
async def my_tool(param: str) -> dict:
    return {"result": param}
```

#### Problem: Async/await syntax errors
```bash
SyntaxError: 'await' outside function
```

**Solution:**
```python
# ✅ Correct: Async tool definition
@mcp.tool
async def analyze_data(data: str) -> dict:
    result = await process_data(data)  # Await inside async function
    return {"result": result}

# ❌ Incorrect: Missing async keyword
@mcp.tool
def analyze_data(data: str) -> dict:
    result = await process_data(data)  # Can't await in sync function
    return {"result": result}
```

### Tool Execution Problems

#### Problem: Tool parameter validation fails
```bash
ValidationError: 1 validation error for ToolParams
sequence
  field required (type=value_error.missing)
```

**Solution:**
1. **Check tool schema:**
   ```python
   # In your MCP server, verify required parameters
   @mcp.tool
   async def sequence_stats(sequence: str, sequence_type: str = "dna") -> dict:
       """Parameters: sequence (required), sequence_type (optional)"""
   ```

2. **Check agent tool call:**
   ```python
   # Ensure all required parameters are provided
   result = await agent._call_mcp_tool(
       "sequence_stats", 
       sequence="ATCGATCG",  # Required parameter
       sequence_type="dna"   # Optional parameter
   )
   ```

#### Problem: write_json_report expects dict but gets string
```bash
ERROR:__main__:Tool call failed: Error calling tool 'write_json_report': 1 validation error for call[write_json_report]
data
  Input should be a valid dictionary [type=dict_type, input_value='The data from...', input_type=str]
```

**Solution:**
This happens when the agent tries to pass a string description instead of actual data. The agent now supports tool chaining:

1. **Use ANALYSIS_RESULTS placeholder:**
   ```json
   {
     "suggested_tools": [
       {
         "tool_name": "analyze_fasta_file",
         "parameters": {"file_path": "../../example/file.fna"}
       },
       {
         "tool_name": "write_json_report", 
         "parameters": {
           "data": "ANALYSIS_RESULTS",
           "output_path": "../../reports/analysis.json"
         }
       }
     ]
   }
   ```

2. **Create reports directory first:**
   ```json
   {
     "tool_name": "create_directory",
     "parameters": {"path": "../../reports"}
   }
   ```

3. **The agent automatically:**
   - Replaces "ANALYSIS_RESULTS" with actual data from previous tool
   - Passes the dictionary result to write_json_report
   - Saves all reports to the reports/ subdirectory

#### Problem: Tool execution timeout
```bash
TimeoutError: Tool execution exceeded maximum time limit
```

**Solutions:**
1. **Increase timeout in agent:**
   ```python
   # In agent.py, modify timeout settings
   async with Client(connection_params, timeout=60) as client:
       result = await client.call_tool(tool_name, kwargs)
   ```

2. **Optimize slow tools:**
   ```python
   # Use async patterns for I/O operations
   @mcp.tool
   async def fast_file_processing(file_paths: list) -> dict:
       # Process files concurrently
       tasks = [process_single_file(path) for path in file_paths]
       results = await asyncio.gather(*tasks)
       return {"results": results}
   ```

### LLM Integration Issues

#### Problem: CBORG API connection fails
```bash
ConnectionError: Unable to connect to CBORG API
```

**Solutions:**
1. **Check API credentials:**
   ```bash
   # Verify environment variables
   echo $CBORG_API_KEY
   
   # Check .env file
   cat .env | grep CBORG
   ```

2. **Test connection:**
   ```python
   # Test CBORG API directly
   from agents.template.src.llm_interface import LLMInterface, LLMProvider
   
   llm = LLMInterface(provider=LLMProvider.CBORG)
   response = await llm.generate("Hello, world!")
   print(response)
   ```

#### Problem: LLM response parsing fails
```bash
JSONDecodeError: Expecting ',' delimiter: line 1 column 45 (char 44)
```

**Solutions:**
1. **Check prompt format:**
   ```python
   # Ensure prompts request valid JSON
   prompt = '''
   Respond in valid JSON format:
   {
     "response_type": "direct_answer" or "use_tools",
     "direct_answer": "your response here"
   }
   '''
   ```

2. **Add JSON cleaning:**
   ```python
   # In agent.py, clean LLM responses
   response = response.strip()
   if response.startswith("```json"):
       response = response[7:]
   if response.endswith("```"):
       response = response[:-3]
   ```

### Memory and Performance Issues

#### Problem: Agent consumes too much memory
```bash
MemoryError: Unable to allocate array
```

**Solutions:**
1. **Monitor memory usage:**
   ```python
   import psutil
   
   def check_memory():
       process = psutil.Process()
       memory_mb = process.memory_info().rss / 1024 / 1024
       print(f"Memory usage: {memory_mb:.1f} MB")
   ```

2. **Implement memory management:**
   ```python
   # Process large datasets in chunks
   async def process_large_dataset(data: list, chunk_size: int = 100):
       for i in range(0, len(data), chunk_size):
           chunk = data[i:i + chunk_size]
           await process_chunk(chunk)
           
           # Force garbage collection
           import gc
           gc.collect()
   ```

#### Problem: Slow tool discovery
```bash
INFO:__main__:🔧 Discovering tools from: BioPython Tools
# ... long pause ...
INFO:__main__:✅ Loaded 8 tools
```

**Solutions:**
1. **Cache tool discovery:**
   ```python
   # In agent.py, implement caching
   @lru_cache(maxsize=10)
   async def _discover_server_tools_cached(self, server_id: str):
       return await self._discover_server_tools(server_id, server_config)
   ```

2. **Reduce logging verbosity:**
   ```python
   # In FastMCP servers, reduce startup logging
   logging.getLogger('fastmcp').setLevel(logging.WARNING)
   ```

## 🔧 Debugging Techniques

### Enable Detailed Logging

```python
# In agent.py, increase logging level
import logging
logging.basicConfig(level=logging.DEBUG)

# For FastMCP servers
logging.getLogger('fastmcp').setLevel(logging.DEBUG)
```

### Test Individual Components

1. **Test MCP Server Directly:**
   ```bash
   cd mcps/template/src
   python -c "
   import asyncio
   from server_fastmcp import mcp
   
   async def test():
       tools = list(mcp.tool_registry.keys())
       print(f'Available tools: {tools}')
       
       result = await mcp.tool_registry['sequence_stats']('ATCG', 'dna')
       print(f'Test result: {result}')
   
   asyncio.run(test())
   "
   ```

2. **Test Agent Components:**
   ```python
   # Test prompt manager
   from agents.template.src.prompt_manager import PromptManager
   pm = PromptManager()
   prompt = pm.load_prompt("tool_selection")
   print(prompt[:200])
   
   # Test LLM interface
   from agents.template.src.llm_interface import LLMInterface
   llm = LLMInterface()
   response = await llm.generate("Test message")
   print(response)
   ```

### Network and Connectivity

1. **Test FastMCP Client Connection:**
   ```python
   from fastmcp import Client
   
   async def test_connection():
       script_path = "../../mcps/template/src/server_fastmcp.py"
       try:
           async with Client(script_path) as client:
               tools = await client.list_tools()
               print(f"Connected successfully, found {len(tools)} tools")
       except Exception as e:
           print(f"Connection failed: {e}")
   
   asyncio.run(test_connection())
   ```

2. **Check Path Resolution:**
   ```python
   import os
   from pathlib import Path
   
   # Check if paths resolve correctly
   config_path = Path("mcp_config.json").resolve()
   print(f"Config path: {config_path}")
   print(f"Exists: {config_path.exists()}")
   
   script_path = Path("../../mcps/template/src/server_fastmcp.py").resolve()
   print(f"Script path: {script_path}")
   print(f"Exists: {script_path.exists()}")
   ```

## 🏥 Health Checks

### System Health Check Script

```python
#!/usr/bin/env python3
# scripts/health_check.py

import asyncio
import json
import sys
from pathlib import Path

async def health_check():
    """Comprehensive system health check"""
    
    results = {
        "environment": {},
        "configuration": {},
        "mcp_servers": {},
        "agent": {},
        "overall_status": "unknown"
    }
    
    # 1. Environment Check
    try:
        import fastmcp
        results["environment"]["fastmcp"] = "✅ Available"
    except ImportError:
        results["environment"]["fastmcp"] = "❌ Missing"
    
    try:
        from agents.template.src.agent import UniversalMCPAgent
        results["environment"]["agent"] = "✅ Available"
    except ImportError as e:
        results["environment"]["agent"] = f"❌ Import failed: {e}"
    
    # 2. Configuration Check
    config_path = Path("agents/template/mcp_config.json")
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            results["configuration"]["mcp_config"] = "✅ Valid JSON"
            results["configuration"]["servers_count"] = len(config.get("mcp_servers", {}))
        except json.JSONDecodeError as e:
            results["configuration"]["mcp_config"] = f"❌ Invalid JSON: {e}"
    else:
        results["configuration"]["mcp_config"] = "❌ File not found"
    
    # 3. MCP Server Check
    if "servers_count" in results["configuration"]:
        with open(config_path) as f:
            config = json.load(f)
        
        for server_id, server_config in config.get("mcp_servers", {}).items():
            if server_config.get("enabled", False):
                script_path = Path(server_config.get("fastmcp_script", ""))
                if script_path.exists():
                    try:
                        # Test server import
                        spec = importlib.util.spec_from_file_location("test_server", script_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        results["mcp_servers"][server_id] = "✅ Loads successfully"
                    except Exception as e:
                        results["mcp_servers"][server_id] = f"❌ Load failed: {e}"
                else:
                    results["mcp_servers"][server_id] = f"❌ Script not found: {script_path}"
            else:
                results["mcp_servers"][server_id] = "⚠️ Disabled"
    
    # 4. Overall Status
    errors = [v for v in results.values() if isinstance(v, dict) and any("❌" in str(val) for val in v.values())]
    if not errors:
        results["overall_status"] = "✅ Healthy"
    else:
        results["overall_status"] = "❌ Issues detected"
    
    return results

if __name__ == "__main__":
    results = asyncio.run(health_check())
    print(json.dumps(results, indent=2))
    
    if "❌" in results["overall_status"]:
        sys.exit(1)
```

### Performance Monitoring

```python
import time
import asyncio
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            success = True
        except Exception as e:
            result = {"error": str(e)}
            success = False
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Function: {func.__name__}")
        print(f"Duration: {duration:.3f}s")
        print(f"Success: {success}")
        
        if not success:
            print(f"Error: {result.get('error')}")
        
        return result
    
    return wrapper

# Usage
@monitor_performance
async def test_tool_execution():
    """Test tool execution performance"""
    from agents.template.src.agent import UniversalMCPAgent, AgentConfig
    
    config = AgentConfig(name="test-agent")
    agent = UniversalMCPAgent(config)
    await agent.initialize()
    
    # Test tool execution
    result = await agent._call_mcp_tool("sequence_stats", sequence="ATCGATCG", sequence_type="dna")
    return result
```

## 🆘 Getting Help

### Before Asking for Help

1. **Check the logs** - Enable debug logging and examine output
2. **Verify environment** - Run health check script
3. **Test components individually** - Isolate the failing component
4. **Check documentation** - Review relevant guides and examples
5. **Search existing issues** - Look for similar problems

### Creating Effective Bug Reports

```markdown
## Bug Report Template

**Environment:**
- OS: [Linux/macOS/Windows]
- Python version: [3.11.x]
- Pixi version: [x.x.x]
- FastMCP version: [x.x.x]

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Steps to Reproduce:**
1. Clone repository
2. Run `pixi install`
3. Execute `pixi run agent-run`
4. [Additional steps]

**Error Messages:**
```
[Paste full error traceback here]
```

**Configuration:**
```json
[Paste relevant config sections]
```

**Additional Context:**
[Any other relevant information]
```

### Common Resources

- **Documentation**: `/docs/` directory
- **Examples**: `/examples/` directory
- **Test Files**: `/tests/` directory
- **Health Check**: `scripts/health_check.py`
- **Configuration**: `agents/template/mcp_config.json`

Most issues with the NeLLi AI Scientist Agent Template can be resolved by following this troubleshooting guide. The Universal MCP Agent architecture and FastMCP integration are designed to be robust, but proper configuration and environment setup are essential for smooth operation.