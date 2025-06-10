# Context7 MCP Server Integration Test

This guide shows how to integrate and test the Context7 vector database MCP server with your bioinformatics AI agents.

## What is Context7?

Context7 is a vector database MCP server that provides:
- **Semantic search** capabilities
- **Vector storage** for text and documents
- **Knowledge retrieval** for AI applications
- **Embeddings management** with automatic vectorization

Perfect for storing and searching bioinformatics literature, research notes, and experimental data.

## Step-by-Step Integration

### 1. Install Node.js via Pixi

Node.js is already included in the pixi environment. Just ensure it's installed:

```bash
# Install all dependencies including Node.js
pixi install

# Verify Node.js is available
pixi run node --version
pixi run npm --version
```

**✅ Advantage:** No system-wide installation needed - everything stays in the project environment!

### 2. Test Context7 Standalone

First, test Context7 independently to ensure it works:

```bash
# Test Context7 MCP server via pixi
pixi run context7-test

# You should see "Context7 Documentation MCP Server running on stdio"
# This confirms Context7 is working correctly
```

### 3. Add Context7 to Claude Desktop

Edit your Claude Desktop configuration file:

**Location:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

**Configuration:**
```json
{
  "mcpServers": {
    "Context7": {
      "command": "pixi",
      "args": ["run", "npx", "-y", "@upstash/context7-mcp"],
      "cwd": "/path/to/nelli-ai-scientist"
    }
  }
}
```

### 4. Restart Claude Desktop

Close and reopen Claude Desktop application to load the new MCP server.

## Testing Context7 Integration

### Test 1: Basic Connection Test

In Claude Desktop, ask:
```
Do you have access to Context7 tools? Please list what Context7 capabilities are available.
```

**Expected Response:** Claude should list Context7 tools like `store_vectors`, `search_vectors`, `list_vectors`, etc.

### Test 2: Store Bioinformatics Knowledge

```
Please store this bioinformatics information in the vector database:

"CRISPR-Cas9 is a revolutionary gene-editing technology that allows precise modification of DNA sequences. It consists of a guide RNA that targets specific DNA sequences and a Cas9 nuclease that cuts the DNA at the target site."

Use the identifier "crispr_overview" for this entry.
```

**Expected Response:** Context7 should confirm the data was stored with a unique ID.

### Test 3: Semantic Search Test

```
Search the vector database for information related to "gene editing tools" and show me what you find.
```

**Expected Response:** Should return the CRISPR information we just stored, demonstrating semantic search capability.

### Test 4: Store Multiple Research Papers

```
Store these research abstracts in the vector database:

1. "Machine learning approaches for predicting protein folding have shown remarkable success. AlphaFold2 represents a breakthrough in computational biology, achieving accuracy comparable to experimental methods."

2. "Single-cell RNA sequencing (scRNA-seq) enables analysis of gene expression at individual cell resolution. This technology has revolutionized our understanding of cellular heterogeneity and development."

3. "Phylogenetic analysis using maximum likelihood methods provides robust evolutionary relationships. These approaches are essential for understanding species evolution and biodiversity."
```

### Test 5: Complex Query Test

```
Search for content related to "computational methods in biology" and rank the results by relevance.
```

**Expected Response:** Should return relevant entries from our stored content, ranked by semantic similarity.

## Combining with Our Bioinformatics Agent

### Complete Configuration with Both Servers

```json
{
  "mcpServers": {
    "biopython-server": {
      "command": "pixi",
      "args": ["run", "python", "-m", "mcps.template.src.server"],
      "cwd": "/path/to/nelli-ai-scientist",
      "env": {
        "PYTHONPATH": "/path/to/nelli-ai-scientist"
      }
    },
    "Context7": {
      "command": "pixi",
      "args": ["run", "npx", "-y", "@upstash/context7-mcp"],
      "cwd": "/path/to/nelli-ai-scientist"
    }
  }
}
```

### Integrated Workflow Test

```
Please help me with this bioinformatics workflow:

1. First, analyze this DNA sequence using the biopython tools: "ATGCGTACGTAGCTAGCTAGCTACGTACGTACG"

2. Then store the analysis results in the Context7 vector database with the identifier "dna_analysis_001"

3. Finally, search the vector database for any related sequence analysis information we might have stored previously.
```

**Expected Response:** Claude should:
1. Use BioPython tools to analyze the sequence (GC content, length, etc.)
2. Store the results in Context7 vector database
3. Search for related analysis information

## Programmatic Testing

### Python Test Script

Create a test script to verify Context7 integration:

```python
# test_context7_integration.py
import asyncio
import json
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

async def test_context7():
    """Test Context7 MCP server connection and functionality"""
    
    try:
        # Connect to Context7 MCP server
        read, write = await stdio_client([
            "npx", "-y", "@upstash/context7-mcp"
        ])
        session = ClientSession(read, write)
        await session.initialize()
        
        print("✅ Successfully connected to Context7 MCP server")
        
        # List available tools
        tools = await session.list_tools()
        print(f"📋 Available tools: {[tool.name for tool in tools]}")
        
        # Test vector storage
        store_result = await session.call_tool("store_vectors", {
            "id": "test_bio_data",
            "text": "Bioinformatics combines computer science and biology to analyze biological data",
            "metadata": {"category": "definition", "field": "bioinformatics"}
        })
        print(f"💾 Stored vector: {store_result}")
        
        # Test vector search
        search_result = await session.call_tool("search_vectors", {
            "query": "computational biology analysis",
            "limit": 3
        })
        print(f"🔍 Search results: {search_result}")
        
        # Test listing vectors
        list_result = await session.call_tool("list_vectors", {})
        print(f"📝 All vectors: {list_result}")
        
        print("✅ All Context7 tests passed!")
        
    except Exception as e:
        print(f"❌ Context7 test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_context7())
    exit(0 if success else 1)
```

### Run the Test

```bash
# Install required dependencies
pip install mcp

# Run the test
python test_context7_integration.py
```

## Troubleshooting

### Common Issues

#### 1. "npx command not found"
```bash
# Install Node.js first
brew install node  # macOS
sudo apt install nodejs npm  # Ubuntu/Debian
```

#### 2. "Context7 server failed to start"
```bash
# Test Context7 manually
npx -y @upstash/context7-mcp

# Check for error messages in the output
# Ensure internet connection for package download
```

#### 3. "Tools not appearing in Claude"
- Restart Claude Desktop completely
- Check configuration file syntax (use JSON validator)
- Verify file path is correct for your OS

#### 4. "Permission denied errors"
```bash
# On macOS, check app permissions
# On Linux, ensure proper file permissions
chmod 644 ~/config/Claude/claude_desktop_config.json
```

### Debug Configuration

For debugging, add logging to your configuration:

```json
{
  "mcpServers": {
    "Context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"],
      "env": {
        "DEBUG": "mcp*",
        "MCP_LOG_LEVEL": "debug"
      }
    }
  }
}
```

## Advanced Usage Examples

### Store Research Papers with Metadata

```
Please store this research paper information in Context7:

Title: "Deep Learning for Protein Structure Prediction"
Abstract: "Recent advances in deep learning have transformed protein structure prediction. Neural networks can now predict 3D protein structures with remarkable accuracy, revolutionizing structural biology research."
Authors: ["Smith, J.", "Doe, A."]
Year: 2024
Journal: "Nature Biotechnology"
Keywords: ["protein folding", "deep learning", "structural biology"]

Use the ID: "protein_dl_2024_001"
```

### Search with Filters

```
Search the Context7 database for:
- Content related to "machine learning in biology"
- From papers published after 2020
- Limit to 5 results
- Sort by relevance
```

### Knowledge Graph Queries

```
Find connections between stored research papers about:
1. Protein structure prediction
2. Gene expression analysis
3. Evolutionary biology

Show me how these research areas are connected based on our stored knowledge.
```

## Integration Benefits

### For Bioinformatics Research

1. **Literature Management**: Store and search research papers semantically
2. **Experimental Knowledge**: Keep track of experimental procedures and results
3. **Cross-Reference Discovery**: Find connections between different research areas
4. **Hypothesis Generation**: Use stored knowledge to generate new research questions

### For AI Agents

1. **Memory Enhancement**: Persistent storage for agent learning
2. **Context Retrieval**: Quickly find relevant information for current tasks
3. **Knowledge Sharing**: Multiple agents can access shared knowledge base
4. **Incremental Learning**: Build knowledge over time through interactions

## Next Steps

1. **Test the integration** following this guide
2. **Store your research data** in Context7
3. **Experiment with semantic search** for your specific domain
4. **Combine with BioPython tools** for comprehensive analysis workflows
5. **Develop custom workflows** that leverage both vector search and computational biology tools

This integration provides a powerful foundation for AI-assisted bioinformatics research with persistent memory and semantic search capabilities.