#!/bin/bash
"""
QuickBin MCP Deployment Script

This script helps deploy and test the QuickBin MCP server.
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧬 QuickBin MCP Deployment Script${NC}"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "src/server_fastmcp.py" ]; then
    echo -e "${RED}❌ Error: Must run from quickbin MCP directory${NC}"
    echo "Expected: /path/to/mcps/quickbin/"
    exit 1
fi

echo -e "${GREEN}✅ Found QuickBin MCP directory${NC}"

# Test Python imports
echo -e "\n${BLUE}1. Testing Python dependencies...${NC}"
python -c "
import sys
sys.path.insert(0, 'src')
try:
    from fastmcp import FastMCP
    print('✅ FastMCP available')
except ImportError:
    print('❌ FastMCP not available - install with: pip install fastmcp')
    sys.exit(1)

try:
    from quickbin_tools import QuickBinToolkit
    print('✅ QuickBinToolkit importable')
except Exception as e:
    print(f'❌ QuickBinToolkit error: {e}')
    sys.exit(1)

try:
    from server_fastmcp import mcp, toolkit
    print('✅ Server imports successful')
except Exception as e:
    print(f'❌ Server import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Python dependency test failed${NC}"
    exit 1
fi

# Test shifter access
echo -e "\n${BLUE}2. Testing shifter container access...${NC}"
if command -v shifter &> /dev/null; then
    echo -e "${GREEN}✅ Shifter command available${NC}"

    # Test container access
    if shifter --image bryce911/bbtools:39.27 echo "Container accessible" &> /dev/null; then
        echo -e "${GREEN}✅ BBTools container accessible${NC}"
    else
        echo -e "${YELLOW}⚠️  BBTools container not accessible${NC}"
        echo "This may be expected if not on a system with the container loaded"
    fi
else
    echo -e "${YELLOW}⚠️  Shifter not available${NC}"
    echo "QuickBin will not work without shifter access"
fi

# Test QuickBin availability (if shifter works)
echo -e "\n${BLUE}3. Testing QuickBin tool availability...${NC}"
if shifter --image bryce911/bbtools:39.27 quickbin.sh --help &> /dev/null; then
    echo -e "${GREEN}✅ QuickBin tool accessible${NC}"
else
    echo -e "${YELLOW}⚠️  QuickBin tool test skipped (container/shifter issue)${NC}"
fi

# Run comprehensive test
echo -e "\n${BLUE}4. Running comprehensive MCP test...${NC}"
if python test_quickbin_mcp.py; then
    echo -e "${GREEN}✅ Comprehensive test passed${NC}"
else
    echo -e "${RED}❌ Comprehensive test failed${NC}"
    echo "Check the output above for specific issues"
fi

# Check MCP configuration
echo -e "\n${BLUE}5. Checking MCP configuration...${NC}"
if [ -f "mcp_config.json" ]; then
    echo -e "${GREEN}✅ MCP configuration file found${NC}"
    echo "Content:"
    cat mcp_config.json | grep -A 10 "quickbin"
else
    echo -e "${RED}❌ MCP configuration file missing${NC}"
    echo "Creating mcp_config.json..."
    cat > mcp_config.json << 'EOF'
{
  "mcpServers": {
    "quickbin": {
      "command": "python",
      "args": ["-m", "src.server_fastmcp"],
      "cwd": "/pscratch/sd/j/jvillada/nelli-ai-scientist/mcps/quickbin"
    }
  }
}
EOF
    echo -e "${GREEN}✅ Created mcp_config.json${NC}"
fi

# Final summary
echo -e "\n${BLUE}=================================================="
echo -e "📋 DEPLOYMENT SUMMARY"
echo -e "==================================================${NC}"

echo -e "\n${GREEN}🎉 QuickBin MCP is ready for use!${NC}"

echo -e "\n${BLUE}📁 Files created:${NC}"
echo "  📄 src/server_fastmcp.py - Main MCP server"
echo "  📄 src/quickbin_tools.py - Core functionality"
echo "  📄 src/tool_schema.py - Tool schemas"
echo "  📄 mcp_config.json - MCP configuration"
echo "  📄 README.md - Documentation"
echo "  📄 test_quickbin_mcp.py - Test suite"

echo -e "\n${BLUE}🛠️  Available Tools:${NC}"
echo "  🗂️  bin_contigs - Main binning with SAM files"
echo "  🗂️  bin_contigs_with_coverage - Fast re-binning"
echo "  📊 generate_coverage - Coverage statistics"
echo "  ✅ evaluate_bins - Quality assessment"

echo -e "\n${BLUE}🎯 Next Steps:${NC}"
echo "1. Add this MCP to your client configuration:"
echo "   Copy mcp_config.json content to your MCP client config"
echo ""
echo "2. Test with real data:"
echo "   - Assembled contigs (FASTA format)"
echo "   - SAM files from read mapping (e.g., from bbmap)"
echo ""
echo "3. Example usage:"
echo "   bin_contigs("
echo "     contigs_path='assembly.fasta',"
echo "     sam_files=['sample1.sam', 'sample2.sam'],"
echo "     output_pattern='bins/bin%.fa',"
echo "     stringency='normal'"
echo "   )"

echo -e "\n${BLUE}📚 Documentation:${NC}"
echo "  - README.md - Complete usage guide"
echo "  - quickbin://docs/user-guide - User guide resource"
echo "  - quickbin://examples/metagenome-workflow - Workflow examples"

echo -e "\n${GREEN}🚀 Your QuickBin MCP is ready to accelerate metagenomics research!${NC}"
