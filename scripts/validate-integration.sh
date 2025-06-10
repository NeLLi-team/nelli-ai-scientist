#!/bin/bash
# Validate agent/MCP integration readiness

set -e

component_path=$1

if [ -z "$component_path" ]; then
    echo "Usage: $0 <component-path>"
    exit 1
fi

echo "🔍 Validating component: $component_path"

# Check required files
required_files=(
    "README.md"
    "src/"
    "tests/"
)

for file in "${required_files[@]}"; do
    if [ ! -e "$component_path/$file" ]; then
        echo "❌ Missing required file/directory: $file"
        exit 1
    fi
done

# Run tests
echo "🧪 Running tests..."
cd "$component_path"
pixi run pytest tests/ -v

# Check if it follows standards
echo "🎨 Checking code standards..."
pixi run black --check src/
pixi run mypy src/ --ignore-missing-imports

echo "✅ Component validation passed!"