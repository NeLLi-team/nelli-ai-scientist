#!/bin/bash
# Setup script for hackathon participants

set -e

echo "🚀 NeLLi AI Scientist Workspace Setup"
echo "===================================="

# Check if git is configured
if [ -z "$(git config --global user.name)" ]; then
    read -p "Enter your name for git: " git_name
    git config --global user.name "$git_name"
fi

if [ -z "$(git config --global user.email)" ]; then
    read -p "Enter your email for git: " git_email
    git config --global user.email "$git_email"
fi

# Get participant info
read -p "Enter your name (lowercase, no spaces): " participant_name
read -p "Are you building an (a)gent or (m)cp server? [a/m]: " component_type

if [ "$component_type" = "a" ]; then
    component_dir="agents/$participant_name"
    branch_name="agent/$participant_name"
    template_dir="agents/template"
else
    read -p "Enter MCP server name (e.g., blast, kraken2): " mcp_name
    component_dir="mcps/$mcp_name"
    branch_name="mcp/$mcp_name"
    template_dir="mcps/template"
fi

# Create branch
echo "📌 Creating branch: $branch_name"
git checkout -b "$branch_name"

# Copy template
echo "📁 Setting up your workspace at: $component_dir"
cp -r "$template_dir" "$component_dir"

# Update README with participant info
sed -i "s/template/$participant_name/g" "$component_dir/README.md"

# Create initial commit
git add "$component_dir"
git commit -m "feat: Initialize $participant_name workspace from template"

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. cd $component_dir"
echo "2. pixi install"
echo "3. Start developing!"
echo ""
echo "To push your branch:"
echo "git push origin $branch_name"