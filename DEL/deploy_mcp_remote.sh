#!/bin/bash
"""
Deploy MCP Servers to Remote Machines
Supports various deployment scenarios
"""

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_NAME="${1:-bioseq}"
DEPLOYMENT_TYPE="${2:-ssh}"

print_usage() {
    echo "🚀 MCP Remote Deployment Script"
    echo "Usage: $0 <mcp_name> <deployment_type>"
    echo ""
    echo "MCP Names:"
    echo "  bioseq      - Nucleic acid analysis MCP"
    echo "  filesystem  - File operations MCP"
    echo ""
    echo "Deployment Types:"
    echo "  ssh         - Deploy via SSH to remote server"
    echo "  docker      - Build and deploy Docker container"
    echo "  kubernetes  - Deploy to Kubernetes cluster"
    echo ""
    echo "Examples:"
    echo "  $0 bioseq ssh"
    echo "  $0 bioseq docker"
    echo "  $0 filesystem ssh"
}

deploy_ssh() {
    local mcp_name="$1"
    local remote_host="${REMOTE_HOST:-bioserver}"
    local remote_path="${REMOTE_PATH:-/opt/nelli-mcps}"
    
    echo "📡 Deploying $mcp_name MCP via SSH to $remote_host:$remote_path"
    
    # Create deployment package
    echo "📦 Creating deployment package..."
    local temp_dir=$(mktemp -d)
    cp -r "mcps/$mcp_name" "$temp_dir/"
    
    # Create deployment script
    cat > "$temp_dir/deploy.sh" << 'EOF'
#!/bin/bash
# Remote deployment script
set -e

MCP_DIR="$1"
echo "🏠 Setting up MCP in $MCP_DIR"

cd "$MCP_DIR"

# Install pixi if not available
if ! command -v pixi &> /dev/null; then
    echo "📥 Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi

# Install dependencies
echo "📦 Installing dependencies..."
pixi install

# Create systemd service
echo "🔧 Creating systemd service..."
sudo tee "/etc/systemd/system/nelli-mcp-${MCP_DIR##*/}.service" > /dev/null << EOL
[Unit]
Description=NeLLi MCP Server - ${MCP_DIR##*/}
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$MCP_DIR
ExecStart=$HOME/.pixi/bin/pixi run run
Restart=always
RestartSec=10
StandardInput=null
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOL

echo "✅ MCP deployed successfully!"
echo "To start: sudo systemctl start nelli-mcp-${MCP_DIR##*/}"
echo "To enable: sudo systemctl enable nelli-mcp-${MCP_DIR##*/}"
EOF
    chmod +x "$temp_dir/deploy.sh"
    
    # Transfer to remote
    echo "📤 Transferring to remote server..."
    ssh "$remote_host" "mkdir -p $remote_path"
    scp -r "$temp_dir/$mcp_name" "$remote_host:$remote_path/"
    scp "$temp_dir/deploy.sh" "$remote_host:$remote_path/"
    
    # Execute deployment
    echo "🔧 Executing remote deployment..."
    ssh "$remote_host" "cd $remote_path && ./deploy.sh $mcp_name"
    
    # Cleanup
    rm -rf "$temp_dir"
    
    echo "✅ SSH deployment complete!"
    echo "🔌 Connect via: ssh $remote_host 'cd $remote_path/$mcp_name && pixi run run'"
}

deploy_docker() {
    local mcp_name="$1"
    
    echo "🐳 Building Docker container for $mcp_name MCP"
    
    cd "mcps/$mcp_name"
    
    # Build container
    echo "🔨 Building Docker image..."
    docker build -t "nelli/${mcp_name}-mcp:latest" .
    
    # Test container
    echo "🧪 Testing container..."
    timeout 5s docker run --rm -i "nelli/${mcp_name}-mcp:latest" || true
    
    echo "✅ Docker deployment complete!"
    echo "🔌 Run with: docker run -i nelli/${mcp_name}-mcp:latest"
    echo "📤 Push with: docker push nelli/${mcp_name}-mcp:latest"
    
    cd - > /dev/null
}

deploy_kubernetes() {
    local mcp_name="$1"
    
    echo "☸️  Creating Kubernetes deployment for $mcp_name MCP"
    
    # Create Kubernetes manifests
    local manifest_dir="k8s-manifests"
    mkdir -p "$manifest_dir"
    
    cat > "$manifest_dir/${mcp_name}-mcp-deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${mcp_name}-mcp
  labels:
    app: ${mcp_name}-mcp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${mcp_name}-mcp
  template:
    metadata:
      labels:
        app: ${mcp_name}-mcp
    spec:
      containers:
      - name: ${mcp_name}-mcp
        image: nelli/${mcp_name}-mcp:latest
        stdin: true
        tty: true
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "512Mi" 
            cpu: "250m"
---
apiVersion: v1
kind: Service
metadata:
  name: ${mcp_name}-mcp-service
spec:
  selector:
    app: ${mcp_name}-mcp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: ClusterIP
EOF
    
    echo "📝 Kubernetes manifest created: $manifest_dir/${mcp_name}-mcp-deployment.yaml"
    echo "🚀 Deploy with: kubectl apply -f $manifest_dir/${mcp_name}-mcp-deployment.yaml"
    echo "🔌 Access with: kubectl exec -it deployment/${mcp_name}-mcp -- /bin/bash"
}

main() {
    if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        print_usage
        exit 0
    fi
    
    if [[ -z "$MCP_NAME" ]] || [[ -z "$DEPLOYMENT_TYPE" ]]; then
        print_usage
        exit 1
    fi
    
    if [[ ! -d "mcps/$MCP_NAME" ]]; then
        echo "❌ MCP directory not found: mcps/$MCP_NAME"
        exit 1
    fi
    
    echo "🚀 Deploying $MCP_NAME MCP using $DEPLOYMENT_TYPE"
    
    case "$DEPLOYMENT_TYPE" in
        ssh)
            deploy_ssh "$MCP_NAME"
            ;;
        docker)
            deploy_docker "$MCP_NAME"
            ;;
        kubernetes|k8s)
            deploy_kubernetes "$MCP_NAME"
            ;;
        *)
            echo "❌ Unknown deployment type: $DEPLOYMENT_TYPE"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"