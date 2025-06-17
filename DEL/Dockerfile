# Dockerfile for Nucleic Acid Analysis MCP Server
FROM mambaorg/micromamba:1.5.8

# Set working directory
WORKDIR /app

# Copy environment file
COPY pixi.toml .

# Install pixi
USER root
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://pixi.sh/install.sh | bash && \
    echo 'export PATH="/root/.pixi/bin:$PATH"' >> ~/.bashrc
ENV PATH="/root/.pixi/bin:$PATH"

# Copy source code
COPY src/ ./src/
COPY README.md .

# Install dependencies
RUN pixi install

# Expose MCP over stdio
ENTRYPOINT ["pixi", "run", "python", "-m", "src.server"]