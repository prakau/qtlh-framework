# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the framework files
COPY . .

# Install the package
RUN pip install -e ".[dev,docs]"

# Run tests to verify installation
RUN pytest -v tests/ -m "not gpu"

# Create non-root user
RUN useradd -m -u 1000 qtlh
USER qtlh

# Set environment variables for the framework
ENV QTLH_CACHE_DIR=/app/.cache \
    QTLH_DATA_DIR=/app/data \
    QTLH_RESULTS_DIR=/app/results

# Create necessary directories with correct permissions
RUN mkdir -p /app/.cache /app/data /app/results

# Set default command
CMD ["python", "-m", "qtlh"]

# Add labels
LABEL maintainer="QTL-H Development Team <contact@qtlh-framework.org>" \
      version="0.1.0" \
      description="QTL-H Framework - Quantum-enhanced Topological Linguistic Hyperdimensional Framework for Genomic Analysis"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Volume configuration for persistent storage
VOLUME ["/app/data", "/app/results"]

# Expose ports for potential API/UI
EXPOSE 8000

# Add documentation about the image
RUN echo "QTL-H Framework Docker Image" > /README.docker && \
    echo "==========================" >> /README.docker && \
    echo "" >> /README.docker && \
    echo "This image contains the QTL-H framework and all its dependencies." >> /README.docker && \
    echo "" >> /README.docker && \
    echo "Usage:" >> /README.docker && \
    echo "  docker run -it --rm -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results qtlh-framework" >> /README.docker

# Optional: Set up development environment
RUN if [ "$BUILD_ENV" = "development" ]; then \
    pip install -e ".[dev,docs]" && \
    pip install jupyter jupyterlab && \
    jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py; \
    fi

# Optional: GPU support
RUN if [ "$USE_GPU" = "true" ]; then \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116; \
    fi

# Optional: Development ports
EXPOSE 8888 6006

# Development entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]
