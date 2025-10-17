# Dockerfile for MetaEvaluator Annotation Platform
# This image provides a containerized environment for running the annotation platform
# with all dependencies pre-installed.

FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install meta-evaluator from GitHub
# This ensures the latest version is always installed when building the image
RUN pip install --no-cache-dir git+https://github.com/govtech-responsibleai/meta-evaluator

# Set working directory
# User's project files will be mounted here via volume mapping
WORKDIR /app/workspace

# Expose Streamlit default port
EXPOSE 8501

# Default command (typically overridden by docker-compose or docker run)
CMD ["python"]
