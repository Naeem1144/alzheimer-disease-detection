# Multi-stage Dockerfile for optimized Alzheimer's prediction app
# This creates a minimal production image with optimized dependencies

# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-optimized.txt .
RUN pip install --no-cache-dir --user -r requirements-optimized.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Make sure scripts are executable
RUN chmod +x deploy_optimized.sh model_optimizer.py

# Environment variables for production
ENV PYTHONPATH=/root/.local/lib/python3.11/site-packages:$PYTHONPATH
ENV PATH=/root/.local/bin:$PATH
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the optimized application
CMD ["streamlit", "run", "Alzheimer-ui-optimized.py", "--server.address=0.0.0.0", "--server.port=8501"]