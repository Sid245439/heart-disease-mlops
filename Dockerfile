# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file and create conda env (alternative: use pip)
COPY environment.yml /app/

# Install dependencies via pip (faster than conda in Docker)
RUN pip install --no-cache-dir \
    numpy==1.24.* \
    pandas==2.0.* \
    scikit-learn==1.3.* \
    matplotlib==3.7.* \
    seaborn==0.12.* \
    mlflow==2.10.* \
    flask==3.0.* \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    python-multipart==0.0.6 \
    requests==2.31.* \
    joblib==1.3.2 \
    onnx==1.15.0 \
    skl2onnx==1.16.0 \
    prometheus-client==0.18.* \
    gunicorn==21.*

# Copy application code
COPY . /app/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
