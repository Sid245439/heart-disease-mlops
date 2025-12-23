#!/bin/bash
# Complete local MLOps pipeline setup and execution
# Run: bash setup_and_run.sh

set -e

echo "============================================================"
echo "Heart Disease MLOps - Complete Setup & Execution Pipeline"
echo "============================================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Create conda environment
echo -e "\n${BLUE}Step 1: Creating Conda Environment...${NC}"
if conda env list | grep -q heart-disease-mlops; then
    echo -e "${YELLOW}Environment already exists. Updating...${NC}"
    conda env update -f environment.yml --prune
else
    echo "Creating new environment..."
    conda env create -f environment.yml
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate heart-disease-mlops
echo -e "${GREEN}✓ Environment activated${NC}"

# Step 2: Download data
echo -e "\n${BLUE}Step 2: Downloading Dataset...${NC}"
python download_data.py
echo -e "${GREEN}✓ Data downloaded${NC}"

# Step 3: Run unit tests
echo -e "\n${BLUE}Step 3: Running Unit Tests...${NC}"
pytest tests/ -v --tb=short
echo -e "${GREEN}✓ Tests passed${NC}"

# Step 4: Train models
echo -e "\n${BLUE}Step 4: Training Models...${NC}"
python -c "
import logging
logging.basicConfig(level=logging.INFO)
from src.training import train_pipeline
train_pipeline('data/raw/heart_disease_raw.csv')
"
echo -e "${GREEN}✓ Model training complete${NC}"

# Step 5: Check MLflow UI
echo -e "\n${BLUE}Step 5: Starting MLflow UI...${NC}"
echo "MLflow will start on http://localhost:5000"
echo "Press Ctrl+C to stop MLflow and continue..."
mlflow ui --host 0.0.0.0 --port 5000 &
MLFLOW_PID=$!
sleep 3
echo -e "${GREEN}✓ MLflow started (PID: $MLFLOW_PID)${NC}"

# Step 6: Build Docker image
echo -e "\n${BLUE}Step 6: Building Docker Image...${NC}"
docker build -t heart-disease-mlops:latest .
echo -e "${GREEN}✓ Docker image built${NC}"

# Step 7: Run Docker container
echo -e "\n${BLUE}Step 7: Starting Docker Container...${NC}"
docker run -d \
  --name heart-disease-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  heart-disease-mlops:latest
echo -e "${GREEN}✓ Docker container running${NC}"

# Wait for API to start
echo "Waiting for API to be ready..."
sleep 5

# Step 8: Test API
echo -e "\n${BLUE}Step 8: Testing API Endpoints...${NC}"
echo "Testing /health endpoint..."
curl -s http://localhost:8000/health | python -m json.tool

echo -e "\nTesting /predict endpoint..."
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }' | python -m json.tool

echo -e "\n${GREEN}✓ API tests completed${NC}"

# Step 9: Setup Kubernetes (optional)
echo -e "\n${BLUE}Step 9: Kubernetes Deployment (Optional)...${NC}"
echo "To deploy to Kubernetes, run:"
echo "  kubectl apply -f k8s/deployment.yaml"
echo "Check status with:"
echo "  kubectl get pods -l app=heart-disease"
echo "  kubectl port-forward svc/heart-disease-service 8000:8000"

# Final summary
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}✓ Complete MLOps Pipeline Execution Successful!${NC}"
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${YELLOW}Summary:${NC}"
echo "  Data:     $(ls -lh data/raw/*.csv | awk '{print $5, $9}')"
echo "  Model:    $(ls -lh models/best_model.pkl | awk '{print $5, $9}')"
echo "  API:      http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Metrics:  http://localhost:8000/metrics"
echo "  MLflow:   http://localhost:5000"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "  1. View API documentation: http://localhost:8000/docs"
echo "  2. Monitor experiments: http://localhost:5000"
echo "  3. Deploy to Kubernetes: kubectl apply -f k8s/deployment.yaml"
echo "  4. Clean up: ./cleanup.sh"
echo ""
