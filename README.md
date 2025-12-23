# Copy to Cloud/Documents/heart-disease-mlops
Follow Commands Setu by step

conda env create -f environment.yml

conda activate heart-disease-mlops

python download_data.py

pytest tests/ -v

python -c "from src.training import train_pipeline; train_pipeline('data/raw/heart_disease_raw.csv')"


#   API
### On New terminal
conda activate heart-disease-mlops

uvicorn app:app --reload
### Visit: http://localhost:8000/docs


# MLFLOW
### On New Terminal
conda activate heart-disease-mlops

mlflow ui --host 0.0.0.0 --port 5000
### Visit: http://localhost:5000


# Docker- Add VS extension for docker before this
### Build and run - On new terminal
conda activate heart-disease-mlops

docker build -t heart-disease-mlops:latest .

docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro heart-disease-mlops:latest
### Visit: http://localhost:8000/docs


### Health check
curl http://localhost:8000/health

### Test prediction
curl -X POST http://localhost:8000/predict \
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
  }'




