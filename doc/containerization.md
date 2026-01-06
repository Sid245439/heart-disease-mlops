# Containerization (Docker)

The service is containerized using the root `Dockerfile`.

## Build

For a fully working image, train first (so `models/` exists), then build.

```bash
nox -s train

docker build -t heart-disease-mlops:latest .
```

## Run

```bash
docker run -p 8000:8000 heart-disease-mlops:latest
```

Open:

- http://localhost:8000/docs
- http://localhost:8000/health
- http://localhost:8000/metrics

## Optional: mount external model artifacts

If you want to rebuild models without rebuilding the image, you can mount `models/` into the container.

PowerShell:

```powershell
docker run -p 8000:8000 -v ${PWD}/models:/app/models:ro heart-disease-mlops:latest
```

## Notes

- Dependencies are installed from `requirements.txt` (generated via `nox -s requirements`).
- The container runs Uvicorn on port 8000.
