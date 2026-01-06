# Production Deployment (Kubernetes)

A Kubernetes manifest is provided at `k8s/deployment.yaml`.

## What is deployed

- Deployment: `heart-disease-predictor` (2 replicas)
- Service: `heart-disease-service` (type: `LoadBalancer`)
- HPA: `heart-disease-hpa` (CPU/memory based)

The Pod template includes Prometheus scrape annotations for `/metrics`.

## Important: model artifacts in Kubernetes

The API loads these files at startup:

- `models/best_model.pkl`
- `models/preprocessor.pkl`

The simplest approach for this assignment is:

1. Train locally: `nox -s train`
2. Build the image after training (so the `models/` directory is baked into the image).

The current manifest does **not** mount an external volume at `/app/models` so that the baked-in model artifacts remain available.

## Deploy (Minikube / Docker Desktop Kubernetes)

### 1. Build the image

```bash
nox -s train

docker build -t heart-disease-mlops:latest .
```

### 2. Ensure the cluster can see the image

- **Minikube** (example):

```bash
minikube image load heart-disease-mlops:latest
```

Or build directly inside minikube’s Docker daemon.

### 3. Apply the manifest

```bash
kubectl apply -f k8s/deployment.yaml
```

### 4. Verify

```bash
kubectl get pods -l app=heart-disease
kubectl get svc heart-disease-service
```

### 5. Access the service

If you don’t have a cloud load balancer (typical for local clusters), use port-forward:

```bash
kubectl port-forward svc/heart-disease-service 8000:8000
```

Then open:

- http://localhost:8000/docs

## What to screenshot for the report

- `kubectl get pods` showing READY pods
- `kubectl get svc` showing service exposure
- A successful `/health` and `/predict` call
- (Optional) HPA status
