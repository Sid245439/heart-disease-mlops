## Experiment Tracking (MLflow)

This project uses **MLflow Tracking** to record model-training experiments (parameters, metrics, and artifacts) so you can compare runs and reproduce results.

### What gets tracked

During training, the code in `src/training.py`:

- Sets/uses an MLflow experiment named `heart-disease-mlops`.
- Creates runs for:
  - Logistic Regression (`run_name="logistic-regression"`)
  - Random Forest (`run_name="random-forest"`)
- Logs:
  - **Params**: hyperparameters (e.g., `C` for LR, `n_estimators`, `max_depth` for RF)
  - **Metrics**: accuracy/precision/recall/F1 + AUC values
  - **Artifacts**: a feature-importance plot for Random Forest
  - **Models**: each trained estimator via `mlflow.sklearn.log_model(...)`

In addition to MLflow tracking, the pipeline also writes local model artifacts:

- `models/best_model.pkl`
- `models/preprocessor.pkl`

### Prerequisites

You need Python dependencies installed.

Recommended (this repo’s standard):

- Install `uv`
- Use `nox` sessions (which use `uv` under the hood via `nox-uv`)

If you haven’t installed `nox` locally yet:

```bash
uv pip install --system nox nox-uv
```

### Option A (simplest): track locally with the default file store

By default, MLflow stores runs in a local folder named `./mlruns` (created automatically).

1. Run training

```bash
nox -s train
```

This downloads the dataset (to `data/raw/heart_disease_raw.csv`) and runs the training pipeline.

2. Start the MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5000
```

3. Open the UI in your browser

- `http://127.0.0.1:5000`

### Option B: run an MLflow Tracking Server (recommended for repeatable local work)

If you want a more “server-like” setup (separate backend store + artifact directory), run:

```bash
mlflow server \
	--backend-store-uri sqlite:///mlflow.db \
	--default-artifact-root ./mlartifacts \
	--host 127.0.0.1 \
	--port 5000
```

Then, in a separate terminal, point clients at it:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
nox -s train
```

Windows PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
nox -s train
```

### Where to find results

- MLflow UI:
  - Experiments → `heart-disease-mlops`
  - Runs → compare LR vs RF, metrics, params
  - Artifacts tab → model artifacts + feature importance plot
- Local files:
  - `models/best_model.pkl` and `models/preprocessor.pkl` are used by the inference service.

### Common troubleshooting

- **UI shows no runs**

  - Make sure the UI points at the same backend store where runs are written.
  - If you used the default file store, run `mlflow ui --backend-store-uri file:./mlruns` from the repo root.

- **Port already in use (5000)**

  - Pick another port, e.g. `--port 5001`.

- **Runs are created but artifacts are missing**

  - Check file permissions in the artifact root directory.
  - For a server setup, confirm `--default-artifact-root` is writable.

- **CI vs local behavior**
  - In CI, a long-running MLflow server usually isn’t started; tracking typically writes to the local file store.
