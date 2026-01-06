# CI/CD

This project uses `nox` + `uv` locally and in GitHub Actions.

## Nox sessions

Defined in `noxfile.py`:

- `nox -s lint` – `ruff check` (generates JUnit XML + HTML report)
- `nox -s format` – `ruff format --check --diff` (diff patch under `doc/reports/format/`)
- `nox -s typing` – `mypy` (generates JUnit XML + HTML report)
- `nox -s test` – `pytest` + coverage (JUnit XML + HTML, coverage XML + HTML)
- `nox -s train` – downloads data + trains models
- `nox -s requirements` – regenerates `requirements.txt` via `uv export`
- `nox -s docs` – builds MkDocs site

## GitHub Actions workflows

Workflows are located under `.github/workflows/`.

### CI (`ci.yml`)

Runs in parallel jobs:

- `lint`
- `format`
- `typing`
- `test`

Each job uploads artifacts from `doc/reports/` so you can view reports per workflow run.

### CI/CD (`ci-cd.yml`)

Orchestrates:

1. Reuses CI via `uses: ./.github/workflows/ci.yml`
2. Builds Docker image (after regenerating `requirements.txt`)
3. Triggers training via `nox -s train` and uploads the `models/` folder as an artifact
4. Runs a security scan (Trivy)

### Docs (`docs.yml`)

- Builds MkDocs site via `nox -s docs`
- Deploys to GitHub Pages (on push to main/master)

## Where reports are stored

- Ruff: `doc/reports/ruff/`
- Format diff: `doc/reports/format/`
- Mypy: `doc/reports/typing/`
- Pytest + JUnit HTML: `doc/reports/pytest/`
- Coverage HTML: `doc/reports/coverage/htmlcov/`
