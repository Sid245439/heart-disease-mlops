from __future__ import annotations

from pathlib import Path
from nox import Session
import nox
from nox_uv import session

MODULE_NAME = "src"


PROJECT_DIRECTORY = Path(__file__).parent.resolve()
DOCS_DIRECTORY = PROJECT_DIRECTORY / "doc"
REPORTS_DIRECTORY = DOCS_DIRECTORY / "reports"
RUFF_DIRECTORY = REPORTS_DIRECTORY / "ruff"
PYTEST_DIRECTORY = REPORTS_DIRECTORY / "pytest"
COVERAGE_DIRECTORY = REPORTS_DIRECTORY / "coverage"
FORMAT_DIRECTORY = REPORTS_DIRECTORY / "format"
TYPING_DIRECTORY = REPORTS_DIRECTORY / "typing"

DATA_DIRECTORY = PROJECT_DIRECTORY / "data" / "raw"


nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = True
# Single command to run all sessions in order
nox.options.sessions = ["lint", "typing", "format", "test"]


@session(
    venv_backend="uv",
    uv_only_groups=["lint"],
    python="3.12",
    uv_no_install_project=True,
)
def lint(session: Session) -> None:
    """Run ruff to check linting issues."""
    # First, generate the report and save it to an XML file
    # Ensure report directories exist
    RUFF_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with (RUFF_DIRECTORY / "ruff-lint-report.xml").open("w") as f:
        # The --exit-zero flag ensures that ruff exits with code 0
        # even if there are linting issues, so that we can capture
        session.run(
            "ruff",
            "check",
            ".",
            "--output-format=junit",
            "--exit-zero",
            stdout=f,
            stderr=None,
        )
    session.run(
        "junit2html",
        str(RUFF_DIRECTORY / "ruff-lint-report.xml"),
        "--report-matrix",
        str(RUFF_DIRECTORY / "ruff-lint-report.html"),
    )

    # Run the command normally to show issues in the console
    # And to set the exit code appropriately
    session.run("ruff", "check", ".")


@session(
    venv_backend="uv",
    uv_only_groups=["format"],
    python="3.12",
    uv_no_install_project=True,
)
def format(session: Session) -> None:
    """Run ruff to format the code."""
    FORMAT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with (FORMAT_DIRECTORY / "ruff-format-diff.patch").open("w") as f:
        session.run("ruff", "format", ".", "--check", "--diff", stdout=f, stderr=None)


@session(
    venv_backend="uv",
    uv_groups=["typing", "train"],
    python="3.12",
)
def typing(session: Session) -> None:
    """Run mypy to check typing issues."""
    TYPING_DIRECTORY.mkdir(parents=True, exist_ok=True)

    mypy_xml = TYPING_DIRECTORY / "mypy-report.xml"
    mypy_html = TYPING_DIRECTORY / "mypy-report.html"

    # Generate report even when there are typing errors.
    session.run("mypy", ".", "--junit-xml", str(mypy_xml), success_codes=[0, 1])
    session.run("junit2html", str(mypy_xml), "--report-matrix", str(mypy_html))

    # Re-run normally to set the proper exit code.
    session.run("mypy", ".")


@session(venv_backend="uv", uv_groups=["dev"], python="3.12", uv_no_install_project=True)
def dev(session: Session) -> None:
    """Set up development environment."""
    # No need to implement anything here for now
    # The virtual environment will be created with the specified packages
    # when this session is called.
    pass


@session(
    venv_backend="uv",
    uv_groups=["test"],
    python="3.12",
)
def test(session: Session) -> None:
    """Run the test suite."""
    # Ensure report directories exist
    PYTEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    COVERAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    pytest_xml = PYTEST_DIRECTORY / "pytest-report.xml"
    pytest_html = PYTEST_DIRECTORY / "pytest-report.html"
    coverage_xml = COVERAGE_DIRECTORY / "coverage-report.xml"

    # Run pytest with coverage and generate XML report
    session.run(
        "pytest",
        f"--junitxml={pytest_xml}",
        f"--cov={MODULE_NAME}",
        f"--cov-report=xml:{coverage_xml}",
        f"--cov-report=html:{COVERAGE_DIRECTORY / 'htmlcov'}",
        "tests/",
    )

    # Convert pytest XML report to HTML
    session.run(
        "junit2html",
        str(pytest_xml),
        "--report-matrix",
        str(pytest_html),
    )


@session(
    venv_backend="uv",
    uv_groups=["train"],
    python="3.12",
)
def train(session: Session) -> None:
    """Download data and train models."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    session.run("python", "download.py")
    session.run(
        "python",
        "-c",
        "from src.training import train_pipeline; train_pipeline('data/raw/heart_disease_raw.csv')",
    )


@session(
    venv_backend="uv",
    uv_groups=["train"],
    python="3.12",
    uv_no_install_project=True,
)
def mlflow_ui(session: Session) -> None:
    """Start the MLflow UI locally for development."""
    session.chdir(PROJECT_DIRECTORY)

    args = list(session.posargs)
    if not args:
        args = ["--host", "localhost", "--port", "5001"]

    session.run("mlflow", "ui", *args, external=True)


@session(
    venv_backend="uv",
    uv_only_groups=["docs"],
    python="3.12",
    uv_no_install_project=True,
)
def docs(session: Session) -> None:
    """Build documentation site (MkDocs)."""
    session.run("mkdocs", "build", "--clean", "--strict")


@session(
    venv_backend="uv",
    python="3.12",
    uv_no_install_project=True,
)
def requirements(session: Session) -> None:
    """Generate a fresh requirements.txt from pyproject/lock using uv."""
    session.run(
        "uv",
        "export",
        "--no-dev",
        "-o",
        "requirements.txt",
        external=True,
    )
