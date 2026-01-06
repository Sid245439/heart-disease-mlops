from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from nox import Session
import nox
from nox_uv import session

MODULE_NAME = "heart_disease_mlops"

PROJECT_DIRECTORY = Path(__file__).parent.resolve()
DOCS_DIRECTORY = PROJECT_DIRECTORY / "doc"
REPORTS_DIRECTORY = DOCS_DIRECTORY / "reports"
RUFF_DIRECTORY = REPORTS_DIRECTORY / "ruff"
PYTEST_DIRECTORY = REPORTS_DIRECTORY / "pytest"
COVERAGE_DIRECTORY = REPORTS_DIRECTORY / "coverage"
FORMAT_DIRECTORY = REPORTS_DIRECTORY / "format"


nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = True


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
    """Run black and ruff to format the code."""
    FORMAT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Save formatter diffs for review.
    with (FORMAT_DIRECTORY / "black-diff.patch").open("w") as f:
        session.run("black", ".", "--diff", stdout=f, stderr=None)

    with (FORMAT_DIRECTORY / "ruff-format-diff.patch").open("w") as f:
        session.run("ruff", "format", ".", "--diff", stdout=f, stderr=None)

    # Apply formatting.
    session.run("black", ".", "--check")
    session.run("ruff", "format", ".", "--check")


@session(
    venv_backend="uv",
    uv_only_groups=["typing"],
    python="3.12",
    uv_no_install_project=True,
)
def typing(session: Session) -> None:
    """Run mypy to check typing issues."""
    session.run("mypy", ".")


@session(
    venv_backend="uv", uv_only_groups=["dev"], python="3.12", uv_no_install_project=True
)
def dev(session: Session) -> None:
    """Set up development environment."""
    # No need to implement anything here for now
    # The virtual environment will be created with the specified packages
    # when this session is called.
    pass
