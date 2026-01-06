from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from nox import Session
import nox
from nox_uv import session

MODULE_NAME = "heart_disease_mlops"

nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = True

@session(venv_backend="uv", uv_groups=["lint"], python="3.12", uv_no_install_project=True)
def lint(session: Session) -> None:
    """Run ruff to check linting issues."""
    session.run("ruff", "check", ".")

@session(venv_backend="uv", uv_groups=["format"], python="3.12", uv_no_install_project=True)
def format(session: Session) -> None:
    """Run black and ruff to format the code."""
    session.run("ruff", "format", ".")

@session(venv_backend="uv", uv_groups=["typing"], python="3.12", uv_no_install_project=True)
def typing(session: Session) -> None:
    """Run mypy to check typing issues."""
    session.run("mypy", ".")

@session(venv_backend="uv", uv_groups=["dev"], python="3.12", uv_no_install_project=True)
def dev(session: Session) -> None:
    """Set up development environment."""
    # No need to implement anything here for now
    # The virtual environment will be created with the specified packages
    # when this session is called.
    pass
