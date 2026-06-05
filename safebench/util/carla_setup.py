"""Ensure CARLA Python API and ``agents`` are importable without manual shell setup."""

from __future__ import annotations

import os
import sys


def _repo_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    )


def ensure_carla_pythonpath(repo_root: str | None = None) -> str:
    """Add CARLA egg/agents paths to ``sys.path``. Returns ``CARLA_ROOT`` used."""
    root = repo_root or _repo_root()
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root:
        carla_root = os.path.join(root, "CARLA_0.9.13_safebench")
    carla_root = os.path.abspath(carla_root)
    os.environ.setdefault("CARLA_ROOT", carla_root)

    candidates = [
        os.path.join(
            carla_root,
            "PythonAPI",
            "carla",
            "dist",
            "carla-0.9.13-py3.8-linux-x86_64.egg",
        ),
        os.path.join(carla_root, "PythonAPI", "carla", "agents"),
        os.path.join(carla_root, "PythonAPI", "carla"),
        os.path.join(carla_root, "PythonAPI"),
    ]
    for path in candidates:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    return carla_root


ensure_carla_pythonpath()
