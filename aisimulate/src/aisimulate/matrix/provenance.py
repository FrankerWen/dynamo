# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime linkage and AIC database provenance for formal run manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import subprocess
from pathlib import Path
from typing import Any

from ._utils import sha256_file
from .config import CatalogBundle


def _module_origin(name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_snapshot(path: Path) -> dict[str, Any]:
    root = _git_output(path, "rev-parse", "--show-toplevel")
    if root is None:
        return {"root": str(path), "revision": None, "dirty": None}
    status = _git_output(Path(root), "status", "--porcelain=v1")
    return {
        "root": root,
        "revision": _git_output(Path(root), "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "status_sha256": None
        if status is None
        else hashlib.sha256(status.encode()).hexdigest(),
    }


def _tree_sha256(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def capture_runtime_provenance(catalogs: CatalogBundle) -> dict[str, Any]:
    """Capture exact binding origin plus selected AIC system-data tree hashes."""
    dynamo_root = Path(__file__).resolve().parents[4]
    binding_origin = _module_origin("dynamo._core")
    aic_core_origin = _module_origin("aiconfigurator_core")
    aic_origin = _module_origin("aiconfigurator")
    aic_system_root = (
        None if aic_core_origin is None else aic_core_origin.parent / "systems" / "data"
    )
    selected_systems: dict[str, Any] = {}
    for system_id in sorted({sku.aic_system for sku in catalogs.skus.values()}):
        system_path = None if aic_system_root is None else aic_system_root / system_id
        selected_systems[system_id] = {
            "path": None if system_path is None else str(system_path),
            "exists": False if system_path is None else system_path.is_dir(),
            "sha256": None if system_path is None else _tree_sha256(system_path),
        }
    aic_git_anchor = aic_core_origin or aic_origin
    return {
        "dynamo_code": _git_snapshot(dynamo_root),
        "binding": {
            "origin": None if binding_origin is None else str(binding_origin),
            "sha256": None
            if binding_origin is None or not binding_origin.is_file()
            else sha256_file(binding_origin),
            "distribution_version": _distribution_version("ai-dynamo-runtime"),
        },
        "aiconfigurator": {
            "core_origin": None if aic_core_origin is None else str(aic_core_origin),
            "python_origin": None if aic_origin is None else str(aic_origin),
            "core_distribution_version": _distribution_version("aiconfigurator-core"),
            "python_distribution_version": _distribution_version("aiconfigurator"),
            "git": None if aic_git_anchor is None else _git_snapshot(aic_git_anchor),
            "system_database_root": None
            if aic_system_root is None
            else str(aic_system_root),
            "selected_systems": selected_systems,
        },
    }
