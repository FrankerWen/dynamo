# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduling markers for the GPU-free matrix unit tests."""

from pathlib import Path

import pytest

_TEST_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item_path = Path(str(item.path))
        if _TEST_ROOT not in item_path.parents:
            continue
        item.add_marker(pytest.mark.gpu_0)
        item.add_marker(pytest.mark.planner)
        item.add_marker(pytest.mark.unit)
        item.add_marker(pytest.mark.pre_merge)
