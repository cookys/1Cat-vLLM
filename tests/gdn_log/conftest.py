# SPDX-License-Identifier: Apache-2.0
"""Guard: the vllm under test must be the stage (or this worktree), never the
serving venv's copy. Set GLOG_EXPECT_ROOT to the directory that must contain
``vllm.__file__``."""

import os

import vllm


def pytest_sessionstart(session) -> None:
    root = os.environ.get("GLOG_EXPECT_ROOT")
    if root:
        real = os.path.realpath(vllm.__file__)
        assert real.startswith(os.path.realpath(root) + os.sep), (
            f"vllm imported from {real}, expected under {root}"
        )
