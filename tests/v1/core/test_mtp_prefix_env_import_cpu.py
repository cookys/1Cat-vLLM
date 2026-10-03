# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real vllm.envs import smoke, without model loading or CUDA initialization.

Run directly using an environment with vLLM's Python import dependencies:
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -B \
      tests/v1/core/test_mtp_prefix_env_import_cpu.py

Unlike the allocator tests, this does not extract AST or replace modules. Each
subprocess imports this checkout's complete envs registry and real env_var.
"""

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMES = (
    "VLLM_SM70_MTP_COMMITTED_PREFIX_CACHE",
    "VLLM_SM70_MTP_COMMITTED_PREFIX_CACHE_LOG",
)


class EnvImportTests(unittest.TestCase):
    def run_import(self, checks):
        environment = os.environ.copy()
        for name in NAMES:
            environment.pop(name, None)
        environment.update(
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            PYTHONPATH=str(ROOT),
        )
        script = (
            "import os, sys\n"
            "from pathlib import Path\n"
            "import vllm.envs as e\n"
            "from vllm.envs_metadata import EnvVar\n"
            "import torch\n"
            f"expected_path = Path({str(ROOT / 'vllm/envs.py')!r})\n"
            "assert Path(e.__file__).resolve() == expected_path\n"
            f"names = {NAMES!r}\n"
            + textwrap.dedent(checks)
            + "\nassert not torch.cuda.is_initialized()\n"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_import_defaults_and_metadata(self):
        self.run_import("""
            for name in names:
                assert getattr(e, name) is False, name
                registration = e.environment_variables[name]
                assert isinstance(registration, EnvVar), name
                metadata = registration.metadata
                assert metadata.description
                assert metadata.declared_default == "0"
                assert metadata.effective_default == "False"
                assert metadata.automatic_conditions == ()
                assert metadata.acceleration_paths
            assert e.environment_variables[names[0]].metadata.category == "experimental"
            assert e.environment_variables[names[1]].metadata.category == "debug"
            description = e.environment_variables[names[0]].metadata.description
            assert "Warm-producer policy" in description
            # The integration base must keep both production features available.
            for name in (
                "VLLM_SM70_SAMPLING_CUDAGRAPH",
                "VLLM_QWEN4EXP_PLE_PREFILL_LOW_MEMORY",
            ):
                assert name in e.environment_variables, name
        """)

    def test_real_getters_validate_values(self):
        self.run_import("""
            for name in names:
                for value, expected in (("0", False), ("1", True)):
                    os.environ[name] = value
                    assert getattr(e, name) is expected, (name, value)
                for value in ("2", "true", ""):
                    os.environ[name] = value
                    try:
                        getattr(e, name)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError((name, value))
                del os.environ[name]
                assert getattr(e, name) is False
        """)


if __name__ == "__main__":
    unittest.main()
