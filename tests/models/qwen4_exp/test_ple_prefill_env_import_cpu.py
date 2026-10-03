# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import this checkout's full env registry with real metadata, CPU only.

Run directly with an environment containing vLLM's Python dependencies.
This does not AST-extract the registry, stub env_var or load the model.
"""

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FLAG = "VLLM_QWEN4EXP_PLE_PREFILL_LOW_MEMORY"


class EnvImportTests(unittest.TestCase):
    def run_import(self, checks):
        environment = os.environ.copy()
        environment.pop(FLAG, None)
        environment.update(
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            PYTHONPATH=str(ROOT),
        )
        script = (
            "import os\nfrom pathlib import Path\n"
            "import vllm.envs as e\n"
            "from vllm.envs_metadata import EnvVar\n"
            "import torch\n"
            f"expected = Path({str(ROOT / 'vllm/envs.py')!r})\n"
            "assert Path(e.__file__).resolve() == expected\n"
            f"name = {FLAG!r}\n"
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

    def test_import_default_and_complete_metadata(self):
        self.run_import("""
            assert getattr(e, name) is False
            registration = e.environment_variables[name]
            assert isinstance(registration, EnvVar)
            m = registration.metadata
            assert m.category == 'experimental'
            assert m.description and m.acceleration_paths
            assert m.declared_default == '0'
            assert m.effective_default == 'False'
            assert m.automatic_conditions == ()
            assert 'VLLM_SM70_SAMPLING_CUDAGRAPH' in e.environment_variables
        """)

    def test_strict_values(self):
        self.run_import("""
            for value, expected in (('0', False), ('1', True)):
                os.environ[name] = value
                assert getattr(e, name) is expected
            for value in ('2', 'true', ''):
                os.environ[name] = value
                try:
                    getattr(e, name)
                except ValueError:
                    pass
                else:
                    raise AssertionError(value)
            del os.environ[name]
            assert getattr(e, name) is False
        """)


if __name__ == "__main__":
    unittest.main()
