# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only import provenance gate for the isolated host-parking window."""

import argparse
import importlib
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path

PACKAGES = ("vllm", "torch", "numpy", "triton", "flash_qla", "flashinfer")
PREBUILT = "flash_qla.ops.gated_delta_rule.chunk.sm70.flash_qla_sm70_gdn_strided"


def checked_origin(name, origin, worktree, venv):
    root = worktree / "vllm" if name == "vllm" else venv
    if not origin or not Path(origin).resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"{name} origin {origin!r} is outside {root}")
    return str(Path(origin).resolve())


def check(worktree, venv):
    report = {"status": "FAIL", "prefix": sys.prefix, "packages": {}}
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("import preflight requires CUDA_VISIBLE_DEVICES=''")
        if Path(sys.prefix).resolve() != venv.resolve():
            raise RuntimeError(f"wrong Python prefix {sys.prefix}; expected {venv}")
        # Resolve every top-level package before importing any of them: refuse
        # shadowed source trees without triggering their imports or JIT loaders.
        for name in PACKAGES:
            spec = importlib.util.find_spec(name)
            origin = None if spec is None else spec.origin
            report["packages"][name] = {"spec_origin": origin}
            checked_origin(name, origin, worktree, venv)
        for name in PACKAGES:
            module = importlib.import_module(name)
            origin = getattr(module, "__file__", None)
            report["packages"][name]["__file__"] = origin
            checked_origin(name, origin, worktree, venv)
            print(f"{name}.__file__ = {origin}", flush=True)
        # Import the bundled binary directly. Never call fused_fwd._load_ext(),
        # which may try a CUDA device check and a source JIT fallback.
        spec = importlib.util.find_spec(PREBUILT)
        origin = None if spec is None else spec.origin
        report["packages"][PREBUILT] = {"spec_origin": origin}
        checked_origin(PREBUILT, origin, worktree, venv)
        if not any(origin.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
            raise RuntimeError("FlashQLA SM70 prebuilt module is not a native binary")
        module = importlib.import_module(PREBUILT)
        origin = getattr(module, "__file__", None)
        report["packages"][PREBUILT]["__file__"] = origin
        checked_origin(PREBUILT, origin, worktree, venv)
        print(f"{PREBUILT}.__file__ = {origin}", flush=True)
        report["cuda_initialized"] = sys.modules["torch"].cuda.is_initialized()
        if report["cuda_initialized"]:
            raise RuntimeError("import preflight unexpectedly initialized CUDA")
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = repr(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.worktree, args.venv)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
