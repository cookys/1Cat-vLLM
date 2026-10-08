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

PACKAGES = (
    "vllm",
    "torch",
    "numpy",
    "triton",
    "flash_qla",
    "flashinfer",
    "flash_attn_v100",
)
PREBUILT = "flash_qla.ops.gated_delta_rule.chunk.sm70.flash_qla_sm70_gdn_strided"
PREBUILTS = (PREBUILT, "flash_attn_v100.flash_attn_v100_cuda")


def checked_staging(staging, worktree):
    contents = {entry.name for entry in staging.iterdir()}
    if contents != {"vllm"}:
        raise RuntimeError(
            f"staging contents must be {{'vllm'}}, got {sorted(contents)}"
        )
    link = staging / "vllm"
    if not link.is_symlink() or link.resolve() != (worktree / "vllm").resolve():
        raise RuntimeError(f"staging vllm must be a symlink to {worktree / 'vllm'}")
    if not link.is_dir():
        raise RuntimeError(f"staging vllm target is missing: {link}")
    return sorted(contents)


def checked_origin(name, origin, worktree, venv):
    root = worktree / "vllm" if name == "vllm" else venv
    if not origin or not Path(origin).resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"{name} origin {origin!r} is outside {root}")
    return str(Path(origin).resolve())


def checked_native_origin(name, origin, worktree, venv):
    checked_origin(name, origin, worktree, venv)
    if not any(origin.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
        raise RuntimeError(f"{name} prebuilt module is not a native binary")


def check(worktree, venv, staging):
    report = {"status": "FAIL", "prefix": sys.prefix, "packages": {}}
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("import preflight requires CUDA_VISIBLE_DEVICES=''")
        if Path(sys.prefix).resolve() != venv.resolve():
            raise RuntimeError(f"wrong Python prefix {sys.prefix}; expected {venv}")
        report["staging"] = str(staging)
        report["staging_contents"] = checked_staging(staging, worktree)
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
        # Import bundled binaries directly. Never call fused_fwd._load_ext(),
        # which may try a CUDA device check and a source JIT fallback.
        for name in PREBUILTS:
            spec = importlib.util.find_spec(name)
            origin = None if spec is None else spec.origin
            report["packages"][name] = {"spec_origin": origin}
            checked_native_origin(name, origin, worktree, venv)
            module = importlib.import_module(name)
            origin = getattr(module, "__file__", None)
            report["packages"][name]["__file__"] = origin
            checked_native_origin(name, origin, worktree, venv)
            print(f"{name}.__file__ = {origin}", flush=True)
        report["cuda_initialized"] = sys.modules["torch"].cuda.is_initialized()
        if report["cuda_initialized"]:
            raise RuntimeError("import preflight unexpectedly initialized CUDA")
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = repr(exc)
        report["error_type"] = type(exc).__name__
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.worktree, args.venv, args.staging)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
