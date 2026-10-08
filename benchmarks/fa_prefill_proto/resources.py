# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU resource gate; no driver calls. Missing evidence fails closed."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse(resource, build_log, sass):
    def number(name):
        hit = re.search(r"\b" + name + r":(\d+)", resource)
        if not hit:
            raise ValueError("missing resource " + name)
        return int(hit[1])
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", build_log)
    if spill is None:
        raise ValueError("missing PTXAS spill report")
    fields = ("REG", "STACK", "SHARED", "LOCAL")
    result = {name.lower(): number(name) for name in fields}
    result.update(spill_store_bytes=int(spill[1]), spill_load_bytes=int(spill[2]))
    result["hmma_static"] = len(re.findall(r"\bHMMA\.", sass))
    result["ldl_stl_static"] = len(re.findall(r"\b(?:LDL|STL)(?:\.|\s)", sass))
    # Volta allocation granularity: 256 registers/warp; 256 shared bytes/CTA.
    regs_cta = ((result["reg"] * 32 + 255) // 256) * 256 * 16
    shared_cta = ((result["shared"] + 255) // 256) * 256
    result["resource_ctas_per_sm"] = min(65536 // regs_cta, 98304 // shared_cta, 4)
    result["gate_pass"] = bool(
        result["reg"] <= 64 and result["shared"] <= 48 * 1024
        and result["local"] == result["stack"] == 0
        and result["spill_store_bytes"] == result["spill_load_bytes"] == 0
        and result["ldl_stl_static"] == 0 and result["resource_ctas_per_sm"] >= 2
        and result["hmma_static"] > 0
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build_dir", type=Path)
    args = p.parse_args()
    cases = {}
    for variant in range(3):
        paths = {ext: args.build_dir / f"proto{variant}.{ext}" for ext in
                 ("cu", "so", "resources.txt", "build.log", "sass")}
        item = parse(paths["resources.txt"].read_text(),
                     paths["build.log"].read_text(), paths["sass"].read_text())
        item["files_sha256"] = {ext: digest(path) for ext, path in paths.items()}
        item["library"] = str(paths["so"].resolve())
        item["status"] = "GPU_GATE_PENDING" if item["gate_pass"] else "RESOURCE_VETO"
        cases[str(variant)] = item
    print(json.dumps({"cpu_only": True, "variants": cases}, indent=2))


if __name__ == "__main__":
    main()
