# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate isolated sm70 prototype translation units from a pinned kernel body.

P0 is verbatim production body, apart from its kernel symbol/namespace. P1 and
P2 are independent; no serving dispatch or installed library is modified.
"""

import argparse
import hashlib
from pathlib import Path

SOURCE_SHA = "848d8f64228bfd206d8485a9e7d9614d13a3d873a3f56c7e319559ad885b8971"
BEGIN = "constexpr int D256_BM32_PHASE_BLOCK_M = 32;"
END = "template <bool CHECK_SPLIT_EMPTY>\n__global__"
KERNEL = "flash_attention_forward_paged_d256_bm32_phase_kernel"
QK_BEGIN = "    if (warp_id < D256_BM32_PHASE_BLOCK_N / WMMA_N &&"
QK_END = "    __syncthreads();\n\n    const int causal_q_offset"
HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected one source anchor: {old!r}")
    return text.replace(old, new, 1)


def core(source, variant):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError("kernel source changed; review extraction before rebuilding")
    body = source[source.index(BEGIN):source.index(END)]
    qk_start = body.index(QK_BEGIN)
    qk_end = body.index(QK_END, qk_start)
    if variant == 1:
        # Only physical score addresses/WMMA leading dimensions change.
        body = replace_once(body, BEGIN, BEGIN +
                            "\nconstexpr int D256_BM32_PHASE_SCORE_LD = 144;")
        body = replace_once(body,
            "float score[D256_BM32_PHASE_BLOCK_M * D256_BM32_PHASE_BLOCK_N];",
            "float score[D256_BM32_PHASE_BLOCK_M * D256_BM32_PHASE_SCORE_LD];")
        start = body.index(
            "__device__ __forceinline__ void d256_bm32_phase_spill_pair_scratch("
        )
        stop = body.index(
            "__device__ __forceinline__ void d256_bm32_phase_sync_warp_pair(", start
        )
        scratch = (HERE / "scratch_own_warp.cuh").read_text()
        body = body[:start] + scratch + "\n" + body[stop:]
        qk_start, qk_end = body.index(QK_BEGIN), body.index(QK_END)
        qk = body[qk_start:qk_end]
        # WMMA leading dimensions and top/bottom score-row addresses only.
        qk = qk.replace("D256_BM32_PHASE_BLOCK_N,", "D256_BM32_PHASE_SCORE_LD,")
        qk = qk.replace("D256_BM32_PHASE_PANEL_M * D256_BM32_PHASE_BLOCK_N",
                        "D256_BM32_PHASE_PANEL_M * D256_BM32_PHASE_SCORE_LD")
        old = """        const int partner_warp = warp_id ^ 1;
        if (partner_warp * WMMA_N < valid_k_rows) {
          d256_bm32_phase_sync_warp_pair(warp_id >> 1);
        }"""
        new = """        // All scratch words belong to this warp's score strip. Ensure
        // every lane has restored its PV values before WMMA overwrites it.
        __syncwarp();"""
        qk = replace_once(qk, old, new)
        body = body[:qk_start] + qk + body[qk_end:]
        body = body.replace("shared.score[index]",
                            "shared.score[row * D256_BM32_PHASE_SCORE_LD + column]")
        body = body.replace("shared.score + warp_id * D256_BM32_PHASE_BLOCK_N",
                            "shared.score + warp_id * D256_BM32_PHASE_SCORE_LD")
        body = body.replace(
            "(D256_BM32_PHASE_PANEL_M + warp_id) * D256_BM32_PHASE_BLOCK_N",
            "(D256_BM32_PHASE_PANEL_M + warp_id) * D256_BM32_PHASE_SCORE_LD"
        )
        # P1-r2 only: PTXAS spilled the loop-invariant total_n_blocks (4 B).
        # Keep it explicitly in shared rather than live across QK/softmax/PV.
        # The existing initialization barrier publishes it; it never changes.
        old = """  const int total_n_blocks =
      (shared.actual_n + D256_BM32_PHASE_BLOCK_N - 1) / D256_BM32_PHASE_BLOCK_N;
"""
        body = replace_once(body, old, "")
        body = body.replace("total_n_blocks", "shared.total_n_blocks")
        body = replace_once(
            body, "  int actual_n;",
            "  int actual_n;\n  volatile int total_n_blocks;"
        )
        old = """      shared.actual_n = seqused_k[shared.batch_id];
    }
  }
  __syncthreads();"""
        new = """      shared.actual_n = seqused_k[shared.batch_id];
    }
    shared.total_n_blocks = (shared.actual_n + 127) / 128;
  }
  __syncthreads();"""
        body = replace_once(body, old, new)
        # Pad score by 2048 B and account for the shared loop bound plus
        # structure alignment (16 B). Replace original sizes once: sequential
        # intermediate-size replacements can alias the adjacent struct size.
        # The derived split field reuses the enlarged base's tail padding.
        for old_size, new_size in ((35408, 37472), (41936, 44000),
                                   (41952, 44000)):
            body = replace_once(body, str(old_size), str(new_size))

    elif variant == 2:
        body = body[:qk_start] + (HERE / "qk16.cuh").read_text() + body[qk_end:]
    elif variant != 0:
        raise ValueError("variant must be 0, 1 or 2")
    return body.replace(KERNEL, f"astra_fa_proto{variant}_kernel")


def generate(source, variant):
    header = """// Generated from pinned 1Cat source; see generate.py and manifest.json.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
using namespace nvcuda::wmma;
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
"""
    body = core(source, variant)
    wrapper = (HERE / "wrapper.cuh").read_text().replace("PROTO", str(variant))
    return header + f"namespace astra{variant} {{\n" + body + "\n}\n" + wrapper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path,
        default=(HERE.parents[1]
                 / "flash-attention-v100/kernel/fused_mha_forward_paged.cu")
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source.read_text()
    for variant in range(3):
        (args.out / f"proto{variant}.cu").write_text(generate(source, variant))


if __name__ == "__main__":
    main()
