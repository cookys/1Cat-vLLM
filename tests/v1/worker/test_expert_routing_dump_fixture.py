# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The committed synthetic fixture matches the published dump schema."""

import hashlib
import importlib.util
import json
import os

import numpy as np

from vllm.model_executor.layers.fused_moe import expert_routing_dump as erd

HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "fixtures", "expert_routing_dump")
RANK0 = os.path.join(FIXTURE, "rank0")

DTYPES = {
    "step_idx": np.int32,
    "step_wall_ns": np.int64,
    "num_tokens": np.int32,
    "padded_num_tokens": np.int32,
    "step_phase": np.int8,
    "active_requests": np.int32,
    "request_uid": np.int32,
    "position": np.int32,
    "token_is_prefill": np.bool_,
    "row_is_padding": np.bool_,
    "target_topk": np.int16,
    "target_topk_weight": np.float16,
    "draft_topk": np.int16,
    "draft_num_rows": np.int32,
    "shared_expert_used": np.bool_,
}


def _load_all():
    index = [json.loads(line) for line in open(os.path.join(RANK0, "index.jsonl"))]
    return index, {
        e["file"]: dict(np.load(os.path.join(RANK0, e["file"]))) for e in index
    }


def test_fixture_files_index_and_meta_use_the_final_keys():
    index, arrays = _load_all()
    assert all(tuple(e) == erd.INDEX_KEYS for e in index)
    for e in index:
        data = open(os.path.join(RANK0, e["file"]), "rb").read()
        assert e["sha256"] == hashlib.sha256(data).hexdigest()
        z = arrays[e["file"]]
        assert e["valid_steps"] == len(z["step_idx"])
        assert (e["first_step"], e["last_step"]) == (
            z["step_idx"].min(),
            z["step_idx"].max(),
        )
    meta = json.load(open(os.path.join(RANK0, "meta.json")))
    assert set(meta) == set(erd.META_KEYS)
    assert meta["synthetic"] is True
    assert meta["tp_rank"] == 0 and meta["target_layers"] == 48
    assert meta["concurrency"] == 2 and meta["tp"] == 4
    assert meta["num_speculative_tokens"] == 4
    assert meta["padding_traffic"] == "recorded" and meta["dump_padded"] is True
    assert meta["padded_rows_invalid"] == 1
    assert meta["valid_steps"] == 5 and meta["dropped_steps"] == 1
    assert meta["dropped_rows"] == 10
    assert meta["valid_steps"] == sum(e["valid_steps"] for e in index)
    assert meta["dropped_steps"] == sum(e["dropped_steps"] for e in index)
    assert meta["dropped_rows"] == sum(e["dropped_rows"] for e in index)
    assert "env" not in meta and "argv" not in meta


def test_fixture_arrays_have_the_final_names_dtypes_and_shapes():
    _, arrays = _load_all()
    for z in arrays.values():
        assert set(z) == set(erd.NPZ_ARRAYS)
        s, mf = z["target_topk"].shape[:2]
        shapes = {
            "step_idx": (s,),
            "step_wall_ns": (s,),
            "num_tokens": (s,),
            "padded_num_tokens": (s,),
            "step_phase": (s,),
            "active_requests": (s,),
            "request_uid": (s, mf),
            "position": (s, mf),
            "token_is_prefill": (s, mf),
            "row_is_padding": (s, mf),
            "target_topk": (s, mf, 48, 10),
            "target_topk_weight": (s, mf, 48, 10),
            "draft_topk": (s, 4, mf, 1, 10),
            "draft_num_rows": (s, 4),
            "shared_expert_used": (s, mf, 48),
        }
        for name, dtype in DTYPES.items():
            assert z[name].dtype == dtype, name
            assert z[name].shape == shapes[name], name


def test_fixture_content_serves_the_analyser_contract():
    _, arrays = _load_all()
    step_idx = np.concatenate([z["step_idx"] for z in arrays.values()])
    assert sorted(step_idx) == [0, 1, 2, 4, 5]  # step 3 is a hole
    phase = np.concatenate([z["step_phase"] for z in arrays.values()])
    active = np.concatenate([z["active_requests"] for z in arrays.values()])
    pure = (phase == 0) & (active == 2)
    assert pure.sum() == 4  # >= 2 pure-decode steps at nominal c=2
    assert sorted(phase.tolist()) == [0, 0, 0, 0, 2]  # one mixed, never pure
    invalid_padding = 0
    for z in arrays.values():
        for i in range(len(z["step_idx"])):
            n = int(z["num_tokens"][i])
            padded = int(z["padded_num_tokens"][i])
            assert padded == n + 2  # two CUDA-graph padding rows per step
            assert z["target_topk"].shape[1] >= padded
            ids = z["target_topk"][i, :n]
            assert (ids >= 0).all() and (ids < 512).all()
            srt = np.sort(ids, axis=-1)
            assert (srt[..., 1:] != srt[..., :-1]).all()  # 10 distinct per row
            # padding rows: flagged, no request/position, ids kept unless invalid
            assert not z["row_is_padding"][i, :n].any()
            assert z["row_is_padding"][i, n:].all()
            assert (z["request_uid"][i, n:] == -1).all()
            assert (z["position"][i, n:] == -1).all()
            assert not z["token_is_prefill"][i, n:].any()
            pad_rows = z["target_topk"][i, n:padded]
            blank = (pad_rows == -1).all(axis=(1, 2))
            invalid_padding += int(blank.sum())
            assert ((pad_rows >= 0).all(axis=(1, 2)) | blank).all()
            assert (z["target_topk"][i, padded:] == -1).all()
            if z["step_phase"][i] == 2:
                assert z["token_is_prefill"][i, :n].sum() == 4
            assert list(z["draft_num_rows"][i]) == [n, 2, 2, 2]
    assert invalid_padding == 1  # == meta["padded_rows_invalid"]


def test_regenerating_the_fixture_reproduces_the_routing(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "make_fixture", os.path.join(HERE, "fixtures", "make_expert_routing_fixture.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build(str(tmp_path / "run"))
    _, committed = _load_all()
    for name, z in committed.items():
        fresh = dict(np.load(tmp_path / "run" / "rank0" / name))
        for key in ("target_topk", "target_topk_weight", "draft_topk", "step_idx"):
            np.testing.assert_array_equal(fresh[key], z[key])
