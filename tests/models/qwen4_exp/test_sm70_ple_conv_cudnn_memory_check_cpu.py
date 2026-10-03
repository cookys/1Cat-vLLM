# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for benchmarks/sm70_ple_conv_cudnn_memory_check.py.

The script needs a GPU to answer its question. What can run without one is
everything around the measurement: argument parsing, the geometry it reads from
the model config, the scenario planning against a fake ``mem_get_info``, the
pure-PyTorch reference convolution, the hashing and diff helpers, the cuDNN log
parser, the verdict logic, the worker command lines and environments, a full
worker run on CPU tensors with a fake memory probe, and the driver with a fake
worker.

Reference convolution: it is compared with ``F.conv1d`` on CPU. With
integer-valued inputs every partial sum is exactly representable in fp32, so the
two are bit-equal whatever order the backend adds the taps in; with random fp32
inputs the order matters in the last bits, so that case checks a relative
tolerance of 1e-5.

    cd <worktree> && CUDA_VISIBLE_DEVICES= PYTHONPATH=<worktree> \
        uv run --no-project --python <venv>/bin/python --with pytest -- \
        python -m pytest --noconftest \
        tests/models/qwen4_exp/test_sm70_ple_conv_cudnn_memory_check_cpu.py -q
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "sm70_ple_conv_cudnn_memory_check.py"
)
REAL_CONFIG = Path("/data/models/Qwen3.8-Flash-Next-NVFP4/config.json")
MIB = 1 << 20
GIB = 1 << 30


@pytest.fixture(scope="module")
def chk():
    spec = importlib.util.spec_from_file_location("sm70_ple_conv_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


# ------------------------------------------------------------ argument parsing


def test_defaults_match_the_task(chk):
    args = chk.parse_args([])
    assert args.dtype == "float16"
    assert args.num_prefills == 2
    assert args.max_len == [2048, 8192]
    assert args.repeats == 3
    assert args.free_mib == [16384, 8192, 4096, 2048, 1024, 512]
    assert args.wscap_mb == [256, 64]
    assert args.config == "/data/models/Qwen3.8-Flash-Next-NVFP4/config.json"
    assert (args.channels, args.kernel_size, args.dilation) == (None, None, None)
    assert args.worker is None and args.out is None
    assert args.algo_settings == "default" and args.settings == "all"


def test_overrides_and_lists_are_parsed(chk):
    args = chk.parse_args(
        [
            "--channels", "64", "--kernel-size", "3", "--dilation", "2",
            "--dtype", "bfloat16", "--num-prefills", "1",
            "--max-len", "128", "256", "512", "--wscap-mb", "32",
            "--settings", "default,benchmark", "--algo-settings", "none",
            "--out", "/tmp/x.json",
        ]
    )  # fmt: skip
    assert (args.channels, args.kernel_size, args.dilation) == (64, 3, 2)
    assert args.dtype == "bfloat16" and args.num_prefills == 1
    assert args.max_len == [128, 256, 512] and args.wscap_mb == [32]
    assert args.out == "/tmp/x.json"


@pytest.mark.parametrize(
    "argv",
    [
        ["--repeats", "0"],
        ["--num-prefills", "0"],
        ["--max-len", "0"],
        ["--dtype", "int8"],
    ],
)
def test_bad_arguments_exit_2(chk, argv):
    with pytest.raises(SystemExit) as exc:
        chk.parse_args(argv)
    assert exc.value.code == 2


def test_setting_selection(chk):
    available = ["default", "deterministic", "benchmark", "wscap256"]
    assert chk._select_names("all", available) == available
    assert chk._select_names("none", available) == []
    assert chk._select_names("default, wscap256", available) == [
        "default",
        "wscap256",
    ]
    with pytest.raises(SystemExit):
        chk._select_names("nonsense", available)


def test_settings_and_flags(chk):
    settings = chk.build_settings([256, 64])
    assert list(settings) == [
        "default",
        "deterministic",
        "benchmark",
        "wscap256",
        "wscap64",
    ]
    assert chk.flags_for("default") == (False, False)
    assert chk.flags_for("deterministic") == (False, True)
    assert chk.flags_for("benchmark") == (True, False)
    assert chk.flags_for("wscap64") == (False, False)
    assert settings["wscap256"]["env"] == {"CUDNN_CONV_WSCAP_DBG": "256"}
    assert settings["default"]["env"] == {}


# -------------------------------------------------------------------- geometry


@pytest.mark.skipif(not REAL_CONFIG.exists(), reason="model config not on this host")
def test_geometry_from_the_real_model_config(chk):
    geometry = chk.read_geometry_from_config(REAL_CONFIG)
    # hidden_size 2560 x hc_count 4, ple_conv_kernel_size 4, ngram_size 3.
    assert geometry == {"channels": 10240, "kernel_size": 4, "dilation": 3}
    args = chk.parse_args(["--config", str(REAL_CONFIG)])
    resolved = chk.resolve_geometry(args)
    assert resolved.conv_state_len == 9  # the production conv_state is [*, 10240, 9]
    assert str(REAL_CONFIG) in resolved.source


def test_geometry_keys_in_text_config_or_top_level(chk, tmp_path):
    keys = {"hidden_size": 8, "hc_count": 2, "ple_conv_kernel_size": 3, "ngram_size": 2}
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"text_config": keys, "vision_config": {}}))
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps(keys))
    expected = {"channels": 16, "kernel_size": 3, "dilation": 2}
    assert chk.read_geometry_from_config(nested) == expected
    assert chk.read_geometry_from_config(flat) == expected


def test_geometry_missing_or_incomplete_config(chk, tmp_path):
    assert chk.read_geometry_from_config(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert chk.read_geometry_from_config(broken) is None
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"text_config": {"hidden_size": 8}}))
    assert chk.read_geometry_from_config(partial) is None


def test_overrides_beat_the_config_and_defaults_are_the_production_values(
    chk, tmp_path
):
    nowhere = tmp_path / "absent.json"
    args = chk.parse_args(["--config", str(nowhere)])
    geometry = chk.resolve_geometry(args)
    assert (geometry.channels, geometry.kernel_size, geometry.dilation) == (
        10240,
        4,
        3,
    )
    assert "production defaults" in geometry.source

    args = chk.parse_args(
        ["--config", str(nowhere), "--dilation", "5", "--channels", "6"]
    )
    geometry = chk.resolve_geometry(args)
    assert (geometry.channels, geometry.dilation) == (6, 5)
    assert geometry.conv_state_len == 15 and "overridden" in geometry.source


# --------------------------------------------------------- scenario planning


def test_scenarios_from_a_fake_mem_get_info_with_plenty_of_memory(chk):
    free = 31 * GIB
    plans = chk.plan_scenarios(lambda: (free, 32 * GIB))
    assert [p.name for p in plans] == [
        "whole",
        "free16GiB",
        "free8GiB",
        "free4GiB",
        "free2GiB",
        "free1GiB",
        "free512MiB",
    ]
    assert all(p.reachable for p in plans)
    assert plans[0].dummy_bytes == 0 and plans[0].target_free_bytes is None
    for plan in plans[1:]:
        assert plan.dummy_bytes == free - plan.target_free_bytes
    assert plans[1].target_free_bytes == 16 * GIB
    assert plans[-1].target_free_bytes == 512 * MIB


def test_scenarios_that_cannot_be_reached_are_skipped(chk):
    # 6 GiB free: the 16 and 8 GiB targets would need negative dummies.
    plans = chk.plan_scenarios(lambda: (6 * GIB, 32 * GIB))
    reachable = {p.name: p.reachable for p in plans}
    assert reachable == {
        "whole": True,
        "free16GiB": False,
        "free8GiB": False,
        "free4GiB": True,
        "free2GiB": True,
        "free1GiB": True,
        "free512MiB": True,
    }
    skipped = next(p for p in plans if p.name == "free8GiB")
    assert skipped.dummy_bytes == 0 and "nothing to take away" in skipped.reason


def test_a_target_within_the_margin_of_the_free_memory_is_the_whole_scenario(chk):
    free = 16 * GIB + chk.MIN_DUMMY_BYTES - 1
    plan = chk.plan_scenario(free, 16 * GIB)
    assert not plan.reachable
    plan = chk.plan_scenario(16 * GIB + chk.MIN_DUMMY_BYTES, 16 * GIB)
    assert plan.reachable and plan.dummy_bytes == chk.MIN_DUMMY_BYTES


def test_scenario_names(chk):
    assert chk.scenario_name(None) == "whole"
    assert chk.scenario_name(2 * GIB) == "free2GiB"
    assert chk.scenario_name(512 * MIB) == "free512MiB"
    assert chk.scenario_name(1536 * MIB) == "free1536MiB"


# ------------------------------------------------------------ reference conv


@pytest.mark.parametrize("dilation", [1, 3])
def test_reference_conv_is_bit_equal_to_conv1d_on_exactly_summable_inputs(
    chk, dilation
):
    generator = torch.Generator().manual_seed(0)
    x = torch.randint(-8, 9, (2, 6, 40), generator=generator).float()
    w = torch.randint(-4, 5, (6, 1, 4), generator=generator).float()
    got = chk.reference_depthwise_dilated_conv(x, w, dilation)
    want = F.conv1d(x, w, groups=6, dilation=dilation)
    assert got.shape == want.shape == (2, 6, 40 - dilation * 3)
    assert got.dtype == torch.float32
    assert torch.equal(got, want)


def test_reference_conv_matches_conv1d_within_fp32_tolerance_on_random_input(chk):
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(2, 16, 200, generator=generator)
    w = torch.randn(16, 1, 4, generator=generator) * 0.5
    got = chk.reference_depthwise_dilated_conv(x, w, 3)
    want = F.conv1d(x, w, groups=16, dilation=3)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_reference_conv_accumulates_in_fp32_for_half_inputs(chk):
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(1, 8, 64, generator=generator).half()
    w = torch.randn(8, 1, 4, generator=generator).half()
    got = chk.reference_depthwise_dilated_conv(x, w, 2)
    assert got.dtype == torch.float32
    want = F.conv1d(x.float(), w.float(), groups=8, dilation=2)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_reference_conv_is_deterministic_and_rejects_short_inputs(chk):
    x = torch.randn(1, 4, 30)
    w = torch.randn(4, 1, 4)
    a = chk.reference_depthwise_dilated_conv(x, w, 3)
    assert torch.equal(a, chk.reference_depthwise_dilated_conv(x, w, 3))
    with pytest.raises(ValueError, match="shorter"):
        chk.reference_depthwise_dilated_conv(torch.randn(1, 4, 9), w, 3)


# ----------------------------------------------------------- hashing and diff


def test_sha256_is_over_the_raw_bytes(chk):
    t = torch.arange(10, dtype=torch.float16)
    assert chk.sha256_tensor(t) == hashlib.sha256(t.numpy().tobytes()).hexdigest()
    other = t.clone()
    other[3] += 1
    assert chk.sha256_tensor(other) != chk.sha256_tensor(t)
    # Same values, different dtype: different bytes.
    assert chk.sha256_tensor(t.float()) != chk.sha256_tensor(t)
    # A non-contiguous view hashes like its contiguous copy.
    m = torch.arange(12, dtype=torch.float32).view(3, 4)
    assert chk.sha256_tensor(m.t()) == chk.sha256_tensor(m.t().contiguous())
    b = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert (
        chk.sha256_tensor(b)
        == hashlib.sha256(b.view(torch.int16).numpy().tobytes()).hexdigest()
    )


def test_max_abs_diff_across_chunks_and_dtypes(chk):
    a = torch.zeros(5, 7, dtype=torch.float16)
    b = a.clone()
    b[4, 6] = 0.5
    b[0, 0] = -0.25
    assert chk.max_abs_diff(a, b) == 0.5
    assert chk.max_abs_diff(a, b, chunk=4) == 0.5  # the maximum is in a later chunk
    assert chk.max_abs_diff(a, a.clone()) == 0.0
    assert chk.max_abs_diff(a, b.float()) == 0.5  # fp16 against fp32
    with pytest.raises(ValueError, match="shape"):
        chk.max_abs_diff(a, torch.zeros(7, 5))


# -------------------------------------------------------------- cuDNN log parse

LEGACY_LOG = """\
I! CuDNN (v91002 17) function cudnnSetConvolutionGroupCount() called:
i!     groupCount: type=int; val=10240;
I! CuDNN (v91002 17) function cudnnConvolutionForward() called:
i!     handle: type=cudnnHandle_t; streamId=0x0;
i!     algo: type=cudnnConvolutionFwdAlgo_t; val=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM (1);
i!     workSpaceSizeInBytes: type=size_t; val=1048576;
I! CuDNN (v91002 17) function cudnnConvolutionForward() called:
i!     algo: type=cudnnConvolutionFwdAlgo_t; val=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM (1);
"""  # noqa: E501

BACKEND_LOG = """\
I! CuDNN (v91002 17) function cudnnBackendSetAttribute() called:
i!     descriptor: type=CUDNN_BACKEND_ENGINE_DESCRIPTOR; val=0x55;
i!     attributeName: type=cudnnBackendAttributeName_t; val=CUDNN_ATTR_ENGINE_GLOBAL_INDEX (200);
i!     attributeType: type=cudnnBackendAttributeType_t; val=CUDNN_TYPE_INT64 (0);
i!     elementCount: type=int64_t; val=1;
i!     arrayOfElements: type=int64_t; val=[42];
I! CuDNN (v91002 17) function cudnnBackendExecute() called:
i!     handle: type=cudnnHandle_t; streamId=0x0;
"""  # noqa: E501

FRONTEND_LOG = """\
[cudnn_frontend] INFO: Executing "eng14_k2=2_k3=0_k6=1" plan with workspace 262144
[cudnn_frontend] INFO: Executing "eng14_k2=2_k3=0_k6=1" plan with workspace 262144
[cudnn_frontend] INFO: Trying plan "eng3_k1=1"
"""


def test_legacy_algorithm_enum_is_found_once_with_the_call_count(chk):
    parsed = chk.parse_cudnn_log(LEGACY_LOG)
    assert parsed["legacy_algos"] == [
        "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"
    ]
    assert parsed["convolution_forward_calls"] == 2
    assert parsed["backend_execute_calls"] == 0
    assert parsed["has_algorithm_info"] is True


def test_backend_api_engine_index_and_execute_calls(chk):
    parsed = chk.parse_cudnn_log(BACKEND_LOG)
    assert parsed["engine_global_indices"] == ["42"]
    assert parsed["backend_execute_calls"] == 1
    assert parsed["legacy_algos"] == [] and parsed["convolution_forward_calls"] == 0
    assert parsed["has_algorithm_info"] is True


def test_frontend_plan_tags_keep_their_order_without_duplicates(chk):
    parsed = chk.parse_cudnn_log(FRONTEND_LOG)
    assert parsed["engine_tags"] == ["eng14_k2=2_k3=0_k6=1", "eng3_k1=1"]
    assert parsed["has_algorithm_info"] is True


def test_a_log_without_algorithm_entries_says_so(chk):
    for text in ("", "I! CuDNN (v91002 17) function cudnnCreate() called:\n"):
        parsed = chk.parse_cudnn_log(text)
        assert parsed["has_algorithm_info"] is False
        assert parsed["legacy_algos"] == parsed["engine_tags"] == []
    line = chk.describe_algorithm(chk.parse_cudnn_log(""), None)
    assert "unknown" in line


def test_algorithm_description_prefers_the_log_then_the_kernel(chk):
    parsed = chk.parse_cudnn_log(LEGACY_LOG + FRONTEND_LOG)
    assert "IMPLICIT_PRECOMP_GEMM" in chk.describe_algorithm(parsed, ["k"])
    empty = chk.parse_cudnn_log("")
    assert chk.describe_algorithm(empty, ["conv_depthwise2d_forward_kernel"]) == (
        "kernel conv_depthwise2d_forward_kernel"
    )
    assert chk.describe_algorithm(None, ["a", "b"]) == "kernel a"
    only_tags = chk.parse_cudnn_log(FRONTEND_LOG)
    assert chk.describe_algorithm(only_tags, ["k"]).startswith(
        "cudnn engine_tags=eng14"
    )


# ------------------------------------------------------------------- verdict


def scenario(name: str, digest: str, repeats: bool = True, status: str = "ok") -> dict:
    if status != "ok":
        return {"name": name, "status": status}
    return {
        "name": name,
        "status": "ok",
        "sha256": [digest, digest, digest if repeats else digest + "x"],
        "repeats_identical": repeats,
    }


def worker_result(digests: dict[str, str], **kwargs) -> dict:
    return {
        "scenarios": [scenario(n, d, **kwargs) for n, d in digests.items()],
        "backend": {"cudnn_used": False, "selected": "_ConvBackend.CudaDepthwise2d"},
    }


SAME = {"whole": "aa", "free8GiB": "aa", "free512MiB": "aa"}
DIFFERENT = {"whole": "aa", "free8GiB": "aa", "free512MiB": "bb"}


def test_verdict_when_nothing_depends_on_memory(chk):
    results = {
        "default": {"2048": worker_result(SAME), "8192": worker_result(SAME)},
        "deterministic": {"2048": worker_result(SAME)},
        "wscap256": {"2048": worker_result(SAME)},
    }
    backends = {"default/2048": {"cudnn_used": False}}
    verdict = chk.compute_verdict(results, backends)
    assert verdict["MEMORY_DEPENDENT"] == "NO"
    assert verdict["REPEATS_IDENTICAL"] == "YES"
    assert verdict["DETERMINISTIC_FLAG_FIXES"] == "N/A"
    assert verdict["WSCAP_FIXES"] == {"wscap256": "N/A"}
    assert verdict["CUDNN_USED"] == "NO"
    assert (
        verdict["settings"]["default"]["max_len"]["2048"]["distinct_output_hashes"] == 1
    )


def test_verdict_when_the_default_depends_on_memory_and_flags_fix_it(chk):
    results = {
        "default": {"2048": worker_result(DIFFERENT)},
        "deterministic": {"2048": worker_result(SAME)},
        "benchmark": {"2048": worker_result(DIFFERENT)},
        "wscap256": {"2048": worker_result(SAME)},
        "wscap64": {"2048": worker_result(DIFFERENT)},
    }
    verdict = chk.compute_verdict(results, {"default/2048": {"cudnn_used": True}})
    assert verdict["MEMORY_DEPENDENT"] == "YES"
    assert verdict["DETERMINISTIC_FLAG_FIXES"] == "YES"
    assert verdict["BENCHMARK_FLAG_FIXES"] == "NO"
    assert verdict["WSCAP_FIXES"] == {"wscap256": "YES", "wscap64": "NO"}
    assert verdict["CUDNN_USED"] == "YES"
    # The bytes of the whole-memory run are the same in every setting here.
    assert all(verdict["SAME_BYTES_AS_DEFAULT_WHOLE"].values())


def test_dependence_in_any_max_len_counts(chk):
    results = {
        "default": {"2048": worker_result(SAME), "8192": worker_result(DIFFERENT)}
    }
    verdict = chk.compute_verdict(results, None)
    assert verdict["MEMORY_DEPENDENT"] == "YES"
    assert verdict["CUDNN_USED"] == "UNKNOWN"
    assert (
        verdict["settings"]["default"]["max_len"]["8192"]["distinct_output_hashes"] == 2
    )


def test_repeats_that_disagree_are_reported_separately(chk):
    results = {"default": {"2048": worker_result(SAME, repeats=False)}}
    verdict = chk.compute_verdict(results, None)
    assert verdict["REPEATS_IDENTICAL"] == "NO"
    # The first repeat of every scenario still agrees across scenarios.
    assert verdict["MEMORY_DEPENDENT"] == "NO"


def test_skipped_and_oom_scenarios_do_not_count(chk):
    per = {
        "scenarios": [
            scenario("whole", "aa"),
            scenario("free512MiB", "", status="oom"),
            scenario("free16GiB", "", status="skipped"),
        ],
        "backend": {"cudnn_used": None},
    }
    verdict = chk.compute_verdict(
        {"default": {"2048": per}}, {"x": {"cudnn_used": None}}
    )
    summary = verdict["settings"]["default"]["max_len"]["2048"]
    assert summary["scenarios_run"] == 1
    assert summary["oom_scenarios"] == ["free512MiB"]
    assert verdict["MEMORY_DEPENDENT"] == "NO"
    assert verdict["CUDNN_USED"] == "UNKNOWN"


def test_verdict_without_the_default_setting_is_unknown(chk):
    verdict = chk.compute_verdict({"benchmark": {"2048": worker_result(SAME)}}, None)
    assert verdict["MEMORY_DEPENDENT"] == "UNKNOWN"


# ----------------------------------------------------------------- subprocess


def test_worker_command_carries_the_resolved_geometry(chk):
    args = chk.parse_args(["--repeats", "2", "--free-mib", "1024", "512"])
    geometry = chk.Geometry(channels=10240, kernel_size=4, dilation=3, source="t")
    command = chk.build_worker_command(
        "/x/script.py", "algo", "wscap64", 8192, args, geometry, "free512MiB"
    )
    assert command[1:3] == ["/x/script.py", "--worker"]
    assert command[3] == "algo"
    pairs = dict(zip(command[4:], command[5:]))
    assert pairs["--setting"] == "wscap64"
    assert pairs["--worker-max-len"] == "8192"
    assert (pairs["--channels"], pairs["--kernel-size"], pairs["--dilation"]) == (
        "10240",
        "4",
        "3",
    )
    assert pairs["--repeats"] == "2" and pairs["--worker-scenario"] == "free512MiB"
    assert "--free-mib" in command and command[command.index("--free-mib") + 1 :][
        :2
    ] == [
        "1024",
        "512",
    ]
    # The worker parses its own command line back to the same values.
    parsed = chk.parse_args(command[2:])
    assert parsed.worker == "algo" and parsed.setting == "wscap64"
    assert parsed.worker_max_len == 8192 and parsed.free_mib == [1024, 512]
    assert parsed.worker_scenario == "free512MiB" and parsed.channels == 10240


def test_worker_environment(chk, monkeypatch):
    monkeypatch.setenv("KEEP_ME", "1")
    env = chk.build_worker_env({"CUDNN_CONV_WSCAP_DBG": "256"})
    assert env["CUDNN_CONV_WSCAP_DBG"] == "256" and env["KEEP_ME"] == "1"
    assert "CUDNN_LOGLEVEL_DBG" not in env
    logged = chk.build_worker_env({}, ("/tmp/api.log", "/tmp/fe.log"))
    assert logged["CUDNN_LOGLEVEL_DBG"] == "3"
    assert logged["CUDNN_LOGDEST_DBG"] == "/tmp/api.log"
    assert logged["CUDNN_FRONTEND_LOG_INFO"] == "1"
    assert logged["CUDNN_FRONTEND_LOG_FILE"] == "/tmp/fe.log"


def test_result_extraction(chk):
    out = 'warning noise\nRESULT_JSON:{"a": 1}\nRESULT_JSON:{"a": 2}\ntrailing\n'
    assert chk.extract_result(out) == {"a": 2}
    with pytest.raises(ValueError, match="no result"):
        chk.extract_result("nothing here\n")


def test_run_worker_reads_the_result_of_a_real_subprocess(chk):
    code = (
        "import sys; print('noise'); "
        "print('RESULT_JSON:{\"ok\": 7}'); sys.stderr.write('w')"
    )
    result = chk.run_worker([sys.executable, "-c", code], {})
    assert result == {"ok": 7}


def test_run_worker_raises_with_the_stderr_tail_on_failure(chk):
    code = "import sys; sys.stderr.write('Traceback: boom\\n'); sys.exit(3)"
    command = [sys.executable, "-c", code, "x", "y", "z", "u", "v", "w"]
    with pytest.raises(RuntimeError, match="worker exited 3") as exc:
        chk.run_worker(command, {})
    assert "boom" in str(exc.value)
    with pytest.raises(ValueError, match="no result"):
        chk.run_worker([sys.executable, "-c", "print('no result line')"], {})


# ------------------------------------------------- full worker on CPU tensors


class FakeMemory:
    """mem_get_info for a device with ``free`` bytes and no real allocation."""

    def __init__(self, free: int):
        self.free = free

    def __call__(self, device):
        return self.free, self.free * 2


@pytest.fixture
def cpu_worker(chk, monkeypatch):
    monkeypatch.setattr(chk, "WORKER_DEVICE", "cpu")
    monkeypatch.setattr(chk, "MIN_DUMMY_BYTES", 1 * MIB)
    memory = FakeMemory(64 * MIB)
    monkeypatch.setattr(chk, "_mem_get_info", memory)
    return memory


def worker_args(chk, setting="default", free_mib=("32", "16", "100"), extra=()):
    return chk.parse_args(
        [
            "--worker", "scenarios", "--setting", setting, "--worker-max-len", "24",
            "--channels", "8", "--kernel-size", "3", "--dilation", "2",
            "--num-prefills", "2", "--repeats", "3", "--free-mib", *free_mib, *extra,
        ]
    )  # fmt: skip


def small_geometry(chk):
    return chk.Geometry(channels=8, kernel_size=3, dilation=2, source="test")


def test_a_full_worker_run_on_cpu(chk, cpu_worker):
    result = chk.worker_scenarios(worker_args(chk), small_geometry(chk))
    by_name = {s["name"]: s for s in result["scenarios"]}
    assert list(by_name) == ["whole", "free32MiB", "free16MiB", "free100MiB"]
    # 100 MiB is more than the 64 MiB the fake device has free.
    assert by_name["free100MiB"]["status"] == "skipped"
    assert "nothing to take away" in by_name["free100MiB"]["reason"]
    assert by_name["whole"]["dummy_mib"] == 0
    assert by_name["free32MiB"]["dummy_mib"] == 32
    assert by_name["free16MiB"]["dummy_mib"] == 48
    ok = [s for s in result["scenarios"] if s["status"] == "ok"]
    assert len(ok) == 3
    for s in ok:
        assert len(s["sha256"]) == 3 and s["repeats_identical"] is True
        # The CPU conv does not depend on memory: every scenario matches the first.
        assert s["max_abs_diff_vs_first_scenario"] == 0.0
        # fp16 output against the fp32 reference: only the output rounding.
        assert s["max_abs_diff_vs_reference_fp32"] < 5e-2
    assert len({s["sha256"][0] for s in ok}) == 1
    assert result["setting"] == "default" and result["max_len"] == 24
    assert result["torch_cudnn"] == {"benchmark": False, "deterministic": False}
    verdict = chk.compute_verdict({"default": {"24": result}}, {"d": result["backend"]})
    assert verdict["MEMORY_DEPENDENT"] == "NO"
    assert verdict["CUDNN_USED"] == "NO"  # CPU tensors never select cuDNN


def test_worker_applies_the_cudnn_flags_of_its_setting(chk, cpu_worker):
    saved = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        result = chk.worker_scenarios(
            worker_args(chk, "benchmark"), small_geometry(chk)
        )
        assert result["torch_cudnn"] == {"benchmark": True, "deterministic": False}
        result = chk.worker_scenarios(
            worker_args(chk, "deterministic"), small_geometry(chk)
        )
        assert result["torch_cudnn"] == {"benchmark": False, "deterministic": True}
        result = chk.worker_scenarios(worker_args(chk, "wscap64"), small_geometry(chk))
        assert result["torch_cudnn"] == {"benchmark": False, "deterministic": False}
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved


def test_an_out_of_memory_conv_is_recorded_not_raised(chk, cpu_worker, monkeypatch):
    real_conv = chk._conv
    calls = {"n": 0}

    def flaky(history, weights, dilation):
        calls["n"] += 1
        if calls["n"] > 3:  # the whole-memory scenario (3 repeats) succeeds
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate")
        return real_conv(history, weights, dilation)

    monkeypatch.setattr(chk, "_conv", flaky)
    result = chk.worker_scenarios(
        worker_args(chk, free_mib=("32",)), small_geometry(chk)
    )
    whole, small = result["scenarios"]
    assert whole["status"] == "ok" and small["status"] == "oom"
    assert "out of memory" in small["reason"].lower()
    assert "sha256" not in small


def test_a_cudnn_error_is_recorded_as_an_error_scenario(chk, cpu_worker, monkeypatch):
    """No engine fits the workspace: cuDNN raises a RuntimeError, not an OOM."""
    real_conv = chk._conv
    calls = {"n": 0}

    def no_engine(history, weights, dilation):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. details")
        return real_conv(history, weights, dilation)

    monkeypatch.setattr(chk, "_conv", no_engine)
    result = chk.worker_scenarios(
        worker_args(chk, free_mib=("32",)), small_geometry(chk)
    )
    assert [s["status"] for s in result["scenarios"]] == ["ok", "error"]
    assert "CUDNN_STATUS_NOT_SUPPORTED" in result["scenarios"][1]["reason"]
    summary = chk.compute_verdict({"default": {"24": result}}, None)["settings"][
        "default"
    ]["max_len"]["24"]
    assert (
        summary["error_scenarios"] == ["free32MiB"] and summary["oom_scenarios"] == []
    )
    assert summary["scenarios_run"] == 1


def test_a_differing_output_in_a_later_scenario_is_visible(
    chk, cpu_worker, monkeypatch
):
    """The worker must report a memory-dependent conv, not hide it."""
    real_conv = chk._conv
    state = {"scenario_calls": 0}

    def memory_dependent(history, weights, dilation):
        state["scenario_calls"] += 1
        out = real_conv(history, weights, dilation)
        if state["scenario_calls"] > 3:  # every scenario after the first
            out = out + torch.tensor(2.0**-9, dtype=out.dtype)
        return out

    monkeypatch.setattr(chk, "_conv", memory_dependent)
    result = chk.worker_scenarios(
        worker_args(chk, free_mib=("32",)), small_geometry(chk)
    )
    whole, small = result["scenarios"]
    assert whole["max_abs_diff_vs_first_scenario"] == 0.0
    assert small["max_abs_diff_vs_first_scenario"] > 0.0
    assert small["sha256"][0] != whole["sha256"][0]
    verdict = chk.compute_verdict({"default": {"24": result}}, None)
    assert verdict["MEMORY_DEPENDENT"] == "YES"


def test_algo_worker_on_cpu(chk, cpu_worker):
    args = chk.parse_args(
        [
            "--worker", "algo", "--setting", "default", "--worker-max-len", "24",
            "--worker-scenario", "free16MiB", "--channels", "8", "--kernel-size", "3",
            "--dilation", "2", "--num-prefills", "2", "--free-mib", "32", "16",
        ]
    )  # fmt: skip
    record = chk.worker_algo(args, small_geometry(chk))
    assert record["device"] == {"name": "cpu", "capability": None}
    assert record["status"] == "ok" and record["scenario"] == "free16MiB"
    assert record["backend"]["cudnn_used"] is False
    assert record["run2_matches_run1"] is True
    assert len(record["sha256"]) == 64
    # No CUDA profiler on a CPU run: the kernel list is absent, not an error.
    assert record["kernels_run1"] is None or isinstance(record["kernels_run1"], list)


# --------------------------------------------------------------- the driver


def fake_worker_factory(chk, calls, fail: str | None = None):
    names = ["whole", "free16GiB", "free512MiB"]

    def run_worker(command: list[str], env: dict[str, str]) -> dict[str, Any]:
        calls.append(command)
        mode = command[command.index("--worker") + 1]
        setting = command[command.index("--setting") + 1]
        max_len = int(command[command.index("--worker-max-len") + 1])
        if fail == setting:
            raise RuntimeError("worker exited 1: boom")
        if mode == "scenarios":
            memory_dependent = setting == "default" and max_len == 8192
            return {
                "setting": setting,
                "max_len": max_len,
                "device": {"name": "Fake V100", "capability": [7, 0]},
                "backend": {
                    "cudnn_used": False,
                    "selected": "_ConvBackend.CudaDepthwise2d",
                },
                "scenarios": [
                    {
                        "name": n,
                        "status": "ok",
                        "sha256": [
                            "bb" if memory_dependent and n == "free512MiB" else "aa"
                        ]
                        * 3,
                        "repeats_identical": True,
                    }
                    for n in names
                ],
            }
        scenario_name = command[command.index("--worker-scenario") + 1]
        Path(env["CUDNN_LOGDEST_DBG"]).write_text(LEGACY_LOG)
        Path(env["CUDNN_FRONTEND_LOG_FILE"]).write_text("")
        return {
            "setting": setting,
            "max_len": max_len,
            "scenario": scenario_name,
            "backend": {"selected": "_ConvBackend.CudaDepthwise2d"},
            "kernels_run1": ["conv_depthwise2d_forward_kernel"],
            "kernels_run2": ["conv_depthwise2d_forward_kernel"],
            "status": "ok",
        }

    return run_worker


@pytest.fixture
def fake_cuda(chk, monkeypatch):
    """CUDA looks available, but the driver must never open a context itself.

    A context in the driver process would take memory from the GPU the workers
    measure, so anything that initializes CUDA fails the test.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError("the driver process must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("get_device_name", "get_device_capability", "mem_get_info", "init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def test_the_driver_orchestrates_workers_and_prints_the_verdict(
    chk, fake_cuda, monkeypatch, tmp_path, capsys
):
    calls: list[list[str]] = []
    monkeypatch.setattr(chk, "run_worker", fake_worker_factory(chk, calls))
    out = tmp_path / "report.json"
    code = chk.main(
        [
            "--config", str(tmp_path / "absent.json"), "--max-len", "2048", "8192",
            "--wscap-mb", "256", "--free-mib", "16384", "512", "--out", str(out),
        ]
    )  # fmt: skip
    assert code == 0
    text = capsys.readouterr().out
    assert "device {'name': 'Fake V100', 'capability': [7, 0]}" in text
    assert "MEMORY_DEPENDENT: YES" in text  # the fake default setting differs at 8192
    assert "CUDNN_USED: NO" in text
    assert "DETERMINISTIC_FLAG_FIXES: YES" in text
    assert "WSCAP_FIXES (wscap256): YES" in text
    assert "ALGO: default 2048 whole: cudnn legacy_algos=" in text
    assert "NOTE: PyTorch does not route this conv to cuDNN" in text

    modes = [c[c.index("--worker") + 1] for c in calls]
    # 4 settings x 2 max_len scenario workers, then algo workers for `default` only.
    assert modes.count("scenarios") == 8
    assert modes.count("algo") == 2 * 3  # 2 max_len x 3 scenarios, default only
    assert all(
        c[c.index("--setting") + 1] == "default"
        for c in calls
        if c[c.index("--worker") + 1] == "algo"
    )

    report = json.loads(out.read_text())
    assert (
        report["geometry"]["channels"] == 10240
        and report["geometry"]["conv_state_len"] == 9
    )
    assert report["verdict"]["MEMORY_DEPENDENT"] == "YES"
    assert report["errors"] == []
    assert report["device"] == {"name": "Fake V100", "capability": [7, 0]}
    parsed = report["algo"]["default"]["2048"]["whole"]["cudnn_log"]
    assert parsed["legacy_algos"] == [
        "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"
    ]


def test_the_driver_exits_1_when_a_worker_fails_but_keeps_going(
    chk, fake_cuda, monkeypatch, tmp_path, capsys
):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        chk, "run_worker", fake_worker_factory(chk, calls, fail="benchmark")
    )
    out = tmp_path / "report.json"
    code = chk.main(
        ["--config", str(tmp_path / "absent.json"), "--max-len", "2048",
         "--algo-settings", "none", "--out", str(out)]
    )  # fmt: skip
    assert code == 1
    assert "ERROR: benchmark/2048" in capsys.readouterr().out
    report = json.loads(out.read_text())
    assert len(report["errors"]) == 1
    assert "benchmark" not in report["results"]["benchmark"]  # recorded as empty
    assert report["verdict"]["MEMORY_DEPENDENT"] == "NO"


def test_the_driver_exits_2_without_cuda(chk, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert chk.main([]) == 2
    assert "needs a CUDA device" in capsys.readouterr().out


def test_a_worker_invocation_prints_one_json_line(chk, cpu_worker, capsys):
    code = chk.main(
        [
            "--worker", "scenarios", "--setting", "default", "--worker-max-len", "16",
            "--channels", "4", "--kernel-size", "3", "--dilation", "2",
            "--free-mib", "16",
        ]
    )  # fmt: skip
    assert code == 0
    lines = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if ln.startswith("RESULT_JSON:")
    ]
    assert len(lines) == 1
    result = json.loads(lines[0][len("RESULT_JSON:") :])
    assert result["setting"] == "default" and result["max_len"] == 16
    assert [s["name"] for s in result["scenarios"]] == ["whole", "free16MiB"]
