# SPDX-License-Identifier: Apache-2.0
"""plan 075 S2 M1 (P2-6): ``--spec-hot-switch-elastic`` defaults off, parses,
needs the cumem allocator and does not change the compile hash; CPU-only."""

import json

import pytest

from vllm.config import CacheConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _parse(*argv):
    p = EngineArgs.add_cli_args(FlexibleArgumentParser())
    return EngineArgs.from_cli_args(p.parse_args(list(argv)))


def test_flag_default_off_and_parses():
    assert _parse().spec_hot_switch_elastic is False
    assert _parse("--spec-hot-switch-elastic").spec_hot_switch_elastic is True
    assert CacheConfig().spec_hot_switch_elastic is False


def test_flag_does_not_change_compile_hash():
    assert (
        CacheConfig(spec_hot_switch_elastic=True).compute_hash()
        == CacheConfig().compute_hash()
    )


@pytest.fixture
def local_model(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "ffn_dim": 128,
                "max_position_embeddings": 2048,
                "vocab_size": 128,
                "bos_token_id": 2,
                "eos_token_id": 2,
                "pad_token_id": 1,
            }
        )
    )
    return str(tmp_path)


def test_elastic_without_cumem_is_rejected(local_model):
    args = EngineArgs(
        model=local_model, skip_tokenizer_init=True, spec_hot_switch_elastic=True
    )
    with pytest.raises(ValueError, match="enable-cumem-allocator"):
        args.create_engine_config()


def test_elastic_with_cumem_reaches_cache_config(local_model):
    args = EngineArgs(
        model=local_model,
        skip_tokenizer_init=True,
        spec_hot_switch_elastic=True,
        enable_cumem_allocator=True,
    )
    cfg = args.create_engine_config()
    assert cfg.cache_config.spec_hot_switch_elastic is True
    off = EngineArgs(model=local_model, skip_tokenizer_init=True).create_engine_config()
    assert off.cache_config.spec_hot_switch_elastic is False
