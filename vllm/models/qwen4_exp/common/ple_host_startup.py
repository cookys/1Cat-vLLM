# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in host-memory pacing for PLE loading (stdlib only).

File-cache advice never changes tensor contents or unmaps private/COW pages.
The lock is host-local and does not depend on a distributed collective.
"""

import fcntl
import json
import logging
import math
import os
import re
import stat
import struct
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
_PREFIX = "VLLM_QWEN4EXP_PLE_PIN_"
_SHARD = re.compile(r"(?:^|\.)ngram_embedding\.shard_\d+\.weight$")
_MAX_HEADER_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class PinStartupOptions:
    serialize: bool = False
    drop_cache: bool = False
    lock_path: str = ""
    timeout_s: float = 600.0
    pause_s: float = 0.0

    @classmethod
    def from_env(cls):
        def flag(name):
            value = os.getenv(_PREFIX + name, "0")
            if value not in ("0", "1"):
                raise ValueError(f"{_PREFIX + name} must be 0 or 1")
            return value == "1"

        timeout = float(os.getenv(_PREFIX + "TIMEOUT_S", "600"))
        pause = float(os.getenv(_PREFIX + "PAUSE_MS", "0")) / 1000.0
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("PLE pin timeout must be finite and positive")
        if not math.isfinite(pause) or pause < 0:
            raise ValueError("PLE pin pause must be finite and non-negative")
        return cls(
            serialize=flag("SERIALIZE"),
            drop_cache=flag("DROP_CACHE"),
            lock_path=os.getenv(
                _PREFIX + "LOCK_PATH",
                f"{tempfile.gettempdir()}/vllm-ple-pin-{os.getuid()}.lock",
            ),
            timeout_s=timeout,
            pause_s=pause,
        )


@contextmanager
def host_pin_lock(path: str, timeout_s: float):
    """Serialize cooperating local processes; crash/exception releases the fd.

    Never unlink the lock file: waiters must continue to use the same inode.
    A timeout fails closed instead of starting another concurrent registration.
    """
    started = time.monotonic()
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or info.st_mode & 0o022
        ):
            raise RuntimeError("PLE pin lock must be a user-owned private regular file")
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for PLE pin lock: {path}"
                    ) from None
                time.sleep(min(0.05, remaining))
        yield time.monotonic() - started
    finally:
        os.close(fd)


def ple_checkpoint_ranges(model_path: str) -> dict[str, list[tuple[int, int]]]:
    """Read index/headers only; return exact PLE byte ranges, never weight data."""
    root = Path(model_path)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    by_file = defaultdict(list)
    for name, filename in index["weight_map"].items():
        if _SHARD.search(name):
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("PLE checkpoint filenames must be relative")
            by_file[str(root / relative)].append(name)
    ranges = {}
    for filename, names in by_file.items():
        with open(filename, "rb") as source:
            size = os.fstat(source.fileno()).st_size
            header_size = struct.unpack("<Q", source.read(8))[0]
            if not 0 < header_size <= min(_MAX_HEADER_BYTES, size - 8):
                raise ValueError(f"Invalid safetensors header size: {filename}")
            header = json.loads(source.read(header_size))
        file_ranges = []
        for name in names:
            start, end = header[name]["data_offsets"]
            if not (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= size - 8 - header_size
            ):
                raise ValueError(f"Invalid PLE tensor offsets: {name}")
            file_ranges.append((8 + header_size + start, end - start))
        ranges[filename] = file_ranges
    return ranges


def advise_file_ranges(filename: str, ranges: list[tuple[int, int]]) -> int:
    """Best-effort eviction of clean file cache, excluding boundary pages.

    The returned byte count is *advised*, not measured reclaimed memory.
    Mapped or dirty pages can remain resident. COW tensor contents are safe.
    """
    page = os.sysconf("SC_PAGE_SIZE")
    with open(filename, "rb") as source:
        size = os.fstat(source.fileno()).st_size
        aligned = []
        for offset, length in ranges:
            if offset < 0 or length < 0 or offset + length > size:
                raise ValueError("PLE cache advice exceeds file bounds")
            start = (offset + page - 1) // page * page
            end = (offset + length) // page * page
            if start < end:
                aligned.append((start, end))
        merged = []
        for start, end in sorted(aligned):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        for start, end in merged:
            os.posix_fadvise(
                source.fileno(), start, end - start, os.POSIX_FADV_DONTNEED
            )
    return sum(end - start for start, end in merged)


def drop_ple_checkpoint_cache(model_path: str | None) -> int:
    """Advice failure is an optimization miss, not a change in model loading."""
    if model_path is None:
        return 0
    try:
        ranges = ple_checkpoint_ranges(model_path)
        if not ranges:
            logger.warning("PLE_PIN no checkpoint shard ranges found: %s", model_path)
        return sum(advise_file_ranges(path, spans) for path, spans in ranges.items())
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        logger.warning("PLE_PIN cache advice unavailable: %s", error)
        return 0


def advise_loaded_shard(address: int, nbytes: int) -> int:
    """Advise only a copied contiguous file-backed tensor, without madvise."""
    if nbytes <= 0:
        return 0
    try:
        with open("/proc/self/maps") as mappings:
            for line in mappings:
                fields = line.rstrip().split(maxsplit=5)
                start, end = (int(x, 16) for x in fields[0].split("-"))
                if not start <= address < end:
                    continue
                if (
                    address + nbytes > end
                    or len(fields) != 6
                    or not fields[5].startswith("/")
                    or fields[5].endswith(" (deleted)")
                    or fields[5].startswith("/dev/")
                ):
                    return 0
                path = fields[5]
                info = os.stat(path)
                major, minor = (int(x, 16) for x in fields[3].split(":"))
                if info.st_ino != int(fields[4]) or info.st_dev != os.makedev(
                    major, minor
                ):
                    return 0
                offset = int(fields[2], 16) + address - start
                return advise_file_ranges(path, [(offset, nbytes)])
    except (OSError, ValueError) as error:
        logger.warning("PLE_PIN copied-shard cache advice unavailable: %s", error)
    return 0


@contextmanager
def pin_startup_guard(options: PinStartupOptions, model_path: str | None):
    """Hold the local lock through allocation, registration and optional pacing."""
    from contextlib import nullcontext

    started = time.monotonic()
    lock = (
        host_pin_lock(options.lock_path, options.timeout_s)
        if options.serialize
        else nullcontext(0.0)
    )
    with lock as waited:
        advice_started = time.monotonic()
        advised = drop_ple_checkpoint_cache(model_path) if options.drop_cache else 0
        advice_s = time.monotonic() - advice_started
        pin_started = time.monotonic()
        logger.info(
            "PLE_PIN begin pid=%d serialized=%s wait_s=%.6f advice_s=%.6f "
            "advised_bytes=%d",
            os.getpid(),
            options.serialize,
            waited,
            advice_s,
            advised,
        )
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            pin_s = time.monotonic() - pin_started
            # Retain the lock during the pause, so another rank cannot pin yet.
            if options.pause_s:
                time.sleep(options.pause_s)
            logger.info(
                "PLE_PIN end pid=%d success=%s pin_s=%.6f pause_s=%.6f total_s=%.6f",
                os.getpid(),
                succeeded,
                pin_s,
                options.pause_s,
                time.monotonic() - started,
            )
