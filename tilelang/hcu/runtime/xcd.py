# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Literal


_XCD_CONFIG_ABI_VERSION = 1
_XCD_MODE_BLOCK = 0
_XCD_MODE_LINEAR = 1
_XCD_MODE_FIXED = 2


class _LinearConfig(ctypes.Structure):
    _fields_ = [("chunk_size", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]


class _BlockConfig(ctypes.Structure):
    _fields_ = [("block_x", ctypes.c_uint32), ("block_y", ctypes.c_uint32)]


class _FixedConfig(ctypes.Structure):
    _fields_ = [("reserved0", ctypes.c_uint32), ("reserved1", ctypes.c_uint32)]


class _DispatchConfig(ctypes.Union):
    _fields_ = [("linear", _LinearConfig), ("block", _BlockConfig), ("fixed", _FixedConfig)]


class _CXCDLaunchConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("die_mask", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("dispatch", _DispatchConfig),
        ("reserved", ctypes.c_uint64 * 4),
    ]


def _require_plain_int(name: str, value: int, *, minimum: int, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be >= {minimum}{upper}, got {value}")


@dataclass(frozen=True)
class XCDLaunchConfig:
    """Per-call physical XCD dispatch configuration for the HCU Cython backend."""

    mode: Literal["linear", "block", "fixed"]
    die_mask: int = 0
    chunk_size: int = 1
    block_x: int = 1
    block_y: int = 1

    def __post_init__(self) -> None:
        if self.mode not in ("linear", "block", "fixed"):
            raise ValueError(f"unsupported XCD dispatch mode: {self.mode!r}")
        _require_plain_int("die_mask", self.die_mask, minimum=0, maximum=0xF)
        _require_plain_int("chunk_size", self.chunk_size, minimum=1, maximum=(1 << 31) - 1)
        _require_plain_int("block_x", self.block_x, minimum=1, maximum=2047)
        _require_plain_int("block_y", self.block_y, minimum=1, maximum=2047)
        if self.mode == "linear" and (self.block_x != 1 or self.block_y != 1):
            raise ValueError("linear mode does not accept block_x or block_y")
        if self.mode == "block" and self.chunk_size != 1:
            raise ValueError("block mode does not accept chunk_size")
        if self.mode == "fixed" and (self.die_mask != 0 or self.chunk_size != 1 or self.block_x != 1 or self.block_y != 1):
            raise ValueError("fixed mode does not accept die_mask, chunk_size, block_x, or block_y")

    @classmethod
    def linear(cls, *, die_mask: int = 0, chunk_size: int = 1) -> XCDLaunchConfig:
        return cls("linear", die_mask=die_mask, chunk_size=chunk_size)

    @classmethod
    def block(cls, *, die_mask: int = 0, block_x: int = 1, block_y: int = 1) -> XCDLaunchConfig:
        return cls("block", die_mask=die_mask, block_x=block_x, block_y=block_y)

    @classmethod
    def fixed(cls) -> XCDLaunchConfig:
        return cls("fixed")

    def _as_ctypes(self) -> _CXCDLaunchConfig:
        config = _CXCDLaunchConfig()
        config.struct_size = ctypes.sizeof(_CXCDLaunchConfig)
        config.abi_version = _XCD_CONFIG_ABI_VERSION
        config.die_mask = self.die_mask
        if self.mode == "linear":
            config.mode = _XCD_MODE_LINEAR
            config.dispatch.linear.chunk_size = self.chunk_size
        elif self.mode == "block":
            config.mode = _XCD_MODE_BLOCK
            config.dispatch.block.block_x = self.block_x
            config.dispatch.block.block_y = self.block_y
        else:
            config.mode = _XCD_MODE_FIXED
        return config
