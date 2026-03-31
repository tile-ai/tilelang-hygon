"""Utility for HCU backend"""

import os


def get_hcu_compile_flags(arch: str):
    # DTK toolchain (e.g. ROCM_PATH=/opt/dtk/...) uses its own defaults; do not inject LLVM hacks.
    rocm_path = os.environ.get("ROCM_PATH", "")
    if "dtk" in rocm_path.lower() or os.path.isdir("/opt/dtk"):
        return []
    if arch in ["gfx928", "gfx936", "gfx938", "gfx92a", "gfx946"]:
        flags = [
            #"-mllvm=-support-768-vgprs=true",
            "-mllvm=-enable-latency-hack=true",
            "-mllvm=-mmac-latency=5",
            "-mllvm=-ds-load-store-latency=6",
        ]
        if arch in ["gfx938", "gfx92a", "gfx946"]:
            flags.append("-mllvm=-hcu-update-wait-by-reverse-search=true")
        return flags
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
