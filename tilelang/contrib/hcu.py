"""Utility for HCU backend"""

def get_hcu_compile_flags(arch: str):
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
        return []
