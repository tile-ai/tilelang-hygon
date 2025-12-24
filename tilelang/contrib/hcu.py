"""Utility for HCU backend"""

def get_hcu_compile_flags(arch: str):
    if arch in ["gfx936", "gfx938"]:
        return [
            #"-mllvm=-support-768-vgprs=true",
            "-mllvm=-enable-latency-hack=true",
            "-mllvm=-mmac-latency=5",
            "-mllvm=-ds-load-store-latency=6",
        ]
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
        return []
