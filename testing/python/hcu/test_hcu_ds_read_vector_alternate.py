"""Matrix-read output order is an explicit instruction contract."""

import pytest
import torch
import tilelang as tl
import tilelang.hcu.language as T
from tilelang import tvm
import tilelang.testing
from hcu_test_utils import current_hcu_arch_string


@pytest.mark.parametrize("alt", [1, 2])
@pytest.mark.parametrize("direct_annotation", [False, True])
@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_ds_read_vector_register_order(alt, direct_annotation):
    source = r"""
#include <hip/hip_runtime.h>
#include <tl_templates/hcu/common.h>
__device__ __forceinline__ void probe_alt(bfloat16_t* dst, bfloat16_t* src) {
  auto x = __builtin_hcu_ds_read_m32x16_bf16_alt(
      (__attribute__((address_space(3))) short*)src);
  *reinterpret_cast<uint4*>(dst) = __builtin_bit_cast(uint4, x);
}
"""

    @T.prim_func
    def main(
        A: T.Tensor((16, 32), "bfloat16"),
        X: T.Tensor((64, 8), "bfloat16"),
        Y: T.Tensor((64, 8), "bfloat16"),
    ):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            s = T.alloc_shared((16, 32), "bfloat16")
            x = T.alloc_local((8,), "bfloat16")
            y = T.alloc_local((8,), "bfloat16")
            T.import_source(source)
            T.copy(A, s)
            T.sync_threads()
            T.call_extern("handle", "probe_alt", x.data, T.address_of(s[tx // 4, (tx % 4) * 8]))
            if direct_annotation:
                T.evaluate(
                    T.call_intrin(
                        "handle",
                        tvm.ir.Op.get("tl.ds_read_vector"),
                        y.data,
                        0,
                        s[tx // 4, (tx % 4) * 8],
                        annotations={"tl.hcu_ds_read_alt": alt},
                    )
                )
            else:
                T.ds_read_vector(y.data, 0, s[tx // 4, (tx % 4) * 8], alt=alt)
            T.s_waitcnt(0, "lgkmcnt")
            for j in T.vectorized(8):
                X[tx, j] = x[j]
                Y[tx, j] = y[j]

    kernel = tl.compile(main, out_idx=[1, 2], target={"kind": "hcu", "mcpu": "gfx938"})
    a = torch.arange(512, device="cuda", dtype=torch.float32).reshape(16, 32).to(torch.bfloat16)
    x, y = kernel(a)
    if alt == 2:
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    else:
        assert not torch.equal(x, y)


@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_raw_alt_frontend_contract():
    dst = tvm.tirx.Var("dst", "handle")
    bf = tvm.tirx.decl_buffer((16, 32), "bfloat16", scope="shared")
    assert not T.ds_read_vector(dst, 0, bf[0, 0]).annotations
    alt = T.ds_read_vector(dst, 0, bf[0, 0], alt=2).annotations["tl.hcu_ds_read_alt"]
    assert int(alt) == 2
    assert alt.dtype == "int32"
    for invalid_alt in (0, 3, 4, True):
        with pytest.raises(ValueError, match="alt must be 1 or 2"):
            T.ds_read_vector(dst, 0, bf[0, 0], alt=invalid_alt)
    global_buffer = tvm.tirx.decl_buffer((16, 32), "bfloat16")
    with pytest.raises(TypeError, match="shared BufferLoad"):
        T.ds_read_vector(dst, 0, global_buffer[0, 0])
    with pytest.raises(TypeError, match="shared BufferLoad"):
        T.ds_read_vector(dst, 0, bf.data)


@pytest.mark.parametrize("alt,dtype", [(0, "bfloat16"), (3, "bfloat16"), (4, "bfloat16"), (True, "bfloat16"), (2, "float16")])
@tilelang.testing.requires_rocm
@pytest.mark.skipif(current_hcu_arch_string() != "gfx938", reason="requires gfx938 HCU")
def test_invalid_ds_read_alt_annotation(alt, dtype):
    @T.prim_func
    def main():
        with T.Kernel(1, threads=64):
            s = T.alloc_shared((16, 32), dtype)
            r = T.alloc_local((8,), dtype)
            T.clear(s)
            T.sync_threads()
            tx = T.get_thread_binding()
            T.evaluate(
                T.call_intrin(
                    "handle",
                    tvm.ir.Op.get("tl.ds_read_vector"),
                    r.data,
                    0,
                    s[tx // 4, (tx % 4) * 8],
                    annotations={"tl.hcu_ds_read_alt": alt},
                )
            )

    with pytest.raises(Exception, match="Alt must be a compile-time integer interleave|unsupported Alt=|requires a bfloat16 source"):
        tl.compile(main, target={"kind": "hcu", "mcpu": "gfx938"})
