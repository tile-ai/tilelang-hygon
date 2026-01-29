# ruff: noqa
import torch
import tilelang
import triton
from tilelang import language as T
# from utils import assert_tensors_similar
from functools import partial
from perf.utils.device import get_free_devices
from tilelang.profiler import do_bench_cudagraph

def test_sparse_mla_fwd(B=32,
                        S=1,
                        SKV=8192,
                        H=128,
                        HKV=1,
                        DQK=576,
                        DV=512,
                        topk=2048,
                        dtype=torch.bfloat16,
                        check_correctness=True,
                        block_I=None,
                        num_stages=None,
                        threads=None,
                        q_start_s_index=4096,
                        kv_stride=1,
                        num_split=None,
                        arch="nmz"):
    torch.random.manual_seed(0)

    if arch == "bmz":
        from perf.sparse_mla.sparse_mla_fwd import sparse_mla_fwd_interface, ref_sparse_mla_fwd_interface
    elif arch == "nmz":
        from perf.sparse_mla.sparse_mla_fwd_nmz import sparse_mla_fwd_interface, ref_sparse_mla_fwd_interface
    else:
        from perf.sparse_mla.sparse_mla_fwd import sparse_mla_fwd_interface, ref_sparse_mla_fwd_interface
    # q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    # kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda")
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda")

    indices = torch.full((B, S, HKV, topk), -1, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(min(max(1, ((t + q_start_s_index) // kv_stride)), SKV))[:topk]
                indices[b, t, h, :len(i_i)] = i_i

    # Allocate required tensors
    tl_out = torch.empty((B, S, H, DV), dtype=q.dtype, device="cuda")
    tl_out_bench = torch.empty((B, S, H, DV), dtype=q.dtype, device="cuda")

    if dtype == torch.bfloat16:
        dtype_str = "bfloat16"
    else:
        dtype_str = "float16"

    tilelang_kernel, num_split, num_split_tail, batch_head = sparse_mla_fwd_interface(
            q, kv, indices, dtype=dtype_str,
            kv_stride=kv_stride, block_I=block_I, num_stages=num_stages, threads=threads,
            num_split=num_split)

    if num_split_tail > 0:
        assert B > batch_head, "B must be greater than batch_head"
        glse = torch.empty((batch_head, S, H, num_split), dtype=q.dtype, device="cuda")
        output_partial = torch.empty((batch_head, S, H, num_split, DV), dtype=q.dtype, device="cuda")
        glse_tail = torch.empty((B - batch_head, S, H, num_split_tail), dtype=q.dtype, device="cuda")
        output_partial_tail = torch.empty((B - batch_head, S, H, num_split_tail, DV), dtype=q.dtype, device="cuda")
        tilelang_kernel(
            q, kv, indices, glse, output_partial, glse_tail, output_partial_tail, tl_out)

        def fn():
            return tilelang_kernel(q, kv, indices, glse, output_partial, glse_tail, output_partial_tail, tl_out_bench)
    else:
        glse = torch.empty((B, S, H, num_split), dtype=q.dtype, device="cuda")
        output_partial = torch.empty((B, S, H, num_split, DV), dtype=q.dtype, device="cuda")
        tilelang_kernel(
            q, kv, indices, glse, output_partial, tl_out)

        def fn():
            return tilelang_kernel(q, kv, indices, glse, output_partial, tl_out_bench)

    def ref_fn():
        return ref_sparse_mla_fwd_interface(q, kv, indices, q_start_s_index=q_start_s_index, kv_stride=kv_stride)

    ms = do_bench_cudagraph(fn)
    print(f"{B=} {S=} {SKV=} {H=} {HKV=} {q_start_s_index=} {topk=} {num_split=} {num_split_tail=} {batch_head=}")
    print(f"Average time: {ms:.3f} ms")
    print("fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e9, "GB/s")
    print("fwd tflops = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)

    if check_correctness:
        # otherwise may cause out of memory
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices, dtype, q_start_s_index=q_start_s_index, kv_stride=kv_stride)
        # assert_tensors_similar(tl_out, ref_out, eps=1e-2, name="out")
        atol=1e-2
        if dtype_str == "bfloat16":
            atol = 2e-2
        torch.testing.assert_close(tl_out, ref_out, atol=atol, rtol=1e-2)
        print("assert_tensors_similar passed")


if __name__ == "__main__":
    device_id = 0
    if device_id == -1:
        free_hcus = get_free_devices()
        if len(free_hcus) == 0:
            raise RuntimeError("No free HCU devices found")
        device_id = free_hcus[len(free_hcus) - 1]
    torch.cuda.set_device(device_id)
    
    for B in [1, 2, 4, 8, 16, 32, 64, 128]:
        test_sparse_mla_fwd(
            B=B,
            S=1,
            SKV=8192,
            H=16,
            HKV=1,
            DQK=576,
            DV=512,
            topk=2048,
            dtype=torch.float16,
            check_correctness=True,
            q_start_s_index=2048)
            # block_I=32,
            # num_stages=0,
            # threads=256,
            # num_split=8)
