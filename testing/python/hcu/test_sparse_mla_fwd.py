# ruff: noqa
import torch
import tilelang
import triton
from tilelang import language as T
# from utils import assert_tensors_similar
from functools import partial
from perf.sparse_mla.sparse_mla_fwd import sparse_mla_fwd_interface, ref_sparse_mla_fwd_interface
from perf.utils.device import get_free_devices

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
                        num_split=None):
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda")
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda")

    indices = torch.full((B, S, HKV, topk), -1, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(min(max(1, ((t + q_start_s_index) // kv_stride)), SKV))[:topk]
                indices[b, t, h, :len(i_i)] = i_i

    # Allocate required tensors
    q_start_s_index_t = torch.tensor([q_start_s_index], dtype=torch.int32, device="cuda")
    glse = torch.empty((B, S, H, num_split), dtype=q.dtype, device="cuda")
    output_partial = torch.empty((B, S, H, num_split, DV), dtype=q.dtype, device="cuda")
    tl_out = torch.empty((B, S, H, DV), dtype=q.dtype, device="cuda")

    if dtype == torch.bfloat16:
        dtype_str = "bfloat16"
    else:
        dtype_str = "float16"

    tilelang_kernel, num_split_ = sparse_mla_fwd_interface(
            q, kv, indices, dtype=dtype_str,
            kv_stride=kv_stride, block_I=block_I, num_stages=num_stages, threads=threads,
            num_split=num_split)

    glse = torch.empty((B, S, H, num_split_), dtype=q.dtype, device="cuda")
    output_partial = torch.empty((B, S, H, num_split_, DV), dtype=q.dtype, device="cuda")
    tilelang_kernel(
        q, kv, indices, q_start_s_index_t, glse, output_partial, tl_out)

    print(f"{B=} {S=} {SKV=} {H=} {HKV=} {q_start_s_index=} {num_split=}")

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
    device = -1
    free_hcus = get_free_devices()
    if len(free_hcus) == 0:
        raise RuntimeError("No free HCU devices found")
    if device == -1:
        device_id = free_hcus[len(free_hcus) - 1]
    else:
        device_id = device
    torch.cuda.set_device(device_id)

    for num_split in [1, 2, 4, 8, 16]:
        test_sparse_mla_fwd(
            B=128,
            S=1,
            SKV=8192,
            H=16,
            HKV=1,
            DQK=576,
            DV=512,
            topk=2048,
            dtype=torch.float16,
            check_correctness=True,
            block_I=32,
            num_stages=1,
            threads=256,
            q_start_s_index=2048,
            num_split=num_split)

        test_sparse_mla_fwd(
            B=128,
            S=1,
            SKV=8192,
            H=16,
            HKV=1,
            DQK=576,
            DV=512,
            topk=2048,
            dtype=torch.float16,
            check_correctness=True,
            block_I=32,
            num_stages=1,
            threads=256,
            q_start_s_index=2048,
            num_split=num_split)

        test_sparse_mla_fwd(
            B=12,
            S=7,
            SKV=2048,
            H=16,
            HKV=1,
            DQK=576,
            DV=512,
            topk=2048,
            dtype=torch.float16,
            check_correctness=True,
            block_I=32,
            num_stages=1,
            threads=256,
            q_start_s_index=177,
            num_split=num_split)

        test_sparse_mla_fwd(
            B=4,
            S=71,
            SKV=1024,
            H=16,
            HKV=1,
            DQK=576,
            DV=512,
            topk=2048,
            dtype=torch.float16,
            check_correctness=True,
            block_I=32,
            num_stages=1,
            threads=256,
            q_start_s_index=0,
            num_split=num_split)
        
        test_sparse_mla_fwd(
            B=16,
            S=1,
            SKV=996,
            H=128,
            HKV=1,
            DQK=576,
            DV=512,
            topk=2048,
            dtype=torch.float16,
            check_correctness=True,
            block_I=32,
            num_stages=1,
            threads=256,
            q_start_s_index=2048,
            num_split=num_split)
