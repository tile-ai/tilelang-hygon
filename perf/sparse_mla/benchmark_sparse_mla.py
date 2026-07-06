# ruff: noqa
import torch
from tilelang.profiler import do_bench_cudagraph
from perf.sparse_mla.sparse_mla_fwd import tilelang_sparse_fwd, ref_sparse_mla_fwd_interface


def test_sparse_mla_fwd(
    B=1,
    S=128,
    SKV=8192,
    H=16,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    check_correctness=True,
    q_start_s_index=4096,
    kv_stride=1,
    sm_scale=None,
):
    torch.random.manual_seed(0)

    q = torch.randn((S, H, DQK), dtype=dtype, device="cuda")
    kv = torch.randn((SKV, HKV, DQK), dtype=dtype, device="cuda")

    indices = torch.full((S, HKV, topk), -1, dtype=torch.int32, device="cuda")
    for t in range(S):
        for h in range(HKV):
            i_i = torch.randperm(min(max(1, ((t + q_start_s_index) // kv_stride)), SKV))[:topk]
            indices[t, h, : len(i_i)] = i_i

    if dtype == torch.bfloat16:
        dtype_str = "bfloat16"
    else:
        dtype_str = "float16"

    if sm_scale is None:
        sm_scale = (1.0 / (DQK + (DQK - DV))) ** 0.5

    tl_out = tilelang_sparse_fwd(q, kv, indices, sm_scale, d_v=DV)

    def fn():
        return tilelang_sparse_fwd(q, kv, indices, sm_scale, d_v=DV)

    ms = do_bench_cudagraph(fn)
    print(f"{B=} {S=} {SKV=} {H=} {HKV=} {q_start_s_index=} {topk=} {dtype=}")
    print(f"Average time: {ms:.3f} ms")
    print(f"fwd io bandwidth = {(B * S * DQK * topk * 2) / (ms * 1e-3) / 1e9} GB/s")
    print(f"fwd tflops = {(B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12}")

    if check_correctness:
        ref_out = ref_sparse_mla_fwd_interface(
            q, kv, indices, dtype, q_start_s_index=q_start_s_index, kv_stride=kv_stride, sm_scale=sm_scale
        )
        atol = 1e-2 if dtype_str == "bfloat16" else 1e-2
        torch.testing.assert_close(tl_out, ref_out, atol=atol, rtol=1e-2)
        print("assert_tensors_similar passed")


if __name__ == "__main__":
    device_id = 0
    torch.cuda.set_device(device_id)
    for s in [1, 2, 4, 8, 16, 32, 64, 72, 96, 128, 144]:
        test_sparse_mla_fwd(
            B=1, S=s, SKV=8192, H=16, HKV=1, DQK=576, DV=512, topk=2048, dtype=torch.bfloat16, check_correctness=True, q_start_s_index=2048
        )
