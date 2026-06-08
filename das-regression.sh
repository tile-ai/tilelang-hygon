#!/usr/bin/env bash

set -euo pipefail

export HIPBLASLT_ALLOW_TF32=1

REPO_DIR="${REPO_DIR:-/workspace/TileKernels}"
TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-./cache_regression}"
TEST_RESULTS_DIR="${TEST_RESULTS_DIR:-${REPO_DIR}/test-results}"

NODEIDS=(
  # mhc_pre_big_fuse
  "tests/mhc/test_pre_big_fuse.py::test_correctness[4-4096-1]"
  "tests/mhc/test_pre_big_fuse.py::test_correctness[4-4096-577]"
  "tests/mhc/test_pre_big_fuse.py::test_correctness[4-4096-21111]"
  "tests/mhc/test_pre_big_fuse.py::test_correctness[4-7168-1]"
  "tests/mhc/test_pre_big_fuse.py::test_correctness[4-7168-577]"
  "tests/mhc/test_pre_big_fuse.py::test_correctness[4-7168-9217]"
  "tests/mhc/test_pre_big_fuse.py::test_correctness[4-4096-2722]"
  "tests/mhc/test_pre_big_fuse.py::test_correctness[4-7168-21111]"
  "tests/mhc/test_pre_big_fuse.py::test_correctness[4-1280-21111]"
  "tests/mhc/test_pre_big_fuse.py::test_correctness[4-4096-9217]"
  "tests/mhc/test_pre_big_fuse.py::test_correctness[4-7168-2722]"

  # mhc_post
  "tests/mhc/test_post.py::test_mhc_post_comprehensive[4096-1-1]"
  "tests/mhc/test_post.py::test_mhc_post_comprehensive[4096-32-1]"
  "tests/mhc/test_post.py::test_mhc_post_comprehensive[4096-4096-1]"
  "tests/mhc/test_post.py::test_mhc_post_comprehensive[7168-1-1]"
  "tests/mhc/test_post.py::test_mhc_post_comprehensive[7168-32-1]"
  "tests/mhc/test_post.py::test_mhc_post_comprehensive[7168-4096-1]"

  # Pre-path unit coverage
  "tests/mhc/test_pre_apply_mix.py::test_pre_apply_mix_comprehensive[1280-1024-1]"
  "tests/mhc/test_pre_apply_mix.py::test_pre_apply_mix_comprehensive[7680-4096-1]"
  "tests/mhc/test_pre_split_mixes.py::test_pre_split_mixes_comprehensive[4-1024-1]"
  "tests/mhc/test_pre_split_mixes.py::test_pre_split_mixes_comprehensive[4-4096-1]"
  "tests/mhc/test_norm_fn.py::test_correctness[False-1280-4096]"
  "tests/mhc/test_norm_fn.py::test_correctness[True-1280-4096]"
  "tests/mhc/test_norm_fn.py::test_correctness[False-7168-8192]"
  "tests/mhc/test_norm_fn.py::test_correctness[True-7168-8192]"
  "tests/mhc/test_norm_fn.py::test_split_k_correctness[1280-13]"
  "tests/mhc/test_norm_fn.py::test_split_k_correctness[4096-512]"

  # Remaining MHC correctness coverage
  "tests/mhc/test_sinkhorn.py::test_sinkhorn_comprehensive[4-1-1]"
  "tests/mhc/test_sinkhorn.py::test_sinkhorn_comprehensive[4-4096-2]"
  "tests/mhc/test_expand.py::test_expand_comprehensive[1280-2-1024-1]"
  "tests/mhc/test_expand.py::test_expand_comprehensive[7168-8-4096-2]"
  "tests/mhc/test_head_compute_mix.py::test_head_compute_mix_comprehensive[4-1024-1]"
  "tests/mhc/test_hc_split_sinkhorn.py::test_hc_split_sinkhorn_comprehensive[4-1-1]"
  "tests/mhc/test_hc_split_sinkhorn.py::test_hc_split_sinkhorn_comprehensive[4-257-2]"
  "tests/mhc/test_multilayer_recompute.py::test_mhc_multilayer_recompute_correctness[1-1-2560]"

  # MOE correctness coverage
  "tests/moe/test_aux_fi.py::test_aux_fi[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8-num_aux_topk=1]"
  "tests/moe/test_expand_to_fused.py::test_expand_to_fused[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8-hidden=2048]"
  "tests/moe/test_expand_to_fused.py::test_expand_to_fused_with_sf[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8-hidden=2048-num_per_channels=128-use_tma_aligned_col_major_sf=True-round_sf=True-use_packed_ue8m0=True]"
  "tests/moe/test_get_fused_mapping.py::test_get_fused_mapping[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8-alignment=128]"
  "tests/moe/test_group_count.py::test_group_count[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8]"
  "tests/moe/test_inplace_unique_group_indices.py::test_inplace_unique_group_indices[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8-num_groups=8]"
  "tests/moe/test_mask_indices_by_tp.py::test_mask_indices_by_tp[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8-num_tp_ranks=8]"
  "tests/moe/test_normalize_weight.py::test_normalize_weight[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8]"
  # "tests/moe/test_reduce_fused.py::test_reduce_fused[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8-hidden=2048-with_weights=True-in_dtype=fp32-out_dtype=fp32-with_sf=False]"
  "tests/moe/test_top2_sum_gate.py::test_top2_sum_gate[num_groups=0-num_topk_groups=0-num_routed_experts=72-num_shared_experts=1-num_topk=6]"
  "tests/moe/test_topk_gate.py::test_topk_gate[num_tokens=4001-num_experts=72-num_topk=6]"
  "tests/moe/test_topk_sum_and_topk_idx.py::test_topk_sum_and_topk_group_idx[num_tokens=4001-num_experts=72-num_groups=4-num_group_sum_topk=1-num_topk_groups=2]"

  # Engram correctness coverage
  "tests/engram/test_engram_fused_weight.py::test_engram_fused_weight[hc=4-hidden=2048]"
  "tests/engram/test_engram_gate_bwd.py::test_engram_gate_bwd[num_tokens=4001-hc=4-hidden=2048]"
  "tests/engram/test_engram_gate_fwd.py::test_engram_gate_fwd_alg_ref_matches_tilelang_accum_order[num_tokens=4001-hc=4-hidden=2048]"
  "tests/engram/test_engram_gate_fwd.py::test_engram_gate_fwd[num_tokens=4001-hc=4-hidden=2048]"
  "tests/engram/test_engram_grad_w_reduce.py::test_engram_grad_w_reduce[hidden=2048]"
  "tests/engram/test_engram_hash.py::test_engram_hash[num_tokens=4001]"

  # Quant correctness coverage
  "tests/quant/test_cast_back.py::test_cast_back_per_token[num_tokens=4001-hidden=7168-fmt=e4m3-use_tma_aligned_col_major_sf=False-round_sf=True-use_packed_ue8m0=False-num_per_channels=128-out_dtype=bf16]"
  "tests/quant/test_cast_back.py::test_cast_back[num_tokens=4001-hidden=7168-round_sf=True-fmt=e4m3-out_dtype=bf16-num_per_tokens=128-num_per_channels=128]"
  "tests/quant/test_cast_back_e5m6.py::test_cast_back_e5m6[num_tokens=4001-hidden=7168-num_per_channels=7168-use_tma_aligned_col_major_sf=False-round_sf=True-use_packed_ue8m0=False-out_dtype=bf16]"
  "tests/quant/test_per_block_cast.py::test_per_block_cast[num_tokens=4001-hidden=2048-in_dtype=bf16-fmt=e4m3-use_tma_aligned_col_major_sf=True-round_sf=True-use_packed_ue8m0=True-block_size=128x128]"
  "tests/quant/test_per_block_cast_lossless.py::test_per_block_cast_lossless[num_tokens=4001-hidden=2048-in_use_tma_aligned_col_major_sf=True-in_round_sf=True-in_use_packed_ue8m0=True-out_use_tma_aligned_col_major_sf=True-out_round_sf=True-out_use_packed_ue8m0=True-out_sf_block=128x128-in_sf_block=1x32-in_fmt=e2m1]"
  "tests/quant/test_per_channel_cast.py::test_per_channel_cast[num_per_tokens=128-num_tokens=4096-hidden=7168-round_sf=True-dtype=bf16]"
  "tests/quant/test_per_channel_cast_and_transpose.py::test_per_channel_cast_and_transpose[num_tokens=4096-hidden=7168-round_sf=True-dtype=bf16-num_per_channels=128]"
  "tests/quant/test_per_channel_cast_fused.py::test_per_channel_cast_fused[num_send_tokens=4096-num_topk=0-num_experts=0-num_ep_ranks=0-hidden=7168-num_per_tokens=128-num_per_channels=128-is_fused_cast_back=True-round_sf=True]"
  "tests/quant/test_per_token_cast.py::test_per_token_cast[num_tokens=4001-hidden=7168-use_tma_aligned_col_major_sf=True-round_sf=True-use_packed_ue8m0=True-in_dtype=bf16-num_per_channels=128-x_block_size=None-fmt=e4m3]"
  "tests/quant/test_per_token_cast_to_e5m6.py::test_per_token_cast_to_e5m6[num_tokens=4001-hidden=7168-use_tma_aligned_col_major_sf=True-round_sf=True-use_packed_ue8m0=True-in_dtype=bf16]"
  "tests/quant/test_swiglu_backward_and_per_token_cast.py::test_swiglu_backward_and_per_token_cast[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8-hidden=7168-num_per_channels=128-round_sf=True-swiglu_clamp_value=None]"
  # "tests/quant/test_swiglu_forward_and_per_channel_cast_and_transpose.py::test_swiglu_forward_and_per_channel_cast_and_transpose[num_tokens=4096-hidden=7168-num_per_tokens=128-without_transpose=False-round_sf=True-swiglu_clamp_value=None]"
  "tests/quant/test_swiglu_forward_and_per_token_cast.py::test_swiglu_forward_and_per_token_cast[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8-hidden=3584-enable_pos_to_expert=True-with_weights=False-num_per_channels=128-use_tma_aligned_col_major_sf=True-round_sf=True-use_packed_ue8m0=True-swiglu_clamp_value=None]"

  # Transpose correctness coverage
  "tests/transpose/test_transpose.py::test_transpose[num_tokens=4032-hidden=7168-dtype=bf16]"
  "tests/transpose/test_transpose.py::test_batched_transpose[num_tokens=4032-hidden=2048-num_experts=8-dtype=bf16]"
)

echo "Running ${#NODEIDS[@]} selected pytest cases in ${REPO_DIR}"

cd "${REPO_DIR}"

ALLURE_RESULTS_DIR="${ALLURE_RESULTS_DIR:-${TEST_RESULTS_DIR}/allure-results}"
ALLURE_REPORT_DIR="${ALLURE_REPORT_DIR:-${TEST_RESULTS_DIR}/allure-report}"
GENERATE_ALLURE_REPORT="${GENERATE_ALLURE_REPORT:-0}"

mkdir -p "${ALLURE_RESULTS_DIR}" "${TEST_RESULTS_DIR}"

extra_pytest_args=("$@")
allure_args=()
pytest_help_file="$(mktemp)"
trap 'rm -f "${pytest_help_file}"' EXIT

python -m pytest --help >"${pytest_help_file}" 2>/dev/null || true
if grep -q -- "--alluredir" "${pytest_help_file}"; then
  allure_args=(--alluredir "${ALLURE_RESULTS_DIR}")
else
  echo "allure-pytest plugin not available, skipping --alluredir" >&2
fi

total=${#NODEIDS[@]}
failed=0
passed_count=0
failed_count=0

for i in "${!NODEIDS[@]}"; do
  idx=$((i + 1))
  nodeid="${NODEIDS[$i]}"
  echo "[${idx}/${total}] ${nodeid}"
  if TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR}" python -m pytest -p no:warnings "${extra_pytest_args[@]}" "${allure_args[@]}" "${nodeid}"; then
    passed_count=$((passed_count + 1))
  else
    failed=1
    failed_count=$((failed_count + 1))
  fi
done

if [[ "${GENERATE_ALLURE_REPORT}" == "1" ]]; then
  if command -v allure >/dev/null 2>&1; then
    rm -rf "${ALLURE_REPORT_DIR}"
    allure generate "${ALLURE_RESULTS_DIR}" -o "${ALLURE_REPORT_DIR}"
    echo "Allure report: ${ALLURE_REPORT_DIR}"
  else
    echo "allure command not found, skipped report generation" >&2
  fi
fi

echo
echo "Pytest summary: total=${total}, passed=${passed_count}, failed=${failed_count}"

exit "${failed}"
