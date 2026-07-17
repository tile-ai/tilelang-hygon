#!/usr/bin/env bash

set -euo pipefail

export HIPBLASLT_ALLOW_TF32=1

finish_pytest_status() {
  local status="$1"
  local phase="$2"
  if [[ "${status}" == "5" ]]; then
    echo "${phase}: pytest collected no tests (exit 5)" >&2
    return 5
  fi
  if [[ "${status}" != "0" ]]; then
    echo "${phase}: pytest failed with exit ${status}" >&2
    return "${status}"
  fi
  return 0
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/workspace/TileKernels}"
TILELANG_REPO_DIR="${TILELANG_REPO_DIR:-${SCRIPT_DIR}}"
TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-./cache_regression}"
TEST_RESULTS_DIR="${TEST_RESULTS_DIR:-${REPO_DIR}/test-results}"
PYTEST_ROOTDIR="${PYTEST_ROOTDIR:-$(dirname "${REPO_DIR}")}"
PRECOMPILE_TEST_KERNELS="${PRECOMPILE_TEST_KERNELS:-1}"
PRECOMPILE_JOBS="${PRECOMPILE_JOBS:-4}"

TILE_KERNEL_NODEIDS=(
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
  # "tests/quant/test_swiglu_backward_and_per_token_cast.py::test_swiglu_backward_and_per_token_cast[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8-hidden=7168-num_per_channels=128-round_sf=True-swiglu_clamp_value=None]"
  # "tests/quant/test_swiglu_forward_and_per_channel_cast_and_transpose.py::test_swiglu_forward_and_per_channel_cast_and_transpose[num_tokens=4096-hidden=7168-num_per_tokens=128-without_transpose=False-round_sf=True-swiglu_clamp_value=None]"
  "tests/quant/test_swiglu_forward_and_per_token_cast.py::test_swiglu_forward_and_per_token_cast[num_send_tokens=4001-num_topk=2-num_experts=9-num_ep_ranks=8-hidden=3584-enable_pos_to_expert=True-with_weights=False-num_per_channels=128-use_tma_aligned_col_major_sf=True-round_sf=True-use_packed_ue8m0=True-swiglu_clamp_value=None]"

  # Transpose correctness coverage
  "tests/transpose/test_transpose.py::test_transpose[num_tokens=4032-hidden=7168-dtype=bf16]"
  "tests/transpose/test_transpose.py::test_batched_transpose[num_tokens=4032-hidden=2048-num_experts=8-dtype=bf16]"

)

TILELANG_NODEIDS=(
  # HCU coverage from the tilelang repository itself. Keep these relative to
  # REPO_DIR so pytest progress output shows stable file paths after cd.
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/testing/python/hcu/test_tilelang_gemm_mls.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/testing/python/hcu/test_tilelang_gemm_mmac_intrinsic.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/testing/python/hcu/test_tilelang_gemm_mmac_preshuffle.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/testing/python/hcu/test_tilelang_test_gemm_hcu.py")"
)

TILELANG_EXAMPLES_NODEIDS=(
  # Example coverage from the tilelang repository itself. Keep these relative
  # to REPO_DIR for stable pytest progress output after cd.
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/analyze/test_example_analyze.py")"
  # "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/attention_sink/test_example_attention_sink.py")"
  # "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/blocksparse_attention/test_example_blocksparse_attention.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/blocksparse_gemm/test_example_blocksparse_gemm.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/cast/test_example_cast.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/convolution/test_example_convolution.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/deepseek_deepgemm/test_example_deepgemm_fp8_2xAcc.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/deepseek_mhc/test_example_mhc.py")"
  # "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/deepseek_mla/test_example_mla_decode.py")"
  # "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/deepseek_nsa/test_example_tilelang_nsa.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/deepseek_v32/test_tilelang_example_deepseek_v32.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/dequantize_gemm/test_example_dequantize_gemm.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/elementwise/test_example_elementwise.py")"
  # "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/flash_attention/test_example_flash_attention.py")"
  # "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/flash_decoding/test_example_flash_decoding.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/fusedmoe/test_example_fusedmoe.py")"
  # "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/gdn/test_example_gdn_compilation.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/gdn/test_utils.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/gemm/test_example_gemm.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/gemm_fp8/test_example_gemm_fp8.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/gemm_sp/test_example_gemm_sp.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/gemm_splitk/test_example_gemm_splitk.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/gemm_streamk/test_example_tilelang_gemm_streamk.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/gemv/test_example_gemv.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/grouped_gemm/test_example_grouped_gemm.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/kda/test_utils_kda.py")"
  # "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/linear_attention/test_linear_attn.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/minference/test_vs_sparse_attn.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/norm/test_rms_norm.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/seer_attention/test_block_sparse_attn_tilelang.py")"
  "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/topk/test_topk_tilelang.py")"
  # "$(realpath --relative-to="${REPO_DIR}" "${TILELANG_REPO_DIR}/examples/warp_specialize/test_example_warp_specialize.py")"
)

ALL_NODEIDS=("${TILE_KERNEL_NODEIDS[@]}" "${TILELANG_NODEIDS[@]}" "${TILELANG_EXAMPLES_NODEIDS[@]}")

dedupe_param_nodeids_by_func() {
  local input_name="$1"
  local output_name="$2"
  local -n input_ref="${input_name}"
  local -n output_ref="${output_name}"
  local nodeid key
  local -A seen=()

  output_ref=()
  for nodeid in "${input_ref[@]}"; do
    key="${nodeid%%[*}"
    if [[ -z "${seen[${key}]+x}" ]]; then
      seen["${key}"]=1
      output_ref+=("${nodeid}")
    fi
  done
}

absolute_collected_nodeid() {
  local nodeid="$1"
  local path_part test_part

  if [[ "${nodeid}" != *::* ]]; then
    printf '%s\n' "${nodeid}"
    return
  fi

  path_part="${nodeid%%::*}"
  test_part="${nodeid#*::}"
  if [[ "${path_part}" == /* ]]; then
    printf '%s::%s\n' "${path_part}" "${test_part}"
  else
    printf '%s/%s::%s\n' "${PYTEST_ROOTDIR}" "${path_part}" "${test_part}"
  fi
}

collect_first_nodeids_by_func() {
  local input_name="$1"
  local output_name="$2"
  local confcutdir="$3"
  local -n input_ref="${input_name}"
  local -n output_ref="${output_name}"
  local line key
  local -A seen=()

  output_ref=()
  while IFS= read -r line; do
    [[ "${line}" == *"::"* ]] || continue
    key="${line%%[*}"
    if [[ -z "${seen[${key}]+x}" ]]; then
      seen["${key}"]=1
      output_ref+=("$(absolute_collected_nodeid "${line}")")
    fi
  done < <(
    PYTHONPATH="${TILELANG_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR}" \
      python -m pytest -p no:warnings --rootdir="${PYTEST_ROOTDIR}" \
        --collect-only -q --confcutdir="${confcutdir}" \
        "${input_ref[@]}" 2>/dev/null || true
  )
}

echo "Running ${#ALL_NODEIDS[@]} selected pytest cases"

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

echo "Running ${#ALL_NODEIDS[@]} test cases from ${REPO_DIR}"
cd "${REPO_DIR}"

if [[ "${PRECOMPILE_TEST_KERNELS}" == "1" ]]; then
  xdist_args=()
  precompile_status=0
  TILE_KERNEL_PRECOMPILE_NODEIDS=()
  TILELANG_PRECOMPILE_NODEIDS=()
  TILELANG_EXAMPLES_PRECOMPILE_NODEIDS=()
  if grep -q -- "--numprocesses" "${pytest_help_file}"; then
    xdist_args=(-n "${PRECOMPILE_JOBS}")
  else
    echo "pytest-xdist plugin not available, precompiling serially" >&2
  fi

  dedupe_param_nodeids_by_func TILE_KERNEL_NODEIDS TILE_KERNEL_PRECOMPILE_NODEIDS
  collect_first_nodeids_by_func TILELANG_NODEIDS TILELANG_PRECOMPILE_NODEIDS "${TILELANG_REPO_DIR}/testing/python/hcu"
  collect_first_nodeids_by_func TILELANG_EXAMPLES_NODEIDS TILELANG_EXAMPLES_PRECOMPILE_NODEIDS "${TILELANG_REPO_DIR}/examples"

  echo "Precompiling ${#TILE_KERNEL_PRECOMPILE_NODEIDS[@]} TileKernels test cases from ${#TILE_KERNEL_NODEIDS[@]} selected nodeids with ${PRECOMPILE_JOBS} workers"
  if ((${#TILE_KERNEL_PRECOMPILE_NODEIDS[@]} > 0)); then
    PYTHONPATH="${TILELANG_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR}" \
      TILELANG_COMPILE_ONLY=1 \
      TILEKERNELS_PRECOMPILE_ONLY=1 \
      python -m pytest -p no:warnings "${xdist_args[@]}" "${TILE_KERNEL_PRECOMPILE_NODEIDS[@]}" || precompile_status=$?
  else
    echo "No TileKernels test cases collected for precompile; skipping" >&2
  fi

  echo "Precompiling ${#TILELANG_PRECOMPILE_NODEIDS[@]} tilelang HCU test functions from ${#TILELANG_NODEIDS[@]} selected files with ${PRECOMPILE_JOBS} workers"
  if ((${#TILELANG_PRECOMPILE_NODEIDS[@]} > 0)); then
    PYTHONPATH="${TILELANG_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR}" \
      TILELANG_COMPILE_ONLY=1 \
      TILEKERNELS_PRECOMPILE_ONLY=1 \
      python -m pytest -p no:warnings "${xdist_args[@]}" \
        --confcutdir="${TILELANG_REPO_DIR}/testing/python/hcu" \
        "${TILELANG_PRECOMPILE_NODEIDS[@]}" || precompile_status=$?
  else
    echo "No tilelang HCU test functions collected for precompile; skipping" >&2
  fi

  echo "Precompiling ${#TILELANG_EXAMPLES_PRECOMPILE_NODEIDS[@]} tilelang example test functions from ${#TILELANG_EXAMPLES_NODEIDS[@]} selected files with ${PRECOMPILE_JOBS} workers"
  if ((${#TILELANG_EXAMPLES_PRECOMPILE_NODEIDS[@]} > 0)); then
    PYTHONPATH="${TILELANG_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR}" \
      TILELANG_COMPILE_ONLY=1 \
      TILEKERNELS_PRECOMPILE_ONLY=1 \
      python -m pytest -p no:warnings "${xdist_args[@]}" \
        --confcutdir="${TILELANG_REPO_DIR}/examples" \
        "${TILELANG_EXAMPLES_PRECOMPILE_NODEIDS[@]}" || precompile_status=$?
  else
    echo "No tilelang example test functions collected for precompile; skipping" >&2
  fi

  if [[ "${precompile_status}" != "0" ]]; then
    echo "Precompile failed with status ${precompile_status}; continuing to formal pytest" >&2
  fi
fi

pytest_status=0
PYTHONPATH="${TILELANG_REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR}" \
  python -m pytest -p no:warnings --rootdir="${PYTEST_ROOTDIR}" \
    "${extra_pytest_args[@]}" "${allure_args[@]}" "${ALL_NODEIDS[@]}" || pytest_status=$?

if [[ "${GENERATE_ALLURE_REPORT}" == "1" ]]; then
  if command -v allure >/dev/null 2>&1; then
    rm -rf "${ALLURE_REPORT_DIR}"
    allure generate "${ALLURE_RESULTS_DIR}" -o "${ALLURE_REPORT_DIR}"
    echo "Allure report: ${ALLURE_REPORT_DIR}"
  else
    echo "allure command not found, skipped report generation" >&2
  fi
fi

finish_pytest_status "${pytest_status}" "das-regression"
