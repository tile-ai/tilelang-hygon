# Third-party notices (TileLang HCU templates)

## Copyright policy (Hygon)

| Case | Action |
| --- | --- |
| Hygon original (Python / Shell / YAML) | `# Copyright (c) 2026 Hygon Information Technology Co., Ltd.` + `# SPDX-License-Identifier: MIT` |
| Substantively modified upstream MIT file | Retain upstream copyright and license; append Hygon copyright + `Modified by Hygon Information Technology Co., Ltd., 2026.`; do not duplicate SPDX |
| Comment / format-only edits | Do not add Hygon copyright |
| Unchanged third-party copy | Keep third-party notice only; record here |

## AMD Composable Kernel — ck_tile (MIT)

The following files under `../core/` are derived from AMD Composable Kernel
`ck_tile` headers (MIT License). Hygon/HCU-specific MLS/matrix code under `../mls/`
is **not** part of this AMD attribution.

### AMD MIT infrastructure (representative paths)

- `core/algorithm/coordinate_transform.hpp`
- `core/algorithm/indexing_adaptor.hpp`
- `core/algorithm/space_filling_curve.hpp`
- `core/arch/amd_buffer_addressing.hpp`
- `core/container/*` (subset used by descriptor stack)
- `core/numeric/*` (subset)
- `core/tensor/tensor_{adaptor,descriptor,coordinate}*.hpp`
- `core/tensor/tile_distribution_encoding.hpp`
- `core/utility/*` (subset)

### Hygon HCU modifications (ck_tile-derived)

Files under `../core/` retain AMD MIT attribution. Hygon Information Technology
Co., Ltd. has modified portions of this tree for HCU targets, including but
not limited to:

- compile-time macros and feature toggles (`core/config.hpp`, …)
- buffer load/store builtins, LLVM intrinsics, and inline assembly
  (`core/arch/amd_buffer_addressing.hpp`, …)
- architecture-specific paths and guards (e.g. gfx938, gfx946)

Files with material Hygon changes append Hygon copyright in the file header per
the policy above. Representative modified files:

- `core/config.hpp` — derived from `ck_tile/core/config.hpp`
- `core/arch/amd_buffer_addressing.hpp` — derived from `ck_tile` buffer-addressing
  headers (aggregated in this tree)

### HCU / TileLang self-developed (no AMD attribution)

- `../mls/mls_resource.hpp`
- `../mls/tilelang_mls_base.hpp`, `mls_generic_detail.hpp`, `tl_mls_atom_*`, …

Upstream extraction reference: [AMD Composable Kernel](https://github.com/ROCm/composable_kernel) `ck_tile` headers.
Vendored MIT portions live under `../core/` in this tree.

See `LICENSE-MIT` for the full MIT license text.
