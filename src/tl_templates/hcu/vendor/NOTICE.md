# Third-party notices (TileLang HCU templates)

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

### HCU / TileLang self-developed (no AMD attribution)

- `../mls/mls_resource.hpp`
- `../mls/tilelang_mls_base.hpp`, `mls_generic_detail.hpp`, `tl_mls_atom_*`, …

Upstream extraction reference: [AMD Composable Kernel](https://github.com/ROCm/composable_kernel) `ck_tile` headers.
Vendored MIT portions live under `../core/` in this tree.

See `LICENSE-MIT` for the full MIT license text.
