#pragma once

/*
 * tilelang MLS (Matrix Load Store) templates.
 *
 * Usage:
 *   #include "mls/tile_window_mls.h"
 *
 *   // Specialization match: uses ck_tile hand-tuned Detail
 *   using T1 = tl::mls::tile_window_mls_param_traits<
 *       ck_tile::sequence<128, 64>, ck_tile::sequence<16, 64>, 4, 1, 16, 1, true,
 *       ck_tile::hcu_target_enum::gfx938>;  // 16 = bits (b16)
 *
 * To add new specializations: edit mls_param_traits.hpp
 * To add new MlsAtom mappings: edit tl_mls_atom_dispatcher.hpp
 */

#include "tl_mls_atom_dispatcher.hpp"
#include "mls_ds_traits.hpp"
#include "mls_generic_detail.hpp"
#include "mls_param_traits.hpp"
