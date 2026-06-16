#pragma once

/*
 * tilelang MLS (Matrix Load Store) templates.
 *
 * Usage:
 *   #include "mls/tile_window_mls.h"
 *
 *   // Specialization match: uses ck_tile hand-tuned Detail
 *   using T1 = tl::mls::tile_window_mls_param_traits<
 *       ::tl::sequence<128, 64>, ::tl::sequence<16, 64>, 4, 1, 16, 1, true,
 *       ::tl::hcu_target_enum::gfx938>;  // 16 = bits (b16)
 *
 * To add new specializations: edit mls_param_traits.hpp
 * To add new MlsAtom mappings: edit tl_mls_atom_dispatcher.hpp
 */

#include <tl_templates/hcu/mls/tl_mls_atom_dispatcher.hpp>
#include <tl_templates/hcu/mls/mls_ds_traits.hpp>
#include <tl_templates/hcu/mls/mls_generic_detail.hpp>
#include <tl_templates/hcu/mls/mls_param_traits.hpp>
