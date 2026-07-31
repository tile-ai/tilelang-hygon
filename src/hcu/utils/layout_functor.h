// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file layout_functor.h
 * \brief Generate compile-time HCU C++ offset functors from resolved layouts.
 */

#ifndef TVM_TL_HCU_UTILS_LAYOUT_FUNCTOR_H_
#define TVM_TL_HCU_UTILS_LAYOUT_FUNCTOR_H_

#include "layout/layout.h"

#include <tvm/arith/analyzer.h>

#include <string>

namespace tvm {
namespace tl {
namespace hcu {

struct GeneratedLayoutFunctor {
  std::string name;
  std::string source;
};

/*! \brief Convert Layout::Forward into a row-major byte-offset C++ functor.
 *
 * The generated ABI has five integer inputs so instruction templates can use a
 * uniform interface for layouts of rank 1..5. Unused trailing inputs do not
 * occur in the generated offset expression.
 */
GeneratedLayoutFunctor MakeLayoutOffsetFunctor(
    const Layout &layout, const DataType &dtype, arith::Analyzer *analyzer,
    const std::string &name_prefix = "TLGeneratedLayoutOffset_",
    size_t leading_broadcast_dims = 0);

} // namespace hcu
} // namespace tl
} // namespace tvm

#endif // TVM_TL_HCU_UTILS_LAYOUT_FUNCTOR_H_
