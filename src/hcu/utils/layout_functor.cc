// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

/*!
 * \file layout_functor.cc
 */

#include "layout_functor.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

#include <iomanip>
#include <sstream>

namespace tvm {
namespace tl {
namespace hcu {

using namespace tirx;
using namespace ffi;

namespace {

std::string LayoutExprToCpp(const PrimExpr &expr, const Array<Var> &vars) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return std::to_string(imm->value);
  }
  if (const auto *var = expr.as<VarNode>()) {
    for (size_t i = 0; i < vars.size(); ++i) {
      if (vars[i].get() == var) {
        return "i" + std::to_string(i);
      }
    }
    LOG(FATAL) << "resolved layout contains a free variable: "
               << GetRef<Var>(var);
  }
  auto binary = [&](const PrimExpr &a, const char *op,
                    const PrimExpr &b) -> std::string {
    return "(" + LayoutExprToCpp(a, vars) + " " + op + " " +
           LayoutExprToCpp(b, vars) + ")";
  };
  if (const auto *n = expr.as<AddNode>())
    return binary(n->a, "+", n->b);
  if (const auto *n = expr.as<SubNode>())
    return binary(n->a, "-", n->b);
  if (const auto *n = expr.as<MulNode>())
    return binary(n->a, "*", n->b);
  if (const auto *n = expr.as<FloorDivNode>())
    return binary(n->a, "/", n->b);
  if (const auto *n = expr.as<FloorModNode>())
    return binary(n->a, "%", n->b);
  if (const auto *n = expr.as<CastNode>()) {
    return LayoutExprToCpp(n->value, vars);
  }
  if (const auto *call = expr.as<CallNode>()) {
    ICHECK_EQ(call->args.size(), 2u)
        << "layout functor only supports binary bitwise calls, got " << expr;
    if (call->op.same_as(builtin::bitwise_xor()))
      return binary(call->args[0], "^", call->args[1]);
    if (call->op.same_as(builtin::bitwise_and()))
      return binary(call->args[0], "&", call->args[1]);
    if (call->op.same_as(builtin::bitwise_or()))
      return binary(call->args[0], "|", call->args[1]);
    if (call->op.same_as(builtin::shift_left()))
      return binary(call->args[0], "<<", call->args[1]);
    if (call->op.same_as(builtin::shift_right()))
      return binary(call->args[0], ">>", call->args[1]);
  }
  LOG(FATAL) << "unsupported expression in resolved layout: " << expr;
  return "0";
}

} // namespace

GeneratedLayoutFunctor MakeLayoutOffsetFunctor(const Layout &layout,
                                               const DataType &dtype,
                                               arith::Analyzer *analyzer,
                                               const std::string &name_prefix,
                                               size_t leading_broadcast_dims) {
  ICHECK_LE(leading_broadcast_dims, layout->InputDim());
  const size_t plane_rank = layout->InputDim() - leading_broadcast_dims;
  ICHECK_LE(plane_rank, 5u)
      << "layout offset functor supports at most five input dimensions";
  const int element_bits = dtype.bits() * dtype.lanes();
  ICHECK_EQ(element_bits % 8, 0)
      << "layout byte-offset functor requires a whole-byte element dtype";
  Array<Var> vars;
  for (size_t i = 0; i < plane_rank; ++i) {
    vars.push_back(Var("i" + std::to_string(i), DataType::Int(32)));
  }
  Array<PrimExpr> layout_inputs;
  for (size_t i = 0; i < leading_broadcast_dims; ++i) {
    layout_inputs.push_back(make_zero(DataType::Int(32)));
  }
  for (const Var &var : vars) {
    layout_inputs.push_back(var);
  }
  Array<PrimExpr> physical = layout->Forward(layout_inputs);
  Array<PrimExpr> shape = layout->OutputShape();
  ICHECK_EQ(physical.size(), shape.size());
  PrimExpr offset = make_zero(DataType::Int(32));
  PrimExpr stride = make_const(DataType::Int(32), 1);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    offset = offset + physical[i] * stride;
    stride = stride * shape[i];
  }
  offset = offset * make_const(DataType::Int(32), element_bits / 8);
  std::string cpp_expr = LayoutExprToCpp(analyzer->Simplify(offset), vars);

  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char c : cpp_expr) {
    hash ^= c;
    hash *= 1099511628211ULL;
  }
  std::ostringstream name;
  name << name_prefix << std::hex << std::setw(16) << std::setfill('0') << hash;
  std::ostringstream source;
  source << "struct " << name.str() << " {\n"
         << "  TL_DEVICE static int CalculateOffset(int i0, int i1, int i2, "
            "int i3, int i4) {\n"
         << "    return " << cpp_expr << ";\n"
         << "  }\n"
         << "};\n";
  return {name.str(), source.str()};
}

} // namespace hcu
} // namespace tl
} // namespace tvm
