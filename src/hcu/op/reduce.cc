/*!
 * \file hcu/op/reduce.cc
 * \brief HCU implementation for tl.reduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "hcu/target_utils.h"
#include "hcu/utils/auto_ebarrier.h"
#include "layout/layout.h"
#include "layout/utils.h"
#include "op/builtin.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace hcu {

namespace reduce {
using backend::reduce::MakeCodegenReducer;
using backend::reduce::MakeInitValue;
using backend::reduce::MakeReduce;
} // namespace reduce

Stmt LowerWarpReduce(const ReduceOpNode &op, const LowerArgs &lower_args,
                     arith::Analyzer *) {
  ICHECK(op.src.scope() == "local.fragment" &&
         op.dst.scope() == "local.fragment")
      << "Reduce for shared memory not implemented.";
  ICHECK(TargetIsHCU(lower_args.target))
      << "Warp reduce (dim=-1) is only supported on HCU target.";

  auto get_buffer = [&](const Buffer &buf) {
    if (lower_args.buffer_remap.count(buf)) {
      return lower_args.buffer_remap[buf];
    }
    return buf;
  };

  auto src_buffer = get_buffer(op.src);
  auto dst_buffer = get_buffer(op.dst);
  Fragment src_layout = lower_args.layout_map[op.src].as<Fragment>().value();
  Fragment dst_layout = lower_args.layout_map[op.dst].as<Fragment>().value();
  size_t src_dim = src_layout->InputDim();
  size_t dst_dim = dst_layout->InputDim();

  ICHECK(src_dim == dst_dim)
      << "Warp reduce requires same input/output dimensions.";

  Array<IterVar> dst_vars;
  for (size_t i = 0; i < dst_dim; i++) {
    Var var = Var(std::string{char('i' + i)});
    dst_vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), var,
                               IterVarType::kDataPar));
  }

  Array<Stmt> stmts;
  Buffer clear_buffer = dst_buffer;
  bool clear_buffer_same_as_src = clear_buffer->data.same_as(src_buffer->data);

  auto output_shape = src_layout->OutputShape();
  ICHECK(output_shape.size() > 0)
      << "Warp reduce requires at least one output dimension.";
  PrimExpr num_registers = output_shape[0];

  int warp_size = TargetHcuGetWarpSize(lower_args.target);
  auto all_threads_int = as_const_int(lower_args.thread_bounds->extent);
  ICHECK(all_threads_int)
      << "Thread bounds extent must be constant for warp reduce.";
  int all_threads = *all_threads_int;
  auto thread_offset_int = as_const_int(lower_args.thread_bounds->min);
  ICHECK(thread_offset_int)
      << "Thread bounds minimum must be constant for warp reduce.";
  int thread_offset_value = *thread_offset_int;

  int reducing_threads = all_threads;

  auto rep_extent = as_const_int(dst_layout->ReplicateExtent());
  ICHECK(rep_extent) << "ReplicateExtent must be constant for warp reduce.";
  ICHECK_EQ(all_threads % (*rep_extent), 0)
      << "Total threads must be divisible by ReplicateExtent for warp reduce.";
  int scale = all_threads / (*rep_extent);

  ICHECK(scale >= warp_size)
      << "Scale must be greater than or equal to warp size for warp reduce.";
  ICHECK_EQ(scale % warp_size, 0)
      << "HCU warp reduce scale must contain complete waves.";

  Var rv = Var("rv");
  Array<PrimExpr> register_dst_indices;
  ICHECK(dst_layout->OutputDim() == 1)
      << "Warp reduce currently only supports 1D output layout.";
  register_dst_indices = {rv};

  Stmt warp_reduce_body;
  bool need_allreduce = (all_threads != scale);
  if (need_allreduce) {
    ICHECK_EQ(thread_offset_value % warp_size, 0)
        << "HCU cross-wave reduce must start at a wave boundary";
    ICHECK_EQ(all_threads % warp_size, 0)
        << "HCU cross-wave reduce must contain complete waves";
    int reducing_waves = all_threads / warp_size;
    ICHECK_EQ(reducing_waves & (reducing_waves - 1), 0)
        << "HCU cross-wave reduce requires a power-of-two number of waves";
  }
  auto codegen_reducer = reduce::MakeCodegenReducer(op).value();

  if (clear_buffer_same_as_src) {
    PrimExpr clear_value = BufferLoad(clear_buffer, register_dst_indices);
    if (op.type->IsAbsSum() || op.type->IsAbsMax()) {
      clear_value = Max(clear_value, -clear_value);
    }

    if (need_allreduce) {
      Array<PrimExpr> allreduce_args = {clear_value};
      ICHECK(lower_args.add_workspace != nullptr);
      allreduce_args.push_back(
          lower_args.add_workspace(all_threads, clear_buffer->dtype));

      std::stringstream ss;
      auto thread_offset = lower_args.thread_bounds->min;
      ss << "tl::AllReduce<" << codegen_reducer << ", " << reducing_threads
         << ", " << scale << ", " << thread_offset;
      if (TargetSupportsHcuEBarrier(lower_args.target)) {
        ss << ", " << kAutoEBarrierPolicyMarker;
      }
      ss << ">::run";

      allreduce_args.insert(allreduce_args.begin(), StringImm(ss.str()));

      warp_reduce_body = BufferStore(
          clear_buffer,
          Call(clear_buffer->dtype, builtin::call_extern(), allreduce_args),
          register_dst_indices);
    } else {
      warp_reduce_body =
          BufferStore(clear_buffer, clear_value, register_dst_indices);
    }
  } else {
    Stmt init_stmt;
    bool need_clear = op.clear;
    if (op.type->IsSum() || op.type->IsAbsSum()) {
      ICHECK(op.clear) << "Warp reduce requires clear=true when src_buffer "
                          "!= dst_buffer for sum/abssum reduce.";
    }

    if (need_clear) {
      Stmt clear_stmt = BufferStore(clear_buffer, reduce::MakeInitValue(op),
                                    register_dst_indices);
      init_stmt = SeqStmt(
          {clear_stmt,
           BufferStore(clear_buffer,
                       reduce::MakeReduce(
                           op, 1,
                           BufferLoad(clear_buffer, register_dst_indices),
                           BufferLoad(src_buffer, register_dst_indices)),
                       register_dst_indices)});
    } else {
      init_stmt =
          BufferStore(clear_buffer,
                      reduce::MakeReduce(
                          op, 1, BufferLoad(clear_buffer, register_dst_indices),
                          BufferLoad(src_buffer, register_dst_indices)),
                      register_dst_indices);
    }

    if (need_allreduce) {
      Array<PrimExpr> allreduce_args = {
          BufferLoad(clear_buffer, register_dst_indices)};
      ICHECK(lower_args.add_workspace != nullptr);
      allreduce_args.push_back(
          lower_args.add_workspace(all_threads, clear_buffer->dtype));

      std::stringstream ss;
      auto thread_offset = lower_args.thread_bounds->min;
      ss << "tl::AllReduce<" << codegen_reducer << ", " << reducing_threads
         << ", " << scale << ", " << thread_offset;
      if (TargetSupportsHcuEBarrier(lower_args.target)) {
        ss << ", " << kAutoEBarrierPolicyMarker;
      }
      ss << ">::run";

      allreduce_args.insert(allreduce_args.begin(), StringImm(ss.str()));

      Stmt reduce_stmt = BufferStore(
          clear_buffer,
          Call(clear_buffer->dtype, builtin::call_extern(), allreduce_args),
          register_dst_indices);
      warp_reduce_body = SeqStmt({init_stmt, reduce_stmt});
    } else {
      warp_reduce_body = init_stmt;
    }
  }

  Stmt warp_reduce_loop =
      For(rv, 0, num_registers, ForKind::kUnrolled, warp_reduce_body,
          std::nullopt, {{tirx::attr::pragma_unroll_explicit, Bool(false)}});

  stmts.push_back(warp_reduce_loop);
  return stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
}

struct HCUReduce : backend::ReduceLowerer<HCUReduce> {
  static bool SupportsFp16Bf16NanReduce(Target) { return false; }

  static int GetPreferedVectorizedSize(DataType, Target) { return 1; }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (reducing_threads > TargetHcuGetWarpSize(target) &&
        TargetSupportsHcuEBarrier(target)) {
      ss << ", " << kAutoEBarrierPolicyMarker;
    } else {
      ss << ", tl::SyncThreadsBarrier";
    }
    ss << ", " << batch << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (reducing_threads > TargetHcuGetWarpSize(target) &&
        TargetSupportsHcuEBarrier(target)) {
      ss << ", " << kAutoEBarrierPolicyMarker;
    }
    ss << ">::run";
    return ss.str();
  }
};

Stmt HcuReduceLower(const ReduceOpNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
  if (op.dim == -1) {
    return LowerWarpReduce(op, lower_args, analyzer);
  }
  return HCUReduce::Lower(op, lower_args, analyzer);
}

} // namespace hcu

namespace {

bool MatchHCUReduceTarget(Target target) { return TargetIsHCU(target); }

bool RegisterHCUReduce() {
  RegisterReduceImpl(ReduceImpl{
      "hcu.Reduce",
      MatchHCUReduceTarget,
      hcu::HcuReduceLower,
  });
  return true;
}

const bool hcu_reduce_registered = RegisterHCUReduce();

} // namespace

} // namespace tl
} // namespace tvm
