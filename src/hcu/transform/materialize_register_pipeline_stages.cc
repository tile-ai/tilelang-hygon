/*!
 * \file materialize_register_pipeline_stages.cc
 * \brief Materialize the rotated register slots of a pipelined GEMM K loop.
 *
 * A register pipeline rotates its shared->register buffers: the producer of
 * one iteration fills a slot while the consumer of another drains it. The
 * injector expresses that rotation by multi-versioning each local buffer with
 * a leading slot dimension, which leaves every access with a
 * `floor_mod(iteration, versions)` index. Resolving such an index needs the
 * rank-relaxed layout lowering a plain fragment access never uses.
 *
 * This pass removes the need for that relaxation. It unrolls the K loop by a
 * multiple of the rotation period -- peeling the leading iterations instead of
 * leaving a tail behind the loop, so no live rotation is left outside the
 * unrolled body -- and gives every slot its own same-shape buffer. From this
 * point on the compiler only sees rank-2 fragments that any ordinary kernel
 * would allocate. The per-iteration work the body carries shrinks by the
 * unroll factor, which the loop annotation records for the passes that count
 * commits per iteration.
 */

#include "hcu/op/builtin.h"
#include "hcu/target_utils.h"
#include "hcu/transform/gemm_k_loop_utils.h"
#include "op/builtin.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/pipeline_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;
using ffi::Map;
using ffi::String;

namespace {

/*!
 * \brief Fold a rotation index whose loop-var coefficient is a multiple of the
 * rotation period.
 *
 * The unrolled body indexes a slot with `floor_mod(k * factor + offset,
 * versions)`. Because the unroll factor is a multiple of every version count,
 * the coefficient vanishes and the index folds to a constant per phase.
 */
class PipelineIndexFolder : public StmtExprMutator {
public:
  explicit PipelineIndexFolder(const Var &loop_var) : loop_var_(loop_var) {}

private:
  static int64_t FloorModI64(int64_t a, int64_t b) {
    int64_t r = a % b;
    if (r < 0) {
      r += (b > 0 ? b : -b);
    }
    return r;
  }

  std::optional<int64_t> MatchDivisor(const PrimExpr &rhs) const {
    if (auto val = GetConstIntValue(rhs)) {
      if (*val > 0) {
        return *val;
      }
    }
    return std::nullopt;
  }

  std::optional<int64_t> MatchAndMaskDivisor(const PrimExpr &rhs) const {
    if (auto val = GetConstIntValue(rhs)) {
      int64_t mask = *val;
      int64_t divisor = mask + 1;
      if (mask > 0 && (divisor & (divisor - 1)) == 0) {
        return divisor;
      }
    }
    return std::nullopt;
  }

  std::optional<PrimExpr> TryFoldMod(const PrimExpr &lhs,
                                     int64_t divisor) const {
    if (divisor <= 0) {
      return std::nullopt;
    }
    if (auto lhs_val = GetConstIntValue(lhs)) {
      return make_const(loop_var_.dtype(), FloorModI64(*lhs_val, divisor));
    }
    ffi::Array<PrimExpr> linear = arith::DetectLinearEquation(lhs, {loop_var_});
    if (linear.size() != 2) {
      return std::nullopt;
    }
    auto coeff = GetConstIntValue(linear[0]);
    if (!coeff.has_value()) {
      return std::nullopt;
    }
    if (*coeff == 0) {
      if (auto c = GetConstIntValue(linear[1])) {
        return make_const(loop_var_.dtype(), FloorModI64(*c, divisor));
      }
      return std::nullopt;
    }
    if ((*coeff % divisor) != 0) {
      return std::nullopt;
    }
    PrimExpr folded =
        floormod(linear[1], make_const(loop_var_.dtype(), divisor));
    if (auto folded_val = GetConstIntValue(folded)) {
      return make_const(loop_var_.dtype(), *folded_val);
    }
    return std::nullopt;
  }

  PrimExpr VisitExpr_(const FloorModNode *op) final {
    PrimExpr lhs = VisitExpr(op->a);
    PrimExpr rhs = VisitExpr(op->b);
    if (auto divisor = MatchDivisor(rhs)) {
      if (auto folded = TryFoldMod(lhs, *divisor)) {
        return *folded;
      }
    }
    if (!lhs.same_as(op->a) || !rhs.same_as(op->b)) {
      return FloorMod(lhs, rhs);
    }
    return GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::bitwise_and()) && op->args.size() == 2) {
      PrimExpr lhs = VisitExpr(op->args[0]);
      PrimExpr rhs = VisitExpr(op->args[1]);
      if (auto divisor = MatchAndMaskDivisor(rhs)) {
        if (auto folded = TryFoldMod(lhs, *divisor)) {
          return *folded;
        }
      }
      if (!lhs.same_as(op->args[0]) || !rhs.same_as(op->args[1])) {
        return Call(op->dtype, op->op, {lhs, rhs});
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Var loop_var_;
};

Stmt FoldPipelineIndices(const Stmt &stmt, const Var &loop_var) {
  return PipelineIndexFolder(loop_var)(stmt);
}

/*! \brief Read the injector's versioned-local record for \p loop. */
std::unordered_map<std::string, int64_t>
ReadVersionedLocals(const ForNode *loop) {
  std::unordered_map<std::string, int64_t> versions;
  auto value = loop->annotations.Get(kRegisterPipelineVersionedLocals);
  if (!value.has_value()) {
    return versions;
  }
  auto record = value.value().as<Map<String, Any>>();
  if (!record.has_value()) {
    return versions;
  }
  for (const auto &entry : record.value()) {
    if (const auto *count = entry.second.as<IntImmNode>()) {
      versions[static_cast<std::string>(entry.first)] = count->value;
    }
  }
  return versions;
}

/*! \brief Unroll a register-pipeline K loop into whole rotation periods. */
class KLoopStageUnroller : public StmtMutator {
public:
  KLoopStageUnroller(int64_t base_factor,
                     std::unordered_map<std::string, int64_t> *versions,
                     bool *transformed)
      : base_factor_(base_factor), versions_(versions),
        transformed_(transformed) {}

private:
  Stmt VisitStmt_(const ForNode *op) final {
    if (!LoopHasAppliedRegisterPipeline(op)) {
      return StmtMutator::VisitStmt_(op);
    }
    Stmt body = StmtMutator::VisitStmt(op->body);
    std::unordered_map<std::string, int64_t> versions =
        ReadVersionedLocals(op);
    if (versions.empty()) {
      // Nothing rotates, so nothing has to become a constant slot.
      if (body.same_as(op->body)) {
        return GetRef<For>(op);
      }
      For loop = GetRef<For>(op);
      auto *n = loop.CopyOnWrite();
      n->body = std::move(body);
      return loop;
    }
    auto extent_opt = GetConstIntValue(op->extent);
    ICHECK(extent_opt.has_value())
        << "a register-pipeline K loop needs a constant extent to materialize "
           "its register slots, got "
        << op->extent;

    const int64_t extent = *extent_opt;
    int64_t period = 1;
    for (const auto &entry : versions) {
      period = std::lcm(period, entry.second);
    }
    int64_t factor = period;
    while (factor < base_factor_) {
      factor += period;
    }
    // A K loop shorter than the configured unroll window still holds whole
    // rotation periods, so shrink the window to the largest one the loop does
    // hold. A loop shorter than a single period cannot give its slots constant
    // indices while keeping the loop the wait plan and the warp split drive
    // the pipeline off; RegisterPipelinePlanning declines such loops.
    ICHECK_GE(extent, period)
        << "a register-pipeline K loop of " << extent
        << " iteration(s) does not complete one " << period
        << "-iteration register rotation; RegisterPipelinePlanning should have "
           "declined it";
    if (extent < factor) {
      factor = extent / period * period;
    }
    const int64_t peeled = extent % factor;
    const int64_t main_extent = extent / factor;
    const DataType index_dtype = op->loop_var.dtype();

    Array<Stmt> parts;
    // Peel the leading iterations instead of trailing ones: a tail after the
    // loop would be read as pipeline epilogue by the fence planner, which
    // schedules its waits for the loop body alone.
    for (int64_t i = 0; i < peeled; ++i) {
      PrimExpr tile = op->min + make_const(index_dtype, i);
      Stmt peeled_body = Substitute(body, {{op->loop_var, tile}});
      parts.push_back(FoldPipelineIndices(peeled_body, op->loop_var));
    }

    // The main loop counts whole rotation periods, so every slot index folds
    // to a constant.
    Var iter = op->loop_var.copy_with_suffix("");
    Array<Stmt> main_parts;
    main_parts.reserve(static_cast<size_t>(factor));
    for (int64_t i = 0; i < factor; ++i) {
      PrimExpr tile = iter * make_const(index_dtype, factor) +
                      (op->min + make_const(index_dtype, peeled + i));
      Stmt phase_body = Substitute(body, {{op->loop_var, tile}});
      main_parts.push_back(FoldPipelineIndices(phase_body, iter));
    }
    Map<String, Any> annotations = op->annotations;
    annotations.erase(kRegisterPipelineVersionedLocals);
    // The per-statement schedule described one iteration; the body no longer
    // matches it and no pass after the injector reads it back.
    annotations.erase(kRegisterPipelineStage);
    annotations.erase(kRegisterPipelineOrder);
    annotations.erase(kRegisterPipelineAsyncStages);
    annotations.Set(kRegisterPipelinePhaseUnroll, Integer(factor));
    parts.push_back(For(iter, Integer(0),
                        make_const(op->extent.dtype(), main_extent), op->kind,
                        SeqStmt::Flatten(main_parts), op->thread_binding,
                        std::move(annotations), op->step, op->span));

    for (const auto &entry : versions) {
      auto [it, inserted] = versions_->emplace(entry.first, entry.second);
      if (!inserted) {
        ICHECK_EQ(it->second, entry.second)
            << "register-pipeline buffer " << entry.first
            << " is versioned with " << it->second << " and " << entry.second
            << " slots by different loops";
      }
    }
    *transformed_ = true;
    return SeqStmt::Flatten(parts);
  }

  int64_t base_factor_;
  std::unordered_map<std::string, int64_t> *versions_;
  bool *transformed_;
};

/*! \brief Find the buffers the injector versioned, by the names it recorded. */
class VersionedBufferFinder : public StmtVisitor {
public:
  explicit VersionedBufferFinder(
      const std::unordered_map<std::string, int64_t> *versions)
      : versions_(versions) {}

  std::unordered_map<std::string, Buffer> found;

private:
  void Note(const Buffer &buffer) {
    if (versions_->count(static_cast<std::string>(buffer->name))) {
      found[static_cast<std::string>(buffer->name)] = buffer;
    }
  }

  void VisitStmt_(const SBlockNode *op) final {
    for (const Buffer &buffer : op->alloc_buffers) {
      Note(buffer);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    Note(op->buffer);
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    Note(op->buffer);
    StmtVisitor::VisitStmt_(op);
  }

  const std::unordered_map<std::string, int64_t> *versions_;
};

/*!
 * \brief Give every rotation slot of \p source its own same-shape buffer.
 *
 * The slot buffers share nothing: a distinct data variable is what keeps the
 * layout inference and register allocation from treating two stages as one
 * buffer, so the leading version dimension can be dropped entirely.
 */
Array<Buffer> MakeSlotBuffers(const Buffer &source, int64_t versions) {
  ICHECK_GE(source->shape.size(), 2U)
      << "a versioned register-pipeline buffer must carry a slot dimension "
         "and its fragment shape, got rank "
      << source->shape.size() << " for " << source->name;
  Array<Buffer> slots;
  slots.reserve(static_cast<size_t>(versions));
  for (int64_t i = 0; i < versions; ++i) {
    std::string name = source->name + "_" + std::to_string(i);
    ffi::ObjectPtr<BufferNode> node =
        ffi::make_object<BufferNode>(*source.get());
    Var data = source->data->type_annotation.defined()
                   ? Var(name, source->data->type_annotation)
                   : Var(name, source->data.dtype());
    node->data = std::move(data);
    node->name = name;
    node->shape.erase(node->shape.begin());
    if (!node->strides.empty()) {
      node->strides.erase(node->strides.begin());
    }
    slots.push_back(Buffer(node));
  }
  return slots;
}

/*! \brief Replace every access of a versioned local with its slot buffer. */
class RegisterSlotSplitter : public StmtExprMutator {
public:
  explicit RegisterSlotSplitter(
      std::unordered_map<std::string, Array<Buffer>> slots)
      : slots_(std::move(slots)) {}

private:
  std::optional<Array<Buffer>> SlotsOf(const Buffer &buffer) const {
    auto it = slots_.find(static_cast<std::string>(buffer->name));
    if (it == slots_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  int64_t ConstAfterSimplify(const PrimExpr &expr) const {
    auto value = GetConstIntValue(analyzer_.Simplify(expr));
    ICHECK(value.has_value()) << "expected a constant register-pipeline stage "
                                 "expression, got "
                              << expr;
    return *value;
  }

  int64_t SlotIndex(const PrimExpr &index, const Buffer &buffer) const {
    auto value = GetConstIntValue(analyzer_.Simplify(index));
    ICHECK(value.has_value())
        << "register-pipeline stage index of " << buffer->name
        << " did not fold to a constant after unrolling: " << index;
    ICHECK_GE(*value, 0) << "register-pipeline stage index of "
                         << buffer->name << " is negative: " << *value;
    return *value;
  }

  Buffer SlotBuffer(const Array<Buffer> &slots, int64_t index,
                    const Buffer &buffer) const {
    ICHECK_LT(index, static_cast<int64_t>(slots.size()))
        << "register-pipeline stage index " << index << " of " << buffer->name
        << " is outside the " << slots.size() << " materialized slot(s)";
    return slots[index];
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto slots = SlotsOf(op->buffer);
    if (!slots.has_value()) {
      return StmtExprMutator::VisitExpr_(op);
    }
    ICHECK_GE(op->indices.size(), 2U);
    const Buffer slot =
        SlotBuffer(slots.value(), SlotIndex(op->indices[0], op->buffer),
                   op->buffer);
    Array<PrimExpr> indices;
    indices.reserve(op->indices.size() - 1);
    for (size_t i = 1; i < op->indices.size(); ++i) {
      indices.push_back(VisitExpr(op->indices[i]));
    }
    return BufferLoad(slot, indices, std::nullopt, op->span);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto slots = SlotsOf(op->buffer);
    if (!slots.has_value()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    ICHECK_GE(op->indices.size(), 2U);
    const Buffer slot =
        SlotBuffer(slots.value(), SlotIndex(op->indices[0], op->buffer),
                   op->buffer);
    Array<PrimExpr> indices;
    indices.reserve(op->indices.size() - 1);
    for (size_t i = 1; i < op->indices.size(); ++i) {
      indices.push_back(VisitExpr(op->indices[i]));
    }
    return BufferStore(slot, VisitExpr(op->value), indices, std::nullopt,
                       op->span);
  }

  /*!
   * \brief Rewrite a selector region around a versioned local.
   *
   * The leading slot dimension is dropped and a region covering `n` slots
   * becomes one region per slot, because each slot is now a buffer of its own.
   * The body accesses one slot at a time and its selector has already been
   * folded to a constant; a region whose selector is still symbolic is a block
   * read/write set declared against the pre-loop tile index, and there the
   * whole slot range is the safe reading.
   */
  std::optional<Array<BufferRegion>> SplitRegion(const BufferRegion &region) {
    auto slots = SlotsOf(region->buffer);
    if (!slots.has_value()) {
      return std::nullopt;
    }
    ICHECK_GE(region->region.size(), 2U);
    int64_t first = 0;
    int64_t count = static_cast<int64_t>(slots.value().size());
    if (auto folded_min = GetConstIntValue(
            analyzer_.Simplify(region->region[0]->min));
        folded_min.has_value()) {
      if (auto folded_extent =
              GetConstIntValue(analyzer_.Simplify(region->region[0]->extent));
          folded_extent.has_value() && *folded_extent > 0) {
        first = *folded_min;
        count = *folded_extent;
      }
    }
    Array<Range> ranges;
    ranges.reserve(region->region.size() - 1);
    for (size_t i = 1; i < region->region.size(); ++i) {
      ranges.push_back(
          Range::FromMinExtent(VisitExpr(region->region[i]->min),
                               VisitExpr(region->region[i]->extent)));
    }
    Array<BufferRegion> split;
    split.reserve(count);
    for (int64_t i = 0; i < count; ++i) {
      split.push_back(
          BufferRegion(SlotBuffer(slots.value(), first + i, region->buffer),
                       ranges));
    }
    return split;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(tl::region())) {
      return StmtExprMutator::VisitExpr_(op);
    }
    const auto *load = op->args[0].as<BufferLoadNode>();
    if (load == nullptr || !SlotsOf(load->buffer).has_value()) {
      return StmtExprMutator::VisitExpr_(op);
    }
    ICHECK_EQ(load->indices.size() + 2, op->args.size());
    auto slots = SlotsOf(load->buffer).value();
    const int64_t slot = SlotIndex(load->indices[0], load->buffer);
    ICHECK_EQ(ConstAfterSimplify(VisitExpr(op->args[2])), 1)
        << "a rotated register-pipeline buffer is accessed one slot at a "
           "time, but the region on "
        << load->buffer->name << " covers " << op->args[2] << " slots";
    Array<PrimExpr> indices;
    Array<PrimExpr> args;
    indices.reserve(load->indices.size() - 1);
    for (size_t i = 1; i < load->indices.size(); ++i) {
      indices.push_back(VisitExpr(load->indices[i]));
      args.push_back(VisitExpr(op->args[2 + i]));
    }
    PrimExpr base = BufferLoad(SlotBuffer(slots, slot, load->buffer), indices);
    Array<PrimExpr> new_args;
    new_args.reserve(args.size() + 2);
    new_args.push_back(std::move(base));
    new_args.push_back(op->args[1]);
    for (const PrimExpr &extent : args) {
      new_args.push_back(extent);
    }
    return Call(op->dtype, op->op, new_args, op->annotations, op->span);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    bool changed = false;
    Array<Buffer> alloc_buffers;
    for (const Buffer &buffer : op->alloc_buffers) {
      auto slots = SlotsOf(buffer);
      if (slots.has_value()) {
        for (const Buffer &slot : slots.value()) {
          alloc_buffers.push_back(slot);
        }
        changed = true;
      } else {
        alloc_buffers.push_back(buffer);
      }
    }
    auto split_regions = [&](const Array<BufferRegion> &regions) {
      Array<BufferRegion> result;
      for (const BufferRegion &region : regions) {
        if (auto split = SplitRegion(region)) {
          for (const BufferRegion &slot_region : split.value()) {
            result.push_back(slot_region);
          }
          changed = true;
        } else {
          result.push_back(BufferRegion(
              region->buffer,
              region->region.Map([&](const Range &r) {
                return Range::FromMinExtent(VisitExpr(r->min),
                                            VisitExpr(r->extent));
              })));
        }
      }
      return result;
    };
    Array<BufferRegion> reads = split_regions(op->reads);
    Array<BufferRegion> writes = split_regions(op->writes);
    ffi::Optional<Stmt> init;
    bool init_changed = false;
    if (op->init.defined()) {
      Stmt new_init = VisitStmt(op->init.value());
      init_changed = !new_init.same_as(op->init.value());
      init = new_init;
    }
    Stmt body = VisitStmt(op->body);
    if (!changed && !init_changed && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    SBlock block = GetRef<SBlock>(op);
    auto *n = block.CopyOnWrite();
    n->alloc_buffers = std::move(alloc_buffers);
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->init = std::move(init);
    n->body = std::move(body);
    return block;
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    auto slots = SlotsOf(op->buffer);
    if (!slots.has_value()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Array<Stmt> allocs;
    for (const Buffer &slot : slots.value()) {
      allocs.push_back(AllocBuffer(slot, op->annotations, op->span));
    }
    return SeqStmt::Flatten(allocs);
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    auto slots = SlotsOf(op->buffer);
    if (!slots.has_value()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Array<Stmt> decls;
    for (const Buffer &slot : slots.value()) {
      decls.push_back(DeclBuffer(slot, op->span));
    }
    return SeqStmt::Flatten(decls);
  }

  std::unordered_map<std::string, Array<Buffer>> slots_;
  mutable arith::Analyzer analyzer_;
};

int64_t GetUnrollFactor(const tvm::transform::PassContext &ctx) {
  if (ctx.defined()) {
    auto val = ctx->GetConfig<Integer>(kPipelinedKUnrollFactor, Integer(4));
    if (val.defined()) {
      return static_cast<int64_t>(val.value()->value);
    }
  }
  return 4;
}

/*!
 * \brief Collect the loops the register injector rewrote.
 *
 * The GEMM K loop cannot be recognized by its body here: this pass runs before
 * LayoutInference and GEMM lowering, so the loop still holds tile ops rather
 * than the MMA and DMA calls the later analysis keys on. The injector's own
 * marker is the only evidence available this early, and it is exact.
 */
class RegisterPipelineLoopCollector : public StmtVisitor {
public:
  std::vector<const ForNode *> loops;

 private:
  void VisitStmt_(const ForNode *op) final {
    if (LoopHasAppliedRegisterPipeline(op)) {
      loops.push_back(op);
    }
    StmtVisitor::VisitStmt_(op);
  }
};

} // namespace

tirx::transform::Pass MaterializeRegisterPipelineStages() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &ctx) {
    Target target = f->GetAttr<Target>(tvm::attr::kTarget).value_or(Target());
    if (!target.defined() || !TargetIsHCU(target)) {
      return f;
    }
    RegisterPipelineLoopCollector collector;
    collector(f->body);
    if (collector.loops.empty()) {
      return f;
    }
    std::unordered_map<std::string, int64_t> versions;
    bool transformed = false;
    auto *n = f.CopyOnWrite();
    n->body = KLoopStageUnroller(GetUnrollFactor(ctx), &versions, &transformed)(
        std::move(n->body));
    if (!transformed || versions.empty()) {
      return f;
    }
    // Duplicating the body duplicates the scalar Bind statements the injector
    // emits per iteration; give every copy its own variables before the split
    // rewrites the accesses.
    n->body = ConvertSSA(std::move(n->body));

    VersionedBufferFinder finder(&versions);
    finder(n->body);
    std::unordered_map<std::string, Array<Buffer>> slots;
    for (const auto &entry : versions) {
      auto found = finder.found.find(entry.first);
      ICHECK(found != finder.found.end())
          << "the injector versioned " << entry.first
          << " but no allocation of it is left in the kernel";
      slots.emplace(entry.first,
                    MakeSlotBuffers(found->second, entry.second));
    }
    n->body = RegisterSlotSplitter(std::move(slots))(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0,
                            "tl.MaterializeRegisterPipelineStages", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MaterializeRegisterPipelineStages",
                        MaterializeRegisterPipelineStages);
}

} // namespace tl
} // namespace tvm
