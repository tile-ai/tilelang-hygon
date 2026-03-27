/*!
 * \file insert_matrix_load_waitcnt.cc
 * \brief Insert __builtin_amdgcn_s_waitcnt(0) before consumers of mls_load_tile.
 *
 * matrix_load (mls_load_tile) is async; its consumers (tl_gemm, ds_read_format)
 * must wait via s_waitcnt before using the loaded data.
 */
#include <tvm/ir/transform.h>
#include <tvm/node/structural_equal.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::transform::PassContext;

static const int kPathNotReached = -1;
static const int kPathAtProducer = 0;
static const int kPathSyncInserted = 1;
static const int kPathOtherConsumerGotSync = -2;

static bool IsTlGemm(const CallNode *call) {
  return call->op.same_as(tl::tl_gemm());
}


// tvm_access_ptr(dtype, buffer_var, offset, extent, access_mask)
struct BufferRegion {
  Var buffer_var;
  PrimExpr offset;
  PrimExpr extent;
};

static std::optional<BufferRegion> GetBufferRegionFromAccessPtr(const PrimExpr &expr) {
  if (const auto *call = expr.as<CallNode>()) {
    if (call->op.same_as(builtin::tvm_access_ptr()) && call->args.size() >= 5) {
      if (const auto *var = call->args[1].as<VarNode>()) {
        BufferRegion r;
        r.buffer_var = tvm::ffi::GetRef<Var>(var);
        r.offset = call->args[2];
        r.extent = call->args[3];
        return r;
      }
    }
  }
  return std::nullopt;
}

// Check if two regions overlap. For constant offset/extent, do numeric check.
// For symbolic, use structural equality on offset (conservative: same expr => overlap).
static bool RegionsOverlap(const BufferRegion &a, const BufferRegion &b) {
  if (!a.buffer_var.same_as(b.buffer_var)) {
    return false;
  }
  const auto *a_off = a.offset.as<IntImmNode>();
  const auto *a_ext = a.extent.as<IntImmNode>();
  const auto *b_off = b.offset.as<IntImmNode>();
  const auto *b_ext = b.extent.as<IntImmNode>();
  if (a_off && a_ext && b_off && b_ext) {
    int64_t a_begin = a_off->value;
    int64_t a_end = a_begin + a_ext->value;
    int64_t b_begin = b_off->value;
    int64_t b_end = b_begin + b_ext->value;
    return a_begin < b_end && b_begin < a_end;
  }
  return StructuralEqual()(a.offset, b.offset);
}

struct QueueEntry {
  enum Type { kProducer, kConsumer, kManualWaitcnt };
  Type type;
  const Object *stmt{};
  int index{-1};
  std::vector<BufferRegion> regions;  // producer: dst regions; consumer: src regions
  std::vector<int> producer_indices;  // nearest producer per consumed region
};

class MlsLoadConsumerCollector : public StmtVisitor {
public:
  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

  std::vector<QueueEntry> &GetQueue() { return queue_; }
  std::unordered_set<const Object *> &GetNeedWaitcnt() { return need_waitcnt_; }

  void Analyze() {
    std::map<std::pair<int, int>, int> path_state;

    for (size_t i = 0; i < queue_.size(); i++) {
      QueueEntry &e = queue_[i];
      if (e.type == QueueEntry::kProducer) {
        for (int cidx : GetConsumerIndices(i)) {
          path_state[{i, cidx}] = kPathAtProducer;
        }
      } else if (e.type == QueueEntry::kManualWaitcnt) {
        for (auto &kv : path_state) {
          if (kv.second == kPathAtProducer) {
            kv.second = kPathSyncInserted;
          }
        }
      } else if (e.type == QueueEntry::kConsumer) {
        int nearest_producer = *std::max_element(e.producer_indices.begin(),
                                                e.producer_indices.end());
        int &state = path_state[{nearest_producer, e.index}];
        bool insert_waitcnt = (state == kPathAtProducer);

        if (insert_waitcnt) {
          need_waitcnt_.insert(e.stmt);
          for (auto &kv : path_state) {
            if (kv.second == kPathAtProducer) {
              kv.second = (kv.first.second == e.index) ? kPathSyncInserted
                                                       : kPathOtherConsumerGotSync;
            }
          }
        }
      }
    }
  }

private:
  std::vector<int> GetConsumerIndices(int producer_idx) {
    std::vector<int> result;
    for (size_t i = producer_idx + 1; i < queue_.size(); i++) {
      if (queue_[i].type == QueueEntry::kConsumer) {
        for (int pidx : queue_[i].producer_indices) {
          if (pidx == static_cast<int>(producer_idx)) {
            result.push_back(static_cast<int>(i));
            break;
          }
        }
      }
    }
    return result;
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (IsMlsLoadTileExternCall(call)) {
        auto region = GetBufferRegionFromAccessPtr(call->args[7]);
        if (region) {
          QueueEntry e;
          e.type = QueueEntry::kProducer;
          e.stmt = op;
          e.index = static_cast<int>(queue_.size());
          e.regions.push_back(*region);
          queue_.push_back(e);
        }
      } else if (IsTlGemm(call)) {
        std::vector<BufferRegion> consumer_regions;
        for (int arg_idx : {1, 2}) {
          auto region = GetBufferRegionFromAccessPtr(call->args[arg_idx]);
          if (region) {
            consumer_regions.push_back(*region);
          }
        }
        if (consumer_regions.empty()) return;

        QueueEntry e;
        e.type = QueueEntry::kConsumer;
        e.stmt = op;
        e.index = static_cast<int>(queue_.size());
        e.regions = consumer_regions;
        std::unordered_set<int> seen_producers;
        for (const BufferRegion &creg : consumer_regions) {
          bool found = false;
          for (int j = static_cast<int>(queue_.size()) - 1; j >= 0 && !found; j--) {
            if (queue_[j].type == QueueEntry::kProducer) {
              for (const BufferRegion &preg : queue_[j].regions) {
                if (RegionsOverlap(creg, preg)) {
                  if (seen_producers.insert(j).second) {
                    e.producer_indices.push_back(j);
                  }
                  found = true;
                  break;
                }
              }
            }
          }
        }
        if (!e.producer_indices.empty()) {
          queue_.push_back(e);
        }
      } else if (IsDsReadFormatExternCall(call)) {
        auto region = GetBufferRegionFromAccessPtr(call->args[1]);
        if (!region) return;

        QueueEntry e;
        e.type = QueueEntry::kConsumer;
        e.stmt = op;
        e.index = static_cast<int>(queue_.size());
        e.regions = {*region};
        std::unordered_set<int> seen_producers;
        for (int j = static_cast<int>(queue_.size()) - 1; j >= 0; j--) {
          if (queue_[j].type == QueueEntry::kProducer) {
            for (const BufferRegion &preg : queue_[j].regions) {
              if (RegionsOverlap(*region, preg)) {
                if (seen_producers.insert(j).second) {
                  e.producer_indices.push_back(j);
                }
                break;
              }
            }
          }
        }
        if (!e.producer_indices.empty()) {
          queue_.push_back(e);
        }
      } else if (IsManualSWaitcntCall(call)) {
        QueueEntry e;
        e.type = QueueEntry::kManualWaitcnt;
        e.stmt = op;
        e.index = static_cast<int>(queue_.size());
        queue_.push_back(e);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  static bool IsManualSWaitcntCall(const CallNode *call) {
    if (!call->op.same_as(builtin::call_extern()) || call->args.empty()) {
      return false;
    }
    if (const auto *name = call->args[0].as<StringImmNode>()) {
      return name->value == "__builtin_amdgcn_s_waitcnt";
    }
    return false;
  }

  std::vector<QueueEntry> queue_;
  std::unordered_set<const Object *> need_waitcnt_;
};

class InsertWaitcntMutator : public StmtMutator {
public:
  explicit InsertWaitcntMutator(const std::unordered_set<const Object *> &need_waitcnt)
      : need_waitcnt_(need_waitcnt) {}

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt ret = StmtMutator::VisitStmt_(op);
    if (need_waitcnt_.count(op)) {
      Stmt wait_stmt = MakeSWaitcnt(0);
      return SeqStmt({wait_stmt, ret});
    }
    return ret;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> new_seq;
    for (const Stmt &s : op->seq) {
      new_seq.push_back(VisitStmt(s));
    }
    return SeqStmt(std::move(new_seq));
  }

private:
  // Pack vmcnt into s_waitcnt immediate. AMD GCN s_waitcnt format:
  //   [3:0]   = Vmcnt[3:0]   (VMEM counter, 0=wait for all)
  //   [6:4]   = Expcnt       (Export counter, 7=don't wait)
  //   [7]     = update-wait-by-research (1=default)
  //   [11:8]  = Lgkmcnt      (LDS/GDS/Constant, 15=don't wait)
  //   [13:12] = 1 (default)
  //   [15:14] = Vmcnt[5:4]   (VMEM counter extension)
  // Other fields default to max (don't wait); only vmcnt is set from cnt.
  static int PackSWaitcntImm(int vmcnt) {
    const int kExpcntDontWait = 7;       // [6:4]
    const int kLgkmcntDontWait = 15;     // [11:8]
    const int kUpdateWaitByResearch = 1;  // [7]
    const int kBits12_13 = 3;             // [13:12] = 11b
    int imm = (vmcnt & 0xF) |                           // Vmcnt[3:0]
              (kExpcntDontWait << 4) |                  // Expcnt
              (kUpdateWaitByResearch << 7) |            // [7]
              (kLgkmcntDontWait << 8) |                 // Lgkmcnt
              (kBits12_13 << 12) |                      // [13:12]
              ((vmcnt & 0x30) << 10);                   // Vmcnt[5:4] -> [15:14]
    return imm;
  }

  static Stmt MakeSWaitcnt(int vmcnt) {
    int imm = PackSWaitcntImm(vmcnt);
    return Evaluate(Call(DataType::Int(32), builtin::call_extern(),
                         {StringImm("__builtin_amdgcn_s_waitcnt"),
                          IntImm(DataType::Int(32), imm)}));
  }

  const std::unordered_set<const Object *> &need_waitcnt_;
};

PrimFunc InsertMatrixLoadWaitcnt(PrimFunc func) {
  Optional<Target> opt_target = func->GetAttr<Target>(tvm::attr::kTarget);
  if (!opt_target.defined() || !TargetIsHCU(opt_target.value())) {
    return func;
  }

  MlsLoadConsumerCollector collector;
  collector.Collect(func->body);
  if (collector.GetQueue().empty()) {
    return func;
  }

  collector.Analyze();
  if (collector.GetNeedWaitcnt().empty()) {
    return func;
  }

  auto *n = func.CopyOnWrite();
  n->body = InsertWaitcntMutator(collector.GetNeedWaitcnt())(std::move(func->body));
  return func;
}

}  // namespace tl
}  // namespace tvm

namespace tvm {
namespace tl {
namespace transform {

tvm::transform::Pass InsertMatrixLoadWaitcntPass() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return tl::InsertMatrixLoadWaitcnt(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0, "tl.InsertMatrixLoadWaitcnt", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InsertMatrixLoadWaitcnt", InsertMatrixLoadWaitcntPass);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
