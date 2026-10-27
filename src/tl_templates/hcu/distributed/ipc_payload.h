// IPC-private payload offsets; never part of another backend's payload.
#ifndef TILELANG_HCU_DISTRIBUTED_IPC_PAYLOAD_H_
#define TILELANG_HCU_DISTRIBUTED_IPC_PAYLOAD_H_

#include "metadata.h"

namespace tl {
namespace ipc_payload {
constexpr int kBackendId = 1;
constexpr int kArenaSize = 0;
constexpr int kPeerBases = 1;
constexpr int kHeaderSize = kPeerBases;
constexpr int kMaxWorldSize =
    distributed_metadata::kCapacity - distributed_metadata::kPayload - kHeaderSize;
// Reserved legacy span-status values; new code uses ordinary predication.
constexpr int kInvalidSourceCount = 256;
constexpr int kInvalidSourceSpan = 257;
constexpr int kInvalidSourceArena = 258;
} // namespace ipc_payload
} // namespace tl
#endif // TILELANG_HCU_DISTRIBUTED_IPC_PAYLOAD_H_
