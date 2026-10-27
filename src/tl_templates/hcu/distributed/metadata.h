// Distributed wire ABI. This header is the sole source of host/device constants.
// Bump kAbiVersion for incompatible layout, field semantics or helper signatures.
#ifndef TILELANG_HCU_DISTRIBUTED_METADATA_H_
#define TILELANG_HCU_DISTRIBUTED_METADATA_H_
namespace tl {
namespace distributed_metadata {
constexpr int kAbiVersion = 1;
constexpr int kRequirementsVersion = 4;
constexpr int kCapacity = 1024;
constexpr int kVersion = 0;
constexpr int kBackend = 1;
constexpr int kRank = 2;
constexpr int kWorldSize = 3;
constexpr int kPayloadSize = 4;
constexpr int kStatus = 5;
constexpr int kStatusDetail = 6;
constexpr int kPayload = 7;
constexpr int kStatusWords = kPayload - kStatus;
constexpr int kSuccess = 0;
constexpr int kInvalidPeer = 1;
constexpr int kIncompatibleVersion = 2;
constexpr int kInvalidEnvelope = 3;
constexpr int kBackendMismatch = 4;
constexpr int kInvalidPayload = 5;
} // namespace distributed_metadata
} // namespace tl
#endif // TILELANG_HCU_DISTRIBUTED_METADATA_H_
