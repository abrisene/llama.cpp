#pragma once

// Bounded, immutable storage for the server's persistent prefix-cache
// artifacts.  The cache deliberately has no dependency on llama_context or
// server-context: the state/range API can hand it opaque serialized bytes.

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace server_prefix_cache {

constexpr uint32_t FORMAT_VERSION = 1;

struct Key {
    std::array<uint8_t, 32> bytes{};

    bool operator==(const Key & other) const noexcept { return bytes == other.bytes; }
    bool operator!=(const Key & other) const noexcept { return !(*this == other); }

    std::string hex() const;
    static std::optional<Key> from_hex(const std::string & value);
};

struct KeyHash {
    size_t operator()(const Key & key) const noexcept;
};

enum class ArtifactKind : uint8_t {
    ATTENTION = 1,
    RECURRENT = 2,
};

struct ArtifactId {
    ArtifactKind kind = ArtifactKind::ATTENTION;
    Key key;

    bool operator==(const ArtifactId & other) const noexcept {
        return kind == other.kind && key == other.key;
    }
};

struct ArtifactIdHash {
    size_t operator()(const ArtifactId & id) const noexcept;
};

// The key encoding is intentionally explicit and independent of C++ object
// layout.  `parent` is omitted for the first block (pass std::nullopt).
Key chain_key(const Key & signature,
              ArtifactKind kind,
              const std::optional<Key> & parent,
              const std::vector<int32_t> & positions,
              const std::vector<int32_t> & tokens);

// Hash an already canonicalized signature.  The returned value is also the
// namespace name under the store root.
Key signature_key(const std::vector<uint8_t> & canonical_signature);

struct Artifact {
    ArtifactKind kind = ArtifactKind::ATTENTION;
    Key signature;
    Key key;
    Key parent{};
    bool has_parent = false;
    int64_t position_start = 0;
    int64_t position_end = 0;
    uint64_t token_count = 0;
    std::vector<uint8_t> payload;
    uint64_t created_ns = 0;
};

struct CacheStats {
    uint64_t lookups = 0;
    uint64_t attention_lookups = 0;
    uint64_t recurrent_lookups = 0;
    uint64_t hot_hits = 0;
    uint64_t attention_hot_hits = 0;
    uint64_t recurrent_hot_hits = 0;
    uint64_t disk_hits = 0;
    uint64_t attention_disk_hits = 0;
    uint64_t recurrent_disk_hits = 0;
    uint64_t misses = 0;
    uint64_t attention_misses = 0;
    uint64_t recurrent_misses = 0;
    uint64_t corrupt = 0;
    uint64_t incompatible = 0;
    uint64_t published = 0;
    uint64_t duplicate_publications = 0;
    uint64_t enqueue_skips = 0;
    uint64_t restore_failures = 0;
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    uint64_t hot_bytes = 0;
    uint64_t hot_artifacts = 0;
    uint64_t disk_bytes = 0; // Required, nonzero hard limit.
    uint64_t disk_artifacts = 0;
    uint64_t disk_io_inflight = 0;
    uint64_t pending_write_bytes = 0;
    uint64_t pending_write_peak_bytes = 0;
    uint64_t pending_write_items = 0;
    uint64_t pending_write_capacity = 0;
    uint64_t reserved_hot_bytes = 0;
};

struct Options {
    std::filesystem::path root;
    Key signature;
    uint64_t hot_bytes = 0;
    uint64_t disk_bytes = 0;
    uint64_t disk_files = 0; // 0 means no file-count limit.
    // Zero derives a hard limit from the first successfully reserved
    // attention block plus the first recurrent sidecar.  Each first artifact
    // is still bounded by max_artifact_bytes and disk_bytes.
    uint64_t pending_write_bytes = 0;
    uint64_t pending_write_items = 0;
    // Zero derives the per-artifact cap from disk_bytes.
    uint64_t max_artifact_bytes = 0;
    std::chrono::milliseconds enqueue_wait{ 5 };
};

class PrefixCacheStore {
public:
    class Reservation {
    public:
        Reservation();
        ~Reservation();
        Reservation(Reservation &&) noexcept;
        Reservation & operator=(Reservation &&) noexcept;
        Reservation(const Reservation &) = delete;
        Reservation & operator=(const Reservation &) = delete;

        uint8_t * payload_data();
        size_t payload_size() const;
        bool reserved_hot() const;

        // Metadata must describe the reserved kind and have an empty payload.
        // Commit fills the immutable envelope in-place and transfers the same
        // allocation to the hot tier (when reserved) and writer queue.
        bool commit(const Artifact & metadata);

    private:
        struct State;
        explicit Reservation(std::unique_ptr<State> state);
        std::unique_ptr<State> state_;
        friend class PrefixCacheStore;
    };

    // Opens (and, when needed, creates) root/v1/<signature>/ and performs a
    // bounded header-only startup scan.  A symlink anywhere in the owned
    // layout is rejected.
    static std::unique_ptr<PrefixCacheStore> open(const Options & options, std::string * error = nullptr);

    ~PrefixCacheStore();

    PrefixCacheStore(const PrefixCacheStore &) = delete;
    PrefixCacheStore & operator=(const PrefixCacheStore &) = delete;

    // Lookup validates the complete immutable artifact.  The returned value
    // owns its bytes, so eviction cannot invalidate an in-progress restore.
    std::optional<Artifact> lookup(ArtifactKind kind, const Key & key);

    // Reserve queue bytes before allocating or serializing model state.
    // WRITE is mandatory. HOT is best-effort, so a disabled/full hot tier
    // does not prevent disk capture. The returned reservation owns a
    // pre-sized payload region and releases all accounting if abandoned.
    std::optional<Reservation> reserve(
            ArtifactKind kind,
            uint64_t payload_bytes,
            bool want_hot = true,
            std::chrono::milliseconds timeout = std::chrono::milliseconds(-1));

    // Insert into the immutable hot tier.  Existing valid bytes win.
    bool put_hot(const Artifact & artifact);
    void clear_hot();

    // Synchronous publication, useful for controlled shutdown and tests.
    bool publish(const Artifact & artifact);

    // Queue an immutable serialized artifact for the bounded writer.  The
    // method returns false when item/byte limits cannot be reserved.
    bool enqueue(const Artifact & artifact);

    // Wait for queued writes.  shutdown() stops accepting new work, drains up
    // to the supplied deadline, then safely discards the remainder.
    bool flush(std::chrono::milliseconds timeout);
    void shutdown(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));

    // Remove only this validated signature's disk tier.  Readers/writers are
    // serialized and symlinks are never followed.
    bool clear_disk(std::string * error = nullptr);

    CacheStats stats() const;
    std::filesystem::path signature_directory() const;

private:
    PrefixCacheStore(const Options & options);
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace server_prefix_cache
