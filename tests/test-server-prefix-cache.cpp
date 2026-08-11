#include "server-prefix-cache.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <process.h>
static int test_pid() { return _getpid(); }
#else
#  include <unistd.h>
static int test_pid() { return getpid(); }
#endif

namespace fs = std::filesystem;
using namespace server_prefix_cache;

static Key make_key(uint8_t seed) {
    Key key;
    for (size_t i = 0; i < key.bytes.size(); ++i) {
        key.bytes[i] = uint8_t(seed + i);
    }
    return key;
}

static Artifact make_artifact(const Key & signature, ArtifactKind kind, const Key & key, size_t payload_size = 16) {
    Artifact artifact;
    artifact.kind = kind;
    artifact.signature = signature;
    artifact.key = key;
    artifact.position_start = 0;
    artifact.position_end = 4;
    artifact.token_count = 4;
    artifact.payload.resize(payload_size);
    for (size_t i = 0; i < artifact.payload.size(); ++i) {
        artifact.payload[i] = uint8_t(i ^ 0x5a);
    }
    return artifact;
}

static bool flip_byte(const fs::path & path, std::streamoff offset) {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!file) {
        return false;
    }
    file.seekg(offset);
    char value = 0;
    file.read(&value, 1);
    if (!file) {
        return false;
    }
    value ^= char(0x01);
    file.seekp(offset);
    file.write(&value, 1);
    return bool(file);
}

int main() {
    const fs::path root = fs::temp_directory_path() /
                          ("llama-prefix-cache-test-" + std::to_string((long long) test_pid()));
    std::error_code ec;
    fs::remove_all(root, ec);

    const Key signature = make_key(0x11);
    const Key key = make_key(0x91);
    const fs::path disk_root = root / "disk";

    Options options;
    options.root = disk_root;
    options.signature = signature;
    options.hot_bytes = 512;
    options.disk_bytes = 1 << 20;
    options.pending_write_bytes = 1 << 20;
    options.pending_write_items = 8;
    options.max_artifact_bytes = 1 << 16;

    auto store = PrefixCacheStore::open(options);
    if (!store) {
        return 1;
    }

    Artifact artifact = make_artifact(signature, ArtifactKind::ATTENTION, key);
    if (!store->publish(artifact)) {
        return 1;
    }
    auto hit = store->lookup(ArtifactKind::ATTENTION, key);
    if (!hit || hit->payload != artifact.payload || hit->position_end != artifact.position_end) {
        return 1;
    }

    // A duplicate publication must not replace a valid immutable artifact.
    Artifact duplicate = artifact;
    duplicate.payload.assign(artifact.payload.size(), 0xa5);
    if (!store->publish(duplicate)) {
        return 1;
    }
    hit = store->lookup(ArtifactKind::ATTENTION, key);
    if (!hit || hit->payload != artifact.payload) {
        return 1;
    }

    // Two independent writers race for one immutable final path. Exactly one
    // complete artifact must win; neither writer may expose a temporary file.
    const Key raced_key = make_key(0xa1);
    Artifact raced = make_artifact(signature, ArtifactKind::RECURRENT, raced_key);
    auto contender_a = PrefixCacheStore::open(options);
    auto contender_b = PrefixCacheStore::open(options);
    if (!contender_a || !contender_b) {
        return 1;
    }
    bool published_a = false;
    bool published_b = false;
    std::thread writer_a([&] { published_a = contender_a->publish(raced); });
    std::thread writer_b([&] { published_b = contender_b->publish(raced); });
    writer_a.join();
    writer_b.join();
    if (!published_a || !published_b || !contender_a->lookup(ArtifactKind::RECURRENT, raced_key)) {
        return 1;
    }
    contender_a->shutdown(std::chrono::milliseconds(1000));
    contender_b->shutdown(std::chrono::milliseconds(1000));

    // Hot entries are independently bounded and survive a disk clear only in RAM.
    if (!store->put_hot(artifact) || !store->lookup(ArtifactKind::ATTENTION, key)) {
        return 1;
    }
    store->clear_hot();
    if (!store->lookup(ArtifactKind::ATTENTION, key)) {
        return 1;
    }

    const std::string artifact_hex = key.hex();
    const fs::path artifact_file = store->signature_directory() / "attention" /
                                   artifact_hex.substr(0, 2) / artifact_hex.substr(2, 2) /
                                   (artifact_hex + ".bin");
    if (!fs::is_regular_file(artifact_file, ec)) {
        return 1;
    }
    store->shutdown(std::chrono::milliseconds(1000));

    // Restart discovers the immutable file without loading its payload first.
    store = PrefixCacheStore::open(options);
    if (!store || !store->lookup(ArtifactKind::ATTENTION, key)) {
        return 1;
    }
    store->shutdown(std::chrono::milliseconds(1000));

    // A checksum failure is a cache miss, never a partial restore.
    if (!flip_byte(artifact_file, 184)) {
        return 1;
    }
    store = PrefixCacheStore::open(options);
    if (!store || store->lookup(ArtifactKind::ATTENTION, key)) {
        return 1;
    }
    store->shutdown(std::chrono::milliseconds(1000));

    // Pending writes have hard item and byte limits.
    Options bounded = options;
    bounded.root = root / "bounded";
    bounded.pending_write_bytes = 4;
    bounded.pending_write_items = 1;
    bounded.hot_bytes = 4;
    bounded.max_artifact_bytes = 1 << 16;
    auto limited = PrefixCacheStore::open(bounded);
    if (!limited || limited->enqueue(artifact) || limited->put_hot(artifact)) {
        return 1;
    }
    limited->shutdown(std::chrono::milliseconds(100));

    // Reservation happens before payload allocation. Abandonment releases all
    // staging accounting, and commit transfers the same allocation to the
    // bounded writer (with HOT remaining optional).
    Options adaptive = options;
    adaptive.root = root / "adaptive";
    adaptive.hot_bytes = 0;
    adaptive.pending_write_bytes = 0;
    adaptive.pending_write_items = 2;
    auto adaptive_store = PrefixCacheStore::open(adaptive);
    if (!adaptive_store) {
        return 1;
    }
    {
        auto abandoned = adaptive_store->reserve(ArtifactKind::ATTENTION, 32);
        if (!abandoned || abandoned->reserved_hot() || abandoned->payload_size() != 32 ||
            adaptive_store->stats().pending_write_bytes != 32 + 184) {
            return 1;
        }
    }
    if (adaptive_store->stats().pending_write_bytes != 0) {
        return 1;
    }
    auto reserved = adaptive_store->reserve(ArtifactKind::ATTENTION, 32);
    if (!reserved) {
        return 1;
    }
    for (size_t i = 0; i < reserved->payload_size(); ++i) {
        reserved->payload_data()[i] = uint8_t(i);
    }
    Artifact reserved_metadata = make_artifact(signature, ArtifactKind::ATTENTION, make_key(0xb1), 0);
    if (!reserved->commit(reserved_metadata) ||
        !adaptive_store->flush(std::chrono::milliseconds(1000)) ||
        !adaptive_store->lookup(ArtifactKind::ATTENTION, reserved_metadata.key)) {
        return 1;
    }
    // The first recurrent size extends the derived hard capacity exactly once.
    auto recurrent_reservation = adaptive_store->reserve(ArtifactKind::RECURRENT, 48);
    if (!recurrent_reservation || adaptive_store->stats().pending_write_capacity != 32 + 48 + 2 * 184) {
        return 1;
    }
    recurrent_reservation.reset(); // RAII cancellation
    adaptive_store->shutdown(std::chrono::milliseconds(1000));

    // Full payload reads/checksums run outside the metadata mutex. If lookup
    // held that mutex, stats could never observe disk_io_inflight and reserve
    // would serialize behind the read.
    Options concurrent_io = options;
    concurrent_io.root = root / "concurrent-io";
    concurrent_io.hot_bytes = 0;
    concurrent_io.disk_bytes = 64ull << 20;
    concurrent_io.max_artifact_bytes = 16ull << 20;
    concurrent_io.pending_write_bytes = 16ull << 20;
    auto io_store = PrefixCacheStore::open(concurrent_io);
    Artifact large = make_artifact(signature, ArtifactKind::ATTENTION, make_key(0xb9), 8ull << 20);
    if (!io_store || !io_store->publish(large)) {
        return 1;
    }
    std::atomic<int> readers_done{0};
    auto read_repeatedly = [&] {
        for (int i = 0; i < 4; ++i) {
            if (!io_store->lookup(large.kind, large.key)) break;
        }
        readers_done.fetch_add(1);
    };
    std::thread disk_reader_a(read_repeatedly);
    std::thread disk_reader_b(read_repeatedly);
    bool overlapped = false;
    while (readers_done.load() != 2) {
        if (io_store->stats().disk_io_inflight >= 2) {
            auto concurrent_reservation = io_store->reserve(ArtifactKind::RECURRENT, 64, false);
            overlapped = concurrent_reservation.has_value();
            break;
        }
        std::this_thread::yield();
    }
    disk_reader_a.join();
    disk_reader_b.join();
    if (!overlapped) {
        return 1;
    }

    Artifact large_write = make_artifact(signature, ArtifactKind::RECURRENT, make_key(0xba), 8ull << 20);
    std::atomic<bool> publish_done{false};
    bool publish_ok = false;
    std::thread disk_writer([&] {
        publish_ok = io_store->publish(large_write);
        publish_done.store(true);
    });
    bool reserve_during_publish = false;
    while (!publish_done.load()) {
        if (io_store->stats().disk_io_inflight != 0) {
            auto concurrent_reservation = io_store->reserve(ArtifactKind::ATTENTION, 64, false);
            reserve_during_publish = concurrent_reservation.has_value();
            break;
        }
        std::this_thread::yield();
    }
    disk_writer.join();
    if (!publish_ok || !reserve_during_publish) {
        return 1;
    }
    io_store->shutdown(std::chrono::milliseconds(1000));

    // Oversized disk artifacts are rejected before publication.
    Options tiny_disk = options;
    tiny_disk.root = root / "tiny-disk";
    tiny_disk.disk_bytes = 4;
    tiny_disk.max_artifact_bytes = 1 << 16;
    auto disk_limited = PrefixCacheStore::open(tiny_disk);
    if (!disk_limited || disk_limited->publish(artifact)) {
        return 1;
    }
    disk_limited->shutdown(std::chrono::milliseconds(100));

    // Once shutdown begins, staging and writer admission are closed. This is
    // the storage-side contract a deferred descriptor path depends on when a
    // slot is reused or the server is stopping.
    Options shutdown_options = options;
    shutdown_options.root = root / "shutdown-root";
    shutdown_options.pending_write_items = 1;
    shutdown_options.pending_write_bytes = 1 << 20;
    auto shutdown_store = PrefixCacheStore::open(shutdown_options);
    if (!shutdown_store) {
        return 1;
    }
    auto shutdown_reservation = shutdown_store->reserve(ArtifactKind::ATTENTION, 32);
    if (!shutdown_reservation) {
        return 1;
    }
    shutdown_store->shutdown(std::chrono::milliseconds(1000));
    Artifact shutdown_artifact = make_artifact(signature, ArtifactKind::ATTENTION, make_key(0xcb), 0);
    if (shutdown_reservation->commit(shutdown_artifact) ||
        shutdown_store->reserve(ArtifactKind::ATTENTION, 32) ||
        shutdown_store->enqueue(artifact)) {
        return 1;
    }

    // A cache-owned symlink is rejected rather than followed.
#if !defined(_WIN32)
    const fs::path symlink_root = root / "symlink-root";
    fs::create_directory_symlink(disk_root, symlink_root, ec);
    if (ec) {
        return 1;
    }
    Options symlink_options = options;
    symlink_options.root = symlink_root;
    if (PrefixCacheStore::open(symlink_options)) {
        return 1;
    }

    // Replacing a validated owned directory cannot redirect later operations:
    // the store keeps directory descriptors and never resolves this path
    // again. Publication remains in the original inode.
    Options fd_options = options;
    fd_options.root = root / "fd-root";
    auto fd_store = PrefixCacheStore::open(fd_options);
    if (!fd_store) {
        return 1;
    }
    const fs::path fd_signature = fd_store->signature_directory();
    const fs::path held_attention = fd_signature / "attention-held";
    const fs::path outside = root / "outside";
    fs::create_directories(outside, ec);
    fs::rename(fd_signature / "attention", held_attention, ec);
    if (ec) {
        return 1;
    }
    fs::create_directory_symlink(outside, fd_signature / "attention", ec);
    if (ec) {
        return 1;
    }
    Artifact fd_artifact = make_artifact(signature, ArtifactKind::ATTENTION, make_key(0xc1));
    if (!fd_store->publish(fd_artifact) || !fd_store->lookup(fd_artifact.kind, fd_artifact.key)) {
        return 1;
    }
    const std::string fd_hex = fd_artifact.key.hex();
    if (!fs::is_regular_file(held_attention / fd_hex.substr(0, 2) / fd_hex.substr(2, 2) /
                             (fd_hex + ".bin"), ec) ||
        fs::exists(outside / fd_hex.substr(0, 2), ec)) {
        return 1;
    }
    std::string clear_error;
    const bool cleared = fd_store->clear_disk(&clear_error);
    if (!cleared ||
        fs::exists(held_attention / fd_hex.substr(0, 2) / fd_hex.substr(2, 2) /
                   (fd_hex + ".bin"), ec) ||
        fs::exists(outside / fd_hex.substr(0, 2), ec)) {
        return 1;
    }
    fd_store->shutdown(std::chrono::milliseconds(1000));

    // A shard replaced by a symlink after startup is refused at the openat
    // boundary; publication cannot escape to the symlink target.
    Options shard_options = options;
    shard_options.root = root / "shard-root";
    auto shard_store = PrefixCacheStore::open(shard_options);
    Artifact first_shard = make_artifact(signature, ArtifactKind::ATTENTION, make_key(0xd1));
    if (!shard_store || !shard_store->publish(first_shard)) {
        return 1;
    }
    const std::string shard_hex = first_shard.key.hex();
    const fs::path shard_path = shard_store->signature_directory() / "attention" / shard_hex.substr(0, 2);
    const fs::path held_shard = shard_path.string() + "-held";
    fs::rename(shard_path, held_shard, ec);
    if (ec) {
        return 1;
    }
    fs::create_directory_symlink(outside, shard_path, ec);
    if (ec) {
        return 1;
    }
    Key escaped_key = first_shard.key;
    escaped_key.bytes.back() ^= 1;
    Artifact escaped = make_artifact(signature, ArtifactKind::ATTENTION, escaped_key);
    if (shard_store->publish(escaped) || fs::exists(outside / shard_hex.substr(2, 2), ec)) {
        return 1;
    }
    shard_store->shutdown(std::chrono::milliseconds(1000));
#endif

    fs::remove_all(root, ec);
    return 0;
}
