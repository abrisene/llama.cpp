#include "server-prefix-cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <list>
#include <mutex>
#include <random>
#include <system_error>
#include <thread>
#include <unordered_map>

#if defined(_WIN32)
#  include <io.h>
#else
#  include <dirent.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

namespace server_prefix_cache {
namespace {

using Bytes = std::vector<uint8_t>;
// magic/version/kind/flags/size (16) + signature/key/parent (96) +
// positions/token count/payload size/timestamp (40) + checksum (32).
constexpr size_t HEADER_SIZE = 184;
constexpr char MAGIC[8] = { 'L', 'L', 'M', 'P', 'R', 'F', 'X', 1 };
constexpr char KEY_DOMAIN[] = "llama.cpp.prefix-cache.v1";

uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

class Sha256 {
public:
    Sha256() {
        state_ = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    }

    void update(const uint8_t * data, size_t size) {
        total_ += size;
        while (size != 0) {
            const size_t take = std::min(size, block_.size() - used_);
            std::memcpy(block_.data() + used_, data, take);
            used_ += take;
            data += take;
            size -= take;
            if (used_ == block_.size()) {
                transform(block_.data());
                used_ = 0;
            }
        }
    }

    Key finish() {
        const uint64_t bit_count = total_ * 8;
        uint8_t one = 0x80;
        update(&one, 1);
        uint8_t zero = 0;
        while (used_ != 56) update(&zero, 1);
        uint8_t length[8];
        for (int i = 0; i < 8; ++i) length[i] = static_cast<uint8_t>(bit_count >> (56 - 8 * i));
        update(length, sizeof(length));

        Key out;
        for (size_t i = 0; i < state_.size(); ++i) {
            out.bytes[i * 4 + 0] = static_cast<uint8_t>(state_[i] >> 24);
            out.bytes[i * 4 + 1] = static_cast<uint8_t>(state_[i] >> 16);
            out.bytes[i * 4 + 2] = static_cast<uint8_t>(state_[i] >> 8);
            out.bytes[i * 4 + 3] = static_cast<uint8_t>(state_[i]);
        }
        return out;
    }

private:
    void transform(const uint8_t * data) {
        static constexpr uint32_t k[] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
            0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
            0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
            0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
            0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(data[i * 4]) << 24) | (uint32_t(data[i * 4 + 1]) << 16) |
                   (uint32_t(data[i * 4 + 2]) << 8) | uint32_t(data[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = h + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_{};
    std::array<uint8_t, 64> block_{};
    size_t used_ = 0;
    uint64_t total_ = 0;
};

void put_u16(Bytes & out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
}
void put_u32(Bytes & out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(value >> (i * 8)));
}
void put_u64(Bytes & out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(value >> (i * 8)));
}
void put_i64(Bytes & out, int64_t value) { put_u64(out, static_cast<uint64_t>(value)); }

bool take_u16(const Bytes & in, size_t & at, uint16_t & out) {
    if (at + 2 > in.size()) return false;
    out = uint16_t(in[at]) | (uint16_t(in[at + 1]) << 8); at += 2; return true;
}
bool take_u32(const Bytes & in, size_t & at, uint32_t & out) {
    if (at + 4 > in.size()) return false;
    out = uint32_t(in[at]) | (uint32_t(in[at + 1]) << 8) | (uint32_t(in[at + 2]) << 16) |
          (uint32_t(in[at + 3]) << 24); at += 4; return true;
}
bool take_u64(const Bytes & in, size_t & at, uint64_t & out) {
    if (at + 8 > in.size()) return false;
    out = 0; for (int i = 0; i < 8; ++i) out |= uint64_t(in[at + i]) << (i * 8); at += 8; return true;
}
bool take_i64(const Bytes & in, size_t & at, int64_t & out) {
    uint64_t value = 0; if (!take_u64(in, at, value)) return false; out = static_cast<int64_t>(value); return true;
}

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

uint64_t next_temp_nonce() {
    static std::atomic<uint64_t> nonce {
        now_ns() ^ (static_cast<uint64_t>(std::random_device{}()) << 32) ^ std::random_device{}()
    };
    return nonce.fetch_add(1, std::memory_order_relaxed);
}

bool valid_hex(const std::string & value, size_t n) {
    if (value.size() != n) return false;
    for (char c : value) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
    return true;
}

Bytes encode_header(const Artifact & input, const uint8_t * payload, size_t payload_size) {
    Artifact a = input;
    if (a.created_ns == 0) a.created_ns = now_ns();
    Bytes out;
    out.reserve(HEADER_SIZE);
    out.insert(out.end(), MAGIC, MAGIC + sizeof(MAGIC));
    put_u32(out, FORMAT_VERSION);
    out.push_back(static_cast<uint8_t>(a.kind));
    out.push_back(static_cast<uint8_t>(a.has_parent ? 1 : 0));
    put_u16(out, HEADER_SIZE);
    out.insert(out.end(), a.signature.bytes.begin(), a.signature.bytes.end());
    out.insert(out.end(), a.key.bytes.begin(), a.key.bytes.end());
    if (a.has_parent) out.insert(out.end(), a.parent.bytes.begin(), a.parent.bytes.end());
    else out.insert(out.end(), 32, 0);
    put_i64(out, a.position_start); put_i64(out, a.position_end);
    put_u64(out, a.token_count); put_u64(out, payload_size); put_u64(out, a.created_ns);
    Sha256 sha; sha.update(payload, payload_size);
    const Key checksum = sha.finish();
    out.insert(out.end(), checksum.bytes.begin(), checksum.bytes.end());
    return out;
}

Bytes encode_artifact(const Artifact & input) {
    Bytes out = encode_header(input, input.payload.data(), input.payload.size());
    out.reserve(HEADER_SIZE + input.payload.size());
    out.insert(out.end(), input.payload.begin(), input.payload.end());
    return out;
}

bool finalize_reserved_artifact(Bytes & bytes, const Artifact & metadata) {
    if (!metadata.payload.empty() || bytes.size() < HEADER_SIZE) return false;
    const size_t payload_size = bytes.size() - HEADER_SIZE;
    const Bytes header = encode_header(metadata, bytes.data() + HEADER_SIZE, payload_size);
    if (header.size() != HEADER_SIZE) return false;
    std::copy(header.begin(), header.end(), bytes.begin());
    return true;
}

bool decode_artifact(const Bytes & bytes, uint64_t max_bytes, Artifact & out) {
    if (bytes.size() < HEADER_SIZE) return false;
    if (!std::equal(MAGIC, MAGIC + sizeof(MAGIC), bytes.begin())) return false;
    size_t at = sizeof(MAGIC);
    uint32_t version = 0; uint16_t header_size = 0;
    if (!take_u32(bytes, at, version)) return false;
    if (version != FORMAT_VERSION || at + 2 > bytes.size()) return false;
    const uint8_t kind = bytes[at++];
    const uint8_t flags = bytes[at++];
    if (kind != uint8_t(ArtifactKind::ATTENTION) && kind != uint8_t(ArtifactKind::RECURRENT)) return false;
    if ((flags & ~uint8_t(1)) != 0 || !take_u16(bytes, at, header_size) || header_size != HEADER_SIZE) return false;
    if (at + 96 > bytes.size()) return false;
    std::memcpy(out.signature.bytes.data(), bytes.data() + at, 32); at += 32;
    std::memcpy(out.key.bytes.data(), bytes.data() + at, 32); at += 32;
    std::memcpy(out.parent.bytes.data(), bytes.data() + at, 32); at += 32;
    out.kind = static_cast<ArtifactKind>(kind); out.has_parent = (flags & 1) != 0;
    if (!take_i64(bytes, at, out.position_start) || !take_i64(bytes, at, out.position_end) ||
        !take_u64(bytes, at, out.token_count)) return false;
    uint64_t payload_size = 0, created = 0;
    if (!take_u64(bytes, at, payload_size) || !take_u64(bytes, at, created) || at + 32 > bytes.size()) return false;
    if (payload_size > max_bytes || payload_size != bytes.size() - HEADER_SIZE || at + 32 != HEADER_SIZE) return false;
    Key checksum; std::memcpy(checksum.bytes.data(), bytes.data() + at, 32); at += 32;
    if (out.position_start < 0 || out.position_end < out.position_start) return false;
    out.created_ns = created;
    out.payload.assign(bytes.begin() + HEADER_SIZE, bytes.end());
    Sha256 sha; sha.update(out.payload.data(), out.payload.size());
    if (sha.finish() != checksum) return false;
    return true;
}

// Header-only validation used during startup.  It deliberately does not read
// or checksum payload bytes: scan must remain bounded even when a stale cache
// contains large artifacts.  The complete checksum is verified by lookup.
bool decode_header(const Bytes & bytes, uint64_t file_size, uint64_t max_bytes, Artifact & out) {
    if (bytes.size() != HEADER_SIZE || file_size < HEADER_SIZE) return false;
    if (!std::equal(MAGIC, MAGIC + sizeof(MAGIC), bytes.begin())) return false;
    size_t at = sizeof(MAGIC); uint32_t version = 0; uint16_t header_size = 0;
    if (!take_u32(bytes, at, version) || version != FORMAT_VERSION || at + 2 > bytes.size()) return false;
    const uint8_t kind = bytes[at++], flags = bytes[at++];
    if ((kind != uint8_t(ArtifactKind::ATTENTION) && kind != uint8_t(ArtifactKind::RECURRENT)) || (flags & ~uint8_t(1)) != 0 ||
        !take_u16(bytes, at, header_size) || header_size != HEADER_SIZE || at + 96 > bytes.size()) return false;
    std::memcpy(out.signature.bytes.data(), bytes.data() + at, 32); at += 32;
    std::memcpy(out.key.bytes.data(), bytes.data() + at, 32); at += 32;
    std::memcpy(out.parent.bytes.data(), bytes.data() + at, 32); at += 32;
    out.kind = static_cast<ArtifactKind>(kind); out.has_parent = (flags & 1) != 0;
    if (!take_i64(bytes, at, out.position_start) || !take_i64(bytes, at, out.position_end) || !take_u64(bytes, at, out.token_count)) return false;
    uint64_t payload_size = 0;
    if (!take_u64(bytes, at, payload_size) || !take_u64(bytes, at, out.created_ns) || at + 32 != HEADER_SIZE) return false;
    return payload_size <= max_bytes && file_size == HEADER_SIZE + payload_size && out.position_start >= 0 && out.position_end >= out.position_start;
}

#if defined(_WIN32)
bool read_header(const std::filesystem::path & path, Bytes & bytes, uint64_t & file_size) {
    std::error_code ec; const auto status = std::filesystem::symlink_status(path, ec);
    if (ec || status.type() == std::filesystem::file_type::symlink || !std::filesystem::is_regular_file(status)) return false;
    file_size = std::filesystem::file_size(path, ec); if (ec || file_size < HEADER_SIZE) return false;
    std::ifstream input(path, std::ios::binary); if (!input) return false;
    bytes.resize(HEADER_SIZE); input.read(reinterpret_cast<char *>(bytes.data()), HEADER_SIZE);
    return input.gcount() == static_cast<std::streamsize>(HEADER_SIZE);
}
#endif

#if defined(_WIN32)
bool read_file(const std::filesystem::path & path, Bytes & bytes) {
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    if (ec || status.type() == std::filesystem::file_type::symlink || !std::filesystem::is_regular_file(status)) return false;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size > (1ull << 31)) return false;
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    bytes.resize(static_cast<size_t>(size));
    if (size != 0) input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));
    return input.good() || input.eof();
}
#endif

bool ensure_directory(const std::filesystem::path & path, std::string * error) {
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    if (!ec && status.type() == std::filesystem::file_type::symlink) {
        if (error) *error = "cache path is a symlink: " + path.string(); return false;
    }
    if (!ec && std::filesystem::is_directory(status)) return true;
    if (!ec && std::filesystem::exists(status)) {
        if (error) *error = "cache path is not a directory: " + path.string(); return false;
    }
    ec.clear();
    if (!std::filesystem::create_directories(path, ec) && ec) {
        if (error) *error = "cannot create cache directory " + path.string() + ": " + ec.message(); return false;
    }
    return true;
}

#if !defined(_WIN32)
bool write_all_fd(int fd, const uint8_t * data, size_t size) {
    while (size != 0) {
        const ssize_t n = ::write(fd, data, size);
        if (n <= 0) return false;
        data += n; size -= static_cast<size_t>(n);
    }
    return true;
}

bool read_all_fd(int fd, uint8_t * data, size_t size) {
    size_t offset = 0;
    while (offset != size) {
        const ssize_t n = ::pread(fd, data + offset, size - offset, static_cast<off_t>(offset));
        if (n <= 0) return false;
        offset += static_cast<size_t>(n);
    }
    return true;
}

bool regular_file_size(int fd, uint64_t & size) {
    struct stat st {};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) return false;
    size = static_cast<uint64_t>(st.st_size);
    return true;
}

bool ensure_dir_at(int parent_fd, const std::string & name, int & result, std::string * error) {
    if (::mkdirat(parent_fd, name.c_str(), 0700) != 0 && errno != EEXIST) {
        if (error) *error = "cannot create cache directory '" + name + "': " + std::strerror(errno);
        return false;
    }
    result = ::openat(parent_fd, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (result < 0) {
        if (error) *error = "cache directory is unsafe or inaccessible '" + name + "': " + std::strerror(errno);
        return false;
    }
    struct stat st {};
    if (::fstat(result, &st) != 0 || !S_ISDIR(st.st_mode)) {
        if (error) *error = "cache path is not a directory: " + name;
        ::close(result);
        result = -1;
        return false;
    }
    return true;
}

bool list_names_at(int dir_fd, std::vector<std::string> & names) {
    // dup() shares the directory offset with the retained descriptor. Open
    // "." instead so every scan has an independent file description.
    const int copy = ::openat(dir_fd, ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (copy < 0) return false;
    DIR * dir = ::fdopendir(copy);
    if (!dir) {
        ::close(copy);
        return false;
    }
    errno = 0;
    while (dirent * entry = ::readdir(dir)) {
        const std::string name = entry->d_name;
        if (name != "." && name != "..") names.push_back(name);
    }
    const bool ok = errno == 0;
    ::closedir(dir);
    return ok;
}
#endif

} // namespace

std::string Key::hex() const {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out; out.resize(bytes.size() * 2);
    for (size_t i = 0; i < bytes.size(); ++i) { out[i * 2] = digits[bytes[i] >> 4]; out[i * 2 + 1] = digits[bytes[i] & 15]; }
    return out;
}

std::optional<Key> Key::from_hex(const std::string & value) {
    if (!valid_hex(value, 64)) return std::nullopt;
    Key out;
    for (size_t i = 0; i < out.bytes.size(); ++i) {
        auto nibble = [](char c) -> uint8_t { return c <= '9' ? uint8_t(c - '0') : uint8_t((c | 32) - 'a' + 10); };
        out.bytes[i] = static_cast<uint8_t>((nibble(value[i * 2]) << 4) | nibble(value[i * 2 + 1]));
    }
    return out;
}

size_t KeyHash::operator()(const Key & key) const noexcept {
    uint64_t hash = 1469598103934665603ull;
    for (uint8_t b : key.bytes) { hash ^= b; hash *= 1099511628211ull; }
    return static_cast<size_t>(hash);
}

size_t ArtifactIdHash::operator()(const ArtifactId & id) const noexcept {
    return KeyHash{}(id.key) ^ (static_cast<size_t>(id.kind) * 0x9e3779b9u);
}

Key signature_key(const std::vector<uint8_t> & canonical_signature) {
    Sha256 sha; sha.update(canonical_signature.data(), canonical_signature.size()); return sha.finish();
}

Key chain_key(const Key & signature, ArtifactKind kind, const std::optional<Key> & parent,
              const std::vector<int32_t> & positions, const std::vector<int32_t> & tokens) {
    Key out;
    Sha256 sha;
    sha.update(reinterpret_cast<const uint8_t *>(KEY_DOMAIN), sizeof(KEY_DOMAIN) - 1);
    const uint8_t kind_byte = static_cast<uint8_t>(kind); sha.update(&kind_byte, 1);
    sha.update(signature.bytes.data(), signature.bytes.size());
    if (parent) sha.update(parent->bytes.data(), parent->bytes.size());
    if (positions.size() != tokens.size()) return out;
    for (size_t i = 0; i < positions.size(); ++i) {
        uint8_t p[4], t[4];
        const uint32_t pv = static_cast<uint32_t>(positions[i]), tv = static_cast<uint32_t>(tokens[i]);
        for (int j = 0; j < 4; ++j) { p[j] = uint8_t(pv >> (j * 8)); t[j] = uint8_t(tv >> (j * 8)); }
        sha.update(p, 4); sha.update(t, 4);
    }
    return sha.finish();
}

struct PrefixCacheStore::Impl {
    struct DiskEntry { std::filesystem::path path; uint64_t size = 0; uint64_t last_access = 0; uint32_t pins = 0; ArtifactId id; };
    struct HotEntry { std::shared_ptr<const Bytes> bytes; uint64_t size = 0; uint32_t pins = 0; std::list<ArtifactId>::iterator lru; };
    struct Job { ArtifactId id; std::shared_ptr<const Bytes> bytes; };

    Options options;
    std::filesystem::path root, signature_dir, attention_dir, recurrent_dir, quarantine_dir;
#if !defined(_WIN32)
    int root_fd = -1, v1_fd = -1, signature_fd = -1;
    int attention_fd = -1, recurrent_fd = -1, quarantine_fd = -1;
#endif
    mutable std::mutex mutex;
    std::condition_variable cv, done_cv, disk_cv;
    std::unordered_map<ArtifactId, DiskEntry, ArtifactIdHash> disk;
    std::unordered_map<ArtifactId, HotEntry, ArtifactIdHash> hot;
    std::list<ArtifactId> hot_lru;
    std::deque<Job> jobs;
    CacheStats counters;
    uint64_t hot_used = 0, disk_used = 0, pending_bytes = 0, in_flight = 0;
    uint64_t reserved_write_bytes = 0, reserved_hot_bytes = 0, reserved_items = 0;
    uint64_t adaptive_attention_bytes = 0, adaptive_recurrent_bytes = 0;
    uint64_t write_capacity = 0;
    uint64_t disk_reserved_bytes = 0, disk_reserved_files = 0, disk_io_inflight = 0;
    bool accepting = true, stopping = false, abort = false, clearing = false;
    std::thread writer;

    explicit Impl(const Options & opts) : options(opts) {}
    ~Impl() {
#if !defined(_WIN32)
        for (int * fd : {&quarantine_fd, &recurrent_fd, &attention_fd, &signature_fd, &v1_fd, &root_fd}) {
            if (*fd >= 0) {
                ::close(*fd);
                *fd = -1;
            }
        }
#endif
    }

    bool initialize(std::string * error) {
        if (options.root.empty()) { if (error) *error = "persistent cache root is empty"; return false; }
        if (options.disk_bytes == 0) {
            if (error) *error = "persistent cache requires a nonzero disk byte limit";
            return false;
        }
        if (options.max_artifact_bytes == 0) {
            options.max_artifact_bytes = options.disk_bytes > HEADER_SIZE
                ? options.disk_bytes - HEADER_SIZE
                : 0;
        }
        if (options.max_artifact_bytes == 0) {
            if (error) *error = "persistent cache requires a disk or artifact byte limit";
            return false;
        }
        if (options.pending_write_items == 0) {
            options.pending_write_items = 2;
        }
        write_capacity = options.pending_write_bytes;
        counters.pending_write_capacity = write_capacity;
        root = std::filesystem::absolute(options.root);
        if (!ensure_directory(root, error)) return false;
        const auto v1 = root / "v1";
        signature_dir = v1 / options.signature.hex();
        attention_dir = signature_dir / "attention";
        recurrent_dir = signature_dir / "recurrent";
        quarantine_dir = signature_dir / "quarantine";
#if defined(_WIN32)
        if (!ensure_directory(v1, error) || !ensure_directory(signature_dir, error)) return false;
        if (!ensure_directory(attention_dir, error) || !ensure_directory(recurrent_dir, error) || !ensure_directory(quarantine_dir, error)) return false;
#else
        root_fd = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (root_fd < 0) {
            if (error) *error = "cache root is unsafe or inaccessible: " + std::string(std::strerror(errno));
            return false;
        }
        if (!ensure_dir_at(root_fd, "v1", v1_fd, error) ||
            !ensure_dir_at(v1_fd, options.signature.hex(), signature_fd, error) ||
            !ensure_dir_at(signature_fd, "attention", attention_fd, error) ||
            !ensure_dir_at(signature_fd, "recurrent", recurrent_fd, error) ||
            !ensure_dir_at(signature_fd, "quarantine", quarantine_fd, error)) return false;
#endif
        if (!scan_kind(ArtifactKind::ATTENTION, attention_dir, error) || !scan_kind(ArtifactKind::RECURRENT, recurrent_dir, error)) return false;
        writer = std::thread([this] { writer_loop(); });
        return true;
    }

    std::filesystem::path artifact_path(ArtifactKind kind, const Key & key) const {
        const auto base = kind == ArtifactKind::ATTENTION ? attention_dir : recurrent_dir;
        const std::string h = key.hex();
        return base / h.substr(0, 2) / h.substr(2, 2) / (h + ".bin");
    }

    bool make_shards(ArtifactKind kind, const Key & key, std::string * error) {
#if defined(_WIN32)
        const auto base = kind == ArtifactKind::ATTENTION ? attention_dir : recurrent_dir;
        const std::string h = key.hex();
        return ensure_directory(base / h.substr(0, 2), error) && ensure_directory(base / h.substr(0, 2) / h.substr(2, 2), error);
#else
        const int base = kind == ArtifactKind::ATTENTION ? attention_fd : recurrent_fd;
        const std::string h = key.hex();
        int first = -1, second = -1;
        const bool ok = ensure_dir_at(base, h.substr(0, 2), first, error) &&
                        ensure_dir_at(first, h.substr(2, 2), second, error);
        if (second >= 0) ::close(second);
        if (first >= 0) ::close(first);
        return ok;
#endif
    }

#if !defined(_WIN32)
    int open_shard_fd(ArtifactKind kind, const Key & key, bool create, std::string * error = nullptr) const {
        const int base = kind == ArtifactKind::ATTENTION ? attention_fd : recurrent_fd;
        const std::string h = key.hex();
        int first = -1, second = -1;
        if (create) {
            if (!ensure_dir_at(base, h.substr(0, 2), first, error) ||
                !ensure_dir_at(first, h.substr(2, 2), second, error)) {
                if (first >= 0) ::close(first);
                if (second >= 0) ::close(second);
                return -1;
            }
        } else {
            first = ::openat(base, h.substr(0, 2).c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
            if (first < 0) return -1;
            second = ::openat(first, h.substr(2, 2).c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
            if (second < 0) {
                ::close(first);
                return -1;
            }
        }
        ::close(first);
        return second;
    }

    bool read_id(const ArtifactId & id, Bytes & bytes, bool header_only, uint64_t & file_size) const {
        const int shard = open_shard_fd(id.kind, id.key, false);
        if (shard < 0) return false;
        const std::string name = id.key.hex() + ".bin";
        const int fd = ::openat(shard, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        ::close(shard);
        if (fd < 0) return false;
        const bool sized = regular_file_size(fd, file_size);
        const uint64_t read_size = header_only ? HEADER_SIZE : file_size;
        const bool bounded = sized && file_size >= HEADER_SIZE &&
                             file_size <= HEADER_SIZE + options.max_artifact_bytes &&
                             read_size <= static_cast<uint64_t>(SIZE_MAX);
        if (bounded) {
            bytes.resize(static_cast<size_t>(read_size));
        }
        const bool ok = bounded && read_all_fd(fd, bytes.data(), bytes.size());
        ::close(fd);
        return ok;
    }

    bool unlink_id(const ArtifactId & id) const {
        const int shard = open_shard_fd(id.kind, id.key, false);
        if (shard < 0) return false;
        const std::string name = id.key.hex() + ".bin";
        const bool ok = ::unlinkat(shard, name.c_str(), 0) == 0 || errno == ENOENT;
        ::close(shard);
        return ok;
    }
#endif

    bool scan_kind(ArtifactKind kind, const std::filesystem::path & base, std::string * error) {
#if !defined(_WIN32)
        (void) base;
        const int base_fd = kind == ArtifactKind::ATTENTION ? attention_fd : recurrent_fd;
        std::vector<std::string> first_names;
        if (!list_names_at(base_fd, first_names)) {
            if (error) *error = "cannot scan cache directory";
            return false;
        }
        for (const std::string & d1 : first_names) {
            if (!valid_hex(d1, 2)) continue;
            const int first = ::openat(base_fd, d1.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
            if (first < 0) {
                if (error) *error = "unsafe cache shard: " + d1;
                return false;
            }
            std::vector<std::string> second_names;
            if (!list_names_at(first, second_names)) {
                ::close(first);
                if (error) *error = "cannot scan cache shard: " + d1;
                return false;
            }
            for (const std::string & d2 : second_names) {
                if (!valid_hex(d2, 2)) continue;
                const int second = ::openat(first, d2.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
                if (second < 0) {
                    ::close(first);
                    if (error) *error = "unsafe cache shard: " + d1 + "/" + d2;
                    return false;
                }
                std::vector<std::string> files;
                if (!list_names_at(second, files)) {
                    ::close(second);
                    ::close(first);
                    if (error) *error = "cannot scan cache shard files";
                    return false;
                }
                for (const std::string & name : files) {
                    struct stat st {};
                    if (::fstatat(second, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) != 0) continue;
                    if (S_ISLNK(st.st_mode)) {
                        ::close(second);
                        ::close(first);
                        if (error) *error = "symlink in cache shard: " + name;
                        return false;
                    }
                    if (!S_ISREG(st.st_mode)) continue;
                    if (name.find(".tmp.") != std::string::npos) {
                        ::unlinkat(second, name.c_str(), 0);
                        continue;
                    }
                    if (name.size() != 68 || name.substr(64) != ".bin") continue;
                    const auto key = Key::from_hex(name.substr(0, 64));
                    if (!key) continue;
                    const ArtifactId id{kind, *key};
                    Bytes header;
                    uint64_t file_size = 0;
                    Artifact artifact;
                    const bool valid = read_id(id, header, true, file_size) &&
                                       decode_header(header, file_size, options.max_artifact_bytes, artifact) &&
                                       artifact.kind == kind && artifact.signature == options.signature && artifact.key == *key;
                    if (!valid) {
                        ++counters.corrupt;
                        continue;
                    }
                    disk.emplace(id, DiskEntry{artifact_path(kind, *key), file_size, now_ns(), 0, id});
                    disk_used += file_size;
                }
                ::close(second);
            }
            ::close(first);
        }
        evict_disk_locked();
        return true;
#else
        std::error_code ec;
        for (const auto & first : std::filesystem::directory_iterator(base, ec)) {
            if (ec) { if (error) *error = ec.message(); return false; }
            const auto st = std::filesystem::symlink_status(first.path(), ec);
            if (ec || st.type() == std::filesystem::file_type::symlink) { if (error) *error = "symlink in cache directory: " + first.path().string(); return false; }
            if (!std::filesystem::is_directory(st)) continue;
            const std::string d1 = first.path().filename().string();
            if (!valid_hex(d1, 2)) continue;
            for (const auto & second : std::filesystem::directory_iterator(first.path(), ec)) {
                if (ec) { if (error) *error = ec.message(); return false; }
                const auto st2 = std::filesystem::symlink_status(second.path(), ec);
                if (ec || st2.type() == std::filesystem::file_type::symlink) { if (error) *error = "symlink in cache directory: " + second.path().string(); return false; }
                if (!std::filesystem::is_directory(st2)) continue;
                const std::string d2 = second.path().filename().string();
                if (!valid_hex(d2, 2)) continue;
                for (const auto & file : std::filesystem::directory_iterator(second.path(), ec)) {
                    if (ec) { if (error) *error = ec.message(); return false; }
                    const auto sf = std::filesystem::symlink_status(file.path(), ec);
                    if (ec || sf.type() == std::filesystem::file_type::symlink) { if (error) *error = "symlink in cache directory: " + file.path().string(); return false; }
                    if (!std::filesystem::is_regular_file(sf)) continue;
                    const std::string name = file.path().filename().string();
                    if (name.find(".tmp.") != std::string::npos) { std::filesystem::remove(file.path(), ec); continue; }
                    if (name.size() != 68 || name.substr(64) != ".bin") continue;
                    const auto parsed = Key::from_hex(name.substr(0, 64)); if (!parsed) continue;
                    Bytes bytes; Artifact a; uint64_t file_size = 0;
                    bool valid = read_header(file.path(), bytes, file_size) && decode_header(bytes, file_size, options.max_artifact_bytes, a) &&
                                 a.kind == kind && a.signature == options.signature && a.key == *parsed;
                    if (!valid) { ++counters.corrupt; continue; }
                    const ArtifactId id{ kind, *parsed };
                    const uint64_t size = file_size;
                    disk.emplace(id, DiskEntry{ file.path(), size, now_ns(), 0, id }); disk_used += size;
                }
            }
        }
        evict_disk_locked();
        return true;
#endif
    }

    bool add_hot_locked(const ArtifactId & id, const std::shared_ptr<const Bytes> & bytes) {
        if (options.hot_bytes == 0 || bytes->size() > options.hot_bytes) return false;
        auto existing = hot.find(id);
        if (existing != hot.end()) return true;
        evict_hot_locked(bytes->size());
        if (hot_used + bytes->size() > options.hot_bytes) return false;
        hot_lru.push_front(id);
        hot.emplace(id, HotEntry{ bytes, bytes->size(), 0, hot_lru.begin() });
        hot_used += bytes->size();
        counters.hot_bytes = hot_used; counters.hot_artifacts = hot.size();
        return true;
    }

    void evict_hot_locked(uint64_t needed) {
        size_t scanned = 0;
        while (hot_used + needed > options.hot_bytes && !hot_lru.empty()) {
            const ArtifactId id = hot_lru.back(); auto it = hot.find(id);
            if (it == hot.end()) { hot_lru.pop_back(); continue; }
            if (it->second.pins != 0) {
                hot_lru.pop_back(); hot_lru.push_front(id); it->second.lru = hot_lru.begin();
                if (++scanned >= hot.size()) break;
                continue;
            }
            hot_used -= it->second.size; hot.erase(it); hot_lru.pop_back();
            scanned = 0;
        }
        counters.hot_bytes = hot_used; counters.hot_artifacts = hot.size();
    }

    void evict_disk_locked(uint64_t incoming = 0, uint64_t incoming_files = 0) {
        while ((options.disk_bytes != 0 && disk_used + incoming > options.disk_bytes) ||
               (options.disk_files != 0 && disk.size() + incoming_files > options.disk_files)) {
            auto victim = disk.end();
            for (auto it = disk.begin(); it != disk.end(); ++it) if (it->second.pins == 0 && (victim == disk.end() || it->second.last_access < victim->second.last_access)) victim = it;
            if (victim == disk.end()) break;
#if defined(_WIN32)
            std::error_code ec; const auto st = std::filesystem::symlink_status(victim->second.path, ec);
            if (!ec && st.type() != std::filesystem::file_type::symlink) std::filesystem::remove(victim->second.path, ec);
#else
            if (!unlink_id(victim->second.id)) break;
#endif
            if (disk_used >= victim->second.size) disk_used -= victim->second.size; else disk_used = 0;
            disk.erase(victim);
        }
        counters.disk_bytes = disk_used; counters.disk_artifacts = disk.size();
    }

    void update_pending_stats_locked() {
        counters.pending_write_bytes = pending_bytes + reserved_write_bytes;
        counters.pending_write_items = jobs.size() + in_flight + reserved_items;
        counters.pending_write_peak_bytes = std::max(counters.pending_write_peak_bytes, counters.pending_write_bytes);
        counters.pending_write_capacity = write_capacity;
        counters.reserved_hot_bytes = reserved_hot_bytes;
    }

    bool reserve_bytes(
            ArtifactKind kind,
            uint64_t payload_size,
            bool want_hot,
            std::chrono::milliseconds timeout,
            std::shared_ptr<Bytes> & bytes,
            bool & got_hot) {
        if (payload_size > options.max_artifact_bytes ||
            payload_size > UINT64_MAX - HEADER_SIZE ||
            payload_size > static_cast<uint64_t>(SIZE_MAX) - HEADER_SIZE) return false;
        const uint64_t total = payload_size + HEADER_SIZE;
        std::unique_lock<std::mutex> lock(mutex);
        if (!accepting || stopping) return false;

        if (options.pending_write_bytes == 0) {
            uint64_t & learned = kind == ArtifactKind::ATTENTION ? adaptive_attention_bytes : adaptive_recurrent_bytes;
            if (learned == 0) {
                const uint64_t other = kind == ArtifactKind::ATTENTION ? adaptive_recurrent_bytes : adaptive_attention_bytes;
                if (other > UINT64_MAX - total) {
                    ++counters.enqueue_skips;
                    return false;
                }
                learned = total;
                write_capacity = adaptive_attention_bytes + adaptive_recurrent_bytes;
            } else if (total > learned) {
                ++counters.enqueue_skips;
                return false;
            }
        }
        if (total > write_capacity) {
            ++counters.enqueue_skips;
            return false;
        }

        const auto wait_for = timeout.count() < 0 ? options.enqueue_wait : timeout;
        const auto deadline = std::chrono::steady_clock::now() + wait_for;
        const auto available = [&] {
            return stopping ||
                   (jobs.size() + in_flight + reserved_items < options.pending_write_items &&
                    pending_bytes + reserved_write_bytes + total <= write_capacity);
        };
        if (!available() && (!cv.wait_until(lock, deadline, available) || stopping)) {
            ++counters.enqueue_skips;
            return false;
        }
        if (stopping) return false;

        got_hot = false;
        if (want_hot && options.hot_bytes != 0 && total <= options.hot_bytes) {
            evict_hot_locked(reserved_hot_bytes + total);
            if (hot_used + reserved_hot_bytes + total <= options.hot_bytes) {
                reserved_hot_bytes += total;
                got_hot = true;
            }
        }
        reserved_write_bytes += total;
        ++reserved_items;
        update_pending_stats_locked();
        lock.unlock();

        try {
            bytes = std::make_shared<Bytes>(static_cast<size_t>(total), 0);
        } catch (...) {
            release_reservation(total, got_hot);
            return false;
        }
        return true;
    }

    void release_reservation(uint64_t total, bool had_hot) {
        std::lock_guard<std::mutex> lock(mutex);
        reserved_write_bytes = reserved_write_bytes >= total ? reserved_write_bytes - total : 0;
        if (reserved_items != 0) --reserved_items;
        if (had_hot) reserved_hot_bytes = reserved_hot_bytes >= total ? reserved_hot_bytes - total : 0;
        update_pending_stats_locked();
        cv.notify_all();
    }

    bool commit_reservation(
            ArtifactKind reserved_kind,
            const std::shared_ptr<Bytes> & bytes,
            bool had_hot,
            const Artifact & metadata) {
        const uint64_t total = bytes ? bytes->size() : 0;
        if (!bytes || total < HEADER_SIZE || metadata.kind != reserved_kind ||
            metadata.signature != options.signature || !metadata.payload.empty() ||
            metadata.position_start < 0 || metadata.position_end < metadata.position_start ||
            !finalize_reserved_artifact(*bytes, metadata)) {
            release_reservation(total, had_hot);
            return false;
        }

        const ArtifactId id{metadata.kind, metadata.key};
        std::lock_guard<std::mutex> lock(mutex);
        if (!accepting || stopping) {
            reserved_write_bytes = reserved_write_bytes >= total ? reserved_write_bytes - total : 0;
            if (reserved_items != 0) --reserved_items;
            if (had_hot) reserved_hot_bytes = reserved_hot_bytes >= total ? reserved_hot_bytes - total : 0;
            update_pending_stats_locked();
            cv.notify_all();
            return false;
        }

        reserved_write_bytes -= total;
        --reserved_items;
        pending_bytes += total;
        jobs.push_back(Job{id, std::static_pointer_cast<const Bytes>(bytes)});

        if (had_hot) {
            reserved_hot_bytes -= total;
            if (hot.find(id) == hot.end()) {
                hot_lru.push_front(id);
                hot.emplace(id, HotEntry{std::static_pointer_cast<const Bytes>(bytes), total, 0, hot_lru.begin()});
                hot_used += total;
            }
        }
        counters.hot_bytes = hot_used;
        counters.hot_artifacts = hot.size();
        update_pending_stats_locked();
        cv.notify_all();
        return true;
    }

    bool publish_bytes(const ArtifactId & id, const std::shared_ptr<const Bytes> & bytes) {
        Artifact parsed;
        if (!decode_artifact(*bytes, options.max_artifact_bytes, parsed) || parsed.kind != id.kind || parsed.key != id.key || parsed.signature != options.signature) {
            std::lock_guard<std::mutex> lock(mutex); ++counters.corrupt; return false;
        }
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (clearing) return false;
            if (disk.find(id) != disk.end()) {
                ++counters.duplicate_publications;
                return true;
            }
            if (options.disk_bytes != 0 && bytes->size() > options.disk_bytes) return false;
            evict_disk_locked(bytes->size() + disk_reserved_bytes, disk_reserved_files + 1);
            if ((options.disk_bytes != 0 && disk_used + disk_reserved_bytes + bytes->size() > options.disk_bytes) ||
                (options.disk_files != 0 && disk.size() + disk_reserved_files + 1 > options.disk_files)) return false;
            disk_reserved_bytes += bytes->size();
            ++disk_reserved_files;
            ++disk_io_inflight;
            counters.disk_io_inflight = disk_io_inflight;
        }

        bool io_ok = false;
        bool duplicate = false;
        uint64_t actual_size = bytes->size();
        std::string error;
        if (make_shards(id.kind, id.key, &error)) {
#if defined(_WIN32)
            const auto path = artifact_path(id.kind, id.key);
            std::error_code ec;
            const auto st = std::filesystem::symlink_status(path, ec);
            if (!ec && std::filesystem::is_regular_file(st)) {
                Bytes existing_bytes; Artifact existing_artifact;
                if (read_file(path, existing_bytes) &&
                    decode_artifact(existing_bytes, options.max_artifact_bytes, existing_artifact) &&
                    existing_artifact.kind == id.kind && existing_artifact.key == id.key &&
                    existing_artifact.signature == options.signature) {
                    duplicate = true;
                    actual_size = existing_bytes.size();
                    io_ok = true;
                }
            } else if (!ec && !std::filesystem::exists(st)) {
                const std::string suffix = ".tmp.0." + std::to_string(next_temp_nonce());
                const auto tmp = path.string() + suffix;
                std::ofstream output(tmp, std::ios::binary | std::ios::out);
                if (output) {
                    output.write(reinterpret_cast<const char *>(bytes->data()), static_cast<std::streamsize>(bytes->size()));
                    output.flush();
                    output.close();
                    if (!std::filesystem::exists(path)) std::filesystem::rename(tmp, path, ec);
                    else ec = std::make_error_code(std::errc::file_exists);
                    if (ec) std::filesystem::remove(tmp, ec);
                    else io_ok = true;
                }
            }
#else
            const int shard = open_shard_fd(id.kind, id.key, true, &error);
            if (shard >= 0) {
                const std::string final_name = id.key.hex() + ".bin";
                int existing_fd = ::openat(shard, final_name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
                bool may_write = existing_fd < 0 && errno == ENOENT;
                if (existing_fd >= 0) {
                    Bytes existing_bytes;
                    const bool bounded = regular_file_size(existing_fd, actual_size) &&
                                         actual_size <= HEADER_SIZE + options.max_artifact_bytes &&
                                         actual_size <= static_cast<uint64_t>(SIZE_MAX);
                    if (bounded) existing_bytes.resize(static_cast<size_t>(actual_size));
                    const bool read_ok = bounded && read_all_fd(existing_fd, existing_bytes.data(), existing_bytes.size());
                    ::close(existing_fd);
                    Artifact existing_artifact;
                    if (read_ok && decode_artifact(existing_bytes, options.max_artifact_bytes, existing_artifact) &&
                        existing_artifact.kind == id.kind && existing_artifact.key == id.key &&
                        existing_artifact.signature == options.signature) {
                        duplicate = true;
                        io_ok = true;
                    } else {
                        const std::string q = final_name + "." + std::to_string(next_temp_nonce());
                        if (::linkat(shard, final_name.c_str(), quarantine_fd, q.c_str(), 0) == 0 &&
                            ::unlinkat(shard, final_name.c_str(), 0) == 0) {
                            may_write = true;
                        }
                    }
                }
                if (!io_ok && may_write) {
                    const std::string tmp_name = final_name + ".tmp." +
                        std::to_string(static_cast<unsigned long long>(::getpid())) + "." +
                        std::to_string(next_temp_nonce());
                    const int fd = ::openat(shard, tmp_name.c_str(),
                        O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
                    if (fd >= 0 && write_all_fd(fd, bytes->data(), bytes->size()) && ::fsync(fd) == 0) {
                        ::close(fd);
                        if (::linkat(shard, tmp_name.c_str(), shard, final_name.c_str(), 0) == 0) {
                            io_ok = true;
                            actual_size = bytes->size();
                        } else if (errno == EEXIST) {
                            Bytes existing_bytes;
                            if (read_id(id, existing_bytes, false, actual_size)) {
                                Artifact existing_artifact;
                                duplicate = decode_artifact(existing_bytes, options.max_artifact_bytes, existing_artifact) &&
                                            existing_artifact.kind == id.kind && existing_artifact.key == id.key &&
                                            existing_artifact.signature == options.signature;
                                io_ok = duplicate;
                            }
                        }
                    } else if (fd >= 0) {
                        ::close(fd);
                    }
                    ::unlinkat(shard, tmp_name.c_str(), 0);
                    if (io_ok && !duplicate) ::fsync(shard);
                }
                ::close(shard);
            }
#endif
        }

        std::unique_lock<std::mutex> lock(mutex);
        disk_reserved_bytes -= bytes->size();
        --disk_reserved_files;
        --disk_io_inflight;
        counters.disk_io_inflight = disk_io_inflight;
        if (io_ok) {
            auto inserted = disk.emplace(id, DiskEntry{artifact_path(id.kind, id.key), actual_size, now_ns(), 0, id});
            if (inserted.second) disk_used += actual_size;
            if (duplicate || !inserted.second) ++counters.duplicate_publications;
            else {
                ++counters.published;
                counters.bytes_written += bytes->size();
            }
            evict_disk_locked();
        }
        disk_cv.notify_all();
        return io_ok;
    }

    std::optional<Artifact> lookup(ArtifactKind kind, const Key & key) {
        const ArtifactId id{kind, key};
        std::unique_lock<std::mutex> lock(mutex);
        ++counters.lookups;
        if (kind == ArtifactKind::ATTENTION) ++counters.attention_lookups;
        else ++counters.recurrent_lookups;
        auto h = hot.find(id);
        if (h != hot.end()) {
            Artifact out;
            if (decode_artifact(*h->second.bytes, options.max_artifact_bytes, out)) {
                hot_lru.erase(h->second.lru); hot_lru.push_front(id); h->second.lru = hot_lru.begin();
                ++counters.hot_hits;
                if (kind == ArtifactKind::ATTENTION) ++counters.attention_hot_hits;
                else ++counters.recurrent_hot_hits;
                return out;
            }
            hot_used -= h->second.size; hot_lru.erase(h->second.lru); hot.erase(h); ++counters.corrupt;
            counters.hot_bytes = hot_used; counters.hot_artifacts = hot.size();
        }
        auto d = disk.find(id);
        if (d == disk.end() || clearing) {
            ++counters.misses;
            if (kind == ArtifactKind::ATTENTION) ++counters.attention_misses;
            else ++counters.recurrent_misses;
            return std::nullopt;
        }
        ++d->second.pins;
        ++disk_io_inflight;
        counters.disk_io_inflight = disk_io_inflight;
        const uint64_t expected_size = d->second.size;
#if defined(_WIN32)
        const auto disk_path = d->second.path;
#endif
        lock.unlock();
        Bytes bytes; Artifact out;
#if defined(_WIN32)
        const bool read_ok = read_file(disk_path, bytes) && bytes.size() == expected_size;
#else
        uint64_t file_size = 0;
        const bool read_ok = read_id(id, bytes, false, file_size) && file_size == expected_size;
#endif
        const bool ok = read_ok && decode_artifact(bytes, options.max_artifact_bytes, out) &&
                        out.kind == kind && out.key == key && out.signature == options.signature;
        lock.lock();
        d = disk.find(id);
        if (d != disk.end() && d->second.pins != 0) --d->second.pins;
        --disk_io_inflight;
        counters.disk_io_inflight = disk_io_inflight;
        disk_cv.notify_all();
        if (!ok) {
            ++counters.corrupt; ++counters.misses;
            if (kind == ArtifactKind::ATTENTION) ++counters.attention_misses;
            else ++counters.recurrent_misses;
            if (d != disk.end()) {
                if (disk_used >= d->second.size) disk_used -= d->second.size; else disk_used = 0;
                disk.erase(d);
            }
            counters.disk_bytes = disk_used; counters.disk_artifacts = disk.size();
            return std::nullopt;
        }
        if (d == disk.end()) {
            ++counters.misses;
            if (kind == ArtifactKind::ATTENTION) ++counters.attention_misses;
            else ++counters.recurrent_misses;
            return std::nullopt;
        }
        d->second.last_access = now_ns(); ++counters.disk_hits;
        if (kind == ArtifactKind::ATTENTION) ++counters.attention_disk_hits;
        else ++counters.recurrent_disk_hits;
        counters.bytes_read += bytes.size();
        add_hot_locked(id, std::make_shared<const Bytes>(bytes));
        return out;
    }

    bool put_hot(const Artifact & a) {
        const auto bytes = std::make_shared<const Bytes>(encode_artifact(a)); Artifact parsed;
        if (!decode_artifact(*bytes, options.max_artifact_bytes, parsed) || parsed.signature != options.signature || parsed.key != a.key || parsed.kind != a.kind) return false;
        std::lock_guard<std::mutex> lock(mutex); return add_hot_locked(ArtifactId{a.kind, a.key}, bytes);
    }

    bool publish(const Artifact & a) { const auto bytes = std::make_shared<const Bytes>(encode_artifact(a)); return publish_bytes(ArtifactId{a.kind, a.key}, bytes); }

    bool enqueue(const Artifact & a) {
        const auto bytes = std::make_shared<const Bytes>(encode_artifact(a));
        Artifact parsed; if (!decode_artifact(*bytes, options.max_artifact_bytes, parsed) || parsed.signature != options.signature || parsed.key != a.key || parsed.kind != a.kind) return false;
        std::unique_lock<std::mutex> lock(mutex);
        if (!accepting || options.pending_write_items == 0) { ++counters.enqueue_skips; return false; }
        if (options.pending_write_bytes == 0) {
            uint64_t & learned = a.kind == ArtifactKind::ATTENTION ? adaptive_attention_bytes : adaptive_recurrent_bytes;
            if (learned == 0) {
                const uint64_t other = a.kind == ArtifactKind::ATTENTION ? adaptive_recurrent_bytes : adaptive_attention_bytes;
                if (other > UINT64_MAX - bytes->size()) {
                    ++counters.enqueue_skips;
                    return false;
                }
                learned = bytes->size();
                write_capacity = adaptive_attention_bytes + adaptive_recurrent_bytes;
            } else if (bytes->size() > learned) {
                ++counters.enqueue_skips;
                return false;
            }
        }
        if (bytes->size() > write_capacity) { ++counters.enqueue_skips; return false; }
        const auto deadline = std::chrono::steady_clock::now() + options.enqueue_wait;
        if (!cv.wait_until(lock, deadline, [&] {
                return stopping || (jobs.size() + in_flight + reserved_items < options.pending_write_items &&
                                    pending_bytes + reserved_write_bytes + bytes->size() <= write_capacity);
            }) || stopping) { ++counters.enqueue_skips; return false; }
        jobs.push_back(Job{ArtifactId{a.kind, a.key}, bytes}); pending_bytes += bytes->size();
        update_pending_stats_locked(); cv.notify_all(); return true;
    }

    bool flush(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        return done_cv.wait_for(lock, timeout, [&] { return jobs.empty() && in_flight == 0; });
    }

    void shutdown(std::chrono::milliseconds timeout) {
        {
            std::lock_guard<std::mutex> lock(mutex); if (!writer.joinable()) return; accepting = false; stopping = true; cv.notify_all();
        }
        if (!flush(timeout)) {
            std::lock_guard<std::mutex> lock(mutex); abort = true;
            for (const auto & job : jobs) if (pending_bytes >= job.bytes->size()) pending_bytes -= job.bytes->size();
            jobs.clear(); update_pending_stats_locked(); cv.notify_all();
        }
        if (writer.joinable()) writer.join();
    }

    void writer_loop() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return stopping || !jobs.empty(); });
                if (jobs.empty() && stopping) return;
                if (jobs.empty()) continue;
                job = std::move(jobs.front()); jobs.pop_front(); ++in_flight;
                update_pending_stats_locked(); cv.notify_all();
            }
            if (!abort) publish_bytes(job.id, job.bytes);
            {
                std::lock_guard<std::mutex> lock(mutex);
                --in_flight;
                if (pending_bytes >= job.bytes->size()) pending_bytes -= job.bytes->size(); else pending_bytes = 0;
                update_pending_stats_locked();
                done_cv.notify_all(); cv.notify_all();
            }
        }
    }

    void clear_hot() {
        std::lock_guard<std::mutex> lock(mutex); hot.clear(); hot_lru.clear(); hot_used = 0; counters.hot_bytes = 0; counters.hot_artifacts = 0;
    }

#if !defined(_WIN32)
    bool validate_tree_fd(int dir_fd, std::string * error) const {
        std::vector<std::string> names;
        if (!list_names_at(dir_fd, names)) {
            if (error) *error = "cannot enumerate cache directory";
            return false;
        }
        for (const std::string & name : names) {
            struct stat st {};
            if (::fstatat(dir_fd, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) != 0) {
                if (error) *error = "cannot inspect cache entry: " + name;
                return false;
            }
            if (S_ISLNK(st.st_mode)) {
                if (error) *error = "symlink in cache: " + name;
                return false;
            }
            if (S_ISDIR(st.st_mode)) {
                const int child = ::openat(dir_fd, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
                if (child < 0) {
                    if (error) *error = "unsafe cache directory: " + name;
                    return false;
                }
                const bool ok = validate_tree_fd(child, error);
                ::close(child);
                if (!ok) return false;
            } else if (!S_ISREG(st.st_mode)) {
                if (error) *error = "unsupported cache entry: " + name;
                return false;
            }
        }
        return true;
    }

    bool clear_tree_fd(int dir_fd, std::string * error) const {
        std::vector<std::string> names;
        if (!list_names_at(dir_fd, names)) {
            if (error) *error = "cannot enumerate cache directory";
            return false;
        }
        for (const std::string & name : names) {
            struct stat st {};
            if (::fstatat(dir_fd, name.c_str(), &st, AT_SYMLINK_NOFOLLOW) != 0 || S_ISLNK(st.st_mode)) {
                if (error) *error = "unsafe cache entry: " + name;
                return false;
            }
            if (S_ISDIR(st.st_mode)) {
                const int child = ::openat(dir_fd, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
                if (child < 0) {
                    if (error) *error = "unsafe cache directory: " + name;
                    return false;
                }
                const bool ok = clear_tree_fd(child, error);
                ::close(child);
                if (!ok || ::unlinkat(dir_fd, name.c_str(), AT_REMOVEDIR) != 0) {
                    if (error && error->empty()) *error = "cannot remove cache directory: " + name;
                    return false;
                }
            } else if (!S_ISREG(st.st_mode) || ::unlinkat(dir_fd, name.c_str(), 0) != 0) {
                if (error) *error = "cannot remove cache file: " + name;
                return false;
            }
        }
        return true;
    }
#endif

    bool clear_disk(std::string * error) {
        if (!flush(std::chrono::seconds(30))) { if (error) *error = "timed out waiting for cache writer"; return false; }
        std::unique_lock<std::mutex> lock(mutex);
        if (clearing) {
            if (error) *error = "cache clear is already active";
            return false;
        }
        clearing = true;
        disk_cv.wait(lock, [&] { return disk_io_inflight == 0; });
        for (const auto & entry : disk) {
#if defined(_WIN32)
            std::error_code ec; const auto st = std::filesystem::symlink_status(entry.second.path, ec);
            if (ec || st.type() == std::filesystem::file_type::symlink) {
                if (error) *error = "symlink in cache: " + entry.second.path.string();
                clearing = false;
                disk_cv.notify_all();
                return false;
            }
#endif
            if (entry.second.pins != 0) {
                if (error) *error = "cache artifact is busy";
                clearing = false;
                disk_cv.notify_all();
                return false;
            }
        }
        lock.unlock();
        bool io_ok = false;
#if !defined(_WIN32)
        io_ok = validate_tree_fd(attention_fd, error) && validate_tree_fd(recurrent_fd, error) &&
                validate_tree_fd(quarantine_fd, error) &&
                clear_tree_fd(attention_fd, error) && clear_tree_fd(recurrent_fd, error) &&
                clear_tree_fd(quarantine_fd, error);
#else
        // Validate the complete owned tree before deleting anything.  In
        // particular, never let a symlink turn a model-scoped clear into a
        // recursive delete outside the configured root.
        const auto validate_tree = [&](const auto & self, const std::filesystem::path & dir) -> bool {
            std::error_code ec;
            for (const auto & item : std::filesystem::directory_iterator(dir, ec)) {
                if (ec) { if (error) *error = ec.message(); return false; }
                const auto st = std::filesystem::symlink_status(item.path(), ec);
                if (ec || st.type() == std::filesystem::file_type::symlink) { if (error) *error = "symlink in cache: " + item.path().string(); return false; }
                if (std::filesystem::is_directory(st) && !self(self, item.path())) return false;
            }
            return true;
        };
        const bool valid = validate_tree(validate_tree, attention_dir) &&
                           validate_tree(validate_tree, recurrent_dir) &&
                           validate_tree(validate_tree, quarantine_dir);
        const auto delete_tree = [&](const auto & self, const std::filesystem::path & dir) -> bool {
            std::error_code ec;
            for (const auto & item : std::filesystem::directory_iterator(dir, ec)) {
                if (ec) { if (error) *error = ec.message(); return false; }
                const auto st = std::filesystem::symlink_status(item.path(), ec);
                if (ec) { if (error) *error = ec.message(); return false; }
                if (std::filesystem::is_directory(st)) { if (!self(self, item.path())) return false; }
                else if (!std::filesystem::remove(item.path(), ec) || ec) { if (error) *error = ec ? ec.message() : "cannot remove cache file"; return false; }
            }
            return true;
        };
        io_ok = valid && delete_tree(delete_tree, attention_dir) &&
                delete_tree(delete_tree, recurrent_dir) &&
                delete_tree(delete_tree, quarantine_dir);
#endif
        lock.lock();
        if (io_ok) {
            disk.clear();
            disk_used = 0;
            counters.disk_bytes = 0;
            counters.disk_artifacts = 0;
        }
        clearing = false;
        disk_cv.notify_all();
        return io_ok;
    }
};

struct PrefixCacheStore::Reservation::State {
    std::shared_ptr<PrefixCacheStore::Impl> owner;
    std::shared_ptr<Bytes> bytes;
    ArtifactKind kind = ArtifactKind::ATTENTION;
    bool hot = false;
    bool active = true;

    ~State() {
        if (active && owner && bytes) owner->release_reservation(bytes->size(), hot);
    }
};

std::unique_ptr<PrefixCacheStore> PrefixCacheStore::open(const Options & options, std::string * error) {
    auto result = std::unique_ptr<PrefixCacheStore>(new PrefixCacheStore(options));
    if (!result->impl_->initialize(error)) return nullptr;
    return result;
}

PrefixCacheStore::Reservation::Reservation() = default;
PrefixCacheStore::Reservation::Reservation(std::unique_ptr<State> state) : state_(std::move(state)) {}
PrefixCacheStore::Reservation::~Reservation() = default;
PrefixCacheStore::Reservation::Reservation(Reservation &&) noexcept = default;
PrefixCacheStore::Reservation & PrefixCacheStore::Reservation::operator=(Reservation &&) noexcept = default;
uint8_t * PrefixCacheStore::Reservation::payload_data() {
    return state_ && state_->bytes ? state_->bytes->data() + HEADER_SIZE : nullptr;
}
size_t PrefixCacheStore::Reservation::payload_size() const {
    return state_ && state_->bytes ? state_->bytes->size() - HEADER_SIZE : 0;
}
bool PrefixCacheStore::Reservation::reserved_hot() const { return state_ && state_->hot; }
bool PrefixCacheStore::Reservation::commit(const Artifact & metadata) {
    if (!state_ || !state_->active) return false;
    state_->active = false;
    return state_->owner->commit_reservation(state_->kind, state_->bytes, state_->hot, metadata);
}

PrefixCacheStore::PrefixCacheStore(const Options & options) : impl_(std::make_shared<Impl>(options)) {}
PrefixCacheStore::~PrefixCacheStore() { if (impl_) impl_->shutdown(std::chrono::milliseconds(5000)); }
std::optional<Artifact> PrefixCacheStore::lookup(ArtifactKind kind, const Key & key) { return impl_->lookup(kind, key); }
std::optional<PrefixCacheStore::Reservation> PrefixCacheStore::reserve(
        ArtifactKind kind, uint64_t payload_bytes, bool want_hot, std::chrono::milliseconds timeout) {
    std::shared_ptr<Bytes> bytes;
    bool got_hot = false;
    if (!impl_->reserve_bytes(kind, payload_bytes, want_hot, timeout, bytes, got_hot)) return std::nullopt;
    auto state = std::make_unique<Reservation::State>();
    state->owner = impl_;
    state->bytes = std::move(bytes);
    state->kind = kind;
    state->hot = got_hot;
    return Reservation(std::move(state));
}
bool PrefixCacheStore::put_hot(const Artifact & artifact) { return impl_->put_hot(artifact); }
void PrefixCacheStore::clear_hot() { impl_->clear_hot(); }
bool PrefixCacheStore::publish(const Artifact & artifact) { return impl_->publish(artifact); }
bool PrefixCacheStore::enqueue(const Artifact & artifact) { return impl_->enqueue(artifact); }
bool PrefixCacheStore::flush(std::chrono::milliseconds timeout) { return impl_->flush(timeout); }
void PrefixCacheStore::shutdown(std::chrono::milliseconds timeout) { impl_->shutdown(timeout); }
bool PrefixCacheStore::clear_disk(std::string * error) { return impl_->clear_disk(error); }
CacheStats PrefixCacheStore::stats() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->counters; }
std::filesystem::path PrefixCacheStore::signature_directory() const { return impl_->signature_dir; }

} // namespace server_prefix_cache
