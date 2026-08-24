#include "server-decode-arbiter.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

}

server_decode_arbiter::lease::lease(server_decode_arbiter * owner, int64_t acquired_at_us)
        : owner(owner), acquired_at_us(acquired_at_us) {
}

server_decode_arbiter::lease::lease(lease && other) noexcept
        : owner(std::exchange(other.owner, nullptr)),
          acquired_at_us(std::exchange(other.acquired_at_us, 0)) {
}

server_decode_arbiter::lease & server_decode_arbiter::lease::operator=(lease && other) noexcept {
    if (this != &other) {
        release();
        owner = std::exchange(other.owner, nullptr);
        acquired_at_us = std::exchange(other.acquired_at_us, 0);
    }
    return *this;
}

server_decode_arbiter::lease::~lease() {
    release();
}

void server_decode_arbiter::lease::release() {
    if (owner != nullptr) {
        owner->release(acquired_at_us);
        owner = nullptr;
        acquired_at_us = 0;
    }
}

server_decode_arbiter::server_decode_arbiter(const std::string & path) {
    std::string error;
    if (!open(path, error)) {
        throw std::runtime_error(error);
    }
}

server_decode_arbiter::~server_decode_arbiter() {
#if !defined(_WIN32)
    if (fd >= 0) {
        ::close(fd);
    }
#endif
}

bool server_decode_arbiter::open(const std::string & path, std::string & error) {
    if (path.empty()) {
        return true;
    }
    if (fd >= 0) {
        error = "decode arbiter is already open";
        return false;
    }

#if defined(_WIN32)
    error = "cross-model decode arbitration is not supported on Windows";
    return false;
#else
    const int opened = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (opened < 0) {
        error = "failed to open decode arbiter '" + path + "': " + std::strerror(errno);
        return false;
    }

    struct stat st {};
    if (::fstat(opened, &st) != 0 || !S_ISREG(st.st_mode)) {
        error = "decode arbiter path is not a regular file: " + path;
        ::close(opened);
        return false;
    }

    fd = opened;
    return true;
#endif
}

server_decode_arbiter::lease server_decode_arbiter::acquire() {
    if (fd < 0) {
        return {};
    }

#if defined(_WIN32)
    return {};
#else
    const int64_t started_at = now_us();
    bool contended = false;

    while (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EINTR) {
            continue;
        }
        if (errno != EWOULDBLOCK && errno != EAGAIN) {
            throw std::runtime_error(std::string("failed to acquire decode arbiter: ") + std::strerror(errno));
        }
        contended = true;
        while (::flock(fd, LOCK_EX) != 0) {
            if (errno != EINTR) {
                throw std::runtime_error(std::string("failed to wait for decode arbiter: ") + std::strerror(errno));
            }
        }
        break;
    }

    const int64_t acquired_at = now_us();
    acquisitions.fetch_add(1, std::memory_order_relaxed);
    wait_us.fetch_add((uint64_t) (acquired_at - started_at), std::memory_order_relaxed);
    if (contended) {
        contentions.fetch_add(1, std::memory_order_relaxed);
    }
    return lease(this, acquired_at);
#endif
}

server_decode_arbiter_stats server_decode_arbiter::stats() const {
    return {
        fd >= 0,
        acquisitions.load(std::memory_order_relaxed),
        contentions.load(std::memory_order_relaxed),
        wait_us.load(std::memory_order_relaxed),
        hold_us.load(std::memory_order_relaxed),
    };
}

void server_decode_arbiter::release(int64_t acquired_at_us) {
#if !defined(_WIN32)
    hold_us.fetch_add((uint64_t) (now_us() - acquired_at_us), std::memory_order_relaxed);
    if (::flock(fd, LOCK_UN) != 0) {
        std::terminate();
    }
#else
    (void) acquired_at_us;
#endif
}
