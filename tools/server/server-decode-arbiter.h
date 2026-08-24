#pragma once

#include <atomic>
#include <cstdint>
#include <string>

struct server_decode_arbiter_stats {
    bool enabled = false;
    uint64_t acquisitions = 0;
    uint64_t contentions = 0;
    uint64_t wait_us = 0;
    uint64_t hold_us = 0;
};

class server_decode_arbiter {
public:
    class lease {
    public:
        lease() = default;
        lease(const lease &) = delete;
        lease & operator=(const lease &) = delete;
        lease(lease && other) noexcept;
        lease & operator=(lease && other) noexcept;
        ~lease();

        explicit operator bool() const {
            return owner != nullptr;
        }

    private:
        friend class server_decode_arbiter;

        lease(server_decode_arbiter * owner, int64_t acquired_at_us);
        void release();

        server_decode_arbiter * owner = nullptr;
        int64_t acquired_at_us = 0;
    };

    server_decode_arbiter() = default;
    explicit server_decode_arbiter(const std::string & path);
    ~server_decode_arbiter();

    server_decode_arbiter(const server_decode_arbiter &) = delete;
    server_decode_arbiter & operator=(const server_decode_arbiter &) = delete;

    bool open(const std::string & path, std::string & error);
    lease acquire();
    server_decode_arbiter_stats stats() const;

private:
    friend class lease;

    void release(int64_t acquired_at_us);

    int fd = -1;
    std::atomic<uint64_t> acquisitions{0};
    std::atomic<uint64_t> contentions{0};
    std::atomic<uint64_t> wait_us{0};
    std::atomic<uint64_t> hold_us{0};
};
