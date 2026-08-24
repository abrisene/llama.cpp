#include "server-decode-arbiter.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

int main() {
    server_decode_arbiter disabled;
    if (disabled.acquire() || disabled.stats().enabled) {
        return 1;
    }

#if defined(_WIN32)
    return 0;
#else
    const fs::path root = fs::temp_directory_path() /
            ("llama-decode-arbiter-test-" + std::to_string((long long) getpid()));
    const fs::path lock_path = root / "decode.lock";
    std::error_code ec;
    fs::remove_all(root, ec);
    if (!fs::create_directories(root, ec)) {
        return 1;
    }

    server_decode_arbiter parent;
    std::string error;
    if (!parent.open(lock_path.string(), error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    int pipe_fd[2] = {-1, -1};
    if (::pipe(pipe_fd) != 0) {
        return 1;
    }

    {
        auto parent_lease = parent.acquire();
        if (!parent_lease) {
            return 1;
        }

        const pid_t child = ::fork();
        if (child < 0) {
            return 1;
        }
        if (child == 0) {
            ::close(pipe_fd[0]);
            server_decode_arbiter contender;
            std::string child_error;
            if (!contender.open(lock_path.string(), child_error)) {
                _exit(2);
            }
            auto lease = contender.acquire();
            const auto stats = contender.stats();
            const uint64_t result[2] = {stats.contentions, stats.wait_us};
            const bool wrote = ::write(pipe_fd[1], result, sizeof(result)) == (ssize_t) sizeof(result);
            _exit(wrote ? 0 : 3);
        }

        ::close(pipe_fd[1]);
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        parent_lease = {};

        uint64_t result[2] = {};
        const bool read_ok = ::read(pipe_fd[0], result, sizeof(result)) == (ssize_t) sizeof(result);
        ::close(pipe_fd[0]);

        int child_status = 1;
        if (::waitpid(child, &child_status, 0) < 0 ||
                !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0 ||
                !read_ok || result[0] != 1 || result[1] < 50000) {
            return 1;
        }
    }

    const auto stats = parent.stats();
    if (!stats.enabled || stats.acquisitions != 1 || stats.hold_us < 50000) {
        return 1;
    }

    const fs::path symlink_path = root / "symlink.lock";
    if (::symlink(lock_path.c_str(), symlink_path.c_str()) != 0) {
        return 1;
    }
    server_decode_arbiter unsafe;
    error.clear();
    if (unsafe.open(symlink_path.string(), error)) {
        return 1;
    }

    fs::remove_all(root, ec);
    return 0;
#endif
}
