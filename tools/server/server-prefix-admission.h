#pragma once

#include "server-prefix-cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace server_prefix_cache {

struct AdmissionStats {
    uint64_t boundaries_observed = 0;
    uint64_t first_observations = 0;
    uint64_t repeat_admissions = 0;
    uint64_t explicit_admissions = 0;
    uint64_t lru_evictions = 0;
    uint64_t items = 0;
    uint64_t admitted_items = 0;
    uint64_t capacity = 0;
};

struct AdmissionEntry {
    uint8_t observations = 0;
    uint64_t last_observed = 0;
    bool admitted = false;
};

class PrefixAdmission {
public:
    explicit PrefixAdmission(size_t capacity);
    ~PrefixAdmission();

    PrefixAdmission(PrefixAdmission &&) noexcept;
    PrefixAdmission & operator=(PrefixAdmission &&) noexcept;
    PrefixAdmission(const PrefixAdmission &) = delete;
    PrefixAdmission & operator=(const PrefixAdmission &) = delete;

    // Records one cold boundary observation. The second observation admits the
    // key. Counts saturate at two and each call counts at most once.
    bool observe_cold(const Key & key);

    // Explicit capture bypasses frequency admission and does not require an
    // LRU entry to remain eligible.
    bool observe_explicit(const Key & key);

    // Failed capture leaves an admitted hint in place for a later cold retry.
    void capture_failed(const Key & key);

    // Successful publication makes the durable artifact authoritative, so the
    // process-local hint can be removed.
    void published(const Key & key);

    bool admitted(const Key & key) const;
    bool contains(const Key & key) const;
    size_t size() const;
    size_t capacity() const;
    AdmissionStats stats() const;
    AdmissionEntry entry(const Key & key) const;
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace server_prefix_cache
