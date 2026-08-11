#include "server-prefix-admission.h"

#include <list>
#include <unordered_map>

namespace server_prefix_cache {

struct PrefixAdmission::Impl {
    struct Entry {
        uint8_t observations = 0;
        uint64_t last_observed = 0;
        bool admitted = false;
        std::list<Key>::iterator lru;
    };

    explicit Impl(size_t capacity_) : capacity(capacity_) {}

    void touch(std::unordered_map<Key, Entry, KeyHash>::iterator it) {
        lru.erase(it->second.lru);
        lru.push_front(it->first);
        it->second.lru = lru.begin();
        it->second.last_observed = ++clock;
    }

    void evict_to_capacity() {
        while (items.size() > capacity) {
            const Key victim = lru.back();
            lru.pop_back();
            const auto it = items.find(victim);
            if (it != items.end()) {
                if (it->second.admitted) {
                    --admitted_items;
                }
                items.erase(it);
                ++stats.lru_evictions;
            }
        }
    }

    size_t capacity = 0;
    uint64_t clock = 0;
    uint64_t admitted_items = 0;
    AdmissionStats stats;
    std::list<Key> lru; // Most recently observed first.
    std::unordered_map<Key, Entry, KeyHash> items;
};

PrefixAdmission::PrefixAdmission(size_t capacity) : impl_(new Impl(capacity)) {}

PrefixAdmission::~PrefixAdmission() = default;

PrefixAdmission::PrefixAdmission(PrefixAdmission &&) noexcept = default;

PrefixAdmission & PrefixAdmission::operator=(PrefixAdmission &&) noexcept = default;

bool PrefixAdmission::observe_cold(const Key & key) {
    ++impl_->stats.boundaries_observed;

    auto it = impl_->items.find(key);
    if (it == impl_->items.end()) {
        if (impl_->capacity == 0) {
            return false;
        }
        impl_->lru.push_front(key);
        Impl::Entry inserted;
        inserted.observations = 1;
        inserted.last_observed = ++impl_->clock;
        inserted.lru = impl_->lru.begin();
        impl_->items.emplace(key, inserted);
        ++impl_->stats.first_observations;
        impl_->evict_to_capacity();
        return false;
    }

    impl_->touch(it);
    if (it->second.observations < 2) {
        ++it->second.observations;
    }
    if (!it->second.admitted && it->second.observations >= 2) {
        it->second.admitted = true;
        ++impl_->admitted_items;
        ++impl_->stats.repeat_admissions;
    }
    return it->second.admitted;
}

bool PrefixAdmission::observe_explicit(const Key &) {
    ++impl_->stats.explicit_admissions;
    return true;
}

void PrefixAdmission::capture_failed(const Key &) {
    // Admission is intentionally sticky across failed capture/backpressure.
}

void PrefixAdmission::published(const Key & key) {
    const auto it = impl_->items.find(key);
    if (it == impl_->items.end()) {
        return;
    }
    if (it->second.admitted) {
        --impl_->admitted_items;
    }
    impl_->lru.erase(it->second.lru);
    impl_->items.erase(it);
}

bool PrefixAdmission::admitted(const Key & key) const {
    const auto it = impl_->items.find(key);
    return it != impl_->items.end() && it->second.admitted;
}

bool PrefixAdmission::contains(const Key & key) const {
    return impl_->items.find(key) != impl_->items.end();
}

size_t PrefixAdmission::size() const {
    return impl_->items.size();
}

size_t PrefixAdmission::capacity() const {
    return impl_->capacity;
}

AdmissionStats PrefixAdmission::stats() const {
    AdmissionStats out = impl_->stats;
    out.items = impl_->items.size();
    out.admitted_items = impl_->admitted_items;
    out.capacity = impl_->capacity;
    return out;
}

AdmissionEntry PrefixAdmission::entry(const Key & key) const {
    const auto it = impl_->items.find(key);
    if (it == impl_->items.end()) {
        return {};
    }
    AdmissionEntry out;
    out.observations = it->second.observations;
    out.last_observed = it->second.last_observed;
    out.admitted = it->second.admitted;
    return out;
}

void PrefixAdmission::clear() {
    impl_->items.clear();
    impl_->lru.clear();
    impl_->admitted_items = 0;
}

} // namespace server_prefix_cache
