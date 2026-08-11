#include "server-prefix-admission.h"

#include <cstdint>
#include <cstdlib>

using namespace server_prefix_cache;

static Key make_key(uint8_t seed) {
    Key key;
    for (size_t i = 0; i < key.bytes.size(); ++i) {
        key.bytes[i] = uint8_t(seed + i);
    }
    return key;
}

static void require(bool ok) {
    if (!ok) {
        std::abort();
    }
}

int main() {
    const Key a = make_key(0x10);
    const Key b = make_key(0x20);
    const Key c = make_key(0x30);
    const Key d = make_key(0x40);

    PrefixAdmission admission(3);

    require(!admission.observe_cold(a));
    require(admission.contains(a));
    require(!admission.admitted(a));
    require(admission.entry(a).observations == 1);

    require(admission.observe_cold(a));
    require(admission.admitted(a));
    require(admission.entry(a).observations == 2);

    // Counts saturate at two, but each explicit observe call is still counted.
    require(admission.observe_cold(a));
    require(admission.entry(a).observations == 2);
    require(admission.stats().boundaries_observed == 3);
    require(admission.stats().repeat_admissions == 1);

    // Failed capture/backpressure leaves the repeat admission sticky.
    admission.capture_failed(a);
    require(admission.admitted(a));

    // Successful publication removes the process-local hint because the store
    // is now authoritative.
    admission.published(a);
    require(!admission.contains(a));
    require(!admission.admitted(a));

    // After removal, the same key must need two fresh cold observations.
    require(!admission.observe_cold(a));
    require(admission.entry(a).observations == 1);
    require(admission.observe_cold(a));
    require(admission.admitted(a));

    // LRU eviction is bounded and evicts the least recently observed entry.
    require(!admission.observe_cold(b));
    require(!admission.observe_cold(c));
    require(admission.size() == 3);
    require(admission.observe_cold(a)); // refresh a
    require(!admission.observe_cold(d));
    require(admission.size() == 3);
    require(admission.contains(a));
    require(!admission.contains(b));
    require(admission.contains(c));
    require(admission.contains(d));
    require(admission.stats().lru_evictions == 1);

    // Explicit admission bypasses frequency and capacity hints, and records a
    // distinct metric without changing the LRU item count.
    const size_t before_size = admission.size();
    require(admission.observe_explicit(b));
    require(admission.size() == before_size);
    require(admission.stats().explicit_admissions == 1);

    const AdmissionStats stats = admission.stats();
    require(stats.first_observations == 5);
    require(stats.repeat_admissions == 2);
    require(stats.items == 3);
    require(stats.admitted_items == 1);
    require(stats.capacity == 3);

    PrefixAdmission disabled(0);
    require(!disabled.observe_cold(a));
    require(!disabled.contains(a));
    require(disabled.stats().boundaries_observed == 1);
    require(disabled.observe_explicit(a));
    require(disabled.stats().explicit_admissions == 1);

    return 0;
}
