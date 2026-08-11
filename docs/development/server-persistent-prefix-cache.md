# Persistent Prefix Cache for llama-server

Status: draft specification

Target: the local multi-model llama.cpp branch

Primary target models: Qwen 3.5 and Qwen 3.6 hybrid text models

## 1. Purpose

llama-server currently keeps complete prompt states in host RAM. Each saved prompt owns one complete serialized state. Long prompts with related prefixes therefore repeat most of the same KV data. The cache is also lost when a model unloads or the server restarts.

This change replaces that representation, when explicitly enabled, with a reusable prefix cache made of:

1. token-aligned attention KV blocks;
2. recurrent-state snapshots at the same token boundaries;
3. a bounded host-RAM hot tier;
4. a bounded persistent disk tier.

The design is inspired by the updated oMLX cache architecture, but it follows llama.cpp memory and sequence semantics. It does not copy MLX cache objects or assume that MLX and ggml have the same storage rules.

## 2. Required results

The implementation is correct only if all of the following are true:

- A repeated text prefix can be restored after its original slot is gone.
- A repeated text prefix can be restored after its model is unloaded and loaded again.
- Hybrid models restore both attention KV and recurrent state from the same exact boundary.
- Shared prefixes store attention KV bytes once instead of once per complete prompt.
- Cache files are rejected when the model, adapter, context, memory layout, or cache format is incompatible.
- A partial write, corrupt file, missing sidecar, or failed restore causes a safe cache miss.
- A failed restore leaves no partial state in the destination sequence.
- Host staging memory, hot-cache memory, disk use, and queued writes have hard limits.
- Cache hits and misses are observable without reading debug logs.
- The feature is disabled by default and does not change the existing prompt cache path unless configured.

## 3. Non-goals

Version 1 does not:

- quantize KV data or recurrent state;
- change model weights, inference kernels, Metal shaders, or token sampling;
- promise bit-for-bit identical logits across different batch shapes;
- reuse a cache across different models, model files, adapter states, or context configurations;
- cache multimodal prefixes;
- cache sequences after context shifting or position remapping;
- cache a sequence while speculative decoding is active;
- cache a prompt whose adapter state changes during the prefix, including aLoRA;
- cache SWA attention unless `swa_full` is active;
- cache generic multidimensional positions; the only exception is the explicit
  canonical text-only M-RoPE contract `[p, p, p, 0]`;
- support an attention layout until its range-save and additive-restore behavior is tested;
- make cache files portable between different llama.cpp cache format versions.

These limits are correctness boundaries. An unsupported request uses normal prompt evaluation.

## 4. Terms

### 4.1 Prefix block

A prefix block covers a fixed, half-open token-position range:

```text
[position_start, position_end)
```

The default block size is 2048 positions. Only complete blocks are persisted. A shorter unmatched suffix is evaluated normally.

### 4.2 Chain key

Attention KV values for a token depend on the prefix before that token. A block cannot be identified by its local token IDs alone.

Each block key is therefore chained:

```text
block_key[0] = SHA256(domain || cache_signature || position_data[0] || token_data[0])
block_key[n] = SHA256(domain || cache_signature || block_key[n - 1] ||
                      position_data[n] || token_data[n])
```

`domain` includes the cache format version and artifact kind. Each token is encoded as a fixed-width little-endian signed `int32_t` (`llama_token`). Each position is encoded as a fixed-width little-endian signed `int32_t` (`llama_pos`). Strings use a fixed-width length followed by UTF-8 bytes. The input must not depend on C++ object layout, variable-length integer encoding, or `std::hash`.

### 4.3 Attention artifact

An attention artifact contains the serialized attention memory cells for one prefix block. It does not contain recurrent state.

### 4.4 Recurrent sidecar

A recurrent sidecar contains the complete recurrent state immediately after the last token of a prefix block has been decoded. It is keyed by the final block key plus the recurrent layout signature.

It is a boundary snapshot, not a token slice. A sidecar from one boundary must never be combined with KV blocks from another boundary.

### 4.5 Cache signature

The cache signature identifies every input that can change the meaning or layout of serialized state. Section 8 defines its required fields.

## 5. Core invariants

### 5.1 Boundary invariant

For boundary `B`, the persisted state is valid only when:

```text
attention blocks cover [0, B) without gaps
AND
the recurrent sidecar was captured after position B - 1 was decoded
AND
the prompt chain key at B matches the sidecar key
```

For an attention-only model, the sidecar condition is omitted.

### 5.2 Empty-destination invariant

Persistent restore starts with an empty destination sequence. The restore operation:

1. clears the destination sequence;
2. restores attention blocks in increasing position order;
3. restores the recurrent sidecar;
4. verifies the restored position range;
5. publishes the restored token prefix to the slot.

If any step fails, it clears the destination sequence again and reports a miss.

### 5.3 No mixed generations

An artifact is immutable after publication. Writers use a temporary file in the target directory, flush and close it, then publish it with an atomic no-replace operation. If another writer already published the same key, the existing valid artifact wins.

Readers never open temporary files.

Plain POSIX `rename()` is not sufficient because it may replace an existing file. The implementation uses a platform no-replace primitive when available. The fallback uses an exclusive final-path claim and never exposes the final name before the complete bytes are durable.

### 5.4 Bounded-memory invariant

The total bytes held by:

- hot attention artifacts;
- hot recurrent sidecars;
- pending serialized writes;
- in-flight disk reads;

must each be accounted for. The writer queue has both an item limit and a byte limit. When it reaches either limit, capture applies bounded backpressure or skips persistence. It must not grow without limit.

### 5.5 Fail-closed compatibility

Unknown format fields, unknown memory components, unsupported flags, signature mismatches, invalid sizes, invalid position ranges, and checksum failures are cache misses. They are not warnings followed by a best-effort restore.

## 6. llama.cpp state API

The existing `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` flag does not provide the required contract:

- in `llama_memory_hybrid`, it omits attention and keeps recurrent state;
- in `llama_memory_hybrid_iswa`, it is passed to both children;
- it does not select a position range;
- its serialized bytes do not identify their component kind.

The implementation must add explicit component and range APIs. The exact C symbol names may follow project naming conventions, but the contract must be equivalent to:

```c
enum llama_state_seq_component {
    LLAMA_STATE_SEQ_COMPONENT_ATTENTION = 1 << 0,
    LLAMA_STATE_SEQ_COMPONENT_RECURRENT = 1 << 1,
};

#define LLAMA_STATE_SEQ_FLAGS_MROPE_TEXT (1u << 2)

struct llama_state_seq_range {
    llama_pos p0;
    llama_pos p1;
};

uint32_t llama_state_seq_components(struct llama_context * ctx);

uint32_t llama_state_seq_capabilities(struct llama_context * ctx);

size_t llama_state_seq_get_size_range(
        struct llama_context * ctx,
        llama_seq_id seq_id,
        uint32_t components,
        struct llama_state_seq_range range,
        llama_state_seq_flags flags);

size_t llama_state_seq_get_data_range(
        struct llama_context * ctx,
        uint8_t * dst,
        size_t size,
        llama_seq_id seq_id,
        uint32_t components,
        struct llama_state_seq_range range,
        llama_state_seq_flags flags);

size_t llama_state_seq_set_data_range(
        struct llama_context * ctx,
        const uint8_t * src,
        size_t size,
        llama_seq_id dest_seq_id,
        uint32_t components,
        struct llama_state_seq_range range,
        llama_state_seq_flags flags);
```

The public header is C-compatible, but the implementation remains C++.

The capability result must distinguish at least:

- component presence;
- attention range save;
- additive attention range restore;
- recurrent boundary save;
- recurrent replace restore;
- safe restore with unified KV;
- canonical text-only M-RoPE range save and restore;
- supported on-device serialization.

Component presence alone does not make a context eligible. The server requires every capability used by its restore transaction.

### 6.1 Component behavior

- Plain KV models expose `ATTENTION`.
- Plain recurrent models expose `RECURRENT`.
- `llama_memory_hybrid` exposes both.
- A composite attention implementation exposes `ATTENTION` only after all of its child caches support the same range contract.
- Requests for an unsupported component return zero and log one clear error.
- Passing both components is allowed for compatibility tests, but persistent storage writes them as separate artifacts.

### 6.2 Attention range behavior

An attention range save includes only cells for `seq_id` whose logical positions are in `[p0, p1)`.

The save verifies that every required position in the range is present exactly once. A missing, duplicate, SWA-evicted, or remapped position rejects the save.

The range-envelope v2 contract supports:

- ordinary one-dimensional sequential positions with no special flag;
- canonical text-only M-RoPE positions `[p, p, p, 0]` with the explicit `LLAMA_STATE_SEQ_FLAGS_MROPE_TEXT` provenance flag and advertised capability.

The M-RoPE flag is not generic multidimensional support. Save validates the retained extended coordinates, restore reconstructs the fixed fourth coordinate as zero, and the server may use the flag only for a media-free token request. Arbitrary embedding or multimodal position planes remain unsupported.

An attention range restore is additive. It must:

- reject an overlap with existing cells for the destination sequence;
- preserve serialized logical positions;
- reject a range that does not match the artifact header;
- fail without clearing other sequences;
- return the exact number of bytes consumed.

Range save and restore always use one concrete `seq_id >= 0`. The `seq_id == -1` whole-cache path is not part of this API and must not be reachable from persistent restore.

In unified KV, restore may:

- share an existing cell only when the chain identity and logical position prove that its K/V bytes are identical, then add the destination sequence tag without rewriting the bytes; or
- allocate an empty cell, write the K/V bytes, and add the destination sequence tag.

It must not overwrite a cell owned by another sequence, remove another sequence's tag, or treat untagged stale bytes as a valid match. If safe cells are unavailable, restore fails and the request uses normal evaluation.

The first implementation target is `llama_kv_cache` inside `llama_memory_hybrid`. Other KV implementations remain unsupported until they pass the same tests.

### 6.3 Recurrent behavior

A recurrent save ignores token slicing and requires a boundary:

```text
range.p0 == 0
range.p1 == boundary
```

The serialized state is the complete recurrent state for `seq_id` at that boundary. Restore replaces the recurrent state for the destination sequence.

The API must reject a recurrent save when the sequence memory does not end at `range.p1 - 1`.

The API must also reject the save when the sequence has a pending recurrent rollback plane. The live recurrent rollback index for `seq_id` must be zero.

Recurrent replace restore must use the existing metadata reconstruction semantics. It rebuilds the destination tail, sets the destination cell source to itself, restores the position, and resets the rollback index to zero. Copying tensor rows without rebuilding `tail`, `src`, and rollback metadata is invalid.

`get_size_range` for `RECURRENT` returns exactly one logical state row per non-null recurrent layer for one destination sequence. It does not reserve every rollback plane.

### 6.4 Stream envelope

Every range state stream starts with a versioned envelope:

```text
magic
format_version
component_mask
range_flags
source_seq_id
position_start
position_end
payload_size
payload_checksum
payload
```

The envelope prevents a recurrent blob from being loaded as attention data and prevents the caller from supplying a false range.

The attention payload keeps the existing per-layer validation fields, including layer count, tensor type, row size, element size, and embedding dimensions. A matching outer signature does not replace those checks.

Existing full-sequence state APIs and file versions keep their current behavior. The new range format has its own magic and version.

## 7. Capture lifecycle

### 7.1 Eligibility

A sequence is eligible only when:

- prompt caching is enabled for the request;
- the cache signature is complete;
- the input is pure text;
- token positions are sequential and start at zero;
- the model uses one-dimensional positions, or advertises the canonical text-only M-RoPE range capability and the request supplies only text tokens;
- no context shift or position rewrite has happened;
- SWA is absent or `swa_full` is active;
- required memory components report range-cache support;
- active adapters match the signature captured when the chain began;
- no adapter scale or activation transition occurred inside the prefix;
- speculative decoding is inactive;
- the recurrent rollback index is zero.

Eligibility is checked again at every boundary.

### 7.2 Decode boundary

The server must never infer that a queued token is already in model memory.

To capture boundary `B`, the prompt batch builder stops at `B`. After `llama_decode()` succeeds, the server verifies:

```text
llama_memory_seq_pos_max(memory, seq_id) == B - 1
```

For hybrid memory this query returns the minimum of the attention and recurrent maxima, so both components must have reached the boundary. The server also verifies that the recurrent rollback index is zero. It then captures:

- attention range `[B - block_size, B)`;
- recurrent range `[0, B)`, when present;
- the chain key and cache signature.

Generation may also create a boundary after the generated token has been decoded and added to the sequence. The same position check applies.

### 7.3 Host staging and writer

Device-to-host serialization occurs on the inference thread unless the backend provides a proven asynchronous transfer contract. Disk I/O occurs on one bounded background writer per loaded model context.

The capture path:

1. computes the exact required sizes;
2. reserves pending bytes;
3. serializes into owned host buffers;
4. enqueues immutable write jobs;
5. releases the inference thread;
6. writes and publishes artifacts in the background.

If the reservation cannot be obtained within a short bounded wait, the capture is skipped and a counter is incremented. Inference must continue.

## 8. Compatibility signature

The signature is serialized canonically and hashed. It must include:

- range-cache format version;
- llama.cpp state ABI version;
- model architecture;
- canonical identity for every GGUF shard: resolved path, file size, and nanosecond modification time where available;
- a content digest for every GGUF shard, computed before persistent lookup is enabled;
- tensor and memory layout identity needed by the state reader;
- target or draft role;
- KV data types;
- context size;
- batch-related settings that change cached numerical state;
- RoPE and position settings;
- SWA settings;
- recurrent rollback configuration;
- active LoRA and adapter file identities, order, and scales;
- control-vector identity and range;
- embedding mode or pooling mode when relevant;
- multimodal projector identity if multimodal caching is added later.

The model alias, UI name, server port, slot number, and request ID are not part of the signature.

The target model and draft model always use different namespaces and artifacts.

The server may cache a shard digest beside the cache index and reuse it when the canonical path, file identity, size, and nanosecond modification time are unchanged. It must recompute the digest when any of those values change. A user-supplied namespace may further separate stores, but it cannot replace the model identity fields.

## 9. Persistent store

### 9.1 Directory layout

The disk root is user configured. The server creates:

```text
<root>/
  v1/
    <signature-hash>/
      attention/
        <key[0:2]>/<key[2:4]>/<key>.bin
      recurrent/
        <key[0:2]>/<key[2:4]>/<key>.bin
      quarantine/
```

The implementation refuses a symlinked cache root, signature directory, shard directory, destination file, or temporary file. It does not follow cache-owned symlinks during scan, clear, or eviction.

### 9.2 Artifact header

Each file contains:

- file magic and version;
- artifact kind;
- complete cache signature hash;
- chain key;
- parent chain key for attention artifacts;
- position start and end;
- token count;
- uncompressed payload size;
- payload checksum;
- creation timestamp;
- payload.

Payloads are stored in llama.cpp's native serialized representation. Version 1 does not compress or quantize them.

### 9.3 Discovery

The chain keys are computed directly from an incoming prompt. No prompt-to-file manifest is required for lookup.

At model load, the server scans only the active signature directory. It validates file names and bounded headers without loading payloads. Invalid entries move to `quarantine` when safe, or are ignored.

### 9.4 LRU and limits

Disk use has a hard byte limit. The in-memory index tracks file size, last successful access, artifact kind, and pin count.

- A reader pins an entry before opening it.
- Eviction skips pinned entries and temporary files.
- Access timestamps are written back in batches, not for every hit.
- Eviction removes the oldest unpinned artifacts until both the byte and file-count limits are satisfied.
- Removing an attention block does not require eagerly removing descendants. A later lookup stops at the first missing block.
- A recurrent sidecar whose final block is missing is eligible for early eviction.

### 9.5 Restart and crash behavior

Temporary files include the process ID and a random suffix. Startup removes stale temporary files only inside the validated active signature directory.

A crash can lose queued writes. It cannot publish a partial artifact.

## 10. Hot RAM tier

The existing `--cache-ram` byte budget becomes the hot serialized-artifact budget when the persistent cache is enabled.

The hot tier:

- uses the same keys and validated envelope as disk;
- stores immutable shared byte buffers;
- keeps attention blocks and recurrent sidecars independently;
- uses byte-based LRU;
- pins entries during restore;
- coalesces concurrent loads of the same key;
- never consumes an entry on hit.

When persistent caching is disabled, existing whole-prompt RAM cache behavior remains unchanged.

## 11. Lookup and restore

For an incoming eligible prompt:

1. compute complete block chain keys from the prompt;
2. find the longest contiguous attention chain available in hot RAM or on disk;
3. for a hybrid model, require the recurrent sidecar at that exact final key;
4. if it is missing or invalid, walk back one block at a time;
5. reserve enough destination memory;
6. clear the destination sequence;
7. restore attention blocks in order;
8. restore the recurrent sidecar;
9. verify sequence positions and component state;
10. publish the matching token prefix to the slot;
11. evaluate the unmatched suffix normally.

An attention-only model stops at step 2 and omits sidecar operations.

Restoring from the persistent cache must not delete or consume artifacts.

llama-server requires at least one prompt token to be evaluated to produce the logits used for generation. Therefore, for completion requests that require fresh logits, the restored boundary must be strictly shorter than the input prompt:

```text
restored_boundary < prompt_token_count
```

If the deepest cached boundary equals the complete prompt length, lookup walks back to an earlier boundary. Version 1 does not persist logits. This may require reevaluating one full block when the prompt length is exactly block aligned, but it avoids invalid recurrent rollback.

If walkback reaches position zero without a usable attention chain and matching sidecar, the request performs normal prompt evaluation with no restore.

## 12. Interaction with existing server features

### 12.1 Multi-model residency

Each loaded target or draft context owns its hot tier and writer. Disk artifacts remain after the context unloads. Reloading the same compatible model attaches to the same signature directory.

The router must shut down a model writer cleanly before destroying its context. Shutdown has a bounded flush timeout. Jobs not flushed in time are discarded safely.

### 12.2 Slots and unified KV

Artifacts never contain a destination slot ID as identity. Serialized source sequence IDs are remapped through the state reader.

Restore is tested with:

- non-unified per-slot KV;
- unified KV;
- two concurrent slots sharing the same prefix.

If an implementation cannot make additive range restore safe for unified KV, persistent caching is disabled for that context rather than clearing unrelated sequences.

### 12.3 Speculative decoding

Version 1 treats speculative decoding as ineligible because target memory, draft memory, and implementation-specific speculative state must resume at one coherent boundary. Restoring target and draft KV alone is not sufficient.

A later version may add target, draft, and `common_speculative` state artifacts. It can enable a hit only when all required states restore to the same boundary and the existing speculative subsystem proves continuation from that state.

### 12.4 LoRA and control vectors

Changing an adapter scale or control vector ends the current cache chain. The next boundary uses a new signature. No artifact is reused across the change.

Version 1 requires one fixed adapter and control-vector configuration for the complete cached prefix. aLoRA is ineligible because its activation transition occurs inside the prefix.

### 12.5 Context checkpoints

Speculative and rollback checkpoints remain runtime objects. The persistent cache does not store the current `common_prompt_checkpoint` list in version 1.

A persistent restore clears stale checkpoints and starts a new checkpoint history from the restored boundary.

### 12.6 Existing slot save files

The `/slots/{id}?action=save|restore` file format and behavior remain unchanged. Persistent prefix artifacts are an internal server cache and are not slot-save files.

## 13. Configuration

Proposed server options:

```text
--cache-disk PATH
--cache-disk-size N
--cache-block-size N
--cache-write-buffer N
```

Semantics:

- `--cache-disk PATH`: enables persistent prefix caching at `PATH`.
- `--cache-disk-size N`: maximum disk use in MiB. It is required when disk caching is enabled.
- `--cache-block-size N`: positions per block. Default 2048. Minimum 256. Must be a power of two.
- `--cache-write-buffer N`: maximum pending serialized bytes in MiB. A conservative default is derived from one block and one recurrent snapshot.
- `--cache-ram N`: existing option; becomes the hot-tier budget while persistent caching is enabled.
- `--cache-ram 0`: disables the hot tier but does not disable disk caching.

Invalid or unsafe configuration fails server startup with a clear error. It does not silently choose an unbounded cache.

## 14. Observability and control

The server exposes per-model counters and gauges:

- eligible requests;
- ineligible requests by reason;
- lookup attempts;
- attention block hits from RAM;
- attention block hits from disk;
- recurrent sidecar hits from RAM;
- recurrent sidecar hits from disk;
- walkbacks;
- corrupt or incompatible artifacts;
- restore failures;
- capture attempts;
- capture skips due to backpressure;
- bytes read and written;
- hot bytes and artifact count;
- disk bytes and artifact count;
- pending write bytes and peak bytes;
- tokens restored;
- tokens evaluated after restore;
- time spent hashing, reading, restoring, serializing, and writing.

Model status JSON includes a compact `prefix_cache` object. A control endpoint supports:

- stats;
- clear hot tier;
- clear this model signature's disk tier.

Disk clear is serialized with readers and writers and never follows symlinks. The UI may expose the live size, hit state, and clear action in the existing model controls. It must not add permanent explanatory text.

## 15. Implementation plan

### Phase 0: frozen test fixtures and benchmarks

1. Record current behavior for full state save and restore.
2. Add a small hybrid test model fixture or deterministic synthetic memory fixture.
3. Add baseline measurements for cold prefill, current RAM prompt-cache hit, peak host RAM, and state size.
4. Record real baselines for the resident Qwen 3.5 9B and Qwen 3.6 35B GGUF models.

Exit gate:

- Baselines are reproducible.
- Tests detect a recurrent state captured at the wrong token boundary.

### Phase 1: component-aware range serialization

1. Add component capability introspection.
2. Add the versioned range-state envelope.
3. Add position filtering to `llama_kv_cache` state write.
4. Add additive, overlap-rejecting KV range restore.
5. Add exact-boundary recurrent save and replace restore.
6. Route component selection through `llama_memory_hybrid`.
7. Leave `llama_memory_hybrid_iswa` unsupported until its child layout has dedicated tests.
8. Preserve all existing state APIs and tests.

Likely files:

- `include/llama.h`
- `src/llama-context.cpp`
- `src/llama-memory.h`
- `src/llama-memory-hybrid.cpp`
- `src/llama-memory-hybrid-iswa.cpp`
- `src/llama-kv-cache.cpp`
- `src/llama-memory-recurrent.cpp`
- `tests/test-save-load-state.cpp`
- a new focused range-state test

Exit gate:

- A hybrid sequence can be split into attention blocks plus one recurrent sidecar, restored into a different sequence ID, and continued with matching output.
- Corrupt, overlapping, wrong-component, wrong-range, and wrong-boundary inputs fail closed.
- Existing state tests still pass.

### Phase 2: store and hot tier

1. Add canonical chain-key encoding and SHA-256.
2. Add cache-signature construction.
3. Add artifact headers and validation.
4. Add the bounded hot LRU.
5. Add the persistent store, atomic publication, startup scan, pinning, and eviction.
6. Add bounded writer staging and shutdown.

Likely files:

- new `tools/server/server-prefix-cache.h`
- new `tools/server/server-prefix-cache.cpp`
- `tools/server/CMakeLists.txt`
- focused unit tests for storage and concurrency

Exit gate:

- Restart reuse works.
- Concurrent identical writers publish one valid artifact.
- Eviction cannot remove an in-use artifact.
- A simulated crash before rename leaves no visible artifact.
- Queue and hot-tier limits hold under stress.

### Phase 3: server capture and restore

1. Add configuration.
2. Split prompt batches at eligible block boundaries.
3. Capture after successful decode.
4. Restore the deepest complete compatible boundary at request start.
5. Add safe walkback and transactional cleanup.
6. Reject speculative decoding and mid-prefix adapter transitions.
7. Clear stale runtime checkpoints after restore.

Likely files:

- `common/common.h`
- `common/arg.cpp`
- `tools/server/server-context.cpp`
- `tools/server/server-task.h`
- `tools/server/server-task.cpp`
- `tools/server/server-schema.cpp`
- `tools/server/README.md`
- server unit and integration tests

Exit gate:

- Two slots reuse one shared prefix.
- A model unload and reload reuses its disk prefix.
- A failed restore leaves the slot empty and completes by normal evaluation.
- Unsupported requests take the existing path.

### Phase 4: controls, UI, and metrics

1. Expose stats and clear operations.
2. Add model-level cache state to router status.
3. Add compact controls to the existing model UI.
4. Add logging for one-line hit, miss, walkback, and corruption summaries.

Exit gate:

- The UI reports actual server state.
- Clear operations are race-safe and model-scoped.
- No cache control can delete outside the configured root.

### Phase 5: real-model qualification

Run cold, warm, restart, concurrency, and corruption tests against:

- Qwen 3.5 9B;
- Qwen 3.6 35B;
- one attention-only GGUF;
- one speculative configuration as a negative eligibility test.

Measure:

- cold prompt throughput;
- warm restore throughput;
- time to first generated token;
- generated continuation equivalence under the existing batch nondeterminism contract;
- peak and steady host RAM;
- VRAM before and after capture;
- bytes stored for several prompts sharing long prefixes;
- reload latency after model eviction.

Exit gate:

- Warm requests evaluate only the unmatched suffix.
- Shared-prefix disk growth is approximately linear in unique blocks plus recurrent boundary snapshots, not in the sum of full prompt lengths.
- Persistent caching does not increase steady VRAM.
- Cold prompt throughput regression from boundary capture is below 5 percent at the default block size, or capture is automatically reduced or disabled.
- No correctness fallback requires manual cache deletion.

## 16. Test matrix

### Core serialization

- attention-only range round trip;
- recurrent-only boundary round trip;
- hybrid split round trip;
- source and destination sequence IDs differ;
- two restored ranges are additive;
- overlapping ranges fail;
- missing and duplicate positions fail;
- a gap is detected by the server restore transaction;
- wrong component, range, version, and checksum fail;
- truncated data fails;
- recurrent save with a nonzero rollback index fails;
- recurrent restore rebuilds tail, source, and rollback metadata;
- recurrent size reports one logical state row, not all rollback planes;
- the `seq_id == -1` range path is rejected;
- existing per-layer tensor layout checks remain active;
- existing full state and partial state behavior is unchanged.

### Store

- same prefix produces the same chain;
- same local tokens after a different parent produce a different key;
- a model or adapter signature change produces a different namespace;
- atomic duplicate writers;
- interrupted temporary write;
- corrupt header and corrupt payload;
- symlinked root, directory, and file rejection;
- disk byte limit;
- hot byte limit;
- writer byte limit;
- eviction with pinned readers;
- restart scan;
- access-time batching.

### Server

- exact block-aligned prompt walks back to preserve fresh logits;
- partial hit;
- sidecar walkback;
- missing middle attention block;
- corrupt deepest sidecar;
- empty slot and occupied slot;
- two concurrent slots;
- unified KV;
- unified KV restore preserves unrelated sequence tags and bytes;
- SWA without `swa_full` is ineligible;
- a model with `n_pos_per_embd() > 1` is ineligible;
- model unload and reload;
- speculative decoding is ineligible;
- LoRA change;
- context shift makes the request ineligible;
- media makes the request ineligible;
- backpressure skip;
- clear while idle;
- clear while a read or write is active.

## 17. Adversarial review

This section treats the preceding design as hostile input and looks for ways it can return incorrect model state, corrupt cache data, stall inference, or claim optimization it did not achieve. It includes a separate read-only Pi review with GLM-5.2. Each accepted external finding was checked against the current source before it was added here.

### Finding A: token-local hashing would silently return wrong KV

Severity: critical

Attack: use the same 2048-token block after two different earlier prefixes. If the key contains only the local tokens, the server restores KV computed from the wrong hidden states.

Resolution in this specification: every block key includes its parent key and the complete cache signature. A block belongs to one exact prefix chain.

### Finding B: a correct KV chain with the wrong recurrent sidecar is still wrong

Severity: critical

Attack: restore attention blocks through boundary `B`, then choose the nearest recurrent snapshot from boundary `B - block_size` or from a sibling prompt.

Resolution: a hybrid hit exists only when the sidecar is keyed by the exact final block key. Missing or invalid sidecars force whole-block walkback.

### Finding C: capture timing can be off by one decode

Severity: critical

Attack: the server appends tokens to `slot.prompt` before `llama_decode()`, captures at a nominal boundary, and stores recurrent state that ends before that boundary.

Resolution: capture only after successful decode and verify the actual sequence position before serialization. Tests must intentionally capture before decode and prove rejection.

### Finding D: the proposed compatibility signature can miss weight changes

Severity: high

Attack: replace a GGUF in place with a different file that has the same architecture and tensor layout.

Resolution: the signature includes a content digest for every shard. Path, file identity, size, and modification time may validate a cached digest, but they do not replace it.

### Finding E: `PARTIAL_ONLY` cannot be reused as a new component API

Severity: high

Attack: pass the existing flag through a composite memory type and assume it always means recurrent-only. The two hybrid implementations already interpret it differently.

Resolution: add explicit component selection and a self-describing range envelope. Do not overload the old flag.

### Finding F: sliding-window and composite attention can overflow or reconstruct invalid state

Severity: high

Attack: load every historical SWA block into a cache that only owns a rotating window, or omit base-attention cells needed by a composite cache.

Resolution: version 1 requires `swa_full` for every SWA layout and supports only memory implementations that advertise and pass the range contract. `llama_memory_hybrid_iswa` is disabled until it has a component-specific storage plan and tests. The server must not infer support from the model name.

### Finding G: restore is not transactional at the llama state layer

Severity: high

Attack: load several valid blocks, fail on a corrupt later block, and continue inference with a partial prefix.

Resolution: the server owns the transaction. Any failure clears the destination sequence and its runtime checkpoints before falling back to normal evaluation. Other sequences must remain intact.

### Finding H: asynchronous writes can become a second unbounded prompt cache

Severity: high

Attack: submit long prompts faster than the disk writer can publish them. Each boundary holds one KV block and one large recurrent snapshot in RAM.

Resolution: pending writes have independent byte and item limits. Reservation happens before serialization. A bounded wait may apply backpressure; after it expires, capture is skipped.

### Finding I: recurrent sidecars may dominate disk even after KV deduplication

Severity: medium

Attack: save a fixed-size recurrent snapshot every 256 tokens for many related prompts. Disk use grows quickly even though KV blocks are shared.

Resolution: default to 2048 positions, enforce a minimum of 256, account for sidecars separately, and let eviction prefer orphaned or low-value sidecars. Qualification reports KV bytes and recurrent bytes separately. Adaptive boundary spacing is a later optimization only after the fixed format is proven.

### Finding J: persistent restore may be slower than prompt evaluation for short prefixes

Severity: medium

Attack: restore a small prefix from a slow disk and add hashing, validation, and host-to-device copy cost.

Resolution: only complete blocks are eligible, default block size is 2048, and metrics keep separate restore and evaluation time. A model may maintain a measured minimum useful restore length. This is a performance decision, never a correctness exception.

### Finding K: draft and target can resume at different boundaries

Severity: high

Attack: the target has a deep hit while the draft has only a shallow hit, then speculative decoding assumes aligned contexts.

Resolution: version 1 rejects speculative configurations. Later support requires target, draft, and implementation-specific speculative state at one boundary.

### Finding L: multimodal placeholder tokens collide

Severity: critical

Attack: two different images produce the same placeholder token sequence and therefore the same chain key.

Resolution: version 1 rejects requests that contain media. Future support requires a stable digest of the actual encoded media contribution and its position mapping.

### Finding M: clearing or eviction can escape the cache root

Severity: critical

Attack: replace a cache subdirectory with a symlink before a recursive clear.

Resolution: validate every owned path, refuse symlinks, scope deletion to a previously validated model-signature directory, and use file-descriptor-relative operations where the platform supports them.

### Finding N: the design may improve disk use but not VRAM

Severity: medium

Attack: advertise lower VRAM although serialized cache artifacts live in host RAM and SSD while active model KV allocation remains unchanged.

Resolution: the feature claims lower repeated prefill work and deduplicated host/disk cache storage. It does not claim smaller active KV allocation. Real-model qualification reports VRAM separately and requires no steady increase.

### Finding O: cache-hit output comparisons can demand false bit equality

Severity: medium

Attack: compare a restored continuation with a cold continuation that used a different prompt batch shape and reject valid behavior already documented by llama-server.

Resolution: use the server's existing cache nondeterminism contract. Core state tests use controlled batch shapes; end-to-end tests compare tokens or logits with documented tolerances.

### Finding P: an exact full-prompt restore may not provide generation logits

Severity: critical

Attack: restore memory through the final prompt token and begin sampling without evaluating any token. llama-server currently forces at least one prompt token through decode for this reason.

Resolution: version 1 restores only a boundary strictly shorter than a completion prompt. A block-aligned exact prompt walks back one block. Persisting logits is a separate future format.

### Finding Q: component presence can be mistaken for safe range support

Severity: high

Attack: a composite memory reports attention data but cannot save one position range or add it to an occupied unified cache safely.

Resolution: the core API exposes explicit capability bits. Eligibility requires the exact save, restore, and unified-KV capabilities used by the transaction.

### Finding R: ordinary rename can overwrite an existing artifact

Severity: high

Attack: two writers publish the same key. On macOS, plain `rename()` can replace the first complete file even though the design says the existing artifact wins.

Resolution: publication uses an atomic no-replace primitive or an exclusive final-path claim. Plain overwrite rename is forbidden.

### Finding S: a pending recurrent rollback plane can pair the wrong state with KV

Severity: critical

Attack: capture after a partial recurrent rollback while `rs_idx[seq_id]` selects an older state plane. The sidecar is labeled with boundary `B`, but its tensors represent an earlier boundary.

Resolution: recurrent save is eligible only when the rollback index is zero. The API rejects any other value.

### Finding T: copying recurrent tensors without metadata rebuild breaks the next decode

Severity: critical

Attack: restore recurrent tensor rows but leave `tail`, `src`, or rollback metadata from the old destination sequence.

Resolution: replace restore must follow the existing recurrent metadata reconstruction semantics and reset the rollback index.

### Finding U: unified KV restore can overwrite another sequence

Severity: critical

Attack: treat a destination sequence cleared of tags as if its former cells are free, then overwrite cells that still belong to another sequence in the unified cache.

Resolution: restore allocates empty cells or shares a cell only when chain identity proves the bytes are identical. It preserves every unrelated tag and byte. A capacity failure becomes a cache miss.

### Finding V: the whole-cache sequence ID path defeats transaction isolation

Severity: high

Attack: call range restore with `seq_id == -1`. Existing state readers clear the complete memory cache on failure.

Resolution: persistent range APIs require one concrete nonnegative sequence ID. The whole-cache path is unreachable.

### Finding W: generic multi-dimensional positions are not restored safely

Severity: high

Attack: persist scalar ranges for a context whose KV cells also use extended position coordinates. A generic single-sequence restore cannot reconstruct an arbitrary fourth coordinate because the KV cell metadata retains only the primary, x, and y values.

Resolution: the range-envelope v2 format remains fail-closed for generic multidimensional positions. It adds one explicit exception for pure-text M-RoPE, whose position contract is `[p, p, p, 0]`. Save validates the stored primary/x/y values, the envelope binds the provenance flag, and restore reconstructs the fourth coordinate as zero. Media, arbitrary embeddings, and any non-canonical extended coordinates are rejected.

### Reviewer verdict

Proceed only with the following gates intact:

1. Explicit component and range serialization lands before persistent storage.
2. Hybrid restore requires the exact recurrent sidecar.
3. SWA without `swa_full`, `llama_memory_hybrid_iswa`, generic multi-dimensional positions, media, shifted contexts, and unknown layouts fail closed. Canonical text-only M-RoPE requires its explicit v2 flag and capability.
4. Model and adapter identity are part of the cache namespace.
5. Capture occurs after decode with a checked memory boundary.
6. Recurrent capture requires a zero rollback index and recurrent restore rebuilds metadata.
7. Unified-KV restore preserves unrelated sequence bytes and tags.
8. Range APIs reject `seq_id == -1`.
9. Restore failure clears the destination sequence.
10. All RAM and disk paths are byte bounded.
11. Completion restore leaves at least one token for fresh logits.
12. Artifact publication cannot replace an existing valid file.
13. Real Qwen tests must prove continuation correctness and actual suffix skipping.

Removing any of these gates turns the cache from an optimization into a source of silent wrong-model state.
