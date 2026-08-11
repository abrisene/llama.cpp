# Selective Prefix Capture and Explicit Precache

Status: implementation specification
Depends on: `server-persistent-prefix-cache.md`
Target: the local multi-model llama.cpp server and its bundled UI

## 1. Purpose

The persistent prefix cache already makes a repeated long prompt substantially
faster after model unload or process restart. Its remaining cost is paid on the
first evaluation: every complete block is copied from Metal to host memory,
serialized, and queued for storage even when the prefix is never reused.

This change makes capture demand-driven.

It adds:

- selective automatic admission;
- an explicit per-request capture override;
- a UI action to precache the current conversation prefix;
- deferred attention serialization after the response;
- sparse recurrent sidecars for long prompts;
- metrics that separate observation, admission, staging, and publication.

It does not change cache lookup, compatibility signatures, exact restoration,
model weights, KV precision, or generation sampling.

## 2. Required results

1. A one-off long prompt does not serialize persistent attention or recurrent
   state under the default policy.
2. A prefix observed twice becomes eligible for capture on its second cold
   evaluation.
3. A user can explicitly precache the current conversation from the UI without
   adding a generated assistant message to the conversation.
4. Explicit capture bypasses admission frequency, but never bypasses the
   existing correctness and compatibility gates.
5. Recurrent state required for a boundary is copied while that boundary is
   live. Attention serialization is deferred until after the response has been
   handed to the client.
6. Long prompts do not store a recurrent sidecar for every attention block.
7. Restore remains fail-closed and produces the same continuation as ordinary
   evaluation.
8. Persistent capture adds no steady VRAM allocation.
9. Every queue and staging path has a hard byte and item bound.
10. The existing `always` behavior remains available for controlled workloads.

## 3. Non-goals

This version does not:

- benchmark or automatically tune Metal batch sizes, ubatch sizes, or threads;
- persist automatic-admission observations across model-process restarts;
- perform concurrent llama-context reads while inference is running;
- keep hot cache artifacts on Metal;
- compress or quantize cache payloads;
- cache media, speculative state, shifted contexts, unsupported SWA layouts, or
  arbitrary multidimensional positions;
- promise that an explicitly requested precache succeeds when the store is
  full, the model is unloading, or the request is otherwise ineligible.

The M5 Max benchmark matrix is specified in section 14 and intentionally
deferred.

## 4. Capture modes

Add:

```text
--cache-capture-mode MODE
--cache-admission-items N
--cache-recurrent-stride N
```

`MODE` is one of:

- `always`: preserve the current behavior. Every eligible complete boundary is
  captured.
- `repeat`: default when persistent caching is enabled. The first cold
  observation records a bounded admission hint. The second cold observation
  admits the boundary for capture.
- `explicit`: automatic observation does not capture. Only a request with the
  explicit capture flag may capture.

Defaults:

```text
cache-capture-mode = repeat
cache-admission-items = 8192
cache-recurrent-stride = 4
```

`cache-admission-items` is a hard item bound. Zero is valid only for `always`
or `explicit`. The admission structure must not allocate in proportion to
context size or request count without eviction.

`cache-recurrent-stride` is measured in attention blocks. It must be at least
one. A value of one preserves the existing sidecar frequency.

## 5. Request contract

Add an optional completion request field:

```json
{
  "cache_persist": true
}
```

Semantics:

- absent or `false`: use the configured capture mode;
- `true`: force capture admission for this request;
- the flag affects capture only, never lookup;
- the flag does not override media, adapter, context-shift, speculative,
  capability, range, memory, disk, or backpressure checks;
- unsupported or ineligible explicit capture completes normal inference and
  reports the reason in response metadata;
- OpenAI-compatible and native completion routes accept the same field.

The final response includes:

```json
{
  "persistent_cache": {
    "requested": true,
    "eligible": true,
    "staged_blocks": 1,
    "published_blocks": 1,
    "durable": true,
    "reason": ""
  }
}
```

`durable` is true only after all required attention and recurrent artifacts
have been committed to the disk writer successfully. A normal chat request
does not wait for disk `fsync`; an explicit UI precache request does.

## 6. Admission state

Automatic `repeat` admission uses a bounded LRU keyed by the attention chain
key for a complete boundary.

Each entry stores:

- key;
- observation count, saturated at two;
- last-observed monotonic timestamp;
- admitted bit.

Rules:

1. Count a boundary at most once per request.
2. Do not count capture-helper retries, generated tokens, or repeated calls
   while the evaluated boundary is unchanged.
3. A cache hit does not need admission and does not increment the count.
4. First cold observation inserts or refreshes count one and skips all tensor
   serialization.
5. Second cold observation sets the admitted bit and may stage capture.
6. Backpressure or a failed capture leaves the entry admitted so a later cold
   evaluation can retry.
7. Successful publication removes the admission entry because the artifact is
   now authoritative.
8. LRU eviction loses only a performance hint. It cannot change correctness.

The admission LRU is process-local in this version. Explicit precache is the
durable mechanism for important prefixes that must survive model churn.

## 7. Boundary selection and sparse recurrent sidecars

Attention remains block-addressed at every complete block.

A recurrent sidecar is required for:

- block one;
- every block whose one-based index is divisible by
  `cache-recurrent-stride`;
- the deepest complete block in the request prompt.

Examples with block size 2048 and stride four:

| Prompt length | Attention blocks | Recurrent sidecars |
| --- | ---: | --- |
| 2049 | 1 | 1 |
| 8193 | 4 | 2: blocks 1 and 4 |
| 16385 | 8 | 3: blocks 1, 4, and 8 |
| 65535 | 31 | 9: blocks 1, 4, 8, 12, 16, 20, 24, 28, and 31 |

The final sidecar rule prevents a long prompt from walking back farther than
one stride merely because its end is not stride-aligned.

Lookup already walks back from the deepest contiguous attention chain to a
compatible recurrent sidecar. That behavior remains authoritative.

## 8. Deferred capture lifecycle

### 8.1 During prompt decode

At each selected and admitted boundary:

1. Confirm the request and memory implementation remain eligible.
2. Reserve bounded staging for the recurrent sidecar and a small capture
   descriptor.
3. Serialize the recurrent boundary while it is live.
4. Record the attention range, key, parent, token count, and slot generation.
5. Do not serialize attention yet.

If recurrent staging fails, record a backpressure skip and continue ordinary
inference. No partial artifact is published.

### 8.2 After response handoff

After the final response is handed to the client, but before the slot may be
cleared, retagged, or reused:

1. Validate that the slot generation and attention range still match the
   descriptor.
2. Serialize the attention range.
3. Validate both envelopes and checksums.
4. Commit attention and recurrent artifacts as one logical pair.
5. Release staging reservations.

The llama context is accessed only on the server execution thread. The disk
writer remains asynchronous. No background thread may read llama memory.

For non-recurrent models, the descriptor contains only the deferred attention
range.

### 8.3 Failure behavior

- If response handoff succeeds but deferred capture fails, the client response
  remains successful.
- If the slot changed before materialization, drop the descriptor.
- If only one artifact commits, lookup still requires a valid sidecar for a
  recurrent model and therefore fails closed.
- Shutdown flushes committed writer jobs but may discard unmaterialized slot
  descriptors.

## 9. Bounded staging

The existing write-buffer reservation system owns staged recurrent payloads.
It must account for:

- reserved payloads;
- deferred descriptors;
- queued writer jobs;
- in-flight writer jobs.

No second unaccounted copy of a tensor payload is allowed.

When `--cache-write-buffer 0` is used, adaptive capacity learns one admitted
attention block plus one recurrent sidecar, as it does today. Selective capture
does not weaken the hard capacity.

Per-slot pending descriptors are bounded by the number of selected sidecars in
the current prompt and by the global reservation budget.

## 10. Explicit precache endpoint

Add:

```text
POST /cache/prefix
```

The router proxies this route to the selected model like completion routes.

The request accepts the same prompt forms as completion:

- native `prompt`;
- OpenAI-style `messages`;
- model ID;
- adapters and template options already supported by the selected route.

The endpoint internally performs a no-visible-output completion with:

- persistent capture forced;
- normal template and tokenization;
- generation limited to the minimum required to obtain valid logits;
- output discarded;
- synchronous wait until selected artifacts are committed to the disk writer.

The response is:

```json
{
  "model": "qwen3.5-9b",
  "tokens_evaluated": 4097,
  "boundary": 4096,
  "attention_blocks": 2,
  "recurrent_sidecars": 2,
  "bytes_published": 188000000,
  "durable": true,
  "reason": ""
}
```

An ineligible request returns a normal 4xx request result with a stable reason.
Backpressure returns 429. Corrupt or failed serialization returns 500 and
publishes no usable recurrent boundary pair.

## 11. UI contract

Add a compact `Precache prefix` action to the existing chat overflow menu.

Behavior:

1. Use the currently selected model.
2. Build messages through the same chat request builder used for the next
   completion.
3. Include the current conversation and configured system prompt.
4. Exclude an empty draft. Include a non-empty draft only after the user
   confirms by invoking the action from that draft's menu.
5. Call `POST /cache/prefix`.
6. Show one transient state on the action: working, complete, or failed.
7. Do not append a hidden or generated assistant message to the conversation.
8. Disable the action while generation is active.
9. If the model reports persistent caching disabled, omit the action.

The action is a low-frequency invoked affordance. It does not receive permanent
primary placement, a badge, explanatory copy, or a new settings panel.

The model-management UI may show live cache bytes and a clear action only where
it already shows mutable model runtime state.

## 12. Observability

Add counters:

- boundaries observed;
- first observations;
- repeat admissions;
- explicit admissions;
- admission LRU evictions;
- recurrent stages;
- recurrent stage bytes;
- deferred attention materializations;
- dropped deferred descriptors by reason;
- explicit precache requests, successes, and failures;
- capture time before response handoff;
- capture time after response handoff.

Existing capture and store counters remain.

`/props` exposes:

```json
{
  "prefix_cache": {
    "capture_mode": "repeat",
    "admission_items": 127,
    "admission_capacity": 8192,
    "recurrent_stride": 4,
    "deferred_items": 0,
    "deferred_bytes": 0
  }
}
```

## 13. Implementation plan

### Phase A: admission and request contract

1. Add configuration fields, parsing, validation, and documentation.
2. Add `cache_persist` to native and OpenAI-compatible request schemas.
3. Add the bounded admission LRU.
4. Separate observed-boundary tracking from captured-boundary tracking.
5. Preserve `always` behavior and add `repeat` and `explicit`.
6. Add metrics and unit tests.

Exit gate:

- a one-off prefix publishes zero artifacts in `repeat`;
- the second cold observation captures;
- `cache_persist=true` captures on the first observation;
- retries within one request do not count as a second observation.

### Phase B: sparse sidecars

1. Add boundary-selection policy.
2. Capture sidecars at block one, stride boundaries, and the final complete
   prompt boundary.
3. Extend walkback tests.
4. Report attention and recurrent publication counts separately.

Exit gate:

- a 31-block prompt with stride four publishes 31 attention blocks and nine
  recurrent sidecars;
- deepest restore walks back no farther than required by the declared policy;
- exact continuation tests pass.

### Phase C: deferred attention materialization

1. Add per-slot deferred descriptors and generation IDs.
2. Stage recurrent data during decode.
3. Materialize attention after response handoff on the server thread.
4. Commit only complete logical boundary pairs.
5. Add cancellation, slot reuse, shutdown, and backpressure tests.

Exit gate:

- attention tensor serialization is absent from pre-response timing;
- no llama-context access occurs on the writer thread;
- forced slot reuse drops stale descriptors safely;
- cold request output and correctness are unchanged.

### Phase D: explicit endpoint and UI

1. Add `/cache/prefix` to server and router proxy routes.
2. Reuse existing prompt, messages, template, model, and adapter parsing.
3. Add a durable-writer wait for explicit precache.
4. Add the chat overflow action and transient states.
5. Add server integration and UI component tests.

Exit gate:

- the UI action creates reusable disk artifacts;
- no assistant message is appended;
- restart reuse evaluates only the unmatched suffix;
- disabled and ineligible states are represented without permanent copy.

### Phase E: qualification and deployment

1. Run generated attention, recurrent, and hybrid fixtures.
2. Run real Qwen 3.5 9B unified Metal cold, repeat, explicit, restart, and
   corrupt-artifact cases.
3. Run Qwen 3.6 35B capture-count and restart smoke tests.
4. Set the local production preset to `repeat`, stride four.
5. Restart the LaunchAgent and verify both resident children.

Exit gate:

- one-off prompts have no persistent serialization cost;
- explicit and repeat-admitted prefixes restore exactly after restart;
- the live UI exposes precache for enabled models;
- production reports zero restore failures and no unbounded staging.

## 14. Deferred M5 Max benchmark matrix

Benchmarking is intentionally held until the feature behavior is complete.
Record this matrix without executing it during Phases A-D:

| Variable | Values |
| --- | --- |
| Model | Qwen 3.5 9B, Qwen 3.6 35B |
| Prompt tokens | 2049, 8193, 32769 |
| `ubatch-size` | 512, 1024, 2048 |
| threads | 8, 12, 18 |
| recurrent stride | 1, 2, 4, 8 |
| capture mode | always, repeat admitted, explicit, disabled |
| restore source | hot RAM, filesystem page cache, cold disk |

Measure:

- prompt tokens per second;
- time to first generated token;
- pre-response capture time;
- post-response materialization time;
- recurrent and attention bytes copied;
- host RAM peak and steady state;
- Metal allocated bytes;
- disk bytes;
- restart restore latency;
- generation tokens per second.

Use fixed tokens and a fixed seed. Compare continuation tokens and direct logits
where the test harness supports them. Do not select production values from one
run; use at least five measured runs after one warm-up.

## 15. Adversarial checks

Review must challenge:

- accidental admission twice within one request;
- explicit flags bypassing correctness eligibility;
- response completion before recurrent state is safely staged;
- slot reuse while a deferred descriptor exists;
- materialization after LoRA, model, or prompt identity changes;
- orphan attention publication without a usable sidecar;
- sidecar sparsity causing unbounded walkback;
- writer reservations double-counted or not counted;
- UI precache mutating the conversation;
- router proxying precache to the wrong model;
- shutdown losing a response that claimed `durable=true`;
- permanent UI text or controls that do not earn their placement.
