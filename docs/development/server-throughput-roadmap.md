# M5 Max Server Throughput Roadmap

Status: local-fork implementation record

This document tracks generated-token throughput work for the local multi-model
server. It does not cover prompt-prefill acceleration, quantization changes, or
the broader tuning matrix in `server-prefix-cache-admission-and-precache.md`.

## Stage 3: Metal backend sampling

Production setting:

```ini
backend-sampling = true
```

The option moves sampling work from the CPU path to the active backend. It is
experimental in llama.cpp, so it remains an explicit deployment setting rather
than a changed global default.

### M5 Max qualification

The production router served one model at a time during each sample. Each run
requested up to 128 generated tokens with the same short prompt and sampling
settings. This was a narrow feature qualification, not the deferred tuning
matrix.

| Model | CPU sampling median | Metal sampling median | Change |
| --- | ---: | ---: | ---: |
| Qwen3.5 9B Q4_K_M | 54.07 t/s | 54.06 t/s | effectively flat |
| Qwen3.6 35B-A3B Q6_K | 76.07 t/s | 79.28 t/s | +4.2% |

Decision: keep backend sampling enabled. It materially helps the MoE model and
does not show a meaningful regression on the dense 9B model.

## Stage 4: bandwidth-aware model scheduling

Goal: prevent independently resident model processes from degrading each
other's generated-token rate when both submit Metal decode work.

Implementation status: the short-lease arbiter is implemented behind
`--models-decode-arbiter`. The production throughput policy remains concurrent
decode because the focused simultaneous-model sample was faster with the
arbiter bypassed.

Required behavior:

- coordinate decode ownership in the router, not in individual model kernels;
- leave idle resident models loaded;
- allow prompt processing and disk-cache work while another model owns decode;
- use short, bounded leases so one long response cannot starve another model;
- expose queue time, lease time, handoffs, and concurrent-demand metrics;
- provide a bypass mode for comparison and recovery;
- preserve normal continuous batching inside each model process.

The POSIX implementation uses one shared advisory file lock whose path is
passed to router children. A child acquires it only for a generation decode
step, including speculative follow-up. Prompt decode, cache reads, and model
residency do not take the lease. Child properties expose acquisition,
contention, handoff, concurrent-demand, wait, and hold counters. Because one
child has one decode thread, each contended acquisition is both observed
concurrent demand and a handoff from another child.

The router operator selects the policy explicitly:

- interactive mode favors per-request t/s and predictable latency;
- throughput mode permits concurrent decode when it improves combined t/s.

The deployment decision must not assume that serial execution is always
faster. It is made from simultaneous-model measurements and remains reversible
through the bypass flag.

### M5 Max policy qualification

Three simultaneous 128-token samples were taken with Qwen3.5 9B and
Qwen3.6 35B-A3B resident. This is a narrow policy qualification, not the
deferred tuning matrix.

| Policy | Median effective combined rate | Decision |
| --- | ---: | --- |
| Concurrent child decode | 102.71 t/s | production default |
| Short serialized leases | 82.64 t/s | available for latency/fairness diagnosis |

Serialization reduced combined throughput by about 19.5% on this pair. The
correct throughput setting for this machine is therefore the bypass mode.

## Stage 5: adaptive decode microbatching

Goal: improve combined t/s when several slots on one model are generating.

Implementation status: implemented behind `--batch-coalesce-us`. The coalescer
acts only at admission when at least two generation requests were already
received in one queue cycle and free batch slots remain. It never sleeps
between generated tokens, does not delay an isolated request, and does not wait
after a burst already fills every configured server slot.

Production remains at the default value of `0` until the deferred benchmark
matrix qualifies a workload where partial request bursts benefit from the
window.

Required behavior:

- use the existing continuous-batching path;
- add a bounded coalescing window only when more work is expected;
- begin with a maximum window of 1-2 ms;
- disable coalescing for a single active request;
- shrink the window when first-token or inter-token latency worsens;
- expose batch width, wait time, decode time, and tokens-per-decode metrics;
- keep streaming responses and cancellation responsive.

The coalescer and cross-model scheduler must share one latency budget. Their
waits must not stack independently.

The current window grows toward the configured maximum when another task
arrives during the wait. It shrinks after a timeout, because that timeout added
first-token latency without increasing batch width. When the cross-model decode
arbiter is enabled, admission coalescing is disabled so the two waits cannot
stack.

Child `/props` exposes the maximum and current window, waits, wakeups, timeouts,
admitted tasks, decode calls, total and maximum batch tokens, average tokens and
generation slots per decode, and full decode-step wall time including
post-decode sampling.

## Verification order

1. Functional build and server tests.
2. Single-request output and cancellation checks.
3. Two simultaneous requests to one model.
4. Simultaneous requests to both resident models.
5. Rollback/bypass checks.
6. A focused before/after throughput sample.

The broader batch, thread, prompt-length, and cache-stride matrix remains
deferred until these mechanisms exist.
