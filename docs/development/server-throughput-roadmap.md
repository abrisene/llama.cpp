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

Required behavior:

- coordinate decode ownership in the router, not in individual model kernels;
- leave idle resident models loaded;
- allow prompt processing and disk-cache work while another model owns decode;
- use short, bounded leases so one long response cannot starve another model;
- expose queue time, lease time, handoffs, and concurrent-demand metrics;
- provide a bypass mode for comparison and recovery;
- preserve normal continuous batching inside each model process.

The scheduler must optimize the selected policy explicitly:

- interactive mode favors per-request t/s and predictable latency;
- throughput mode permits concurrent decode when it improves combined t/s.

Implementation must not assume that serial execution is always faster. The
admission decision will use measured active-model demand and rolling decode
timings.

## Stage 5: adaptive decode microbatching

Goal: improve combined t/s when several slots on one model are generating.

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

## Verification order

1. Functional build and server tests.
2. Single-request output and cancellation checks.
3. Two simultaneous requests to one model.
4. Simultaneous requests to both resident models.
5. Rollback/bypass checks.
6. A focused before/after throughput sample.

The broader batch, thread, prompt-length, and cache-stride matrix remains
deferred until these mechanisms exist.
