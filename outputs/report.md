# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,660  
**Projected savings:** $12,473  (**46%**)

## Unit economics ($/1M-token)

| Scope | Baseline | Optimized | Change |
|---|---|---|---|
| Inference API only (M2) | $6.488 | $1.275 | -80% |
| All-in: GPU fleet + API spend / tokens served | $120.063 | $64.870 | -46% |

## Savings by lever

| Lever | Savings (USD) | Share of savings |
|---|---|---|
| Inference (cascade/cache/batch) | $1,178 | 9% |
| Purchasing (spot/reserved) | $10,040 | 80% |
| Right-size util-lies | $655 | 5% |
| Kill idle GPUs | $600 | 5% |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1 (cleanest; cheapest power: us-east-wa)

_Tokens served: 7,533,027/day (226M/month). The all-in row spreads the whole GPU bill over these tokens, so it is a blended figure, not an API price._

## The GPU-Util lie

| GPU | Type (workload) | GPU-Util | MFU | MBU | Best same-type MFU | Right-size to | $/month |
|---|---|---|---|---|---|---|---|
| gpu-h100-4 | H100 (train) | 98% | 0.19 | 0.21 | 0.43 | A100 | $511 |
| gpu-a10g-1 | A10G (infer) | 97% | 0.27 | 0.30 | 0.22 | L4 | $144 |

**Why it happens.** `nvidia-smi` GPU-Util is the share of time *at least one kernel* is running on *any* SM. It does not measure how many SMs are busy or whether the tensor cores are fed. A GPU that launches many small kernels, waits on NCCL all-reduce, spins on a slow dataloader or stalls on HBM still reads ~100%.

**What the numbers say.** gpu-h100-4 shows 98% util but MFU 0.19, while its H100 peers reach 0.43 on the same job type. Per useful FLOP it costs **2.2x** a healthy H100. On gpu-h100-4 both MFU and MBU are low (~0.2), so it is neither compute-bound nor memory-bound. It is *starved*: input pipeline, host-device sync, kernel-launch overhead or communication waits. gpu-a10g-1 is an inference GPU. Decode is memory-bound, so MBU is the fairer metric there, and MBU ~0.30 at 97% util points to small batches. Fix batching before downsizing.

**Cost impact.** You pay full on-demand price for hours that yield a fraction of the work. Right-sizing both one tier down is worth **$655/month**. Profiling the H100 first (dataloader workers, larger micro-batch, overlap comms) may recover the throughput instead.

## Recommended actions (ranked by $ impact / effort)

| # | Lever | $/month | Why this order |
|---|---|---|---|
| 1 | Purchasing (spot/reserved) | $10,040 | Billing change only, no code: spot + checkpoint for the interruptible training jobs, 3yr reserved for 24/7 inference (duty >= 55% break-even). |
| 2 | Inference (cascade/cache/batch) | $1,178 | Router + cache + Batch API. Ship cascade first (80% of requests are easy); cache only the groups that clear break-even reads (Ext 3). |
| 3 | Right-size util-lies | $655 | Profile first (MFU/MBU), then move to a cheaper GPU tier. Needs a throughput benchmark before the switch. |
| 4 | Kill idle GPUs | $600 | Scheduler / auto-shutdown for gpu-h100-5 overnight. Small $, near-zero effort: do it this week. |

Kill-idle is ranked last by dollars but costs nothing to do, so run it in parallel with #1.

## Extension 3 — `cache_is_worth_it()`

Assumptions (CSV has no prefix ids or read counts): one cache entry per (team, tier) per 5-min TTL window, and the first request in the window writes it. Write cost = 0.25x input price (Anthropic 5-min write) + $1/1M-tok/hr storage (Gemini-style). Reads are billed at 10% of input price.

| Team | Tier | Cached reqs | Writes | Avg reads/write | Break-even reads | Cache? |
|---|---|---|---|---|---|---|
| assistant | large | 173 | 138 | 0.25 | 0.31 | no |
| assistant | small | 617 | 260 | 1.37 | 0.74 | yes |
| eval | large | 84 | 71 | 0.18 | 0.31 | no |
| eval | small | 331 | 198 | 0.67 | 0.74 | no |
| rag | large | 104 | 87 | 0.20 | 0.31 | no |
| rag | small | 462 | 231 | 1.00 | 0.74 | yes |
| search | large | 137 | 110 | 0.25 | 0.31 | no |
| search | small | 492 | 236 | 1.08 | 0.74 | yes |

| Cache mode | $/day | $/1M-token |
|---|---|---|
| No cache | $9.66 | $1.282 |
| Before: writes treated as free (original lab) | $8.48 | $1.126 |
| Always cache, pay writes | $9.67 | $1.284 |
| After: cache only where worth it | $9.61 | $1.275 |

**Insight.** The original model treated writes as free, which overstated cache savings: $1.17/day vs no cache. Counting writes, the real figure is $0.05/day. Caching everything is worse than no cache at all. Large-model groups see only 0.18–0.25 reads per write, below the 0.31 break-even, so caching them loses money. eval/small (0.67 < 0.74) is batch traffic spread across the day and also misses. Only chat, RAG and search on the small model reuse prefixes often enough. Longer TTL or grouping requests by prefix would raise reads/write.

## Extension 4 — Reasoning budget

| Group | Requests | Tokens | $/day | Wh/day | Share req / tok / $ / Wh |
|---|---|---|---|---|---|
| Non-reasoning | 2,199 | 6,291,871 | $8.20 | 1,888 | 92% / 84% / 85% / 6% |
| Reasoning | 201 | 1,241,156 | $1.41 | 29,788 | 8% / 16% / 15% / 94% |

| Scenario | Demoted | $/day | Wh/day | Saved $/month | Saved kWh/month |
|---|---|---|---|---|---|
| current | 0 | $9.61 | 31,675 | $0.00 | 0.0 |
| cap 10% traffic | 0 | $9.61 | 31,675 | $0.00 | 0.0 |
| reasoning only if route_tier=large | 162 | $9.50 | 7,771 | $3.16 | 717.1 |

Reasoning is already **8.4%** of traffic, so a 10% cap demotes nothing and saves $0. Proposed rule: **allow reasoning only when the cascade routes the request to the large model** (the router has judged it hard). This demotes 162 easy requests. Assumption: a demoted request emits output/6 (removing the reasoning tax from `data/generate.py`) on the same model.

**Insight.** Reasoning is ~15% of inference $ but ~94% of inference energy (80x Wh multiplier). The rule barely moves dollars but cuts ~75% of Wh. The reasoning budget is a carbon lever more than a cost lever. It is reported here and not added to the $ total.

## Region choice: electricity $ vs carbon

Fleet energy from telemetry (sum of hourly `power_w`, 30-day month): **3,386 kWh/month**.

| Region | $/kWh | gCO2/kWh | Electricity $/month | kg CO2e/month |
|---|---|---|---|---|
| europe-north1 | 0.090 | 30 | $305 | 102 |
| us-east-wa | 0.055 | 90 | $186 | 305 |
| us-west-2 | 0.070 | 120 | $237 | 406 |
| us-east-1 | 0.120 | 380 | $406 | 1,287 |
| europe-central2 | 0.180 | 660 | $609 | 2,235 |

- Cleanest: **europe-north1**. Moving from us-east-1 cuts 1,185 kg CO2e/month (-92%) and electricity $102/month.
- Cheapest power: **us-east-wa** ($0.055/kWh, 90 gCO2/kWh). Balanced pick: **us-east-wa**.
- europe-central2 is worst on both counts. Avoid it.
- Trade-off: the cleanest region may sit far from users, so latency-sensitive chat should stay close to users. Interruptible training and batch eval can move. Electricity is usually bundled into the GPU-hour price, so these $ only show up directly for owned or colocated hardware or region-priced capacity.

_Figures are June-2026 as-of snapshots; re-baseline before acting._