"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}

# Why each lever is ranked where it is (effort / risk), shown next to its computed $.
LEVER_NOTES = {
    "Purchasing (spot/reserved)": "Billing change only, no code: spot + checkpoint for the interruptible training jobs, "
                                  "3yr reserved for 24/7 inference (duty >= 55% break-even).",
    "Inference (cascade/cache/batch)": "Router + cache + Batch API. Ship cascade first (80% of requests are easy); "
                                       "cache only the groups that clear break-even reads (Ext 3).",
    "Right-size util-lies": "Profile first (MFU/MBU), then move to a cheaper GPU tier. Needs a throughput "
                            "benchmark before the switch.",
    "Kill idle GPUs": "Scheduler / auto-shutdown for gpu-h100-5 overnight. Small $, near-zero effort: do it this week.",
}


def _telemetry_by_gpu() -> dict:
    """Workload label and daily energy (Wh) per GPU from hourly telemetry."""
    out = defaultdict(lambda: {"workload": "", "wh_day": 0.0})
    for r in load_csv("gpu_telemetry.csv"):
        out[r["gpu_id"]]["workload"] = r["workload"]
        out[r["gpu_id"]]["wh_day"] += num(r["power_w"])  # 1-hour rows -> W == Wh
    return out


def _util_lie_section(r1: dict, cat: dict, tel: dict) -> tuple[str, float]:
    by_type = defaultdict(list)
    for s in r1["summary"]:
        if s not in r1["lies"]:
            by_type[s["gpu_type"]].append(s["mfu"])
    rows, total = [], 0.0
    for lie in r1["lies"]:
        cur, tgt = lie["gpu_type"], RIGHTSIZE_MAP.get(lie["gpu_type"], lie["gpu_type"])
        save = max(0.0, num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])) * 24 * DAYS
        total += save
        peers = by_type[cur]
        peer = f"{max(peers):.2f}" if peers else "n/a"
        rows.append(f"| {lie['gpu_id']} | {cur} ({tel[lie['gpu_id']]['workload']}) | {lie['gpu_util_pct']:.0f}% "
                    f"| {lie['mfu']:.2f} | {lie['mbu']:.2f} | {peer} | {tgt} | ${save:,.0f} |")
    h100 = next((l for l in r1["lies"] if l["gpu_id"] == "gpu-h100-4"), None)
    paid_flops = ""
    if h100:
        best = max(by_type["H100"])
        paid_flops = (f"gpu-h100-4 shows {h100['gpu_util_pct']:.0f}% util but MFU {h100['mfu']:.2f}, while its H100 peers "
                      f"reach {best:.2f} on the same job type. Per useful FLOP it costs "
                      f"**{best / h100['mfu']:.1f}x** a healthy H100. ")
    return "\n".join([
        "## The GPU-Util lie",
        "",
        "| GPU | Type (workload) | GPU-Util | MFU | MBU | Best same-type MFU | Right-size to | $/month |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "**Why it happens.** `nvidia-smi` GPU-Util is the share of time *at least one kernel* is running on *any* SM. "
        "It does not measure how many SMs are busy or whether the tensor cores are fed. A GPU that launches many "
        "small kernels, waits on NCCL all-reduce, spins on a slow dataloader or stalls on HBM still reads ~100%.",
        "",
        f"**What the numbers say.** {paid_flops}On gpu-h100-4 both MFU and MBU are low (~0.2), so it is "
        "neither compute-bound nor memory-bound. It is *starved*: input pipeline, host-device sync, kernel-launch overhead "
        "or communication waits. gpu-a10g-1 is an inference GPU. Decode is memory-bound, so MBU is the fairer metric "
        "there, and MBU ~0.30 at 97% util points to small batches. Fix batching before downsizing.",
        "",
        f"**Cost impact.** You pay full on-demand price for hours that yield a fraction of the work. "
        f"Right-sizing both one tier down is worth **${total:,.0f}/month**. Profiling the H100 first "
        "(dataloader workers, larger micro-batch, overlap comms) may recover the throughput instead.",
    ]), total


def _actions_section(levers: dict) -> str:
    ranked = sorted(levers.items(), key=lambda kv: kv[1], reverse=True)
    lines = ["## Recommended actions (ranked by $ impact / effort)", "",
             "| # | Lever | $/month | Why this order |", "|---|---|---|---|"]
    for i, (name, amt) in enumerate(ranked, 1):
        lines.append(f"| {i} | {name} | ${amt:,.0f} | {LEVER_NOTES.get(name, '')} |")
    lines += ["", "Kill-idle is ranked last by dollars but costs nothing to do, so run it in parallel with #1."]
    return "\n".join(lines)


def _cache_section(r2: dict) -> str:
    pm = lambda daily: daily / (r2["total_tokens"] / 1e6)
    lines = ["## Extension 3 — `cache_is_worth_it()`", "",
             "Assumptions (CSV has no prefix ids or read counts): one cache entry per (team, tier) per 5-min TTL window, "
             "and the first request in the window writes it. Write cost = 0.25x input price (Anthropic 5-min write) "
             "+ $1/1M-tok/hr storage (Gemini-style). Reads are billed at 10% of input price.", "",
             "| Team | Tier | Cached reqs | Writes | Avg reads/write | Break-even reads | Cache? |",
             "|---|---|---|---|---|---|---|"]
    for (team, tier), p in sorted(r2["cache_policy"].items()):
        lines.append(f"| {team} | {tier} | {p['requests']} | {p['writes']} | {p['avg_reads']:.2f} "
                     f"| {p['break_even_reads']:.2f} | {'yes' if p['worth_it'] else 'no'} |")
    lines += ["", "| Cache mode | $/day | $/1M-token |", "|---|---|---|"]
    modes = r2["cache_modes_daily"]
    for name, key in [("No cache", "none"),
                      ("Before: writes treated as free (original lab)", "free"),
                      ("Always cache, pay writes", "always"),
                      ("After: cache only where worth it", "policy")]:
        lines.append(f"| {name} | ${modes[key]:.2f} | ${pm(modes[key]):.3f} |")
    lines += ["", f"**Insight.** The original model treated writes as free, which overstated cache savings: "
              f"${modes['none'] - modes['free']:.2f}/day vs no cache. Counting writes, the real figure is "
              f"${modes['none'] - modes['policy']:.2f}/day. Caching everything is worse than no cache at all. "
              "Large-model groups see only 0.18–0.25 reads per write, below the 0.31 break-even, so caching them loses money. "
              "eval/small (0.67 < 0.74) is batch traffic spread across the day and also misses. "
              "Only chat, RAG and search on the small model reuse prefixes often enough. "
              "Longer TTL or grouping requests by prefix would raise reads/write."]
    return "\n".join(lines)


def _reasoning_section(r2: dict) -> str:
    rb = r2["reasoning"]
    cur = rb["current"]["groups"]
    tot = {k: cur[0][k] + cur[1][k] for k in ("requests", "tokens", "usd", "wh")}
    lines = ["## Extension 4 — Reasoning budget", "",
             "| Group | Requests | Tokens | $/day | Wh/day | Share req / tok / $ / Wh |", "|---|---|---|---|---|---|"]
    for k, label in ((0, "Non-reasoning"), (1, "Reasoning")):
        g = cur[k]
        share = " / ".join(f"{g[f] / tot[f]:.0%}" for f in ("requests", "tokens", "usd", "wh"))
        lines.append(f"| {label} | {g['requests']:,} | {g['tokens']:,} | ${g['usd']:.2f} | {g['wh']:,.0f} | {share} |")
    lines += ["", "| Scenario | Demoted | $/day | Wh/day | Saved $/month | Saved kWh/month |", "|---|---|---|---|---|---|"]
    for key in ("current", "cap10", "rule_large_only"):
        s = rb[key]
        d_usd = (rb["current"]["usd"] - s["usd"]) * DAYS
        d_kwh = (rb["current"]["wh"] - s["wh"]) * DAYS / 1000
        lines.append(f"| {s['name']} | {s['demoted']} | ${s['usd']:.2f} | {s['wh']:,.0f} | ${d_usd:,.2f} | {d_kwh:,.1f} |")
    rule = rb["rule_large_only"]
    lines += ["", f"Reasoning is already **{rb['reasoning_share']:.1%}** of traffic, so a 10% cap demotes nothing and saves $0. "
              f"Proposed rule: **allow reasoning only when the cascade routes the request to the large model** "
              f"(the router has judged it hard). This demotes {rule['demoted']} easy requests. Assumption: a demoted request "
              "emits output/6 (removing the reasoning tax from `data/generate.py`) on the same model.", "",
              "**Insight.** Reasoning is ~15% of inference $ but ~94% of inference energy (80x Wh multiplier). "
              "The rule barely moves dollars but cuts ~75% of Wh. The reasoning budget is a carbon lever more than "
              "a cost lever. It is reported here and not added to the $ total."]
    return "\n".join(lines)


def _region_section(tel: dict) -> tuple[str, dict]:
    kwh_month = sum(g["wh_day"] for g in tel.values()) * DAYS / 1000
    carbon, price = sustainability.REGION_CARBON, sustainability.REGION_PRICE_KWH
    cleanest = min(carbon, key=carbon.get)
    cheapest = min(price, key=price.get)
    # balanced: normalise both to the worst region and add
    balanced = min(carbon, key=lambda r: carbon[r] / max(carbon.values()) + price[r] / max(price.values()))
    lines = ["## Region choice: electricity $ vs carbon", "",
             f"Fleet energy from telemetry (sum of hourly `power_w`, 30-day month): **{kwh_month:,.0f} kWh/month**.", "",
             "| Region | $/kWh | gCO2/kWh | Electricity $/month | kg CO2e/month |", "|---|---|---|---|---|"]
    for r in sorted(carbon, key=carbon.get):
        lines.append(f"| {r} | {price[r]:.3f} | {carbon[r]} | ${kwh_month * price[r]:,.0f} "
                     f"| {kwh_month * carbon[r] / 1000:,.0f} |")
    base_kg = kwh_month * carbon["us-east-1"] / 1000
    best_kg = kwh_month * carbon[cleanest] / 1000
    lines += ["",
              f"- Cleanest: **{cleanest}**. Moving from us-east-1 cuts {base_kg - best_kg:,.0f} kg CO2e/month "
              f"(-{(1 - carbon[cleanest] / carbon['us-east-1']):.0%}) and electricity "
              f"${kwh_month * (price['us-east-1'] - price[cleanest]):,.0f}/month.",
              f"- Cheapest power: **{cheapest}** (${price[cheapest]}/kWh, {carbon[cheapest]} gCO2/kWh). Balanced pick: **{balanced}**.",
              "- europe-central2 is worst on both counts. Avoid it.",
              "- Trade-off: the cleanest region may sit far from users, so latency-sensitive chat should stay close to users. "
              "Interruptible training and batch eval can move. Electricity is usually bundled into the GPU-hour price, "
              "so these $ only show up directly for owned or colocated hardware or region-priced capacity."]
    return "\n".join(lines), {"cleanest": cleanest, "cheapest": cheapest}


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()
    tel = _telemetry_by_gpu()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    lie_md, rightsize_savings = _util_lie_section(r1, cat, tel)

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- unit economics: $/1M-token ---
    month_tokens_m = r2["total_tokens"] * DAYS / 1e6
    unit = {
        "Inference API only (M2)": (r2["baseline_per_m"], r2["optimized_per_m"]),
        "All-in: GPU fleet + API spend / tokens served": (baseline / month_tokens_m, optimized / month_tokens_m),
    }

    # --- sustainability snapshot ---
    region_md, regions = _region_section(tel)
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": f"{regions['cleanest']} (cleanest; cheapest power: {regions['cheapest']})",
    }

    sections = [
        f"_Tokens served: {r2['total_tokens']:,}/day ({month_tokens_m:,.0f}M/month). The all-in row spreads the whole "
        "GPU bill over these tokens, so it is a blended figure, not an API price._",
        lie_md, _actions_section(levers), _cache_section(r2), _reasoning_section(r2), region_md,
    ]
    md = report.build_report(baseline, optimized, levers, sustainability=sust,
                             unit_economics=unit, sections=sections)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"), baseline=baseline)

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
