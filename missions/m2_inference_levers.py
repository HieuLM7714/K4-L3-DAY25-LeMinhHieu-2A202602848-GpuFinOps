"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Extensions:
  Ext 3 — cache_is_worth_it(): only cache a (team, tier) prefix when reads pay back the write.
  Ext 4 — Reasoning budget: split $ / tokens / Wh by is_reasoning, test a 10% cap + a routing rule.

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# ASSUMPTIONS (CSV has no prefix ids / read counts):
CACHE_TTL_MIN = 5          # one cache entry per (team, tier) per 5-min TTL window; first request writes it
REASONING_OUTPUT_TAX = 6   # data/generate.py multiplies reasoning output by 6x -> demoted request emits out/6
REASONING_CAP = 0.10       # Guide: try capping reasoning at 10% of traffic


def _minute(ts: str) -> int:
    return int(ts[11:13]) * 60 + int(ts[14:16])


def cache_reuse_stats(rows: list[dict], ttl_min: int = CACHE_TTL_MIN) -> dict:
    """Estimate avg reads per cache write for each (team, tier) from timestamps."""
    reqs = defaultdict(int)
    windows = defaultdict(set)
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            key = (r["team"], r["route_tier"])
            reqs[key] += 1
            windows[key].add(_minute(r["ts"]) // ttl_min)
    return {k: {"requests": reqs[k], "writes": len(windows[k]),
                "avg_reads": (reqs[k] - len(windows[k])) / len(windows[k])} for k in reqs}


def cache_policy(rows: list[dict]) -> dict:
    """Ext 3: decide per (team, tier) whether caching pays off once write cost is counted."""
    policy = {}
    for key, s in cache_reuse_stats(rows).items():
        pin, _ = MODEL_PRICES[key[1]]
        wc = pricing.cache_write_cost_per_m(pin)
        policy[key] = {**s, "write_cost_per_m": wc,
                       "break_even_reads": pricing.break_even_cache_reads(wc, pin),
                       "worth_it": pricing.cache_is_worth_it(s["avg_reads"], wc, pin)}
    return policy


def priced(rows: list[dict], mode: str, policy: dict | None = None) -> list[float]:
    """Per-request optimized cost (cascade + batch) under a cache mode.

    mode: 'none'   — no prompt caching at all
          'free'   — original lab model: every cached token is discounted, writes cost nothing
          'always' — cache everything, the first request per TTL window pays the write
          'policy' — like 'always' but only for groups where cache_is_worth_it() is True
    Rows may carry an '_out' override (used by the reasoning scenarios).
    """
    seen = set()
    costs = []
    for r in sorted(rows, key=lambda x: x["ts"]):
        inp = int(num(r["input_tokens"]))
        out = int(r.get("_out", num(r["output_tokens"])))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        key = (r["team"], r["route_tier"])
        use_cache = mode in ("free", "always") or (policy and policy.get(key, {}).get("worth_it"))
        if not use_cache or cached == 0:
            costs.append(pricing.request_cost(inp, out, pin, pout, batch=is_batch))
            continue
        window = (key, _minute(r["ts"]) // CACHE_TTL_MIN)
        if mode != "free" and window not in seen:  # cache writer: full input price + write cost
            seen.add(window)
            wc = pricing.cache_write_cost_per_m(pin) * cached / 1e6
            costs.append(pricing.request_cost(inp, out, pin, pout, batch=is_batch)
                         + wc * (0.5 if is_batch else 1.0))
        else:
            costs.append(pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch))
    return costs


def reasoning_budget(rows: list[dict], policy: dict) -> dict:
    """Ext 4: $ / tokens / Wh split by is_reasoning, plus cap and routing-rule scenarios."""
    rows = sorted(rows, key=lambda x: x["ts"])
    costs = priced(rows, "policy", policy)

    def split(rs, cs):
        g = {k: {"requests": 0, "tokens": 0, "usd": 0.0, "wh": 0.0} for k in (0, 1)}
        for r, c in zip(rs, cs):
            k = int(num(r["is_reasoning"]))
            tok = int(num(r["input_tokens"])) + int(r.get("_out", num(r["output_tokens"])))
            g[k]["requests"] += 1
            g[k]["tokens"] += tok
            g[k]["usd"] += c
            g[k]["wh"] += sustainability.wh_per_query(tok, is_reasoning=bool(k))
        return g

    def demote(selected_ids: set) -> list[dict]:
        out = []
        for i, r in enumerate(rows):
            if i in selected_ids:
                r = {**r, "is_reasoning": "0", "_out": int(num(r["output_tokens"])) // REASONING_OUTPUT_TAX}
            out.append(r)
        return out

    def scenario(name, ids):
        rs = demote(ids)
        g = split(rs, priced(rs, "policy", policy))
        return {"name": name, "demoted": len(ids), "groups": g,
                "usd": g[0]["usd"] + g[1]["usd"], "wh": g[0]["wh"] + g[1]["wh"]}

    reasoning_ids = [i for i, r in enumerate(rows) if r["is_reasoning"] == "1"]
    # 10% cap: demote only the excess above the cap (small-tier = easy requests first)
    allowed = int(REASONING_CAP * len(rows))
    excess = max(0, len(reasoning_ids) - allowed)
    by_ease = sorted(reasoning_ids, key=lambda i: rows[i]["route_tier"] != "small")
    # routing rule: reasoning only for requests the cascade already judged hard (route_tier=large)
    rule_ids = {i for i in reasoning_ids if rows[i]["route_tier"] == "small"}

    current = scenario("current", set())
    return {
        "current": current,
        "cap10": scenario(f"cap {REASONING_CAP:.0%} traffic", set(by_ease[:excess])),
        "rule_large_only": scenario("reasoning only if route_tier=large", rule_ids),
        "reasoning_share": len(reasoning_ids) / len(rows),
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

    # OPTIMIZED: cascade (route_tier), prompt caching, batch API
    policy = cache_policy(rows)
    no_cache_cost = sum(priced(rows, "none"))
    free_cost = sum(priced(rows, "free"))        # before Ext 3: cache writes treated as free
    always_cost = sum(priced(rows, "always"))    # cache everything, pay writes
    opt_cost = sum(priced(rows, "policy", policy))  # Ext 3: cache only where worth it
    reasoning = reasoning_budget(rows, policy)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        pm = lambda c: pricing.dollars_per_million(c, total_tokens)
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print(f"\n-- Ext 3: cache_is_worth_it()  (TTL {CACHE_TTL_MIN} min, write = 0.25x input + $1/1M-tok/hr storage) --")
        print(f"{'team':10}{'tier':7}{'reqs':>6}{'writes':>8}{'avg_reads':>11}{'break_even':>12}  cache?")
        for (team, tier), p in sorted(policy.items()):
            print(f"{team:10}{tier:7}{p['requests']:>6}{p['writes']:>8}{p['avg_reads']:>11.2f}"
                  f"{p['break_even_reads']:>12.2f}  {'YES' if p['worth_it'] else 'no'}")
        print(f"{'cache mode':34}{'$/day':>9}{'$/1M-tok':>10}")
        for name, c in [("no cache", no_cache_cost),
                        ("free writes (before Ext 3)", free_cost),
                        ("always cache, pay writes", always_cost),
                        ("cache_is_worth_it policy (after)", opt_cost)]:
            print(f"{name:34}{c:>9.2f}{pm(c):>10.3f}")
        print("sensitivity (reads per write -> worth it?):")
        for tier, (pin, _) in MODEL_PRICES.items():
            wc = pricing.cache_write_cost_per_m(pin)
            marks = "  ".join(f"{n}:{'Y' if pricing.cache_is_worth_it(n, wc, pin) else 'n'}"
                              for n in (0, 0.25, 0.5, 1, 3, 10))
            print(f"  {tier:6} break-even {pricing.break_even_cache_reads(wc, pin):.2f}  ->  {marks}")

        print(f"\n-- Ext 4: Reasoning budget (Wh via wh_per_query, reasoning x{sustainability.REASONING_ENERGY_MULTIPLIER:.0f}) --")
        cur = reasoning["current"]["groups"]
        tot = {k: cur[0][k] + cur[1][k] for k in ("requests", "tokens", "usd", "wh")}
        print(f"{'group':14}{'requests':>10}{'tokens':>12}{'$/day':>9}{'Wh/day':>10}  share req/tok/$/Wh")
        for k, label in ((0, "non-reasoning"), (1, "reasoning")):
            g = cur[k]
            print(f"{label:14}{g['requests']:>10}{g['tokens']:>12,}{g['usd']:>9.2f}{g['wh']:>10.0f}  "
                  + " / ".join(f"{g[f] / tot[f]:.0%}" for f in ("requests", "tokens", "usd", "wh")))
        print(f"{'scenario':38}{'demoted':>8}{'$/day':>9}{'Wh/day':>10}{'saved $':>9}{'saved Wh':>10}")
        for key in ("current", "cap10", "rule_large_only"):
            s = reasoning[key]
            print(f"{s['name']:38}{s['demoted']:>8}{s['usd']:>9.2f}{s['wh']:>10.0f}"
                  f"{reasoning['current']['usd'] - s['usd']:>9.2f}{reasoning['current']['wh'] - s['wh']:>10.0f}")
        print(f"(reasoning is {reasoning['reasoning_share']:.1%} of traffic -> a 10% cap demotes nothing; "
              f"demoted requests assume output / {REASONING_OUTPUT_TAX})")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_modes_daily": {"none": no_cache_cost, "free": free_cost,
                              "always": always_cost, "policy": opt_cost},
        "optimized_daily_no_cache": round(no_cache_cost, 2),
        "optimized_daily_free_cache": round(free_cost, 2),
        "optimized_daily_always_cache": round(always_cost, 2),
        "cache_policy": policy, "reasoning": reasoning,
    }


if __name__ == "__main__":
    run()
