import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing
from missions import m2_inference_levers


def test_cache_break_even_reads():
    # write = 0.25 x $3 + $1/hr x 5min = 0.8333; each read saves 0.9 x $3 = 2.7
    wc = pricing.cache_write_cost_per_m(3.0)
    assert abs(wc - (0.75 + 1 / 12)) < 1e-9
    assert abs(pricing.break_even_cache_reads(wc, 3.0) - wc / 2.7) < 1e-9


def test_cache_is_worth_it_low_vs_high_reads():
    wc = pricing.cache_write_cost_per_m(0.20)          # small model: storage dominates
    assert not pricing.cache_is_worth_it(0, wc, 0.20)   # never re-read -> pure loss
    assert not pricing.cache_is_worth_it(0.5, wc, 0.20)
    assert pricing.cache_is_worth_it(3, wc, 0.20)
    assert pricing.cache_is_worth_it(10, wc, 0.20)
    # no read discount -> reads never pay back
    assert not pricing.cache_is_worth_it(100, wc, 0.20, read_discount=1.0)


def test_cache_policy_never_worse_than_always_cache():
    r2 = m2_inference_levers.run(verbose=False)
    modes = r2["cache_modes_daily"]
    assert modes["policy"] <= modes["always"]
    assert modes["policy"] <= modes["none"]
    assert modes["free"] <= modes["policy"]             # free writes overstated savings


def test_reasoning_budget():
    rb = m2_inference_levers.run(verbose=False)["reasoning"]
    assert rb["reasoning_share"] < 0.10
    assert rb["cap10"]["demoted"] == 0 and rb["cap10"]["usd"] == rb["current"]["usd"]
    rule = rb["rule_large_only"]
    assert rule["demoted"] > 0
    assert rule["usd"] < rb["current"]["usd"] and rule["wh"] < rb["current"]["wh"]
    cur = rb["current"]["groups"]
    assert cur[1]["wh"] > cur[0]["wh"]                  # reasoning dominates energy
