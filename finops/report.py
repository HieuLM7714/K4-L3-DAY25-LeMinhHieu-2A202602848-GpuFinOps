"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 unit_economics: dict | None = None, sections: list[str] | None = None) -> str:
    """Return a markdown cost-optimization report.

    unit_economics: optional {label: (baseline_per_m, optimized_per_m)} rows for a $/1M-token table.
    sections: optional extra markdown blocks appended after the core report.
    """
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
    ]
    if unit_economics:
        lines += [
            "",
            "## Unit economics ($/1M-token)",
            "",
            "| Scope | Baseline | Optimized | Change |",
            "|---|---|---|---|",
        ]
        for name, (b, o) in unit_economics.items():
            chg = (1 - o / b) * 100 if b else 0.0
            lines.append(f"| {name} | ${b:,.3f} | ${o:,.3f} | -{chg:.0f}% |")
    total = sum(levers.values()) or 1
    lines += [
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) | Share of savings |",
        "|---|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} | {amount / total:.0%} |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}",
        ]
    for block in sections or []:
        lines += ["", block.rstrip()]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str, baseline: float | None = None) -> str:
    """Write a savings chart PNG. Returns the path. No-op if matplotlib absent.

    With `baseline`, draws a true waterfall: baseline -> each lever -> optimized.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if baseline is None:
        ax.bar(names, vals, color="#2e548a")
        ax.set_ylabel("Savings (USD / month)")
    else:
        optimized = baseline - sum(vals)
        labels = ["Baseline"] + names + ["Optimized"]
        level, bottoms, heights, colors = baseline, [0], [baseline], ["#7a7a7a"]
        for v in vals:
            level -= v
            bottoms.append(level)
            heights.append(v)
            colors.append("#2e8a57")
        bottoms.append(0)
        heights.append(optimized)
        colors.append("#2e548a")
        bars = ax.bar(labels, heights, bottom=bottoms, color=colors)
        for b, h, bt in zip(bars, heights, bottoms):
            ax.text(b.get_x() + b.get_width() / 2, bt + h, f"${h:,.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Spend (USD / month)")
    ax.set_title("GPU cost savings by FinOps lever")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
