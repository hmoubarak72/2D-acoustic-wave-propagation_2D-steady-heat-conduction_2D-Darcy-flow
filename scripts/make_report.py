"""Fill README.md and docs/*.md from results/ (no hand-typed numbers).   python scripts/make_report.py"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from physicsveto.gtm.content import VERTICALS, solution_brief

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LABELS = {
    "in_distribution": "In-distribution (normal work)",
    "salt_body": "Salt body", "high_contrast": "Extreme velocity contrast", "rough_geology": "Rough geology",
    "moved_source": "Shot moved to edge / deeper", "mild_shift": "Mild distribution shift", "checkerboard_artifact": "Injected checkerboard artifact",
    "amplitude_drift": "Uniform amplitude drift",
}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def mode_table(s: dict, labels: dict | None = None) -> str:
    labels = labels or LABELS
    lines = ["| Test set | Type | Cases | Truly bad | Vetoed | Vetoed or review | Bad cases caught |",
             "|---|---|---|---|---|---|---|"]
    for k, m in s["per_mode"].items():
        lines.append(f"| {labels.get(k, k)} | {m['group']} | {m['n']} | {pct(m['share_truly_bad'])} | "
                     f"{pct(m['veto_rate'])} | {pct(m['flag_rate'])} | {pct(m['recall_of_bad'])} |")
    return "\n".join(lines)


STEADY_LABELS = {
    "edge_hotspot": "Hot chip next to the cold edge", "extreme_contrast": "Extreme conductivity contrast",
    "many_sources": "Six to nine small hot chips", "overpowered": "3-5x the training power",
    "channels": "Long thin channels", "well_injection": "Concentrated injection well",
    "fine_scale": "Very fine-scale heterogeneity",
}


def steady_vals() -> dict:
    """Placeholders for Cases B and C and the PINN comparison (n/a if a result file is missing)."""
    out = {}
    for ck, tag in (("case_b", "b"), ("case_c", "c")):
        p = RES / ck / "benchmark_summary.json"
        s = json.loads(p.read_text()) if p.exists() else None

        def g(key, s=s):
            return pct(s[key]) if s else "n/a"

        out.update({f"{tag}_fno_err": g("fno_in_dist_rel_l2_median"), f"{tag}_unet_err": g("unet_in_dist_rel_l2_median"),
                    f"{tag}_veto_p": g("veto_precision"), f"{tag}_veto_r": g("veto_recall"),
                    f"{tag}_flag_r": g("flag_recall"), f"{tag}_vr": g("in_dist_veto_rate"),
                    f"{tag}_acc_bad": g("accepted_share_truly_bad"), f"{tag}_raw_bad": g("unfiltered_share_truly_bad"),
                    f"{tag}_n_eval": str(s["n_eval"]) if s else "n/a",
                    f"{tag}_fd_ms": f"{s['fd_solver_ms']:.1f}" if s else "n/a",
                    f"{tag}_fno_ms": f"{s['fno_batched_ms_per_sample']:.2f}" if s else "n/a",
                    f"{tag}_mode_table": mode_table(s, {**LABELS, **STEADY_LABELS, **(
                        {"high_contrast": "Permeability contrast 1:100"} if ck == "case_c" else {})}) if s else "_not run_"})
    lb_p, pinn_p = RES / "case_b" / "leaderboard.json", RES / "case_b" / "pinn.json"
    lines = ["| Approach | Training samples | Median error | PDE residual (median) | Training (CPU min) | New instance costs |",
             "|---|---|---|---|---|---|"]
    if lb_p.exists():
        for r in json.loads(lb_p.read_text()):
            if r["n_train"] > 0:
                lines.append(f"| {r['model']} | {r['n_train']} | {pct(r['rel_l2_median'])} | {r['residual_median']:.3f} | "
                             f"{r['train_minutes']:.1f} | one forward pass ({r['batched_ms_per_sample']:.2f} ms batched) |")
    if pinn_p.exists():
        pn = json.loads(pinn_p.read_text())
        lines.append(f"| Classic PINN, no data ({len(pn['instances'])} instances, {pn['iters']} steps) | 0 | "
                     f"{pct(pn['median_rel_l2'])} | n/a | {pn['median_train_seconds'] / 60:.1f} per instance | "
                     f"a new {pn['median_train_seconds']:.0f} s training run |")
    out["pinn_table"] = "\n".join(lines) if len(lines) > 2 else "_not run_"
    return out


def main() -> None:
    s = json.loads((RES / "benchmark_summary.json").read_text())
    sc = json.loads((RES / "scaling.json").read_text()) if (RES / "scaling.json").exists() else None
    vals = {
        "date": dt.date.today().isoformat(),
        "fno_err": pct(s["fno_in_dist_rel_l2_median"]), "unet_err": pct(s["unet_in_dist_rel_l2_median"]),
        "n_eval": str(s["n_eval"]), "n_bad": str(s["n_truly_bad"]),
        "veto_p": pct(s["veto_precision"]), "veto_r": pct(s["veto_recall"]), "flag_r": pct(s["flag_recall"]),
        "acc_bad": pct(s["accepted_share_truly_bad"]), "raw_bad": pct(s["unfiltered_share_truly_bad"]),
        "vr": pct(s["in_dist_veto_rate"]), "sanity": pct(s["reference_output_accept_rate"]),
        "fd_ms": f"{s['fd_solver_ms']:.1f}", "fno_ms": f"{s['fno_batched_ms_per_sample']:.1f}",
        "speedup": f"{s['speedup_batched']:.2f}", "tol": pct(s["error_tolerance"]),
        "scale_n": str(sc["measured"][-1]["n"]) if sc else "n/a",
        "scale_speedup": f"{sc['measured'][-1]['speedup']:.1f}" if sc else "n/a",
        "mode_table": mode_table(s),
        **steady_vals(),
    }
    targets = {
        "README.md.tmpl": ROOT / "README.md",
        "CASE_STUDY.md.tmpl": ROOT / "docs" / "CASE_STUDY.md",
        "LINKEDIN_POSTS.md.tmpl": ROOT / "docs" / "LINKEDIN_POSTS.md",
    }
    for tmpl, out in targets.items():
        text = (ROOT / "docs" / "templates" / tmpl).read_text(encoding="utf-8")
        text = re.sub(r"\{\{(\w+)\}\}", lambda m: vals[m.group(1)], text)
        out.write_text(text, encoding="utf-8")
        print("wrote", out.relative_to(ROOT))
    brief = solution_brief(next(iter(VERTICALS)), s)
    (ROOT / "docs" / "ONE_PAGER.md").write_text(brief, encoding="utf-8")
    print("wrote docs/ONE_PAGER.md")


if __name__ == "__main__":
    main()
