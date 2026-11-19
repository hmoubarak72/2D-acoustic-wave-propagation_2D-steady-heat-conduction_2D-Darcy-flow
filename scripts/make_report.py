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


def mode_table(s: dict) -> str:
    lines = ["| Test set | Type | Cases | Truly bad | Vetoed | Vetoed or review | Bad cases caught |",
             "|---|---|---|---|---|---|---|"]
    for k, m in s["per_mode"].items():
        lines.append(f"| {LABELS.get(k, k)} | {m['group']} | {m['n']} | {pct(m['share_truly_bad'])} | "
                     f"{pct(m['veto_rate'])} | {pct(m['flag_rate'])} | {pct(m['recall_of_bad'])} |")
    return "\n".join(lines)


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
