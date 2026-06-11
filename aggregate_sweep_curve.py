#!/usr/bin/env python
"""Aggregate a bones_seed_small checkpoint sweep into step-vs-metric curves.

Reads RESULTS/step_<N>/<split>/<cat>.json (the per-category eval JSONs) and emits:
  <out>_long.csv     one row per (step, split, category) with all metrics
  <out>_summary.csv  one row per step, motion-weighted means across the 6 cells
Metric mapping matches the published Kimodo benchmark:
  FID = TMR/FID/gen_gt ; R@3 = TMR/t2m_R/R03 ; Skate(cm/s) = foot_skate_from_pred_contacts*100 ;
  Contact = foot_contact_consistency.  (See benchmark/evaluate_folder.py docstring.)
"""
import argparse, csv, glob, json, os, re

SPLITS = ["content", "repetition"]
CATS = ["overview", "timeline_single", "timeline_multi"]


def load_cell(path):
    d = json.load(open(path))
    t = d.get("tmr", {})
    gen = d.get("per_motion_mean_gen", {})
    gt = d.get("per_motion_mean_gt", {})
    return {
        "num_motions": d.get("num_motions", 0),
        "fid_gen_gt": t.get("TMR/FID/gen_gt"),
        "fid_gen_text": t.get("TMR/FID/gen_text"),
        "t2m_r1": t.get("TMR/t2m_R/R01"),
        "t2m_r3": t.get("TMR/t2m_R/R03"),
        "t2m_medr": t.get("TMR/t2m_R/MedR"),
        "t2m_sim": t.get("TMR/t2m_sim"),
        "m2m_r1": t.get("TMR/m2m_R/R01"),
        "gt_r3_ceiling": t.get("TMR/t2m_gt_R/R03"),
        "skate_cm_s": (gen.get("foot_skate_from_pred_contacts") or 0) * 100,
        "contact": gen.get("foot_contact_consistency"),
        "gt_skate_cm_s": (gt.get("foot_skate_from_pred_contacts") or 0) * 100,
        "gt_contact": gt.get("foot_contact_consistency"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default="sweep_curve")
    args = ap.parse_args()

    step_dirs = sorted(glob.glob(os.path.join(args.results_dir, "step_*")),
                       key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    long_rows, summary_rows = [], []

    for sd in step_dirs:
        step = int(re.search(r"step_(\d+)", sd).group(1))
        cells = []
        for split in SPLITS:
            for cat in CATS:
                p = os.path.join(sd, split, f"{cat}.json")
                if not os.path.isfile(p):
                    continue
                c = load_cell(p)
                c.update(step=step, split=split, category=cat)
                cells.append(c)
                long_rows.append(c)
        if not cells:
            continue
        # motion-weighted means across cells
        tot = sum(c["num_motions"] for c in cells) or 1

        def wmean(key):
            vals = [(c["num_motions"], c[key]) for c in cells if c.get(key) is not None]
            w = sum(n for n, _ in vals) or 1
            return sum(n * v for n, v in vals) / w

        summary_rows.append({
            "step": step, "n_cells": len(cells), "total_motions": tot,
            "fid_gen_gt": round(wmean("fid_gen_gt"), 4),
            "t2m_r1": round(wmean("t2m_r1"), 3),
            "t2m_r3": round(wmean("t2m_r3"), 3),
            "t2m_sim": round(wmean("t2m_sim"), 4),
            "skate_cm_s": round(wmean("skate_cm_s"), 4),
            "contact": round(wmean("contact"), 4),
            "gt_r3_ceiling": round(wmean("gt_r3_ceiling"), 3),
        })

    long_cols = ["step", "split", "category", "num_motions", "fid_gen_gt", "fid_gen_text",
                 "t2m_r1", "t2m_r3", "t2m_medr", "t2m_sim", "m2m_r1", "gt_r3_ceiling",
                 "skate_cm_s", "contact", "gt_skate_cm_s", "gt_contact"]
    with open(f"{args.out}_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=long_cols, extrasaction="ignore")
        w.writeheader(); w.writerows(long_rows)
    sum_cols = ["step", "n_cells", "total_motions", "fid_gen_gt", "t2m_r1", "t2m_r3",
                "t2m_sim", "skate_cm_s", "contact", "gt_r3_ceiling"]
    with open(f"{args.out}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_cols)
        w.writeheader(); w.writerows(summary_rows)

    # console readout
    print(f"\n=== bones_seed_small sweep (motion-weighted across 6 cells) ===")
    print(f"{'step':>8}{'FID':>9}{'t2m_R1':>9}{'t2m_R3':>9}{'sim':>8}{'skate':>9}{'contact':>9}")
    for r in summary_rows:
        print(f"{r['step']:>8}{r['fid_gen_gt']:>9}{r['t2m_r1']:>9}{r['t2m_r3']:>9}"
              f"{r['t2m_sim']:>8}{r['skate_cm_s']:>9}{r['contact']:>9}")
    if summary_rows:
        best_fid = min(summary_rows, key=lambda r: r["fid_gen_gt"])
        best_r3 = max(summary_rows, key=lambda r: r["t2m_r3"])
        print(f"\nbest FID:   step {best_fid['step']}  (FID {best_fid['fid_gen_gt']})")
        print(f"best t2m R@3: step {best_r3['step']}  (R@3 {best_r3['t2m_r3']})")
    print(f"\nwrote {args.out}_long.csv  and  {args.out}_summary.csv")


if __name__ == "__main__":
    main()
