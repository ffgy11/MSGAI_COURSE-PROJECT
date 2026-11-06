#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="CSV containing rcc_score and baseline_correct")
    ap.add_argument("--out_csv", default="rcc_selective_answering_curve.csv")
    ap.add_argument("--out_png", default="rcc_selective_answering_curve.png")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    risk = pd.to_numeric(df["rcc_score"], errors="raise")
    correct = df["baseline_correct"].astype(bool)
    baseline_acc = float(correct.mean())
    baseline_err = 1.0 - baseline_acc

    rows = []
    for abstain_rate in [0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50]:
        if abstain_rate == 0:
            keep = np.ones(len(df), dtype=bool)
            threshold = np.inf
        else:
            threshold = float(np.quantile(risk, 1.0-abstain_rate))
            keep = (risk < threshold).to_numpy()

        acc = float(correct[keep].mean())
        err = 1.0 - acc
        rejected_acc = float(correct[~keep].mean()) if (~keep).sum() else np.nan
        rows.append({
            "abstain_rate": abstain_rate,
            "coverage": float(keep.mean()),
            "retained_n": int(keep.sum()),
            "abstained_n": int((~keep).sum()),
            "risk_threshold": threshold,
            "retained_accuracy": acc,
            "retained_error_rate": err,
            "relative_error_reduction": (
                (baseline_err-err)/baseline_err if baseline_err > 0 else np.nan
            ),
            "abstained_subset_accuracy": rejected_acc,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(out.to_string(index=False))
    print(f"\nBaseline accuracy: {100*baseline_acc:.2f}%")

    fig, ax = plt.subplots(figsize=(6.4,4.2))
    ax.plot(out["coverage"]*100, out["retained_accuracy"]*100, marker="o")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy on answered samples (%)")
    ax.set_title("RCC selective answering on CommonsenseQA")
    ax.grid(True, alpha=0.25)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_png}")

if __name__ == "__main__":
    main()
