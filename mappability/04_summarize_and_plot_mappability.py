#!/usr/bin/env python3
"""
DECODE Mappability Step 4: BedGraph Summarization & Multi-Panel Plotting
=========================================================================

Overview
--------
Provides two subcommands for the dual-strand GenMap mappability pipeline:

1. `summarize` (invoked per track by `03_run_genmap_array.slurm`)
   - Intersects a sorted GenMap `.bedgraph` track against the five dual-strand
     biotype BED files (`whole_genome`, `coding_exons`, `nascent_introns`,
     `lncRNAs`, `multicopy_small_ncRNAs`) in linear two-pointer time.
   - Computes per-biotype uniquely mappable fraction (`freq == 1.0`, i.e.,
     single genomic occurrence) and mean mappability (`mean(1 / occurrences)`).

2. `plot` (invoked after all 13 array tasks complete)
   - Merges all per-track `.tsv` files in `--stats_dir` into
     `<out_prefix>_summary.tsv`.
   - Computes and prints interpolated K_90 and K_95 read-length uniqueness
     thresholds across all five genomic/transcriptomic strata.
   - Renders a 5-panel publication-grade vector PDF (`<out_prefix>.pdf`)
     comparing 4-base (A/C/G/T) vs. 2-base (G/T) uniqueness across read
     lengths K = 20..1000 nt and mismatch tolerances E = 0, 1.

Usage Examples
--------------
Summarize a single GenMap bedgraph track:
    python mappability/04_summarize_and_plot_mappability.py summarize \
        --bedgraph /tmp/track.bedgraph \
        --beds_dir refs_dual_strand/beds \
        --alphabet 2base_GT \
        --k 150 \
        --e 1 \
        --out stats/stats_2base_GT_K150_E1.tsv

Merge all 48 track summaries and generate plots:
    python mappability/04_summarize_and_plot_mappability.py plot \
        --stats_dir stats \
        --out_prefix decode_human_mappability
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

BIOTYPES = [
    ("whole_genome", "Whole Genome"),
    ("coding_exons", "Coding Exons"),
    ("nascent_introns", "Nascent Introns"),
    ("lncRNAs", "lncRNAs"),
    ("multicopy_small_ncRNAs", "Multicopy / Small ncRNAs"),
]


def load_bed_intervals(bed_path):
    """Load sorted 0-based [start, end) intervals grouped by contig."""
    if not os.path.exists(bed_path):
        sys.exit(f"ERROR: Missing BED file: {bed_path}")
    by_chrom = defaultdict(list)
    total_bp = 0
    with open(bed_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            s, e = int(parts[1]), int(parts[2])
            if e > s:
                by_chrom[parts[0]].append((s, e))
                total_bp += e - s
    return by_chrom, total_bp


def intersect_bedgraph_with_beds(bg_path, bed_models):
    """
    Stream a sorted GenMap `.bedgraph` file and intersect with all BED strata.

    Uses per-contig pointer advancement in O(N_bg + N_bed) time and minimal
    memory, accumulating covered bases, uniquely mappable bases (`freq >=
    0.999999`), and frequency-weighted mappable bases per biotype.
    """
    if not os.path.exists(bg_path):
        sys.exit(f"ERROR: BedGraph file not found: {bg_path}")

    n_strata = len(bed_models)
    covered_bp = np.zeros(n_strata, dtype=np.int64)
    unique_bp = np.zeros(n_strata, dtype=np.int64)
    weighted_bp = np.zeros(n_strata, dtype=np.float64)

    curr_chrom = None
    chrom_ivs = [[] for _ in range(n_strata)]
    ptrs = [0] * n_strata

    with open(bg_path) as fh:
        for line in fh:
            if not line or line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[0]
            gs = int(parts[1])
            ge = int(parts[2])
            if ge <= gs:
                continue
            freq = float(parts[3])
            is_uniq = freq >= 0.999999

            if chrom != curr_chrom:
                curr_chrom = chrom
                for idx, (_, by_chrom, _) in enumerate(bed_models):
                    chrom_ivs[idx] = by_chrom.get(chrom, [])
                    ptrs[idx] = 0

            for idx in range(n_strata):
                ivs = chrom_ivs[idx]
                n_iv = len(ivs)
                p = ptrs[idx]
                while p < n_iv and ivs[p][1] <= gs:
                    p += 1
                ptrs[idx] = p
                k = p
                while k < n_iv and ivs[k][0] < ge:
                    ov = min(ge, ivs[k][1]) - max(gs, ivs[k][0])
                    if ov > 0:
                        covered_bp[idx] += ov
                        weighted_bp[idx] += ov * freq
                        if is_uniq:
                            unique_bp[idx] += ov
                    k += 1

    return covered_bp, unique_bp, weighted_bp


def cmd_summarize(args):
    """Run `summarize` subcommand on a single GenMap BedGraph track."""
    bed_models = []
    for bkey, _ in BIOTYPES:
        bpath = os.path.join(args.beds_dir, f"{bkey}.bed")
        by_chrom, total_bp = load_bed_intervals(bpath)
        bed_models.append((bkey, by_chrom, total_bp))

    cov_arr, uniq_arr, wgt_arr = intersect_bedgraph_with_beds(
        args.bedgraph, bed_models
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", newline="") as fout:
        w = csv.writer(fout, delimiter="\t")
        w.writerow([
            "alphabet", "K", "E", "biotype",
            "total_bp", "covered_bp", "unique_bp",
            "frac_unique", "mean_mappability"
        ])
        for idx, (bkey, _, tot_bp) in enumerate(bed_models):
            denom = max(1, tot_bp)
            f_uniq = uniq_arr[idx] / denom
            m_map = wgt_arr[idx] / denom
            w.writerow([
                args.alphabet, args.k, args.e, bkey,
                tot_bp, int(cov_arr[idx]), int(uniq_arr[idx]),
                f"{f_uniq:.6f}", f"{m_map:.6f}"
            ])
    print(f"Wrote mappability summary -> {args.out}")


def interpolate_k_threshold(ks, fracs, target):
    """Estimate minimum K where uniqueness reaches `target` via log-K interp."""
    if len(ks) == 0:
        return None
    if fracs[0] >= target:
        return float(ks[0])
    for i in range(1, len(ks)):
        if fracs[i] >= target:
            y0, y1 = fracs[i - 1], fracs[i]
            if y1 <= y0:
                return float(ks[i])
            w = (target - y0) / (y1 - y0)
            log_k = np.log10(ks[i - 1]) + w * (
                np.log10(ks[i]) - np.log10(ks[i - 1])
            )
            return float(10.0 ** log_k)
    return None


def cmd_plot(args):
    """Merge per-track TSVs, print K90/K95 milestones, and render figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    pattern = os.path.join(args.stats_dir, "*.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"ERROR: No .tsv files found in {args.stats_dir}")

    rows = []
    for fpath in files:
        with open(fpath) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for r in reader:
                rows.append({
                    "alphabet": r["alphabet"],
                    "K": int(r["K"]),
                    "E": int(r["E"]),
                    "biotype": r["biotype"],
                    "total_bp": int(r["total_bp"]),
                    "covered_bp": int(r["covered_bp"]),
                    "unique_bp": int(r["unique_bp"]),
                    "frac_unique": float(r["frac_unique"]),
                    "mean_mappability": float(r["mean_mappability"]),
                })

    rows.sort(key=lambda r: (r["biotype"], r["alphabet"], r["E"], r["K"]))
    summary_tsv = f"{args.out_prefix}_summary.tsv"
    with open(summary_tsv, "w", newline="") as fout:
        w = csv.writer(fout, delimiter="\t")
        w.writerow([
            "alphabet", "K", "E", "biotype",
            "total_bp", "covered_bp", "unique_bp",
            "frac_unique", "mean_mappability"
        ])
        for r in rows:
            w.writerow([
                r["alphabet"], r["K"], r["E"], r["biotype"],
                r["total_bp"], r["covered_bp"], r["unique_bp"],
                f"{r['frac_unique']:.6f}", f"{r['mean_mappability']:.6f}"
            ])
    print(
        f"Merged {len(files)} track files ({len(rows)} rows) -> "
        f"{summary_tsv}"
    )

    # Print K90 and K95 milestones across biotypes
    print("\nUniqueness Read-Length Milestones (K90 / K95):")
    print(
        f"{'Biotype':<25s} {'Series':<16s} "
        f"{'K90 (nt)':>10s} {'K95 (nt)':>10s}"
    )
    print("-" * 65)
    series_specs = [
        ("4base", 0, "4-base (E=0)"),
        ("4base", 1, "4-base (E=1)"),
        ("2base_GT", 0, "2-base G/T (E=0)"),
        ("2base_GT", 1, "2-base G/T (E=1)"),
    ]
    for bkey, blabel in BIOTYPES:
        for alph, e_val, slabel in series_specs:
            sub = [
                r for r in rows
                if r["biotype"] == bkey
                and r["alphabet"] == alph
                and r["E"] == e_val
            ]
            sub.sort(key=lambda r: r["K"])
            ks = [r["K"] for r in sub]
            fs = [r["frac_unique"] for r in sub]
            k90 = interpolate_k_threshold(ks, fs, 0.90)
            k95 = interpolate_k_threshold(ks, fs, 0.95)
            s90 = f"{k90:.1f}" if k90 is not None else ">1000"
            s95 = f"{k95:.1f}" if k95 is not None else ">1000"
            print(f"{blabel:<25s} {slabel:<16s} {s90:>10s} {s95:>10s}")

    # Build 5-panel figure (2x3 grid with legend in panel 6)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharey=True)
    axes_flat = axes.flatten()

    styles = {
        ("4base", 0): dict(
            color="#374151", ls="-", marker="o", label="4-base (E = 0)"
        ),
        ("4base", 1): dict(
            color="#6B7280", ls="--", marker="s", label="4-base (E = 1)"
        ),
        ("2base_GT", 0): dict(
            color="#0284C7", ls="-", marker="o", label="2-base G/T (E = 0)"
        ),
        ("2base_GT", 1): dict(
            color="#D97706", ls="--", marker="^", label="2-base G/T (E = 1)"
        ),
    }

    for idx, (bkey, blabel) in enumerate(BIOTYPES):
        ax = axes_flat[idx]
        for (alph, e_val), st in styles.items():
            sub = [
                r for r in rows
                if r["biotype"] == bkey
                and r["alphabet"] == alph
                and r["E"] == e_val
            ]
            sub.sort(key=lambda r: r["K"])
            if not sub:
                continue
            ks = [r["K"] for r in sub]
            ys = [100.0 * r["frac_unique"] for r in sub]
            ax.plot(
                ks, ys,
                color=st["color"], linestyle=st["ls"], marker=st["marker"],
                markersize=4.5, linewidth=1.8, label=st["label"]
            )
        ax.axhline(90.0, color="#9CA3AF", linestyle=":", linewidth=1.0)
        ax.axhline(95.0, color="#D1D5DB", linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xticks([20, 30, 50, 100, 150, 300, 500, 1000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.tick_params(axis="x", rotation=35, labelsize=8.5)
        ax.set_ylim(0, 103)
        ax.set_title(blabel, fontsize=10.5, fontweight="bold")
        ax.grid(True, which="major", linestyle="--", alpha=0.3)
        if idx % 3 == 0:
            ax.set_ylabel("Uniquely Mappable Bases (%)", fontsize=10)
        if idx >= 3:
            ax.set_xlabel("Read Length K (nt)", fontsize=10)

    # Use 6th subplot cell for shared legend
    ax_leg = axes_flat[5]
    ax_leg.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    ax_leg.legend(
        handles, labels, loc="center", frameon=True, fontsize=10,
        title="Alphabet & Mismatch Tolerance", title_fontsize=10.5
    )

    fig.suptitle(
        "Dual-Strand Transcriptome & Genome Mappability in 2-Base (G/T) Space",
        fontsize=12.5, fontweight="bold", y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    pdf_path = f"{args.out_prefix}.pdf"
    png_path = f"{args.out_prefix}.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figures -> {pdf_path} and {png_path}")


def parse_args():
    """Parse CLI subcommands (`summarize` and `plot`)."""
    p = argparse.ArgumentParser(
        description="Summarize GenMap bedgraphs and plot mappability curves."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summarize", help="Summarize one BedGraph track.")
    p_sum.add_argument("--bedgraph", required=True, help="GenMap .bedgraph")
    p_sum.add_argument(
        "--beds_dir", default="refs_dual_strand/beds",
        help="Directory containing biotype BED files"
    )
    p_sum.add_argument("--alphabet", required=True, help="4base or 2base_GT")
    p_sum.add_argument("--k", type=int, required=True, help="Read length K")
    p_sum.add_argument("--e", type=int, required=True, help="Max errors E")
    p_sum.add_argument("--out", required=True, help="Output .tsv path")

    p_plt = sub.add_parser("plot", help="Merge TSVs and plot multi-panel PDF.")
    p_plt.add_argument(
        "--stats_dir", default="stats", help="Directory with per-track TSVs"
    )
    p_plt.add_argument(
        "--out_prefix", default="decode_human_mappability",
        help="Output prefix for summary TSV and PDF/PNG plots"
    )
    return p.parse_args()


def main():
    """Dispatch `summarize` or `plot` subcommand."""
    args = parse_args()
    if args.cmd == "summarize":
        cmd_summarize(args)
    elif args.cmd == "plot":
        cmd_plot(args)


if __name__ == "__main__":
    main()
