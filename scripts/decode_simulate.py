#!/usr/bin/env python3
"""
DECODE Synthetic Benchmark Simulator: Tunable Deamination Read Generator
========================================================================

Overview
--------
Generates synthetic single-end or paired-end RNA-seq reads with realistic,
structure-modulated chemical deamination mutations to benchmark alignment
recovery and reactivity reconstruction across mutation loads up to the
theoretical 62.5% transition saturation ceiling:

  - Pyrimidine channel (C -> T): Unidirectional conversion of accessible
    cytidines up to saturation (100% of C -> T contributes 25% divergence on
    an equimolar genome).
  - Purine channels (A <-> G): Bidirectional purine equilibration (50% A/G
    randomized to 50/50 A:G contributes up to 37.5% additional divergence,
    for a combined theoretical divergence ceiling of 62.5%).

Workflow & Outputs (`--outdir`)
-------------------------------
1. Structural Reactivity Profile (`ground_truth_reactivity.tsv`)
   - Loads an empirical or user-supplied per-site reactivity table via
     `--reactivity` (columns: `chrom`, `pos`, `strand`, `norm_reactivity` or
     `reactivity`), OR synthesizes a block-structured stem-loop reactivity
     profile (`[0, 1]`) when `--reactivity` is omitted.

2. Read & Mutation Simulation (`sim_R1.fastq.gz`, optional `sim_R2.fastq.gz`)
   - Samples transcripts/contigs (optionally from both `+` and `-` strands
     with `--both-strands`), applies per-base deamination probabilities
     modulated by local reactivity:
         p_mut(i) = clip(p_bg + reactivity(i) * p_max, 0.0, 0.98)
     and injects base-calling errors at `--seq-error` (default: 0.002).
   - Paired-end reads are emitted in standard Illumina directional
     `fr-secondstrand` orientation (R1 sense `5'->3'`, R2 antisense `5'->3'`)
     matching the default `--orientation fr-secondstrand` of `decode_align.py`.

3. Ground-Truth Read Manifest (`ground_truth_reads.tsv.gz`)
   - Records true origin contig, strand, 1-based coordinates, mutation count,
     and exact mutation positions/types for every simulated read.

Usage Examples
--------------
1. Single-end benchmark on human rRNA (auto-generated structure profile):
    python scripts/decode_simulate.py \
        --fasta test_data/hsrRNA.fa \
        --n-reads 5000 \
        --read-len 150 \
        --outdir sim_benchmark

2. Paired-end high-load benchmark (~40-50% divergence):
    python scripts/decode_simulate.py \
        --fasta test_data/hsrRNA.fa \
        --paired-end \
        --n-reads 10000 \
        --read-len 150 \
        --ct-max 0.85 \
        --ag-max 0.80 \
        --outdir sim_high_load
"""

import argparse
import csv
import gzip
import os
import sys

import numpy as np

COMP_TAB = str.maketrans("ACGTUacgtuNn", "TGCAAtgcaaNn")
DNA_TAB = str.maketrans("ACGTUacgtuNn", "ACGTTacgttNn")
BASES = np.array(["A", "C", "G", "T"], dtype="<U1")


def open_text(path, mode="rt"):
    """Open a plain-text or gzip-compressed file transparently."""
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def revcomp(seq):
    """Return the 5'->3' reverse complement of a 4-base DNA/RNA sequence."""
    return seq.translate(COMP_TAB)[::-1]


def load_fasta(fasta_path):
    """Load reference sequences normalized to uppercase DNA alphabet."""
    if not os.path.exists(fasta_path):
        sys.exit(f"ERROR: FASTA file not found: {fasta_path}")
    seqs = {}
    curr, chunks = None, []
    with open_text(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr:
                    seqs[curr] = "".join(chunks).upper().translate(DNA_TAB)
                curr, chunks = line[1:].split()[0], []
            else:
                chunks.append(line)
        if curr:
            seqs[curr] = "".join(chunks).upper().translate(DNA_TAB)
    if not seqs:
        sys.exit(f"ERROR: No sequences found in {fasta_path}")
    return seqs


def generate_synthetic_reactivities(seqs, rng):
    """
    Create realistic stem-loop reactivity profiles in [0, 1] per transcript.

    Alternates paired stem segments (low reactivity, Beta(1.2, 12)) and
    unpaired loop/bulge segments (high reactivity, Beta(4.0, 2.0)) across
    6-14 nt structural blocks.
    """
    profiles = {}
    for cname, seq in seqs.items():
        n = len(seq)
        react = np.zeros(n, dtype=np.float64)
        pos = 0
        paired = True
        while pos < n:
            blk = int(rng.integers(6, 15))
            end = min(n, pos + blk)
            if paired:
                react[pos:end] = rng.beta(1.2, 12.0, size=end - pos)
            else:
                react[pos:end] = rng.beta(4.0, 2.0, size=end - pos)
            paired = not paired
            pos = end
        profiles[cname] = np.clip(react, 0.0, 1.0)
    return profiles


def load_or_build_reactivities(seqs, react_path, rng):
    """
    Load user-provided per-site reactivities or generate synthetic profiles.

    Expects a TSV with columns `chrom`, `pos` (1-based), and either
    `norm_reactivity` or `reactivity`. Values are clipped to [0, 1.5] and
    rescaled to [0, 1] if maximum exceeds 1.0.
    """
    if react_path is None:
        return generate_synthetic_reactivities(seqs, rng)
    if not os.path.exists(react_path):
        sys.exit(f"ERROR: Reactivity file not found: {react_path}")

    profiles = {c: np.zeros(len(s), dtype=np.float64) for c, s in seqs.items()}
    with open_text(react_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            sys.exit("ERROR: Empty reactivity TSV.")
        val_col = (
            "norm_reactivity"
            if "norm_reactivity" in reader.fieldnames
            else "reactivity"
        )
        if val_col not in reader.fieldnames or "pos" not in reader.fieldnames:
            sys.exit(
                "ERROR: Reactivity TSV requires 'pos' and "
                "'norm_reactivity' (or 'reactivity') columns."
            )
        default_chrom = next(iter(seqs))
        for row in reader:
            cname = row.get("chrom", default_chrom)
            if cname not in profiles:
                continue
            raw_v = row[val_col]
            if raw_v in ("", "NA", "nan", "NaN", None):
                continue
            pos1 = int(row["pos"])
            if 1 <= pos1 <= len(profiles[cname]):
                profiles[cname][pos1 - 1] = max(0.0, float(raw_v))

    for cname, arr in profiles.items():
        mx = float(np.max(arr))
        if mx > 1.0:
            profiles[cname] = np.clip(arr / mx, 0.0, 1.0)
    return profiles


def build_channel_probs(
    seqs, reactivities, ct_max, ag_max, ga_max, bg_rate
):
    """
    Compute per-nucleotide transition probabilities [p_CT, p_AG, p_GA].

    Returns:
        dict[str, np.ndarray]: Shape (L, 3) probability table per contig.
    """
    probs = {}
    for cname, seq in seqs.items():
        r = reactivities[cname]
        arr = np.zeros((len(seq), 3), dtype=np.float64)
        for idx, b in enumerate(seq):
            ri = r[idx]
            if b == "C":
                arr[idx, 0] = min(0.98, bg_rate + ri * ct_max)
            elif b == "A":
                arr[idx, 1] = min(0.98, bg_rate + ri * ag_max)
            elif b == "G":
                arr[idx, 2] = min(0.98, bg_rate + ri * ga_max)
        probs[cname] = arr
    return probs


def export_ground_truth_reactivity(
    seqs, reactivities, chan_probs, out_tsv
):
    """Write ground-truth per-nucleotide reactivities and mutation rates."""
    with open(out_tsv, "w", newline="") as fout:
        w = csv.writer(fout, delimiter="\t")
        w.writerow([
            "chrom", "pos", "strand", "ref_base",
            "true_reactivity", "p_CT", "p_AG", "p_GA"
        ])
        for cname, seq in seqs.items():
            r = reactivities[cname]
            p = chan_probs[cname]
            for i, b in enumerate(seq):
                w.writerow([
                    cname, i + 1, "+", b,
                    f"{r[i]:.6f}", f"{p[i, 0]:.6f}",
                    f"{p[i, 1]:.6f}", f"{p[i, 2]:.6f}"
                ])


def mutate_fragment(
    frag_seq, frag_probs, start_0, seq_err, rng
):
    """
    Apply structure-modulated deamination and sequencing errors to a fragment.

    Returns:
        tuple[str, list[int], list[str]]:
            (mutated_sense_seq, mut_positions_1based, mut_labels)
    """
    n = len(frag_seq)
    chars = list(frag_seq)
    u_deam = rng.random(n)
    u_err = rng.random(n)
    mut_pos = []
    mut_lbl = []

    for i in range(n):
        rb = chars[i]
        nb = rb
        if rb == "C" and u_deam[i] < frag_probs[i, 0]:
            nb = "T"
        elif rb == "A" and u_deam[i] < frag_probs[i, 1]:
            nb = "G"
        elif rb == "G" and u_deam[i] < frag_probs[i, 2]:
            nb = "A"

        if nb != rb:
            mut_pos.append(start_0 + i + 1)
            mut_lbl.append(f"{rb}>{nb}")

        if u_err[i] < seq_err:
            alts = [b for b in ("A", "C", "G", "T") if b != nb]
            nb = alts[int(rng.integers(0, 3))]

        chars[i] = nb

    return "".join(chars), mut_pos, mut_lbl


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Simulate deaminated RNA-seq reads for DECODE benchmarks."
    )
    p.add_argument("--fasta", required=True, help="Reference FASTA (.fa/.gz)")
    p.add_argument(
        "--reactivity", default=None,
        help="Optional TSV with 'pos' and 'norm_reactivity' columns"
    )
    p.add_argument("--outdir", default="sim_benchmark", help="Output folder")
    p.add_argument(
        "--n-reads", type=int, default=5000,
        help="Number of reads or read pairs to simulate (default: 5000)"
    )
    p.add_argument(
        "--read-len", type=int, default=150,
        help="Read length in nucleotides (default: 150)"
    )
    p.add_argument(
        "--paired-end", action="store_true",
        help="Emit paired-end R1/R2 FASTQs (fr-secondstrand orientation)"
    )
    p.add_argument(
        "--frag-mean", type=int, default=300,
        help="Mean cDNA fragment length for paired-end mode (default: 300)"
    )
    p.add_argument(
        "--frag-sd", type=int, default=50,
        help="Fragment length SD for paired-end mode (default: 50)"
    )
    p.add_argument(
        "--ct-max", type=float, default=0.70,
        help="Max C->T conversion rate at reactivity=1.0 (default: 0.70)"
    )
    p.add_argument(
        "--ag-max", type=float, default=0.60,
        help="Max A->G conversion rate at reactivity=1.0 (default: 0.60)"
    )
    p.add_argument(
        "--ga-max", type=float, default=0.02,
        help="Max G->A conversion rate at reactivity=1.0 (default: 0.02)"
    )
    p.add_argument(
        "--bg-rate", type=float, default=0.01,
        help="Baseline deamination rate at reactivity=0.0 (default: 0.01)"
    )
    p.add_argument(
        "--seq-error", type=float, default=0.002,
        help="Per-base sequencing error probability (default: 0.002)"
    )
    p.add_argument(
        "--both-strands", action="store_true",
        help="Sample transcripts from both '+' and '-' strands"
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    return p.parse_args()


def main():
    """Simulate deaminated reads and export FASTQs + ground-truth tables."""
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    seqs = load_fasta(args.fasta)
    reactivities = load_or_build_reactivities(seqs, args.reactivity, rng)
    chan_probs = build_channel_probs(
        seqs, reactivities, args.ct_max, args.ag_max, args.ga_max,
        args.bg_rate
    )

    gt_react_path = os.path.join(args.outdir, "ground_truth_reactivity.tsv")
    export_ground_truth_reactivity(
        seqs, reactivities, chan_probs, gt_react_path
    )

    cnames = list(seqs.keys())
    weights = np.array([len(seqs[c]) for c in cnames], dtype=np.float64)
    weights /= weights.sum()

    r1_path = os.path.join(args.outdir, "sim_R1.fastq.gz")
    r2_path = (
        os.path.join(args.outdir, "sim_R2.fastq.gz")
        if args.paired_end
        else None
    )
    gt_reads_path = os.path.join(args.outdir, "ground_truth_reads.tsv.gz")
    qual_str = "I" * args.read_len

    total_bases = 0
    total_muts = 0

    f2 = gzip.open(r2_path, "wt") if r2_path else None
    with gzip.open(r1_path, "wt") as f1, gzip.open(
        gt_reads_path, "wt"
    ) as fgt:
        fgt.write(
            "read_id\tchrom\tstrand\tstart\tend\t"
            "n_mutations\tmut_positions\tmut_types\n"
        )
        for idx in range(args.n_reads):
            cname = cnames[int(rng.choice(len(cnames), p=weights))]
            seq = seqs[cname]
            probs = chan_probs[cname]
            clen = len(seq)
            strand = (
                "-" if (args.both_strands and rng.random() < 0.5) else "+"
            )

            if args.paired_end:
                flen = int(
                    np.clip(
                        rng.normal(args.frag_mean, args.frag_sd),
                        args.read_len,
                        clen,
                    )
                )
            else:
                flen = min(args.read_len, clen)

            max_start = max(0, clen - flen)
            s0 = int(rng.integers(0, max_start + 1))
            e0 = s0 + flen

            frag_seq = seq[s0:e0]
            frag_probs = probs[s0:e0]
            if strand == "-":
                frag_seq = revcomp(frag_seq)
                frag_probs = frag_probs[::-1]

            mut_frag, mpos_rel, mlbl = mutate_fragment(
                frag_seq, frag_probs, 0, args.seq_error, rng
            )
            if strand == "+":
                mpos_nat = [s0 + p for p in mpos_rel]
            else:
                mpos_nat = [e0 - p + 1 for p in reversed(mpos_rel)]
                mlbl = list(reversed(mlbl))

            rlen = min(args.read_len, len(mut_frag))
            q_slice = qual_str[:rlen]
            rid = f"sim_{idx:07d}"

            r1_seq = mut_frag[:rlen]
            f1.write(f"@{rid}/1\n{r1_seq}\n+\n{q_slice}\n")
            if f2 is not None:
                r2_seq = revcomp(mut_frag[-rlen:])
                f2.write(f"@{rid}/2\n{r2_seq}\n+\n{q_slice}\n")

            total_bases += flen
            total_muts += len(mpos_nat)
            p_str = ",".join(map(str, mpos_nat)) if mpos_nat else "."
            t_str = ",".join(mlbl) if mlbl else "."
            fgt.write(
                f"{rid}\t{cname}\t{strand}\t{s0 + 1}\t{e0}\t"
                f"{len(mpos_nat)}\t{p_str}\t{t_str}\n"
            )

    if f2 is not None:
        f2.close()

    div_pct = 100.0 * total_muts / max(1, total_bases)
    print(
        f"Simulated {args.n_reads} templates ({div_pct:.2f}% deamination "
        f"divergence) -> {args.outdir}"
    )


if __name__ == "__main__":
    main()
