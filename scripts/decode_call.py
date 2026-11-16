#!/usr/bin/env python3
"""
DECODE Stage 2: Bayesian Paralog Resolution & Deamination Quantification
=========================================================================

Overview
--------
After Stage 1 (`decode_align.py`) aligns reads in degenerate 2-base (G/T)
space and restores original 4-base sequences, homologous genes, pseudogenes,
or repetitive paralogs that share identical 2-base projections can yield
multiple candidate alignments (`NH > 1`).

Stage 2 resolves locus ambiguity and quantifies single-molecule and ensemble
deamination signals in three steps:

1. Empirical Prior Estimation (Uniquely Mapped Reads, `NH == 1`)
   - Streams uniquely mapped reads to estimate transcript/contig abundance
     priors and empirical Bayes smoothed site-specific transition rates
     (C -> T, A -> G, G -> A).

2. Chemistry- & Structure-Informed Bayesian Multi-Mapper Scoring (`NH > 1`)
   - Evaluates each candidate locus using an exact, normalized 4-base
     transition likelihood model conditioned on Phred base qualities:
       * Directional pyrimidine asymmetry: Cytidine deamination converts
         C -> T/U, whereas T -> C occurs only at sequencing error rates
         (~1e-3). Observing an unconverted C strongly penalizes T-bearing
         paralogs.
       * Asymmetric purine kinetics: Incorporates distinct A -> G (inosine)
         and G -> A transition probabilities plus local structural priors.
   - Multi-mappers exceeding `--min-posterior` (default: 0.80) are assigned
     to their maximum a posteriori (MAP) locus.

3. Native-Coordinate Single-Molecule & Ensemble Calling
   - Projects `_fwd` and `_rev` dual-strand alignments back to native 1-based
     coordinates (`+` and `-` strands) and exports:
       * `ensemble_reactivity.tsv`       Per-base depth, mutation counts,
                                         raw rates, and 2%-8% normalized
                                         structural reactivities.
       * `single_molecule_reads.tsv.gz`  Read-level mutation coordinates,
                                         types, and posterior confidences for
                                         downstream clustering/covariance.

Usage Examples
--------------
Standard calling using a DECODE index directory:
    python scripts/decode_call.py \
        --bam sample.restored.bam \
        --index decode_idx \
        --outdir decode_results

Calling directly from an original reference FASTA:
    python scripts/decode_call.py \
        --bam sample.restored.bam \
        --fasta test_data/hsrRNA.fa \
        --outdir decode_results
"""

import argparse
import csv
import gzip
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pysam

COMP_TAB = str.maketrans("ACGTUacgtuNn", "TGCAAtgcaaNn")
DNA_TAB = str.maketrans("ACGTUacgtuNn", "ACGTTacgttNn")
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def open_text(path, mode="rt"):
    """Open a plain-text or gzip-compressed file transparently."""
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def revcomp(seq):
    """Return the 5'->3' reverse complement of a 4-base DNA/RNA sequence."""
    return seq.translate(COMP_TAB)[::-1]


def load_dual_sequences(fasta_path=None, index_dir=None):
    """
    Load dual-strand (`_fwd` and `_rev`) 4-base reference sequences.

    Accepts either `--index` (loads `reference_4base_dual.fa` directly) or
    `--fasta` (builds `_fwd` and `_rev` contigs in memory if not already
    dual-strand).

    Returns:
        dict[str, str]: Mapping from dual contig name to uppercase 4-base seq.
    """
    target_fa = None
    already_dual = False
    if index_dir:
        cand = os.path.join(index_dir, "reference_4base_dual.fa")
        if os.path.exists(cand):
            target_fa = cand
            already_dual = True
    if target_fa is None and fasta_path:
        target_fa = fasta_path
    if target_fa is None or not os.path.exists(target_fa):
        sys.exit("ERROR: Provide a valid --index or --fasta reference.")

    raw = {}
    curr, chunks = None, []
    with open_text(target_fa) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr:
                    raw[curr] = "".join(chunks).upper().translate(DNA_TAB)
                curr, chunks = line[1:].split()[0], []
            else:
                chunks.append(line)
        if curr:
            raw[curr] = "".join(chunks).upper().translate(DNA_TAB)

    if already_dual or any(k.endswith("_fwd") for k in raw):
        return raw

    dual = {}
    for name, seq in raw.items():
        dual[f"{name}_fwd"] = seq
        dual[f"{name}_rev"] = revcomp(seq)
    return dual


def init_pileup_arrays(dual_seqs):
    """
    Allocate per-contig integer arrays tracking pileup counts.

    Columns per position:
      0: total aligned read coverage
      1: C -> T transitions
      2: A -> G transitions
      3: G -> A transitions
      4: other substitutions (transversions)
    """
    return {
        cname: np.zeros((len(seq), 5), dtype=np.int32)
        for cname, seq in dual_seqs.items()
    }


def extract_read_events(aln, ref_seq, active_transitions):
    """
    Extract aligned bases and deamination events for a single BAM alignment.

    Walks CIGAR operations in 5' -> 3' transcript orientation on the dual
    contig (`_fwd` or `_rev`).

    Returns:
        tuple: (aligned_tuples, mut_positions_1based, mut_labels,
                n_indels, span_start_0, span_end_0)
            where aligned_tuples is a list of (ref_pos_0, ref_b, obs_b, qual).
    """
    qseq = aln.query_sequence.upper()
    quals = aln.query_qualities
    q_pos = 0
    r_pos = aln.reference_start
    aligned = []
    mut_pos = []
    mut_lbl = []
    n_indels = 0
    r_len = len(ref_seq)

    for op, length in aln.cigartuples:
        if op in (0, 7, 8):  # M, =, X
            for i in range(length):
                rp = r_pos + i
                if rp >= r_len:
                    break
                rb = ref_seq[rp]
                qb = qseq[q_pos + i]
                qv = int(quals[q_pos + i]) if quals is not None else 30
                aligned.append((rp, rb, qb, qv))
                if rb in BASE_TO_IDX and qb in BASE_TO_IDX and rb != qb:
                    tcode = f"{rb}{qb}"
                    if tcode in active_transitions:
                        mut_pos.append(rp + 1)
                        mut_lbl.append(f"{rb}>{qb}")
            q_pos += length
            r_pos += length
        elif op == 1:  # I
            q_pos += length
            n_indels += length
        elif op == 2:  # D
            r_pos += length
            n_indels += length
        elif op == 3:  # N (intron skip)
            r_pos += length
        elif op == 4:  # S (soft clip)
            q_pos += length

    return aligned, mut_pos, mut_lbl, n_indels, aln.reference_start, r_pos


def update_pileup(pileup_arr, aligned_tuples):
    """Increment per-position depth and transition counts from aligned bases."""
    for rp, rb, qb, _ in aligned_tuples:
        if rb not in BASE_TO_IDX or qb not in BASE_TO_IDX:
            continue
        pileup_arr[rp, 0] += 1
        if rb == qb:
            continue
        if rb == "C" and qb == "T":
            pileup_arr[rp, 1] += 1
        elif rb == "A" and qb == "G":
            pileup_arr[rp, 2] += 1
        elif rb == "G" and qb == "A":
            pileup_arr[rp, 3] += 1
        else:
            pileup_arr[rp, 4] += 1


def compute_empirical_priors(dual_seqs, pileups, locus_counts, prior_strength):
    """
    Compute empirical Bayes site-specific transition rates & locus priors.

    Shrinks noisy low-coverage sites toward the global base-specific mean
    deamination rate:
        p_hat(c, i) = (k_{c,i} + alpha * p_global) / (n_{c,i} + alpha)

    Returns:
        tuple[dict[str, np.ndarray], dict[str, float]]:
            (site_rates[contig] -> shape (L, 3) for [p_CT, p_AG, p_GA],
             log_locus_prior[contig])
    """
    totals = np.zeros(3, dtype=np.float64)   # C, A, G coverage
    muts = np.zeros(3, dtype=np.float64)     # C->T, A->G, G->A counts

    for cname, seq in dual_seqs.items():
        arr = pileups[cname]
        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        for idx, (ascii_code, col) in enumerate(((67, 1), (65, 2), (71, 3))):
            mask = seq_bytes == ascii_code
            totals[idx] += arr[mask, 0].sum()
            muts[idx] += arr[mask, col].sum()

    defaults = np.array([0.05, 0.03, 0.005], dtype=np.float64)
    g_rates = np.where(totals > 0, muts / np.maximum(1.0, totals), defaults)
    g_rates = np.clip(g_rates, 1e-4, 0.90)

    site_rates = {}
    for cname, seq in dual_seqs.items():
        arr = pileups[cname]
        cov = arr[:, 0].astype(np.float64)
        rates = np.empty((len(seq), 3), dtype=np.float64)
        for idx in range(3):
            k = arr[:, idx + 1].astype(np.float64)
            p_hat = (k + prior_strength * g_rates[idx]) / (
                cov + prior_strength
            )
            rates[:, idx] = np.clip(p_hat, 1e-4, 0.95)
        site_rates[cname] = rates

    total_reads = sum(locus_counts.values()) + len(dual_seqs)
    log_priors = {
        cname: math.log((locus_counts.get(cname, 0) + 1.0) / total_reads)
        for cname in dual_seqs
    }
    return site_rates, log_priors


def score_candidate_hit(
    hit_records, dual_seqs, site_rates, log_priors, indel_penalty=4.0
):
    """
    Compute Bayesian log-posterior numerator for a candidate locus hit.

    Combines locus abundance log-prior with base-by-base 4-base transition
    log-likelihoods across all mate alignments belonging to the candidate hit.
    """
    first_cname = hit_records[0][0]
    ll = log_priors.get(first_cname, -10.0)

    for cname, aligned, n_indels in hit_records:
        rates = site_rates[cname]
        ll -= indel_penalty * n_indels
        for rp, rb, qb, qv in aligned:
            if rb not in BASE_TO_IDX or qb not in BASE_TO_IDX:
                continue
            eps = max(1e-4, min(0.25, 10.0 ** (-qv / 10.0)))
            eps3 = eps / 3.0
            one_minus_eps = 1.0 - eps

            if rb == "C":
                p = rates[rp, 0]
                if qb == "C":
                    prob = (1.0 - p) * one_minus_eps + p * eps3
                elif qb == "T":
                    prob = p * one_minus_eps + (1.0 - p) * eps3
                else:
                    prob = eps3
            elif rb == "T":
                prob = one_minus_eps if qb == "T" else eps3
            elif rb == "A":
                p = rates[rp, 1]
                if qb == "A":
                    prob = (1.0 - p) * one_minus_eps + p * eps3
                elif qb == "G":
                    prob = p * one_minus_eps + (1.0 - p) * eps3
                else:
                    prob = eps3
            else:  # rb == "G"
                p = rates[rp, 2]
                if qb == "G":
                    prob = (1.0 - p) * one_minus_eps + p * eps3
                elif qb == "A":
                    prob = p * one_minus_eps + (1.0 - p) * eps3
                else:
                    prob = eps3

            ll += math.log(max(1e-12, prob))
    return ll


def resolve_multimapper(
    cand_groups, dual_seqs, site_rates, log_priors, min_posterior
):
    """
    Score all candidate hits for a multi-mapping read/pair and select MAP hit.

    Returns:
        tuple[list | None, float]: (winning_hit_records, posterior_prob)
    """
    if not cand_groups:
        return None, 0.0
    scores = [
        score_candidate_hit(grp, dual_seqs, site_rates, log_priors)
        for grp in cand_groups
    ]
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    z = sum(exps)
    posteriors = [e / z for e in exps]
    best_idx = int(np.argmax(posteriors))
    best_p = posteriors[best_idx]
    if best_p < min_posterior:
        return None, best_p
    return cand_groups[best_idx], best_p


def to_native_coords(dual_cname, contig_len, start_0, end_0, mut_pos_1):
    """
    Convert dual-strand (`_fwd` / `_rev`) coordinates to native coordinates.

    For `<chrom>_fwd`: strand is '+', coordinates are unchanged.
    For `<chrom>_rev`: strand is '-', 0-based [start_0, end_0) maps to
    1-based inclusive [contig_len - end_0 + 1, contig_len - start_0], and
    each 1-based position p maps to (contig_len - p + 1) in reverse order.
    """
    if dual_cname.endswith("_rev"):
        native_chrom = dual_cname[:-4]
        strand = "-"
        n_start = contig_len - end_0 + 1
        n_end = contig_len - start_0
        n_muts = [contig_len - p + 1 for p in reversed(mut_pos_1)]
        return native_chrom, strand, n_start, n_end, n_muts

    native_chrom = (
        dual_cname[:-4] if dual_cname.endswith("_fwd") else dual_cname
    )
    return native_chrom, "+", start_0 + 1, end_0, mut_pos_1


def write_single_molecule_record(
    fh, qname, dual_cname, ref_seq, posterior, mut_pos, mut_lbl, start_0, end_0
):
    """Write one read/fragment line to `single_molecule_reads.tsv.gz`."""
    n_chrom, strand, n_start, n_end, n_muts = to_native_coords(
        dual_cname, len(ref_seq), start_0, end_0, mut_pos
    )
    if strand == "-":
        mut_lbl = list(reversed(mut_lbl))
    span = max(1, n_end - n_start + 1)
    m_rate = len(n_muts) / span
    pos_str = ",".join(map(str, n_muts)) if n_muts else "."
    typ_str = ",".join(mut_lbl) if mut_lbl else "."
    fh.write(
        f"{qname}\t{n_chrom}\t{strand}\t{n_start}\t{n_end}\t"
        f"{posterior:.4f}\t{len(n_muts)}\t{m_rate:.5f}\t{pos_str}\t{typ_str}\n"
    )


def winsorized_normalize(raw_rates, valid_mask):
    """
    Normalize raw mutation rates using 2%-8% Winsorized scaling.

    Caps extreme outliers at the 98th percentile and scales by the mean of
    the top 10% below the cap (minimum 90th percentile), placing reactivities
    on a standard [0, ~1.5+] structural probing scale.
    """
    norm = np.full_like(raw_rates, np.nan, dtype=np.float64)
    vals = raw_rates[valid_mask]
    if vals.size == 0:
        return norm
    pos_vals = vals[vals > 0]
    if pos_vals.size < 10:
        scale = float(np.max(vals)) if np.max(vals) > 0 else 1.0
        norm[valid_mask] = vals / scale
        return norm

    cap = float(np.percentile(pos_vals, 98.0))
    capped = np.minimum(pos_vals, cap)
    p90 = float(np.percentile(capped, 90.0))
    top_decile = capped[capped >= p90]
    norm_factor = float(np.mean(top_decile)) if top_decile.size > 0 else cap
    if norm_factor <= 0.0:
        norm_factor = 1.0
    norm[valid_mask] = np.minimum(vals, cap) / norm_factor
    return norm


def export_ensemble_table(
    dual_seqs, pileups, active_transitions, min_depth, out_tsv
):
    """Write native-coordinate per-nucleotide reactivity table."""
    rows = []
    for dual_cname, seq in dual_seqs.items():
        arr = pileups[dual_cname]
        clen = len(seq)
        is_rev = dual_cname.endswith("_rev")
        n_chrom = (
            dual_cname[:-4]
            if dual_cname.endswith(("_fwd", "_rev"))
            else dual_cname
        )
        strand = "-" if is_rev else "+"

        for rp in range(clen):
            depth = int(arr[rp, 0])
            if depth == 0:
                continue
            rb = seq[rp]
            n_ct, n_ag, n_ga, n_oth = map(int, arr[rp, 1:5])
            active_hits = 0
            informative = False
            if rb == "C" and "CT" in active_transitions:
                active_hits = n_ct
                informative = True
            elif rb == "A" and "AG" in active_transitions:
                active_hits = n_ag
                informative = True
            elif rb == "G" and "GA" in active_transitions:
                active_hits = n_ga
                informative = True

            native_pos = (clen - rp) if is_rev else (rp + 1)
            raw_rate = (
                (active_hits / depth) if (informative and depth > 0) else np.nan
            )
            rows.append([
                n_chrom, native_pos, strand, rb, depth,
                int(informative and depth >= min_depth),
                n_ct, n_ag, n_ga, n_oth, raw_rate
            ])

    rows.sort(key=lambda r: (r[0], r[2], r[1]))
    raw_arr = np.array([r[10] for r in rows], dtype=np.float64)
    valid_mask = np.array([bool(r[5]) and not np.isnan(r[10]) for r in rows])
    norm_arr = winsorized_normalize(raw_arr, valid_mask)

    with open(out_tsv, "w", newline="") as fout:
        w = csv.writer(fout, delimiter="\t")
        w.writerow([
            "chrom", "pos", "strand", "ref_base", "depth", "is_informative",
            "n_CT", "n_AG", "n_GA", "n_other", "raw_rate", "norm_reactivity"
        ])
        for row, nval in zip(rows, norm_arr):
            r_str = f"{row[10]:.6f}" if not np.isnan(row[10]) else "NA"
            n_str = f"{nval:.6f}" if not np.isnan(nval) else "NA"
            w.writerow([
                row[0], row[1], row[2], row[3], row[4], row[5],
                row[6], row[7], row[8], row[9], r_str, n_str
            ])


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Bayesian paralog resolution & DECODE mutation calling."
    )
    p.add_argument("--bam", required=True, help="4-base restored BAM file")
    p.add_argument("--index", default=None, help="Directory from decode_index")
    p.add_argument("--fasta", default=None, help="Original reference FASTA")
    p.add_argument("--outdir", default="decode_results", help="Output folder")
    p.add_argument(
        "--transitions", default="CT,AG",
        help="Comma-separated active deamination channels (default: CT,AG)"
    )
    p.add_argument(
        "--min-posterior", type=float, default=0.80,
        help="Minimum posterior probability for multi-mappers (default: 0.80)"
    )
    p.add_argument(
        "--min-mapq", type=int, default=0,
        help="Minimum MAPQ for uniquely mapped reads (default: 0)"
    )
    p.add_argument(
        "--min-depth", type=int, default=20,
        help="Minimum read depth for normalized reactivity (default: 20)"
    )
    p.add_argument(
        "--prior-strength", type=float, default=10.0,
        help="Empirical Bayes pseudo-count strength (default: 10.0)"
    )
    return p.parse_args()


def main():
    """Run two-stage Bayesian paralog resolution and mutation calling."""
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    active_trans = {
        t.strip().upper() for t in args.transitions.split(",") if t.strip()
    }
    dual_seqs = load_dual_sequences(fasta_path=args.fasta, index_dir=args.index)
    pileups = init_pileup_arrays(dual_seqs)
    locus_counts = defaultdict(int)
    multimappers = defaultdict(lambda: defaultdict(list))

    sm_path = os.path.join(args.outdir, "single_molecule_reads.tsv.gz")
    n_unique = 0
    n_multi_reads = 0

    with pysam.AlignmentFile(args.bam, "rb") as bam, gzip.open(
        sm_path, "wt"
    ) as fsm:
        fsm.write(
            "read_id\tchrom\tstrand\tstart\tend\tposterior\t"
            "n_mutations\tmut_rate\tmut_positions\tmut_types\n"
        )
        for aln in bam:
            if aln.is_unmapped or aln.is_duplicate or aln.is_qcfail:
                continue
            if not aln.cigartuples or aln.query_sequence is None:
                continue
            cname = aln.reference_name
            if cname not in dual_seqs:
                continue

            nh = aln.get_tag("NH") if aln.has_tag("NH") else 1
            hi = aln.get_tag("HI") if aln.has_tag("HI") else 1
            aligned, mpos, mlbl, n_indels, s0, e0 = extract_read_events(
                aln, dual_seqs[cname], active_trans
            )

            if nh == 1:
                if aln.mapping_quality < args.min_mapq:
                    continue
                update_pileup(pileups[cname], aligned)
                locus_counts[cname] += 1
                n_unique += 1
                write_single_molecule_record(
                    fsm, aln.query_name, cname, dual_seqs[cname], 1.0,
                    mpos, mlbl, s0, e0
                )
            else:
                multimappers[aln.query_name][hi].append(
                    (cname, aligned, n_indels, mpos, mlbl, s0, e0)
                )

        site_rates, log_priors = compute_empirical_priors(
            dual_seqs, pileups, locus_counts, args.prior_strength
        )
        n_resolved = 0
        for qname, hi_map in multimappers.items():
            n_multi_reads += 1
            cand_groups = []
            full_groups = []
            for hi_idx in sorted(hi_map):
                recs = hi_map[hi_idx]
                cand_groups.append([(r[0], r[1], r[2]) for r in recs])
                full_groups.append(recs)

            best_score_grp, post = resolve_multimapper(
                cand_groups, dual_seqs, site_rates, log_priors,
                args.min_posterior
            )
            if best_score_grp is None:
                continue
            best_idx = cand_groups.index(best_score_grp)
            chosen_recs = full_groups[best_idx]
            n_resolved += 1
            for cname, aligned, _, mpos, mlbl, s0, e0 in chosen_recs:
                update_pileup(pileups[cname], aligned)
                write_single_molecule_record(
                    fsm, qname, cname, dual_seqs[cname], post,
                    mpos, mlbl, s0, e0
                )

    ens_path = os.path.join(args.outdir, "ensemble_reactivity.tsv")
    export_ensemble_table(
        dual_seqs, pileups, active_trans, args.min_depth, ens_path
    )
    print(
        f"Called {n_unique} unique alignments and resolved "
        f"{n_resolved}/{n_multi_reads} multi-mapping templates "
        f"(posterior >= {args.min_posterior})."
    )
    print(f"Saved -> {ens_path} and {sm_path}")


if __name__ == "__main__":
    main()
