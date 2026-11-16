#!/usr/bin/env python3
"""
DECODE Reference Indexing: Dual-Strand 2-Base (G/T) Reference & STAR Index
==========================================================================

Overview
--------
Chemical deamination probing converts purines bidirectionally (A <-> G) and
pyrimidines unidirectionally (C -> T/U). Projecting sequences into a 2-base
purine/pyrimidine alphabet ({A, G} -> G; {C, T, U} -> T) eliminates transition
mismatch penalties during alignment.

Because the Watson-Crick reverse complement of {G, T} is {C, A} rather than
{G, T}, standard reverse-strand search in aligners cannot match 2-base reads
against antisense transcripts. This script solves that by explicitly building
a dual-strand reference containing both forward (`<contig>_fwd`) and reverse-
complemented (`<contig>_rev`) sequences oriented 5' -> 3' in G/T space.

Generated Index Artifacts (`--outdir`)
--------------------------------------
- `reference_2base_GT_dual.fa`  Dual-strand 2-base (G/T) FASTA indexed by STAR
- `reference_4base_dual.fa`     Dual-strand 4-base (A/C/G/T) FASTA used by
                                `decode_align.py` and `decode_call.py` for
                                4-base restoration and mutation calling
- `chrom.sizes`                 Contig lengths for all _fwd and _rev entries
- `annotation_dual.gtf`         Strand-projected GTF (if `--gtf` is provided)
- `STAR_index/`                 Compiled STAR genome/transcriptome index

Usage Examples
--------------
1. Target RNA or viral genome (FASTA only):
    python scripts/decode_index.py \
        --fasta test_data/hsrRNA.fa \
        --outdir decode_idx

2. Full genome or transcriptome with splice-junction annotation (FASTA + GTF):
    python scripts/decode_index.py \
        --fasta GRCh38.primary_assembly.genome.fa.gz \
        --gtf gencode.v33.primary_assembly.annotation.gtf.gz \
        --threads 16 \
        --outdir hg38_decode_idx
"""

import argparse
import gzip
import math
import os
import shutil
import subprocess
import sys

COMP_TAB = str.maketrans("ACGTUacgtuNn", "TGCAAtgcaaNn")
DNA_TAB = str.maketrans("ACGTUacgtuNn", "ACGTTacgttNn")
GT_TAB = str.maketrans("ACGTUacgtuNn", "GTGTGTGTGTNn")


def open_text(path, mode="rt"):
    """Open plain or gzipped text file."""
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def revcomp(seq):
    """Return 5'->3' reverse complement in standard DNA alphabet."""
    return seq.translate(COMP_TAB)[::-1]


def build_dual_references(fasta_path, out_dir):
    """Write dual-strand 4-base/2-base FASTAs and chrom.sizes."""
    fa_4b = os.path.join(out_dir, "reference_4base_dual.fa")
    fa_2b = os.path.join(out_dir, "reference_2base_GT_dual.fa")
    sizes_path = os.path.join(out_dir, "chrom.sizes")
    chrom_lens = {}
    curr, chunks = None, []

    def flush(name, parts, f4, f2):
        if not name:
            return
        fwd = "".join(parts).upper().translate(DNA_TAB)
        rev = revcomp(fwd)
        chrom_lens[f"{name}_fwd"] = len(fwd)
        chrom_lens[f"{name}_rev"] = len(rev)
        f4.write(f">{name}_fwd\n{fwd}\n>{name}_rev\n{rev}\n")
        f2.write(
            f">{name}_fwd\n{fwd.translate(GT_TAB)}\n"
            f">{name}_rev\n{rev.translate(GT_TAB)}\n"
        )

    with open_text(fasta_path) as fin, open(fa_4b, "w") as f4, open(
        fa_2b, "w"
    ) as f2:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush(curr, chunks, f4, f2)
                curr, chunks = line[1:].split()[0], []
            else:
                chunks.append(line)
        flush(curr, chunks, f4, f2)

    with open(sizes_path, "w") as fs:
        for cname, clen in chrom_lens.items():
            fs.write(f"{cname}\t{clen}\n")
    return chrom_lens, fa_4b, fa_2b


def project_gtf_dual_strand(gtf_path, chrom_lens, out_dir):
    """Map GTF features onto _fwd and _rev contigs in 5'->3' orientation."""
    dual_gtf = os.path.join(out_dir, "annotation_dual.gtf")
    kept = 0
    with open_text(gtf_path) as fin, open(dual_gtf, "w") as fout:
        for line in fin:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, src, feat, s_str, e_str, score, strand, frame, attrs = parts
            fwd_name = f"{chrom}_fwd"
            if fwd_name not in chrom_lens:
                continue
            clen = chrom_lens[fwd_name]
            start, end = int(s_str), int(e_str)
            if strand == "+":
                parts[0] = fwd_name
            elif strand == "-":
                parts[0] = f"{chrom}_rev"
                parts[3] = str(clen - end + 1)
                parts[4] = str(clen - start + 1)
                parts[6] = "+"
            else:
                continue
            fout.write("\t".join(parts) + "\n")
            kept += 1
    print(f"Projected {kept} GTF records -> {dual_gtf}")
    return dual_gtf


def run_star_index(
    star_bin, fa_2b, star_dir, chrom_lens, threads, dual_gtf, overhang
):
    """Build STAR genome index on the dual-strand 2-base (G/T) FASTA."""
    if shutil.which(star_bin) is None and not os.path.isfile(star_bin):
        sys.exit(f"ERROR: STAR binary not found: {star_bin}")
    os.makedirs(star_dir, exist_ok=True)
    total_bp = max(16, sum(chrom_lens.values()))
    n_contigs = max(1, len(chrom_lens))
    sa_nbases = max(2, min(14, int(math.log2(total_bp) / 2.0) - 1))
    bin_nbits = max(
        6, min(18, int(math.log2(max(64, total_bp // n_contigs))))
    )
    cmd = [
        star_bin,
        "--runMode", "genomeGenerate",
        "--runThreadN", str(threads),
        "--genomeDir", star_dir,
        "--genomeFastaFiles", fa_2b,
        "--genomeSAindexNbases", str(sa_nbases),
        "--genomeChrBinNbits", str(bin_nbits),
    ]
    if dual_gtf:
        cmd.extend([
            "--sjdbGTFfile", dual_gtf,
            "--sjdbOverhang", str(overhang),
        ])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Build dual-strand 2-base (G/T) reference & STAR index."
    )
    p.add_argument("--fasta", required=True, help="Input FASTA (.fa/.fa.gz)")
    p.add_argument("--gtf", default=None, help="Optional GTF (.gtf/.gtf.gz)")
    p.add_argument("--outdir", default="decode_idx", help="Output directory")
    p.add_argument("--threads", type=int, default=8, help="STAR threads")
    p.add_argument("--sjdb-overhang", type=int, default=100,
                   help="STAR sjdbOverhang (default: 100)")
    p.add_argument("--star-bin", default="STAR", help="Path to STAR binary")
    p.add_argument("--skip-star", action="store_true",
                   help="Only build dual-strand FASTAs and chrom.sizes")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    chrom_lens, _, fa_2b = build_dual_references(args.fasta, args.outdir)
    print(f"Built dual-strand references ({len(chrom_lens)} contigs).")
    dual_gtf = None
    if args.gtf:
        dual_gtf = project_gtf_dual_strand(args.gtf, chrom_lens, args.outdir)
    if not args.skip_star:
        star_dir = os.path.join(args.outdir, "STAR_index")
        run_star_index(
            args.star_bin, fa_2b, star_dir, chrom_lens,
            args.threads, dual_gtf, args.sjdb_overhang
        )
    print(f"Done. Index written to {args.outdir}")


if __name__ == "__main__":
    main()
