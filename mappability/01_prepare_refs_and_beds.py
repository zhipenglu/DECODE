#!/usr/bin/env python3
"""
DECODE Mappability Step 1: Dual-Strand References & Stratified Biotype BEDs
============================================================================

Overview
--------
Prepares dual-strand 4-base (A/C/G/T) and degenerate 2-base (G/T) genome
FASTAs alongside stratified 0-based BED annotations for exact (K, E)
mappability benchmarking with GenMap.

Because reverse-complementing {G, T} yields {C, A}, both forward (`<chr>_fwd`)
and reverse-complement (`<chr>_rev`) strands are written in 5' -> 3' G/T space,
and GTF annotations on the `-` strand are inverted onto `<chr>_rev`:
    bed_start = L - gtf_end
    bed_end   = L - gtf_start + 1

Generated Outputs (`--outdir`)
------------------------------
- `genome_4base_dual.fa`          Dual-strand 4-base reference FASTA
- `genome_2base_GT_dual.fa`       Dual-strand 2-base (G/T) reference FASTA
- `chrom.sizes`                   Contig lengths for `_fwd` and `_rev`
- `beds/whole_genome.bed`         Full dual-strand genomic intervals
- `beds/coding_exons.bed`         Merged protein-coding exon intervals
- `beds/nascent_introns.bed`      Protein-coding intronic intervals
- `beds/lncRNAs.bed`              Merged lncRNA exon intervals
- `beds/multicopy_small_ncRNAs.bed` Merged rRNA/tRNA/snRNA/snoRNA/miRNA loci

Usage Example
-------------
    python mappability/01_prepare_refs_and_beds.py \
        --genome GRCh38.primary_assembly.genome.fa.gz \
        --gtf gencode.v33.primary_assembly.annotation.gtf.gz \
        --outdir refs_dual_strand
"""

import argparse
import gzip
import os
import re
import sys
from collections import defaultdict

COMP_TAB = str.maketrans("ACGTUacgtuNn", "TGCAAtgcaaNn")
DNA_TAB = str.maketrans("ACGTUacgtuNn", "ACGTTacgttNn")
GT_TAB = str.maketrans("ACGTUacgtuNn", "GTGTTgtgttNn")

LNCRNA_TYPES = {
    "lncRNA",
    "lincRNA",
    "antisense",
    "processed_transcript",
    "sense_intronic",
    "sense_overlapping",
    "3prime_overlapping_ncRNA",
    "bidirectional_promoter_lncRNA",
    "macro_lncRNA",
}

SMALL_NCRNA_TYPES = {
    "rRNA",
    "rRNA_pseudogene",
    "tRNA",
    "Mt_rRNA",
    "Mt_tRNA",
    "snRNA",
    "snoRNA",
    "scaRNA",
    "miRNA",
    "scRNA",
    "vault_RNA",
    "Y_RNA",
    "sRNA",
    "ribozyme",
}

BIOTYPE_RE = re.compile(r'(?:gene_type|gene_biotype)\s+"([^"]+)"')


def open_text(path, mode="rt"):
    """Open a plain-text or gzip-compressed file transparently."""
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def revcomp(seq):
    """Return the 5'->3' reverse complement of a 4-base DNA/RNA sequence."""
    return seq.translate(COMP_TAB)[::-1]


def build_dual_fastas(genome_path, out_dir):
    """
    Write dual-strand 4-base and 2-base FASTAs and `chrom.sizes`.

    Returns:
        dict[str, int]: Mapping from dual contig name (`_fwd`/`_rev`) to bp.
    """
    if not os.path.exists(genome_path):
        sys.exit(f"ERROR: Genome FASTA not found: {genome_path}")

    fa_4b = os.path.join(out_dir, "genome_4base_dual.fa")
    fa_2b = os.path.join(out_dir, "genome_2base_GT_dual.fa")
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

    with open_text(genome_path) as fin, open(fa_4b, "w") as f4, open(
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
    return chrom_lens


def merge_intervals(iv_map):
    """Merge overlapping or adjacent [start, end) intervals per contig."""
    merged = {}
    for cname, ivs in iv_map.items():
        if not ivs:
            continue
        ivs_sorted = sorted(ivs)
        out = [list(ivs_sorted[0])]
        for s, e in ivs_sorted[1:]:
            if s <= out[-1][1]:
                if e > out[-1][1]:
                    out[-1][1] = e
            else:
                out.append([s, e])
        merged[cname] = [(s, e) for s, e in out if e > s]
    return merged


def subtract_intervals(base_map, mask_map):
    """Subtract merged `mask_map` intervals from `base_map` per contig."""
    result = {}
    for cname, b_list in base_map.items():
        m_list = mask_map.get(cname, [])
        if not m_list:
            result[cname] = list(b_list)
            continue
        out = []
        j = 0
        n_m = len(m_list)
        for bs, be in b_list:
            cur = bs
            while j < n_m and m_list[j][1] <= cur:
                j += 1
            k = j
            while k < n_m and m_list[k][0] < be:
                ms, me = m_list[k]
                if ms > cur:
                    out.append((cur, min(be, ms)))
                cur = max(cur, me)
                if cur >= be:
                    break
                k += 1
            if cur < be:
                out.append((cur, be))
        if out:
            result[cname] = out
    return result


def write_bed(iv_map, chrom_order, bed_path):
    """Write sorted 3-column 0-based BED file and return total base pairs."""
    total_bp = 0
    with open(bed_path, "w") as fout:
        for cname in chrom_order:
            for s, e in iv_map.get(cname, []):
                fout.write(f"{cname}\t{s}\t{e}\n")
                total_bp += e - s
    return total_bp


def build_biotype_beds(gtf_path, chrom_lens, out_dir):
    """
    Extract dual-strand 0-based BED tracks across 5 transcriptome strata.
    """
    if not os.path.exists(gtf_path):
        sys.exit(f"ERROR: GTF file not found: {gtf_path}")

    beds_dir = os.path.join(out_dir, "beds")
    os.makedirs(beds_dir, exist_ok=True)

    pc_genes = defaultdict(list)
    pc_exons = defaultdict(list)
    all_exons = defaultdict(list)
    lnc_exons = defaultdict(list)
    small_nc = defaultdict(list)

    with open_text(gtf_path) as fin:
        for line in fin:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feat, s_str, e_str, _, strand, _, attrs = parts
            if feat not in ("gene", "exon"):
                continue
            fwd_name = f"{chrom}_fwd"
            if fwd_name not in chrom_lens:
                continue

            m = BIOTYPE_RE.search(attrs)
            if not m:
                continue
            btype = m.group(1)

            clen = chrom_lens[fwd_name]
            s1, e1 = int(s_str), int(e_str)
            if strand == "+":
                d_chrom = fwd_name
                b_start, b_end = max(0, s1 - 1), min(clen, e1)
            elif strand == "-":
                d_chrom = f"{chrom}_rev"
                b_start = max(0, clen - e1)
                b_end = min(clen, clen - s1 + 1)
            else:
                continue

            if b_end <= b_start:
                continue

            if feat == "exon":
                all_exons[d_chrom].append((b_start, b_end))
                if btype == "protein_coding":
                    pc_exons[d_chrom].append((b_start, b_end))
                elif btype in LNCRNA_TYPES:
                    lnc_exons[d_chrom].append((b_start, b_end))
                elif btype in SMALL_NCRNA_TYPES:
                    small_nc[d_chrom].append((b_start, b_end))
            elif feat == "gene":
                if btype == "protein_coding":
                    pc_genes[d_chrom].append((b_start, b_end))
                elif btype in SMALL_NCRNA_TYPES:
                    small_nc[d_chrom].append((b_start, b_end))

    whole_map = {c: [(0, l)] for c, l in chrom_lens.items() if l > 0}
    coding_map = merge_intervals(pc_exons)
    intron_map = subtract_intervals(
        merge_intervals(pc_genes), merge_intervals(all_exons)
    )
    lnc_map = merge_intervals(lnc_exons)
    small_map = merge_intervals(small_nc)

    chrom_order = list(chrom_lens.keys())
    strata = [
        ("whole_genome", whole_map),
        ("coding_exons", coding_map),
        ("nascent_introns", intron_map),
        ("lncRNAs", lnc_map),
        ("multicopy_small_ncRNAs", small_map),
    ]
    for name, iv_map in strata:
        bpath = os.path.join(beds_dir, f"{name}.bed")
        bp = write_bed(iv_map, chrom_order, bpath)
        print(f"  {name:<24s} -> {bp:>14,d} bp ({bpath})")


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Prepare dual-strand FASTAs and stratified biotype BEDs."
    )
    p.add_argument("--genome", required=True, help="Genome FASTA (.fa/.gz)")
    p.add_argument("--gtf", required=True, help="Annotation GTF (.gtf/.gz)")
    p.add_argument(
        "--outdir", default="refs_dual_strand", help="Output directory"
    )
    return p.parse_args()


def main():
    """Build dual-strand references and stratified BED tracks."""
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print("Building dual-strand 4-base and 2-base (G/T) genome FASTAs...")
    chrom_lens = build_dual_fastas(args.genome, args.outdir)
    print(f"Wrote {len(chrom_lens)} dual-strand contigs -> {args.outdir}")
    print("Extracting dual-strand biotype BED intervals...")
    build_biotype_beds(args.gtf, chrom_lens, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
