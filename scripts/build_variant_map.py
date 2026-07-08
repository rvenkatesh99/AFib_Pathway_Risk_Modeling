"""
Build variant_map.tsv for AlphaGenome validation.

Joins snp_pathway_final.tsv (variant→gene→pathway) with AF_GWAS_chunks_dosage.csv
(GWAS summary stats: beta, p-value, alleles) on the chr_pos key.

Output: variant_map.tsv with columns:
    chrom, pos, ref, alt, rsid, pathway, gwas_beta, gwas_p
"""
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snp_pathway", required=True,
                        help="Path to snp_pathway_final.tsv")
    parser.add_argument("--gwas", required=True,
                        help="Path to AF_GWAS_chunks_dosage.csv")
    parser.add_argument("--out", default="variant_map.tsv",
                        help="Output path for variant_map.tsv")
    args = parser.parse_args()

    gwas = pd.read_csv(args.gwas, sep="\t", dtype={"CHR": str, "POS": str})
    gwas["key"] = gwas["CHR"].astype(str) + "_" + gwas["POS"].astype(str)
    gwas = gwas.rename(columns={
        "CHR": "chrom",
        "POS": "pos",
        "Allele1": "ref",
        "Allele2": "alt",
        "MarkerID": "rsid",
        "BETA": "gwas_beta",
        "P": "gwas_p",
    })[["key", "chrom", "pos", "ref", "alt", "rsid", "gwas_beta", "gwas_p"]]

    sp = pd.read_csv(args.snp_pathway, sep="\t", dtype=str)
    sp = sp[["key", "pathway"]].drop_duplicates()

    merged = sp.merge(gwas, on="key", how="inner")
    merged = merged.drop(columns=["key"])

    col_order = ["chrom", "pos", "ref", "alt", "rsid", "pathway", "gwas_beta", "gwas_p"]
    merged = merged[col_order]

    merged.to_csv(args.out, sep="\t", index=False)
    print(f"Written {len(merged)} rows to {args.out}")
    print(f"  {merged['rsid'].nunique()} unique variants")
    print(f"  {merged['pathway'].nunique()} unique pathways")

if __name__ == "__main__":
    main()
