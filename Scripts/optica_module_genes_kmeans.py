#!/usr/bin/env python3
"""
OptICA-faithful significant gene extraction (paper method).

Implements:
- For each independent component (each column of M.csv):
  1) take absolute values of gene weights
  2) run K-means clustering on |weights|
  3) mark genes in the top two clusters (clusters with largest centers) as significant
  4) output significant gene lists per component (iModulons)

Reference: OptICA paper, "Identifying significant genes in an independent component".
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# EPS = 1e-12  # Not needed anymore if log transformation isn't used


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--m_csv",
        required=True,
        help="Path to robust M.csv for a chosen dimension (genes x components).",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory to write significant gene results.",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        required=True,
        help=(
            "Number of K-means clusters to use on |gene weights| per component. "
            "Required because the OptICA paper does not specify K."
        ),
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for KMeans reproducibility (does not change OptICA method).",
    )
    p.add_argument(
        "--n-init",
        type=int,
        default=50,
        help="KMeans n_init (stability of KMeans solution).",
    )
    p.add_argument(
        "--component-prefix",
        default="IC",
        help="Prefix for component naming if M.csv columns are integers.",
    )
    return p.parse_args()


def ensure_component_names(cols, prefix: str):
    # If columns are numeric (0..k-1), name them IC0, IC1, ...
    new_cols = []
    for c in cols:
        try:
            int(c)
            new_cols.append(f"{prefix}{c}")
        except Exception:
            new_cols.append(str(c))
    return new_cols


def top_two_cluster_mask(abs_weights: np.ndarray, n_clusters: int, random_state: int, n_init: int):
    """
    abs_weights: shape (n_genes,)
    Return boolean mask (n_genes,) True if gene is in one of the top-two clusters by center magnitude.
    """
    # Changed: cluster on abs_weights
    # x = np.log10(abs_weights + EPS).reshape(-1, 1)
    x = abs_weights.reshape(-1, 1)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(x)

    centers = km.cluster_centers_.reshape(-1)  # shape (n_clusters,)
    # Identify indices of the two clusters with the largest centers
    top2 = np.argsort(centers)[-2:]  # last two = largest centers
    sig_mask = np.isin(labels, top2)

    return sig_mask, labels, centers, top2


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load robust M.csv (genes x components)
    M = pd.read_csv(args.m_csv, index_col=0)
    M.columns = ensure_component_names(M.columns, args.component_prefix)

    # Output tables
    # 1) long-form table for all genes/components with significance flag
    rows = []
    # 2) compact membership: one row per (component, gene) for significant genes
    sig_rows = []
    top_rows = []

    for comp in M.columns:
        w = M[comp].astype(float).values
        abs_w = np.abs(w)

        sig_mask, labels, centers, top2 = top_two_cluster_mask(
            abs_w, n_clusters=args.n_clusters, random_state=args.random_state, n_init=args.n_init
        )

        top2_sorted = list(np.array(top2)[np.argsort(centers[top2])])
        top_rows.append(
            {
                "component": comp,
                "top_cluster_1": int(top2_sorted[0]),
                "top_cluster_1_center_abs_weight": float(centers[top2_sorted[0]]),
                "top_cluster_2": int(top2_sorted[1]),
                "top_cluster_2_center_abs_weight": float(centers[top2_sorted[1]]),
            }
        )

        # record per-gene, per-component info
        for gene, weight, aw, lab, is_sig in zip(M.index, w, abs_w, labels, sig_mask):
            rows.append(
                {
                    "component": comp,
                    "gene": gene,
                    "weight": float(weight),
                    "abs_weight": float(aw),
                    "kmeans_label": int(lab),
                    "significant": bool(is_sig),
                    "weight_sign": "pos" if weight > 0 else ("neg" if weight < 0 else "zero"),
                }
            )
            if is_sig:
                sig_rows.append(
                    {
                        "component": comp,
                        "gene": gene,
                        "weight": float(weight),
                        "abs_weight": float(aw),
                        "weight_sign": "pos" if weight > 0 else ("neg" if weight < 0 else "zero"),
                    }
                )

        # write component-level gene list
        comp_sig_genes = M.index[sig_mask].tolist()
        (outdir / "per_component").mkdir(exist_ok=True)
        with open(outdir / "per_component" / f"{comp}.significant_genes.txt", "w") as f:
            for g in comp_sig_genes:
                f.write(f"{g}\n")

        # also write cluster centers for auditability
        centers_df = pd.DataFrame({"kmeans_cluster": np.arange(args.n_clusters), "center_abs_weight": centers})
        centers_df.sort_values("center_abs_weight", inplace=True)
        (outdir / "kmeans_centers").mkdir(exist_ok=True)
        centers_df.to_csv(outdir / "kmeans_centers" / f"{comp}.kmeans_centers.tsv", sep="\t", index=False)

    long_df = pd.DataFrame(rows)
    sig_df = pd.DataFrame(sig_rows)

    long_df.to_csv(outdir / "all_genes_all_components_with_significance.tsv", sep="\t", index=False)
    sig_df.to_csv(outdir / "significant_genes_long.tsv", sep="\t", index=False)

    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(outdir / "top_two_clusters_per_component.tsv", sep="\t", index=False)

    # Summary counts per component
    summary = (
        sig_df.groupby("component")
        .agg(
            n_significant=("gene", "count"),
            n_pos=("weight_sign", lambda s: int((s == "pos").sum())),
            n_neg=("weight_sign", lambda s: int((s == "neg").sum())),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "significant_gene_counts_per_component.tsv", sep="\t", index=False)

    print(f"[OK] Loaded M: {M.shape[0]} genes x {M.shape[1]} components from: {args.m_csv}")
    print(f"[OK] Wrote outputs to: {outdir.resolve()}")
    print("[OK] OptICA paper method used: K-means on (|weights|); top two clusters = significant.")
    print("[OK] Wrote top two clusters per component to: top_two_clusters_per_component.tsv")


if __name__ == "__main__":
    main()
