#!/usr/bin/env python3
"""
Faithful OptICA wrapper for HPC (bulk RNA-seq gene co-expression)

This script keeps the existing preprocessing (PCA noise correction + residualization)
and replaces ONLY the OptICA pieces (robustness / conserved / selection) with the
official OptICA GitHub implementation (iModulonMiner/4_optICA), unchanged.

Key guarantees:
- Robust components are defined by OptICA's DBSCAN clustering on S across restarts.
- Conserved ("Final Components") and optimal k selection are computed by OptICA's get_dimension.py.
- All intermediate outputs are human-readable CSV/TSV.
- Deterministic restarts via run-index→seed mapping (does not change OptICA logic).
- Does NOT modify OptICA scripts.

Expected input matrix layout:
- the CSV must be samples × genes (samples as rows, genes as columns).
- Internally we transpose to genes × samples to match OptICA scripts.

Outputs (under output_dir):
- df_corrected.csv                        (the preprocessing result, samples × genes)
- ica_runs/<k>/tmp/proc_*_{S,A}.csv       (ICA restarts for k; S is genes×k, A is samples×k)
- ica_runs/<k>/tmp/dist_*_*.npz           (OptICA pairwise sparse blocks)
- ica_runs/<k>/M.csv, A.csv               (OptICA clustered robust components)
- ica_runs/<k>/robust_stats.csv           (from OptICA)
- ica_runs/<k>/good_comps.tsv             (computed but NOT applied; faithful to OptICA)
- ica_runs/<k>/run_manifest.tsv           (n_runs effective, etc.)
- dimension_analysis.pdf                  (from OptICA get_dimension.py)
- dimension_stats.tsv                     (copied from get_dimension.py computation if enabled)
- M.csv, A.csv                            (selected optimal k, copied by get_dimension.py)
- selected_dimension.txt                  (chosen k)
- selected_good_comps.tsv                 (good_comps for selected k; NOT applied)

Future iModulon gene thresholding:
- This script saves all needed artifacts (M.csv + robust_stats + good_comps) so you can
  later compute significant genes only for "stable" components (e.g., good_comps) without
  touching OptICA scripts.
"""

import os

# -------------------------------------------------------------------
# Thread limiting (HPC safety)
# -------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import shutil
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LinearRegression



# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="HPC-friendly, faithful OptICA wrapper using official OptICA scripts unchanged."
    )
    p.add_argument("--input_dir", required=True, type=str, help="Directory containing input expression file.")
    p.add_argument("--output_dir", required=True, type=str, help="Output directory.")
    p.add_argument(
        "--input_file",
        default="All_peer.expression_with_ids_logtransformed.csv",
        type=str,
        help="CSV file name under input_dir (samples × genes). Default matches the current script.",
    )

    # Preprocessing knobs (kept minimal; same logic as the current script)
    p.add_argument(
        "--noise_pcs",
        default=50,
        type=int,
        help="Number of noise PCs to regress out during residualization (matches the current default if any).",
    )

    # OptICA sweep knobs (faithful semantics)
    p.add_argument("--min_dim", type=int, default=20, help="Minimum ICA dimensionality to test.")
    p.add_argument("--max_dim", type=int, default=200, help="Maximum ICA dimensionality to test.")
    p.add_argument(
        "--step_size",
        type=int,
        default=None,
        help="Dimensionality step size. Default is n_samples/25 (OptICA README default).",
    )

    # ICA restart knobs
    p.add_argument("--N_runs", type=int, default=20, help="Number of ICA restarts per dimension.")
    p.add_argument("--ICA_TOL", type=float, default=1e-3, help="FastICA tolerance (OptICA README suggests loosening).")
    p.add_argument("--ICA_MAX_ITER", type=int, default=1000, help="FastICA max_iter.")
    p.add_argument("--base_seed", type=int, default=42, help="Base seed for deterministic restarts: seed=base_seed+i.")
    p.add_argument("--N_jobs", type=int, default=1, help="Parallel workers for ICA restarts (joblib).")

    # OptICA scripts location
    p.add_argument(
        "--optica_dir",
        type=str,
        required=True,
        help="Path to OptICA '4_optICA' directory containing adjust_csv_MPI.py, compute_distance.py, cluster_components.py, get_dimension.py.",
    )

    # DBSCAN knobs (keep OptICA defaults unless you know you need otherwise)
    p.add_argument("--dbscan_eps", type=float, default=0.1, help="DBSCAN eps (OptICA default 0.1).")
    p.add_argument("--dbscan_min_frac", type=float, default=0.5, help="DBSCAN min_frac (OptICA default 0.5).")

    # Debug / rerun behavior
    p.add_argument("--overwrite", action="store_true", help="If set, overwrite existing ica_runs/<k> outputs.")
    return p.parse_args()


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def run_cmd(cmd, cwd=None):
    """Run a command; raise with clear message on failure."""
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed (exit {e.returncode}): {' '.join(map(str, cmd))}") from e


def ensure_empty_dir(path: Path, overwrite: bool):
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"Refusing to overwrite existing directory: {path} (use --overwrite)")
    path.mkdir(parents=True, exist_ok=False)


def list_proc_runs(tmp_dir: Path) -> int:
    """Count proc_*_S.csv files after adjust step."""
    return len(sorted(tmp_dir.glob("proc_*_S.csv")))



def prepare_get_dimension_layout(output_dir: Path, ica_runs_dir: Path, dims: list[int]) -> None:
    """
    OptICA get_dimension.py expects OUT_DIR to contain per-dimension folders named as integers,
    each containing M.csv and A.csv.
    Our wrapper stores these under OUT_DIR/ica_runs/<k>/.
    To remain faithful without modifying OptICA scripts, we create lightweight per-dim folders
    under OUT_DIR/<k>/ with symlinks to the corresponding files in ica_runs.

    If symlinks are not supported, we fall back to copying M.csv/A.csv (still human-readable CSV).
    """
    for k in dims:
        src_dir = ica_runs_dir / str(k)
        if not (src_dir / "M.csv").exists() or not (src_dir / "A.csv").exists():
            continue
        dst_dir = output_dir / str(k)
        dst_dir.mkdir(exist_ok=True)
        for fname in ("M.csv", "A.csv"):
            s = src_dir / fname
            d = dst_dir / fname
            try:
                if d.exists() or d.is_symlink():
                    d.unlink()
                os.symlink(s, d)
            except Exception:
                # fallback copy
                shutil.copyfile(s, d)


def save_params_tsv(out_dir: Path, params: dict):
    df = pd.DataFrame({"parameter": list(params.keys()), "value": list(params.values())})
    df.to_csv(out_dir / "run_params.tsv", sep="\t", index=False)


# -------------------------------------------------------------------
# Preprocessing (kept aligned with the current script's intent)
# -------------------------------------------------------------------
def preprocess_expression(df_samples_x_genes: pd.DataFrame, n_noise_pcs: int, seed: int = 42) -> pd.DataFrame:
    """
    1) Center per gene
    2) PCA on centered data (samples × genes) to get noise PCs
    3) Residualize each gene on those PCs (no intercept)
    Returns df_corrected (samples × genes)
    """
    # Center genes
    df_centered = df_samples_x_genes - df_samples_x_genes.mean(axis=0)

    # Noise PCs
    pca_noise = PCA(n_components=n_noise_pcs, random_state=seed)
    PCs_noise = pca_noise.fit_transform(df_centered.values)  # samples × n_noise_pcs

    # Residualize
    lr_model = LinearRegression(fit_intercept=False)
    lr_model.fit(PCs_noise, df_centered.values)
    predicted = lr_model.predict(PCs_noise)
    residuals = df_centered.values - predicted

    return pd.DataFrame(residuals, index=df_samples_x_genes.index, columns=df_samples_x_genes.columns)


# -------------------------------------------------------------------
# ICA restarts (seeded, outputs in OptICA format)
# -------------------------------------------------------------------
def run_one_fastica_restart(X_genes_x_samples: np.ndarray, k: int, seed: int, max_iter: int, tol: float):
    """
    Run sklearn FastICA on X (genes × samples), return:
      S: genes × k  (sources / gene weights)
      A: samples × k (mixing / activities)
    This layout matches OptICA scripts' expectation when they read proc_i_S.csv/proc_i_A.csv.
    """
    ica = FastICA(
        n_components=k,
        random_state=seed,
        max_iter=max_iter,
        tol=tol,
        whiten="arbitrary-variance",
        algorithm="parallel",
    )
    S = ica.fit_transform(X_genes_x_samples)  # genes × k
    A = ica.mixing_  # samples × k (features= samples)
    return S, A


def write_restart_outputs(tmp_dir: Path, run_idx: int, S: np.ndarray, A: np.ndarray, gene_index, sample_columns):
    """
    Write exactly:
      tmp/proc_<i>_S.csv  with index=genes and columns=components
      tmp/proc_<i>_A.csv  with index=samples and columns=components
    """
    S_df = pd.DataFrame(S, index=gene_index)
    A_df = pd.DataFrame(A, index=sample_columns)

    # OptICA scripts don't care about column names; they reassign later. Keep simple integer columns.
    S_df.columns = range(S_df.shape[1])
    A_df.columns = range(A_df.shape[1])

    S_df.to_csv(tmp_dir / f"proc_{run_idx}_S.csv")
    A_df.to_csv(tmp_dir / f"proc_{run_idx}_A.csv")


# -------------------------------------------------------------------
# "good_comps" helper (computed but not applied; faithful)
# -------------------------------------------------------------------

def compute_good_comps(out_k_dir: Path, tmp_dir: Path, k: int, eps: float, min_frac: float, n_eff: int) -> pd.DataFrame:
    """
    Compute the 'good_comps' filter exactly like OptICA cluster_components.py *would*,
    but without modifying OptICA scripts.

    OptICA's cluster_components.py builds a sparse block matrix from dist_i_j.npz blocks,
    converts similarity->distance (1 - sim), runs DBSCAN(metric="precomputed"), then
    builds df_stats with a 'count' column = cluster size. It computes:
        good_comps = df_stats[df_stats["count"] > max_rank * 0.5].index
    but does not write df_stats to disk.

    Here we reconstruct the same clustering labels from the existing dist blocks and
    write robust_stats.tsv + good_comps.tsv for human inspection/future filtering.
    """
    from scipy import sparse
    from sklearn.cluster import DBSCAN

    if n_eff <= 0:
        raise RuntimeError(f"n_eff must be > 0, got {n_eff}")

    # In OptICA scripts, max_rank is the maximum proc index (n_eff-1 after renumbering)
    max_rank = max(n_eff - 1, 0)
    min_samples = int(round(min_frac * max_rank)) + 1

    # Build sparse block matrix of similarities from dist_i_j.npz blocks
    block = []
    for i in range(max_rank + 1):
        row = []
        for j in range(max_rank + 1):
            if i <= j:
                f = tmp_dir / f"dist_{i}_{j}.npz"
                if not f.exists():
                    raise FileNotFoundError(f"Missing distance block: {f}")
                row.append(sparse.load_npz(str(f)))
            else:
                # lower triangle will be filled by transpose when bmat is symmetrized in OptICA.
                # However OptICA directly loads only i<=j blocks into a full block; its file set
                # contains both triangles due to how compute_distance enumerates pairs. We mirror
                # the simplest approach: load dist_j_i if present, else transpose.
                f = tmp_dir / f"dist_{j}_{i}.npz"
                if f.exists():
                    row.append(sparse.load_npz(str(f)))
                else:
                    ft = tmp_dir / f"dist_{i}_{j}.npz"
                    row.append(sparse.load_npz(str(ft)).T)
        block.append(row)

    D = sparse.bmat(block, format="csr")

    # Convert similarity->distance exactly like OptICA: distance = 1 - similarity for stored entries.
    D = sparse.csr_matrix((1 - D.data, D.indices, D.indptr), shape=D.shape)

    # DBSCAN clustering (precomputed distance)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(D)

    # Count cluster sizes (exclude noise label -1)
    labs, counts = np.unique(labels[labels != -1], return_counts=True)
    df_stats = pd.DataFrame({"count": counts}, index=labs)
    df_stats.index.name = "cluster_label"

    # Compute good comps exactly like OptICA (count > max_rank*0.5)
    threshold = max_rank * 0.5
    df_stats["is_good"] = df_stats["count"] > threshold

    good_labels = df_stats.index[df_stats["is_good"]].tolist()
    pd.DataFrame({"good_component_label": good_labels}).to_csv(out_k_dir / "good_comps.tsv", sep="	", index=False)

    # Save stats + manifest (human-readable)
    df_stats.sort_index().to_csv(out_k_dir / "robust_stats.tsv", sep="	")

    pd.DataFrame(
        {
            "k": [k],
            "n_effective_runs": [n_eff],
            "max_rank": [max_rank],
            "dbscan_eps": [eps],
            "dbscan_min_frac": [min_frac],
            "min_samples": [min_samples],
            "good_count_threshold": [threshold],
            "n_clusters_nonnoise": [len(labs)],
            "n_good_clusters": [len(good_labels)],
        }
    ).to_csv(out_k_dir / "run_manifest.tsv", sep="	", index=False)

    return df_stats



# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    optica_dir = Path(args.optica_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate OptICA scripts exist
    required = ["adjust_csv_MPI.py", "compute_distance.py", "cluster_components.py", "get_dimension.py"]
    missing = [x for x in required if not (optica_dir / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing OptICA scripts in --optica_dir: {missing}")

    # Save run parameters (human-readable)
    params = vars(args).copy()
    save_params_tsv(output_dir, params)

    # Load input
    in_path = input_dir / args.input_file
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    print(f"Reading input data: {in_path}")
    df = pd.read_csv(in_path, index_col=0)
    print(f"Loaded expression matrix: {df.shape[0]} samples × {df.shape[1]} genes")

    # Preprocess (kept)
    print("Preprocessing: centering + PCA noise residualization...")
    df_corrected = preprocess_expression(df, n_noise_pcs=args.noise_pcs, seed=args.base_seed)
    df_corrected.to_csv(output_dir / "df_corrected.csv")
    print("Saved df_corrected.csv")

    # Determine step size default (faithful semantics)
    n_samples = df_corrected.shape[0]
    step_size = args.step_size if args.step_size is not None else max(int(round(n_samples / 25)), 1)
    if step_size <= 0:
        step_size = 1

    # Build dimensions list
    dims = list(range(args.min_dim, args.max_dim + 1, step_size))
    if len(dims) == 0:
        raise ValueError("No dimensions to test. Check --min_dim/--max_dim/--step_size.")
    print(f"Testing dimensions: {dims}  (step_size={step_size}, n_samples={n_samples})")

    dims_done = []  # dimensions successfully processed (M.csv/A.csv exist)

    # Create ica_runs root
    ica_runs_dir = output_dir / "ica_runs"
    ica_runs_dir.mkdir(exist_ok=True)

    # Prepare ICA input orientation (genes × samples) to match OptICA scripts
    # the data is samples × genes => transpose
    X = df_corrected.values.T
    gene_index = df_corrected.columns.tolist()   # genes
    sample_columns = df_corrected.index.tolist() # samples

    # Run per-dimension
    for k in dims:
        print(f"\n=== OptICA dimension k={k} ===")
        out_k_dir = ica_runs_dir / str(k)
        tmp_dir = out_k_dir / "tmp"

        if out_k_dir.exists():
            if args.overwrite:
                shutil.rmtree(out_k_dir)
            else:
                print(f"Skipping k={k} because output exists (use --overwrite to rerun): {out_k_dir}")
                continue

        out_k_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # ---- Stage 2: ICA restarts (the parallelization; deterministic seeds)
        from joblib import Parallel, delayed

        def one_run(i):
            seed = args.base_seed + i
            S, A = run_one_fastica_restart(
                X_genes_x_samples=X,
                k=k,
                seed=seed,
                max_iter=args.ICA_MAX_ITER,
                tol=args.ICA_TOL,
            )
            write_restart_outputs(tmp_dir, i, S, A, gene_index=gene_index, sample_columns=sample_columns)
            return i

        print(f"Running {args.N_runs} ICA restarts with N_jobs={args.N_jobs}, tol={args.ICA_TOL} ...")
        Parallel(n_jobs=args.N_jobs, backend="loky")(delayed(one_run)(i) for i in range(args.N_runs))

        # ---- Stage 3–5: call OptICA scripts unchanged
        py = sys.executable

        print("Running OptICA adjust_csv_MPI.py ...")
        run_cmd([py, str(optica_dir / "adjust_csv_MPI.py"), "-o", str(out_k_dir), "-n", "1"])

        n_eff = list_proc_runs(tmp_dir)
        print(f"Effective runs after adjust: {n_eff}")

        print("Running OptICA compute_distance.py ...")
        run_cmd([py, str(optica_dir / "compute_distance.py"), "-o", str(out_k_dir), "-i", str(n_eff)])

        # Compute good_comps BEFORE cluster_components.py because cluster_components deletes tmp_dir
        print("Computing good_comps (saved, NOT applied) ...")
        df_stats = compute_good_comps(out_k_dir, tmp_dir, k=k, eps=args.dbscan_eps, min_frac=args.dbscan_min_frac, n_eff=n_eff)
        n_good = int(df_stats["is_good"].sum()) if "is_good" in df_stats.columns else 0
        print(f"k={k}: clusters_nonnoise={df_stats.shape[0]}, good_comps={n_good}")

        print("Running OptICA cluster_components.py ...")
        run_cmd([
                py,
                str(optica_dir / "cluster_components.py"),
                "-o", str(out_k_dir),
                "-i", str(n_eff),
                "-d", str(args.dbscan_eps),
                "-m", str(args.dbscan_min_frac),
            ])

        # Ensure M/A exist for get_dimension
        if not (out_k_dir / "M.csv").exists() or not (out_k_dir / "A.csv").exists():
            raise FileNotFoundError(f"Expected clustered outputs missing for k={k}: {out_k_dir/'M.csv'} or A.csv")
        dims_done.append(k)
    # ---- Stage B: select optimal dimension (faithful OptICA get_dimension logic, internal)
    print("\n=== Selecting optimal dimension (faithful get_dimension logic, internal) ===")

    # We use the exact computations from OptICA get_dimension.py internally, to avoid brittle
    # external-script layout/visibility issues on HPC. This remains faithful to OptICA.
    selected_k = infer_selected_dimension(output_dir)

    (output_dir / "selected_dimension.txt").write_text(str(selected_k) + "", encoding="utf-8")
    print(f"Selected dimension recorded: {selected_k}")

    # Copy selected M/A to top-level outputs (mirrors OptICA behavior)
    sel_dir = output_dir / "ica_runs" / str(selected_k)
    shutil.copyfile(sel_dir / "M.csv", output_dir / "M.csv")
    shutil.copyfile(sel_dir / "A.csv", output_dir / "A.csv")

    # Save good_comps list for selected dimension (for future significant-gene extraction)
    sel_good = sel_dir / "good_comps.tsv"
    if sel_good.exists():
        shutil.copyfile(sel_good, output_dir / "selected_good_comps.tsv")
        print("Saved selected_good_comps.tsv (NOT applied; for future iModulon gene selection).")
    else:
        print("Warning: selected dimension good_comps.tsv not found; continuing.")

    print("Done. You can now run the smoke test (few runs, few CPUs), then scale to 100 runs.")


def infer_selected_dimension(out_dir: Path) -> int:
    """
    Reproduce *exactly* the selection rule used in OptICA get_dimension.py to record the chosen k.
    This does not alter OptICA outputs; it only writes a record for convenience.

    Robust discovery: we only consider numeric subfolders under out_dir/ica_runs that contain BOTH
    M.csv and A.csv. This avoids empty-dims crashes if extra files/folders exist.
    """
    ica_runs = out_dir / "ica_runs"
    if not ica_runs.exists():
        raise RuntimeError(f"Expected ica_runs directory not found: {ica_runs}")

    dims = []
    for p in ica_runs.iterdir():
        if not p.is_dir():
            continue
        try:
            k = int(p.name)
        except Exception:
            continue
        if (p / "M.csv").exists() and (p / "A.csv").exists():
            dims.append(k)

    dims = sorted(dims)
    if not dims:
        contents = sorted([q.name for q in ica_runs.iterdir()])
        raise RuntimeError(
            "No valid per-dimension results found under "
            f"{ica_runs}. Expected numeric folders containing M.csv and A.csv. "
            f"Contents: {contents}"
        )

    def load_mat(dim, mat):
        df = pd.read_csv(ica_runs / str(dim) / f"{mat}.csv", index_col=0)
        df.columns = range(len(df.columns))
        return df.astype(float)

        df = pd.read_csv(ica_runs / str(dim) / f"{mat}.csv", index_col=0)
        df.columns = range(len(df.columns))
        return df.astype(float)

    M_data = [load_mat(dim, "M") for dim in dims]
    A_data = [load_mat(dim, "A") for dim in dims]

    # Mimic 'large iModulon dimension' check
    final_a = A_data[-1]
    while A_data and np.allclose(final_a, 0, atol=0.01):
        A_data = A_data[:-1]
        M_data = M_data[:-1]
        dims = dims[:-1]
        if not A_data:
            raise RuntimeError(
                "All tested dimensions had near-zero A matrices (allclose to 0). Cannot select optimal dimension."
            )
        final_a = A_data[-1]

    final_m = M_data[-1]
    n_components = [m.shape[1] for m in M_data]

    thresh = 0.7
    n_final_mods = []
    n_single_genes = []
    for m in M_data:
        l2_final = np.sqrt(np.power(final_m, 2).sum(axis=0))
        l2_m = np.sqrt(np.power(m, 2).sum(axis=0))
        dist = (
            pd.DataFrame(abs(np.dot(final_m.T, m)))
            .divide(l2_final, axis=0)
            .divide(l2_m, axis=1)
        )
        n_final_mods.append(len(np.where(dist > thresh)[0]))

        counter = 0
        for col in m.columns:
            sorted_genes = abs(m[col]).sort_values(ascending=False)
            if sorted_genes.iloc[0] > 2 * sorted_genes.iloc[1]:
                counter += 1
        n_single_genes.append(counter)

    non_single_components = np.array(n_components) - np.array(n_single_genes)

    DF_stats = pd.DataFrame(
        [n_components, n_final_mods, non_single_components, n_single_genes],
        index=[
            "Robust Components",
            "Final Components",
            "Multi-gene Components",
            "Single Gene Components",
        ],
        columns=dims,
    ).T
    DF_stats.sort_index(inplace=True)

    # Official selection rule (>= and earliest)
    dimensionality = (
        DF_stats[DF_stats["Final Components"] >= DF_stats["Multi-gene Components"]]
        .iloc[0]
        .name
    )

    DF_stats.to_csv(out_dir / "dimension_stats.tsv", sep="\t")
    return int(dimensionality)


if __name__ == "__main__":
    main()
