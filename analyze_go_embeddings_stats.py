import argparse
import os
import math
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from statsmodels.stats.multitest import multipletests


def _normalize_path(p: str) -> str:
    return p.replace("\\", "/")


def _is_vector(x) -> bool:
    if isinstance(x, np.ndarray) and x.ndim == 1:
        return True
    if isinstance(x, (list, tuple)) and len(x) > 3:
        return True
    return False


def _extract_vector(item):
    if _is_vector(item):
        return np.asarray(item, dtype=np.float32)
    if isinstance(item, (list, tuple)) and len(item) >= 1 and _is_vector(item[0]):
        return np.asarray(item[0], dtype=np.float32)
    raise ValueError("Unrecognized embedding element format")


def load_embeddings_dict(path: str, evalue_threshold: float = None) -> Dict[str, List[np.ndarray]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    result: Dict[str, List[np.ndarray]] = {}
    skipped_nonfinite = 0
    skipped_format = 0
    for k, v in data.items():
        vecs: List[np.ndarray] = []
        for it in v:
            try:
                if isinstance(it, (list, tuple)) and len(it) >= 2 and isinstance(it[1], (int, float)):
                    if evalue_threshold is not None and it[1] > evalue_threshold:
                        continue
                    vct = _extract_vector(it)
                else:
                    vct = _extract_vector(it)
            except Exception:
                skipped_format += 1
                continue
            if not np.all(np.isfinite(vct)):
                skipped_nonfinite += 1
                continue
            vecs.append(vct)
        if len(vecs) > 0:
            result[str(k)] = vecs
    if skipped_nonfinite > 0 or skipped_format > 0:
        print(f"[load] skipped_nonfinite={skipped_nonfinite}, skipped_format={skipped_format}")
    return result


def build_matrix(
    emb_dict: Dict[str, List[np.ndarray]],
    max_classes: int = None,
    max_per_class: int = None,
    min_per_class: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.RandomState(seed)
    items = list(emb_dict.items())
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    if max_classes is not None and max_classes > 0:
        items = items[:max_classes]
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    classes: List[str] = []
    cid = 0
    for name, vecs in items:
        if max_per_class is not None and max_per_class > 0 and len(vecs) > max_per_class:
            idx = rng.choice(len(vecs), size=max_per_class, replace=False)
            vecs_use = [vecs[i] for i in idx]
        else:
            vecs_use = vecs
        if len(vecs_use) < min_per_class:
            continue
        X_list.append(np.stack(vecs_use, axis=0))
        y_list.extend([cid] * len(vecs_use))
        classes.append(name)
        cid += 1
    if len(X_list) == 0:
        raise ValueError("No class has enough samples")
    X = np.concatenate(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int32)
    return X, y, classes


def compute_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        D = euclidean_distances(X)
    elif metric == "cosine":
        S = cosine_similarity(X)
        D = 1.0 - S
    else:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    D[D < 0] = 0.0
    return D


def run_permanova(D: np.ndarray, y: np.ndarray, permutations: int) -> Dict[str, float]:
    from skbio import DistanceMatrix
    from skbio.stats.distance import permanova

    dm = DistanceMatrix(D)
    res = permanova(dm, y, permutations=permutations)
    out = {
        "f_statistic": float(res["test statistic"]),
        "p_value": float(res["p-value"]),
        "permutations": int(permutations),
    }
    return out


def compute_eta_squared(X: np.ndarray, y: np.ndarray) -> float:
    overall = np.mean(X, axis=0)
    ss_total = float(np.sum((X - overall) ** 2))
    ss_between = 0.0
    for k in np.unique(y):
        Xg = X[y == k]
        mg = np.mean(Xg, axis=0)
        ss_between += Xg.shape[0] * float(np.sum((mg - overall) ** 2))
    if ss_total <= 0:
        return 0.0
    return float(ss_between / ss_total)


def run_anosim(D: np.ndarray, y: np.ndarray, permutations: int) -> Dict[str, float]:
    from skbio import DistanceMatrix
    from skbio.stats.distance import anosim

    dm = DistanceMatrix(D)
    res = anosim(dm, y, permutations=permutations)
    return {"r_statistic": float(res["test statistic"]), "p_value": float(res["p-value"])}


def summarize_per_class(X: np.ndarray, y: np.ndarray, classes: List[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(classes):
        Xi = X[y == i]
        if Xi.shape[0] < 2:
            continue
        intra = 1.0 - cosine_similarity(Xi)
        iu = np.triu_indices(Xi.shape[0], k=1)
        dvals = intra[iu]
        rows.append(
            {
                "go_term": name,
                "n_samples": int(Xi.shape[0]),
                "intra_cosine_mean": float(np.mean(dvals)),
                "intra_cosine_std": float(np.std(dvals)),
                "norm_mean": float(np.linalg.norm(Xi, axis=1).mean()),
            }
        )
    return pd.DataFrame(rows)


def pairwise_permanova(
    X: np.ndarray,
    y: np.ndarray,
    classes: List[str],
    metric: str,
    permutations: int,
    max_pairs: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            pairs.append((i, j))
    if max_pairs is not None and max_pairs > 0 and len(pairs) > max_pairs:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    rows = []
    for i, j in pairs:
        Xi = X[y == i]
        Xj = X[y == j]
        Xij = np.concatenate([Xi, Xj], axis=0)
        yy = np.array([0] * Xi.shape[0] + [1] * Xj.shape[0], dtype=np.int32)
        Dij = compute_distance_matrix(Xij, metric)
        try:
            res = run_permanova(Dij, yy, permutations)
            rows.append(
                {
                    "term1": classes[i],
                    "term2": classes[j],
                    "n1": int(Xi.shape[0]),
                    "n2": int(Xj.shape[0]),
                    "f_statistic": res["f_statistic"],
                    "p_value": res["p_value"],
                }
            )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["term1", "term2", "n1", "n2", "f_statistic", "p_value", "p_adj", "significant"])
    df = pd.DataFrame(rows)
    adj = multipletests(df["p_value"].values, method="fdr_bh")
    df["p_adj"] = adj[1]
    df["significant"] = df["p_adj"] < 0.05
    return df


def _filter_finite_and_min_samples(
    X: np.ndarray, y: np.ndarray, classes: List[str], min_per_class: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    finite_mask = np.isfinite(X).all(axis=1)
    if not finite_mask.all():
        dropped = int((~finite_mask).sum())
        print(f"[clean] drop {dropped} samples with NaN/Inf")
    X = X[finite_mask]
    y = y[finite_mask]
    keep_cls = []
    old_to_new = {}
    new_id = 0
    for i, name in enumerate(classes):
        cnt = int((y == i).sum())
        if cnt >= min_per_class:
            old_to_new[i] = new_id
            keep_cls.append(name)
            new_id += 1
        else:
            old_to_new[i] = -1
    keep_mask = np.array([old_to_new[int(yi)] >= 0 for yi in y], dtype=bool)
    if not keep_mask.all():
        print(f"[clean] drop {int((~keep_mask).sum())} samples from undersized classes")
    X = X[keep_mask]
    y = y[keep_mask]
    y_new = np.array([old_to_new[int(yi)] for yi in y], dtype=np.int32)
    return X, y_new, keep_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        "-i",
        type=str,
        default="data-bin/uniprotKB/cfpgen_general_dataset/train_go_terms_cls_emb.pkl",
    )
    ap.add_argument("--distance", "-d", choices=["euclidean", "cosine"], default="euclidean")
    ap.add_argument("--permutations", "-p", type=int, default=999)
    ap.add_argument("--max-classes", type=int, default=None)
    ap.add_argument("--max-samples-per-class", type=int, default=100)
    ap.add_argument("--min-samples-per-class", type=int, default=2)
    ap.add_argument("--evalue-threshold", type=float, default=None)
    ap.add_argument("--pairwise", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=5000)
    ap.add_argument("--output-prefix", type=str, default="go_embed_stats")
    args = ap.parse_args()

    in_path = _normalize_path(args.input)
    if not os.path.isabs(in_path):
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
        in_path = os.path.join(base, in_path)
    emb_dict = load_embeddings_dict(in_path, evalue_threshold=args.evalue_threshold)
    X, y, classes = build_matrix(
        emb_dict,
        max_classes=args.max_classes,
        max_per_class=args.max_samples_per_class,
        min_per_class=args.min_samples_per_class,
    )
    X, y, classes = _filter_finite_and_min_samples(X, y, classes, args.min_samples_per_class)
    D = compute_distance_matrix(X, args.distance)
    if not np.isfinite(D).all():
        raise ValueError("Distance matrix still contains NaN/Inf after cleaning")
    if not np.allclose(D, D.T):
        print("[warn] distance matrix not perfectly symmetric; symmetrizing")
        D = 0.5 * (D + D.T)
    perm = run_permanova(D, y, args.permutations)
    eta2 = compute_eta_squared(X, y)
    try:
        anos = run_anosim(D, y, args.permutations)
    except Exception:
        anos = {"r_statistic": math.nan, "p_value": math.nan}
    print("Global PERMANOVA:", perm)
    print("Global eta_squared:", eta2)
    print("Global ANOSIM:", anos)
    per_class_df = summarize_per_class(X, y, classes)
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
    per_class_path = os.path.join(out_dir, f"{args.output_prefix}_per_class.csv")
    per_class_df.to_csv(per_class_path, index=False)
    global_path = os.path.join(out_dir, f"{args.output_prefix}_global.csv")
    pd.DataFrame(
        [
            {
                "distance": args.distance,
                "permutations": args.permutations,
                "n_samples": int(X.shape[0]),
                "n_classes": int(len(classes)),
                "permanova_f": perm["f_statistic"],
                "permanova_p": perm["p_value"],
                "eta_squared": eta2,
                "anosim_r": anos["r_statistic"],
                "anosim_p": anos["p_value"],
            }
        ]
    ).to_csv(global_path, index=False)
    if args.pairwise:
        pair_df = pairwise_permanova(
            X, y, classes, metric=args.distance, permutations=args.permutations, max_pairs=args.max_pairs
        )
        pair_path = os.path.join(out_dir, f"{args.output_prefix}_pairwise.csv")
        pair_df.to_csv(pair_path, index=False)


if __name__ == "__main__":
    main()
