import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
 
 
@dataclass(frozen=True)
class ESMFoldMetrics:
    uniprot_id: str
    plddt: float
    ptm: float
    pdb_path: Path
 
 
_PLDDT_RE = re.compile(r"plddt_([0-9]+(?:\.[0-9]+)?)")
_PTM_RE = re.compile(r"ptm_([0-9]+(?:\.[0-9]+)?)\.pdb$")
_SEQ_ID_RE = re.compile(r"SEQUENCE_ID=([^_]+)_L=")
 
 
def _normalize_path_str(path_str: str) -> str:
    return path_str.strip().strip('"').strip("'").replace("\\", "/")
 
 
def _extract_uniprot_id(filename: str) -> Optional[str]:
    m = _SEQ_ID_RE.search(filename)
    if m:
        return m.group(1)
    if "SEQUENCE_ID=" in filename:
        tail = filename.split("SEQUENCE_ID=", 1)[1]
        if "_L=" in tail:
            return tail.split("_L=", 1)[0]
        return tail.split("_", 1)[0]
    return None
 
 
def _extract_plddt_ptm(filename: str) -> Tuple[Optional[float], Optional[float]]:
    plddt_m = _PLDDT_RE.search(filename)
    ptm_m = _PTM_RE.search(filename)
    plddt = float(plddt_m.group(1)) if plddt_m else None
    ptm = float(ptm_m.group(1)) if ptm_m else None
    return plddt, ptm
 
 
def _iter_pdb_files(pdb_dir: Path) -> Iterable[Path]:
    yield from pdb_dir.glob("*.pdb")
 
 
def build_metrics_index(pdb_dir: Path) -> Tuple[Dict[str, ESMFoldMetrics], List[str], List[str]]:
    by_id: Dict[str, ESMFoldMetrics] = {}
    parse_errors: List[str] = []
    duplicates: List[str] = []
 
    for pdb_path in _iter_pdb_files(pdb_dir):
        fname = pdb_path.name
        uniprot_id = _extract_uniprot_id(fname)
        plddt, ptm = _extract_plddt_ptm(fname)
        if not uniprot_id or plddt is None or ptm is None:
            parse_errors.append(str(pdb_path))
            continue
 
        metrics = ESMFoldMetrics(uniprot_id=uniprot_id, plddt=plddt, ptm=ptm, pdb_path=pdb_path)
        if uniprot_id in by_id:
            duplicates.append(uniprot_id)
            existing = by_id[uniprot_id]
            if (metrics.plddt, metrics.ptm, str(metrics.pdb_path)) > (
                existing.plddt,
                existing.ptm,
                str(existing.pdb_path),
            ):
                by_id[uniprot_id] = metrics
        else:
            by_id[uniprot_id] = metrics
 
    return by_id, parse_errors, duplicates
 
 
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
 
 
def summarize(
    metrics: Sequence[ESMFoldMetrics],
    plddt_threshold: float,
    ptm_threshold: float,
) -> dict:
    plddt_pass = 0
    ptm_pass = 0
    both_pass = 0
    plddts: List[float] = []
    ptms: List[float] = []
 
    for m in metrics:
        plddts.append(m.plddt)
        ptms.append(m.ptm)
        ok_plddt = m.plddt > plddt_threshold
        ok_ptm = m.ptm > ptm_threshold
        plddt_pass += int(ok_plddt)
        ptm_pass += int(ok_ptm)
        both_pass += int(ok_plddt and ok_ptm)
 
    n = len(metrics)
    return {
        "n": n,
        "plddt_pass": plddt_pass,
        "ptm_pass": ptm_pass,
        "both_pass": both_pass,
        "plddt_rate": (plddt_pass / n) if n else 0.0,
        "ptm_rate": (ptm_pass / n) if n else 0.0,
        "both_rate": (both_pass / n) if n else 0.0,
        "plddt_mean": _mean(plddts),
        "ptm_mean": _mean(ptms),
    }
 
 
def write_csv(
    out_csv: Path,
    common_ids: Sequence[str],
    a: Dict[str, ESMFoldMetrics],
    b: Dict[str, ESMFoldMetrics],
    plddt_threshold: float,
    ptm_threshold: float,
    a_label: str,
    b_label: str,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "uniprot_id",
                f"plddt_{a_label}",
                f"ptm_{a_label}",
                f"plddt_{b_label}",
                f"ptm_{b_label}",
                "delta_plddt(b-a)",
                "delta_ptm(b-a)",
                f"pass_plddt_{a_label}",
                f"pass_ptm_{a_label}",
                f"pass_both_{a_label}",
                f"pass_plddt_{b_label}",
                f"pass_ptm_{b_label}",
                f"pass_both_{b_label}",
                f"pdb_{a_label}",
                f"pdb_{b_label}",
            ]
        )
        for uid in common_ids:
            ma = a[uid]
            mb = b[uid]
            a_plddt_ok = ma.plddt > plddt_threshold
            a_ptm_ok = ma.ptm > ptm_threshold
            b_plddt_ok = mb.plddt > plddt_threshold
            b_ptm_ok = mb.ptm > ptm_threshold
            w.writerow(
                [
                    uid,
                    ma.plddt,
                    ma.ptm,
                    mb.plddt,
                    mb.ptm,
                    mb.plddt - ma.plddt,
                    mb.ptm - ma.ptm,
                    int(a_plddt_ok),
                    int(a_ptm_ok),
                    int(a_plddt_ok and a_ptm_ok),
                    int(b_plddt_ok),
                    int(b_ptm_ok),
                    int(b_plddt_ok and b_ptm_ok),
                    str(ma.pdb_path),
                    str(mb.pdb_path),
                ]
            )
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True)
    ap.add_argument("--dir-b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--plddt-threshold", type=float, default=70.0)
    ap.add_argument("--ptm-threshold", type=float, default=0.5)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
 
    dir_a = Path(_normalize_path_str(args.dir_a))
    dir_b = Path(_normalize_path_str(args.dir_b))
 
    if not dir_a.exists():
        raise FileNotFoundError(f"dir-a not found: {dir_a}")
    if not dir_b.exists():
        raise FileNotFoundError(f"dir-b not found: {dir_b}")
 
    a_by_id, a_parse_errors, a_duplicates = build_metrics_index(dir_a)
    b_by_id, b_parse_errors, b_duplicates = build_metrics_index(dir_b)
 
    common_ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
    a_only = sorted(set(a_by_id.keys()) - set(b_by_id.keys()))
    b_only = sorted(set(b_by_id.keys()) - set(a_by_id.keys()))
 
    common_a = [a_by_id[uid] for uid in common_ids]
    common_b = [b_by_id[uid] for uid in common_ids]
 
    a_stats = summarize(common_a, plddt_threshold=args.plddt_threshold, ptm_threshold=args.ptm_threshold)
    b_stats = summarize(common_b, plddt_threshold=args.plddt_threshold, ptm_threshold=args.ptm_threshold)
 
    deltas_plddt = [b_by_id[uid].plddt - a_by_id[uid].plddt for uid in common_ids]
    deltas_ptm = [b_by_id[uid].ptm - a_by_id[uid].ptm for uid in common_ids]
 
    print("=" * 80)
    print("ESMFold pLDDT / pTM Comparison (Common UniProt ID Subset)")
    print("=" * 80)
    print(f"dir_a: {dir_a} ({args.label_a})")
    print(f"dir_b: {dir_b} ({args.label_b})")
    print("-" * 80)
    print(f"files_parsed_{args.label_a}: {len(a_by_id)}")
    print(f"files_parsed_{args.label_b}: {len(b_by_id)}")
    print(f"common_uniprot_ids: {len(common_ids)}")
    print(f"only_in_{args.label_a}: {len(a_only)}")
    print(f"only_in_{args.label_b}: {len(b_only)}")
    if a_parse_errors:
        print(f"parse_errors_{args.label_a}: {len(a_parse_errors)}")
    if b_parse_errors:
        print(f"parse_errors_{args.label_b}: {len(b_parse_errors)}")
    if a_duplicates:
        print(f"duplicate_ids_{args.label_a}: {len(set(a_duplicates))}")
    if b_duplicates:
        print(f"duplicate_ids_{args.label_b}: {len(set(b_duplicates))}")
    print("-" * 80)
    print(f"thresholds: pLDDT > {args.plddt_threshold}, pTM > {args.ptm_threshold}")
    print("-" * 80)
 
    def _print_block(label: str, s: dict) -> None:
        n = s["n"]
        print(f"[{label}] N={n}")
        print(f"  mean_pLDDT: {s['plddt_mean']:.4f}")
        print(f"  mean_pTM  : {s['ptm_mean']:.4f}")
        print(f"  pLDDT_pass: {s['plddt_pass']}/{n} ({s['plddt_rate']:.2%})")
        print(f"  pTM_pass  : {s['ptm_pass']}/{n} ({s['ptm_rate']:.2%})")
        print(f"  both_pass : {s['both_pass']}/{n} ({s['both_rate']:.2%})")
 
    _print_block(args.label_a, a_stats)
    _print_block(args.label_b, b_stats)
 
    if common_ids:
        print("-" * 80)
        print("Delta (dir_b - dir_a) on common subset")
        print(f"  mean_delta_pLDDT: {_mean(deltas_plddt):.4f}")
        print(f"  mean_delta_pTM  : {_mean(deltas_ptm):.4f}")
 
    if args.out_csv:
        out_csv = Path(_normalize_path_str(args.out_csv))
        write_csv(
            out_csv=out_csv,
            common_ids=common_ids,
            a=a_by_id,
            b=b_by_id,
            plddt_threshold=args.plddt_threshold,
            ptm_threshold=args.ptm_threshold,
            a_label=args.label_a,
            b_label=args.label_b,
        )
        print("-" * 80)
        print(f"wrote_csv: {out_csv}")
 
 
if __name__ == "__main__":
    main()
