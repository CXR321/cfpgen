import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
 
 
@dataclass(frozen=True)
class ProteinMetrics:
    record_id: str
    plddt: Optional[float]
    ptm: Optional[float]
    pdb_path: Path
 
 
_SEQ_ID_RE = re.compile(r"SEQUENCE_ID=([^_]+)_L=")
_PLDDT_RE = re.compile(r"(?:^|_)plddt_([0-9]+(?:\.[0-9]+)?)")
_PTM_RE = re.compile(r"(?:^|_)ptm_([0-9]+(?:\.[0-9]+)?)")
_PTM_IN_PDB_RE = re.compile(
    r"(?:\bptm\b|\bpTM\b|predicted\s+tm-score|predicted\s+tm\s+score)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
 
 
def _normalize_path_str(path_str: str) -> str:
    return path_str.strip().strip('"').strip("'").replace("\\", "/")
 
 
def _parse_thresholds(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        out.append(float(p))
    return out
 
 
def _extract_uniprot_id(filename: str) -> Optional[str]:
    m = _SEQ_ID_RE.search(filename)
    if not m:
        return None
    return m.group(1)
 
 
def _extract_sample_id(filename: str) -> str:
    stem = filename
    if stem.endswith(".pdb"):
        stem = stem[:-4]
 
    if "_plddt_" in stem:
        return stem.split("_plddt_", 1)[0]
    if "_ptm_" in stem:
        return stem.split("_ptm_", 1)[0]
    return stem
 
 
def _extract_record_id(filename: str, id_mode: str) -> str:
    if id_mode == "filename":
        return filename
    if id_mode == "sample":
        return _extract_sample_id(filename)
    if id_mode == "uniprot":
        uniprot_id = _extract_uniprot_id(filename)
        if uniprot_id:
            return uniprot_id
        return _extract_sample_id(filename)
    raise ValueError(f"Unknown id_mode: {id_mode}")
 
def _extract_plddt_ptm_from_name(filename: str) -> Tuple[Optional[float], Optional[float]]:
    plddt_m = _PLDDT_RE.search(filename)
    ptm_m = _PTM_RE.search(filename)
    plddt = float(plddt_m.group(1)) if plddt_m else None
    ptm = float(ptm_m.group(1)) if ptm_m else None
    return plddt, ptm
 
 
def _iter_pdb_files(pdb_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from pdb_dir.rglob("*.pdb")
    else:
        yield from pdb_dir.glob("*.pdb")
 
 
def _parse_ptm_from_pdb(pdb_path: Path) -> Optional[float]:
    try:
        with pdb_path.open("r", errors="ignore") as f:
            for line in f:
                if line.startswith("ATOM"):
                    return None
                if not line.startswith("REMARK"):
                    continue
                m = _PTM_IN_PDB_RE.search(line)
                if m:
                    return float(m.group(1))
    except OSError:
        return None
    return None
 
 
def _parse_mean_plddt_from_pdb(pdb_path: Path, mode: str) -> Optional[float]:
    total = 0.0
    n = 0
    try:
        with pdb_path.open("r", errors="ignore") as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                if len(line) < 66:
                    continue
                if mode == "ca":
                    atom_name = line[12:16].strip()
                    if atom_name != "CA":
                        continue
                    alt_loc = line[16:17]
                    if alt_loc not in (" ", "A", ""):
                        continue
                try:
                    b = float(line[60:66])
                except ValueError:
                    continue
                total += b
                n += 1
    except OSError:
        return None
 
    if n == 0:
        return None
    return total / n
 
 
def load_metrics(
    pdb_dir: Path,
    recursive: bool,
    plddt_mode: str,
    id_mode: str,
) -> Tuple[Dict[str, ProteinMetrics], List[str]]:
    by_id: Dict[str, ProteinMetrics] = {}
    parse_errors: List[str] = []
 
    for pdb_path in _iter_pdb_files(pdb_dir, recursive=recursive):
        fname = pdb_path.name
        record_id = _extract_record_id(fname, id_mode=id_mode)
        plddt, ptm = _extract_plddt_ptm_from_name(fname)
        if ptm is None:
            ptm = _parse_ptm_from_pdb(pdb_path)
        if plddt is None:
            plddt = _parse_mean_plddt_from_pdb(pdb_path, mode=plddt_mode)
 
        if record_id in by_id:
            existing = by_id[record_id]
            a = (
                existing.ptm if existing.ptm is not None else -1.0,
                existing.plddt if existing.plddt is not None else -1.0,
            )
            b = (ptm if ptm is not None else -1.0, plddt if plddt is not None else -1.0)
            if b > a:
                by_id[record_id] = ProteinMetrics(
                    record_id=record_id, plddt=plddt, ptm=ptm, pdb_path=pdb_path
                )
            continue
 
        by_id[record_id] = ProteinMetrics(
            record_id=record_id,
            plddt=plddt,
            ptm=ptm,
            pdb_path=pdb_path,
        )
 
        if plddt is None and ptm is None:
            parse_errors.append(str(pdb_path))
 
    return by_id, parse_errors
 
 
def _rate(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v) / len(values)
 
 
def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"
 
 
def summarize_bad_rates(
    metrics: Sequence[ProteinMetrics],
    ptm_thresholds: Sequence[float],
    plddt_thresholds: Sequence[float],
) -> dict:
    ptm_values = [m.ptm for m in metrics if m.ptm is not None]
    plddt_values = [m.plddt for m in metrics if m.plddt is not None]
 
    ptm_bad = {t: _rate([v < t for v in ptm_values]) for t in ptm_thresholds}
    plddt_bad = {t: _rate([v < t for v in plddt_values]) for t in plddt_thresholds}
 
    joint: Dict[Tuple[float, float], dict] = {}
    for ptm_t in ptm_thresholds:
        for plddt_t in plddt_thresholds:
            pairs = [(m.ptm, m.plddt) for m in metrics if m.ptm is not None and m.plddt is not None]
            both = _rate([ptm < ptm_t and plddt < plddt_t for ptm, plddt in pairs])
            either = _rate([ptm < ptm_t or plddt < plddt_t for ptm, plddt in pairs])
            joint[(ptm_t, plddt_t)] = {"both": both, "either": either, "n": len(pairs)}
 
    return {
        "n_total": len(metrics),
        "n_with_ptm": len(ptm_values),
        "n_with_plddt": len(plddt_values),
        "ptm_bad": ptm_bad,
        "plddt_bad": plddt_bad,
        "joint": joint,
    }
 
 
def _print_summary(label: str, s: dict, ptm_thresholds: Sequence[float], plddt_thresholds: Sequence[float]) -> None:
    print(f"[{label}]")
    print(f"  N_total      : {s['n_total']}")
    print(f"  N_with_pTM   : {s['n_with_ptm']}")
    print(f"  N_with_pLDDT : {s['n_with_plddt']}")
    if ptm_thresholds:
        for t in ptm_thresholds:
            print(f"  pTM < {t:g}  : {_fmt_pct(s['ptm_bad'][t])}")
    if plddt_thresholds:
        for t in plddt_thresholds:
            print(f"  pLDDT < {t:g}: {_fmt_pct(s['plddt_bad'][t])}")
 
 
def write_per_protein_csv(
    out_csv: Path, label_a: str, label_b: str, a: Dict[str, ProteinMetrics], b: Dict[str, ProteinMetrics]
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    common = sorted(set(a.keys()) & set(b.keys()))
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "record_id",
                f"plddt_{label_a}",
                f"ptm_{label_a}",
                f"plddt_{label_b}",
                f"ptm_{label_b}",
                "delta_plddt(b-a)",
                "delta_ptm(b-a)",
                f"pdb_{label_a}",
                f"pdb_{label_b}",
            ]
        )
        for pid in common:
            ma = a[pid]
            mb = b[pid]
            da = mb.plddt - ma.plddt if (mb.plddt is not None and ma.plddt is not None) else ""
            db = mb.ptm - ma.ptm if (mb.ptm is not None and ma.ptm is not None) else ""
            w.writerow(
                [
                    pid,
                    "" if ma.plddt is None else f"{ma.plddt:.4f}",
                    "" if ma.ptm is None else f"{ma.ptm:.4f}",
                    "" if mb.plddt is None else f"{mb.plddt:.4f}",
                    "" if mb.ptm is None else f"{mb.ptm:.4f}",
                    da,
                    db,
                    str(ma.pdb_path),
                    str(mb.pdb_path),
                ]
            )
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True, help="baseline pdb folder (e.g. cfpgen)")
    ap.add_argument("--codefp-dir", required=True, help="our pdb folder (e.g. CodeFP)")
    ap.add_argument("--label-baseline", default="baseline")
    ap.add_argument("--label-codefp", default="codefp")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--plddt-mode", choices=["ca", "all"], default="ca")
    ap.add_argument(
        "--id-mode",
        choices=["sample", "uniprot", "filename"],
        default="sample",
        help="How to group multiple PDBs: sample keeps each generated sequence (recommended for 1400); uniprot groups best-of-N per UniProt; filename groups by raw filename",
    )
    ap.add_argument("--ptm-thresholds", type=str, default="0.2,0.3,0.4")
    ap.add_argument("--plddt-thresholds", type=str, default="30,50,70")
    ap.add_argument("--out-csv", default=None, help="optional per-protein csv on common subset")
    args = ap.parse_args()
 
    baseline_dir = Path(_normalize_path_str(args.baseline_dir))
    codefp_dir = Path(_normalize_path_str(args.codefp_dir))
    if not baseline_dir.exists():
        raise FileNotFoundError(f"baseline-dir not found: {baseline_dir}")
    if not codefp_dir.exists():
        raise FileNotFoundError(f"codefp-dir not found: {codefp_dir}")
 
    ptm_thresholds = _parse_thresholds(args.ptm_thresholds)
    plddt_thresholds = _parse_thresholds(args.plddt_thresholds)
 
    a_by_id, a_parse_errors = load_metrics(
        baseline_dir, recursive=args.recursive, plddt_mode=args.plddt_mode, id_mode=args.id_mode
    )
    b_by_id, b_parse_errors = load_metrics(
        codefp_dir, recursive=args.recursive, plddt_mode=args.plddt_mode, id_mode=args.id_mode
    )
 
    common_ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
    a_only = sorted(set(a_by_id.keys()) - set(b_by_id.keys()))
    b_only = sorted(set(b_by_id.keys()) - set(a_by_id.keys()))
 
    a_all = list(a_by_id.values())
    b_all = list(b_by_id.values())
    a_common = [a_by_id[i] for i in common_ids]
    b_common = [b_by_id[i] for i in common_ids]
 
    print("=" * 80)
    print("ESMFold pTM / pLDDT: bad-case ratio comparison")
    print("=" * 80)
    print(f"baseline_dir: {baseline_dir} ({args.label_baseline})")
    print(f"codefp_dir  : {codefp_dir} ({args.label_codefp})")
    print("-" * 80)
    print(f"N_parsed_{args.label_baseline}: {len(a_by_id)}")
    print(f"N_parsed_{args.label_codefp}  : {len(b_by_id)}")
    print(f"N_common                : {len(common_ids)}")
    print(f"only_in_{args.label_baseline}: {len(a_only)}")
    print(f"only_in_{args.label_codefp}  : {len(b_only)}")
    if a_parse_errors:
        print(f"parse_errors_{args.label_baseline}: {len(a_parse_errors)}")
    if b_parse_errors:
        print(f"parse_errors_{args.label_codefp}  : {len(b_parse_errors)}")
    print("-" * 80)
    print(
        f"bad thresholds: pTM < {ptm_thresholds}, pLDDT < {plddt_thresholds} (pLDDT mode: {args.plddt_mode}, id_mode: {args.id_mode})"
    )
    print("-" * 80)
 
    a_all_s = summarize_bad_rates(a_all, ptm_thresholds=ptm_thresholds, plddt_thresholds=plddt_thresholds)
    b_all_s = summarize_bad_rates(b_all, ptm_thresholds=ptm_thresholds, plddt_thresholds=plddt_thresholds)
    _print_summary(f"{args.label_baseline} (all)", a_all_s, ptm_thresholds, plddt_thresholds)
    _print_summary(f"{args.label_codefp} (all)", b_all_s, ptm_thresholds, plddt_thresholds)
 
    print("-" * 80)
    a_common_s = summarize_bad_rates(a_common, ptm_thresholds=ptm_thresholds, plddt_thresholds=plddt_thresholds)
    b_common_s = summarize_bad_rates(b_common, ptm_thresholds=ptm_thresholds, plddt_thresholds=plddt_thresholds)
    _print_summary(f"{args.label_baseline} (common)", a_common_s, ptm_thresholds, plddt_thresholds)
    _print_summary(f"{args.label_codefp} (common)", b_common_s, ptm_thresholds, plddt_thresholds)
 
    pairs_n = next(iter(a_common_s["joint"].values()))["n"] if a_common_s["joint"] else 0
    if pairs_n:
        print("-" * 80)
        print("Common subset: joint bad rates (computed on proteins with both pTM and pLDDT)")
        for ptm_t in ptm_thresholds:
            for plddt_t in plddt_thresholds:
                a_j = a_common_s["joint"][(ptm_t, plddt_t)]
                b_j = b_common_s["joint"][(ptm_t, plddt_t)]
                print(
                    f"  (pTM<{ptm_t:g}, pLDDT<{plddt_t:g}) both: {args.label_baseline}={_fmt_pct(a_j['both'])}, {args.label_codefp}={_fmt_pct(b_j['both'])} | "
                    f"either: {args.label_baseline}={_fmt_pct(a_j['either'])}, {args.label_codefp}={_fmt_pct(b_j['either'])}"
                )
 
    if args.out_csv:
        out_csv = Path(_normalize_path_str(args.out_csv))
        write_per_protein_csv(out_csv, args.label_baseline, args.label_codefp, a_by_id, b_by_id)
        print("-" * 80)
        print(f"wrote_csv: {out_csv}")
 
 
if __name__ == "__main__":
    main()
