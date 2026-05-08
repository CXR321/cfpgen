import argparse
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: torch. Install PyTorch in your environment, then rerun."
    ) from e

from byprot.models.lm.dplm2 import (
    MultimodalDiffusionProteinLanguageModel as DPLM2,
)


def find_motif_in_aa_seq(
    aa_seq: str, motif_segment: str, seq: Optional[str] = None, name: Optional[str] = None
) -> Tuple[Optional[int], Optional[int]]:
    index = aa_seq.find(motif_segment)
    if index == -1:
        print(f"Motif segment not found in aa_seq: {motif_segment}")
        try:
            print(f"seq: {seq}")
            print(f"name: {name}")
        except Exception:
            pass
        return None, None
    return index, index + len(motif_segment) - 1


def compute_motif_span_in_aaseq(
    aa_start: int, aa_end: int, motif_s: int, motif_e: int
) -> Optional[Tuple[int, int]]:
    if motif_s <= aa_start:
        aa_motif_s = aa_start
    elif motif_s > aa_end:
        return None
    else:
        aa_motif_s = motif_s

    if motif_e >= aa_end:
        aa_motif_e = aa_end
    elif motif_e < aa_start:
        return None
    else:
        aa_motif_e = motif_e

    start = aa_motif_s - aa_start
    end = aa_motif_e - aa_start
    return start, end


def ensure_struct_tokens(struct_seq: Any) -> List[str]:
    if struct_seq is None:
        return []
    if isinstance(struct_seq, list):
        return [str(x) for x in struct_seq]
    if isinstance(struct_seq, str):
        s = struct_seq.strip()
        if not s:
            return []
        if "," in s:
            return [tok for tok in s.split(",") if tok != ""]
        return [tok for tok in s.split() if tok != ""]
    return [str(struct_seq)]


def select_one_protein(
    data: List[Dict[str, Any]],
    index: Optional[int],
) -> Tuple[int, Dict[str, Any]]:
    if index is not None:
        return index, data[index]
    for i, item in enumerate(data):
        struct_tokens = ensure_struct_tokens(item.get("struct_seq", ""))
        pfam_motif = item.get("pfam_motif", None)
        if struct_tokens and isinstance(pfam_motif, list) and len(pfam_motif) > 0:
            return i, item
    raise ValueError("No suitable protein found: need non-empty struct_seq and pfam_motif list.")


def pad_batch(input_ids_list: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(int(t.shape[1]) for t in input_ids_list)
    batch = torch.full(
        (len(input_ids_list), max_len),
        fill_value=int(pad_id),
        dtype=input_ids_list[0].dtype,
        device=input_ids_list[0].device,
    )
    for i, t in enumerate(input_ids_list):
        batch[i, : t.shape[1]] = t[0]
    return batch


def build_motif_inputs(
    model: DPLM2,
    item: Dict[str, Any],
    min_len: int,
    max_motifs: Optional[int],
    device: torch.device,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    ori_seq = item.get("sequence", "")
    aa_seq = item.get("aa_seq", ori_seq)
    uniprot_id = item.get("uniprot_id", None)

    aa_start, aa_end = find_motif_in_aa_seq(ori_seq, aa_seq, seq=ori_seq, name=uniprot_id)
    if aa_start is None or aa_end is None:
        aa_start, aa_end = 0, max(0, len(aa_seq) - 1)

    struct_tokens = ensure_struct_tokens(item.get("struct_seq", ""))
    pfam_motif = item.get("pfam_motif", [])

    cls_struct_id = model.tokenizer._token_to_id["<cls_struct>"]
    eos_struct_id = model.tokenizer._token_to_id["<eos_struct>"]
    token_to_id = model.tokenizer._token_to_id

    inputs: List[torch.Tensor] = []
    kept = 0
    skipped = 0
    for motif_info in pfam_motif:
        if max_motifs is not None and kept >= max_motifs:
            break
        if not isinstance(motif_info, dict):
            skipped += 1
            continue
        if "start" not in motif_info or "end" not in motif_info:
            skipped += 1
            continue

        motif_s = int(motif_info["start"]) - 1
        motif_e = int(motif_info["end"]) - 1

        span = compute_motif_span_in_aaseq(aa_start, aa_end, motif_s, motif_e)
        if span is None:
            skipped += 1
            continue
        start, end = span
        if end - start + 1 < min_len:
            skipped += 1
            continue
        if start < 0 or end >= len(struct_tokens):
            skipped += 1
            continue

        segment = struct_tokens[start : end + 1]
        try:
            ids = [cls_struct_id] + [token_to_id[tok] for tok in segment] + [eos_struct_id]
        except KeyError:
            skipped += 1
            continue

        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        inputs.append(input_ids)
        kept += 1

    meta = {
        "uniprot_id": uniprot_id,
        "aa_len": len(aa_seq),
        "ori_len": len(ori_seq),
        "struct_len": len(struct_tokens),
        "motifs_total": len(pfam_motif) if isinstance(pfam_motif, list) else 0,
        "motifs_kept": kept,
        "motifs_skipped": skipped,
    }
    return inputs, meta


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-pkl", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--min-len", type=int, default=20)
    parser.add_argument("--max-motifs", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="airkingbd/dplm2_650m")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    data_pkl = args.data_pkl
    if data_pkl is None:
        data_pkl = (
            f"data-bin/uniprotKB/cfpgen_general_dataset/"
            f"{args.split}_all_old_motif_added_pfamMotif_esmfold.pkl"
        )

    with open(data_pkl, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in pickle, got {type(data)}")

    device_str = args.device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    amp_dtype = dtype_map[args.dtype]

    t0 = time.perf_counter()
    model = DPLM2.from_pretrained(args.model_name)
    model.eval()
    model.to(device)
    maybe_sync(device)
    t1 = time.perf_counter()

    idx, item = select_one_protein(data, args.index)
    motif_inputs, meta = build_motif_inputs(
        model=model,
        item=item,
        min_len=args.min_len,
        max_motifs=args.max_motifs,
        device=device,
    )
    if len(motif_inputs) == 0:
        raise ValueError("No motif segments passed filtering; try lowering --min-len or pick another --index.")

    batch = pad_batch(motif_inputs, pad_id=model.pad_id)
    total_tokens = int(batch.numel())
    maybe_sync(device)

    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            if device.type == "cuda" and amp_dtype != torch.float32:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _ = model(batch)
            else:
                _ = model(batch)
        maybe_sync(device)

        times: List[float] = []
        last_emb_shape = None
        for _ in range(max(1, args.repeat)):
            maybe_sync(device)
            start_t = time.perf_counter()
            if device.type == "cuda" and amp_dtype != torch.float32:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(batch)
            else:
                out = model(batch)
            maybe_sync(device)
            end_t = time.perf_counter()
            times.append(end_t - start_t)
            last_emb_shape = tuple(out["last_hidden_state"].shape)

    mean_s = sum(times) / len(times)
    ms_per_protein = mean_s * 1000.0
    ms_per_motif = ms_per_protein / max(1, meta["motifs_kept"])
    tok_per_s = total_tokens / mean_s

    print(f"data_pkl: {data_pkl}")
    print(f"picked_index: {idx}")
    print(f"uniprot_id: {meta['uniprot_id']}")
    print(f"model_load_s: {(t1 - t0):.3f}")
    print(
        "protein_lens:"
        f" ori={meta['ori_len']}"
        f" aa={meta['aa_len']}"
        f" struct={meta['struct_len']}"
    )
    print(
        "motifs:"
        f" total={meta['motifs_total']}"
        f" kept={meta['motifs_kept']}"
        f" skipped={meta['motifs_skipped']}"
        f" min_len={args.min_len}"
        + (f" max_motifs={args.max_motifs}" if args.max_motifs is not None else "")
    )
    print(f"batch: B={batch.shape[0]} L={batch.shape[1]} total_tokens={total_tokens}")
    print(f"device: {device} amp_dtype: {amp_dtype}")
    print(f"last_hidden_state_shape: {last_emb_shape}")
    print(f"time_ms_per_protein: {ms_per_protein:.2f}")
    print(f"time_ms_per_motif: {ms_per_motif:.2f}")
    print(f"tokens_per_s: {tok_per_s:.0f}")


if __name__ == "__main__":
    main()
