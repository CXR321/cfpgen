import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int


def _compute_content_bbox(img_rgba: Image.Image, bg_threshold: int) -> Optional[CropBox]:
    arr = np.asarray(img_rgba)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("Expected RGBA image array")

    rgb = arr[:, :, :3].astype(np.uint16)
    alpha = arr[:, :, 3].astype(np.uint16)

    content_mask = (alpha > 0) & (np.min(rgb, axis=2) < bg_threshold)
    ys, xs = np.where(content_mask)
    if ys.size == 0 or xs.size == 0:
        return None

    top = int(ys.min())
    bottom = int(ys.max()) + 1
    left = int(xs.min())
    right = int(xs.max()) + 1
    return CropBox(left=left, top=top, right=right, bottom=bottom)


def _expand_bbox(bbox: CropBox, width: int, height: int, pad_x: int, pad_y: int) -> CropBox:
    left = max(0, bbox.left - pad_x)
    right = min(width, bbox.right + pad_x)
    top = max(0, bbox.top - pad_y)
    bottom = min(height, bbox.bottom + pad_y)
    return CropBox(left=left, top=top, right=right, bottom=bottom)


def _parse_band(text: str) -> Tuple[int, int]:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid band: {text}. Expected 'y0:y1'")
    y0 = int(parts[0])
    y1 = int(parts[1])
    if y0 == y1:
        raise ValueError(f"Invalid band: {text}. y0 and y1 must differ")
    return (min(y0, y1), max(y0, y1))


def _normalize_bands(bands: List[Tuple[int, int]], height: int) -> List[Tuple[int, int]]:
    clipped = []
    for y0, y1 in bands:
        y0c = max(0, min(height, y0))
        y1c = max(0, min(height, y1))
        if y1c > y0c:
            clipped.append((y0c, y1c))
    if not clipped:
        return []
    clipped.sort(key=lambda x: x[0])

    merged: List[Tuple[int, int]] = []
    cur0, cur1 = clipped[0]
    for y0, y1 in clipped[1:]:
        if y0 <= cur1:
            cur1 = max(cur1, y1)
        else:
            merged.append((cur0, cur1))
            cur0, cur1 = y0, y1
    merged.append((cur0, cur1))
    return merged


def _remove_horizontal_bands(arr: np.ndarray, bands: List[Tuple[int, int]]) -> np.ndarray:
    h = int(arr.shape[0])
    normalized = _normalize_bands(bands, height=h)
    if not normalized:
        return arr

    keep_slices = []
    start = 0
    for y0, y1 in normalized:
        if y0 > start:
            keep_slices.append(arr[start:y0, :, :])
        start = y1
    if start < h:
        keep_slices.append(arr[start:h, :, :])

    if not keep_slices:
        return arr[0:1, :, :]

    return np.concatenate(keep_slices, axis=0)


def _manual_crop(
    img_rgba: Image.Image, left: int, top: int, right: int, bottom: int
) -> Image.Image:
    w, h = img_rgba.size
    l = max(0, min(w, left))
    t = max(0, min(h, top))
    r = max(0, min(w, w - right))
    b = max(0, min(h, h - bottom))
    if r <= l or b <= t:
        return img_rgba
    return img_rgba.crop((l, t, r, b))


def crop_whitespace(
    input_path: str,
    output_path: str,
    bg_threshold: int = 250,
    pad_x: int = 10,
    pad_y: int = 5,
    remove_bands: Optional[List[Tuple[int, int]]] = None,
    crop_left: int = 0,
    crop_top: int = 0,
    crop_right: int = 0,
    crop_bottom: int = 0,
) -> None:
    with Image.open(input_path) as img:
        img_rgba = img.convert("RGBA")

        if remove_bands:
            arr = np.asarray(img_rgba)
            arr2 = _remove_horizontal_bands(arr, remove_bands)
            img_rgba = Image.fromarray(arr2, mode="RGBA")

        if any(v != 0 for v in (crop_left, crop_top, crop_right, crop_bottom)):
            img_rgba = _manual_crop(
                img_rgba,
                left=crop_left,
                top=crop_top,
                right=crop_right,
                bottom=crop_bottom,
            )

        if remove_bands or any(v != 0 for v in (crop_left, crop_top, crop_right, crop_bottom)):
            img_rgba.save(output_path)
            return

        bbox = _compute_content_bbox(img_rgba, bg_threshold=bg_threshold)
        if bbox is None:
            img_rgba.save(output_path)
            return

        bbox = _expand_bbox(
            bbox, width=img_rgba.width, height=img_rgba.height, pad_x=pad_x, pad_y=pad_y
        )
        cropped = img_rgba.crop((bbox.left, bbox.top, bbox.right, bbox.bottom))
        cropped.save(output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--remove-band", action="append", default=[])
    p.add_argument("--crop-left", type=int, default=0)
    p.add_argument("--crop-top", type=int, default=0)
    p.add_argument("--crop-right", type=int, default=0)
    p.add_argument("--crop-bottom", type=int, default=0)
    p.add_argument("--bg-threshold", type=int, default=250)
    p.add_argument("--pad-x", type=int, default=10)
    p.add_argument("--pad-y", type=int, default=5)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    remove_bands = [_parse_band(t) for t in args.remove_band] if args.remove_band else None
    crop_whitespace(
        input_path=args.input,
        output_path=args.output,
        bg_threshold=args.bg_threshold,
        pad_x=args.pad_x,
        pad_y=args.pad_y,
        remove_bands=remove_bands,
        crop_left=args.crop_left,
        crop_top=args.crop_top,
        crop_right=args.crop_right,
        crop_bottom=args.crop_bottom,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
