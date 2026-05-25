"""Batch design recovery from a folder of mockups.

Process every image in --input-dir against the template PSD. When a mockup's
dimensions don't match the template, the template's alpha is bilinearly resized
to fit (so any DPI variant of the same suitcase template works).

Each mockup produces:
  <name>_design.png         — recovered design, text intact
  <name>_design.pdf         — same as PDF at the configured DPI
  <name>_design-embedded.svg — SVG wrapper around the PNG (vector container)
  <name>_design_blank.png   — text removed (if --remove-text), good as a re-usable template
  <name>_text_mask.png      — OCR mask diagnostic (only when text was removed)

Run:
    python batch_recover.py \\
        --template psd_samples/cabin-suitcase-300dpi.psd \\
        --input-dir samples/ \\
        --out runs/batch \\
        --remove-text --keep-bottom-brand
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

from template_extract_v2 import (
    load_template_alpha, load_mockup_rgb,
    inpaint_with_lama, detect_text_mask,
    crop_to_bbox, save_print_outputs,
)


EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}


def resize_template_alpha_to(tpl_alpha: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize the template alpha to a mockup's pixel size. NEAREST so the
    hard hardware/design boundary stays crisp."""
    if tpl_alpha.shape == (target_h, target_w):
        return tpl_alpha
    return cv2.resize(tpl_alpha, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def list_mockups(input_dir: str) -> list[str]:
    out = []
    for ext in EXTS:
        out.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        out.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    return sorted(set(out))


def process(template_alpha: np.ndarray, mockup_path: str, outdir: str,
            remove_text: bool, keep_bottom_brand: bool, dpi: int) -> dict:
    base = os.path.splitext(os.path.basename(mockup_path))[0]
    t0 = time.time()
    rgb = load_mockup_rgb(mockup_path)
    h, w = rgb.shape[:2]
    tpl = resize_template_alpha_to(template_alpha, h, w)
    hardware_mask = (tpl > 0).astype(np.uint8) * 255
    hardware_mask = cv2.dilate(hardware_mask, np.ones((5, 5), np.uint8), iterations=1)
    design_mask = (tpl == 0).astype(np.uint8)
    rgb_clean = inpaint_with_lama(rgb, hardware_mask)
    meta = save_print_outputs(crop_to_bbox(rgb_clean, design_mask), outdir, f"{base}_design", dpi=dpi)
    info = {"base": base, "size_px_in": (w, h), "size_px_out": meta["size_px"],
            "size_mm_out": meta["size_mm"], "time_s": round(time.time() - t0, 1)}
    if remove_text:
        text_mask = detect_text_mask(rgb_clean, design_mask, keep_bottom_brand=keep_bottom_brand)
        if (text_mask > 0).any():
            rgb_blank = inpaint_with_lama(rgb_clean, text_mask)
            save_print_outputs(crop_to_bbox(rgb_blank, design_mask), outdir, f"{base}_design_blank", dpi=dpi)
            cv2.imwrite(os.path.join(outdir, f"{base}_text_mask.png"), text_mask)
            info["text_removed"] = True
        else:
            info["text_removed"] = False
    info["time_s"] = round(time.time() - t0, 1)
    return info


def main():
    p = argparse.ArgumentParser(description="Batch mockup -> design recovery")
    p.add_argument("--template", required=True, help="Template PSD")
    p.add_argument("--input-dir", required=True, help="Folder of mockup images")
    p.add_argument("--out", default="runs/batch")
    p.add_argument("--remove-text", action="store_true")
    p.add_argument("--keep-bottom-brand", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading template alpha from {args.template}...")
    tpl = load_template_alpha(args.template)
    print(f"  template: {tpl.shape}, design pixels: {(tpl == 0).sum():,}")

    mockups = list_mockups(args.input_dir)
    if not mockups:
        print(f"  no mockups found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"\n{len(mockups)} mockups to process.")

    total0 = time.time()
    summary = []
    for i, m in enumerate(mockups, 1):
        print(f"\n[{i}/{len(mockups)}] {os.path.basename(m)}")
        try:
            info = process(tpl, m, args.out, args.remove_text, args.keep_bottom_brand, args.dpi)
            print(f"  ok  {info['size_px_in']} -> {info['size_px_out']} ({info['size_mm_out']} mm) in {info['time_s']}s")
            summary.append(info)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            summary.append({"base": os.path.basename(m), "error": str(e)})

    print(f"\nDone {len([s for s in summary if 'error' not in s])}/{len(summary)} in {time.time() - total0:.1f}s")
    print(f"Outputs in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
