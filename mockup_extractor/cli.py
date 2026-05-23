"""CLI: batch-extract designs from mockups using a JSON template per mockup.

Template file format (corners in image pixels; reorder=true accepts any click order):

    {
      "input": "samples/tiger.jpg",
      "corners": [[330, 110], [1100, 110], [1100, 1420], [330, 1420]],
      "output_dir": "out/tiger",
      "width_mm": 300,
      "height_mm": 450,
      "dpi": 300,
      "inpaint_hardware": false,
      "brightness": 0,
      "contrast": 1.0,
      "gamma": 1.0,
      "trace": true,
      "trace_color_precision": 6
    }

Run:  python cli.py template.json [template2.json ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

from extractor import (
    apply_levels,
    mask_dark_hardware,
    perspective_unwarp,
    physical_to_pixels,
    save_embedded_svg,
    save_pdf,
    save_png,
    save_traced_svg,
)


def run_template(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    inp = cfg["input"]
    if not os.path.isabs(inp):
        inp = os.path.join(os.path.dirname(os.path.abspath(template_path)), inp)
    image_bgr = cv2.imread(inp, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(inp)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    corners = [tuple(c) for c in cfg["corners"]]
    if len(corners) != 4:
        raise ValueError("`corners` must have 4 entries")

    width_mm = float(cfg.get("width_mm", 300))
    height_mm = float(cfg.get("height_mm", 450))
    dpi = int(cfg.get("dpi", 300))
    px_w, px_h = physical_to_pixels(width_mm, height_mm, dpi)

    flat = perspective_unwarp(image, corners, px_w, px_h)

    if cfg.get("inpaint_hardware", False):
        flat, _ = mask_dark_hardware(flat)

    if any(cfg.get(k, default) != default for k, default in
           (("brightness", 0), ("contrast", 1.0), ("gamma", 1.0))):
        flat = apply_levels(
            flat,
            brightness=float(cfg.get("brightness", 0)),
            contrast=float(cfg.get("contrast", 1.0)),
            gamma=float(cfg.get("gamma", 1.0)),
        )

    outdir = cfg.get("output_dir", "out")
    if not os.path.isabs(outdir):
        outdir = os.path.join(os.path.dirname(os.path.abspath(template_path)), outdir)
    os.makedirs(outdir, exist_ok=True)

    save_png(flat, os.path.join(outdir, "design.png"), dpi=dpi)
    save_pdf(flat, os.path.join(outdir, "design.pdf"), width_mm, height_mm)
    save_embedded_svg(flat, os.path.join(outdir, "design-embedded.svg"), width_mm, height_mm)
    if cfg.get("trace", True):
        save_traced_svg(
            flat,
            os.path.join(outdir, "design-traced.svg"),
            color_precision=int(cfg.get("trace_color_precision", 6)),
        )

    return outdir


def main():
    p = argparse.ArgumentParser(description="Batch mockup design extractor")
    p.add_argument("templates", nargs="+", help="One or more template JSON files")
    args = p.parse_args()
    for t in args.templates:
        out = run_template(t)
        print(f"[ok] {t} -> {out}")


if __name__ == "__main__":
    main()
