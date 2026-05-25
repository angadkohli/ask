"""Lossless design recovery using a mockup template PSD.

Given a 'template' PSD that contains only the suitcase hardware (corner caps,
zipper, handle, wheels) on opaque pixels with the design region transparent,
and a 'mockup' PNG of the same dimensions (which is template+design composited),
we can lift the original design pixels back out by masking with the template's
alpha channel.

For the small fraction of design pixels physically hidden under the hardware
(rounded corner caps, wheel notches, zipper edge), we inpaint using OpenCV's
patch-based algorithm to fill them in. The pattern continuity is usually good
enough that no generative model is required.

Usage:
    python template_extract.py --template path/to/template.psd \\
        --mockup path/to/finished_mockup.png \\
        --out runs/<name>/
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage


def load_template_alpha(psd_path: str) -> np.ndarray:
    """Return the template's alpha channel — opaque = hardware, transparent = design."""
    psd = PSDImage.open(psd_path)
    composite = psd.composite()
    arr = np.array(composite.convert("RGBA"))
    return arr[..., 3]  # (H, W) uint8


def load_image_rgba(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, dtype=np.uint8)])
    return img


def recover_design(template_alpha: np.ndarray, mockup: np.ndarray,
                   inpaint_radius: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgba_recovered, hardware_mask) at template resolution.

    `rgba_recovered` has the original design pixels in their template positions,
    with the hardware regions inpainted from neighbouring design pixels.
    """
    if template_alpha.shape != mockup.shape[:2]:
        raise ValueError(
            f"template {template_alpha.shape} != mockup {mockup.shape[:2]}"
        )
    # Design pixels = template transparent (alpha == 0). Hardware = opaque.
    hardware = (template_alpha > 0).astype(np.uint8)
    # Slightly grow the hardware mask so anti-aliased hardware edges don't bleed.
    hardware = cv2.dilate(hardware, np.ones((3, 3), np.uint8), iterations=1)

    bgr = mockup[..., :3].copy()
    # Inpaint hardware regions from surrounding design pixels.
    inpainted = cv2.inpaint(bgr, hardware * 255, inpaint_radius, cv2.INPAINT_TELEA)

    rgba = np.dstack([inpainted, np.full(bgr.shape[:2], 255, dtype=np.uint8)])
    return rgba, hardware * 255


def crop_to_design_bbox(rgba: np.ndarray, template_alpha: np.ndarray,
                        bleed_px: int = 0) -> np.ndarray:
    """Crop the recovered design down to the bounding box of the design region."""
    design_mask = (template_alpha == 0)
    ys, xs = np.where(design_mask)
    if ys.size == 0:
        return rgba
    y0 = max(0, ys.min() - bleed_px)
    y1 = min(rgba.shape[0], ys.max() + 1 + bleed_px)
    x0 = max(0, xs.min() - bleed_px)
    x1 = min(rgba.shape[1], xs.max() + 1 + bleed_px)
    return rgba[y0:y1, x0:x1]


def save_outputs(rgba: np.ndarray, outdir: str, base: str, dpi: int = 300,
                 width_mm: float | None = None, height_mm: float | None = None) -> dict:
    os.makedirs(outdir, exist_ok=True)
    h, w = rgba.shape[:2]
    if width_mm is None:
        width_mm = w / dpi * 25.4
    if height_mm is None:
        height_mm = h / dpi * 25.4

    png_path = os.path.join(outdir, f"{base}.png")
    Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)).save(png_path, dpi=(dpi, dpi))

    pdf_path = os.path.join(outdir, f"{base}.pdf")
    Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGB)).save(
        pdf_path, "PDF", resolution=dpi
    )

    # Embedded SVG container at physical size
    import base64
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    svg_path = os.path.join(outdir, f"{base}-embedded.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width_mm:.2f}mm" height="{height_mm:.2f}mm" '
            f'viewBox="0 0 {width_mm:.2f} {height_mm:.2f}">\n'
            f'  <image x="0" y="0" width="{width_mm:.2f}" height="{height_mm:.2f}" '
            f'preserveAspectRatio="none" xlink:href="data:image/png;base64,{b64}"/>\n'
            f'</svg>\n'
        )
    return {
        "png": png_path, "pdf": pdf_path, "svg": svg_path,
        "size_px": [w, h], "size_mm": [round(width_mm, 1), round(height_mm, 1)],
    }


def main():
    p = argparse.ArgumentParser(description="Recover original design from mockup using template PSD")
    p.add_argument("--template", required=True, help="Template PSD (suitcase hardware, transparent design region)")
    p.add_argument("--mockup", required=True, action="append",
                   help="Mockup PNG/JPG (same dimensions as template). Repeatable.")
    p.add_argument("--out", default="runs/template_recovered", help="Output directory")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--inpaint-radius", type=int, default=5)
    p.add_argument("--bleed-px", type=int, default=0, help="Extra pixels around the design bbox when cropping")
    args = p.parse_args()

    print(f"Loading template alpha from {args.template}...")
    tpl_alpha = load_template_alpha(args.template)
    print(f"  template alpha: {tpl_alpha.shape}, design pixels: "
          f"{(tpl_alpha == 0).sum():,}, hardware pixels: {(tpl_alpha > 0).sum():,}")

    for m in args.mockup:
        base = os.path.splitext(os.path.basename(m))[0]
        print(f"\nProcessing {m}...")
        mockup = load_image_rgba(m)
        if mockup.shape[:2] != tpl_alpha.shape:
            print(f"  [skip] size mismatch: mockup {mockup.shape[:2]} vs template {tpl_alpha.shape}")
            continue

        rgba, hw_mask = recover_design(tpl_alpha, mockup, inpaint_radius=args.inpaint_radius)
        # Save the un-cropped full-canvas version with masked hardware highlighted
        cv2.imwrite(os.path.join(args.out, f"{base}_hardware_mask.png"), hw_mask)

        cropped = crop_to_design_bbox(rgba, tpl_alpha, bleed_px=args.bleed_px)

        os.makedirs(args.out, exist_ok=True)
        meta = save_outputs(cropped, args.out, f"{base}_design", dpi=args.dpi)
        print(f"  -> {meta['png']}  ({meta['size_px'][0]}x{meta['size_px'][1]} px, "
              f"{meta['size_mm'][0]}x{meta['size_mm'][1]} mm @ {args.dpi} DPI)")


if __name__ == "__main__":
    main()
