"""Sanity test: synthesize a fake mockup (known design warped onto a background),
then unwarp it and verify we recover something close to the original design."""

import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from extractor import (
    perspective_unwarp, save_png, save_pdf, save_embedded_svg, save_traced_svg,
    physical_to_pixels,
)


def make_design(w=800, h=1200):
    img = Image.new("RGB", (w, h), (15, 15, 18))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, w - 40, h - 40], outline=(255, 255, 255), width=6)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 180)
    except Exception:
        font = ImageFont.load_default()
    txt = "AMAIREE"
    bbox = draw.textbbox((0, 0), txt, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) / 2, (h - th) / 2 - 100), txt, fill=(240, 240, 240), font=font)
    for i, col in enumerate([(220, 80, 60), (60, 160, 220), (240, 200, 80)]):
        cy = h // 2 + 200 + i * 80
        draw.ellipse([w // 2 - 60, cy - 30, w // 2 + 60, cy + 30], fill=col)
    return np.array(img)


def warp_to_mockup(design, bg_w=1600, bg_h=2200):
    bg = np.full((bg_h, bg_w, 3), 200, dtype=np.uint8)
    # Simulate a slightly-perspective panel inset
    dst_corners = np.array([
        [300, 180], [bg_w - 320, 160],
        [bg_w - 260, bg_h - 240], [260, bg_h - 200],
    ], dtype=np.float32)
    src_corners = np.array([
        [0, 0], [design.shape[1] - 1, 0],
        [design.shape[1] - 1, design.shape[0] - 1], [0, design.shape[0] - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped = cv2.warpPerspective(design, M, (bg_w, bg_h))
    mask = cv2.warpPerspective(
        np.full(design.shape[:2], 255, dtype=np.uint8), M, (bg_w, bg_h)
    )
    out = bg.copy()
    out[mask > 0] = warped[mask > 0]
    return out, dst_corners.astype(int).tolist()


def main():
    outdir = os.path.join(os.path.dirname(__file__), "test_out")
    os.makedirs(outdir, exist_ok=True)

    design = make_design()
    Image.fromarray(design).save(os.path.join(outdir, "0_original_design.png"))

    mockup, corners = warp_to_mockup(design)
    Image.fromarray(mockup).save(os.path.join(outdir, "1_synthetic_mockup.png"))
    print(f"corners (TL, TR, BR, BL): {corners}")

    px_w, px_h = physical_to_pixels(300, 450, 150)  # low DPI for fast test
    unwarped = perspective_unwarp(mockup, [tuple(c) for c in corners], px_w, px_h)

    save_png(unwarped, os.path.join(outdir, "2_extracted.png"), dpi=150)
    save_pdf(unwarped, os.path.join(outdir, "2_extracted.pdf"), 300, 450)
    save_embedded_svg(unwarped, os.path.join(outdir, "2_extracted_embedded.svg"), 300, 450)
    save_traced_svg(unwarped, os.path.join(outdir, "2_extracted_traced.svg"), color_precision=5)

    print(f"\nFiles written to {outdir}:")
    for f in sorted(os.listdir(outdir)):
        size = os.path.getsize(os.path.join(outdir, f))
        print(f"  {f}  ({size:,} bytes)")

    # Compare: resize extracted to original size, compute mean abs diff.
    extracted_resized = cv2.resize(unwarped, (design.shape[1], design.shape[0]),
                                   interpolation=cv2.INTER_AREA)
    diff = np.mean(np.abs(extracted_resized.astype(int) - design.astype(int)))
    print(f"\nMean per-channel abs diff (extracted vs original): {diff:.2f} / 255")
    if diff > 5:
        print("WARNING: large diff — perspective unwarp may be misaligned")
        sys.exit(1)


if __name__ == "__main__":
    main()
