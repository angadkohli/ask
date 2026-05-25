"""Process every image in samples/ with auto-detected corners and write outputs."""

import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from auto_corners import detect_design_quad, draw_corners
from extractor import (
    perspective_unwarp,
    save_png,
    save_pdf,
    save_embedded_svg,
    save_traced_svg,
    physical_to_pixels,
)


# Default print dimensions per mockup-type (cabin suitcase front panel approx 300x450mm).
WIDTH_MM = 300
HEIGHT_MM = 450
DPI = 300

ROOT = os.path.dirname(__file__)
SAMPLES = os.path.join(ROOT, "samples")
RUNS = os.path.join(ROOT, "runs")


def process(image_path: str) -> dict:
    base = os.path.splitext(os.path.basename(image_path))[0]
    outdir = os.path.join(RUNS, base)
    os.makedirs(outdir, exist_ok=True)

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    corners = detect_design_quad(bgr)
    preview = draw_corners(bgr, corners)
    cv2.imwrite(os.path.join(outdir, "0_corners.jpg"), preview,
                [cv2.IMWRITE_JPEG_QUALITY, 85])

    px_w, px_h = physical_to_pixels(WIDTH_MM, HEIGHT_MM, DPI)
    flat = perspective_unwarp(rgb, corners, px_w, px_h)

    save_png(flat, os.path.join(outdir, "design.png"), dpi=DPI)
    save_pdf(flat, os.path.join(outdir, "design.pdf"), WIDTH_MM, HEIGHT_MM)
    save_embedded_svg(flat, os.path.join(outdir, "design-embedded.svg"),
                      WIDTH_MM, HEIGHT_MM)
    save_traced_svg(flat, os.path.join(outdir, "design-traced.svg"),
                    color_precision=6)

    return {
        "image": base,
        "src_size": (bgr.shape[1], bgr.shape[0]),
        "corners": corners,
        "output_size_px": (px_w, px_h),
        "outdir": outdir,
    }


if __name__ == "__main__":
    targets = [
        "amaira_dinosaur.jpg",
        "amaira_space.jpg",
        "kiyara_leopard.png",
        "amairee_floral.png",
    ]
    for t in targets:
        path = os.path.join(SAMPLES, t)
        if not os.path.exists(path):
            print(f"[skip] {t} (missing)")
            continue
        res = process(path)
        print(f"[ok] {res['image']}: src {res['src_size']} -> {res['output_size_px']} px in {res['outdir']}")
