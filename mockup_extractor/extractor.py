"""Core image-processing primitives for reversing a product mockup.

Pipeline: mockup image + 4 corner clicks
  -> perspective unwarp to a flat rectangle
  -> optional inpaint of dark hardware overlays
  -> optional levels/gamma correction
  -> emit print-ready PNG + PDF + embedded SVG + traced color SVG.
"""

from __future__ import annotations

import base64
import os
import tempfile
from typing import Iterable

import cv2
import numpy as np
import vtracer
from PIL import Image


Corner = tuple[float, float]


def order_corners_clockwise(corners: Iterable[Corner]) -> list[Corner]:
    """Order 4 corners as TL, TR, BR, BL regardless of click order."""
    pts = np.array(list(corners), dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Need exactly 4 corners")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def perspective_unwarp(
    image: np.ndarray,
    corners: list[Corner],
    out_w: int,
    out_h: int,
    reorder: bool = True,
) -> np.ndarray:
    """Project the quadrilateral defined by `corners` into a flat (out_w, out_h) image."""
    if reorder:
        corners = order_corners_clockwise(corners)
    src = np.array(corners, dtype=np.float32)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h), flags=cv2.INTER_LANCZOS4)


def apply_levels(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    out = image.astype(np.float32)
    if contrast != 1.0 or brightness != 0.0:
        mid = 127.5
        out = (out - mid) * contrast + mid + brightness
    if gamma != 1.0:
        out = np.clip(out, 0, 255)
        out = 255.0 * np.power(out / 255.0, 1.0 / gamma)
    return np.clip(out, 0, 255).astype(np.uint8)


def mask_dark_hardware(
    image: np.ndarray,
    dark_thresh: int = 30,
    dilate_px: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Inpaint regions darker than `dark_thresh` (intended for corner protectors / zippers
    that sit on top of the design after unwarping). Returns (inpainted, mask)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = (gray < dark_thresh).astype(np.uint8) * 255
    if dilate_px > 0:
        k = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)
    inpainted = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    return inpainted, mask


def save_png(image: np.ndarray, path: str, dpi: int) -> str:
    Image.fromarray(image).save(path, dpi=(dpi, dpi))
    return path


def save_pdf(image: np.ndarray, path: str, width_mm: float, height_mm: float) -> str:
    """Save a single-page PDF sized to width_mm x height_mm. DPI is derived from
    the image's pixel dimensions so the embedded raster fills the page exactly."""
    img = Image.fromarray(image)
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    # PIL uses one DPI value for the whole PDF; pick the smaller so neither axis overflows.
    dpi = min(img.width / width_in, img.height / height_in)
    img.save(path, "PDF", resolution=dpi)
    return path


def save_embedded_svg(
    image: np.ndarray, path: str, width_mm: float, height_mm: float
) -> str:
    """SVG container with the raster embedded as a base64 PNG, sized in millimetres."""
    buf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    buf.close()
    Image.fromarray(image).save(buf.name, format="PNG")
    with open(buf.name, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    os.unlink(buf.name)
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width_mm}mm" height="{height_mm}mm" '
        f'viewBox="0 0 {width_mm} {height_mm}">\n'
        f'  <image x="0" y="0" width="{width_mm}" height="{height_mm}" '
        f'preserveAspectRatio="none" '
        f'xlink:href="data:image/png;base64,{b64}"/>\n'
        f'</svg>\n'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return path


def save_traced_svg(
    image: np.ndarray,
    path: str,
    colormode: str = "color",
    mode: str = "spline",
    color_precision: int = 6,
    filter_speckle: int = 4,
    path_precision: int = 8,
) -> str:
    """True vector trace via vtracer. Best on flat/limited-palette artwork; complex
    photographic regions are necessarily approximated."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    Image.fromarray(image).save(tmp.name, format="PNG")
    try:
        vtracer.convert_image_to_svg_py(
            tmp.name,
            path,
            colormode=colormode,
            mode=mode,
            color_precision=color_precision,
            filter_speckle=filter_speckle,
            path_precision=path_precision,
        )
    finally:
        os.unlink(tmp.name)
    return path


def physical_to_pixels(width_mm: float, height_mm: float, dpi: int) -> tuple[int, int]:
    """Convert physical print size to pixel dimensions at the given DPI."""
    px_w = max(1, int(round(width_mm / 25.4 * dpi)))
    px_h = max(1, int(round(height_mm / 25.4 * dpi)))
    return px_w, px_h
