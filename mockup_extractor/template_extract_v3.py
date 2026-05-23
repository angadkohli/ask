"""Template-based design recovery v3 — higher quality + multi-size print export.

Key improvements over v2:
- Banner-aware text fill: detect the colored band that holds the personalized
  name; replace text pixels with the exact band color (zero AI artefacts).
  Falls back to LaMa for text outside any detected band.
- Higher-resolution LaMa for hardware inpaint (process windows at native res
  when small, with much larger context padding).
- Default: remove ALL text (including small bottom brand mark) unless
  --keep-bottom-brand is set.
- Multi-size print export at 300 DPI with generative outpainting to fit each
  target aspect ratio.

Run:
    python template_extract_v3.py \\
        --template psd_samples/cabin-suitcase-300dpi.psd \\
        --mockup IMG_5249.png --mockup IMG_5250.png \\
        --out runs/v3 \\
        --print-sizes 16x20,18x24,22x29,24x32.125 \\
        --dpi 300
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage


# Lazy-load heavy ML deps
_LAMA = None
_OCR = None


def _lama():
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama()
    return _LAMA


def _ocr():
    global _OCR
    if _OCR is None:
        import easyocr
        _OCR = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR


# ---------- Loading ----------

def load_template_alpha(psd_path: str) -> np.ndarray:
    psd = PSDImage.open(psd_path)
    composite = psd.composite()
    return np.array(composite.convert("RGBA"))[..., 3]


def load_mockup_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[2] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_alpha_to(alpha: np.ndarray, h: int, w: int) -> np.ndarray:
    if alpha.shape == (h, w):
        return alpha
    return cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------- Inpainting primitives ----------

@dataclass
class Region:
    x0: int; y0: int; x1: int; y1: int
    @property
    def w(self): return self.x1 - self.x0
    @property
    def h(self): return self.y1 - self.y0
    def pad(self, px, max_w, max_h):
        return Region(max(0, self.x0 - px), max(0, self.y0 - px),
                      min(max_w, self.x1 + px), min(max_h, self.y1 + px))


def _mask_regions(mask: np.ndarray, min_area: int = 30) -> list[Region]:
    num, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append(Region(int(x), int(y), int(x + w), int(y + h)))
    return out


def lama_inpaint_hq(
    rgb: np.ndarray,
    mask: np.ndarray,
    context_pad_px: int = 256,
    max_window_dim: int = 2000,
    on_progress=None,
) -> np.ndarray:
    """Inpaint mask regions with LaMa, processing each region in a generous
    context window at as close to native resolution as memory allows.

    `context_pad_px` is how much surrounding image LaMa sees around the masked
    region. More context = better continuation of patterns / colors.
    """
    h, w = rgb.shape[:2]
    out = rgb.copy()
    regions = _mask_regions(mask)
    lama = _lama()
    for i, r in enumerate(regions):
        wnd = r.pad(context_pad_px, w, h)
        crop_rgb = out[wnd.y0:wnd.y1, wnd.x0:wnd.x1].copy()
        crop_mask = mask[wnd.y0:wnd.y1, wnd.x0:wnd.x1].copy()
        ch, cw = crop_rgb.shape[:2]
        scale = 1.0
        if max(ch, cw) > max_window_dim:
            scale = max_window_dim / max(ch, cw)
            new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
            crop_rgb_in = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            crop_mask_in = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            crop_rgb_in = crop_rgb
            crop_mask_in = crop_mask
        t0 = time.time()
        inpainted = np.array(lama(Image.fromarray(crop_rgb_in), Image.fromarray(crop_mask_in)))
        if scale < 1.0:
            inpainted = cv2.resize(inpainted, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        else:
            inpainted = inpainted[:ch, :cw]
        # Soft alpha blend at mask edge so seams disappear
        soft = cv2.GaussianBlur(crop_mask.astype(np.float32), (9, 9), 0) / 255.0
        soft3 = soft[..., None]
        merged = (inpainted.astype(np.float32) * soft3 +
                  out[wnd.y0:wnd.y1, wnd.x0:wnd.x1].astype(np.float32) * (1 - soft3))
        out[wnd.y0:wnd.y1, wnd.x0:wnd.x1] = np.clip(merged, 0, 255).astype(np.uint8)
        if on_progress:
            on_progress(i + 1, len(regions), wnd, time.time() - t0)
    return out


# ---------- Banner detection (the killer feature for clean text removal) ----------

def detect_banner_regions(rgb: np.ndarray, design_mask: np.ndarray,
                          min_rel_width: float = 0.5,
                          min_rel_height: float = 0.04,
                          max_rel_height: float = 0.30) -> list[Region]:
    """Heuristically find horizontal coloured 'name banner' rectangles in the
    design region — they're wide, short, single-colour-dominant strips that
    span most of the design width.

    Returns a list of Region rectangles in image coords.
    """
    H, W = rgb.shape[:2]
    ys, xs = np.where(design_mask > 0)
    if ys.size == 0:
        return []
    dx0, dx1 = xs.min(), xs.max()
    dy0, dy1 = ys.min(), ys.max()
    dW = dx1 - dx0
    dH = dy1 - dy0

    # Quantize colors via OpenCV's k-means on a downsampled image.
    small = cv2.resize(rgb, (max(1, W // 6), max(1, H // 6)), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(design_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)

    # For each row of the design region, compute color variance & coverage.
    # A 'banner row' tends to be a horizontal strip where the dominant color
    # (excluding text) covers >70% of the row width within the design area.
    rows_ds = small.shape[0]
    sx0 = int(dx0 * small.shape[1] / W)
    sx1 = max(sx0 + 1, int(dx1 * small.shape[1] / W))
    sy0 = int(dy0 * rows_ds / H)
    sy1 = max(sy0 + 1, int(dy1 * rows_ds / H))

    banner_rows = np.zeros(rows_ds, dtype=bool)
    for y in range(sy0, sy1):
        row = small[y, sx0:sx1]
        row_mask = small_mask[y, sx0:sx1] > 0
        if row_mask.sum() < (sx1 - sx0) * 0.7:
            continue
        active = row[row_mask]
        if active.size == 0:
            continue
        # Saturation as proxy for "banner-like" colored region
        hsv = cv2.cvtColor(active.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        # Cluster colors via histogram on hue
        h_bins = np.bincount(hsv[:, 0], minlength=180)
        dom_count = h_bins.max()
        dom_frac = dom_count / hsv.shape[0]
        # A banner row has one hue dominating >60% (since text is a different color).
        if dom_frac > 0.55:
            banner_rows[y] = True

    # Find contiguous runs of banner rows
    bands = []
    in_run = False
    start = 0
    for y in range(rows_ds + 1):
        is_b = y < rows_ds and banner_rows[y]
        if is_b and not in_run:
            in_run = True
            start = y
        elif not is_b and in_run:
            in_run = False
            run_h = y - start
            # Scale back to full image coords
            y0_full = int(start * H / rows_ds)
            y1_full = int(y * H / rows_ds)
            rel_h = (y1_full - y0_full) / max(1, dH)
            if min_rel_height < rel_h < max_rel_height:
                bands.append(Region(dx0, y0_full, dx1, y1_full))
    return bands


def fit_band_color(rgb: np.ndarray, band: Region, design_mask: np.ndarray) -> np.ndarray:
    """Estimate per-row band color (handles solid + simple gradient bands).
    Returns an HxWx3 array of size (band.h, band.w, 3) ready to paste.
    """
    band_pixels = rgb[band.y0:band.y1, band.x0:band.x1]
    band_pixels_mask = design_mask[band.y0:band.y1, band.x0:band.x1] > 0
    # For each row, take the MEDIAN of valid (design-area) pixels to ignore text outliers.
    # Then smooth across rows.
    h, w = band.h, band.w
    row_color = np.zeros((h, 3), dtype=np.float32)
    for r in range(h):
        valid = band_pixels[r][band_pixels_mask[r]]
        if valid.size > 0:
            # Median is robust to text outliers — usually band color dominates
            row_color[r] = np.median(valid, axis=0)
        elif r > 0:
            row_color[r] = row_color[r - 1]
    # Smooth across rows
    row_color = cv2.GaussianBlur(row_color[:, None, :], (1, 21), 0).reshape(h, 3)
    return np.repeat(row_color[:, None, :], w, axis=1).astype(np.uint8)


def fill_text_in_bands(rgb: np.ndarray, text_mask: np.ndarray,
                       bands: list[Region], design_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each detected banner: estimate its colour and paint over any
    text-mask pixels inside that banner with the matched colour. Return
    (rgb_filled, remaining_text_mask) — pixels in bands are removed from
    the mask so the LaMa pass downstream doesn't re-touch them.
    """
    out = rgb.copy()
    remaining = text_mask.copy()
    for band in bands:
        if band.w <= 0 or band.h <= 0:
            continue
        fill = fit_band_color(rgb, band, design_mask)
        band_box_mask = np.zeros(rgb.shape[:2], dtype=bool)
        band_box_mask[band.y0:band.y1, band.x0:band.x1] = True
        apply = band_box_mask & (text_mask > 0)
        if not apply.any():
            continue
        # Apply fill (only on the masked text pixels)
        ys = np.where(apply)[0] - band.y0
        xs = np.where(apply)[1] - band.x0
        out[band.y0:band.y1, band.x0:band.x1][apply[band.y0:band.y1, band.x0:band.x1]] = fill[
            apply[band.y0:band.y1, band.x0:band.x1]
        ]
        remaining[band.y0:band.y1, band.x0:band.x1][apply[band.y0:band.y1, band.x0:band.x1]] = 0
    return out, remaining


# ---------- Text detection ----------

@dataclass
class TextRegion:
    polygon: np.ndarray  # (N, 2) int32 points in full-res coords
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    text: str
    confidence: float


def detect_text_regions(rgb: np.ndarray, design_mask: np.ndarray,
                        min_confidence: float = 0.10) -> list[TextRegion]:
    """OCR-based text region list. Returns each detected text element separately."""
    ocr = _ocr()
    H, W = rgb.shape[:2]
    target = 2400
    scale = min(1.0, target / max(H, W))
    if scale < 1.0:
        rgb_small = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    else:
        rgb_small = rgb
    results = ocr.readtext(rgb_small, paragraph=False, decoder="greedy")
    regions = []
    for bbox, text, conf in results:
        if conf < min_confidence:
            continue
        pts = (np.array(bbox, dtype=np.float32) / scale).astype(np.int32)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        # Skip if center is outside design region
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        if cy < 0 or cy >= H or cx < 0 or cx >= W:
            continue
        if design_mask[cy, cx] == 0:
            continue
        regions.append(TextRegion(pts, (int(x0), int(y0), int(x1), int(y1)), text, float(conf)))
    return regions


def detect_text_mask(rgb: np.ndarray, design_mask: np.ndarray,
                     min_confidence: float = 0.10,
                     dilate_px: int = 16) -> np.ndarray:
    """Compatibility wrapper — returns a single dilated mask of all text."""
    regions = detect_text_regions(rgb, design_mask, min_confidence)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for r in regions:
        cv2.fillPoly(mask, [r.polygon], 255)
    if dilate_px > 0:
        mask = cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8), iterations=1)
    mask[design_mask == 0] = 0
    return mask


# ---------- Per-region context-aware text fill ----------

def context_sample(rgb: np.ndarray, bbox: tuple[int, int, int, int],
                   design_mask: np.ndarray,
                   ring_factor: float = 0.6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample pixels in a 'ring' around `bbox` (outside the text but nearby).
    Returns (ring_pixels_HxWx3, ring_mask_HxW, full_outer_bbox_xyxy).

    The ring is used to estimate the background color/pattern around the text.
    """
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    pad_x = max(8, int(bw * ring_factor))
    pad_y = max(8, int(bh * ring_factor))
    H, W = rgb.shape[:2]
    ox0 = max(0, x0 - pad_x)
    oy0 = max(0, y0 - pad_y)
    ox1 = min(W, x1 + pad_x)
    oy1 = min(H, y1 + pad_y)
    outer = rgb[oy0:oy1, ox0:ox1]
    outer_design = design_mask[oy0:oy1, ox0:ox1] > 0
    # Ring = outer area, EXCLUDING the inner bbox, AND inside the design region
    ring_mask = outer_design.copy()
    ring_mask[max(0, y0 - oy0):min(outer.shape[0], y1 - oy0),
              max(0, x0 - ox0):min(outer.shape[1], x1 - ox0)] = False
    return outer, ring_mask, (ox0, oy0, ox1, oy1)


def is_uniform_background(ring_pixels: np.ndarray, ring_mask: np.ndarray,
                          variance_threshold: float = 18.0) -> bool:
    """Decide if the ring around the text is uniform (banner) or patterned."""
    px = ring_pixels[ring_mask]
    if px.size == 0:
        return False
    # Use per-row median to detect bands with gradient: a band has low variance
    # WITHIN a row, even if there's a slight gradient ACROSS rows. So sample
    # the std of per-row medians as the metric.
    h = ring_pixels.shape[0]
    row_meds = []
    for r in range(h):
        rm = ring_mask[r]
        if rm.sum() < 5:
            continue
        row_meds.append(np.median(ring_pixels[r][rm], axis=0))
    if len(row_meds) < 4:
        return False
    row_meds = np.array(row_meds)
    # Per-row variance (within the row) tells us if rows are uniform
    within_row_std = []
    for r in range(h):
        rm = ring_mask[r]
        if rm.sum() < 5:
            continue
        within_row_std.append(np.std(ring_pixels[r][rm], axis=0).mean())
    within_row_std = float(np.median(within_row_std)) if within_row_std else 999.0
    return within_row_std < variance_threshold


def build_row_gradient_fill(ring_pixels: np.ndarray, ring_mask: np.ndarray,
                            target_shape: tuple[int, int]) -> np.ndarray:
    """Build a fill that matches the per-row color of the surrounding ring.
    Returns (target_h, target_w, 3) uint8.

    `ring_pixels` covers the full outer area (text + surroundings); `ring_mask`
    excludes the text bbox itself. We compute the row-wise median (robust to text)
    and use it as the fill color for the corresponding row.
    """
    th, tw = target_shape
    h = ring_pixels.shape[0]
    row_color = np.zeros((h, 3), dtype=np.float32)
    last = np.array([255.0, 255.0, 255.0])
    for r in range(h):
        rm = ring_mask[r]
        if rm.sum() >= 3:
            last = np.median(ring_pixels[r][rm], axis=0)
        row_color[r] = last
    # Smooth across rows
    row_color = cv2.GaussianBlur(row_color[:, None, :], (1, 9), 0).reshape(h, 3)
    # row_color has `h` rows; target_shape may differ — resample if needed
    if h != th:
        row_color = cv2.resize(row_color[:, None, :], (1, th), interpolation=cv2.INTER_LINEAR).reshape(th, 3)
    return np.repeat(row_color[:, None, :], tw, axis=1).astype(np.uint8)


def _row_horizontal_sample(rgb: np.ndarray, bbox: tuple[int, int, int, int],
                           design_mask: np.ndarray, side_px: int = 200
                           ) -> tuple[np.ndarray, np.ndarray]:
    """For each row in the bbox, return the same-row left and right pixel arrays
    (within the design region) up to side_px wide on each side. This stays
    INSIDE the band — never crosses vertically into separator stripes.

    Returns (rows_color_estimate Hx3 float32, valid_row_mask H bool).
    """
    x0, y0, x1, y1 = bbox
    H, W = rgb.shape[:2]
    bh = y1 - y0
    row_color = np.zeros((bh, 3), dtype=np.float32)
    valid = np.zeros(bh, dtype=bool)
    lx0 = max(0, x0 - side_px)
    rx1 = min(W, x1 + side_px)
    for i in range(bh):
        y = y0 + i
        if y < 0 or y >= H:
            continue
        left = rgb[y, lx0:x0]
        right = rgb[y, x1:rx1]
        lmask = design_mask[y, lx0:x0] > 0
        rmask = design_mask[y, x1:rx1] > 0
        px = []
        if lmask.any():
            px.append(left[lmask])
        if rmask.any():
            px.append(right[rmask])
        if not px:
            continue
        all_px = np.concatenate(px, axis=0)
        # Robust per-row color = median of same-row neighbours
        row_color[i] = np.median(all_px, axis=0)
        valid[i] = True
    return row_color, valid


def _row_uniformity_score(rgb: np.ndarray, bbox: tuple[int, int, int, int],
                          design_mask: np.ndarray, side_px: int = 200) -> float:
    """Mean per-row std-dev of the same-row left+right neighbours. Low = uniform
    band (zero-artefact fill safe). High = patterned background (use LaMa)."""
    x0, y0, x1, y1 = bbox
    H, W = rgb.shape[:2]
    lx0 = max(0, x0 - side_px)
    rx1 = min(W, x1 + side_px)
    stds = []
    for y in range(y0, y1):
        if y < 0 or y >= H:
            continue
        left = rgb[y, lx0:x0]
        right = rgb[y, x1:rx1]
        lmask = design_mask[y, lx0:x0] > 0
        rmask = design_mask[y, x1:rx1] > 0
        px = []
        if lmask.any(): px.append(left[lmask])
        if rmask.any(): px.append(right[rmask])
        if not px: continue
        all_px = np.concatenate(px, axis=0)
        if all_px.shape[0] < 8: continue
        stds.append(np.std(all_px, axis=0).mean())
    if not stds:
        return 999.0
    return float(np.median(stds))


def remove_text_smart(rgb: np.ndarray, design_mask: np.ndarray,
                      regions: list[TextRegion],
                      bbox_pad_px: int = 22,
                      uniformity_threshold: float = 22.0,
                      verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """For each text region, sample horizontal neighbours (same row, left+right).
    If those neighbours are uniform per-row → fill the bbox row-by-row with that
    per-row colour (handles solid AND gradient bands perfectly). Otherwise mark
    the region for LaMa.

    Returns (rgb_filled, leftover_lama_mask).
    """
    out = rgb.copy()
    leftover = np.zeros(rgb.shape[:2], dtype=np.uint8)
    H, W = rgb.shape[:2]
    band_count = 0
    lama_count = 0
    for tr in regions:
        x0, y0, x1, y1 = tr.bbox
        x0 = max(0, x0 - bbox_pad_px); y0 = max(0, y0 - bbox_pad_px)
        x1 = min(W, x1 + bbox_pad_px); y1 = min(H, y1 + bbox_pad_px)
        score = _row_uniformity_score(out, (x0, y0, x1, y1), design_mask)
        if score < uniformity_threshold:
            # Build a per-row color from same-row neighbours and fill
            row_color, valid = _row_horizontal_sample(out, (x0, y0, x1, y1), design_mask)
            # Forward-fill any invalid rows
            last = None
            for i in range(row_color.shape[0]):
                if valid[i]:
                    last = row_color[i]
                elif last is not None:
                    row_color[i] = last
            # Light vertical smoothing so anti-aliased text edges blend (3px kernel)
            row_color = cv2.GaussianBlur(row_color[:, None, :], (1, 3), 0).reshape(row_color.shape[0], 3)
            tgt_design = (design_mask[y0:y1, x0:x1] > 0)
            fill_row = np.repeat(row_color[:, None, :], x1 - x0, axis=1).astype(np.uint8)
            out_slice = out[y0:y1, x0:x1]
            out_slice[tgt_design] = fill_row[tgt_design]
            out[y0:y1, x0:x1] = out_slice
            band_count += 1
            if verbose:
                print(f"      band-fill '{tr.text}' (row-std={score:.1f}) @ ({x0},{y0})-({x1},{y1})")
        else:
            cv2.fillPoly(leftover, [tr.polygon], 255)
            lama_count += 1
            if verbose:
                print(f"      LaMa-queue '{tr.text}' (row-std={score:.1f}) @ ({x0},{y0})-({x1},{y1})")
    if (leftover > 0).any():
        leftover = cv2.dilate(leftover, np.ones((22, 22), np.uint8), iterations=1)
        leftover[design_mask == 0] = 0
    if verbose:
        print(f"    text removal: {band_count} band-filled, {lama_count} queued for LaMa")
    return out, leftover


# ---------- Cropping & export ----------

def crop_to_bbox(rgb: np.ndarray, design_mask: np.ndarray, bleed_px: int = 0) -> np.ndarray:
    ys, xs = np.where(design_mask)
    y0 = max(0, ys.min() - bleed_px)
    y1 = min(rgb.shape[0], ys.max() + 1 + bleed_px)
    x0 = max(0, xs.min() - bleed_px)
    x1 = min(rgb.shape[1], xs.max() + 1 + bleed_px)
    return rgb[y0:y1, x0:x1]


def save_print_outputs(rgb: np.ndarray, outdir: str, base: str, dpi: int = 300) -> dict:
    os.makedirs(outdir, exist_ok=True)
    h, w = rgb.shape[:2]
    width_mm = w / dpi * 25.4
    height_mm = h / dpi * 25.4
    png_path = os.path.join(outdir, f"{base}.png")
    pdf_path = os.path.join(outdir, f"{base}.pdf")
    Image.fromarray(rgb).save(png_path, dpi=(dpi, dpi))
    Image.fromarray(rgb).save(pdf_path, "PDF", resolution=dpi)
    return {"png": png_path, "pdf": pdf_path, "size_px": (w, h),
            "size_mm": (round(width_mm, 1), round(height_mm, 1))}


# ---------- Multi-size export with generative outpainting ----------

PRESET_PRINT_SIZES = {
    "16x20":      (16.0,    20.0),
    "18x24":      (18.0,    24.0),
    "22x29":      (22.0,    29.0),
    "24x32.125":  (24.0,    32.125),
}


def export_at_print_size(
    rgb: np.ndarray,
    width_in: float,
    height_in: float,
    dpi: int = 300,
    background_extend: str = "outpaint",
    use_openai: bool = False,
    openai_quality: str = "high",
) -> np.ndarray:
    """Resize+extend the recovered design to a target physical print size at DPI.

    `background_extend`:
      - "outpaint" : generative LaMa outpaint to fill the new aspect ratio
      - "reflect"  : edge-reflection (no AI)
      - "edge"     : edge-replication (mirrors solid borders well)
    """
    target_w = int(round(width_in * dpi))
    target_h = int(round(height_in * dpi))
    src_h, src_w = rgb.shape[:2]
    target_aspect = target_w / target_h
    src_aspect = src_w / src_h

    # Fit the source inside the target preserving aspect, then extend the
    # remaining strips. We don't crop the original artwork.
    if abs(target_aspect - src_aspect) < 1e-3:
        return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Scale source so its smaller-than-target dimension matches.
    if target_aspect > src_aspect:
        # target is wider -> match heights
        scale = target_h / src_h
        new_w = int(round(src_w * scale))
        new_h = target_h
    else:
        scale = target_w / src_w
        new_w = target_w
        new_h = int(round(src_h * scale))
    src_scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    if background_extend in ("reflect", "edge"):
        border_mode = cv2.BORDER_REFLECT_101 if background_extend == "reflect" else cv2.BORDER_REPLICATE
        return cv2.copyMakeBorder(
            src_scaled, pad_y, target_h - new_h - pad_y,
            pad_x, target_w - new_w - pad_x, border_mode,
        )

    # Outpaint: start with a REFLECTED canvas (gives the inpainter real edge
    # content to extend, not black), then mask the reflected border and refine.
    canvas = cv2.copyMakeBorder(
        src_scaled, pad_y, target_h - new_h - pad_y,
        pad_x, target_w - new_w - pad_x, cv2.BORDER_REFLECT_101,
    )
    out_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    out_mask[:pad_y, :] = 255
    out_mask[pad_y + new_h:, :] = 255
    out_mask[:, :pad_x] = 255
    out_mask[:, pad_x + new_w:] = 255
    if use_openai:
        from openai_inpaint import openai_inpaint_hq
        canvas = openai_inpaint_hq(canvas, out_mask, context_pad_px=192,
                                   quality=openai_quality)
    else:
        canvas = lama_inpaint_hq(canvas, out_mask, context_pad_px=192, max_window_dim=2000)
    return canvas


# ---------- Top-level orchestration ----------

def process_one(
    template_alpha: np.ndarray,
    mockup_path: str,
    outdir: str,
    remove_text: bool = True,
    keep_bottom_brand: bool = False,
    dpi: int = 300,
    print_sizes: list[str] | None = None,
    use_openai: bool = False,
    openai_quality: str = "high",
    verbose: bool = True,
):
    base = os.path.splitext(os.path.basename(mockup_path))[0]
    rgb = load_mockup_rgb(mockup_path)
    h, w = rgb.shape[:2]
    tpl = resize_alpha_to(template_alpha, h, w)
    design_mask = (tpl == 0).astype(np.uint8)
    hardware_mask = cv2.dilate((tpl > 0).astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)

    # Pick inpainter: OpenAI gpt-image-1 if requested + key present, else LaMa
    if use_openai:
        from openai_inpaint import openai_inpaint_hq, DEFAULT_PROMPT, TEXT_REMOVAL_PROMPT
        def inpaint(rgb_, mask_, prompt=DEFAULT_PROMPT, **kw):
            return openai_inpaint_hq(rgb_, mask_, quality=openai_quality, prompt=prompt,
                                     context_pad_px=kw.get("context_pad_px", 256))
        engine_name = f"OpenAI gpt-image-1 ({openai_quality})"
    else:
        def inpaint(rgb_, mask_, prompt=None, **kw):
            return lama_inpaint_hq(rgb_, mask_,
                                   context_pad_px=kw.get("context_pad_px", 256),
                                   max_window_dim=kw.get("max_window_dim", 2000))
        engine_name = "LaMa (local)"

    # 1. Hardware inpaint
    if verbose:
        print(f"  [hardware] {engine_name}")
    rgb_clean = inpaint(rgb, hardware_mask, prompt=
        "Seamlessly continue the surrounding pattern across the transparent area. "
        "Match colours, lines and style. No new objects, text, or borders.",
        context_pad_px=256)

    # Save "design with text"
    cropped_keep = crop_to_bbox(rgb_clean, design_mask)
    meta_keep = save_print_outputs(cropped_keep, outdir, f"{base}_design", dpi=dpi)
    if verbose:
        print(f"  -> {meta_keep['png']}  {meta_keep['size_px']} px / {meta_keep['size_mm']} mm")

    # 2. Text removal (default ON)
    if remove_text:
        if verbose:
            print(f"  [text] per-region context-aware removal")
        regions = detect_text_regions(rgb_clean, design_mask, min_confidence=0.08)
        if keep_bottom_brand:
            ys = np.where(design_mask)[0]
            y_bottom = ys.max() - int(0.08 * (ys.max() - ys.min()))
            regions = [r for r in regions if (r.bbox[1] + r.bbox[3]) / 2 < y_bottom]
        if verbose:
            print(f"    found {len(regions)} text region(s): " +
                  ", ".join(f"{r.text!r}(c={r.confidence:.2f})" for r in regions[:8]))
        rgb_blank, lama_text_mask = remove_text_smart(rgb_clean, design_mask, regions, verbose=verbose)
        if (lama_text_mask > 0).any():
            if verbose:
                n = int((lama_text_mask > 0).sum())
                print(f"    {engine_name}-inpaint queued non-band text ({n:,} px)")
            rgb_blank = inpaint(
                rgb_blank, lama_text_mask,
                prompt="Remove the text inside the transparent area. Fill it "
                       "with the matching surrounding background — same colour, "
                       "same pattern, same gradient. No new text or graphics.",
                context_pad_px=256, max_window_dim=1800,
            )

        cropped_blank = crop_to_bbox(rgb_blank, design_mask)
        meta_blank = save_print_outputs(cropped_blank, outdir, f"{base}_design_blank", dpi=dpi)
        if verbose:
            print(f"  -> {meta_blank['png']}  ({len(regions)} text regions removed)")
        primary_for_resizing = cropped_blank
    else:
        primary_for_resizing = cropped_keep

    # 3. Multi-size export
    if print_sizes:
        size_dir = os.path.join(outdir, f"{base}_print_sizes")
        os.makedirs(size_dir, exist_ok=True)
        for size_label in print_sizes:
            if size_label in PRESET_PRINT_SIZES:
                w_in, h_in = PRESET_PRINT_SIZES[size_label]
            else:
                parts = size_label.lower().replace("in", "").split("x")
                w_in, h_in = float(parts[0]), float(parts[1])
            if verbose:
                print(f"  [size] {size_label}  -> {int(w_in*dpi)}x{int(h_in*dpi)}px outpaint")
            sized = export_at_print_size(primary_for_resizing, w_in, h_in, dpi=dpi,
                                         background_extend="outpaint",
                                         use_openai=use_openai,
                                         openai_quality=openai_quality)
            name = f"{base}_{size_label.replace('.', '_')}_in_{dpi}dpi"
            Image.fromarray(sized).save(os.path.join(size_dir, f"{name}.png"), dpi=(dpi, dpi))
            Image.fromarray(sized).save(os.path.join(size_dir, f"{name}.pdf"), "PDF", resolution=dpi)
            if verbose:
                print(f"    -> {os.path.join(size_dir, name + '.png')}")


def main():
    p = argparse.ArgumentParser(description="Template-based design recovery v3")
    p.add_argument("--template", required=True)
    p.add_argument("--mockup", required=True, action="append")
    p.add_argument("--out", default="runs/v3")
    p.add_argument("--no-remove-text", dest="remove_text", action="store_false",
                   help="Keep all text (default: remove all text including logos)")
    p.add_argument("--keep-bottom-brand", action="store_true",
                   help="Preserve the bottom 8%% strip (small AMAIREE brand mark)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--print-sizes", default="",
                   help="Comma-separated print sizes, e.g. 16x20,18x24,22x29,24x32.125")
    p.add_argument("--use-openai", action="store_true",
                   help="Use OpenAI gpt-image-1 instead of local LaMa. "
                        "Requires OPENAI_API_KEY env var. Per-call billing applies.")
    p.add_argument("--openai-quality", default="high", choices=["low", "medium", "high"],
                   help="gpt-image-1 quality (cost scales). Default high.")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading template alpha from {args.template}...")
    tpl = load_template_alpha(args.template)
    print(f"  template: {tpl.shape}, design pixels: {(tpl == 0).sum():,}")

    sizes = [s.strip() for s in args.print_sizes.split(",") if s.strip()] if args.print_sizes else None

    for m in args.mockup:
        print(f"\n{m}")
        process_one(tpl, m, args.out,
                    remove_text=args.remove_text,
                    keep_bottom_brand=args.keep_bottom_brand,
                    dpi=args.dpi,
                    print_sizes=sizes,
                    use_openai=args.use_openai,
                    openai_quality=args.openai_quality)


if __name__ == "__main__":
    main()
