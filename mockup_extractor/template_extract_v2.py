"""Template-based design recovery v2 — adds LaMa (generative) inpainting and OCR-driven text removal.

Pipeline:
  1. Mask the finished mockup with the template PSD's alpha channel
     -> lossless lift of design pixels.
  2. Inpaint hardware-occluded regions (corner caps, wheels, zipper) with LaMa,
     processing each mask region in a cropped window for speed on CPU.
  3. (Optional) Detect text via EasyOCR; LaMa-inpaint over text regions to
     produce a 'blank' design template you can re-personalize.
  4. Crop to the design's bounding box; emit PNG/PDF/SVG-embedded.

Run:
    python template_extract_v2.py \\
        --template psd_samples/cabin-suitcase-300dpi.psd \\
        --mockup /path/to/mockup.png [--mockup ...] \\
        --out runs/v2/ \\
        [--remove-text] \\
        [--keep-bottom-brand]
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


# Lazy imports for the heavy ML deps so module import stays cheap.
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


@dataclass
class Region:
    x0: int; y0: int; x1: int; y1: int

    @property
    def w(self): return self.x1 - self.x0
    @property
    def h(self): return self.y1 - self.y0

    def pad(self, px: int, max_w: int, max_h: int) -> "Region":
        return Region(
            max(0, self.x0 - px), max(0, self.y0 - px),
            min(max_w, self.x1 + px), min(max_h, self.y1 + px),
        )


def _mask_regions(mask: np.ndarray, min_area: int = 50) -> list[Region]:
    """Return bounding-box regions of connected components in a binary mask."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append(Region(int(x), int(y), int(x + w), int(y + h)))
    return out


def inpaint_with_lama(
    rgb: np.ndarray,
    mask: np.ndarray,
    pad_px: int = 64,
    max_window_dim: int = 1024,
    on_region=None,
) -> np.ndarray:
    """Inpaint each mask region in a small cropped window, using LaMa.

    For each connected region we:
      - crop a window around the region (with padding for context),
      - downscale if larger than max_window_dim,
      - run LaMa,
      - upscale and paste back. Hard-edge blending only on the masked pixels.
    """
    h, w = rgb.shape[:2]
    out = rgb.copy()
    regions = _mask_regions(mask)
    lama = _lama()
    for i, r in enumerate(regions):
        wnd = r.pad(pad_px, w, h)
        crop_rgb = out[wnd.y0:wnd.y1, wnd.x0:wnd.x1].copy()
        crop_mask = mask[wnd.y0:wnd.y1, wnd.x0:wnd.x1].copy()
        # Downscale if huge
        ch, cw = crop_rgb.shape[:2]
        scale = 1.0
        if max(ch, cw) > max_window_dim:
            scale = max_window_dim / max(ch, cw)
            new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
            crop_rgb_lo = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            crop_mask_lo = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            crop_rgb_lo = crop_rgb
            crop_mask_lo = crop_mask
        t0 = time.time()
        inpainted_lo = lama(Image.fromarray(crop_rgb_lo), Image.fromarray(crop_mask_lo))
        inpainted_lo = np.array(inpainted_lo)
        if scale < 1.0:
            inpainted = cv2.resize(inpainted_lo, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        else:
            # LaMa pads to a modulo of 8; crop back to original window size.
            inpainted = inpainted_lo[:ch, :cw]
        # Composite back: only replace masked pixels (preserve surrounding context exactly)
        m3 = (crop_mask > 0)[..., None].repeat(3, axis=2)
        out[wnd.y0:wnd.y1, wnd.x0:wnd.x1] = np.where(m3, inpainted, out[wnd.y0:wnd.y1, wnd.x0:wnd.x1])
        if on_region:
            on_region(i + 1, len(regions), wnd, time.time() - t0)
    return out


def detect_text_mask(
    rgb: np.ndarray,
    design_mask: np.ndarray,
    min_confidence: float = 0.2,
    dilate_px: int = 14,
    keep_bottom_brand: bool = False,
) -> np.ndarray:
    """EasyOCR detects text; we return a dilated mask of all detected text regions."""
    ocr = _ocr()
    # Work on a manageable size for OCR (it doesn't need 5k width)
    H, W = rgb.shape[:2]
    target = 2400
    scale = min(1.0, target / max(H, W))
    if scale < 1.0:
        rgb_small = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    else:
        rgb_small = rgb
    results = ocr.readtext(rgb_small, paragraph=False)

    mask = np.zeros((H, W), dtype=np.uint8)
    keep_thresh_y = int(H * 0.92) if keep_bottom_brand else H + 1
    for bbox, text, conf in results:
        if conf < min_confidence:
            continue
        pts = np.array(bbox, dtype=np.float32) / scale
        pts_i = pts.astype(np.int32)
        ymid = pts_i[:, 1].mean()
        # Optional: keep the small "AMAIREE" brand mark at the very bottom
        if keep_bottom_brand and ymid > keep_thresh_y:
            continue
        cv2.fillPoly(mask, [pts_i], 255)
    if dilate_px > 0:
        mask = cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8), iterations=1)
    # Stay inside the design area only — don't try to inpaint over hardware regions.
    mask[design_mask == 0] = 0
    return mask


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
    svg_path = os.path.join(outdir, f"{base}-embedded.svg")
    Image.fromarray(rgb).save(png_path, dpi=(dpi, dpi))
    Image.fromarray(rgb).save(pdf_path, "PDF", resolution=dpi)
    import base64
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width_mm:.2f}mm" height="{height_mm:.2f}mm" '
            f'viewBox="0 0 {width_mm:.2f} {height_mm:.2f}">\n'
            f'  <image x="0" y="0" width="{width_mm:.2f}" height="{height_mm:.2f}" '
            f'preserveAspectRatio="none" xlink:href="data:image/png;base64,{b64}"/>\n'
            f'</svg>\n'
        )
    return {"png": png_path, "pdf": pdf_path, "svg": svg_path,
            "size_px": (w, h), "size_mm": (round(width_mm, 1), round(height_mm, 1))}


def process_one(
    template_alpha: np.ndarray,
    mockup_path: str,
    outdir: str,
    remove_text: bool = False,
    keep_bottom_brand: bool = False,
    dpi: int = 300,
    verbose: bool = True,
):
    base = os.path.splitext(os.path.basename(mockup_path))[0]
    rgb = load_mockup_rgb(mockup_path)
    if rgb.shape[:2] != template_alpha.shape:
        raise ValueError(f"{mockup_path}: size {rgb.shape[:2]} != template {template_alpha.shape}")

    # 1. Hardware mask
    hardware_mask = (template_alpha > 0).astype(np.uint8) * 255
    hardware_mask = cv2.dilate(hardware_mask, np.ones((5, 5), np.uint8), iterations=1)
    design_mask = (template_alpha == 0).astype(np.uint8)

    # 2. LaMa hardware inpaint
    if verbose:
        print(f"  LaMa-inpainting hardware regions on {base}...")
    rgb_clean = inpaint_with_lama(
        rgb, hardware_mask,
        on_region=(lambda i, n, w, dt: print(f"    region {i}/{n}: {w.w}x{w.h}px in {dt:.1f}s")) if verbose else None,
    )

    # Save the no-text version too
    cropped_with_text = crop_to_bbox(rgb_clean, design_mask)
    meta_with_text = save_print_outputs(cropped_with_text, outdir, f"{base}_design", dpi=dpi)
    if verbose:
        print(f"  -> {meta_with_text['png']}  {meta_with_text['size_px']} px / {meta_with_text['size_mm']} mm")

    if remove_text:
        if verbose:
            print(f"  Detecting text on {base}...")
        text_mask = detect_text_mask(rgb_clean, design_mask, keep_bottom_brand=keep_bottom_brand)
        text_pixels = int((text_mask > 0).sum())
        if verbose:
            print(f"  text mask: {text_pixels:,} px")
        if text_pixels > 0:
            rgb_blank = inpaint_with_lama(
                rgb_clean, text_mask,
                on_region=(lambda i, n, w, dt: print(f"    text region {i}/{n}: {w.w}x{w.h}px in {dt:.1f}s")) if verbose else None,
            )
            cropped_blank = crop_to_bbox(rgb_blank, design_mask)
            meta_blank = save_print_outputs(cropped_blank, outdir, f"{base}_design_blank", dpi=dpi)
            if verbose:
                print(f"  -> {meta_blank['png']}  (text removed)")

            # Also save the text mask + a visual diff for debugging
            cv2.imwrite(os.path.join(outdir, f"{base}_text_mask.png"), text_mask)


def main():
    p = argparse.ArgumentParser(description="Template-based design recovery v2 (LaMa + OCR)")
    p.add_argument("--template", required=True)
    p.add_argument("--mockup", required=True, action="append")
    p.add_argument("--out", default="runs/v2")
    p.add_argument("--remove-text", action="store_true")
    p.add_argument("--keep-bottom-brand", action="store_true",
                   help="Don't remove text in the bottom 8%% (small AMAIREE brand mark)")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading template alpha from {args.template}...")
    tpl = load_template_alpha(args.template)
    print(f"  template alpha: {tpl.shape}  design pixels: {(tpl == 0).sum():,}")

    for m in args.mockup:
        print(f"\n{m}")
        process_one(tpl, m, args.out,
                    remove_text=args.remove_text,
                    keep_bottom_brand=args.keep_bottom_brand,
                    dpi=args.dpi)


if __name__ == "__main__":
    main()
