"""Drop-in replacement for `lama_inpaint_hq` that uses OpenAI's image edit API
(gpt-image-1) instead of LaMa.

When you have an OPENAI_API_KEY set, this produces noticeably better results
than LaMa for complex patterns / text removal / hardware fill — the model
understands what a leopard / planet / floral pattern *should* look like, not
just "blend nearby pixels".

Cost is per-call (currently ~$0.02–0.19 per image at 1024px depending on
quality tier). LaMa stays as the default offline fallback.

Architecture:
  - For each connected mask region, crop a padded window around it.
  - Resize that window to a square supported by the API
    (1024 / 1536 / 1024x1536), build an RGBA image where pixels-to-edit are
    transparent, send to /v1/images/edits with a prompt that tells the model
    to seamlessly continue the surrounding design.
  - Resize the result back, composite at the masked pixels.

Run standalone:
  export OPENAI_API_KEY=sk-...
  python openai_inpaint.py --image mockup.png --mask mask.png --out result.png

Use as a module inside template_extract_v3 by replacing `lama_inpaint_hq` with
`openai_inpaint_hq`.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


DEFAULT_MODEL = "gpt-image-1"
DEFAULT_QUALITY = "high"
DEFAULT_PROMPT = (
    "Seamlessly continue the existing design across the transparent area. "
    "Match the surrounding pattern, colours, line weights and style exactly. "
    "Do not add new objects, text, watermarks, or borders. Photographic "
    "fidelity to the surrounding image is required."
)
TEXT_REMOVAL_PROMPT = (
    "Remove the text in the transparent area. Fill it with the matching "
    "background — same colour, same pattern, same gradient as the surrounding "
    "pixels. Do not invent new text or graphics."
)


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


def _square_size_for(crop_w: int, crop_h: int) -> tuple[str, int]:
    """Pick the best API output size for the crop's aspect ratio.

    gpt-image-1 accepts: 1024x1024, 1024x1536, 1536x1024, auto.
    """
    ratio = crop_w / crop_h
    if ratio > 1.3:
        return "1536x1024", 1536
    if ratio < 0.77:
        return "1024x1536", 1536
    return "1024x1024", 1024


def _make_rgba_for_edit(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Build an RGBA PIL Image where pixels-to-edit (mask > 0) have alpha = 0
    and pixels to keep have alpha = 255."""
    h, w = rgb.shape[:2]
    alpha = np.where(mask > 0, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def _make_mask_png(mask: np.ndarray) -> Image.Image:
    """Build a PIL Image to send as the `mask` parameter — transparent = edit,
    opaque white = keep. (This is the separate-mask variant for the API.)"""
    alpha = np.where(mask > 0, 0, 255).astype(np.uint8)
    h, w = mask.shape[:2]
    rgba = np.dstack([np.full((h, w, 3), 255, dtype=np.uint8), alpha])
    return Image.fromarray(rgba, mode="RGBA")


def _resize_pil(img: Image.Image, size_wh: tuple[int, int]) -> Image.Image:
    return img.resize(size_wh, Image.LANCZOS)


def _call_openai_edit(
    client, image_pil: Image.Image, mask_pil: Image.Image, prompt: str,
    size: str = "1024x1024", quality: str = "high", model: str = "gpt-image-1",
) -> Image.Image:
    """Single API call. Returns the result as a PIL Image."""
    buf_img = io.BytesIO()
    buf_msk = io.BytesIO()
    image_pil.save(buf_img, format="PNG")
    mask_pil.save(buf_msk, format="PNG")
    buf_img.seek(0); buf_msk.seek(0)
    buf_img.name = "image.png"
    buf_msk.name = "mask.png"
    resp = client.images.edit(
        model=model,
        image=buf_img,
        mask=buf_msk,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
    )
    b64 = resp.data[0].b64_json
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def openai_inpaint_hq(
    rgb: np.ndarray,
    mask: np.ndarray,
    context_pad_px: int = 256,
    prompt: str = DEFAULT_PROMPT,
    quality: str = DEFAULT_QUALITY,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    on_progress=None,
) -> np.ndarray:
    """Inpaint mask regions using OpenAI's image edit API.

    Drop-in replacement for the LaMa variant — same signature aside from `prompt`,
    `quality`, `model`, `api_key`. Falls back to raising if no key is set.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Either export the env var or pass api_key="
        )
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    h, w = rgb.shape[:2]
    out = rgb.copy()
    regions = _mask_regions(mask)
    for i, r in enumerate(regions):
        wnd = r.pad(context_pad_px, w, h)
        crop_rgb = out[wnd.y0:wnd.y1, wnd.x0:wnd.x1].copy()
        crop_mask = mask[wnd.y0:wnd.y1, wnd.x0:wnd.x1].copy()
        ch, cw = crop_rgb.shape[:2]
        size_str, max_dim = _square_size_for(cw, ch)
        api_w, api_h = map(int, size_str.split("x"))

        image_pil = _make_rgba_for_edit(crop_rgb, crop_mask)
        mask_pil = _make_mask_png(crop_mask)
        image_pil = _resize_pil(image_pil, (api_w, api_h))
        mask_pil = _resize_pil(mask_pil, (api_w, api_h))

        t0 = time.time()
        result_pil = _call_openai_edit(
            client, image_pil, mask_pil, prompt=prompt,
            size=size_str, quality=quality, model=model,
        )
        result = np.array(result_pil)
        if result.shape[:2] != (ch, cw):
            result = cv2.resize(result, (cw, ch), interpolation=cv2.INTER_LANCZOS4)

        # Soft alpha blend at mask edge so seams are invisible
        soft = cv2.GaussianBlur(crop_mask.astype(np.float32), (9, 9), 0) / 255.0
        soft3 = soft[..., None]
        merged = (result.astype(np.float32) * soft3 +
                  out[wnd.y0:wnd.y1, wnd.x0:wnd.x1].astype(np.float32) * (1 - soft3))
        out[wnd.y0:wnd.y1, wnd.x0:wnd.x1] = np.clip(merged, 0, 255).astype(np.uint8)

        if on_progress:
            on_progress(i + 1, len(regions), wnd, time.time() - t0)
    return out


def main():
    p = argparse.ArgumentParser(description="OpenAI image-edit-API inpaint")
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True, help="White-on-black mask PNG (white = edit, black = keep)")
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--quality", default=DEFAULT_QUALITY, choices=["low", "medium", "high"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=None)
    args = p.parse_args()

    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    out = openai_inpaint_hq(
        rgb, mask, prompt=args.prompt, quality=args.quality, model=args.model,
        api_key=args.api_key,
        on_progress=lambda i, n, w, dt: print(f"region {i}/{n}: {w.w}x{w.h}px in {dt:.1f}s"),
    )
    Image.fromarray(out).save(args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
