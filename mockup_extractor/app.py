"""Gradio web app: extract a design from a product mockup and emit print-ready files."""

from __future__ import annotations

import os
import tempfile
import zipfile

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


CORNER_LABELS = ("TL", "TR", "BR", "BL")
PREVIEW_MAX_PX = 1400


def _font():
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, 28)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_corners(image: np.ndarray, corners: list[tuple[int, int]]) -> np.ndarray:
    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font()
    r = max(8, int(min(img.width, img.height) * 0.008))
    for i, (x, y) in enumerate(corners):
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 32, 32), width=4)
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 32, 32))
        label = CORNER_LABELS[i] if i < 4 else str(i + 1)
        draw.text((x + r + 6, y - r - 6), label, fill=(255, 32, 32), font=font)
    if len(corners) >= 2:
        for i in range(len(corners)):
            if i + 1 >= len(corners) and len(corners) < 4:
                break
            x0, y0 = corners[i]
            x1, y1 = corners[(i + 1) % 4 if len(corners) == 4 else i + 1]
            draw.line([(x0, y0), (x1, y1)], fill=(255, 32, 32), width=3)
    return np.array(img)


def on_upload(image):
    if image is None:
        return [], None, None, "Upload a mockup to begin.", None
    return [], image, image, "Click the **4 corners** of the design panel: TL → TR → BR → BL.", None


def on_click(raw, corners, evt: gr.SelectData):
    if raw is None:
        return corners, None, "Upload a mockup first."
    x, y = int(evt.index[0]), int(evt.index[1])
    if len(corners) >= 4:
        # Replace nearest corner so re-clicking refines the polygon.
        dists = [(x - cx) ** 2 + (y - cy) ** 2 for cx, cy in corners]
        idx = dists.index(min(dists))
        corners = list(corners)
        corners[idx] = (x, y)
        msg = f"Refined corner {CORNER_LABELS[idx]} → ({x}, {y}). Click anywhere to refine the closest corner."
    else:
        corners = list(corners) + [(x, y)]
        if len(corners) < 4:
            msg = f"Corner {CORNER_LABELS[len(corners) - 1]} set. {4 - len(corners)} to go."
        else:
            msg = "All 4 corners set. Adjust output settings and click **Generate**."
    annotated = _draw_corners(raw, corners)
    return corners, annotated, msg


def on_reset(raw):
    if raw is None:
        return [], None, "Upload a mockup to begin."
    return [], raw, "Corners cleared. Click the 4 corners of the design panel."


def on_preview(raw, corners, width_mm, height_mm):
    if raw is None or len(corners) != 4:
        return None
    # Preview at modest resolution for speed; full DPI only on generate.
    aspect = float(width_mm) / float(height_mm)
    if aspect >= 1:
        pw = PREVIEW_MAX_PX
        ph = int(PREVIEW_MAX_PX / aspect)
    else:
        ph = PREVIEW_MAX_PX
        pw = int(PREVIEW_MAX_PX * aspect)
    return perspective_unwarp(raw, corners, pw, ph)


def on_generate(
    raw,
    corners,
    width_mm,
    height_mm,
    dpi,
    brightness,
    contrast,
    gamma,
    inpaint_hardware,
    do_trace,
    trace_precision,
):
    if raw is None:
        return None, "Upload a mockup first."
    if len(corners) != 4:
        return None, f"Need 4 corners, got {len(corners)}."
    width_mm = float(width_mm)
    height_mm = float(height_mm)
    dpi = int(dpi)
    if width_mm <= 0 or height_mm <= 0:
        return None, "Width and height must be positive."

    px_w, px_h = physical_to_pixels(width_mm, height_mm, dpi)
    flat = perspective_unwarp(raw, corners, px_w, px_h)

    if inpaint_hardware:
        flat, _ = mask_dark_hardware(flat)

    if brightness != 0 or contrast != 1.0 or gamma != 1.0:
        flat = apply_levels(flat, brightness=brightness, contrast=contrast, gamma=gamma)

    outdir = tempfile.mkdtemp(prefix="mockup_extract_")
    png_path = os.path.join(outdir, "design.png")
    pdf_path = os.path.join(outdir, "design.pdf")
    embed_svg_path = os.path.join(outdir, "design-embedded.svg")
    traced_svg_path = os.path.join(outdir, "design-traced.svg")

    save_png(flat, png_path, dpi=dpi)
    save_pdf(flat, pdf_path, width_mm=width_mm, height_mm=height_mm)
    save_embedded_svg(flat, embed_svg_path, width_mm=width_mm, height_mm=height_mm)

    files = [png_path, pdf_path, embed_svg_path]
    if do_trace:
        save_traced_svg(flat, traced_svg_path, color_precision=int(trace_precision))
        files.append(traced_svg_path)

    zip_path = os.path.join(outdir, "design-print-ready.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=os.path.basename(p))

    summary_lines = [
        f"**Generated** at `{outdir}`:",
        f"- `design.png` — {px_w}×{px_h} px @ {dpi} DPI ({width_mm}×{height_mm} mm)",
        f"- `design.pdf` — single-page, sized to {width_mm}×{height_mm} mm",
        f"- `design-embedded.svg` — raster embedded in a vector container",
    ]
    if do_trace:
        summary_lines.append(
            f"- `design-traced.svg` — true vector paths (vtracer, color_precision={int(trace_precision)})"
        )
    return zip_path, "\n".join(summary_lines)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Mockup Design Extractor") as demo:
        gr.Markdown(
            "# Mockup Design Extractor\n"
            "Reverse a product mockup: upload the rendered mockup, click the 4 corners "
            "of the printed design panel, and download print-ready PNG / PDF / SVG."
        )

        corners_state = gr.State([])
        raw_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                img_input = gr.Image(
                    label="Mockup (click 4 corners: TL → TR → BR → BL)",
                    type="numpy",
                    interactive=True,
                    height=600,
                )
                with gr.Row():
                    reset_btn = gr.Button("Reset corners")
                    preview_btn = gr.Button("Refresh preview")
                status_md = gr.Markdown("Upload a mockup to begin.")

            with gr.Column(scale=2):
                preview_img = gr.Image(
                    label="Unwarped design preview",
                    type="numpy",
                    interactive=False,
                    height=400,
                )
                with gr.Group():
                    gr.Markdown("**Print size**")
                    with gr.Row():
                        width_mm = gr.Number(label="Width (mm)", value=300, precision=1)
                        height_mm = gr.Number(label="Height (mm)", value=450, precision=1)
                    dpi = gr.Slider(label="DPI", minimum=72, maximum=600, step=1, value=300)

                with gr.Accordion("Color correction (optional)", open=False):
                    brightness = gr.Slider(label="Brightness", minimum=-100, maximum=100, value=0)
                    contrast = gr.Slider(label="Contrast", minimum=0.5, maximum=2.0, step=0.05, value=1.0)
                    gamma = gr.Slider(label="Gamma", minimum=0.3, maximum=3.0, step=0.05, value=1.0)
                    inpaint_hardware = gr.Checkbox(
                        label="Inpaint dark hardware (corner caps / zipper bleed-in)",
                        value=False,
                    )

                with gr.Accordion("Vector trace", open=True):
                    do_trace = gr.Checkbox(label="Also produce a true vector trace (SVG paths)", value=True)
                    trace_precision = gr.Slider(
                        label="Color precision (higher = more colors, larger SVG)",
                        minimum=2, maximum=8, step=1, value=6,
                    )

                gen_btn = gr.Button("Generate print-ready files", variant="primary", size="lg")
                download = gr.File(label="Download ZIP")
                result_md = gr.Markdown()

        img_input.upload(
            on_upload,
            inputs=img_input,
            outputs=[corners_state, raw_state, img_input, status_md, preview_img],
        )
        img_input.select(
            on_click,
            inputs=[raw_state, corners_state],
            outputs=[corners_state, img_input, status_md],
        )
        reset_btn.click(
            on_reset,
            inputs=raw_state,
            outputs=[corners_state, img_input, status_md],
        )
        preview_btn.click(
            on_preview,
            inputs=[raw_state, corners_state, width_mm, height_mm],
            outputs=preview_img,
        )
        # Auto-preview after each corner click (cheap at preview resolution).
        img_input.select(
            on_preview,
            inputs=[raw_state, corners_state, width_mm, height_mm],
            outputs=preview_img,
        )

        gen_btn.click(
            on_generate,
            inputs=[
                raw_state, corners_state,
                width_mm, height_mm, dpi,
                brightness, contrast, gamma, inpaint_hardware,
                do_trace, trace_precision,
            ],
            outputs=[download, result_md],
        )

    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        show_error=True,
    )
