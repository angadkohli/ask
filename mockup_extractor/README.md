# Mockup Design Extractor

Reverse a product mockup — given a rendered mockup image (e.g. a suitcase with a
printed design panel), extract the design itself as a flat, print-ready file.

## What it does

Pipeline per mockup:

1. **Perspective unwarp.** You click the 4 corners of the design panel
   (TL → TR → BR → BL). A homography flattens that quadrilateral to a true
   rectangle at the print size you specify.
2. **(Optional) Hardware inpaint.** Dark hardware that overlaps the design after
   unwarp (corner protectors, zipper edges) is masked and inpainted.
3. **(Optional) Levels / gamma.** Quick brightness / contrast / gamma to undo
   the slight darkening that mockup PSDs typically multiply onto the design.
4. **Print-ready export.** PNG at your target DPI, single-page PDF sized in mm,
   SVG with the raster embedded, and a true-vector trace SVG via `vtracer`.

## Honest limits

- A bitmap mockup does not contain the original Illustrator paths. The traced
  SVG is a *new* vector approximation of the (unwarped) bitmap. Flat artwork
  (marble monograms, stripes, text) traces cleanly; photographic detail
  (the tiger, the leopard) traces as posterised regions.
- Lighting baked into the mockup (specular highlights, vignette from the
  curve of the case) cannot be fully removed. The levels controls help; for a
  perfect result you'd need the original blank product photo to do per-pixel
  division.
- The output rectangle is rectangular. If the original design wrapped around a
  curved edge in the mockup, that curvature is undone — print straight onto a
  flat substrate.

## Run

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:7860
```

### CLI (batch)

```bash
python cli.py template.json
```

See `cli.py` docstring for the template format.

## Files

- `app.py` — Gradio web app (click 4 corners, export)
- `extractor.py` — Core image-processing functions (unwarp, inpaint, save)
- `cli.py` — Batch processor driven by JSON templates
