"""Extract original design assets from a mockup PSD.

A mockup PSD typically contains a Smart Object layer holding the designer's
original artwork (vector AI/PDF, raster PNG/JPG/TIFF, or another PSB). When
you double-click that Smart Object in Photoshop, you see the un-rasterised
source. This script pulls those exact bytes out without rasterising anything.

What it does, for each .psd given:
  1. Walks the layer tree and lists every layer with kind + size.
  2. Finds Smart Object (PlacedLayer / SmartObject) layers and dumps each
     Smart Object's embedded raw bytes with the correct extension
     (.ai/.pdf/.svg/.psb/.png/.jpg/.tiff). These are losslessly the
     designer's source.
  3. For every visible layer (including raster art, text, shapes) writes a
     transparent-background PNG of just that layer at its native size.
  4. Writes a `_report.json` summarising what it found.

The output goes to a folder named after the PSD, alongside it.

Run:
    python psd_extract.py path/to/file.psd [more.psd ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from psd_tools import PSDImage
from psd_tools.api.layers import SmartObjectLayer, Group, PixelLayer, TypeLayer, ShapeLayer
from psd_tools.constants import Tag


# Map common Smart Object embedded file signatures to extensions. psd-tools
# usually exposes `.smart_object.filename` and `.smart_object.data` so we get
# the extension for free; this is a fallback when filename is missing.
_MAGIC_EXT = [
    (b"%PDF",                ".pdf"),
    (b"\x89PNG\r\n\x1a\n",   ".png"),
    (b"\xff\xd8\xff",        ".jpg"),
    (b"II*\x00",             ".tif"),
    (b"MM\x00*",             ".tif"),
    (b"8BPS\x00\x02",        ".psb"),
    (b"8BPS\x00\x01",        ".psd"),
    (b"<?xml",               ".svg"),  # could also be .ai; .ai files are PDF-prefixed so handled above
    (b"<svg",                ".svg"),
    (b"%!PS",                ".ai"),
]


def sniff_extension(data: bytes) -> str:
    head = data[:16]
    for sig, ext in _MAGIC_EXT:
        if head.startswith(sig):
            return ext
    return ".bin"


def safe_name(s: str, fallback: str = "layer") -> str:
    s = (s or fallback).strip()
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    return out[:80] or fallback


def describe_layer(layer, depth: int = 0) -> dict[str, Any]:
    info: dict[str, Any] = {
        "name": layer.name,
        "kind": layer.kind,
        "visible": bool(layer.visible),
        "size": [layer.width, layer.height],
        "offset": [layer.left, layer.top],
        "depth": depth,
    }
    if isinstance(layer, SmartObjectLayer):
        so = layer.smart_object
        info["smart_object"] = {
            "filename": getattr(so, "filename", None),
            "type": getattr(so, "type", None),
            "has_data": bool(getattr(so, "data", None)),
            "data_size": len(getattr(so, "data", b"") or b""),
        }
    if isinstance(layer, TypeLayer):
        try:
            info["text"] = layer.text
        except Exception:
            pass
    return info


def walk_layers(layer, depth: int = 0):
    yield layer, depth
    if isinstance(layer, Group) or hasattr(layer, "__iter__"):
        try:
            for child in layer:
                yield from walk_layers(child, depth + 1)
        except TypeError:
            pass


def extract_psd(psd_path: str, out_root: str | None = None) -> dict[str, Any]:
    psd = PSDImage.open(psd_path)
    base = os.path.splitext(os.path.basename(psd_path))[0]
    outdir = out_root or os.path.join(os.path.dirname(os.path.abspath(psd_path)), f"{base}_extracted")
    os.makedirs(outdir, exist_ok=True)
    so_dir = os.path.join(outdir, "smart_objects")
    layer_dir = os.path.join(outdir, "layers_png")
    os.makedirs(so_dir, exist_ok=True)
    os.makedirs(layer_dir, exist_ok=True)

    report: dict[str, Any] = {
        "psd": os.path.basename(psd_path),
        "canvas_size": [psd.width, psd.height],
        "color_mode": str(psd.color_mode),
        "layers": [],
        "smart_objects": [],
        "layer_pngs": [],
    }

    # 1. Composite (the flattened mockup itself) for reference.
    try:
        composite = psd.composite()
        if composite is not None:
            composite.save(os.path.join(outdir, "_composite.png"))
            report["composite"] = "_composite.png"
    except Exception as e:
        report["composite_error"] = str(e)

    # 2. Walk layers.
    seq = 0
    for layer, depth in walk_layers(psd):
        if layer is psd:
            continue
        seq += 1
        desc = describe_layer(layer, depth)
        desc["seq"] = seq
        report["layers"].append(desc)

        # 2a. Smart Object extraction (this is the prize).
        if isinstance(layer, SmartObjectLayer):
            so = layer.smart_object
            data = getattr(so, "data", None) or b""
            if data:
                filename = getattr(so, "filename", None)
                if filename:
                    ext = os.path.splitext(filename)[1] or sniff_extension(data)
                    name_in_file = os.path.splitext(os.path.basename(filename))[0]
                else:
                    ext = sniff_extension(data)
                    name_in_file = "embedded"
                fn = f"{seq:02d}_{safe_name(layer.name)}__{safe_name(name_in_file)}{ext}"
                path = os.path.join(so_dir, fn)
                with open(path, "wb") as f:
                    f.write(data)
                desc["extracted_to"] = os.path.relpath(path, outdir)
                report["smart_objects"].append({
                    "layer": layer.name,
                    "original_filename": filename,
                    "extracted_to": os.path.relpath(path, outdir),
                    "bytes": len(data),
                    "sniffed_ext": sniff_extension(data),
                })

        # 2b. Render each renderable layer as a transparent PNG at its native size.
        if not isinstance(layer, Group) and layer.width > 0 and layer.height > 0:
            try:
                img = layer.composite()
                if img is not None:
                    fn = f"{seq:02d}_{safe_name(layer.name)}.png"
                    path = os.path.join(layer_dir, fn)
                    img.save(path)
                    desc["png"] = os.path.relpath(path, outdir)
                    report["layer_pngs"].append(desc["png"])
            except Exception as e:
                desc["render_error"] = str(e)

    with open(os.path.join(outdir, "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def main():
    p = argparse.ArgumentParser(description="Extract design assets from a mockup PSD")
    p.add_argument("psd", nargs="+", help="One or more .psd files")
    p.add_argument("--out", help="Output root directory (default: alongside each PSD)")
    args = p.parse_args()
    for psd in args.psd:
        if not os.path.exists(psd):
            print(f"[skip] {psd} not found", file=sys.stderr)
            continue
        report = extract_psd(psd, args.out)
        print(f"[ok] {psd}")
        print(f"     canvas {report['canvas_size']}, "
              f"{len(report['layers'])} layers, "
              f"{len(report['smart_objects'])} smart objects extracted")
        for so in report["smart_objects"]:
            print(f"     SO: {so['layer']!r} -> {so['extracted_to']}  ({so['bytes']:,} B, {so['sniffed_ext']})")


if __name__ == "__main__":
    main()
