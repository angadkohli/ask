"""Detect the four corners of the design panel on a clean studio mockup.

Assumption: the mockup has a near-white background and the suitcase fills the
foreground. We threshold the background, take the largest non-background blob,
approximate its outer contour to a quadrilateral, and order corners TL/TR/BR/BL.

This is intended for the studio shots in `samples/` — not for in-the-wild photos.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_design_quad(image_bgr: np.ndarray, bg_thresh: int = 235) -> list[tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Background = pixels near white; foreground = the suitcase.
    fg = (gray < bg_thresh).astype(np.uint8) * 255
    # Close small gaps so the suitcase becomes a single blob.
    k = max(3, int(min(h, w) * 0.005)) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("no foreground contour")
    cnt = max(contours, key=cv2.contourArea)
    # The suitcase outline isn't a perfect quad (rounded corners + hardware).
    # Find the minimum-area enclosing rotated rect first — gives an OK approximation.
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    # Refine: for each box corner, snap to the actual nearest contour point.
    cnt_pts = cnt.reshape(-1, 2)
    refined = []
    for bx, by in box:
        d2 = (cnt_pts[:, 0] - bx) ** 2 + (cnt_pts[:, 1] - by) ** 2
        refined.append(tuple(cnt_pts[int(np.argmin(d2))].tolist()))
    # Order TL, TR, BR, BL
    pts = np.array(refined, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl = tuple(map(int, pts[np.argmin(s)]))
    br = tuple(map(int, pts[np.argmax(s)]))
    tr = tuple(map(int, pts[np.argmin(d)]))
    bl = tuple(map(int, pts[np.argmax(d)]))
    return [tl, tr, br, bl]


def draw_corners(image_bgr: np.ndarray, corners: list[tuple[int, int]]) -> np.ndarray:
    out = image_bgr.copy()
    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(corners):
        cv2.circle(out, (x, y), 14, (32, 32, 255), 4)
        cv2.circle(out, (x, y), 3, (32, 32, 255), -1)
        cv2.putText(out, labels[i], (x + 18, y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (32, 32, 255), 2, cv2.LINE_AA)
    pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, (32, 32, 255), 3)
    return out


if __name__ == "__main__":
    import argparse, json, os
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--preview", help="write annotated preview to this path")
    args = p.parse_args()
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    corners = detect_design_quad(img)
    print(json.dumps({"corners": corners, "image_size": [img.shape[1], img.shape[0]]}))
    if args.preview:
        cv2.imwrite(args.preview, draw_corners(img, corners))
