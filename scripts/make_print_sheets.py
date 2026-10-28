"""Printable sheets for M8c, as a PDF at exact size (print at 100% / "actual size", then measure).

Page 1: checkerboard, 8 x 6 squares of 25 mm (7 x 5 inner corners) for camera intrinsics.
Page 2: table marker (ArUco DICT_4X4_50 id 0, 80 mm) and a 30 mm cube net with markers
        id 1 (top) and 2-5 (sides), 22 mm each, so a marker stays visible when the gripper covers one.
Usage: python scripts/make_print_sheets.py [--out prints/m8c_sheets.pdf]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DPI = 300
A4 = (210, 297)  # mm
ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def px(mm):
    return int(round(mm * DPI / 25.4))


def page():
    return np.full((px(A4[1]), px(A4[0])), 255, np.uint8)


def text(img, s, x_mm, y_mm, scale=1.6):
    cv2.putText(img, s, (px(x_mm), px(y_mm)), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 3, cv2.LINE_AA)


def marker(img, marker_id, size_mm, x_mm, y_mm):
    """Paste a marker with its top-left corner at (x, y) mm."""
    m = cv2.aruco.generateImageMarker(ARUCO, marker_id, px(size_mm))
    img[px(y_mm):px(y_mm) + m.shape[0], px(x_mm):px(x_mm) + m.shape[1]] = m


def checkerboard():
    img, sq = page(), 25
    x0, y0 = (A4[0] - 8 * sq) / 2, 40
    for r in range(6):
        for c in range(8):
            if (r + c) % 2 == 0:
                cv2.rectangle(img, (px(x0 + c * sq), px(y0 + r * sq)),
                              (px(x0 + (c + 1) * sq) - 1, px(y0 + (r + 1) * sq) - 1), 0, -1)
    text(img, "Checkerboard 8 x 6 squares, 25 mm (7 x 5 inner corners)", 15, 25, 1.2)
    text(img, "Measure one square after printing; glue flat on card.", 15, y0 + 6 * sq + 15, 1.2)
    return img


def markers_and_cube():
    img = page()
    # Table marker with axis lines: align the arrow with the robot's +x (forward).
    x0, y0, s = 20, 25, 80
    marker(img, 0, s, x0, y0)
    cx, cy = x0 + s / 2, y0 + s / 2
    cv2.line(img, (px(cx), px(y0 + s + 2)), (px(cx), px(y0 + s + 25)), 0, 3)
    cv2.line(img, (px(x0 + s + 2), px(cy)), (px(x0 + s + 25), px(cy)), 0, 3)
    text(img, "ID 0, 80 mm: table marker", x0 + s + 30, y0 + 20, 1.1)
    text(img, "lines mark the centre", x0 + s + 30, y0 + 35, 1.1)
    text(img, "marker 'up' = robot +x", x0 + s + 30, y0 + 50, 1.1)

    # Cube net (cross): column = side 2, top, side 4, bottom; arms = side 3, side 5.
    f, m = 30, 22
    nx, ny = 75, 160
    faces = {(1, 0): 2, (1, 1): 1, (1, 2): 4, (1, 3): None, (0, 1): 3, (2, 1): 5}
    for (c, r), mid in faces.items():
        x, y = nx + c * f, ny + r * f
        cv2.rectangle(img, (px(x), px(y)), (px(x + f), px(y + f)), 150, 2)
        if mid is not None:
            marker(img, mid, m, x + (f - m) / 2, y + (f - m) / 2)
    text(img, "30 mm cube: cut the outline, fold on grey lines,", 20, ny - 18, 1.1)
    text(img, "tape. Marker ID 1 = top. Add 15-30 g inside.", 20, ny - 8, 1.1)
    return img


def markers_only():
    """For a cube built from cardboard: just the markers, each in a cut-out square with a white border."""
    img = page()
    marker(img, 0, 80, 20, 30)
    cv2.rectangle(img, (px(15), px(25)), (px(105), px(115)), 150, 2)
    text(img, "ID 0 - table marker, 80 mm", 115, 55, 1.1)
    text(img, "top of marker = robot +x", 115, 70, 1.1)
    labels = {1: "ID 1 - cube TOP", 2: "ID 2 - side", 3: "ID 3 - side", 4: "ID 4 - side", 5: "ID 5 - side"}
    for k, mid in enumerate(labels):
        x, y = 20 + (k % 3) * 60, 140 + (k // 3) * 60
        marker(img, mid, 22, x + 4, y + 4)
        cv2.rectangle(img, (px(x), px(y)), (px(x + 30), px(y + 30)), 150, 2)  # cut line: 30 mm face
        text(img, labels[mid], x, y + 40, 0.9)
    text(img, "Each small square = one 30 mm cube face (22 mm marker + 4 mm white border).", 15, 275, 0.9)
    text(img, "Side markers: their top edge towards the cube's top face.", 15, 285, 0.9)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="prints/m8c_sheets.pdf")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pages = [Image.fromarray(checkerboard()), Image.fromarray(markers_and_cube())]
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=DPI)
    for i, pg in enumerate(pages, 1):
        pg.save(out.with_name(f"{out.stem}_p{i}.png"), dpi=(DPI, DPI))
    m = Image.fromarray(markers_only())
    m.save(out.with_name("aruco_markers.pdf"), resolution=DPI)
    m.save(out.with_name("aruco_markers.png"), dpi=(DPI, DPI))
    print(f"wrote {out}, {out.with_name('aruco_markers.pdf')} (and PNG previews)")


if __name__ == "__main__":
    main()
