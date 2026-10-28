"""Camera intrinsics from a printed checkerboard, written as a ROS camera_info YAML.

Reads the camera directly (stop the ROS camera node first): the raw ROS image stream is too big
to arrive at a usable rate. Live preview: SPACE saves the current view when the board is found,
Q or ESC calibrates once at least 15 views are saved.
Usage: python scripts/calibrate_camera.py --device /dev/overhead_camera --square 0.025 \
           --out ros2_ws/src/so101_perception/config/overhead_camera_info.yaml
"""
import argparse
import os
import re

import cv2
import numpy as np

CORNERS = (7, 5)  # inner corners of the 8 x 6 board from make_print_sheets.py


def write_camera_info(path, name, size, K, D):
    P = np.hstack([K, np.zeros((3, 1))])
    fmt = lambda a: ", ".join(f"{x:.6f}" for x in np.asarray(a).ravel())
    with open(path, "w") as f:
        f.write(f"image_width: {size[0]}\nimage_height: {size[1]}\ncamera_name: {name}\n"
                f"camera_matrix:\n  rows: 3\n  cols: 3\n  data: [{fmt(K)}]\n"
                f"distortion_model: plumb_bob\n"
                f"distortion_coefficients:\n  rows: 1\n  cols: 5\n  data: [{fmt(D[:5])}]\n"
                f"rectification_matrix:\n  rows: 3\n  cols: 3\n  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
                f"projection_matrix:\n  rows: 3\n  cols: 4\n  data: [{fmt(P)}]\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="/dev/overhead_camera")
    p.add_argument("--square", type=float, default=0.025, help="measured square size, m")
    p.add_argument("--name", default="overhead_camera")
    p.add_argument("--out", default="ros2_ws/src/so101_perception/config/overhead_camera_info.yaml")
    args = p.parse_args()

    # OpenCV's V4L2 backend opens devices by number, so follow the udev link to /dev/videoN.
    index = int(re.search(r"video(\d+)$", os.path.realpath(args.device)).group(1))
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ok, frame = cap.read()
    assert ok, f"cannot read {args.device} (is the ROS camera node still running?)"
    size = (frame.shape[1], frame.shape[0])
    print(f"camera {args.device}: {size[0]}x{size[1]}")

    board = np.zeros((CORNERS[0] * CORNERS[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:CORNERS[0], 0:CORNERS[1]].T.reshape(-1, 2) * args.square
    obj_pts, img_pts = [], []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect on a half-size image for a smooth preview; refine on full size when saving.
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        found, corners = cv2.findChessboardCorners(small, CORNERS, cv2.CALIB_CB_FAST_CHECK)
        view = cv2.resize(frame, None, fx=0.5, fy=0.5)
        if found:
            cv2.drawChessboardCorners(view, CORNERS, corners, found)
        cv2.putText(view, f"saved {len(img_pts)} (need 15+)  SPACE save  Q done", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calibration", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            full = cv2.cornerSubPix(gray, corners * 2, (11, 11), (-1, -1), crit)
            obj_pts.append(board)
            img_pts.append(full)
            print(f"saved view {len(img_pts)}")
        elif key in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()

    assert len(img_pts) >= 15, f"only {len(img_pts)} views; need at least 15"
    rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    write_camera_info(args.out, args.name, size, K, D.ravel())
    print(f"RMS reprojection error {rms:.3f} px ({'good' if rms < 0.5 else 'ok' if rms < 1.0 else 'redo: too high'})")
    print(f"fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}  distortion {np.round(D.ravel(), 4)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
