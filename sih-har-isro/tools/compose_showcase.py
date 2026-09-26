"""Create a presentation MP4 from the 3D BAS protocol render.

The renderer is the source of truth for the astronaut, rack and procedure.
This file adds a presentation-only monitoring treatment around that footage;
it never substitutes synthetic detections or claims that a scripted state is
an observed model prediction.

    python tools/compose_showcase.py build/showcase/recovery_frames \
        build/showcase/space_har_showcase.mp4 --hold 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


STEPS = {
    1: "Approach main box",
    2: "Open main box",
    3: "Pick red box",
    4: "Examine red box",
    5: "Place red box",
    6: "Pick yellow box",
    7: "Examine yellow box",
    8: "Place yellow box",
}

W, H = 1280, 720
VIDEO = (0, 108, 910, 512)
BG = (22, 22, 20)
PANEL = (35, 35, 31)
TEXT = (238, 238, 232)
MUTED = (160, 158, 150)
TEAL = (190, 210, 74)
GOLD = (90, 195, 245)
RED = (56, 64, 224)
GREEN = (112, 205, 112)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28),
]


def text(img, value, xy, scale=0.46, color=TEXT, thick=1):
    cv2.putText(img, value, xy, FONT, scale, color, thick, cv2.LINE_AA)


def plate(img, x, y, w, h, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)


def fit(frame, w, h):
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    rw, rh = int(fw * scale), int(fh * scale)
    return cv2.resize(frame, (rw, rh)), (w - rw) // 2, (h - rh) // 2


def phase(meta, frame_no):
    for span in meta["spans"]:
        if span["start_frame"] <= frame_no <= span["end_frame"]:
            return span
    return meta["spans"][-1]


def status(meta, span):
    occ, step = span["occurrence"], span["step_id"]
    sequence = meta["sequence"]
    completed = []
    alert = None
    recovered = False
    for prior in meta["spans"]:
        if prior["occurrence"] >= occ:
            break
        if prior["step_id"] not in completed:
            completed.append(prior["step_id"])

    # The rendered recovery take intentionally performs 1 -> 3 -> 2 -> 3...
    # It is an illustrative state-machine scenario, therefore labelled as
    # such, not passed off as a prediction from the current LSTM checkpoint.
    if meta.get("sequence_name") == "recover" and occ == 1 and step == 3:
        alert = "STEP 2 NOT CONFIRMED - GUIDANCE HOLD"
    elif meta.get("sequence_name") == "recover" and occ == 2 and step == 2:
        recovered = True
    elif meta.get("sequence_name") == "recover" and occ > 2 and 2 not in completed:
        completed.append(2)
    return completed, alert, recovered


def overlay_perception(image, pose, payload):
    """Draw labels projected from the Blender scene, not model outputs."""
    h, w = image.shape[:2]
    if pose is not None:
        points = pose.reshape(33, 4)
        for a, b in CONNECTIONS:
            if points[a, 3] < 0.05 or points[b, 3] < 0.05:
                continue
            p1 = (int(points[a, 0] * w), int(points[a, 1] * h))
            p2 = (int(points[b, 0] * w), int(points[b, 1] * h))
            cv2.line(image, p1, p2, (118, 245, 152), 2, cv2.LINE_AA)
        for x, y, _z, vis in points:
            if vis >= 0.05:
                cv2.circle(image, (int(x * w), int(y * h)), 3,
                           (245, 216, 96) if vis >= 0.5 else (130, 130, 210), -1, cv2.LINE_AA)
    for key, info in (payload or {}).items():
        if not info.get("on_screen"):
            continue
        x1, y1, x2, y2 = info["bbox"]
        colour = {"red_box": (55, 65, 235), "yellow_box": (65, 225, 245),
                  "main_box": (225, 225, 225)}.get(key, TEXT)
        x1, x2 = int(x1 / 1280 * w), int(x2 / 1280 * w)
        y1, y2 = int(y1 / 720 * h), int(y2 / 720 * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        text(image, key.replace("_", " "), (x1 + 3, max(16, y1 - 6)), 0.36, colour)
    return image


def draw(frame, meta, frame_no, pose=None, payload=None):
    span = phase(meta, frame_no)
    current = span["step_id"]
    completed, alert, recovered = status(meta, span)
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)

    plate(canvas, 0, 0, W, 70, PANEL)
    text(canvas, "SPACE-HAR", (22, 31), 0.82, TEXT, 2)
    text(canvas, "ON-BOARD EXPERIMENT ASSISTANT", (22, 55), 0.41, MUTED)
    text(canvas, "BAS PAYLOAD PROTOCOL  |  SIMULATION SHOWCASE", (790, 39), 0.38, TEAL)

    plate(canvas, 0, 70, W, 38, (44, 43, 37))
    if alert:
        plate(canvas, 0, 70, W, 38, RED)
        text(canvas, "VOICE GUIDANCE", (22, 96), 0.5, (255, 255, 255), 1)
        text(canvas, "Step 2 was not confirmed. Complete Open main box before proceeding.", (194, 96), 0.48, (255, 255, 255), 1)
    elif recovered:
        text(canvas, "RECOVERY ACCEPTED", (22, 96), 0.5, GREEN, 1)
        text(canvas, "Step 2 confirmed. Returning to the ordered procedure.", (228, 96), 0.48, TEXT)
    else:
        text(canvas, "NEXT ACTION", (22, 96), 0.5, GOLD, 1)
        text(canvas, f"Step {current}: {STEPS[current]}", (180, 96), 0.48, TEXT)

    vx, vy, vw, vh = VIDEO
    plate(canvas, vx, vy, vw, vh, (8, 9, 8))
    resized, ox, oy = fit(frame, vw, vh)
    resized = overlay_perception(resized, pose, payload)
    rh, rw = resized.shape[:2]
    canvas[vy + oy:vy + oy + rh, vx + ox:vx + ox + rw] = resized
    plate(canvas, 16, 124, 410, 26, (0, 0, 0))
    text(canvas, "FIXED PAYLOAD CAMERA / 3D SIM LABELS: POSE 33 + OBJECTS", (25, 143), 0.34, TEXT)
    if alert:
        cv2.rectangle(canvas, (vx + 2, vy + 2), (vx + vw - 3, vy + vh - 3), RED, 4)

    sx = 930
    plate(canvas, sx, 108, 350, 512, PANEL)
    text(canvas, "PROTOCOL STATE", (sx + 18, 139), 0.5, MUTED)
    y = 163
    for step_id in range(1, 9):
        done = step_id in completed
        active = step_id == current
        if active:
            plate(canvas, sx + 12, y - 20, 326, 30, (67, 62, 45) if not alert else (78, 42, 38))
        color = GREEN if done else (TEXT if active else MUTED)
        cv2.circle(canvas, (sx + 31, y - 5), 7, color, -1 if done else 1)
        if done:
            text(canvas, "+", (sx + 27, y), 0.35, (20, 20, 20), 1)
        text(canvas, str(step_id), (sx + 49, y), 0.42, color)
        text(canvas, STEPS[step_id], (sx + 73, y), 0.42, color)
        y += 38

    cv2.line(canvas, (sx + 16, 486), (1262, 486), (78, 77, 70), 1)
    text(canvas, "EDGE OPERATION", (sx + 18, 515), 0.42, MUTED)
    text(canvas, "Local video processing", (sx + 18, 540), 0.45, TEXT)
    text(canvas, "Pose + object + step sequence", (sx + 18, 564), 0.41, TEXT)
    text(canvas, "No ground link required", (sx + 18, 588), 0.41, TEAL)

    plate(canvas, sx + 16, 602, 310, 10, (70, 68, 61))
    span_len = span["end_frame"] - span["start_frame"] + 1
    fill = min(1.0, (frame_no - span["start_frame"] + 1) / max(1, min(30, span_len)))
    plate(canvas, sx + 16, 602, int(310 * fill), 10, TEAL)
    text(canvas, "TEMPORAL HAR WINDOW / 30 FRAMES", (sx + 16, 636), 0.36, MUTED)

    plate(canvas, 0, 620, W, 100, (28, 28, 25))
    text(canvas, "HOW IT HELPS THE CREW", (22, 651), 0.42, MUTED)
    if alert:
        line = "Detects a protocol deviation, holds progression, and gives a clear voice cue."
    elif recovered:
        line = "Confirms recovery, resumes the workflow, and records the corrected outcome."
    else:
        line = "Tracks the procedure, presents the next action, and logs the experiment locally."
    text(canvas, line, (22, 680), 0.52, TEXT, 1)
    text(canvas, "3D labels visualize HAR inputs; scripted protocol state is a demonstration, not a model-score claim.", (22, 706), 0.34, MUTED)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", type=Path, help="Directory produced by blender/render_video.py")
    ap.add_argument("out", type=Path)
    ap.add_argument("--hold", type=int, default=3, help="Copies of each rendered frame in the presentation video")
    args = ap.parse_args()
    if args.hold < 1:
        ap.error("--hold must be at least 1")

    meta = json.loads((args.frames / "protocol_meta.json").read_text(encoding="utf-8"))
    files = sorted(args.frames.glob("*.png"))
    if not files:
        raise SystemExit(f"no source PNG frames in {args.frames}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(meta["fps"]), (W, H))
    if not writer.isOpened():
        raise SystemExit("could not open MP4 writer")
    written = 0
    pose_path = args.frames / "pose_2d.npy"
    payload_path = args.frames / "payload.json"
    poses = np.load(pose_path) if pose_path.exists() else None
    payloads = json.loads(payload_path.read_text(encoding="utf-8")) if payload_path.exists() else None
    for i, path in enumerate(files, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"cannot read {path}")
        pose = poses[i - 1] if poses is not None and i <= len(poses) else None
        payload = payloads[i - 1] if payloads is not None and i <= len(payloads) else None
        card = draw(frame, meta, i, pose, payload)
        for _ in range(args.hold):
            writer.write(card)
            written += 1
    writer.release()
    print(f"VIDEO_OK {args.out} ({written} frames, {written / meta['fps']:.1f} s)")


if __name__ == "__main__":
    main()
