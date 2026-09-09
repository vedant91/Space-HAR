"""
Turn a rendered take into an annotated MONITORING view.

`blender/render_video.py` produces a beauty pass. Nothing in it says there is
a perception system in this repository at all. This script runs the real
pipeline - `HARPipeline.process_frame`, the same call the live camera loop
makes - over a take's frames and draws what the system actually perceives and
decides: MediaPipe skeleton, HSV payload boxes, the LSTM's step prediction,
the protocol checklist, the NEXT prompt and the state machine's alert.

Honesty rules this file enforces, because a monitoring overlay is exactly the
place where a demo starts lying:

  * The skeleton drawn is the one that was fed to the classifier. When
    MediaPipe returns nothing and the take's ground-truth pose is drawn
    instead, the frame says so in amber, on the image, every frame.
  * The classifier's raw output is shown whether or not it is right, next to
    the ground-truth step id when the take has one, with a live running
    accuracy and the 0.125 chance line beside it. Per logs/RESULTS.md the
    shipped model is at chance on rendered video; this overlay will show that.
  * `--source oracle` (ground-truth pose into the classifier) and
    `--source labels` (ground-truth step ids into the state machine, no model
    at all) are opt-in flags and are watermarked across the frame.

Usage:
    # honest end-to-end, the default
    .venv312/Scripts/python.exe tools/showcase_overlay.py \
        --take dataset/blender/take_0010 --out build/showcase_overlay/skip \
        --model models/lstm_final_raw.pt --mp4 build/showcase_overlay/skip.mp4

    # just three frames, for a layout check
    .venv312/Scripts/python.exe tools/showcase_overlay.py \
        --take dataset/blender/take_0000 --only 60,140,220 --out build/x

    python tools/showcase_overlay.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from config.experiment_config import (  # noqa: E402
    EXPERIMENT_STEPS, FRAME_WIDTH, FRAME_HEIGHT, SEQUENCE_WINDOW,
    STEP_CONFIDENCE_THRESHOLD,
)
from overlay_pose import draw_pose  # noqa: E402  (tools/overlay_pose.py)

STEP_NAME = {s["id"]: s["name"] for s in EXPERIMENT_STEPS}
CHANCE = 1.0 / len(EXPERIMENT_STEPS)

# Takes the retrained models never saw: tools/retrain_and_evaluate.sh holds out
# these orientations and build_sequences.py drops the error sequences entirely.
# Anything else is training data and a number measured on it is not a result.
HELD_OUT_TAGS = {"orient_180", "orient_-45", "skip", "recover"}

# ── canvas ───────────────────────────────────────────────────────────────────
W, H = 1280, 720
HEAD_H = 76
SIDE_X, SIDE_W = 920, 1280 - 920
BAR_Y, BAR_H = HEAD_H, 44                      # NEXT / ALERT band, video width
VID = (0, BAR_Y + BAR_H, SIDE_X, 480)          # x, y, w, h
LOG = (0, VID[1] + VID[3], SIDE_X, H - VID[1] - VID[3])

BG = (16, 14, 12)
PANEL = (34, 30, 26)
FG = (240, 240, 240)
DIM = (135, 130, 124)
BLUE = (255, 185, 95)
GREEN = (120, 225, 135)
AMBER = (55, 185, 250)
RED = (60, 60, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_H = cv2.FONT_HERSHEY_DUPLEX


def _t(img, s, org, scale=0.45, color=FG, thick=1, font=FONT):
    cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)


def _box(img, x, y, w, h, color, alpha=1.0):
    if alpha >= 1.0:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        return
    x, y = max(x, 0), max(y, 0)
    sub = img[y:y + h, x:x + w]
    if sub.size:
        cv2.addWeighted(np.full_like(sub, color, dtype=np.uint8), alpha,
                        sub, 1 - alpha, 0, sub)


def _tick(img, cx, cy, color):
    cv2.line(img, (cx - 6, cy), (cx - 2, cy + 5), color, 2, cv2.LINE_AA)
    cv2.line(img, (cx - 2, cy + 5), (cx + 6, cy - 6), color, 2, cv2.LINE_AA)


def _fit(src_w, src_h, box_w, box_h):
    """Letterbox fit -> (offset_x, offset_y, width, height)."""
    s = min(box_w / src_w, box_h / src_h)
    w, h = int(src_w * s), int(src_h * s)
    return (box_w - w) // 2, (box_h - h) // 2, w, h


# ── the overlay ──────────────────────────────────────────────────────────────


class Monitor:
    """Draws one annotated 1280x720 monitoring frame."""

    def __init__(self, header: str, subheader: str, subheader_warn: str = "",
                 watermark: str = ""):
        self.header = header
        self.subheader = subheader
        self.subheader_warn = subheader_warn
        self.watermark = watermark
        self.events: deque = deque(maxlen=4)

    def log(self, t_sec: float, text: str, color=FG):
        if self.events and self.events[-1][1] == text:
            return  # the same hold re-firing; one line is the information
        self.events.append((t_sec, text, color))

    def render(self, frame_bgr, dets, pose, pose_is_gt, sm_status, pred, truth,
               acc, perception, timings, alert, frame_no, n_frames, t_sec):
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        self._header(canvas, pred, truth, frame_no, n_frames)
        self._band(canvas, sm_status, alert)
        self._video(canvas, frame_bgr, dets, pose, pose_is_gt, alert)
        self._log(canvas)
        self._sidebar(canvas, sm_status, dets, pose_is_gt, perception, acc, timings)
        if self.watermark:
            self._watermark(canvas)
        return canvas

    # header: identity + the classifier's raw readout vs ground truth
    def _header(self, c, pred, truth, frame_no, n_frames):
        _box(c, 0, 0, W, HEAD_H, PANEL)
        _t(c, self.header, (16, 26), 0.62, FG, 1, FONT_H)
        _t(c, self.subheader, (16, 48), 0.42, DIM)
        if self.subheader_warn:
            (tw, _), _ = cv2.getTextSize(self.subheader, FONT, 0.42, 1)
            _t(c, self.subheader_warn, (16 + tw + 10, 48), 0.42, AMBER)
        _t(c, f"frame {frame_no}/{n_frames}", (16, 66), 0.4, DIM)

        step, conf = pred
        x = 700
        _t(c, "CLASSIFIER (LSTM)", (x, 20), 0.4, DIM)
        label = f"step {step}  {STEP_NAME.get(step, '')}" if step else "no confident call"
        _t(c, label, (x, 44), 0.5, FG if step else DIM)
        # confidence bar - fires the state machine only above the threshold
        bw = 180
        _box(c, x, 54, bw, 12, (60, 55, 50))
        _box(c, x, 54, max(int(bw * float(conf)), 1), 12,
             GREEN if conf >= STEP_CONFIDENCE_THRESHOLD else AMBER)
        cv2.line(c, (x + int(bw * STEP_CONFIDENCE_THRESHOLD), 50),
                 (x + int(bw * STEP_CONFIDENCE_THRESHOLD), 70), FG, 1)
        _t(c, f"{conf:.2f}  (fires at {STEP_CONFIDENCE_THRESHOLD:g})",
           (x + bw + 8, 65), 0.38, DIM)
        if truth is not None:
            ok = step == truth
            col = GREEN if ok else RED
            _t(c, "GROUND TRUTH", (1030, 20), 0.4, DIM)
            _t(c, f"step {truth}", (1030, 44), 0.5, col)
            _t(c, "CORRECT" if ok else "WRONG", (1030, 66), 0.42, col)

    # the headline product feature, or the alert that pre-empts it
    def _band(self, c, sm, alert):
        nxt = sm.get("current_step")
        if alert:
            head, _, detail = alert.partition("|")
            _box(c, 0, BAR_Y, SIDE_X, BAR_H, RED)
            # warning triangle - the band has to read as an alarm at a glance
            cx, cy = 28, BAR_Y + 22
            cv2.fillPoly(c, [np.array([[cx, cy - 13], [cx + 14, cy + 11],
                                       [cx - 14, cy + 11]])], (255, 255, 255))
            _t(c, "!", (cx - 4, cy + 9), 0.5, RED, 2, FONT_H)
            _t(c, head.strip(), (54, BAR_Y + 29), 0.58, (255, 255, 255), 1, FONT_H)
            (tw, _), _ = cv2.getTextSize(head.strip(), FONT_H, 0.58, 1)
            _t(c, detail.strip(), (68 + tw, BAR_Y + 29), 0.44, (235, 225, 225))
            return
        _box(c, 0, BAR_Y, SIDE_X, BAR_H, (48, 42, 34))
        if nxt:
            _t(c, "NEXT", (16, BAR_Y + 30), 0.6, AMBER, 1, FONT_H)
            _t(c, f"STEP {nxt['id']}  -  {nxt['name']}", (96, BAR_Y + 30), 0.6, FG, 1, FONT_H)
            _t(c, EXPERIMENT_STEPS[nxt["id"] - 1]["description"][:46],
               (516, BAR_Y + 29), 0.4, DIM)
        else:
            _t(c, "PROTOCOL COMPLETE", (16, BAR_Y + 30), 0.6, GREEN, 1, FONT_H)

    def _video(self, c, frame, dets, pose, pose_is_gt, alert):
        vx, vy, vw, vh = VID
        _box(c, vx, vy, vw, vh, (10, 9, 8))
        ox, oy, sw, sh = _fit(frame.shape[1], frame.shape[0], vw, vh)
        vid = cv2.resize(frame, (sw, sh))

        # HSV boxes: detections come back in pipeline (1280x720) coordinates.
        kx, ky = sw / float(FRAME_WIDTH), sh / float(FRAME_HEIGHT)
        scaled = [replace(d,
                          bbox=(int(d.bbox[0] * kx), int(d.bbox[1] * ky),
                                int(d.bbox[2] * kx), int(d.bbox[3] * ky)),
                          centroid=(int(d.centroid[0] * kx), int(d.centroid[1] * ky)))
                  for d in dets]
        vid = _HSV_DRAW(vid, scaled)
        for d in scaled:
            # hsv_detector.draw puts the label above the box; a box touching the
            # top of the frame loses it off-image. Re-place those inside, on a
            # dark plate - the main box is a bright screen and eats white text.
            if d.bbox[1] < 20:
                s = f"{d.label} {d.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(s, FONT, 0.5, 1)
                x0, y0 = d.bbox[0] + 4, d.bbox[1] + 6
                _box(vid, x0, y0, tw + 8, th + 8, (0, 0, 0), 0.65)
                _t(vid, s, (x0 + 4, y0 + th + 2), 0.5, (255, 255, 255), 1)

        if pose is not None and np.any(pose):
            vid = draw_pose(vid, pose)

        c[vy + oy:vy + oy + sh, vx + ox:vx + ox + sw] = vid

        if pose_is_gt:
            _box(c, vx + ox, vy + oy, 430, 26, (0, 0, 0), 0.55)
            _t(c, "SKELETON = GROUND TRUTH (no live detection)",
               (vx + ox + 8, vy + oy + 19), 0.46, AMBER)
        else:
            _box(c, vx + ox, vy + oy, 300, 26, (0, 0, 0), 0.45)
            _t(c, "SKELETON = LIVE MediaPipe detection",
               (vx + ox + 8, vy + oy + 19), 0.44, GREEN)

        if alert:
            cv2.rectangle(c, (vx + 2, vy + 2), (vx + vw - 3, vy + vh - 3), RED, 4)

    def _log(self, c):
        lx, ly, lw, lh = LOG
        _box(c, lx, ly, lw, lh, (26, 23, 20))
        _t(c, "EVENT LOG", (16, ly + 20), 0.4, DIM)
        y = ly + 42
        for t_sec, text, color in list(self.events):
            _t(c, f"t+{t_sec:05.1f}s", (16, y), 0.44, DIM)
            _t(c, text, (108, y), 0.44, color)
            y += 20

    def _sidebar(self, c, sm, dets, pose_is_gt, perception, acc, timings):
        # Fixed row positions: a sidebar whose sections jump around as the
        # detection count changes is unwatchable at 30 fps.
        x = SIDE_X
        _box(c, x, HEAD_H, SIDE_W, H - HEAD_H, PANEL)
        cv2.line(c, (x, HEAD_H), (x, H), (60, 54, 48), 1)

        def rule(y, label):
            cv2.line(c, (x + 12, y), (W - 12, y), (60, 54, 48), 1)
            _t(c, label, (x + 16, y + 22), 0.4, DIM)

        _t(c, "PROTOCOL CHECKLIST", (x + 16, 100), 0.44, DIM)
        cur = (sm.get("current_step") or {}).get("id")
        y = 112
        for s in sm["steps"]:
            done = s["status"] == "COMPLETED"
            active = s["id"] == cur
            if active:
                _box(c, x + 8, y, SIDE_W - 16, 32, (72, 60, 42))
            col = GREEN if done else (FG if active else DIM)
            if done:
                _tick(c, x + 28, y + 17, GREEN)
            else:
                cv2.circle(c, (x + 26, y + 16), 6, col, 2 if active else 1, cv2.LINE_AA)
            _t(c, f"{s['id']}", (x + 44, y + 21), 0.46, col)
            _t(c, s["name"][:22], (x + 62, y + 21), 0.46, col)
            if s["recovered"]:
                _t(c, "recovered", (x + 250, y + 21), 0.34, AMBER)
            y += 34

        rule(392, "OBJECT DETECTION  (HSV)")
        yy = 436
        colors = {"red_box": (60, 60, 220), "yellow_box": (0, 215, 255),
                  "main_box": (210, 210, 210)}
        for d in dets[:3]:
            cv2.rectangle(c, (x + 16, yy - 9), (x + 26, yy + 1),
                          colors.get(d.label, FG), -1)
            _t(c, f"{d.label:<11s} {d.confidence:.2f}", (x + 34, yy), 0.44, FG)
            yy += 22
        if not dets:
            _t(c, "none this frame", (x + 16, yy), 0.44, DIM)

        rule(500, "PERCEPTION")
        _t(c, f"pose: {perception['backend']}", (x + 16, 544), 0.42,
           AMBER if pose_is_gt else GREEN)
        _t(c, f"detect rate {perception['rate']:.2f}  "
              f"({perception['hits']}/{perception['seen']})", (x + 16, 564), 0.42, DIM)

        rule(578, "STEP ACCURACY  (live, this take)")
        if acc["n"]:
            rate = acc["correct"] / acc["n"]
            _t(c, f"{rate:.3f}", (x + 16, 636), 0.8,
               GREEN if rate > 2 * CHANCE else (AMBER if rate > CHANCE else RED), 1, FONT_H)
            _t(c, f"{acc['correct']}/{acc['n']} frames", (x + 130, 626), 0.4, DIM)
            _t(c, f"chance = {CHANCE:.3f}", (x + 130, 644), 0.4, DIM)
        else:
            _t(c, acc.get("note", "filling the 30-frame window..."),
               (x + 16, 630), 0.44, DIM)

        rule(654, "LATENCY  (this frame)")
        _t(c, f"hsv {timings['hsv']:.0f} ms    pose {timings['mediapipe']:.0f} ms    "
              f"lstm {timings['lstm']:.0f} ms", (x + 16, 700), 0.42, FG)

    def _watermark(self, c):
        vx, vy, vw, vh = VID
        _box(c, vx, vy + vh - 34, vw, 34, (0, 0, 140), 0.75)
        _t(c, self.watermark, (vx + 12, vy + vh - 12), 0.52, (255, 255, 255), 1, FONT_H)


def _HSV_DRAW(img, dets):
    # bound at import time in main(); kept as a hook so the selftest can run
    # without constructing a detector.
    return _HSV_DRAW.fn(img, dets)


_HSV_DRAW.fn = lambda img, dets: img


# ── driver ───────────────────────────────────────────────────────────────────


def _frames_from(take: Path | None, video: Path | None, camera: str | None):
    """-> (iterator of BGR frames, n_frames, fps, gt_pose|None, labels|None, title)"""
    if take is not None:
        meta = json.loads((take / "meta.json").read_text(encoding="utf-8"))
        cam = camera or meta["cameras"][0]
        cam_dir = take / cam
        paths = sorted((cam_dir / "frames").glob("*.png"))
        gt = np.load(cam_dir / "pose_2d.npy")
        labels = np.load(cam_dir / "labels.npy")
        n = min(len(paths), len(gt), len(labels))
        seen = meta.get("tag") not in HELD_OUT_TAGS
        title = f"{take.name} / {cam} / tag={meta.get('tag')}"
        warn = "[IN THE MODEL'S TRAINING SET]" if seen else "[held-out take]"
        return ((cv2.imread(str(p)) for p in paths[:n]), n,
                float(meta.get("fps", 30)), gt[:n], labels[:n], title, warn)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _gen():
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
        cap.release()

    return _gen(), n, fps, None, None, video.name, ""


def run(args) -> dict:
    from pipeline.har_pipeline import HARPipeline
    from pipeline.rack_frame import RackFrameNormalizer

    take = Path(args.take) if args.take else None
    video = Path(args.video) if args.video else None
    frames, n_frames, fps, gt_pose, labels, title, warn = _frames_from(
        take, video, args.camera)

    pipe = HARPipeline(source=0, headless=True, enable_recording=False,
                       enable_streaming=False, enable_voice=False)
    _HSV_DRAW.fn = pipe.hsv_detector.draw

    # The deployed config (complexity 0, downscale 2) is NOT the config the
    # sequences were built under (tools/build_sequences.py: complexity 1,
    # downscale 1). That is a train/serve skew, and it is worth being able to
    # show either side of it rather than pretending it isn't there.
    if args.pose_complexity is not None or args.pose_downscale is not None:
        from pipeline.har_pipeline import OptimizedMPWrapper
        from config.experiment_config import (MEDIAPIPE_MODEL_COMPLEXITY,
                                              MEDIAPIPE_DOWNSCALE)
        cx = MEDIAPIPE_MODEL_COMPLEXITY if args.pose_complexity is None else args.pose_complexity
        ds = MEDIAPIPE_DOWNSCALE if args.pose_downscale is None else args.pose_downscale
        pipe.mp_wrapper.close()
        pipe.mp_wrapper = OptimizedMPWrapper(complexity=cx, min_det_conf=0.3,
                                             min_trk_conf=0.3, downscale=ds)

    # Swap in the requested checkpoint. HARPipeline picks models/lstm_classifier
    # (ONNX) by default; that is the model logs/RESULTS.md measures at chance,
    # so which one is running has to be a visible, chosen thing.
    model_label = "models/lstm_classifier.onnx (shipped)"
    if args.model:
        from tools.evaluate_real import LSTMRunner
        runner = LSTMRunner(args.model)
        pipe.lstm_ort_sess = None
        pipe.lstm_model = runner.model
        pipe._lstm_label_map = runner.idx_to_label
        model_label = f"{Path(args.model).name} (val {runner.train_val_acc:.2f})"
    if args.rack_normalize:
        pipe.rack_normalizer = RackFrameNormalizer()
        model_label += " +rack"

    watermark = ""
    if args.source == "oracle":
        watermark = "ORACLE MODE - ground-truth pose injected into the classifier"
    elif args.source == "labels":
        watermark = "SCRIPTED MODE - ground-truth step ids drive the state machine, no model"
    if args.source != "mediapipe" and gt_pose is None:
        sys.exit(f"--source {args.source} needs a take with ground truth")

    mon = Monitor(header="ISRO HAR  -  PAYLOAD PROTOCOL MONITOR",
                  subheader=f"{title}  |  {model_label}",
                  subheader_warn=warn, watermark=watermark)

    # Chain onto the pipeline's own state-machine callbacks so the on-screen log
    # shows the same events the experiment logger records.
    sm = pipe.state_machine
    clock = {"t": 0.0}

    def chain(name, fmt, color):
        prev = getattr(sm, name)

        def wrapped(*a):
            if prev:
                prev(*a)
            mon.log(clock["t"], fmt(*a), color)
        setattr(sm, name, wrapped)

    chain("on_step_started", lambda r: f"step {r.step_id} {r.name}: IN PROGRESS", BLUE)
    chain("on_step_completed", lambda r: f"step {r.step_id} {r.name}: COMPLETE", GREEN)
    chain("on_step_skipped",
          lambda e, o: f"ALERT  step {e} not observed - crew appears to be at step {o}", RED)
    chain("on_out_of_sequence",
          lambda e, o: f"ALERT  out of sequence - expected {e}, observed {o}", RED)
    chain("on_step_recovered", lambda r, o: f"step {r.step_id} recovered - back on protocol", AMBER)
    chain("on_experiment_complete", lambda: "PROTOCOL COMPLETE", GREEN)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = {int(v) for v in args.only.split(",") if v.strip()} if args.only else None

    acc = {"correct": 0, "n": 0}
    if args.source == "labels":
        acc["note"] = "not scored - scripted mode"
    perception = {"hits": 0, "seen": 0, "backend": "-", "rate": 0.0}
    oos_until = -1
    written = 0

    # Capture the raw MediaPipe vector on its way into the pipeline: it is what
    # gets drawn, so the drawn skeleton is by construction the one the model saw
    # (or, when it is empty, is visibly replaced by ground truth).
    spy = {"pose": None}
    if pipe.mp_wrapper is not None:
        inner = pipe.mp_wrapper.process

        def capture(rgb, detections=None):
            spy["pose"] = inner(rgb, detections=detections)
            return spy["pose"]
        pipe.mp_wrapper.process = capture
        perception["backend"] = f"MediaPipe {pipe.mp_wrapper.backend_name}"

    for i, frame in enumerate(frames):
        if args.limit and i >= args.limit:
            break
        if frame is None:
            continue
        clock["t"] = i / fps

        inject = gt_pose[i] if (args.source == "oracle" and gt_pose is not None) else None
        res = pipe.process_frame(frame, injected_skel=inject)
        dets = res["detections"]

        mp_pose = spy["pose"]
        pose_ok = mp_pose is not None and bool(np.any(mp_pose))
        perception["seen"] += 1
        perception["hits"] += int(pose_ok)
        perception["rate"] = perception["hits"] / perception["seen"]

        if args.source == "oracle":
            pose, pose_is_gt = gt_pose[i], True
        elif pose_ok:
            pose, pose_is_gt = mp_pose, False
        elif gt_pose is not None:
            pose, pose_is_gt = gt_pose[i], True
        else:
            pose, pose_is_gt = None, False

        truth = int(labels[i]) if labels is not None else None
        pred = (res["pred_step"], res["confidence"])

        if args.source == "labels" and truth:
            # No model in this path at all - the state machine is driven by the
            # renderer's own step ids. Watermarked; never presented as output.
            sm.feed_prediction(truth, 0.99)
        elif truth and len(pipe.skeleton_buffer) >= SEQUENCE_WINDOW:
            acc["n"] += 1
            acc["correct"] += int(pred[0] == truth)

        status = sm.get_status_summary()
        alert = ""
        if status["recovery"]["active"]:
            exp = status["recovery"]["expected_step_id"]
            obs = status["recovery"]["observed_step_id"]
            alert = (f"PROTOCOL ALERT  -  STEP {exp} SKIPPED"
                     f"|observed step {obs} instead - holding until step {exp} is performed")
        elif mon.events and mon.events[-1][2] == RED and mon.events[-1][0] * fps + 2 * fps > i:
            alert = "PROTOCOL ALERT|" + mon.events[-1][1].split("ALERT  ")[-1]

        if only is None or i in only:
            canvas = mon.render(frame, dets, pose, pose_is_gt, status, pred, truth,
                                acc, perception, res["timings_ms"], alert,
                                i, n_frames, clock["t"])
            cv2.imwrite(str(out_dir / f"{i:05d}.png"), canvas)
            written += 1

    pipe.close()

    summary = {
        "take": title,
        "model": model_label,
        "source": args.source,
        "frames_processed": perception["seen"],
        "frames_written": written,
        "pose_detect_rate": round(perception["rate"], 4),
        "step_acc_framewise": round(acc["correct"] / acc["n"], 4) if acc["n"] else None,
        "scored_frames": acc["n"],
        "chance": round(CHANCE, 4),
        "steps_completed": sum(1 for s in sm.get_status_summary()["steps"]
                               if s["status"] == "COMPLETED"),
        "recovery_events": sm.recovery_events,
        "out_dir": str(out_dir),
    }
    if args.mp4 and written:
        from frames_to_video import encode
        encode(out_dir, Path(args.mp4), fps)
        summary["mp4"] = args.mp4
    return summary


def selftest():
    """Layout renders, and the ground-truth fallback is labelled as such."""
    mon = Monitor("HDR", "SUB", watermark="ORACLE MODE - test")
    mon.log(1.0, "step 1 IN PROGRESS", BLUE)
    sm = {"current_step": {"id": 2, "name": "Open Main Box"},
          "recovery": {"active": False},
          "steps": [{"id": s["id"], "name": s["name"],
                     "status": "COMPLETED" if s["id"] == 1 else "PENDING",
                     "recovered": False} for s in EXPERIMENT_STEPS]}
    frame = np.full((506, 900, 3), 40, dtype=np.uint8)
    pose = np.tile(np.array([0.5, 0.5, 0.0, 1.0], dtype=np.float32), 33)
    t = {"hsv": 1.0, "mediapipe": 2.0, "lstm": 3.0}
    p = {"backend": "MediaPipe tasks", "rate": 0.9, "hits": 9, "seen": 10}

    ok = mon.render(frame, [], pose, False, sm, (3, 0.4), 4,
                    {"correct": 1, "n": 8}, p, t, "", 10, 100, 0.3)
    assert ok.shape == (H, W, 3), ok.shape

    fb = mon.render(frame, [], pose, True, sm, (3, 0.4), 4,
                    {"correct": 1, "n": 8}, p, t,
                    "! PROTOCOL ALERT   STEP 2 SKIPPED", 10, 100, 0.3)
    # The fallback frame must differ from the live one - that difference is the
    # "SKELETON = GROUND TRUTH" label, and it is the honesty guarantee.
    assert np.any(ok != fb)
    print("showcase_overlay selftest passed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", help="A take directory, e.g. dataset/blender/take_0010")
    ap.add_argument("--video", help="A video file instead of a take (no ground truth)")
    ap.add_argument("--camera", default=None, help="Camera sub-directory of the take")
    ap.add_argument("--out", default="build/showcase_overlay",
                    help="Directory for annotated PNGs")
    ap.add_argument("--mp4", default=None, help="Also encode the frames to this .mp4")
    ap.add_argument("--model", default=None,
                    help="LSTM .pt checkpoint (default: whatever HARPipeline loads)")
    ap.add_argument("--pose-complexity", type=int, default=None,
                    help="Override MediaPipe complexity (deployed default 0; "
                         "the training sequences were built at 1)")
    ap.add_argument("--pose-downscale", type=int, default=None,
                    help="Override MediaPipe downscale (deployed default 2; "
                         "the training sequences were built at 1)")
    ap.add_argument("--rack-normalize", action="store_true",
                    help="Rack-frame normalise pose before the classifier "
                         "(match models/*_rack.pt)")
    ap.add_argument("--source", choices=("mediapipe", "oracle", "labels"),
                    default="mediapipe",
                    help="mediapipe = honest end-to-end. oracle = ground-truth pose "
                         "into the classifier. labels = ground-truth step ids into the "
                         "state machine. Both non-default modes are watermarked.")
    ap.add_argument("--only", default=None, help="Write only these frame indices")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N frames")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.take or args.video):
        ap.error("one of --take or --video is required")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
