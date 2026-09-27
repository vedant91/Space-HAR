"""
Before/after for the ROMP crop + rotate + schedule path (pipeline/hmr_backend.py).

"before" = ROMP on the whole frame, one call per frame: the pre-crop
`_romp_features`, reproduced verbatim below so this stays runnable after the
change. "after" = HMRBackend("romp") fed every frame of the take in order, as
tools/build_sequences.py feeds it (tracking and scheduling are temporal, so
it cannot be fed a sample).

Detection rate alone says nothing here: at ROMP_CENTER_THRESH=0.02 ROMP
"finds" someone on every frame, including the frames where it has locked onto
rack geometry. So each scored frame is also checked against the Blender
ground truth (pose_2d.npy) on the 8 upper-body joints that drive the protocol
(shoulders, elbows, wrists, hips): error in units of the ground-truth torso
length (mid-shoulder to mid-hip), a joint is correct under 0.5 torso, and a
frame is "tracked" when its median joint error is under 0.5 torso.

Run under the HMR venv (romp + mediapipe + ultralytics):
    D:/SIH-2026/.venv-hmr/Scripts/python.exe tools/measure_hmr_crop.py --takes 3,4,7,13,17
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import pipeline.hmr_backend as hb  # noqa: E402
from config.experiment_config import FRAME_HEIGHT, FRAME_WIDTH  # noqa: E402

UPPER = [11, 12, 13, 14, 15, 16, 23, 24]


def frame_error(pred: np.ndarray, gt: np.ndarray, wh):
    """(median upper-body error / torso, fraction of joints < 0.5 torso)."""
    if not np.any(pred):
        return np.inf, 0.0
    p = pred.reshape(33, 4)[:, :2] * wh
    g = gt.reshape(33, 4)[:, :2] * wh
    torso = np.linalg.norm(g[[11, 12]].mean(0) - g[[23, 24]].mean(0))
    e = np.linalg.norm(p[UPPER] - g[UPPER], axis=1) / max(torso, 1e-6)
    return float(np.median(e)), float(np.mean(e < 0.5))


def full_frame(backend, frame_rgb):
    """The pre-crop HMRBackend._romp_features, unchanged."""
    with contextlib.redirect_stdout(io.StringIO()):
        out = backend.model(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if out is None or len(out.get("center_confs", [])) == 0:
        return np.zeros(hb.FEATURE_DIM, dtype=np.float32)
    i = int(np.argmax(np.asarray(out["center_confs"]).reshape(-1)))
    h, w = frame_rgb.shape[:2]
    return hb.romp_to_feature_vector(out["joints"][i], out["pj2d_org"][i], out["cam"][i], (w, h))


def summarise(rows):
    med = np.array([r[0] for r in rows])
    return {"n": len(rows), "detect_rate": round(float(np.mean(np.isfinite(med))), 3),
            "tracked": round(float(np.mean(med < 0.5)), 3),
            "pck05": round(float(np.mean([r[1] for r in rows])), 3)}


def take_camera_dir(take_dir: Path, meta: dict) -> Path:
    """Most takes render one camera named in meta['cameras']; use the first
    one that actually has a directory (take_0022 uses 'payload_wide', not
    the usual 'payload_a')."""
    for cam in meta.get("cameras", []):
        d = take_dir / cam
        if d.is_dir():
            return d
    raise FileNotFoundError(f"{take_dir}: none of meta['cameras']={meta.get('cameras')} exist on disk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(REPO / "dataset" / "blender_core"))
    ap.add_argument("--takes", default="3,4,7,13,17", help="take indices, comma-separated")
    ap.add_argument("--backend", default="romp", choices=["romp", "custom"],
                    help="custom = train/hmr3d_model.py checkpoint (models/hmr3d_net.pt); "
                         "the 'before' full-frame comparison only applies to romp and is skipped for custom")
    ap.add_argument("--score-every", type=int, default=10, help="score (and run 'before' on) every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole take")
    ap.add_argument("--stale", type=int, default=None, help="override ROMP_MAX_STALE_FRAMES (1 = ROMP every frame)")
    ap.add_argument("--no-retry", action="store_true", help="disable the rotation-hypothesis fallback (romp only)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.stale is not None:
        hb.ROMP_MAX_STALE_FRAMES = args.stale
    if args.no_retry:
        hb.ROMP_RETRY_CONF = -1.0

    backend = hb.HMRBackend(args.backend)
    if not backend.available:
        raise SystemExit(f"{args.backend} backend did not load (run under D:/SIH-2026/.venv-hmr)")
    wh = np.array([FRAME_WIDTH, FRAME_HEIGHT], dtype=np.float64)
    report = {}
    for t in (int(x) for x in args.takes.split(",")):
        d = Path(args.dataset) / f"take_{t:04d}"
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        tag = meta["tag"]
        cam = take_camera_dir(d, meta)
        gt = np.load(cam / "pose_2d.npy")
        frames = sorted((cam / "frames").glob("*.png"))
        n = min(len(frames), len(gt)) if not args.max_frames else min(args.max_frames, len(frames), len(gt))
        backend.reset()
        before, after, t_before, t_after = [], [], 0.0, 0.0
        for i in range(n):
            rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(frames[i])), (FRAME_WIDTH, FRAME_HEIGHT)),
                               cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            feats, _ = backend.get_pose_features(rgb)
            t_after += time.perf_counter() - t0
            if i % args.score_every:
                continue
            after.append(frame_error(feats, gt[i], wh))
            if args.backend == "romp":   # "before" = pre-crop ROMP; not meaningful for a from-scratch model
                t0 = time.perf_counter()
                before.append(frame_error(full_frame(backend, rgb), gt[i], wh))
                t_before += time.perf_counter() - t0
        st = dict(backend.stats)
        row = {"tag": tag, "frames": n,
               "after": {**summarise(after), "ms_per_frame": round(1000 * t_after / max(n, 1), 1),
                         "romp_runs_per_frame": round(st["romp_runs"] / max(n, 1), 3),
                         "mp_hit_rate": round(st["mp_hits"] / max(n, 1), 3),
                         "yolo_proposals": st["yolo_proposals"]}}
        if args.backend == "romp":
            row["before"] = {**summarise(before), "ms_per_frame": round(1000 * t_before / max(len(before), 1), 1)}
            row["after"].update(retries=st["romp_retries"], retry_switched=st["retry_switched"])
        report[f"take_{t:04d}"] = row
        r = row
        before_str = (f"before: tracked={r['before']['tracked']:.2f} pck={r['before']['pck05']:.2f} "
                     f"det={r['before']['detect_rate']:.2f} {r['before']['ms_per_frame']:.0f}ms | "
                     if args.backend == "romp" else "")
        print(f"take_{t:04d} {tag:22s} {before_str}after: tracked={r['after']['tracked']:.2f} "
              f"pck={r['after']['pck05']:.2f} det={r['after']['detect_rate']:.2f} "
              f"{r['after']['ms_per_frame']:.0f}ms/frame run/frame={r['after']['romp_runs_per_frame']:.2f} "
              f"mp_hit={r['after']['mp_hit_rate']:.2f} yolo={st['yolo_proposals']}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
