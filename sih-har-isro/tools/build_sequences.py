"""
Turn rendered Blender takes into LSTM training sequences.

Writes X_sequences.npy / y_labels.npy / groups.npy (+ metadata.json recording
the input contract) in the layout `train/train_lstm.py` expects.

Per take, per camera:

    frames/*.png (30 fps)
        |
        |  every frame, in order, through har_pipeline.perceive() - the live
        |  loop's own perception step: HSV on frame t, pose (config
        |  complexity, POSE_INPUT_WIDTH px) hinted with frame t-1's detections
        v
    HSV detections + raw 132-dim pose         <- cached per take (see below)
        |
        v
    pipeline.features.build_frame_features()  <- the SAME call har_pipeline
        |  per-frame model vector (132 or 143)   makes on every live frame
        v
    pipeline.features.windows()               <- the live buffer, replayed:
        |  frames stamped i / fps, resampled      resample, window, purity
        |  onto the --sample-every / FPS clock,
        |  30-sample windows, --stride samples
        v
    X (N, 30, dim), y (N,), groups (N,) = take index

Three things this does that the old `data_generation/synthetic_pose.py` path
could not:

1.  `--pose-source mediapipe` runs the real pose estimator over the rendered
    frames, so the training distribution is what the deployed pipeline will
    actually see: MediaPipe's output, with its jitter, its dropouts and its
    confusion on an unusual body orientation. Training on ground-truth pose
    and deploying on MediaPipe output is a train/serve skew no amount of
    held-out splitting will reveal. (`groundtruth` exists for the oracle
    experiment; it still takes its boxes from HSV, like evaluate_real's
    oracle path.)
    `--pose-source hmr` runs the trained HMR backend (HMR_TRAIN_BACKEND, in
    pipeline/hmr_backend.py) the way the live loop's HMR branch does, over
    every take except the tags in HMR_UNRELIABLE_TAGS, and records
    pose_source/hmr_backend in metadata.json;
    train_lstm.py copies them into the checkpoint and the pipeline refuses to
    serve it under any other HMR_BACKEND. Needs D:/SIH-2026/.venv-hmr.

2.  `groups.npy` is the TAKE index. train_lstm.py splits with
    GroupShuffleSplit, so a whole take - one crew orientation, one lighting
    condition, one complete performance - lands entirely in train or entirely
    in val. Splitting by window instead lets two overlapping windows from the
    same second of the same take sit on opposite sides of the split.

3.  Feature building and windowing are not implemented here at all. They live
    in pipeline/features.py and are shared with the live pipeline and the
    evaluator, so the three cannot disagree about what a model input is.

Pose cache: running MediaPipe over every frame dominates a rebuild. The raw
per-frame pose and HSV detections are cached in <take>/<camera>/pose_cache/,
keyed by (dataset path, take, camera, pose source, pose complexity, pose input
width, pose confidences, upright, frame count, newest frame mtime). Pose
settings are not flags: they are config, identical to the live loop. Rack normalisation, layout and --sample-every are
applied downstream of the cache, so changing them reuses it instead of
re-running MediaPipe. `--no-cache` forces a fresh pass.

Usage:
    python tools/build_sequences.py --dataset dataset/blender_core \
        --out dataset/seq_blender_core --pose-source mediapipe --rack-normalize \
        --holdout-tags orient_180,orient_-60,yaw_90,pitch_60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.experiment_config import (  # noqa: E402
    DETECTOR_BACKEND, FEATURE_LAYOUT, FPS, FRAME_HEIGHT, FRAME_WIDTH, MEDIAPIPE_MIN_DET_CONF,
    MEDIAPIPE_MIN_TRK_CONF, MEDIAPIPE_MODEL_COMPLEXITY, POSE_INPUT_WIDTH, SAMPLE_EVERY,
    SEQUENCE_WINDOW, SKELETON_FEATURES, WINDOW_PURITY,
)
from pipeline.features import LAYOUT_DIMS, build_frame_features, stack_windows  # noqa: E402
from pipeline.hmr_backend import ROMP_CENTER_THRESH, ROMP_CONTRACT_VERSION  # noqa: E402

# Bump when anything upstream of the cache changes what it would contain
# (HSV thresholds, pose backend wiring).
# ponytail: manual version, not a hash of the detector/pose code; bump it by hand.
CACHE_VERSION = 2   # 2: shared perceive() (previous-frame pose hint), POSE_INPUT_WIDTH, live frame size

# --pose-source hmr: the HMR backend training runs, i.e. the HMR_BACKEND a
# model built from it must be served under (the pipeline refuses otherwise).
# Was "romp" (the pretrained baseline); switched to "custom" once our own
# HMR3DNet (train/hmr3d_model.py) was trained and measured to match-or-beat
# ROMP on every one of ROMP's problem takes (docs/architecture.html section 3).
HMR_TRAIN_BACKEND = "custom"
HMR_VISIBILITY = ("hmr_only_binary: vis=1.0 on the 27 slots with a ROMP joint, 0.0 on the 6 "
                  "without one (eye inner/outer, mouth corners; their x/y/z copy the nearest "
                  "measured point). A model must be trained on HMR data from scratch; serving "
                  "it under MediaPipe (or a MediaPipe model under HMR) is refused at load. "
                  "UNVALIDATED until a model trained on it is scored.")
# Takes on which the trained HMR backend (see HMR_TRAIN_BACKEND) is still
# unreliable (hmr-env, blender_core, confirmed by visual overlay and by
# tools/measure_hmr_crop.py, not confidence). Keyed by tag, not take index:
# the failure is the body orientation / lighting, and tags are what carry
# that across datasets. These are left out of HMR training rather than fed
# in as mislabelled skeletons. The live HMR path will still meet these
# conditions: a model trained without them has never seen them, which is a
# coverage gap, not a solved problem.
# Updated after the denser-angle retrain (10 more takes at 40/50/65/75/85 deg
# roll): orient_30/60/180, pitch_60 and light_single_fail_180 are no longer
# unreliable for the custom model (0.77-1.00 tracked) - only exactly 90 deg
# roll and dim light at 90 deg remain a real, unclosed gap (see
# docs/architecture.html section 3). orient_-90 stayed on this list despite
# partially improving (0.00 -> 0.22): still too unreliable to train on.
HMR_UNRELIABLE_TAGS = {
    **{t: "custom HMR3DNet is still at or near 0.00 tracked at this exact crew roll"
       for t in ("orient_90", "orient_-90")},
    "light_dim_90": "custom HMR3DNet is still at or near 0.00 tracked under dim light at 90 deg roll",
}


def _load_take(take_dir: Path, camera: Optional[str] = None) -> Optional[dict]:
    meta_path = take_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cam = camera or meta["cameras"][0]
    cam_dir = take_dir / cam
    # labels, not pose_2d: a real recorded take (tools/record_take.py) has no
    # pose ground truth and must still be buildable from MediaPipe output.
    if not (cam_dir / "labels.npy").exists():
        return None
    return {"meta": meta, "camera": cam, "dir": cam_dir}


def _frame_inputs(dataset_dir: str, take_dir: Path, cam_dir: Path, pose_source: str,
                  upright: bool, use_cache: bool = True):
    """Raw per-frame (pose 132, HSV detections) for a take, cached.

    -> (poses (n, 132) float32, detections list[list[Detection]], (w, h), stats)
    """
    import cv2

    frames = sorted((cam_dir / "frames").glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no rendered frames under {cam_dir / 'frames'}")

    from pipeline.har_pipeline import OptimizedMPWrapper, perceive, to_frame_size, _make_box_detector

    # Frames are brought to the live frame size first (to_frame_size), exactly
    # as the live loop does to every source, so the detector and pose see the
    # same pixels. _make_box_detector is the SAME DETECTOR_BACKEND switch the
    # live pipeline uses - building sequences with HSV while serving YOLO (or
    # the reverse) would be exactly the train/serve skew this project has
    # already been burned by once.
    w, h = FRAME_WIDTH, FRAME_HEIGHT
    detector = _make_box_detector(w, h)

    # _make_box_detector silently falls back HSV<-YOLO on any load error (a
    # reasonable default for the LIVE pipeline, where uptime beats a hard
    # crash) - but an offline BUILDER doing that silently is exactly the
    # train/serve skew above: a rebuild would write HSV-derived training data
    # while DETECTOR_BACKEND says "yolo" and the checkpoint contract would
    # never know. Confirmed cause of a real mismatch found in this codebase
    # (2026-09-27 pipeline-engineer audit): the deployed LSTM was trained on
    # HSV boxes and is now served on YOLO. Refuse to build silently-wrong data.
    detector_backend = type(detector).__name__  # "YOLOBoxDetector" or "HSVBoxDetector"
    expected = {"yolo": "YOLOBoxDetector", "hsv": "HSVBoxDetector"}.get(DETECTOR_BACKEND)
    if expected is not None and detector_backend != expected:
        raise SystemExit(
            f"DETECTOR_BACKEND={DETECTOR_BACKEND!r} but _make_box_detector() silently "
            f"fell back to {detector_backend} (missing weights or import error - see the "
            f"warning above). Building training sequences on the wrong detector reproduces "
            f"the exact train/serve mismatch this project has already shipped once. Fix the "
            f"detector, don't build data on the fallback.")

    # Weights identity: mtime+size (cheap, no need to hash 18 MB) so a
    # re-fine-tuned YOLO checkpoint invalidates every cache that used the old
    # weights, rather than silently scoring/training on stale detections.
    detector_weights_id = None
    if detector_backend == "YOLOBoxDetector":
        # _make_box_detector() never overrides weights, so this is always the
        # file actually loaded above.
        from pipeline.yolo_detector import DEFAULT_WEIGHTS
        wp = Path(DEFAULT_WEIGHTS)
        if wp.exists():
            st = wp.stat()
            detector_weights_id = f"{wp.name}:{st.st_size}:{int(st.st_mtime)}"

    key = {"v": CACHE_VERSION, "dataset": str(Path(dataset_dir).resolve()),
           "take": take_dir.name, "camera": cam_dir.name, "pose_source": pose_source,
           "complexity": int(MEDIAPIPE_MODEL_COMPLEXITY),
           "pose_input_width": int(POSE_INPUT_WIDTH),
           "det_conf": MEDIAPIPE_MIN_DET_CONF, "trk_conf": MEDIAPIPE_MIN_TRK_CONF,
           "upright": bool(upright), "frame_wh": [FRAME_WIDTH, FRAME_HEIGHT],
           "n_frames": len(frames), "last_mtime": frames[-1].stat().st_mtime,
           "detector_backend": detector_backend, "detector_weights_id": detector_weights_id}
    if pose_source == "hmr":   # only for hmr: existing MediaPipe caches keep their digests
        key.update(hmr_backend=HMR_TRAIN_BACKEND, hmr_contract=ROMP_CONTRACT_VERSION,
                   center_thresh=ROMP_CENTER_THRESH)
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = cam_dir / "pose_cache" / f"{digest}.pkl"
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("key") == key:
            cached["stats"]["cache"] = "hit"
            return cached["poses"], cached["dets"], tuple(cached["wh"]), cached["stats"]
    # New YOLO weights change only the detections: with YOLO the pose never
    # depends on them (the upright estimator's hint reads Detection.rect,
    # which YOLO never sets), so a cache that differs ONLY in
    # detector_weights_id already holds the exact poses. Reuse them and
    # re-run just the detector - minutes instead of a full MediaPipe pass.
    reuse_poses = None
    if use_cache and pose_source == "mediapipe" and detector_backend == "YOLOBoxDetector":
        want = {k: v for k, v in key.items() if k != "detector_weights_id"}
        for old in sorted((cam_dir / "pose_cache").glob("*.pkl")):
            try:
                with open(old, "rb") as fh:
                    cand = pickle.load(fh)
            except Exception:
                continue
            ck = cand.get("key", {})
            if ck.get("detector_backend") == "YOLOBoxDetector" and                     {k: v for k, v in ck.items() if k != "detector_weights_id"} == want:
                reuse_poses = cand["poses"]
                break
    if reuse_poses is not None:
        t0 = time.time()
        dets_all = [detector.detect(to_frame_size(cv2.imread(str(f)))) for f in frames[:len(reuse_poses)]]
        stats = {"frames": len(reuse_poses), "pose_detected": int(np.any(reuse_poses, axis=1).sum()),
                 "detect_rate": round(float(np.any(reuse_poses, axis=1).mean()), 4),
                 "seconds": round(time.time() - t0, 1), "backend": "tasks (poses reused)",
                 "upright": bool(upright), "detector_backend": detector_backend,
                 "detector_weights_id": detector_weights_id}
        cache_path.parent.mkdir(exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"key": key, "poses": reuse_poses, "dets": dets_all, "wh": (w, h),
                         "stats": dict(stats)}, fh)
        tmp.replace(cache_path)
        stats["cache"] = "poses-reused"
        return reuse_poses, dets_all, (w, h), stats

    gt = None
    backend = None
    hmr = None
    if pose_source == "groundtruth":
        gt = np.load(cam_dir / "pose_2d.npy").astype(np.float32)
    elif pose_source == "hmr":
        from pipeline.hmr_backend import HMRBackend
        hmr = HMRBackend(backend=HMR_TRAIN_BACKEND)
        if not hmr.available:   # never fall back: that would label MediaPipe output "hmr"
            raise SystemExit(f"--pose-source hmr: HMR backend {HMR_TRAIN_BACKEND!r} did not load "
                             f"(run under D:/SIH-2026/.venv-hmr with its Scripts dir first on PATH)")
    else:
        # The live pose stage with config defaults: nothing here to drift.
        backend = OptimizedMPWrapper(frame_wh=(w, h), upright=upright)

    n = len(frames) if gt is None else min(len(frames), len(gt))
    poses = np.zeros((n, SKELETON_FEATURES), dtype=np.float32)
    dets_all = []
    detected = 0
    prev = None      # frame 0 gets no pose hint, exactly as live
    t0 = time.time()
    for i in range(n):
        img = to_frame_size(cv2.imread(str(frames[i])))
        if gt is not None:
            dets, feats = detector.detect(img), gt[i]
        elif hmr is not None:
            # The live loop's HMR branch (HARPipeline.process_frame), step for step.
            dets = detector.detect(img)
            feats, _ = hmr.get_pose_features(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            dets, feats = perceive(img, cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                   detector, backend, prev)
        prev = dets
        detected += int(bool(np.any(feats)))
        poses[i] = feats
        dets_all.append(dets)
    if backend is not None:
        backend.close()

    stats = {
        "frames": n,
        "pose_detected": detected,
        "detect_rate": round(detected / max(n, 1), 4),
        "seconds": round(time.time() - t0, 1),
        "backend": ("blender_groundtruth" if gt is not None
                    else hmr.backend_name if hmr is not None else backend.backend_name),
        "upright": bool(upright),
        "detector_backend": detector_backend,
        "detector_weights_id": detector_weights_id,
    }
    if use_cache:
        cache_path.parent.mkdir(exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"key": key, "poses": poses, "dets": dets_all, "wh": (w, h),
                         "stats": dict(stats)}, fh)
        tmp.replace(cache_path)
    stats["cache"] = "miss" if use_cache else "off"
    return poses, dets_all, (w, h), stats


def take_features(poses: np.ndarray, dets: list, wh: Tuple[int, int],
                  rack_normalize: bool, layout: str) -> np.ndarray:
    """Per-frame model vectors for one take, through the shared builder. One
    fresh normaliser per take: its EMA and latched polarity are per session,
    exactly as HARPipeline.reset_runtime() resets them per trial."""
    from pipeline.rack_frame import RackFrameNormalizer
    from config.experiment_config import RACK_ANGLE_EMA, RACK_SCALE_EMA
    normalizer = (RackFrameNormalizer(angle_ema=RACK_ANGLE_EMA, scale_ema=RACK_SCALE_EMA)
                  if rack_normalize else None)
    out = np.zeros((len(poses), LAYOUT_DIMS[layout]), dtype=np.float32)
    for i in range(len(poses)):
        out[i] = build_frame_features(poses[i], dets[i], wh, normalizer, layout)
    return out


def build(dataset_dir: str, out_dir: str, pose_source: str = "mediapipe",
          stride: int = 1, camera: Optional[str] = None,
          purity: float = WINDOW_PURITY, upright: bool = True,
          holdout_tags: Sequence[str] = (), rack_normalize: bool = False,
          exclude_tags: Sequence[str] = (), sample_every: int = SAMPLE_EVERY,
          layout: str = FEATURE_LAYOUT, use_cache: bool = True) -> dict:
    root = Path(dataset_dir)
    takes = sorted(p for p in root.glob("take_*") if p.is_dir())
    if not takes:
        raise SystemExit(f"no takes under {root}")
    if pose_source == "hmr":
        upright = False   # the live HMR branch never canonicalises the frame
        lost = sorted(set(holdout_tags) & set(HMR_UNRELIABLE_TAGS))
        if lost:
            print(f"  WARNING: holdout tags {lost} are HMR-unreliable and will be skipped; "
                  f"the holdout set will not contain them")
    hmr_excluded: List[dict] = []

    out_train = Path(out_dir)
    out_test = Path(f"{out_dir}_holdout")
    for d in (out_train, out_test):
        d.mkdir(parents=True, exist_ok=True)

    bucket = {"train": {"x": [], "y": [], "g": []},
              "holdout": {"x": [], "y": [], "g": []}}
    report: List[dict] = []
    run_detector_backend = None
    run_detector_weights_id = None

    for take_dir in takes:
        loaded = _load_take(take_dir, camera)
        if loaded is None:
            continue
        meta, cam_dir = loaded["meta"], loaded["dir"]
        tag = meta.get("tag", "nominal")
        if tag in exclude_tags:
            print(f"  skip {take_dir.name} (tag={tag}, excluded)")
            continue
        if pose_source == "groundtruth" and meta.get("real"):
            print(f"  skip {take_dir.name} (real take has no pose ground truth)")
            continue
        if pose_source == "hmr" and tag in HMR_UNRELIABLE_TAGS:
            reason = HMR_UNRELIABLE_TAGS[tag]
            print(f"  SKIP {take_dir.name} (tag={tag}): HMR-unreliable, not trained on: {reason}")
            hmr_excluded.append({"take": take_dir.name, "index": meta["index"], "tag": tag,
                                 "reason": reason})
            continue
        if not meta.get("frames_written", True):
            print(f"  skip {take_dir.name} (no frames rendered)")
            continue

        labels = np.load(cam_dir / "labels.npy")
        poses, dets, wh, stats = _frame_inputs(dataset_dir, take_dir, cam_dir, pose_source,
                                               upright, use_cache)
        n = min(len(poses), len(labels))
        feats = take_features(poses[:n], dets[:n], wh, rack_normalize, layout)

        # Rendered frame i was captured at i / fps; the window resamples that
        # onto the same sample_every / FPS clock the live pipeline uses.
        stamps = np.arange(n) / float(meta.get("fps", FPS))
        xs, ys = stack_windows(feats, labels[:n], stride=stride, purity=purity,
                               sample_every=sample_every, timestamps=stamps)
        split = "holdout" if tag in holdout_tags else "train"
        if len(xs):
            bucket[split]["x"].append(xs)
            bucket[split]["y"].append(ys)
            bucket[split]["g"].append(np.full(len(xs), meta["index"], dtype=np.int64))

        row = {"take": take_dir.name, "index": meta["index"], "tag": tag,
               "roll": meta.get("body_roll_deg"), "lighting": meta.get("lighting"),
               "distractors": meta.get("distractors"), "camera": loaded["camera"],
               "windows": int(len(xs)), "split": split, **stats}
        report.append(row)
        if run_detector_backend is None:
            run_detector_backend = stats.get("detector_backend")
            run_detector_weights_id = stats.get("detector_weights_id")
        elif stats.get("detector_backend") != run_detector_backend:
            # Every take in one build() call shares the same live _make_box_detector
            # instance/config, so this can only happen if the underlying weights
            # file changed mid-run - a build that started under one detector must
            # not silently finish under another.
            raise SystemExit(
                f"{take_dir.name}: detector_backend changed mid-build "
                f"({run_detector_backend!r} -> {stats.get('detector_backend')!r}). "
                f"Something replaced the weights file while this build was running.")
        print(f"  {take_dir.name} tag={tag:16s} -> {len(xs):4d} windows "
              f"[{split}] detect={stats.get('detect_rate')} cache={stats['cache']}",
              flush=True)

    contract = {"feature_layout": layout, "sample_every": int(sample_every),
                "rack_normalized": bool(rack_normalize),
                "detector_backend": run_detector_backend,
                "detector_weights_id": run_detector_weights_id}
    # pose_source (+ hmr_backend) is what train_lstm copies into the checkpoint
    # and the pipeline checks against the runtime HMR_BACKEND at load.
    if pose_source == "hmr":
        pose_cfg = {"pose_source": "hmr", "hmr_backend": HMR_TRAIN_BACKEND,
                    "hmr_contract": ROMP_CONTRACT_VERSION, "center_thresh": ROMP_CENTER_THRESH,
                    "upright": False, "visibility": HMR_VISIBILITY,
                    "hmr_excluded_takes": hmr_excluded}
    else:
        pose_cfg = {"pose_source": pose_source, "pose_complexity": MEDIAPIPE_MODEL_COMPLEXITY,
                    "pose_input_width": POSE_INPUT_WIDTH, "upright": bool(upright)}
    written = {}
    for split, target in (("train", out_train), ("holdout", out_test)):
        if not bucket[split]["x"]:
            continue
        X = np.concatenate(bucket[split]["x"])
        y = np.concatenate(bucket[split]["y"])
        g = np.concatenate(bucket[split]["g"])
        np.save(target / "X_sequences.npy", X)
        np.save(target / "y_labels.npy", y)
        np.save(target / "groups.npy", g)
        meta_out = {
            "total_windows": int(len(X)),
            "n_source_takes": int(len(np.unique(g))),
            "feature_dim": int(X.shape[-1]),
            "window": SEQUENCE_WINDOW,
            "stride": stride,
            "purity": purity,
            "source": f"blender_render/{pose_source}"
                      + (f":{HMR_TRAIN_BACKEND}" if pose_source == "hmr" else ""),
            **contract,
            **pose_cfg,
            "class_counts": {int(k): int(v)
                             for k, v in zip(*np.unique(y, return_counts=True))},
        }
        (target / "metadata.json").write_text(json.dumps(meta_out, indent=2),
                                              encoding="utf-8")
        written[split] = {"dir": str(target), **meta_out}
        print(f"{split}: X={X.shape} y={y.shape} takes={meta_out['n_source_takes']}")

    summary = {"pose_source": pose_source, "stride": stride, **contract,
               "holdout_tags": list(holdout_tags), "takes": report,
               "hmr_excluded_takes": hmr_excluded,
               "written": written}
    (out_train / "build_report.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/blender")
    ap.add_argument("--out", default="dataset/sequences_mp")
    ap.add_argument("--pose-source", default="mediapipe",
                    choices=["mediapipe", "groundtruth", "hmr"],
                    help=f"hmr = HMR backend {HMR_TRAIN_BACKEND!r}, run under .venv-hmr; "
                         f"skips HMR-unreliable tags; serve the model with "
                         f"HMR_BACKEND={HMR_TRAIN_BACKEND!r}")
    ap.add_argument("--stride", type=int, default=1,
                    help="Window stride in SAMPLES (after --sample-every)")
    ap.add_argument("--sample-every", type=int, default=SAMPLE_EVERY,
                    help=f"Keep every Nth frame when windowing (config SAMPLE_EVERY="
                         f"{SAMPLE_EVERY}; the live pipeline refuses a model trained "
                         f"at any other rate)")
    ap.add_argument("--layout", default=FEATURE_LAYOUT, choices=sorted(LAYOUT_DIMS),
                    help=f"Model input layout (config FEATURE_LAYOUT={FEATURE_LAYOUT})")
    ap.add_argument("--purity", type=float, default=WINDOW_PURITY,
                    help="Fraction of a window that must share the majority "
                         "step label for it to be kept (config WINDOW_PURITY)")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--rack-normalize", action="store_true")
    ap.add_argument("--no-upright", action="store_true",
                    help="Disable image canonicalisation before pose "
                         "(see config.UPRIGHT_POSE for why it is default on)")
    ap.add_argument("--no-cache", action="store_true",
                    help="Re-run pose even when a matching per-take cache exists")
    ap.add_argument("--holdout-tags", default="",
                    help="Comma-separated take tags to route into the holdout set")
    ap.add_argument("--exclude-tags", default="skip,recover,heldout_view",
                    help="Tags to leave out entirely (error sequences are for "
                         "state-machine evaluation, and heldout_view is an "
                         "unseen-camera test, neither is step training)")
    args = ap.parse_args()

    holdout = [t for t in args.holdout_tags.split(",") if t]
    exclude = [t for t in args.exclude_tags.split(",") if t]
    build(args.dataset, args.out, args.pose_source, args.stride,
          args.camera, args.purity, not args.no_upright,
          holdout, args.rack_normalize, exclude, args.sample_every,
          args.layout, not args.no_cache)


if __name__ == "__main__":
    main()
