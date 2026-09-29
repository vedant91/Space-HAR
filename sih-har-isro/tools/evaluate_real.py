"""
Honest end-to-end evaluation against rendered video with known ground truth.

This replaces the project's existing self-referential measurements. To be
explicit about what was wrong with those, since the point of this file is to
not repeat it:

  * `lstm_test_acc` was measured on `synthetic_pose.generate_dataset(seed=99)`
    while training used `seed=7` — the same closed-form generator, so the
    model was scored on its ability to invert a 24-waypoint interpolator.
  * `oracle_step_acc` came from `run_pipeline_on_clip(inject_pose=True)`,
    which feeds ground-truth pose vectors straight into the LSTM. MediaPipe
    is bypassed, so the camera->pose stage — the actual deliverable — was
    never measured at all.
  * `hsv_*_recall` was measured on frames whose boxes were painted in colours
    chosen to sit inside the detector's own thresholds, drawn last so nothing
    could occlude them.

Everything here is measured against Blender ground truth that was produced by
a different process than the thing being scored:

  pose_mpjpe        MediaPipe's landmark error vs the true 3-D-derived
                    projection. Never previously measured.
  step_acc_real     camera -> MediaPipe -> LSTM -> step id.
  step_acc_oracle   ground-truth pose -> LSTM -> step id. The GAP between
                    these two is precisely the cost of the perception stage,
                    which is the number the old harness hid.
  hsv_*             detector vs true boxes and true occlusion.
  protocol          the state machine driven by the REAL prediction stream on
                    the skip / recover takes, not by scripted perfect input.
  latency           per-frame, on real frames, broken down by stage.

Usage:
    python tools/evaluate_real.py --dataset dataset/blender \
        --model models/lstm_classifier.pt --out logs/real_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(Path(__file__).resolve().parent))   # tools/ (no package): build_sequences

from config.experiment_config import (  # noqa: E402
    FPS, FRAME_HEIGHT, FRAME_WIDTH, MEDIAPIPE_MODEL_COMPLEXITY, MODEL_FEATURE_DIM, POSE_INPUT_WIDTH,
    SKELETON_FEATURES, STEP_CONFIDENCE_THRESHOLD, WINDOW_PURITY,
)
from pipeline.features import (  # noqa: E402
    LAYOUT_DIMS, build_frame_features, contract_of, windows,
)

NUM_LANDMARKS = 33
# Landmarks whose accuracy actually matters for this protocol: the arms drive
# every step, the shoulders/hips set the body frame. Face and feet are
# reported separately rather than being allowed to flatter the mean.
KEY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24]


# ── LSTM ─────────────────────────────────────────────────────────────────────


class LSTMRunner:
    """Loads the trained checkpoint and classifies 30-frame windows."""

    def __init__(self, model_path: str, runtime_backend: str = "none"):
        import torch
        from train.train_lstm import HARLSTMClassifier

        self.torch = torch
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        # runtime_backend is the pose stage this run scores through: "none"
        # (MediaPipe, OptimizedMPWrapper) or the trained HMR backend, see
        # build_sequences.HMR_TRAIN_BACKEND (--pose-source hmr).
        from pipeline.hmr_backend import check_pose_source
        check_pose_source(model_path, ckpt, runtime_backend)
        self.model = HARLSTMClassifier(
            feature_dim=ckpt["feature_dim"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            num_classes=ckpt["num_classes"],
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.idx_to_label = {int(k): int(v) for k, v in ckpt.get("idx_to_label", {}).items()}
        self.train_val_acc = float(ckpt.get("val_acc", 0.0))
        self.feature_dim = int(ckpt["feature_dim"])
        # What this checkpoint was trained on. A checkpoint with no recorded
        # contract (e.g. models/lstm_final_rack.pt, the 0.201 baseline) is the
        # legacy one: 132-dim pose only, every frame. Its inputs are built
        # that way here, so it stays scorable next to new models.
        self.contract = contract_of(ckpt)
        want = LAYOUT_DIMS.get(self.contract["feature_layout"])
        if want != self.feature_dim:
            raise SystemExit(f"{model_path}: layout {self.contract['feature_layout']!r} "
                             f"is {want}-dim but the model takes {self.feature_dim}")
        if self.feature_dim != MODEL_FEATURE_DIM:
            print(f"[i] {model_path}: layout {self.contract['feature_layout']!r} "
                  f"({self.feature_dim}-dim), not the deployed {MODEL_FEATURE_DIM}-dim layout")

    def predict(self, window: np.ndarray):
        """(T, F) -> (step_id, confidence)."""
        x = self.torch.from_numpy(window[None, ...].astype(np.float32))
        with self.torch.no_grad():
            logits = self.model(x)
            probs = self.torch.softmax(logits, dim=1)[0]
            conf, idx = probs.max(0)
        idx = int(idx.item())
        return int(self.idx_to_label.get(idx, idx + 1)), float(conf.item())


# ── per-take evaluation ──────────────────────────────────────────────────────


def evaluate_take(take_dir: Path, runner: Optional[LSTMRunner],
                  camera: Optional[str] = None,
                  rack_normalize: bool = False,
                  max_frames: int = 0,
                  upright: bool = True,
                  pose_source: str = "mediapipe") -> Optional[dict]:
    import cv2
    from pipeline.har_pipeline import OptimizedMPWrapper, perceive, to_frame_size, _make_box_detector
    from pipeline.rack_frame import RackFrameNormalizer

    meta_path = take_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("frames_written", True):
        return None
    cam = camera or meta["cameras"][0]
    cam_dir = take_dir / cam

    gt_pose = np.load(cam_dir / "pose_2d.npy")
    labels = np.load(cam_dir / "labels.npy")
    payload_gt = json.loads((cam_dir / "payload.json").read_text(encoding="utf-8"))
    frames = sorted((cam_dir / "frames").glob("*.png"))
    if not frames:
        return None

    n = min(len(frames), len(gt_pose), len(labels), len(payload_gt))
    if max_frames:
        n = min(n, max_frames)

    img0 = cv2.imread(str(frames[0]))
    h, w = img0.shape[:2]

    # Perception runs on frames brought to the live frame size (to_frame_size),
    # with the live pose stage's config defaults (complexity, POSE_INPUT_WIDTH).
    # w, h stay the render's native size for pixel pose error / truth boxes.
    fw, fh = FRAME_WIDTH, FRAME_HEIGHT
    box_scale = np.array([fw / w, fh / h, fw / w, fh / h])
    hmr_poses = hmr_dets = hmr_stats = backend = detector = None
    if pose_source == "hmr":
        # The HMR route is tools/build_sequences._frame_inputs, the exact code
        # that built the HMR training set: the configured detector backend +
        # HMRBackend(HMR_TRAIN_BACKEND) per frame, no upright (the live HMR
        # branch never canonicalises the frame). Cached per take, so a scoring
        # run under the system Python reuses an HMR pass made under
        # .venv-hmr; on a miss it runs that backend (and refuses if it does
        # not load, never falling back to MediaPipe).
        from build_sequences import _frame_inputs
        upright = False
        hmr_poses, hmr_dets, _, hmr_stats = _frame_inputs(
            str(take_dir.parent), take_dir, cam_dir, "hmr", upright=False)
        n = min(n, len(hmr_poses))
    else:
        backend = OptimizedMPWrapper(frame_wh=(fw, fh), upright=upright)
        detector = _make_box_detector(fw, fh)
        # Same fatal check as build_sequences._frame_inputs: a scoring run
        # must not silently score through a detector fallback different from
        # config.DETECTOR_BACKEND (confirmed real mismatch, 2026-09-27 audit).
        from config.experiment_config import DETECTOR_BACKEND as _DB
        _expected = {"yolo": "YOLOBoxDetector", "hsv": "HSVBoxDetector"}.get(_DB)
        _actual = type(detector).__name__
        if _expected is not None and _actual != _expected:
            raise SystemExit(
                f"DETECTOR_BACKEND={_DB!r} but _make_box_detector() silently fell back "
                f"to {_actual} - scoring on the fallback would report a number that "
                f"doesn't reflect the configured detector. Fix the detector, don't "
                f"score on the fallback.")
    normalizer = RackFrameNormalizer() if rack_normalize else None
    layout = runner.contract["feature_layout"] if runner is not None else "pose132"
    dim = LAYOUT_DIMS[layout]

    # Two copies of MediaPipe's output are needed once rack normalisation is
    # in play. `mp_raw` stays in image coordinates so it can be compared
    # against the ground truth for landmark error; `mp_poses` is the model
    # input, built by the same pipeline.features call the live pipeline makes.
    # Scoring normalised coordinates against raw ones silently produces a
    # nonsense error (~700 px on a 900 px frame, because rack-frame units are
    # torso-lengths, not pixels).
    mp_poses = np.zeros((n, dim), dtype=np.float32)
    mp_raw = np.zeros((n, SKELETON_FEATURES), dtype=np.float32)
    gt_for_model = np.zeros((n, dim), dtype=np.float32)
    gt_normalizer = RackFrameNormalizer() if rack_normalize else None
    timings = defaultdict(list)
    hsv_counts = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                  for k in ("red_box", "yellow_box", "main_box")}
    iou_sums = defaultdict(list)
    detected = 0
    prev = None      # frame 0 gets no pose hint, exactly as live

    for i in range(n):
        if hmr_poses is not None:
            dets, feats = hmr_dets[i], hmr_poses[i]
        else:
            img = to_frame_size(cv2.imread(str(frames[i])))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # The live loop's own perception step (HSV on this frame, pose
            # hinted with the previous frame's detections).
            tm = {}
            dets, feats = perceive(img, rgb, detector, backend, prev, timings=tm)
            prev = dets
            timings["hsv"].append(tm["hsv_ms"])
            timings["pose"].append(tm["pose_ms"])

        if np.any(feats):
            detected += 1
        mp_raw[i] = feats
        mp_poses[i] = build_frame_features(feats, dets, (fw, fh), normalizer, layout)
        # The oracle path must see the ground truth put through the SAME
        # builder (same rack rect, same boxes, own normaliser state) as the
        # live path, or it measures a feature-space mismatch rather than the
        # cost of perception. Only the pose is swapped; boxes stay HSV.
        gt_for_model[i] = build_frame_features(gt_pose[i], dets, (fw, fh),
                                               gt_normalizer, layout)

        # HSV vs true visibility + true boxes
        by_label = {d.label: d for d in dets}
        for key in hsv_counts:
            truth = bool(payload_gt[i][key]["visible"])
            pred = key in by_label
            if truth and pred:
                hsv_counts[key]["tp"] += 1
                truth_box = np.asarray(payload_gt[i][key]["bbox"], dtype=float) * box_scale
                iou_sums[key].append(_iou(by_label[key].bbox, truth_box))
            elif truth and not pred:
                hsv_counts[key]["fn"] += 1
            elif (not truth) and pred:
                hsv_counts[key]["fp"] += 1
            else:
                hsv_counts[key]["tn"] += 1

    if backend is not None:
        backend.close()
    latency = {k: {"mean": round(float(np.mean(v)), 2),
                   "p95": round(float(np.percentile(v, 95)), 2)}
               for k, v in timings.items()}
    if hmr_stats is not None:
        # Whole-pass wall time of the HMR run that filled the cache (imread +
        # resize + detector + HMR per frame); not separable per stage.
        latency["hmr_pass_incl_io_hsv"] = {
            "mean": round(1000.0 * hmr_stats["seconds"] / max(hmr_stats["frames"], 1), 2)}

    result = {
        "take": take_dir.name,
        "index": meta["index"],
        "tag": meta.get("tag", "nominal"),
        "body_roll_deg": meta.get("body_roll_deg"),
        "lighting": meta.get("lighting"),
        "distractors": meta.get("distractors"),
        "camera": cam,
        "frames": n,
        "pose_detect_rate": round(detected / max(n, 1), 4),
        "pose_backend": hmr_stats["backend"] if hmr_stats is not None else backend.backend_name,
        "upright_pose": bool(upright),
        "latency_ms": latency,
        "hsv": _hsv_summary(hsv_counts, iou_sums),
        "pose_error": _pose_error(mp_raw, gt_pose[:n], w, h),
    }

    if runner is not None:
        stamps = np.arange(n) / float(meta.get("fps", FPS))   # render frame i at i / fps
        result["step"] = _step_accuracy(runner, mp_poses, gt_for_model, labels[:n], stamps)
        result["protocol"] = _protocol_outcome(runner, mp_poses, labels[:n], stamps,
                                               meta.get("sequence", []))
    return result


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = ua + ub - inter
    return float(inter / denom) if denom > 0 else 0.0


def _hsv_summary(counts: Dict[str, Dict[str, int]], ious) -> dict:
    out = {}
    for key, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        out[key] = {
            "recall": round(tp / max(tp + fn, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": c["tn"],
            "mean_iou": round(float(np.mean(ious[key])), 4) if ious[key] else 0.0,
        }
    return out


def _pose_error(mp_pose: np.ndarray, gt_pose: np.ndarray,
                width: int, height: int) -> dict:
    """MediaPipe vs ground truth, in pixels.

    Scored only on frames where MediaPipe returned anything and only on
    landmarks the renderer marked visible — grading a detector on landmarks
    that are genuinely hidden measures occlusion, not the detector.
    """
    mp = mp_pose.reshape(len(mp_pose), NUM_LANDMARKS, 4)
    gt = gt_pose.reshape(len(gt_pose), NUM_LANDMARKS, 4)

    got = np.any(mp.reshape(len(mp), -1) != 0.0, axis=1)
    visible = gt[..., 3] >= 0.9
    usable = visible & got[:, None]

    scale = np.array([width, height], dtype=np.float32)
    err = np.linalg.norm((mp[..., :2] - gt[..., :2]) * scale, axis=-1)

    def _stat(mask):
        vals = err[mask]
        if vals.size == 0:
            return None
        return {
            "mean_px": round(float(vals.mean()), 2),
            "median_px": round(float(np.median(vals)), 2),
            "p90_px": round(float(np.percentile(vals, 90)), 2),
            "pck@5pct": round(float((vals < 0.05 * width).mean()), 4),
            "n": int(vals.size),
        }

    key_mask = np.zeros_like(usable)
    key_mask[:, KEY_LANDMARKS] = True
    return {
        "all_visible_landmarks": _stat(usable),
        "key_landmarks": _stat(usable & key_mask),
        "frames_with_detection": int(got.sum()),
        "frames_total": int(len(mp)),
    }


def _step_accuracy(runner: LSTMRunner, mp_poses: np.ndarray,
                   gt_poses: np.ndarray, labels: np.ndarray,
                   stamps: np.ndarray, stride: int = 1) -> dict:
    """The headline comparison: real perception vs oracle pose, same model,
    same windows, same labels. The difference is the perception cost.

    Windows come from pipeline.features.windows() at the model's own sample
    rate and the one config purity (WINDOW_PURITY) that training used, so a
    3-second-trained model is scored on 3-second windows. Dominant-label
    windows rather than strictly pure ones: pure windows at stride 3 once
    yielded ONE scorable window for a whole take, and a per-take accuracy
    over n=1 is not a measurement.
    """
    res = {}
    for name, source in (("real", mp_poses), ("oracle", gt_poses)):
        correct = total = confident = 0
        confusion = defaultdict(int)
        for window, truth in windows(source, labels, stride=stride, purity=WINDOW_PURITY,
                                     sample_every=runner.contract["sample_every"],
                                     timestamps=stamps):
            pred, conf = runner.predict(window)
            total += 1
            if conf >= STEP_CONFIDENCE_THRESHOLD:
                confident += 1
            if pred == truth:
                correct += 1
            else:
                confusion[f"{truth}->{pred}"] += 1
        res[name] = {
            "accuracy": round(correct / max(total, 1), 4),
            "correct": correct,
            "windows": total,
            "confident_frac": round(confident / max(total, 1), 4),
            "top_confusions": dict(sorted(confusion.items(),
                                          key=lambda kv: -kv[1])[:5]),
        }
    res["perception_cost"] = round(res["oracle"]["accuracy"] - res["real"]["accuracy"], 4)
    return res


def _protocol_outcome(runner: LSTMRunner, mp_poses: np.ndarray,
                      labels: np.ndarray, stamps: np.ndarray,
                      sequence: Sequence[int]) -> dict:
    """Drive the real state machine with the REAL prediction stream.

    `simulation/space_sim.evaluate_state_machine()` feeds it hand-written
    perfect predictions at conf=0.92, which tests the FSM's own branching but
    says nothing about whether the deployed perception produces a clean
    enough stream to reach those branches. This does the latter.
    """
    from pipeline.state_machine import ExperimentStateMachine

    sm = ExperimentStateMachine()
    events = {"skipped": [], "out_of_sequence": [], "recovered": [], "completed": []}
    sm.on_step_skipped = lambda e, o: events["skipped"].append((e, o))
    sm.on_out_of_sequence = lambda e, o: events["out_of_sequence"].append((e, o))
    sm.on_step_recovered = lambda rec, o: events["recovered"].append(rec.step_id)
    sm.on_step_completed = lambda rec: events["completed"].append(rec.step_id)

    # One prediction per sample entering the window, exactly as the live
    # pipeline feeds its state machine.
    for window, _ in windows(mp_poses, None, stride=1, purity=0.0,
                             sample_every=runner.contract["sample_every"], timestamps=stamps):
        pred, conf = runner.predict(window)
        sm.feed_prediction(pred, conf)

    summary = sm.get_status_summary()
    return {
        "rendered_sequence": list(sequence),
        "steps_completed": sorted(set(events["completed"])),
        "n_completed": len({s["id"] for s in summary["steps"]
                            if s["status"] == "COMPLETED"}),
        "experiment_complete": bool(summary["experiment_complete"]),
        "skip_events": events["skipped"],
        "out_of_sequence_events": events["out_of_sequence"],
        "recovery_events": summary["recovery"]["events"],
        "recovered_steps": events["recovered"],
    }


# ── driver ───────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/blender")
    ap.add_argument("--model", default="models/lstm_classifier.pt")
    ap.add_argument("--out", default="logs/real_eval.json")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--rack-normalize", action="store_true")
    ap.add_argument("--no-upright", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--tags", default="", help="Only evaluate takes with these tags")
    ap.add_argument("--label", default="", help="Free-text label for this run")
    ap.add_argument("--pose-source", default="mediapipe", choices=["mediapipe", "hmr"],
                    help="hmr = the trained HMR backend (build_sequences.HMR_TRAIN_BACKEND) "
                         "via tools/build_sequences' HMR pass (the one HMR training used); "
                         "needs .venv-hmr unless every take is cached. The checkpoint must "
                         "have been trained on the same source.")
    args = ap.parse_args()
    if args.pose_source == "hmr":
        args.no_upright = True   # the HMR branch never canonicalises the frame

    runner = None
    if Path(args.model).exists():
        from build_sequences import HMR_TRAIN_BACKEND
        runner = LSTMRunner(args.model,
                            HMR_TRAIN_BACKEND if args.pose_source == "hmr" else "none")
        print(f"LSTM loaded: {args.model} (reported train val_acc={runner.train_val_acc:.3f}, "
              f"contract {runner.contract})")
        recorded = runner.contract["rack_normalized"]
        if recorded is not None and bool(recorded) != args.rack_normalize:
            # The checkpoint knows how it was trained; the flag is a guess.
            print(f"[!] --rack-normalize={args.rack_normalize} contradicts the checkpoint "
                  f"(trained with rack_normalized={recorded}); using the checkpoint's value.")
            args.rack_normalize = bool(recorded)
    else:
        print(f"[!] No model at {args.model} — pose/HSV metrics only.")

    only = {t for t in args.tags.split(",") if t}
    takes = sorted(p for p in Path(args.dataset).glob("take_*") if p.is_dir())
    rows = []
    for take_dir in takes:
        meta_path = take_dir / "meta.json"
        if only and meta_path.exists():
            tag = json.loads(meta_path.read_text(encoding="utf-8")).get("tag", "nominal")
            if tag not in only:
                continue
        row = evaluate_take(take_dir, runner, args.camera,
                            args.rack_normalize, args.max_frames,
                            upright=not args.no_upright, pose_source=args.pose_source)
        if row is None:
            continue
        rows.append(row)
        step = row.get("step", {})
        print(f"  {row['take']} tag={row['tag']:16s} "
              f"real={step.get('real', {}).get('accuracy')} "
              f"oracle={step.get('oracle', {}).get('accuracy')} "
              f"mpjpe={row['pose_error']['key_landmarks']}", flush=True)

    report = {
        "label": args.label,
        "model": args.model,
        "model_reported_val_acc": runner.train_val_acc if runner else None,
        "rack_normalize": args.rack_normalize,
        "model_contract": runner.contract if runner else None,
        "pose_source": args.pose_source,
        "window_purity": WINDOW_PURITY,
        "upright_pose": not args.no_upright,
        "pose_complexity": MEDIAPIPE_MODEL_COMPLEXITY,
        "pose_input_width": POSE_INPUT_WIDTH,
        "takes": rows,
        "aggregate": _aggregate(rows),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"\nWrote {out}")


def _aggregate(rows: List[dict]) -> dict:
    if not rows:
        return {}
    def _mean(path, default=None):
        vals = []
        for r in rows:
            cur = r
            for key in path:
                cur = (cur or {}).get(key) if isinstance(cur, dict) else None
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return round(float(np.mean(vals)), 4) if vals else default

    step_rows = [r for r in rows if "step" in r]

    def _pooled(source):
        """Sum correct / sum windows across takes.

        Averaging per-take rates would weight a take with one scorable
        window the same as one with two hundred.
        """
        c = sum(r.get("step", {}).get(source, {}).get("correct", 0) for r in step_rows)
        n = sum(r.get("step", {}).get(source, {}).get("windows", 0) for r in step_rows)
        return round(c / n, 4) if n else None

    pooled_real, pooled_oracle = _pooled("real"), _pooled("oracle")
    return {
        "n_takes": len(rows),
        "total_frames": int(sum(r["frames"] for r in rows)),
        "step_windows_scored": int(sum(
            r.get("step", {}).get("real", {}).get("windows", 0) for r in step_rows)),
        "step_acc_real": pooled_real,
        "step_acc_oracle": pooled_oracle,
        "step_acc_real_take_mean": _mean(["step", "real", "accuracy"]),
        "perception_cost": (round(pooled_oracle - pooled_real, 4)
                            if (pooled_real is not None and pooled_oracle is not None)
                            else None),
        "pose_detect_rate": _mean(["pose_detect_rate"]),
        "key_landmark_mean_px": _mean(["pose_error", "key_landmarks", "mean_px"]),
        "key_landmark_pck5": _mean(["pose_error", "key_landmarks", "pck@5pct"]),
        "hsv_red_recall": _mean(["hsv", "red_box", "recall"]),
        "hsv_red_precision": _mean(["hsv", "red_box", "precision"]),
        "hsv_yellow_recall": _mean(["hsv", "yellow_box", "recall"]),
        "hsv_yellow_precision": _mean(["hsv", "yellow_box", "precision"]),
        "hsv_main_recall": _mean(["hsv", "main_box", "recall"]),
        "pose_latency_ms_mean": _mean(["latency_ms", "pose", "mean"]),
        "hmr_pass_ms_mean": _mean(["latency_ms", "hmr_pass_incl_io_hsv", "mean"]),
        "hsv_latency_ms_mean": _mean(["latency_ms", "hsv", "mean"]),
        "experiments_completed": sum(
            1 for r in step_rows if r.get("protocol", {}).get("experiment_complete")),
    }


if __name__ == "__main__":
    main()
