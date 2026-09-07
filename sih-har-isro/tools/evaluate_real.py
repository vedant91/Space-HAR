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

from config.experiment_config import (  # noqa: E402
    SEQUENCE_WINDOW, SKELETON_FEATURES, STEP_CONFIDENCE_THRESHOLD,
)

NUM_LANDMARKS = 33
# Landmarks whose accuracy actually matters for this protocol: the arms drive
# every step, the shoulders/hips set the body frame. Face and feet are
# reported separately rather than being allowed to flatter the mean.
KEY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24]


# ── LSTM ─────────────────────────────────────────────────────────────────────


class LSTMRunner:
    """Loads the trained checkpoint and classifies 30-frame windows."""

    def __init__(self, model_path: str):
        import torch
        from train.train_lstm import HARLSTMClassifier

        self.torch = torch
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
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
                  complexity: int = 1,
                  rack_normalize: bool = False,
                  max_frames: int = 0) -> Optional[dict]:
    import cv2
    from pipeline.pose_backend import PoseBackend
    from pipeline.hsv_detector import HSVBoxDetector
    from pipeline.rack_frame import RackFrameNormalizer, pick_rack_rect

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

    backend = PoseBackend(complexity=complexity, min_det_conf=0.3,
                          min_trk_conf=0.3, downscale=1,
                          frame_width=w, frame_height=h)
    detector = HSVBoxDetector(frame_width=w, frame_height=h)
    normalizer = RackFrameNormalizer() if rack_normalize else None

    mp_poses = np.zeros((n, SKELETON_FEATURES), dtype=np.float32)
    timings = defaultdict(list)
    hsv_counts = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                  for k in ("red_box", "yellow_box", "main_box")}
    iou_sums = defaultdict(list)
    detected = 0

    for i in range(n):
        img = cv2.imread(str(frames[i]))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        dets = detector.detect(img)
        t_hsv = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        feats = backend.process(rgb, timestamp_ms=int(i * 1000 / 30))
        t_pose = (time.perf_counter() - t0) * 1000.0

        if np.any(feats):
            detected += 1
        if normalizer is not None:
            feats = normalizer.normalize(feats, rack_rect=pick_rack_rect(dets))
        mp_poses[i] = feats

        timings["hsv"].append(t_hsv)
        timings["pose"].append(t_pose)

        # HSV vs true visibility + true boxes
        by_label = {d.label: d for d in dets}
        for key in hsv_counts:
            truth = bool(payload_gt[i][key]["visible"])
            pred = key in by_label
            if truth and pred:
                hsv_counts[key]["tp"] += 1
                iou_sums[key].append(_iou(by_label[key].bbox, payload_gt[i][key]["bbox"]))
            elif truth and not pred:
                hsv_counts[key]["fn"] += 1
            elif (not truth) and pred:
                hsv_counts[key]["fp"] += 1
            else:
                hsv_counts[key]["tn"] += 1

    backend.close()

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
        "pose_backend": backend.backend,
        "latency_ms": {
            k: {"mean": round(float(np.mean(v)), 2),
                "p95": round(float(np.percentile(v, 95)), 2)}
            for k, v in timings.items()
        },
        "hsv": _hsv_summary(hsv_counts, iou_sums),
        "pose_error": _pose_error(mp_poses, gt_pose[:n], w, h),
    }

    if runner is not None:
        result["step"] = _step_accuracy(runner, mp_poses, gt_pose[:n], labels[:n])
        result["protocol"] = _protocol_outcome(runner, mp_poses, labels[:n],
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


def _windows(poses: np.ndarray, labels: np.ndarray, stride: int = 3):
    for start in range(0, len(poses) - SEQUENCE_WINDOW + 1, stride):
        lab = labels[start:start + SEQUENCE_WINDOW]
        if len(np.unique(lab)) != 1:
            continue
        yield poses[start:start + SEQUENCE_WINDOW], int(lab[-1])


def _step_accuracy(runner: LSTMRunner, mp_poses: np.ndarray,
                   gt_poses: np.ndarray, labels: np.ndarray,
                   stride: int = 3) -> dict:
    """The headline comparison: real perception vs oracle pose, same model,
    same windows, same labels. The difference is the perception cost."""
    res = {}
    for name, source in (("real", mp_poses), ("oracle", gt_poses)):
        correct = total = confident = 0
        confusion = defaultdict(int)
        for window, truth in _windows(source, labels, stride):
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
            "windows": total,
            "confident_frac": round(confident / max(total, 1), 4),
            "top_confusions": dict(sorted(confusion.items(),
                                          key=lambda kv: -kv[1])[:5]),
        }
    res["perception_cost"] = round(res["oracle"]["accuracy"] - res["real"]["accuracy"], 4)
    return res


def _protocol_outcome(runner: LSTMRunner, mp_poses: np.ndarray,
                      labels: np.ndarray, sequence: Sequence[int]) -> dict:
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

    for start in range(0, len(mp_poses) - SEQUENCE_WINDOW + 1):
        pred, conf = runner.predict(mp_poses[start:start + SEQUENCE_WINDOW])
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
    ap.add_argument("--complexity", type=int, default=1)
    ap.add_argument("--rack-normalize", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--tags", default="", help="Only evaluate takes with these tags")
    ap.add_argument("--label", default="", help="Free-text label for this run")
    args = ap.parse_args()

    runner = None
    if Path(args.model).exists():
        runner = LSTMRunner(args.model)
        print(f"LSTM loaded: {args.model} (reported train val_acc={runner.train_val_acc:.3f})")
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
        row = evaluate_take(take_dir, runner, args.camera, args.complexity,
                            args.rack_normalize, args.max_frames)
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
        "pose_complexity": args.complexity,
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
    return {
        "n_takes": len(rows),
        "total_frames": int(sum(r["frames"] for r in rows)),
        "step_acc_real": _mean(["step", "real", "accuracy"]),
        "step_acc_oracle": _mean(["step", "oracle", "accuracy"]),
        "perception_cost": _mean(["step", "perception_cost"]),
        "pose_detect_rate": _mean(["pose_detect_rate"]),
        "key_landmark_mean_px": _mean(["pose_error", "key_landmarks", "mean_px"]),
        "key_landmark_pck5": _mean(["pose_error", "key_landmarks", "pck@5pct"]),
        "hsv_red_recall": _mean(["hsv", "red_box", "recall"]),
        "hsv_red_precision": _mean(["hsv", "red_box", "precision"]),
        "hsv_yellow_recall": _mean(["hsv", "yellow_box", "recall"]),
        "hsv_yellow_precision": _mean(["hsv", "yellow_box", "precision"]),
        "hsv_main_recall": _mean(["hsv", "main_box", "recall"]),
        "pose_latency_ms_mean": _mean(["latency_ms", "pose", "mean"]),
        "hsv_latency_ms_mean": _mean(["latency_ms", "hsv", "mean"]),
        "experiments_completed": sum(
            1 for r in step_rows if r.get("protocol", {}).get("experiment_complete")),
    }


if __name__ == "__main__":
    main()
