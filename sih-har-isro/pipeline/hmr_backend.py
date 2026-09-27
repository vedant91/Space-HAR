"""
3D Human Mesh Recovery Backend (Stage 2) — Optional
====================================================
Stage 1 (pipeline/rack_frame.py) makes the 2D skeleton rack-relative, but a
2D projection still loses out-of-plane depth. True orientation-agnostic
tracking needs 3D Human Mesh Recovery: an HMR model (HMR 2.0, WHAM, OS-X,
etc.) regresses a SMPL body mesh in a ROOT-RELATIVE canonical frame, which
is inherently gravity-free — no floor prior, no fixed 'up'.

This module is a thin, dependency-optional adapter:

  - `HMRBackend` tries to load a supported HMR package: hmr2, wham, or romp
    (the one actually installed/working in .venv-hmr).
  - If none is installed (or there is no GPU), it reports unavailable and the
    pipeline transparently falls back to MediaPipe 2D + rack-frame Stage 1.
  - Output is always the SAME 132-dim contract (33 landmarks x, y, z, vis),
    so the LSTM/state machine never changes, but the coordinate convention
    depends on backend:
      - hmr2/wham: root-relative 3D (meters), gravity-free.
      - romp: MediaPipe's own convention (x, y normalized image coords,
        z hip-relative and scaled to roughly match) via
        `romp_to_feature_vector` — NOT root-relative 3D. See that function
        and ROMP_CONTRACT_VERSION for the exact math.

Install (when GPU hardware is available):
    pip install git+https://github.com/shubham-goenka/hmr2.git   # or
    pip install wham                                              # or
    pip install simple_romp   # (run under D:/SIH-2026/.venv-hmr)

Nothing in this file is required for the current CPU pipeline to run.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import FPS, FRAME_HEIGHT, FRAME_WIDTH, HMR_BACKEND, SKELETON_FEATURES
from pipeline.rack_frame import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, wrap180

logger = logging.getLogger(__name__)

NUM_LANDMARKS = 33
FEATURE_DIM = SKELETON_FEATURES  # single source of truth: config.SKELETON_FEATURES (132)

# SMPL -> MediaPipe slot mapping.
#
# Both index spaces are fixed and public, so this table is checkable rather
# than a matter of taste:
#
#   SMPL (24-joint body skeleton)
#     0 pelvis   1 L_hip     2 R_hip    3 spine1   4 L_knee   5 R_knee
#     6 spine2   7 L_ankle   8 R_ankle  9 spine3  10 L_foot  11 R_foot
#    12 neck    13 L_collar 14 R_collar 15 head   16 L_shoulder 17 R_shoulder
#    18 L_elbow 19 R_elbow  20 L_wrist 21 R_wrist 22 L_hand  23 R_hand
#
#   MediaPipe Pose (33 landmarks)
#     0 nose   11/12 shoulders  13/14 elbows  15/16 wrists  19/20 index
#    23/24 hips  25/26 knees  27/28 ankles  31/32 foot_index
#
# The previous table was wrong on the MediaPipe side for every single entry:
# it sent hips to 11/12 (which are the SHOULDERS), knees to 13/14 (ELBOWS),
# ankles to 15/16 (WRISTS), wrists to 17/18 (PINKY), and shoulders to 20/21
# (INDEX and THUMB). Every mapped joint landed one limb away from where it
# belonged, so an active HMR backend would have fed the LSTM a scrambled
# skeleton — silently, because the vector still had the right shape and the
# right magnitudes. It was dormant only because HMR_BACKEND defaults to
# "none".
_SMPL_TO_SLOT = {
    15: 0,    # head        -> nose (closest available head landmark)
    16: 11,   # L_shoulder  -> left_shoulder
    17: 12,   # R_shoulder  -> right_shoulder
    18: 13,   # L_elbow     -> left_elbow
    19: 14,   # R_elbow     -> right_elbow
    20: 15,   # L_wrist     -> left_wrist
    21: 16,   # R_wrist     -> right_wrist
    22: 19,   # L_hand      -> left_index
    23: 20,   # R_hand      -> right_index
    1: 23,    # L_hip       -> left_hip
    2: 24,    # R_hip       -> right_hip
    4: 25,    # L_knee      -> left_knee
    5: 26,    # R_knee      -> right_knee
    7: 27,    # L_ankle     -> left_ankle
    8: 28,    # R_ankle     -> right_ankle
    10: 31,   # L_foot      -> left_foot_index
    11: 32,   # R_foot      -> right_foot_index
}

# MediaPipe slots for the hips, used to root the output the same way
# MediaPipe itself does (z is expressed relative to the hip midpoint).
_L_HIP_SLOT, _R_HIP_SLOT = 23, 24

# ── ROMP (backend="romp") ────────────────────────────────────────────────────
# Unlike hmr2/wham above (root-relative metres, a DIFFERENT distribution from
# MediaPipe), the ROMP path emits MediaPipe's own coordinate convention, per
# the contract hmr-contract verified on blender_core:
#   x, y = ROMP's pj2d_org (image pixels) / frame W, H   - absolute, NOT hip-centred
#   z    = (J_z - hip_mid_z) * cam_s * max(W,H)/2 / W    - hip-relative, x-pixel units
#   vis  = 1.0 on the 27 slots with a ROMP joint, 0.0 on the 6 without one
# That makes rack normalisation (pipeline/rack_frame.py) behave on it exactly
# as on MediaPipe. It is still NOT MediaPipe: a model trained on MediaPipe
# must not be served on it, nor the reverse. check_pose_source() enforces that.
#
# MediaPipe slot -> ROMP joints54 index (romp/utils.py SMPL_ALL_54). SMPL left
# is MediaPipe left throughout.
_ROMP_SLOTS = {
    0: 24,                                  # Nose (joints54, beats SMPL 15 'Jaw')
    2: 26, 5: 25, 7: 28, 8: 27,             # L_Eye, R_Eye, L_Ear, R_Ear
    11: 16, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21,   # shoulders, elbows, wrists
    17: 39, 18: 44,                         # L/R_Hand_pinky
    19: 22, 20: 23,                         # SMPL L/R_Hand -> index (approximate)
    21: 35, 22: 40,                         # L/R_Hand_thumb
    23: 1, 24: 2, 25: 4, 26: 5, 27: 7, 28: 8,         # hips, knees, ankles
    29: 31, 30: 34,                         # L/R_Heel
    31: 10, 32: 11,                         # SMPL L/R_Toe -> foot_index (approximate)
}
# No ROMP joint at all (eye inner/outer, mouth corners). They copy the nearest
# measured point with vis 0: left at (0, 0) they would still be rotated and
# scaled by rack normalisation into a rack-dependent phantom trajectory, and
# would bias the z re-centring.
_ROMP_PROXY = {1: 2, 3: 2, 4: 5, 6: 5, 9: 0, 10: 0}
ROMP_CENTER_THRESH = 0.02    # default 0.25 finds nobody on these renders (hmr-env)
# Bump when what the ROMP path emits changes (keys the build cache).
# 2: crop + rotate + adaptive schedule (below) instead of one full-frame call per frame.
# 3: each actual run also tries the plain full frame and keeps whichever ROMP
#    is more confident in - bounds the crop's accuracy floor at the pre-crop
#    behaviour (see _romp_features), which can change the emitted pose on
#    frames where the full-frame candidate wins.
ROMP_CONTRACT_VERSION = 3

# ── ROMP input: crop, rotate, schedule ───────────────────────────────────────
# ROMP pads its input to a square and resizes it to 512 px before the backbone,
# and was trained on upright, well-framed people. Handed the whole frame, the
# astronaut gets only the fraction of those 512 px he covers, at whatever roll
# he floats at. So ROMP runs on a square crop around the tracked body, rotated
# so the torso points up the crop, and its joints are mapped back to full-frame
# pixels: the 132-dim convention above is unchanged.
#
# The box and the rotation come from MediaPipe every frame (cheap next to
# ROMP): the live pose stage's own UprightPoseEstimator with its config, on
# the whole frame. Measured on blender_core, running MediaPipe on the crop
# instead found the body on only 17-28% of take_0000 frames (vs ~95% on the
# frame), and each hallucinated inverted torso then rotated the next crop
# upside down. Stock COCO YOLOv8n is the (re)acquisition fallback only: it
# almost never calls the rendered astronaut a `person` (it scores him as a
# "traffic light" / "parking meter"), so the track is carried by MediaPipe.
# Tuning numbers: tools/measure_hmr_crop.py.
ROMP_CROP_PAD = 1.3            # crop side = pad x diagonal of the landmark box: fits any rotation
ROMP_REROTATE_DEG = 15.0       # re-run ROMP once the torso's image angle drifts this far ...
ROMP_MAX_STALE_FRAMES = FPS    # ... or this many frames (1 s) after the last run, whichever first
ROMP_RETRY_CONF = 0.1          # centre confidence below which ROMP also tries ROMP_RETRY_ANGLES
ROMP_RETRY_ANGLES = (90.0, -90.0, 180.0)
TRACK_EMA = 0.5                # weight of this frame's landmark box in the tracked box
TRACK_MAX_COAST = 5            # frames with no body before the box is dropped (full frame again)
TRACK_MIN_VIS = 0.3            # landmarks at/above this visibility define box and torso
TRACK_MIN_POINTS = 4           # fewer visible landmarks than this is a miss, not a box
PERSON_CONF = 0.25             # stock YOLO `person` confidence for (re)acquisition
PERSON_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"


def torso_tilt_deg(xy: np.ndarray, vis: np.ndarray, min_vis: float = TRACK_MIN_VIS
                   ) -> Optional[float]:
    """cv2.getRotationMatrix2D angle that turns the hip->shoulder vector to
    point straight up the image (xy in pixels, y down). None when no shoulder
    or no hip is visible. Partial landmarks are enough: one of each will do."""
    sh = [i for i in (L_SHOULDER, R_SHOULDER) if vis[i] >= min_vis]
    hp = [i for i in (L_HIP, R_HIP) if vis[i] >= min_vis]
    if not sh or not hp:
        return None
    v = xy[sh].mean(axis=0) - xy[hp].mean(axis=0)
    return math.degrees(math.atan2(v[0], -v[1]))


def square_crop(center: Tuple[float, float], side: float, angle: float):
    """(2x3 affine, side): frame -> side x side crop centred on `center`,
    rotated by `angle` (cv2.getRotationMatrix2D sense)."""
    import cv2
    side = int(round(side))
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle), 1.0)
    m[0, 2] += side / 2.0 - center[0]
    m[1, 2] += side / 2.0 - center[1]
    return m, side


def romp_to_feature_vector(joints54: np.ndarray, pj2d_org: np.ndarray, cam: np.ndarray,
                           wh: Tuple[int, int], input_side: Optional[float] = None) -> np.ndarray:
    """One ROMP person -> 132-dim vector in MediaPipe's coordinate convention.

    `pj2d_org` is in full-frame pixels. `input_side` is the longer side of the
    image ROMP actually saw (a crop), which sets its pixels-per-unit scale;
    default: the frame itself."""
    w, h = wh
    side = float(input_side) if input_side else float(max(w, h))
    slots = np.fromiter(_ROMP_SLOTS.keys(), dtype=np.int64)
    idx = np.fromiter(_ROMP_SLOTS.values(), dtype=np.int64)
    f = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    f[slots, 0] = pj2d_org[idx, 0] / w
    f[slots, 1] = pj2d_org[idx, 1] / h
    hip_z = 0.5 * (joints54[1, 2] + joints54[2, 2])
    f[slots, 2] = (joints54[idx, 2] - hip_z) * float(cam[0]) * side / 2.0 / w
    f[slots, 3] = 1.0
    for slot, src in _ROMP_PROXY.items():
        f[slot, :3] = f[src, :3]
    return f.reshape(-1)


def trained_hmr_backend(meta: dict) -> str:
    """HMR backend a checkpoint / sequence set's pose came from, or "none" for
    MediaPipe-convention 2D (MediaPipe, Blender ground truth, synthetic).
    Unrecorded = "none": everything built before --pose-source hmr existed."""
    return str(meta.get("hmr_backend", "none")) if meta.get("pose_source") == "hmr" else "none"


def check_pose_source(name: str, meta: dict, runtime_backend: str) -> None:
    """Refuse to serve a model on a pose source it was not trained on.

    `runtime_backend` is the HMR backend ACTUALLY producing pose ("none" when
    MediaPipe does, including when HMR_BACKEND asked for one that failed to
    load). Both sides fill the same 132 slots, so a mismatch never throws on
    its own; it just makes the model quietly worse."""
    trained = trained_hmr_backend(meta)
    if trained == runtime_backend:
        return
    side = lambda b: "MediaPipe 2D (no HMR backend)" if b == "none" else f"HMR backend {b!r}"
    raise RuntimeError(
        f"{name}: pose source mismatch. The model was TRAINED on {side(trained)} "
        f"(pose_source={meta.get('pose_source', 'unrecorded')!r}) but this runtime SERVES "
        f"{side(runtime_backend)} (config HMR_BACKEND={HMR_BACKEND!r}). Same 132 slots, "
        f"different coordinates: refusing to run rather than degrade silently. Set "
        f"HMR_BACKEND={trained!r}{' (with its package importable)' if trained != 'none' else ''}, "
        f"or serve a model built "
        f"with tools/build_sequences.py --pose-source "
        f"{'hmr' if runtime_backend != 'none' else 'mediapipe'}.")


class HMRBackend:
    """
    Optional 3D HMR adapter with graceful degradation.

    Example:
        backend = HMRBackend()           # never raises
        feats, name = backend.get_pose_features(frame_rgb)
        # name == "mediapipe" → Stage-1 rack-frame path handles it
    """

    def __init__(self, backend: str = "auto", device: Optional[str] = None):
        self.requested = (backend or "none").lower()
        self.device = device
        self.model = None
        self.backend_name = "none"
        if self.requested in ("none", "mediapipe"):
            return
        if self.requested == "romp":
            try:
                self.model = self._load_romp()
                self.backend_name = "romp"
                logger.info("HMR backend active: romp (MediaPipe-convention 2D + hip-relative z)")
            except Exception as e:   # ImportError included: same graceful fallback as below
                logger.warning("HMR backend 'romp' unavailable (%s); falling back to MediaPipe.", e)
            return
        if self.requested == "custom":
            try:
                self.model = self._load_custom()
                self.backend_name = "custom"
                logger.info("HMR backend active: custom (train/hmr3d_model.py, "
                           "trained on synthetic ground truth)")
            except Exception as e:
                logger.warning("HMR backend 'custom' unavailable (%s); falling back to MediaPipe.", e)
            return
        self._try_load()

    # ── loading ───────────────────────────────────────────────────────────

    def _try_load(self) -> None:
        for pkg, loader in (("hmr2", self._load_hmr2), ("wham", self._load_wham)):
            try:
                self.model = loader()
                self.backend_name = pkg
                logger.info("HMR backend active: %s (3D root-relative mesh)", pkg)
                return
            except ImportError:
                continue
            except Exception as e:  # weights/IO problems must not kill the pipeline
                logger.warning("HMR package '%s' found but failed to load: %s", pkg, e)
        logger.warning(
            "No 3D HMR backend available (requested='%s'). "
            "Falling back to MediaPipe 2D + rack-frame normalization. "
            "Install one with: pip install hmr2  (GPU required).",
            self.requested,
        )

    def _load_hmr2(self):
        import torch  # noqa: F401  (import proves the runtime exists)
        from hmr2.models import load_hmr2  # type: ignore
        model, _ = load_hmr2()
        model.eval()
        return model

    def _load_wham(self):
        import torch  # noqa: F401
        from wham_api import WHAM  # type: ignore
        return WHAM()

    def _load_tracker(self) -> None:
        """MediaPipe (box + torso angle) and the stock person detector, shared
        by every learned backend that needs a tracked crop (romp, custom).
        Missing either must fail the load the same way a missing backend
        model does: a backend that silently ran uncropped would emit
        different features under the same contract version."""
        from config.experiment_config import (MEDIAPIPE_MIN_DET_CONF, MEDIAPIPE_MIN_TRK_CONF,
                                              MEDIAPIPE_MODEL_COMPLEXITY, POSE_INPUT_WIDTH)
        from pipeline.upright_pose import UprightPoseEstimator
        self._mp_wh = (int(POSE_INPUT_WIDTH), max(int(round(FRAME_HEIGHT * POSE_INPUT_WIDTH / FRAME_WIDTH)), 16))
        self._mp = UprightPoseEstimator(complexity=MEDIAPIPE_MODEL_COMPLEXITY,
                                        min_det_conf=MEDIAPIPE_MIN_DET_CONF,
                                        min_trk_conf=MEDIAPIPE_MIN_TRK_CONF, downscale=1,
                                        frame_width=self._mp_wh[0], frame_height=self._mp_wh[1])
        if not self._mp.available:
            raise RuntimeError("MediaPipe pose (needed for the crop) is unavailable")
        if not PERSON_WEIGHTS.exists():   # never let ultralytics fetch weights: offline system
            raise FileNotFoundError(f"stock person detector weights missing: {PERSON_WEIGHTS}")
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # its auto-pip must not run either
        os.environ.setdefault("YOLO_OFFLINE", "True")
        from ultralytics import YOLO  # type: ignore
        self._person = YOLO(str(PERSON_WEIGHTS))
        self.reset()

    def _load_romp(self):
        # simple_romp shells out to `pip install ...` resolved via PATH, not
        # sys.executable: put this interpreter's Scripts dir first so any such
        # call lands in this environment and never in the system Python.
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        import romp  # type: ignore
        from romp.main import romp_settings  # type: ignore
        model = romp.ROMP(romp_settings(["--center_thresh", str(ROMP_CENTER_THRESH)]))
        self._load_tracker()
        return model

    def _load_custom(self):
        """train/hmr3d_model.py, fine-tuned on our own synthetic ground truth
        instead of ROMP's real-photo pretraining - see that module's
        docstring for why. Checkpoint from train/train_hmr3d.py."""
        import torch
        from train.hmr3d_model import HMR3DNet
        ckpt_path = Path(__file__).resolve().parent.parent / "models" / "hmr3d_net.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"custom HMR checkpoint missing: {ckpt_path} (train it with train/train_hmr3d.py)")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = HMR3DNet(pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self._custom_device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(self._custom_device)
        self._load_tracker()
        return model

    def reset(self) -> None:
        """Forget the tracked body. Call between takes/trials."""
        if getattr(self, "_mp", None) is not None:
            self._mp.reset()      # the upright estimator's working angle
        self._box = None         # (x1, y1, x2, y2) full-frame px, EMA of landmark boxes
        self._angle = 0.0         # MediaPipe torso angle (cv2 rotation that makes it upright)
        self._offset = 0.0        # quarter-turn correction the ROMP retry chose on top of it
        self._misses = 0
        self._last = None         # last ROMP 132-vector, reused between runs
        self._last_angle = 0.0
        self._since = 0           # frames since the last ROMP run
        self.stats = {"frames": 0, "romp_runs": 0, "romp_retries": 0, "retry_switched": 0,
                      "mp_hits": 0, "yolo_proposals": 0, "full_frame_tries": 0, "full_frame_won": 0}

    def _crop(self, rgb: np.ndarray, box, angle: float):
        import cv2
        h, w = rgb.shape[:2]
        if box is None:   # no track: the whole frame, padded square exactly as ROMP pads it
            m, side = square_crop((w / 2.0, h / 2.0), max(w, h), angle)
        else:
            m, side = square_crop(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
                                  max(ROMP_CROP_PAD * math.hypot(box[2] - box[0], box[3] - box[1]), 64.0),
                                  angle)
        return cv2.warpAffine(rgb, m, (side, side), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT), m, side

    @staticmethod
    def _to_frame(m: np.ndarray, xy: np.ndarray) -> np.ndarray:
        import cv2
        return np.c_[xy, np.ones(len(xy))] @ cv2.invertAffineTransform(m).T

    def _landmarks(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Upright MediaPipe on the whole frame -> (33, 4), x, y in frame pixels."""
        import cv2
        h, w = rgb.shape[:2]
        lm = self._mp.process(cv2.resize(rgb, self._mp_wh)).reshape(NUM_LANDMARKS, 4).astype(np.float64)
        if int((lm[:, 3] >= TRACK_MIN_VIS).sum()) < TRACK_MIN_POINTS:
            return None
        lm[:, :2] *= (w, h)
        return lm

    def _person_box(self, rgb: np.ndarray):
        r = self._person.predict(np.ascontiguousarray(rgb[..., ::-1]), classes=[0],
                                 conf=PERSON_CONF, verbose=False)[0]
        if len(r.boxes) == 0:
            return None
        self.stats["yolo_proposals"] += 1
        return r.boxes.xyxy[int(r.boxes.conf.argmax())].cpu().numpy().astype(np.float64)

    def _track(self, rgb: np.ndarray) -> None:
        """Per-frame, cheap: move the box and torso angle with MediaPipe."""
        lm = self._landmarks(rgb)
        if lm is None:
            # Coast on the last box a few frames (blur, occlusion), then drop
            # it; with no box, a stock-YOLO person box stands in until
            # MediaPipe finds the body again, else ROMP sees the whole frame.
            self._misses += 1
            if self._misses > TRACK_MAX_COAST:
                self._box = None
            if self._box is None:
                self._box = self._person_box(rgb)
                if self._box is not None:
                    self._misses = TRACK_MAX_COAST   # unconfirmed: dropped on the next miss
            return
        self.stats["mp_hits"] += 1
        self._misses = 0
        seen = lm[lm[:, 3] >= TRACK_MIN_VIS, :2]
        new = np.r_[seen.min(axis=0), seen.max(axis=0)]
        self._box = new if self._box is None else TRACK_EMA * new + (1.0 - TRACK_EMA) * self._box
        tilt = torso_tilt_deg(lm[:, :2], lm[:, 3])
        if tilt is not None:
            self._angle = tilt

    def _romp_at(self, rgb: np.ndarray, box, angle: float) -> Tuple[Optional[np.ndarray], float]:
        import contextlib
        import io
        import cv2
        crop, m, side = self._crop(rgb, box, angle)
        with contextlib.redirect_stdout(io.StringIO()):   # ROMP prints a timing line per call
            out = self.model(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if out is None or len(out.get("center_confs", [])) == 0:
            return None, 0.0
        confs = np.asarray(out["center_confs"]).reshape(-1)
        i = int(np.argmax(confs))   # one astronaut: most confident centre wins
        h, w = rgb.shape[:2]
        pj2d = self._to_frame(m, np.asarray(out["pj2d_org"][i])[:, :2])
        return romp_to_feature_vector(out["joints"][i], pj2d, out["cam"][i], (w, h),
                                      input_side=side), float(confs[i])

    def _romp_features(self, frame_rgb: np.ndarray) -> np.ndarray:
        self.stats["frames"] += 1
        self._track(frame_rgb)
        self._since += 1
        stale = (self._last is None or self._since >= ROMP_MAX_STALE_FRAMES
                 or abs(wrap180(self._angle - self._last_angle)) > ROMP_REROTATE_DEG)
        if stale:
            self.stats["romp_runs"] += 1
            feats, conf = self._romp_at(frame_rgb, self._box, self._angle + self._offset)
            if conf < ROMP_RETRY_CONF:
                # Low confidence on the corrected crop: the torso angle may be
                # wrong (MediaPipe missed, or flipped an inverted body). Try
                # the other quarter turns. The winner is kept as an offset on
                # MediaPipe's angle, not written over it: MediaPipe re-measures
                # every frame, and overwriting would read as drift and re-run.
                self.stats["romp_retries"] += 1
                best = self._offset
                for d in ROMP_RETRY_ANGLES:
                    off = wrap180(self._offset + d)
                    f2, c2 = self._romp_at(frame_rgb, self._box, self._angle + off)
                    if c2 > conf:
                        feats, conf, best = f2, c2, off
                if best != self._offset:
                    self.stats["retry_switched"] += 1
                    self._offset = best
            # ROMP's centre confidence says "a person-shaped blob is here", not
            # "this crop contains the whole body correctly" - measured on
            # blender_core, a landmark box that's tight or mispositioned under
            # foreshortening (pitch) gets a confidently-WRONG crop the checks
            # above never catch, regressing a case ROMP was already reading
            # correctly uncropped (pitch_60: 0.74 tracked before this file's
            # crop/rotate change, 0.21 after - tools/measure_hmr_crop.py). So on
            # every actual run, also try the plain full frame (no box, no
            # rotation: exactly what ROMP saw before this file existed) and
            # keep whichever the model itself is more confident in. This
            # bounds the accuracy floor at the pre-crop behaviour; it can only
            # help or tie, never regress a frame the old code handled fine.
            # Cost: one extra ROMP call, but only on the already-throttled
            # ~3-18% of frames a run actually happens on (measure_hmr_crop.py).
            if self._box is not None:
                self.stats["full_frame_tries"] += 1
                f3, c3 = self._romp_at(frame_rgb, None, 0.0)
                if c3 > conf:
                    self.stats["full_frame_won"] += 1
                    feats, conf = f3, c3
            self._last, self._last_angle, self._since = feats, self._angle, 0
        if self._last is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)       # nobody: same as MediaPipe
        return self._last.copy()

    def _custom_at(self, rgb: np.ndarray, box, angle: float) -> Tuple[Optional[np.ndarray], float]:
        """Same crop -> inverse-affine-back-to-full-frame shape as _romp_at,
        but through train/hmr3d_model.py instead of ROMP. z passes straight
        through (an in-plane crop rotation does not change true depth - see
        that module's docstring); "confidence" is the mean predicted
        visibility, a simple proxy since this network has no ROMP-style
        multi-person center score to fall back on."""
        import torch
        from train.hmr3d_model import preprocess_crop
        crop, m, side = self._crop(rgb, box, angle)
        x = preprocess_crop(crop).unsqueeze(0).to(self._custom_device)
        with torch.no_grad():
            pred = self.model(x)[0].cpu().numpy()   # (33, 4), crop-local
        h, w = rgb.shape[:2]
        xy_full = self._to_frame(m, pred[:, :2] * side)
        feats = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        feats[:, 0] = xy_full[:, 0] / w
        feats[:, 1] = xy_full[:, 1] / h
        feats[:, 2] = pred[:, 2]
        feats[:, 3] = np.clip(pred[:, 3], 0.0, 1.0)
        return feats.reshape(-1), float(np.mean(feats[:, 3]))

    def _custom_features(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Same track-crop-schedule shape as _romp_features, without the
        confidence-retry/full-frame-fallback machinery that exists
        specifically to work around ROMP's known failure modes - this
        network's own failure modes are not measured yet, so that complexity
        is not warranted until evidence says it is (see docs/architecture.html
        section 3 on why the ROMP fixes were each driven by a measured
        problem, not applied speculatively)."""
        self.stats["frames"] += 1
        self._track(frame_rgb)
        self._since += 1
        stale = (self._last is None or self._since >= ROMP_MAX_STALE_FRAMES
                 or abs(wrap180(self._angle - self._last_angle)) > ROMP_REROTATE_DEG)
        if stale:
            self.stats["romp_runs"] += 1
            feats, conf = self._custom_at(frame_rgb, self._box, self._angle)
            self._last, self._last_angle, self._since = feats, self._angle, 0
        if self._last is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        return self._last.copy()

    # ── public API ────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self.model is not None

    def get_pose_features(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Returns (features132, backend_name).

        - HMR active:  root-relative 3D joints mapped into the 33-slot
          contract (x, y, z in meters; visibility=1 for mapped joints).
        - HMR absent:  zero features + "mediapipe"; the caller must run its
          MediaPipe path (this keeps the pipeline's latency contract intact).
        """
        if not self.available:
            return np.zeros(FEATURE_DIM, dtype=np.float32), "mediapipe"
        try:
            if self.backend_name == "romp":
                return self._romp_features(frame_rgb), "romp"
            if self.backend_name == "custom":
                return self._custom_features(frame_rgb), "custom"
            joints = self._infer_joints(frame_rgb)  # (J, 3) root-relative meters
            return self._to_feature_vector(joints), self.backend_name
        except Exception as e:
            logger.warning("HMR inference failed (%s); caller should fall back.", e)
            return np.zeros(FEATURE_DIM, dtype=np.float32), "mediapipe"

    # ── internals ─────────────────────────────────────────────────────────

    def _infer_joints(self, frame_rgb: np.ndarray) -> np.ndarray:
        import torch
        import cv2

        img = cv2.resize(frame_rgb, (256, 256))
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        if self.device:
            x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            # hmr2 returns dict with 'smpl_vertices'/'joints3d'; wham returns tensor
            if isinstance(out, dict):
                joints = out.get("joints3d", out.get("smpl_joints3d"))
            else:
                joints = out
        return joints[0].detach().cpu().numpy().astype(np.float32)

    def _to_feature_vector(self, joints: np.ndarray) -> np.ndarray:
        """(J, 3) root-relative 3D → 132-dim MediaPipe-style contract."""
        feats = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        smpl = np.asarray(joints, dtype=np.float32).reshape(-1, 3)

        for smpl_idx, slot in _SMPL_TO_SLOT.items():
            if smpl_idx < len(smpl):
                feats[slot, :3] = smpl[smpl_idx]
                feats[slot, 3] = 1.0

        # Root on the hip midpoint, not the mean of all joints. MediaPipe's
        # own convention is hip-centred, and a mean over whatever subset of
        # joints happens to be mapped would move the origin whenever the
        # mapping changed — making features from two builds incomparable.
        if feats[_L_HIP_SLOT, 3] > 0.0 and feats[_R_HIP_SLOT, 3] > 0.0:
            origin = 0.5 * (feats[_L_HIP_SLOT, :3] + feats[_R_HIP_SLOT, :3])
        else:
            mapped = feats[feats[:, 3] > 0.0, :3]
            origin = mapped.mean(axis=0) if len(mapped) else np.zeros(3, dtype=np.float32)
        feats[feats[:, 3] > 0.0, :3] -= origin
        return feats.reshape(-1)


def _selftest() -> None:
    import cv2
    b = HMRBackend(backend="auto")
    feats, name = b.get_pose_features(np.zeros((720, 1280, 3), dtype=np.uint8))
    print(f"backend={name} available={b.available} feats_shape={feats.shape}")
    assert feats.shape == (FEATURE_DIM,) and np.isfinite(feats).all()

    # ROMP contract: pixel -> normalised image coords, hips at the right slots.
    j54 = np.zeros((54, 3), np.float32); p2d = np.zeros((54, 2), np.float32)
    p2d[1], p2d[2], p2d[16] = (640, 360), (660, 360), (600, 200)   # L_hip, R_hip, L_shoulder
    j54[16, 2], j54[1, 2], j54[2, 2] = 0.3, 0.1, 0.1
    f = romp_to_feature_vector(j54, p2d, np.array([0.5, 0, 0]), (1280, 720)).reshape(33, 4)
    assert np.allclose(f[23, :2], (0.5, 0.5)) and np.allclose(f[11, :2], (600 / 1280, 200 / 720))
    assert np.isclose(f[11, 2], 0.2 * 0.5 * 1280 / 2 / 1280) and f[23, 2] == 0.0
    assert f[:, 3].sum() == 27 and f[[1, 3, 4, 6, 9, 10], 3].sum() == 0
    assert np.allclose(f[1, :3], f[2, :3])                           # proxy copies neighbour

    # Pose-source guard: match passes, either mismatch direction raises.
    check_pose_source("m", {}, "none")
    check_pose_source("m", {"pose_source": "hmr", "hmr_backend": "romp"}, "romp")
    for meta, rt in (({"pose_source": "hmr", "hmr_backend": "romp"}, "none"), ({}, "romp")):
        try:
            check_pose_source("m", meta, rt)
            raise AssertionError("mismatch not refused")
        except RuntimeError:
            pass

    # z follows the scale of the image ROMP actually saw (a crop), x/y do not.
    f2 = romp_to_feature_vector(j54, p2d, np.array([0.5, 0, 0]), (1280, 720), input_side=640).reshape(33, 4)
    assert np.isclose(f2[11, 2], f[11, 2] / 2) and np.allclose(f2[:, :2], f[:, :2])

    # Torso angle sign: an upright torso rotated by A (cv2 sense) reads -A,
    # and a crop rotated by the reading stands it back up.
    xy = np.full((33, 2), (110.0, 100.0)); vis = np.ones(33)
    xy[[11, 12, 23, 24]] = (100, 50), (120, 50), (100, 150), (120, 150)
    assert abs(torso_tilt_deg(xy, vis)) < 1e-9 and torso_tilt_deg(xy, np.zeros(33)) is None
    for a in (30.0, 90.0, -120.0, 180.0):
        rxy = np.c_[xy, np.ones(33)] @ cv2.getRotationMatrix2D((110, 100), a, 1.0).T
        t = torso_tilt_deg(rxy, vis)
        assert abs(wrap180(t + a)) < 1e-6, (a, t)
        m, side = square_crop((110, 100), 200, t)
        assert abs(torso_tilt_deg(np.c_[rxy, np.ones(33)] @ m.T, vis)) < 1e-6
        assert np.allclose(HMRBackend._to_frame(m, np.c_[rxy, np.ones(33)] @ m.T), rxy)
    m, side = square_crop((300, 200), 101, 37.0)
    assert np.allclose(np.array([300, 200, 1.0]) @ m.T, (side / 2, side / 2))

    # Tracker + scheduler, with MediaPipe / ROMP / YOLO faked out.
    body = {"tilt": 0.0, "seen": True, "good": 0.0}
    calls, yolo, full_calls, full_conf = [], [], [], {"value": 0.0}

    def fake_landmarks(rgb):
        if not body["seen"]:
            return None
        lm = np.ones((33, 4))
        lm[:, :2] = np.c_[xy, np.ones(33)] @ cv2.getRotationMatrix2D((110, 100), -body["tilt"], 1.0).T
        return lm

    def fake_romp(rgb, box, angle):
        if box is None:   # the full-frame candidate _romp_features always also tries
            full_calls.append(angle)
            return np.full(FEATURE_DIM, -1.0, np.float32), full_conf["value"]
        calls.append(angle)   # confident only when the crop is rotated by tilt + "good"
        ok = abs(wrap180(angle - body["tilt"] - body["good"])) < 1.0
        return np.full(FEATURE_DIM, len(calls), np.float32), (0.5 if ok else 0.01)

    hb = HMRBackend(backend="none")
    hb.backend_name = "romp"
    hb.reset()
    hb._landmarks, hb._romp_at = fake_landmarks, fake_romp
    hb._person_box = lambda rgb: yolo.append(1)   # None: no person proposal
    rgb = np.zeros((720, 1280, 3), np.uint8)
    step = lambda: hb._romp_features(rgb)

    out = step()
    assert len(calls) == 1 and calls[0] == 0.0 and np.all(out == 1)         # first frame runs
    body["tilt"] = ROMP_REROTATE_DEG - 1.0                                  # small drift: reuse
    for _ in range(ROMP_MAX_STALE_FRAMES - 1):   # frames 2..MAX reuse frame 1
        assert np.all(step() == 1)
    assert len(calls) == 1
    step()
    assert len(calls) == 2, "max staleness must force a re-run"
    body["tilt"] = 2 * ROMP_REROTATE_DEG + 1.0                              # big drift: re-run now
    step()
    assert len(calls) == 3 and np.isclose(calls[-1], body["tilt"])
    # Low confidence -> quarter-turn hypotheses; the winning offset sticks
    # without making every following frame look like drift.
    body["tilt"], body["good"] = 100.0, 180.0
    step()
    assert len(calls) == 3 + 1 + len(ROMP_RETRY_ANGLES) and wrap180(hb._offset - 180.0) == 0.0
    n = len(calls)
    for _ in range(5):
        step()
    assert len(calls) == n, "a retry offset must not cause re-runs"
    assert hb._box is not None and not yolo
    # Lost body: coast TRACK_MAX_COAST frames on the old box, then drop it and ask YOLO.
    body["seen"] = False
    for _ in range(TRACK_MAX_COAST):
        step()
        assert hb._box is not None and not yolo
    step()
    assert hb._box is None and len(yolo) == 1
    hb.reset()
    assert hb._last is None and hb._box is None and hb._offset == 0.0

    # A confidently-wrong crop (measured for real on pitch_60: ROMP stayed
    # confident on a foreshortening-mangled crop while its actual accuracy
    # collapsed) must not beat a plain full frame that ROMP is MORE confident
    # in - the full-frame candidate has to win when it genuinely scores higher.
    body["seen"], body["tilt"], body["good"] = True, 0.0, 0.0
    calls.clear(); full_calls.clear()
    full_conf["value"] = 0.9   # far above the crop's 0.5 "confident" case
    out = step()
    assert len(full_calls) == 1 and full_calls[0] == 0.0, "full frame is tried untouched: box=None, angle=0"
    assert np.all(out == -1.0), "the full-frame candidate must win when it scores higher"
    assert hb.stats["full_frame_tries"] == 1 and hb.stats["full_frame_won"] == 1
    hb.reset()

    # _custom_at: a known crop-local prediction must map back to the correct
    # full-frame normalised coordinates through the crop's own inverse
    # affine, with z passed straight through and vis clipped to [0,1] -
    # isolated from MediaPipe/YOLO/a real checkpoint entirely (no _load_tracker,
    # no real model), same spirit as the crop-geometry round-trip test above.
    class _ConstOut:
        def __init__(self, val):
            self.val = val
        def __call__(self, x):
            import torch
            return self.val.unsqueeze(0).expand(x.shape[0], -1, -1)

    import torch
    hb2 = HMRBackend(backend="none")
    hb2.backend_name = "custom"
    hb2._custom_device = "cpu"
    known = torch.zeros(NUM_LANDMARKS, 4)
    known[11] = torch.tensor([0.5, 0.5, 0.15, 1.3])   # crop centre; vis > 1 must clip to 1.0
    hb2.model = _ConstOut(known)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # box=None, angle=0: the crop is the whole frame padded to a square,
    # centred - crop-local centre (0.5, 0.5) must land on the FRAME's own
    # centre, normalised (0.5, 0.5).
    feats, conf = hb2._custom_at(frame, None, 0.0)
    f = feats.reshape(NUM_LANDMARKS, 4)
    assert np.allclose(f[11, :2], (0.5, 0.5), atol=1e-3), f[11, :2]
    assert np.isclose(f[11, 2], 0.15), "z must pass through the crop's inverse affine unchanged"
    assert np.isclose(f[11, 3], 1.0), "vis must clip to [0, 1]"
    assert np.isclose(conf, f[:, 3].mean())

    # A real (non-centred, rotated) crop: the SAME centre prediction must now
    # land on the box's own centre, not the frame's.
    box = (400.0, 200.0, 600.0, 400.0)   # 200x200 box, centred at (500, 300)
    feats2, _ = hb2._custom_at(frame, box, 90.0)
    f2 = feats2.reshape(NUM_LANDMARKS, 4)
    assert np.allclose(f2[11, :2], (500 / 1280, 300 / 720), atol=1e-3), f2[11, :2]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
    print("hmr_backend self-test passed (graceful fallback, ROMP contract, pose-source guard, "
          "crop geometry, tracker + scheduler, full-frame-vs-crop fallback, custom-backend transform).")
