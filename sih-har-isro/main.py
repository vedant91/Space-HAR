"""
ISRO HAR — Main Entry Point
============================
Run this file to launch the full system:
    python main.py                  → full pipeline with GUI
    python main.py --headless       → no GUI (terminal only)
    python main.py --stream --stream-host <ip> --stream-port 5000
                                     → also push video to that IP over UDP
    python main.py --cnn-ensemble   → fuse CNN + LSTM predictions (retrain CNN first)
    python main.py --mode demo      → run on demo video
    python main.py --mode tuner     → HSV calibration tuner
    python main.py --mode train     → run full training sequence
    python main.py --mode e2e       → data → train → space sim loop until gates pass
    python main.py --mode sim       → space simulation only (requires trained LSTM)
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def run_pipeline(source, headless: bool = False,
                 enable_streaming: bool | None = None,
                 stream_host: str | None = None, stream_port: int | None = None,
                 enable_cnn_ensemble: bool | None = None):
    """Launch the real-time HAR inference pipeline."""
    import queue
    import threading
    from pipeline.har_pipeline import HARPipeline

    # Qt (if installed) must own the main thread's event loop — decide up
    # front whether a real dashboard is available, since that determines
    # which thread runs the capture loop vs. the GUI.
    use_qt = False
    if not headless:
        try:
            from gui.qt_dashboard import launch_qt_dashboard
            use_qt = True
        except ImportError as e:
            logger.warning("PyQt6 dashboard not available (%s) — "
                           "falling back to the cv2 debug preview window.", e)

    gui_queue = queue.Queue(maxsize=30) if use_qt else None
    pipeline = HARPipeline(source=source, headless=headless, gui_queue=gui_queue,
                           enable_streaming=enable_streaming,
                           stream_host=stream_host, stream_port=stream_port,
                           enable_cnn_ensemble=enable_cnn_ensemble)

    if not use_qt:
        pipeline.run()
        return

    pipeline_thread = threading.Thread(target=pipeline.run, daemon=True)
    pipeline_thread.start()
    # Route the actual launch through gui.dashboard.launch_dashboard (its own
    # try/except around qt_dashboard is now redundant with the probe above,
    # but it's the one real GUI entry point other callers should use too —
    # calling launch_qt_dashboard directly here left gui/dashboard.py orphaned).
    from gui.dashboard import launch_dashboard
    launch_dashboard(gui_queue, on_close=pipeline.stop,
                     enable_streaming=pipeline.enable_streaming,
                     stream_host=pipeline.stream_host, stream_port=pipeline.stream_port)
    pipeline_thread.join()


def run_training(use_real_data: bool = True, skip_posenet: bool = False):
    """Full training sequence: PoseNet -> (real autolabel + fine-tune) -> LSTM data -> CNN -> LSTM.

    Every model here is trained from scratch on this project's own data — no
    open-source/pretrained weights anywhere:
      1. PoseNet (pipeline/pose_net.py) on the synthetic renderer's exact
         ground truth, then fine-tuned on classical-CV pseudo-labels from the
         real "gravitational mimic" footage (config.REAL_VIDEO_DIR), if present.
      2. Custom CNN on synthetic + (if available) real annotated frames.
      3. LSTM on sequences built from PoseNet's OWN predictions (not the
         ground-truth vectors PoseNet was trained against) — see
         data_generation/build_posenet_realistic_dataset.py's docstring for
         why: training the LSTM mostly on clean ground truth measurably hurt
         real end-to-end accuracy (0.125 step-window accuracy without oracle
         pose injection, vs. 1.0 with it) because PoseNet's actual inference
         is noisier than what the LSTM had ever seen at training time.
    """
    from config.experiment_config import (
        POSENET_PATH, REAL_VIDEO_DIR, REAL_PSEUDO_DIR, REAL_ANNOTATED_DIR,
        REAL_SEQUENCES_DIR, COMBINED_SEQUENCES_DIR,
    )
    print("\n" + "="*60)
    print("ISRO HAR — Training Sequence (no open-source/pretrained models)")
    print("="*60)

    data_dir = Path("dataset/annotated")
    skel_dir = Path("dataset/skeleton_sequences")               # PoseNet's OWN ground-truth data
    posenet_lstm_dir = Path("dataset/skeleton_sequences_posenet")  # LSTM's real training data
    real_video_dir = Path(REAL_VIDEO_DIR)
    have_real_videos = use_real_data and real_video_dir.exists() and \
        any(real_video_dir.glob("*.mp4"))

    # ── 1. Synthetic ground-truth data (PoseNet Stage 1's own training
    # target — separate from what the LSTM trains on, see step 3) ───
    if not (skel_dir / "X_sequences.npy").exists():
        print("\n[data] No synthetic ground-truth pose sequences found — generating...")
        from data_generation.synthetic_pose import generate_dataset
        generate_dataset(str(skel_dir), n_sequences_per_step=48)
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("[data] No synthetic annotated frames found — rendering...")
        from simulation.space_sim import save_step_frames
        save_step_frames(data_dir, frames_per_step=96)

    # ── 2. PoseNet (Stage 1: synthetic; Stage 2: real fine-tune) ─
    step = 1
    total_steps = 5 if have_real_videos else 4
    if not skip_posenet:
        if not Path(POSENET_PATH).exists():
            print(f"\n[{step}/{total_steps}] Training PoseNet from scratch on synthetic ground truth...")
            from train.train_posenet import train_posenet
            r = train_posenet()
            print(f"      PoseNet Stage 1 done. val_PCK={r['val_pck']:.3f}")
        else:
            print(f"\n[{step}/{total_steps}] PoseNet checkpoint already exists — skipping Stage 1 "
                 f"(delete {POSENET_PATH} to retrain).")
    step += 1

    if have_real_videos:
        print(f"\n[{step}/{total_steps}] Auto-labeling real 'gravitational mimic' footage "
             f"({real_video_dir})...")
        from data_generation.real_video_autolabel import run_autolabel
        run_autolabel()
        if not skip_posenet:
            from train.train_posenet import finetune_on_real
            ft = finetune_on_real(pseudo_dir=REAL_PSEUDO_DIR)
            if ft.get("rejected"):
                print(f"      PoseNet Stage 2 REJECTED by the regression guard (synthetic PCK "
                     f"would have dropped {ft['regression_frac']:.0%}) — kept Stage-1 weights. "
                     f"See {ft['attempt_path']}.")
            elif not ft.get("skipped"):
                print(f"      PoseNet Stage 2 (real fine-tune) accepted. "
                     f"n_real={ft['n_real_samples']}, val_loss={ft['best_val_loss']:.4f}")
        step += 1

    # ── 3. LSTM training data: the TRAINED PoseNet's own predictions on
    # freshly-rendered synthetic frames (free, exact step-id labels — the
    # renderer drove them), not the ground-truth vectors from step 1. ──
    print(f"\n[{step}/{total_steps}] Building LSTM training data from PoseNet's own predictions...")
    from data_generation.build_posenet_realistic_dataset import build_dataset as build_posenet_lstm_data
    build_posenet_lstm_data(str(posenet_lstm_dir), n_sequences_per_step=24)
    step += 1

    if have_real_videos:
        from data_generation.build_real_dataset import (
            build_real_annotated_frames, build_real_sequences, merge_with_synthetic,
        )
        build_real_annotated_frames(REAL_PSEUDO_DIR, REAL_ANNOTATED_DIR)
        build_real_sequences(REAL_PSEUDO_DIR, REAL_SEQUENCES_DIR)
        merge_with_synthetic(str(posenet_lstm_dir), REAL_SEQUENCES_DIR, COMBINED_SEQUENCES_DIR)

    # ── 4. Custom CNN — synthetic + real frames (whichever exist) ─
    print(f"\n[{step}/{total_steps}] Training Custom CNN from scratch...")
    cnn_roots = [str(data_dir)]
    if Path(REAL_ANNOTATED_DIR).exists() and any(Path(REAL_ANNOTATED_DIR).iterdir()):
        cnn_roots.append(REAL_ANNOTATED_DIR)
    from train.train_cnn import train_cnn
    cnn_acc = train_cnn(data_dir=cnn_roots)
    print(f"      CNN done. Val Acc: {cnn_acc:.3f} (roots: {cnn_roots})")
    step += 1

    # ── 5. LSTM — PoseNet-realistic (+ real, when available) sequences ─
    combined_x = Path(COMBINED_SEQUENCES_DIR) / "X_sequences.npy"
    lstm_dir = COMBINED_SEQUENCES_DIR if combined_x.exists() else str(posenet_lstm_dir)
    print(f"\n[{step}/{total_steps}] Training LSTM from scratch (data: {lstm_dir})...")
    from train.train_lstm import train_model
    lstm_acc = train_model(data_dir=lstm_dir)
    print(f"      LSTM done. Val Acc: {lstm_acc:.3f}")

    print("\n✅ Training complete.")
    print("   PoseNet:    models/pose_net.pt / .onnx")
    print("   CNN model:  models/activity_cnn.pt / .onnx")
    print("   LSTM model: models/lstm_classifier.pt / .onnx")
    print("\n   See README.md's 'Current results' section for honest, measured accuracy —")
    print("   including the real (non-oracle) end-to-end check, not just held-out windows.")


def run_data_generation(use_mock: bool = False):
    """Run Gemini synthetic data generation."""
    if use_mock:
        from data_generation.gemini_video_gen import create_mock_dataset
        create_mock_dataset()
    else:
        from data_generation.gemini_video_gen import run_generation
        run_generation()


def run_hsv_tuner(camera: int = 0):
    """Launch interactive HSV calibration tool."""
    from pipeline.hsv_detector import run_hsv_tuner
    run_hsv_tuner(camera)


def print_status():
    """Print current system status: models, data, etc."""
    print("\n" + "="*60)
    print("ISRO HAR — System Status")
    print("="*60)

    from config.experiment_config import REAL_VIDEO_DIR, REAL_PSEUDO_DIR, REAL_ANNOTATED_DIR

    checks = {
        "PoseNet model (.pt)":     Path("models/pose_net.pt"),
        "PoseNet model (.onnx)":   Path("models/pose_net.onnx"),
        "CNN model (.pt)":         Path("models/activity_cnn.pt"),
        "CNN model (.onnx)":       Path("models/activity_cnn.onnx"),
        "LSTM model":              Path("models/lstm_classifier.pt"),
        "Annotated data (synth)":  Path("dataset/annotated"),
        "Annotated data (real)":   Path(REAL_ANNOTATED_DIR),
        "Skeleton sequences":      Path("dataset/skeleton_sequences/X_sequences.npy"),
        "Combined (synth+real)":   Path("dataset/skeleton_sequences_combined/X_sequences.npy"),
        "Real video source":       Path(REAL_VIDEO_DIR),
        "Real pseudo-labels":      Path(REAL_PSEUDO_DIR),
        "Synthetic videos":        Path("dataset/synthetic_gemini/videos"),
        "Logs":                    Path("logs"),
    }

    for name, path in checks.items():
        exists = "✅" if path.exists() else "❌"
        print(f"  {exists}  {name:<30}  {path}")

    if Path("models/pose_net.pt").exists():
        try:
            import torch
            ckpt = torch.load("models/pose_net.pt", map_location="cpu", weights_only=False)
            print(f"     └─ PoseNet: stage={ckpt.get('stage')} val_PCK={ckpt.get('val_pck', 0):.3f} "
                 f"epoch={ckpt.get('epoch')}")
        except Exception:
            pass

    print("\nHardware:")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  {'✅' if cuda else '❌'}  CUDA               {'Available' if cuda else 'Not found (CPU-only)'}")
        if cuda:
            print(f"     └─ {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    except ImportError:
        print("  ❌  PyTorch not installed")

    print("  ✅  Pose/movement model: custom HARPoseNet (trained from scratch — no "
         "MediaPipe/YOLO/pretrained weights)")

    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    print(f"  {'✅' if ffmpeg_path else '❌'}  ffmpeg (video stream) {ffmpeg_path or 'not found on PATH'}")

    try:
        import PyQt6  # noqa: F401
        print("  ✅  PyQt6 (GUI dashboard)")
    except ImportError:
        print("  ❌  PyQt6 not installed — falls back to cv2 debug preview")

    from config.experiment_config import (
        ENABLE_STREAMING, CNN_ENSEMBLE_ENABLED, RACK_FRAME_NORMALIZE,
    )
    print("\nOptional features (config defaults, all off until explicitly enabled):")
    print(f"  streaming={ENABLE_STREAMING}  cnn_ensemble={CNN_ENSEMBLE_ENABLED}  "
         f"rack_frame_normalize={RACK_FRAME_NORMALIZE}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ISRO SIH 2026 — AI HAR System for BAS Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  pipeline   → Run real-time HAR pipeline (default)
  train      → Train PoseNet + CNN + LSTM from scratch (synthetic + real video)
  posenet    → Train/fine-tune only the custom pose model
  autolabel  → Pseudo-label the real 'gravitational mimic' videos only
  datagen    → Generate synthetic data via Gemini Veo 2
  tuner      → Interactive HSV color calibration
  status     → Show system status and model availability
  e2e        → End-to-end loop: data → train → space sim until gates pass
  sim        → Space simulation latency/accuracy test only
        """
    )
    parser.add_argument("--mode",
                        choices=["pipeline", "train", "posenet", "autolabel", "datagen",
                                "tuner", "status", "e2e", "sim"],
                        default="pipeline")
    parser.add_argument("--camera",   type=int,  default=0)
    parser.add_argument("--video",    type=str,  default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--stream",   action="store_true",
                        help="Also push video to --stream-host:--stream-port via ffmpeg")
    parser.add_argument("--stream-host", type=str, default=None,
                        help="Overrides config.STREAM_HOST")
    parser.add_argument("--stream-port", type=int, default=None,
                        help="Overrides config.STREAM_PORT")
    parser.add_argument("--cnn-ensemble", action="store_true",
                        help="Fuse the CNN's frame-level prediction with the LSTM's "
                            "(overrides config.CNN_ENSEMBLE_ENABLED=True). Off by "
                            "default — retrain the CNN with the current (leakage-fixed) "
                            "train/train_cnn.py before relying on this in production.")
    parser.add_argument("--mock",     action="store_true",
                        help="Use mock data (no Gemini API needed)")
    parser.add_argument("--no-real-data", action="store_true",
                        help="--mode train: ignore config.REAL_VIDEO_DIR, train on synthetic only")
    parser.add_argument("--finetune-real", action="store_true",
                        help="--mode posenet: also run the Stage-2 real-video fine-tune")
    args = parser.parse_args()

    if args.mode == "status":
        print_status()

    elif args.mode == "pipeline":
        source = args.video if args.video else args.camera
        # None (not just-False) when the flag is absent, so an operator who
        # flips ENABLE_STREAMING/CNN_ENSEMBLE_ENABLED to True in config isn't
        # silently overridden back to False by argparse's default — only an
        # explicit --stream/--cnn-ensemble forces it on, matching
        # HARPipeline's own `None means use the config default` contract.
        run_pipeline(source, headless=args.headless,
                    enable_streaming=True if args.stream else None,
                    stream_host=args.stream_host, stream_port=args.stream_port,
                    enable_cnn_ensemble=True if args.cnn_ensemble else None)

    elif args.mode == "train":
        run_training(use_real_data=not args.no_real_data)

    elif args.mode == "posenet":
        from train.train_posenet import train_posenet, finetune_on_real
        train_posenet()
        if args.finetune_real:
            from data_generation.real_video_autolabel import run_autolabel
            run_autolabel()
            finetune_on_real()

    elif args.mode == "autolabel":
        from data_generation.real_video_autolabel import run_autolabel
        run_autolabel()

    elif args.mode == "datagen":
        run_data_generation(use_mock=args.mock)

    elif args.mode == "tuner":
        run_hsv_tuner(args.camera)

    elif args.mode == "e2e":
        from end_to_end_loop import run_loop
        result = run_loop()
        sys.exit(0 if result.get("passed") else 1)

    elif args.mode == "sim":
        from simulation.space_sim import run_full_space_sim
        import json
        report = run_full_space_sim(test_sequences_dir="dataset/skeleton_sequences_test")
        print(json.dumps({k: v for k, v in report.items() if k != "sm_statuses"},
                         indent=2, default=str))
