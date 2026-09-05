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


def run_training():
    """Run the full training sequence: CNN → LSTM."""
    print("\n" + "="*60)
    print("ISRO HAR — Training Sequence")
    print("="*60)

    # Check if annotated data exists
    data_dir = Path("dataset/annotated")
    skel_dir = Path("dataset/skeleton_sequences")

    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("[!] No annotated data found.")
        print("    Run data collection first:")
        print("    python data_generation/mediapipe_labeler.py")
        sys.exit(1)

    # Train CNN from scratch
    print("\n[1/2] Training Custom CNN from scratch (RTX 3050 6GB)...")
    from train.train_cnn import train_cnn
    cnn_acc = train_cnn(data_dir=str(data_dir))
    print(f"      CNN done. Val Acc: {cnn_acc:.3f}")

    # Generate skeleton sequences if not done
    if not (skel_dir / "X_sequences.npy").exists():
        print("\n[!] Skeleton sequences not found. Generating...")
        from data_generation.mediapipe_labeler import run_labeling, build_default_label_map
        run_labeling(str(data_dir), str(skel_dir), build_default_label_map())

    # Train LSTM from scratch
    print("\n[2/2] Training LSTM from scratch...")
    from train.train_lstm import train_model
    lstm_acc = train_model(data_dir=str(skel_dir))
    print(f"      LSTM done. Val Acc: {lstm_acc:.3f}")

    print("\n✅ Training complete.")
    print("   CNN model:  models/activity_cnn.pt / .onnx")
    print("   LSTM model: models/lstm_classifier.pt")


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

    checks = {
        "CNN model (.pt)":   Path("models/activity_cnn.pt"),
        "CNN model (.onnx)": Path("models/activity_cnn.onnx"),
        "LSTM model":        Path("models/lstm_classifier.pt"),
        "Annotated data":    Path("dataset/annotated"),
        "Skeleton sequences":Path("dataset/skeleton_sequences/X_sequences.npy"),
        "Synthetic videos":  Path("dataset/synthetic_gemini/videos"),
        "Logs":              Path("logs"),
    }

    for name, path in checks.items():
        exists = "✅" if path.exists() else "❌"
        print(f"  {exists}  {name:<30}  {path}")

    print("\nHardware:")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  {'✅' if cuda else '❌'}  CUDA (RTX 3050)    {'Available' if cuda else 'Not found'}")
        if cuda:
            print(f"     └─ {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    except ImportError:
        print("  ❌  PyTorch not installed")

    try:
        import mediapipe
        print(f"  ✅  MediaPipe        {mediapipe.__version__}")
    except ImportError:
        print("  ❌  MediaPipe not installed")

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
  train      → Train CNN + LSTM from scratch
  datagen    → Generate synthetic data via Gemini Veo 2
  tuner      → Interactive HSV color calibration
  status     → Show system status and model availability
  e2e        → End-to-end loop: data → train → space sim until gates pass
  sim        → Space simulation latency/accuracy test only
        """
    )
    parser.add_argument("--mode", choices=["pipeline", "train", "datagen", "tuner", "status", "e2e", "sim"],
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
        run_training()

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
