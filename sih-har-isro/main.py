"""
ISRO HAR — Main Entry Point
============================
Run this file to launch the full system:
    python main.py                  → full pipeline with GUI
    python main.py --headless       → no GUI (terminal only)
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


def run_pipeline(source, headless: bool = False):
    """Launch the real-time HAR inference pipeline."""
    import queue
    from pipeline.har_pipeline import HARPipeline

    gui_queue = None
    gui_thread = None

    if not headless:
        try:
            from gui.dashboard import launch_dashboard
            gui_queue = queue.Queue(maxsize=30)
            import threading
            gui_thread = threading.Thread(
                target=launch_dashboard, args=(gui_queue,), daemon=True
            )
            gui_thread.start()
        except ImportError as e:
            logger.warning("GUI failed to launch (%s). Running headless.", e)

    pipeline = HARPipeline(source=source, gui_queue=gui_queue)
    pipeline.run()


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
    parser.add_argument("--mock",     action="store_true",
                        help="Use mock data (no Gemini API needed)")
    args = parser.parse_args()

    if args.mode == "status":
        print_status()

    elif args.mode == "pipeline":
        source = args.video if args.video else args.camera
        run_pipeline(source, headless=args.headless)

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
