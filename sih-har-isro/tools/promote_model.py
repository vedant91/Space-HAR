"""
Put a trained LSTM checkpoint where the live demo will actually load it.

The pipeline loads exactly two files: LSTM_ONNX_PATH (the model) and LSTM_PATH
(its label map). Training deliberately writes its .onnx beside its own
checkpoint instead, so a retrain never reaches the demo on its own. This is
the one step that does, and it checks what it ships:

  * refuses anything that is not an LSTM checkpoint (models/ also holds a CNN)
  * re-saves the ONNX as ONE self-contained file. The exporter writes weights
    to an external-data sidecar (*.onnx.data) whose filename is stored INSIDE
    the model, so a renamed copy of the pair can silently point at the wrong
    weights - or at the stale sidecar still sitting under the new name.
  * runs the promoted file in onnxruntime on a dummy window of the
    checkpoint's own feature dim and requires it to match the PyTorch output,
    so the .onnx and the .pt are provably the same model
  * before the first promotion ever, keeps the published baseline
    (lstm_final_rack.pt, 0.201 held-out) as baseline_rack_0201.pt, never
    overwritten

    python tools/promote_model.py models/lstm_blender_core.pt
    python tools/promote_model.py <ckpt.pt> --models-dir <dir>   # dry target
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from config.experiment_config import LSTM_ONNX_PATH, LSTM_PATH, SEQUENCE_WINDOW  # noqa: E402

BASELINE_SRC = "lstm_final_rack.pt"
BASELINE_KEEP = "baseline_rack_0201.pt"
LSTM_KEYS = ("model_state", "feature_dim", "hidden_size", "num_layers", "num_classes")


def load_lstm(ckpt_path: Path):
    """Return (model, ckpt), or exit if this is not an LSTM checkpoint."""
    import torch
    from train.train_lstm import HARLSTMClassifier

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing = [k for k in LSTM_KEYS if not (isinstance(ckpt, dict) and k in ckpt)]
    if missing:
        sys.exit(f"refusing {ckpt_path}: not an LSTM checkpoint (missing {missing})")
    model = HARLSTMClassifier(feature_dim=ckpt["feature_dim"], hidden_size=ckpt["hidden_size"],
                              num_layers=ckpt["num_layers"], num_classes=ckpt["num_classes"])
    try:
        model.load_state_dict(ckpt["model_state"])   # strict: wrong arch fails here
    except RuntimeError as e:
        sys.exit(f"refusing {ckpt_path}: weights do not fit HARLSTMClassifier: {e}")
    return model.eval(), ckpt


def promote(ckpt_path: Path, models_dir: Path) -> None:
    import onnx
    import onnxruntime as ort
    import torch
    from train.train_lstm import _export_lstm_onnx

    ckpt_path = ckpt_path.resolve()
    model, ckpt = load_lstm(ckpt_path)
    dst_pt = models_dir / Path(LSTM_PATH).name
    dst_onnx = models_dir / Path(LSTM_ONNX_PATH).name
    models_dir.mkdir(parents=True, exist_ok=True)

    keep = models_dir / BASELINE_KEEP
    if not keep.exists():
        src = models_dir / BASELINE_SRC
        if not src.exists():
            sys.exit(f"refusing: baseline {src} missing, cannot preserve it as {keep}")
        shutil.copy2(src, keep)
        print(f"baseline preserved: {src} -> {keep}")

    src_onnx = ckpt_path.with_suffix(".onnx")
    tmp = dst_onnx.with_name(dst_onnx.name + ".promoting")
    with tempfile.TemporaryDirectory() as scratch:
        if not src_onnx.exists():
            print(f"no {src_onnx.name} beside the checkpoint - exporting one")
            src_onnx = Path(scratch) / src_onnx.name
            _export_lstm_onnx(str(ckpt_path), str(src_onnx))
            if not src_onnx.exists():
                sys.exit(f"ONNX export of {ckpt_path} failed (see log above)")
        # load_external_data pulls the sidecar in; a plain save embeds it.
        onnx.save(onnx.load(str(src_onnx), load_external_data=True), str(tmp))

    dim = int(ckpt["feature_dim"])
    x = np.random.default_rng(0).standard_normal((1, SEQUENCE_WINDOW, dim)).astype(np.float32)
    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        want = model(torch.from_numpy(x)).numpy()
    if got.shape != want.shape or not np.allclose(got, want, atol=1e-4):
        tmp.unlink()
        sys.exit(f"refusing: {src_onnx.name} is not this checkpoint "
                 f"(onnx {got.shape} vs torch {want.shape}, "
                 f"max diff {np.abs(got - want).max() if got.shape == want.shape else 'n/a'})")

    if ckpt_path != dst_pt.resolve():
        shutil.copy2(ckpt_path, dst_pt)
    os.replace(tmp, dst_onnx)
    print(f"promoted {ckpt_path.name}: feature_dim={dim} hidden={ckpt['hidden_size']} "
          f"classes={ckpt['num_classes']} val_acc={ckpt.get('val_acc')}")
    print(f"  {dst_pt}\n  {dst_onnx}  (self-contained, onnxruntime output matches torch)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--models-dir", type=Path, default=REPO / Path(LSTM_PATH).parent,
                    help="Where the live pipeline's model files live (default: config)")
    args = ap.parse_args()
    if not args.checkpoint.exists():
        sys.exit(f"no checkpoint at {args.checkpoint}")
    promote(args.checkpoint, args.models_dir.resolve())


if __name__ == "__main__":
    main()
