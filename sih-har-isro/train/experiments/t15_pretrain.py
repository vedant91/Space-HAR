"""T15: masked landmark reconstruction pretraining, then fine-tune.

Pretext: hide random (timestep, landmark) groups and whole timesteps of a
window, reconstruct the hidden values from context with the classifier's own
encoder (input_bn + bi-LSTM) and a throwaway linear head. No labels used.

Pretraining data = the `--pretrain` build's TRAIN split. Build it with the
four holdout tags still routed to holdout and nothing excluded, so it covers
the 16 train takes plus skip/recover/heldout_view but never sees a held-out
take's frames, not even unlabelled.

Fine-tuning goes through train.train_lstm.train_model unchanged (same split,
optimiser, early stopping); the only difference is that the classifier is
constructed with the pretrained encoder weights already loaded.

Usage:
    python train/experiments/t15_pretrain.py --pretrain D:/SIH-2026/.tmp/ml/seq_pretrain_mp \
        --finetune dataset/seq_blender_core --seeds 7 11 23 --out logs/t15_pretrain.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train.train_lstm as tl  # noqa: E402
from config.experiment_config import LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS  # noqa: E402
from t14_oracle import load_split, predict, score  # noqa: E402

N_LANDMARKS = 33


def groups_of(dim: int):
    """Feature-index groups that are masked together: one per landmark
    (x, y, z, vis), plus the object channels as one group."""
    g = [list(range(4 * i, 4 * i + 4)) for i in range(N_LANDMARKS)]
    if dim > 4 * N_LANDMARKS:
        g.append(list(range(4 * N_LANDMARKS, dim)))
    return g


def pretrain(X: np.ndarray, epochs: int, p_group=0.15, p_frame=0.10, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = X.shape[-1]
    mean = X.reshape(-1, dim).mean(0)
    std = X.reshape(-1, dim).std(0) + 1e-6
    enc = tl.HARLSTMClassifier(dim, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, 8).to(dev)
    head = nn.Linear(2 * LSTM_HIDDEN_SIZE, dim).to(dev)
    opt = torch.optim.Adam(list(enc.input_bn.parameters()) + list(enc.lstm.parameters())
                           + list(head.parameters()), lr=1e-3)
    Xt = torch.from_numpy(X).float()
    mean_t, std_t = torch.from_numpy(mean).to(dev), torch.from_numpy(std).to(dev)
    G = groups_of(dim)
    gidx = torch.zeros(len(G), dim, dtype=torch.bool)
    for k, cols in enumerate(G):
        gidx[k, cols] = True
    gidx = gidx.to(dev)
    log = []
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        tot = n = 0
        enc.train()
        for i in range(0, len(Xt), 64):
            x = Xt[perm[i:i + 64]].to(dev)
            B, T, _ = x.shape
            gm = torch.rand(B, T, len(G), device=dev) < p_group
            gm |= (torch.rand(B, T, 1, device=dev) < p_frame)
            mask = (gm.float() @ gidx.float()) > 0          # (B, T, dim)
            x_in = torch.where(mask, mean_t.expand_as(x), x)
            h = enc.input_bn(x_in.reshape(B * T, dim)).reshape(B, T, dim)
            out, _ = enc.lstm(h)
            rec = head(out)
            target = (x - mean_t) / std_t
            loss = ((rec - target) ** 2)[mask].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            tot += loss.item() * B
            n += B
        log.append(round(tot / n, 4))
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"pretrain epoch {ep:03d} masked-MSE {tot / n:.4f}", flush=True)
    state = {k: v.cpu() for k, v in enc.state_dict().items()
             if k.startswith(("input_bn.", "lstm."))}
    return state, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", required=True)
    ap.add_argument("--finetune", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--pretrain-epochs", type=int, default=40)
    ap.add_argument("--model-dir", default="D:/SIH-2026/.tmp/ml/models")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; never overwrite a result file")

    Xp, _, gp, mp = load_split(args.pretrain)
    print(f"pretrain windows {Xp.shape} from {mp['n_source_takes']} takes", flush=True)
    t0 = time.time()
    enc_state, curve = pretrain(Xp, args.pretrain_epochs)
    pre_wall = round(time.time() - t0, 1)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    torch.save(enc_state, f"{args.model_dir}/t15_encoder.pt")

    Base = tl.HARLSTMClassifier

    class Pretrained(Base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.load_state_dict(enc_state, strict=False)

    Xh, yh, gh, _ = load_split(args.finetune + "_holdout")
    rep = json.loads((Path(args.finetune) / "build_report.json").read_text())
    tag_of = {r["index"]: r["tag"] for r in rep["takes"]}
    runs = []
    tl.HARLSTMClassifier = Pretrained       # ponytail: monkeypatch keeps train_model byte-identical
    try:
        for seed in args.seeds:
            ck = f"{args.model_dir}/t15_ft_s{seed}.pt"
            t1 = time.time()
            val = tl.train_model(data_dir=args.finetune, output_path=ck, epochs=80, seed=seed)
            r = score(predict(ck, Xh), yh, gh, tag_of)
            runs.append({"seed": seed, "checkpoint": ck, "val_acc": round(val, 4),
                         "train_wall_s": round(time.time() - t1, 1), "holdout": r})
            print(f"RUN pretrained seed={seed} val={val:.3f} holdout={r['acc']:.4f}{r['ci95']}",
                  flush=True)
    finally:
        tl.HARLSTMClassifier = Base
    accs = [r["holdout"]["acc"] for r in runs]
    report = {"pretrain_dir": args.pretrain, "pretrain_takes": mp["n_source_takes"],
              "pretrain_windows": int(len(Xp)), "pretrain_wall_s": pre_wall,
              "pretrain_curve": curve, "finetune_dir": args.finetune, "runs": runs,
              "holdout_mean": round(float(np.mean(accs)), 4)}
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
