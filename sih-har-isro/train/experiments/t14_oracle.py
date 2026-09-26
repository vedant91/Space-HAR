"""T14 oracle experiment: train on one pose source, score on held-out windows
of every pose source, over several seeds.

The windows come from tools/build_sequences.py (the `<out>_holdout` dirs), so
they are built by the same pipeline.features calls evaluate_real uses; the
point of scoring here rather than through evaluate_real is to afford several
seeds and the cross terms (gt-trained model on MediaPipe windows and back).

95% CI: cluster bootstrap over (take, step) units, because overlapping
stride-1 windows of one step performance are not independent samples. Four
held-out takes x 8 steps = 32 clusters, which is the real sample size.

Usage:
    python train/experiments/t14_oracle.py \
        --train gt=D:/SIH-2026/.tmp/ml/seq_gt mp=dataset/seq_blender_core \
        --seeds 7 11 23 --out logs/t14_oracle.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_split(d: str):
    p = Path(d)
    return (np.load(p / "X_sequences.npy"), np.load(p / "y_labels.npy"),
            np.load(p / "groups.npy"), json.loads((p / "metadata.json").read_text()))


def predict(ckpt_path: str, X: np.ndarray) -> np.ndarray:
    import torch
    from train.train_lstm import HARLSTMClassifier
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = HARLSTMClassifier(ck["feature_dim"], ck["hidden_size"], ck["num_layers"],
                          ck["num_classes"])
    m.load_state_dict(ck["model_state"])
    m.eval()
    with torch.no_grad():
        idx = torch.cat([m(torch.from_numpy(X[i:i + 512]).float()).argmax(1)
                         for i in range(0, len(X), 512)]).numpy()
    lut = {int(k): int(v) for k, v in ck["idx_to_label"].items()}
    return np.array([lut[i] for i in idx])


def score(pred, y, g, tag_of, n_boot=2000, seed=0) -> dict:
    ok = (pred == y).astype(float)
    units = [(a, b) for a, b in sorted(set(zip(g.tolist(), y.tolist())))]
    per_unit = np.array([[ok[(g == a) & (y == b)].sum(), ((g == a) & (y == b)).sum()]
                         for a, b in units])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = per_unit[rng.integers(0, len(units), len(units))].sum(0)
        boots.append(s[0] / s[1])
    per_take = {tag_of.get(int(t), str(t)): round(float(ok[g == t].mean()), 4)
                for t in np.unique(g)}
    per_step = {int(s): round(float(ok[y == s].mean()), 4) for s in np.unique(y)}
    return {"acc": round(float(ok.mean()), 4), "windows": int(len(y)),
            "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                     round(float(np.percentile(boots, 97.5)), 4)],
            "n_clusters": len(units), "per_take": per_take, "per_step": per_step}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="name=seq_dir (train split)")
    ap.add_argument("--test-only", nargs="*", default=[],
                    help="name=seq_dir scored on its _holdout split, never trained on")
    ap.add_argument("--seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--model-dir", default="D:/SIH-2026/.tmp/ml/models")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; never overwrite a result file")

    from train.train_lstm import train_model

    sources = dict(s.split("=", 1) for s in args.train)
    tests = {**sources, **dict(s.split("=", 1) for s in args.test_only)}
    hold = {n: load_split(d + "_holdout") for n, d in tests.items()}
    tag_of = {}
    for n, d in tests.items():
        rep = json.loads((Path(d) / "build_report.json").read_text())
        tag_of.update({r["index"]: r["tag"] for r in rep["takes"]})
    train_meta = {n: load_split(d)[3] for n, d in sources.items()}

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    runs = []
    for n, d in sources.items():
        for seed in args.seeds:
            ck = f"{args.model_dir}/t14_{n}_s{seed}.pt"
            t0 = time.time()
            val = train_model(data_dir=d, output_path=ck, epochs=args.epochs, seed=seed)
            wall = round(time.time() - t0, 1)
            row = {"train_source": n, "seed": seed, "checkpoint": ck, "val_acc": round(val, 4),
                   "train_takes": train_meta[n]["n_source_takes"], "train_wall_s": wall,
                   "test": {}}
            for tn, (X, y, g, _) in hold.items():
                row["test"][tn] = score(predict(ck, X), y, g, tag_of)
            print(f"RUN train={n} seed={seed} val={val:.3f} "
                  + " ".join(f"{tn}={r['acc']:.4f}{r['ci95']}" for tn, r in row["test"].items()),
                  flush=True)
            runs.append(row)

    summary = {}
    for n in sources:
        for tn in hold:
            accs = [r["test"][tn]["acc"] for r in runs if r["train_source"] == n]
            summary[f"{n}->{tn}"] = {"mean": round(float(np.mean(accs)), 4),
                                     "min": min(accs), "max": max(accs), "seeds": len(accs)}
    report = {"test_set": "holdout takes " + ",".join(sorted(set(tag_of[int(t)]
              for t in np.unique(next(iter(hold.values()))[2])))),
              "chance": 0.125, "sources": sources, "runs": runs, "summary": summary}
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
