"""Refit the deployed activity model's final layer using labeled mistakes and replay."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.promote_model import load_lstm
from pipeline.features import contract_of


def read_split(path, checkpoint):
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("project_id") != checkpoint.get("project_id"):
        raise ValueError(f"{path}: sequence set and checkpoint have different project IDs")
    for key, expected in contract_of(checkpoint).items():
        if meta.get(key) is not None and meta[key] != expected:
            raise ValueError(f"{path}: {key} differs from checkpoint")
    x = np.load(path / "X_sequences.npy")
    labels = np.load(path / "y_labels.npy")
    groups = np.load(path / "groups.npy")
    if x.shape != (len(labels), checkpoint["window"], checkpoint["feature_dim"]) or len(groups) != len(x):
        raise ValueError(f"{path}: incompatible arrays")
    y = torch.tensor([int(checkpoint["label_to_idx"][int(label)]) for label in labels])
    if y.unique().numel() != checkpoint["num_classes"]:
        raise ValueError(f"{path}: all project steps must be represented")
    return torch.from_numpy(x).float(), y, set(groups.tolist())


def reviewed(paths, checkpoint):
    windows, labels, sources = [], [], set()
    for path in paths:
        with np.load(path) as bundle:
            source = str(bundle["source"])
            sources.add(source)
            expected = {"project_id": checkpoint.get("project_id", ""), "layout": checkpoint["feature_layout"],
                        "sample_every": checkpoint["sample_every"], "rack_normalized": checkpoint["rack_normalized"]}
            if any(bundle[key].item() != value for key, value in expected.items()):
                raise ValueError(f"{path}: mismatched project or features")
            data = torch.from_numpy(bundle["x"]).float()
            if data.shape[1:] != (checkpoint["window"], checkpoint["feature_dim"]) or not len(data):
                raise ValueError(f"{path}: incompatible windows")
            windows.append(data)
            labels.append(torch.full((len(data),), int(checkpoint["label_to_idx"][int(bundle["step"])])))
    return torch.cat(windows), torch.cat(labels), sources


def learn(checkpoint, training, audit, output, corrections=(), review_audit=()):
    if output.exists():
        raise ValueError(f"Choose a fresh candidate path: {output}")
    model, ckpt = load_lstm(checkpoint)
    x, y, training_groups = read_split(training, ckpt)
    test, truth, audit_groups = read_split(audit, ckpt)
    if training_groups & audit_groups:
        raise ValueError("Training and audit share source takes")
    head = model.classifier[-1]
    model.classifier[-1] = torch.nn.Identity()
    with torch.no_grad():
        features = torch.cat([model(batch) for batch in x.split(32)])
        audit_features = torch.cat([model(batch) for batch in test.split(32)])
        reference = head(features).detach()
        before = head(audit_features).argmax(1)
    if corrections or review_audit:
        if not corrections or not review_audit:
            raise ValueError("Supply both training corrections and independently reviewed audit videos")
        correction_x, correction_y, sources = reviewed(corrections, ckpt)
        clip_x, clip_y, audit_sources = reviewed(review_audit, ckpt)
        if sources & audit_sources:
            raise ValueError("Training and audit contain the same video")
        with torch.no_grad():
            correction_features = torch.cat([model(batch) for batch in correction_x.split(32)])
            clip_features = torch.cat([model(batch) for batch in clip_x.split(32)])
            clip_before = head(clip_features).argmax(1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    for _ in range(100):
        logits = head(features)
        # Recompute mistakes after each update. Replay every labeled window
        # and penalize drift from the original classifier on familiar actions.
        weights = 1.0 + (logits.detach().argmax(1) != y).float()
        supervised = (weights * F.cross_entropy(logits, y, reduction="none")).mean()
        drift = F.kl_div(logits.log_softmax(1), reference.softmax(1), reduction="batchmean")
        loss = supervised + 0.1 * drift
        if corrections:
            # Keep the incumbent responses on replay data while adapting to the reviewed domain.
            loss = (0.1 * loss + drift
                    + F.cross_entropy(head(correction_features), correction_y))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after = head(audit_features).detach().argmax(1)
    report = {}
    for index, label in ckpt["idx_to_label"].items():
        mask = truth == int(index)
        report[int(label)] = [int((before[mask] == truth[mask]).sum()),
                              int((after[mask] == truth[mask]).sum()), int(mask.sum())]
    if corrections:
        clip_after = head(clip_features).detach().argmax(1)
        report["reviewed_video"] = [int((clip_before == clip_y).sum()), int((clip_after == clip_y).sum()), len(clip_y)]
    model.classifier[-1] = head
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(dict(ckpt, model_state=model.state_dict(), error_replay_audit=report), stream)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "training", "audit", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--corrections", nargs="+", type=Path, default=[])
    parser.add_argument("--review-audit", nargs="+", type=Path, default=[])
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(learn(**vars(args)), indent=2))
