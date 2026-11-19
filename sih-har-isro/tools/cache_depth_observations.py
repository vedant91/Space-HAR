"""Pair measured 2D skeletons and the existing RGB HMR baseline with rig labels."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.har_pipeline import OptimizedMPWrapper, _make_box_detector
from pipeline.hmr_backend import HMRBackend


def cache(root):
    pose = OptimizedMPWrapper()
    incumbent = HMRBackend("custom")
    if not pose.available or not incumbent.available:
        raise RuntimeError("Scoring requires both the MediaPipe bundle and the RGB HMR checkpoint")
    try:
        for path in sorted(root.glob("*/*/pose.npz")):
            with np.load(path) as labels:
                width, height = labels["size"]
            detector = _make_box_detector(int(width), int(height))
            pose.reset()
            incumbent.reset()
            offsets, skeletons, rgb_depths = [], [], []
            previous = None
            for image in sorted(path.parent.glob("*.png")):
                bgr = cv2.imread(str(image))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                skeletons.append(pose.process(rgb, detections=previous).reshape(33, 4))
                previous = detector.detect(bgr)
                rgb_depths.append(incumbent.get_pose_features(rgb)[0].reshape(33, 4))
                offsets.append(int(image.stem))
            if not offsets:
                raise ValueError(f"{path}: export with --previews to measure image inputs")
            np.savez_compressed(path.with_name("observations.npz"), offset=offsets,
                                skeleton=skeletons, rgb_hmr=rgb_depths)
            print(path.parent.name, len(offsets), flush=True)
    finally:
        pose.close()
        incumbent.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    cache(parser.parse_args().dataset)
