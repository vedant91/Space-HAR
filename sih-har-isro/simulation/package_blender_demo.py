"""Package Blender RGB frames into a playable HAR evaluation video.

This is deliberately separate from rendering.  It makes two artifacts:
  * bas_blender_rgb.mp4: clean camera feed for actual HAR inference
  * bas_blender_ground_truth_preview.mp4: review-only video, labelled as
    simulation ground truth rather than model detection.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "blender_twin"
FRAMES = OUT / "raw_frames"


def main() -> dict:
    labels = json.loads((OUT / "annotations_ground_truth.json").read_text(encoding="utf-8"))
    width, height = labels["image_size"]
    fps = labels["fps"]
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    clean = cv2.VideoWriter(str(OUT / "bas_blender_rgb.mp4"), codec, fps, (width, height))
    review = cv2.VideoWriter(str(OUT / "bas_blender_ground_truth_preview.mp4"), codec, fps, (width, height))
    colors = {"astronaut": (255, 220, 60), "red_payload": (55, 55, 255), "yellow_payload": (0, 220, 255)}
    try:
        for frame in labels["frames"]:
            image = cv2.imread(str(OUT / frame["image"]))
            if image is None:
                raise FileNotFoundError(frame["image"])
            clean.write(image)
            annotated = image.copy()
            cv2.rectangle(annotated, (12, 12), (510, 45), (15, 20, 30), -1)
            cv2.putText(annotated, "SIMULATION GROUND TRUTH - NOT LIVE INFERENCE", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (120, 235, 255), 1, cv2.LINE_AA)
            for ann in frame["annotations"]:
                x, y, w, h = (round(value) for value in ann["bbox_xywh"])
                color = colors[ann["label"]]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(annotated, ann["label"], (x, max(20, y - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
            review.write(annotated)
    finally:
        clean.release(); review.release()
    return {"clean_video": str(OUT / "bas_blender_rgb.mp4"),
            "review_video": str(OUT / "bas_blender_ground_truth_preview.mp4"),
            "annotations": str(OUT / "annotations_ground_truth.json"), "frames": len(labels["frames"])}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
