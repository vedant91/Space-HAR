"""
Upload dataset/yolo_detection/images/unlabeled/'s real frames to a Roboflow
project for hand-labeling, matching the class schema already in
dataset/yolo_detection/data.yaml (0:hand, 1:red_box, 2:yellow_box,
3:main_box_open, 4:main_box_closed, 5:person).

Needs a Roboflow API key (Settings -> Roboflow API in your workspace,
https://app.roboflow.com/settings/api) — never hardcode it, read from the
ROBOFLOW_API_KEY env var only.

Usage:
    export ROBOFLOW_API_KEY=...
    python tools/upload_to_roboflow.py --workspace <your-workspace-slug> \
        --project har-real-frames --create
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CLASS_NAMES = ["hand", "red_box", "yellow_box", "main_box_open", "main_box_closed", "person"]
DEFAULT_SRC = REPO / "dataset" / "yolo_detection" / "images" / "unlabeled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Roboflow workspace slug (URL after app.roboflow.com/)")
    ap.add_argument("--project", default="har-real-frames")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--create", action="store_true",
                    help="Create the project if it doesn't exist yet (object-detection, this schema)")
    args = ap.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY first (Settings -> Roboflow API in your workspace).")

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(args.workspace)

    if args.create:
        try:
            ws.create_project(project_name=args.project, project_type="object-detection",
                              project_license="MIT", annotation="hand")
            print(f"Created project '{args.project}' in workspace '{args.workspace}'.")
        except Exception as e:
            print(f"create_project skipped ({e}) — assuming it already exists.")

    project = ws.project(args.project)

    files = sorted(glob.glob(str(Path(args.src) / "*.jpg")))
    if not files:
        raise SystemExit(f"No .jpg files under {args.src}")

    n_ok, n_fail = 0, 0
    for i, f in enumerate(files, 1):
        try:
            project.upload(f, batch_name="real_webcam_frames")
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"  FAILED {f}: {e}")
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)} uploaded ({n_ok} ok, {n_fail} failed)")

    print(f"\nDone. {n_ok} uploaded, {n_fail} failed. "
         f"Open https://app.roboflow.com/{args.workspace}/{args.project} to label them — "
         f"class list should match {CLASS_NAMES}.")


if __name__ == "__main__":
    main()
