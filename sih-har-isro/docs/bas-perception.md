# BAS pose and payload evaluation

The custom view combines measured MediaPipe landmarks with two depth estimates. The original RGB model contains 1,108,644 trainable parameters after removal of the MobileNet classifier. The new 100,385-parameter regressor learns hip-relative joint depths from normalized 2D landmarks and their visibility. It was trained from scratch. The MediaPipe Tasks weights are the supplied bundle. All three stages produce a 33-joint skeleton. Body shape and mesh vertices remain unmeasured.

The Blender rig supplies supervision for roll, pitch, yaw, rack interaction and out-of-plane motion. The old RGB scheduler could hold a pose for one second during a reach at constant torso roll. Each new custom view uses the current skeleton and RGB crop. A weak image joint can borrow a nearby RGB estimate, and capture timestamps control depth smoothing. The step classifier receives its trained feature layout.

## Running the models

From `sih-har-isro`, with model weights materialized by Git LFS, run the fused view with the twin-trained detector.

```sh
python main.py --mode pipeline --camera 0 --viz-3d hmr --detector-weights models/yolo_boxes_twin.pt
python tools/render_har_video.py --video clip.mp4 --out monitor.mp4 --viz-3d hmr --detector-weights models/yolo_boxes_twin.pt
```

Use `--viz-3d lift` for the skeleton-only depth model or `--viz-3d blazepose` for the default MediaPipe world view. The original payload detector remains the default for existing callers. The new YOLO checkpoint was fine-tuned for 24 epochs using the visible payloads in the rendered views.

## Results

Training used 42 animations with 2,724 labelled poses and 357 corresponding RGB observations. Nine separate validation animations supplied 76 RGB observations for checkpoint selection. Epoch 3 of 100 gave the best skeleton-only validation depth. Validation selected an equal blend of skeleton and RGB depth. A 12-take audit used new animation seeds, mixed attitudes, three lighting conditions and four camera mounts. Both depth estimators produced usable inputs on 99 of its 100 RGB frames.

| Audit measure | Existing RGB depth and MediaPipe xy | Fused BAS pose |
|---|---:|---:|
| Mean absolute depth error | 21.34 cm | 20.74 cm |
| Root-relative mean per-joint position error | 439.5 mm | 408.7 mm |
| Visible body joints within half a projected torso length | 25.2% | 35.2% |

The pose comparisons use the same 497 visible joint observations from shoulders through ankles. PCK also requires predicted visibility of at least 0.15. The metric reconstruction uses the known Blender hip distance and camera calibration to unproject each prediction. Live display uses an estimated torso scale. The skeleton-only model reached 22.30 cm depth error. Across individual audit takes, fused depth error ranged from 8.5 to 38.1 cm. Left and right wrist depth errors averaged 29.4 and 30.1 cm.

Four additional motion takes supplied 260 frames 100 ms apart. Smoothing reduced the fused model’s mean error in frame-to-frame depth change from 5.72 to 2.93 cm per interval. With the same filter the RGB depth baseline reached 2.78 cm. Filtered fused depth error was 14.04 cm against 11.12 cm for filtered RGB, while fused 3D joint error was 373 mm against 410 mm for RGB. These measurements precede the final rigid-bone display pass. The audit table uses individual unfiltered frames.

| Payload at IoU ≥ 0.5 and confidence ≥ 0.25 | Existing correct / visible | Twin model correct / visible | False positives, existing → twin |
|---|---:|---:|---:|
| Red box | 43 / 66 | 45 / 66 | 2 → 14 |
| Yellow box | 38 / 92 | 77 / 92 | 24 → 23 |
| Main box | 83 / 100 | 100 / 100 | 4 → 0 |
| All classes | 164 / 258 | 222 / 258 | 30 → 37 |

All-class precision increased from 84.5% to 85.7%. The red-box precision regression matters for applications that act on a single detection. The audit measures synthetic performance on one astronaut rig. A NASA clip exercised the file capture path and a generated astronaut clip exercised video export. Neither clip supplies metric 3D ground truth. [Per-take results and joint errors](bas-perception-results.json)

## Reproducing the measurements

The archive contains exact per-take poses, camera scales, measured landmarks and RGB HMR outputs for training, validation, audit and motion evaluation. Unpacking it is sufficient to train or score the depth model without Blender.

```sh
python -m zipfile -e dataset/bas_depth_samples.zip /tmp/bas-depth
python train/train_depth_lifter.py /tmp/bas-depth --evaluate --split audit
python train/train_depth_lifter.py /tmp/bas-depth --evaluate --split motion
python train/train_depth_lifter.py /tmp/bas-depth --out /tmp/bas-depth-fit.pt
```

To regenerate the images and detections, use the repository’s Blender scene builder. These runs used Blender 4.3, Python 3.11, MediaPipe 0.10.21 and Ultralytics 8.4.215. The detector evaluator matches predictions one-to-one with ray-tested visible objects. Boxes projected behind opaque objects do not enter its recall denominator.

```sh
blender --background --python blender/export_depth_takes.py -- --out /tmp/bas-rendered --previews
python tools/cache_depth_observations.py /tmp/bas-rendered
python train/train_twin_boxes.py /tmp/bas-rendered /tmp/bas-yolo
python tools/score_payload_boxes.py --dataset /tmp/bas-rendered/audit --weights models/yolo_boxes_twin.pt
```
