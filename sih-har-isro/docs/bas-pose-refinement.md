# BAS pose refinement

The refined monitor view runs MediaPipe and the existing RGB HMR crop model. The first BAS lifter combines their estimates, then a 1,561,699-parameter residual network corrects the horizontal, vertical and hip-relative depth coordinates of all 33 joints. The new network was trained from scratch on labelled Blender renders. It retains the input visibility scores and uses the existing timestamp-based depth filter and rigid-bone renderer.

Training used 42 original training takes and 28 additional, more densely rendered takes (632 additional RGB observations) from four camera mounts and three lighting conditions. Checkpoint selection used nine separate validation takes. Epoch 44 of 45 had the lowest validation joint-position error. Twelve further animation seeds were reserved for evaluation.

| Metric on 101 held-out frames | Published fusion | Refined pose |
|---|---:|---:|
| Mean root-relative 3D joint error | 434 mm | 326 mm |
| 90th percentile 3D joint error | 782 mm | 656 mm |
| Mean absolute depth error | 24.5 cm | 19.6 cm |
| Visible joints within half a projected torso length | 40.8% | 52.4% |
| Left wrist 3D joint error | 578 mm | 397 mm |
| Right wrist 3D joint error | 569 mm | 408 mm |

The comparison uses 593 visible shoulder, arm, hip and leg joints. Both estimates use the known Blender camera calibration and hip distance when scored. Capture uses an estimated display scale. All 12 held-out takes had lower mean 3D error with refinement.

On the separate 260-frame motion set, mean 3D error fell from 375 to 225 mm. Error in frame-to-frame depth change after filtering increased from 2.93 to 3.19 cm per 100 ms. These values measure pose estimates before rigid-bone rendering. Quantitative validation on real astronauts and across suit designs remains open.

## Run and reproduce

Run from the `sih-har-isro` directory. Fetch the models through Git LFS when initializing a new checkout.

```sh
python main.py --mode pipeline --camera 0 --viz-3d refined
python tools/render_har_video.py --video clip.mp4 --out monitor.mp4 --viz-3d refined
python -m zipfile -e dataset/bas_depth_samples.zip /tmp/bas-refinement
python -m zipfile -e dataset/bas_pose_refinement_samples.zip /tmp/bas-refinement
python train/train_pose_refiner.py /tmp/bas-refinement --evaluate refine_test
python train/train_pose_refiner.py /tmp/bas-refinement --evaluate motion
python train/train_pose_refiner.py /tmp/bas-refinement --epochs 45 --out /tmp/bas-refit.pt
```

CPU retraining from the two observation archives reproduced every saved tensor. To regenerate the added images and observations, run the Blender exporter with `--previews --split refine_train` and `--previews --split refine_test` followed by `tools/cache_depth_observations.py` on those splits.
