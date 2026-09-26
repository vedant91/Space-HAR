# Datasets

What each source feeds, where it lives, and how to regenerate it. A source
that feeds nothing in this pipeline is listed under *Not collected* with the
reason, so the decision is not re-made from scratch.

## Collected / generated here

| Folder | What | Feeds | Status |
|---|---|---|---|
| `blender_core/` | Digital-twin renders: 23 takes, roll/yaw/pitch attitude sweeps, lighting, clutter, skip/recover, unseen camera. Exact per-frame labels, 2D + 3D pose ground truth, true box positions | LSTM training + held-out eval; the pitch/yaw takes are the HMR gate (does 2D break out-of-plane?) | **Rendering** |
| `real/` | Real performances of the 8-step protocol on a webcam, labelled live | The only real-domain evaluation; fine-tuning | **Needs you** — see below |
| `nasa_iss/` | 8 NASA public-domain clips of crew working hardware in microgravity (`manifest.json` has provenance) | Qualitative check: does pose detection hold on real astronauts at real attitudes? Unlabelled | Done, 74 MB |
| `seq_final_rack*/`, `seq_final_raw*/` | Built windows from the old 12-take render, whose frames no longer exist | The published 0.201 baseline only | Frozen, do not rebuild |
| `annotated/`, `skeleton_sequences*/`, `space_sim/` | Output of the retired 2D simulator (boxes painted in the detector's own colours) | Nothing honest. Kept for history | Retired |
| `synthetic_gemini/` | 2 generated clips | Nothing — no per-frame labels, uncontrolled timing | Retired |

### Regenerate

```bash
# check the generator without rendering images (minutes)
blender --background --factory-startup --python blender/render_dataset.py -- \
    --out .tmp/gtcheck --plan core --gt-only
python tools/check_dataset.py .tmp/gtcheck

# full render (resumable; relaunches across driver crashes)
BLENDER='D:/SIH-2026/tools/blender-5.2.0/blender-5.2.0-windows-x64/blender.exe' \
    tools/render_dataset_resilient.sh dataset/blender_core core 900 506 16
python tools/check_dataset.py dataset/blender_core
```

Held out by default (never trained on): `orient_180`, `orient_-60`, `yaw_90`,
`pitch_60`. Excluded from training entirely: `skip`, `recover`, `heldout_view`.

### Recording real takes

Props: a white container with a lid, a red box, a yellow box, two "trays"
(left and right). Fixed webcam, framed like the payload camera.

```bash
python tools/record_take.py --out dataset/real --tag real_nominal
# SPACE starts; press 1..8 the moment each step BEGINS; q saves.
python tools/record_take.py --out dataset/real --tag real_skip      # leave a step out
python tools/record_take.py --out dataset/real --tag real_recover   # skip, then go back
python tools/check_dataset.py dataset/real --holdout ""
```

Target: 10 nominal, 3 skip, 2 recover, from at least 2 different people.
Vary where you stand and which hand you use; that variety is the point.

## Needs a registration only you can do

| What | Why we need it | Where |
|---|---|---|
| **SMPL body model** (`SMPL_NEUTRAL.pkl` v1.1.0 + ROMP metadata) | Required by every HMR backend. No SMPL file, no 3D HMR, whichever model | smpl.is.tue.mpg.de (free, non-commercial) |
| NTU RGB+D 120 **skeletons only** (5.8 GB + 4.5 GB) | Optional pretraining track only. 25 Kinect joints, not MediaPipe's 33 — not drop-in training data | rose1.ntu.edu.sg/dataset/actionRecognition |

## Not collected, and why

| Source | Reason |
|---|---|
| EPIC-Kitchens | Egocentric (camera on the head): the actor's body is not in frame. Our camera is fixed and third-person |
| Kaggle smartphone HAR | Accelerometer data, not vision |
| UCF101, Kinetics-400 | General/sports actions, nothing like the protocol; Kinetics is ~450 GB against 145 GB free |
| Assembly101 full | ~700 GB. Closest domain, but only worth a curated subset once real-take evaluation exists |
| Something-Something v2 | Would pretrain a video CNN; the CNN plan uses a frozen ImageNet backbone instead |
| COIN | YouTube IDs, many dead, generic tasks |
| MPII | Single images, no sequences |
| Veo / Gemini generation | No per-frame labels, uncontrolled timing, needs network and an API key |
| SURREAL | Registration-gated synthetic humans; our own twin is closer to the domain |
