# Set up an activity recognition project

A project JSON file defines the order of the procedure, the activity classifier to load, and the spoken instructions. The launcher reads it at startup. The built-in BAS procedure is used when the launcher has no project file.

1. Copy [the equipment inspection example](../projects/equipment-inspection.example.json) to a new JSON file. Set its unique project `id`, display `name`, and ordered `steps`. Each step needs a positive integer model label `id` and a `name`. Use an action phrase for the name so the generated speech reads naturally. Set `voice_cue` on a step to use exact wording.
2. Set `checkpoint` to the trained activity classifier's `.pt` file. Relative paths start at the project file. The checkpoint's `idx_to_label` mapping must cover every step ID. Label `0` can represent idle activity. The file's step order determines the procedure and the numbers spoken aloud.
3. Launch using that project.

```sh
python main.py --project projects/my-project.json --camera 0 --headless
```

The GUI uses the same procedure when you omit `--headless`. Use `--video recording.mp4` to run the project on a saved recording. `--voice` and `--no-voice` override the project's voice setting for a run.

## Train for new actions

Voice prompts are generated from the procedure. The activity classifier must first learn the actions from labelled recordings. For pose-based projects, collect multiple takes with representative operators, camera viewpoints, and lighting. Record each action long enough to fill the classifier's sample window. Reserve complete takes to check recognition before using voice guidance in the real procedure.

From `sih-har-isro`, run the recorder once per take, build pose sequences, then train the project checkpoint.

```sh
python tools/record_take.py --project projects/my-project.json --out dataset/my_project
python tools/build_sequences.py --dataset dataset/my_project --out dataset/my_project_sequences --layout pose132
python train/train_lstm.py --project projects/my-project.json --data dataset/my_project_sequences
```

Record with Space. Use the number keys to label the first nine steps or the bracket keys to move through the list. Save with Q. The trainer writes the configured checkpoint and records the project ID. The runner also accepts an existing compatible checkpoint that records matching class IDs. Check that the labels in an older checkpoint refer to the same actions. Changing the cue text requires only a restart. Changing which action a label represents requires matching training data and weights.

The pose classifier tracks camera-visible actions represented in its training data. `pose132` is the general pose input. The existing `pose132+obj11` layout uses the BAS payload channels. A procedure that depends on new objects or signals invisible to the camera may require additional perception features and training.

## Choose the voice

- `voice.enabled` turns spoken guidance on or off for the project. It defaults to on.
- `voice.reminder_seconds` sets the interval for a step that has not been confirmed. It defaults to 20. Set it to 0 to disable reminders.
- `voice.system_voice` selects an installed system voice. On macOS, list the installed names with `say -v '?'`.
- `voice.piper_model` points to a local Piper `.onnx` voice. Keep its `.onnx.json` config beside it. When Piper and `ffplay` are available, the configured Piper voice takes precedence over the system voice.

Speech is synthesized offline from the configured text. Startup, confirmation, correction, reminder, and completion cues use the same selected voice.

### Learn from reviewed activity mistakes

For recordings that begin partway through a procedure, pass the first expected step
ID with `--video clip.mp4 --start-step 3`. This limits the checklist and spoken
guidance to the recorded part of the procedure. Camera sessions use the full protocol.

Review an interval in which the same action is visible throughout. In the example
below, 3 is the confirmed action ID and 2 and 7.8 are seconds from the start. The
tool runs the project's actual perception pipeline and exports complete feature windows.
Repeat on a different recording for evaluation. Use the same detector weights for
both recordings and playback. For custom projects pass `--project` during extraction.

```sh
.venv/bin/python tools/review_activity_clip.py train-clip.mp4 3 2 7.8 corrections/train.npz
.venv/bin/python tools/review_activity_clip.py audit-clip.mp4 3 1 5 corrections/audit.npz
.venv/bin/python tools/learn_from_errors.py models/lstm_blender_core.pt \
  dataset/seq_blender_core dataset/seq_blender_core_holdout runs/activity-learning/new.pt \
  --corrections corrections/train.npz --review-audit corrections/audit.npz
```

The learner updates the final activity layer, automatically giving extra weight to
mislabeled-by-the-model windows on each pass. The reference labels come from the
digital twin or reviewed clips. Training and audit take IDs must be disjoint. Use
the same take-ID namespace and preprocessing for both sequence sets. Older sequence
sets may omit detector provenance. The output reports baseline correct, candidate
correct and total windows per action, plus the separate reviewed-video result.
Overlapping windows from one video describe that recording only. Preview a candidate
with `--activity-checkpoint runs/activity-learning/new.pt` or a copy of the project's
JSON pointing to that file. The default classifier is updated separately after review.
