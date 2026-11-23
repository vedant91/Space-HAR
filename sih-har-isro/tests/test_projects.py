import json

import pytest
import torch

from config.experiment_config import STEP_CONFIRM_FRAMES
from config.project import load_project
from pipeline.har_pipeline import HARPipeline


def write_project(tmp_path, **overrides):
    path = tmp_path / "procedure.json"
    data = {
        "id": "inspection",
        "name": "Panel inspection",
        "checkpoint": "inspection.pt",
        "steps": [
            {"id": 30, "name": "Position the equipment", "voice_cue": "Place the unit on the bench."},
            {"id": 10, "name": "Inspect the panel"},
            {"id": 20, "name": "Store the equipment"},
        ],
        "voice": {"enabled": False, "reminder_seconds": 3, "system_voice": "Alex"},
    }
    path.write_text(json.dumps(data | overrides))
    return path


def test_custom_checkpoint_drives_order_and_spoken_instructions(tmp_path, tiny_lstm_ckpt):
    path = write_project(tmp_path)
    checkpoint = tiny_lstm_ckpt("inspection.pt", feature_dim=132, feature_layout="pose132",
                               num_classes=3, rack_normalized=False, sample_every=3)
    data = torch.load(checkpoint, weights_only=False)
    data.update(project_id="inspection", idx_to_label={0: 10, 1: 20, 2: 30})
    torch.save(data, checkpoint)
    project = load_project(path)
    assert project.checkpoint == checkpoint
    pipeline = HARPipeline(project=project, headless=True, use_threaded=False)
    state = pipeline.state_machine
    history = pipeline.voice.history

    def observe(step):
        for _ in range(2 * STEP_CONFIRM_FRAMES):
            state.feed_prediction(step, .95, entropy=.1)

    try:
        assert pipeline._lstm_label_map[2] == 30
        assert not pipeline.voice.enabled
        assert pipeline.voice.system_voice == "Alex"
        pipeline._start_guidance()
        assert history[-1] == "Starting Panel inspection. Place the unit on the bench."
        observe(0)
        observe(99)
        assert state.expected_step_id == 30
        assert not state.recovery_required

        observe(30)
        assert history[-1] == "Step 1 confirmed. Next, Step 2. Inspect the panel."
        observe(20)
        assert state.recovery_required and state.expected_step_id == 10
        pipeline._guide_waiting(0)
        pipeline._guide_waiting(3)
        assert history[-1] == "Pause. Step 2 has not been confirmed. Step 2. Inspect the panel."
        observe(10)
        assert history[-1] == "Step 2 is now correct. Next, Step 3. Store the equipment."
        observe(20)
        assert state.experiment_complete
        pipeline.reset_runtime()
        assert state.expected_step_id == 30
    finally:
        pipeline.close()

    data["idx_to_label"][2] = 3
    torch.save(data, checkpoint)
    with pytest.raises(ValueError, match="must map classifier outputs"):
        HARPipeline(project=project, headless=True)


def test_loader_rejects_duplicate_ids_and_resolves_voice_asset(tmp_path):
    path = write_project(tmp_path, steps=[{"id": 3, "name": "Pick"}, {"id": 3, "name": "Place"}])
    with pytest.raises(ValueError, match="unique positive integer"):
        load_project(path)

    path = write_project(tmp_path, voice={"piper_model": "voices/guide.onnx", "reminder_seconds": 0})
    project = load_project(path)
    assert project.piper_model == tmp_path / "voices/guide.onnx"
    assert project.reminder_seconds == 0


def test_partial_video_starts_at_selected_project_id():
    from pipeline.state_machine import ExperimentStateMachine
    steps = [{"id": step, "name": str(step)} for step in (30, 10, 20)]
    sm = ExperimentStateMachine(steps, start_step=10)
    assert sm.expected_step_id == 10
    assert [record.step_id for record in sm.step_records] == [10, 20]
    sm.reset()
    assert sm.expected_step_id == 10
    with pytest.raises(ValueError, match="Unknown playback start"):
        ExperimentStateMachine(steps, start_step=99)
