import queue
import threading

from config.experiment_config import STEP_CONFIRM_FRAMES, VOICE_REMINDER_INTERVAL_SEC
from pipeline.har_pipeline import HARPipeline
from pipeline.voice_alert import VoiceAlertSystem


def test_procedure_is_guided_with_a_full_dashboard_queue(tiny_lstm_ckpt):
    checkpoint = tiny_lstm_ckpt("voice.pt", feature_dim=143, feature_layout="pose132+obj11",
                               sample_every=3, rack_normalized=True)
    dashboard = queue.Queue(maxsize=1)
    dashboard.put("full")
    pipeline = HARPipeline(lstm_path=str(checkpoint), headless=True, enable_voice=False,
                           enable_recording=False, use_threaded=False, gui_queue=dashboard)
    state = pipeline.state_machine
    history = pipeline.voice.history

    def observe(step, confidence=.95):
        for _ in range(2 * STEP_CONFIRM_FRAMES):
            state.feed_prediction(step, confidence, entropy=.1)

    try:
        pipeline._start_guidance()
        pipeline._start_guidance()
        assert history == ["Experiment started. Step 1: Approach the main box."]
        observe(1, confidence=.2)
        assert len(history) == 1

        observe(1)
        assert history[-1] == "Step 1 confirmed. Next, Step 2: Open the main box."
        observe(3)
        assert state.expected_step_id == 2
        assert history[-1] == "Pause. Step 2 has not been confirmed. Step 2: Open the main box."
        count = len(history)
        observe(3)
        observe(4)
        assert len(history) == count
        pipeline._guide_waiting(0)
        pipeline._guide_waiting(VOICE_REMINDER_INTERVAL_SEC)
        assert history[-1] == history[-2]
        assert len(history) == count + 1

        observe(2)
        assert history[-1] == "Step 2 is now correct. Next, Step 3: Pick up the red box."
        observe(3)
        observe(1)
        assert history[-1] == "Pause. Action out of sequence. Step 4: Examine the red box."

        for step in range(4, 9):
            observe(step)
        assert history[-1] == "Experiment complete. All steps confirmed."
        pipeline._guide_waiting(2 * VOICE_REMINDER_INTERVAL_SEC)
        assert history[-1] == "Experiment complete. All steps confirmed."

        pipeline.reset_runtime()
        pipeline._start_guidance()
        pipeline._guide_waiting(0)
        pipeline._guide_waiting(VOICE_REMINDER_INTERVAL_SEC - 1)
        assert history[-1] == "Experiment started. Step 1: Approach the main box."
        pipeline._guide_waiting(VOICE_REMINDER_INTERVAL_SEC)
        assert history[-1] == "I could not confirm the action. Step 1: Approach the main box."
    finally:
        pipeline.close()


def test_voice_worker_falls_back_and_replaces_pending_cues(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    heard = []

    def broken(text):
        entered.set()
        assert release.wait(timeout=3)
        raise RuntimeError("Playback unavailable")

    def backends(self):
        yield "broken", broken
        yield "working", heard.append

    monkeypatch.setattr(VoiceAlertSystem, "_backends", backends)
    voice = VoiceAlertSystem()
    try:
        voice.speak("First cue")
        assert entered.wait(timeout=3)
        voice.speak("Old pending guidance")
        voice.speak("Current instruction", priority=True)
    finally:
        release.set()
        voice.close()
    assert heard == ["Current instruction"]
