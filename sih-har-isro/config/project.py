"""Load a procedure and its activity checkpoint from a project JSON file."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from config.experiment_config import VOICE_ENABLED, VOICE_REMINDER_INTERVAL_SEC


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    steps: tuple[dict, ...]
    checkpoint: Path
    voice_enabled: bool
    reminder_seconds: float
    system_voice: str | None
    piper_model: Path | None

    @property
    def step_ids(self) -> set[int]:
        return {step["id"] for step in self.steps}


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def load_project(path: str | Path) -> Project:
    path = Path(path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("expected a JSON object")
        project_id = _text(raw.get("id"), "id")
        name = _text(raw.get("name"), "name")
        checkpoint = path.parent / _text(raw.get("checkpoint"), "checkpoint")
        if checkpoint.suffix != ".pt":
            raise ValueError("checkpoint must name the activity classifier's .pt file")

        steps = raw.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("steps must be a nonempty ordered list")
        seen = set()
        normalized = []
        for index, step in enumerate(steps, start=1):
            prefix = f"steps[{index - 1}]"
            step_id = step.get("id") if isinstance(step, dict) else None
            if type(step_id) is not int or step_id < 1 or step_id in seen:
                raise ValueError(f"{prefix}.id must be a unique positive integer")
            seen.add(step_id)
            name_of_step = _text(step.get("name"), f"{prefix}.name")
            entry = {"id": step_id, "name": name_of_step, "number": index}
            if "voice_cue" in step:
                entry["voice_cue"] = _text(step["voice_cue"], f"{prefix}.voice_cue")
            if "required_objects" in step:
                objects = step["required_objects"]
                if not isinstance(objects, list):
                    raise ValueError(f"{prefix}.required_objects must be a list of names")
                entry["required_objects"] = [_text(obj, f"{prefix}.required_objects") for obj in objects]
            normalized.append(entry)

        voice = raw.get("voice", {})
        if not isinstance(voice, dict):
            raise ValueError("voice must be an object")
        enabled = voice.get("enabled", VOICE_ENABLED)
        if type(enabled) is not bool:
            raise ValueError("voice.enabled must be true or false")
        interval = voice.get("reminder_seconds", VOICE_REMINDER_INTERVAL_SEC)
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval < 0:
            raise ValueError("voice.reminder_seconds must be a finite, nonnegative number")
        system_voice = _text(voice["system_voice"], "voice.system_voice") if "system_voice" in voice else None
        piper = path.parent / _text(voice["piper_model"], "voice.piper_model") if "piper_model" in voice else None
        return Project(project_id, name, tuple(normalized), checkpoint, enabled, interval, system_voice, piper)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Project {path}: {exc}") from exc
