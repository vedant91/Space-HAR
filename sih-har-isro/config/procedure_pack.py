"""
Procedure Pack Loader
========================
Loads an experiment protocol from a YAML "pack" instead of a hardcoded
Python step table, so a second experiment can be demoed by swapping a file,
not editing `pipeline/state_machine.py` or any other FSM code.

Fails loud: a missing/invalid pack raises ProcedurePackError immediately at
load time (config/experiment_config.py loads the active pack at import
time) — there is deliberately no silent fallback to a default protocol,
since running the WRONG protocol without noticing would be worse than a
crash. See packs/box_sort_v1.yaml for the default pack (the current
protocol, exported 1:1) and packs/toy_3step.yaml for a second, minimal
pack proving the format actually swaps freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REQUIRED_TOP_FIELDS = ("pack_version", "experiment_id", "name", "steps")
REQUIRED_STEP_FIELDS = ("id", "name", "required_objects", "voice_cue")


class ProcedurePackError(ValueError):
    """Raised for any malformed pack — never caught into a silent fallback."""


@dataclass
class ProcedurePack:
    pack_version: int
    experiment_id: str
    name: str
    steps: List[Dict]
    camera_hints: Dict = field(default_factory=dict)
    fsm: Dict = field(default_factory=dict)
    alerts: Dict = field(default_factory=lambda: {"voice": True})
    source_path: Optional[str] = None

    @property
    def experiment_steps(self) -> List[Dict]:
        """The shape config.experiment_config.EXPERIMENT_STEPS has always
        had — pipeline/state_machine.py and everything downstream of it
        needs no changes to consume a pack loaded through this property."""
        out = []
        for s in self.steps:
            out.append({
                "id": s["id"],
                "name": s["name"],
                "description": s.get("description", s["name"]),
                "duration_hint_sec": s.get("duration_hint_sec", s.get("timeout_s", 30)),
                "required_objects": list(s["required_objects"]),
                "voice_cue": s.get("voice_cue") or s.get("next_prompt")
                            or f"Step {s['id']}: {s['name']}.",
                # Forward-compatible optional fields (unused by the FSM today —
                # G3/G4 in the build plan read these; keep them passed through
                # rather than dropped so a pack author can set them now).
                "success_evidence": s.get("success_evidence", []),
                "forbidden_zones": s.get("forbidden_zones", []),
                "timeout_s": s.get("timeout_s", s.get("duration_hint_sec", 30)),
                "irreversible": bool(s.get("irreversible", False)),
            })
        return out


def load_pack(path: str) -> ProcedurePack:
    p = Path(path)
    if not p.exists():
        raise ProcedurePackError(f"Procedure pack not found: {path}")

    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ProcedurePackError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(data, dict):
        raise ProcedurePackError(f"{path}: top level must be a mapping, got {type(data).__name__}")

    missing = [f for f in REQUIRED_TOP_FIELDS if f not in data]
    if missing:
        raise ProcedurePackError(f"{path}: missing required field(s): {missing}")

    steps = data["steps"]
    if not isinstance(steps, list) or not steps:
        raise ProcedurePackError(f"{path}: 'steps' must be a non-empty list")

    seen_ids = set()
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            raise ProcedurePackError(f"{path}: steps[{i}] must be a mapping, got {type(s).__name__}")
        missing_step = [f for f in REQUIRED_STEP_FIELDS if f not in s]
        if missing_step:
            raise ProcedurePackError(f"{path}: steps[{i}] missing required field(s): {missing_step}")
        if not isinstance(s["required_objects"], list):
            raise ProcedurePackError(f"{path}: steps[{i}].required_objects must be a list")
        if not isinstance(s["id"], int) or s["id"] < 1:
            raise ProcedurePackError(f"{path}: steps[{i}].id must be a positive integer")
        if s["id"] in seen_ids:
            raise ProcedurePackError(f"{path}: duplicate step id {s['id']}")
        seen_ids.add(s["id"])

    # pipeline/state_machine.py indexes steps 0..N-1 and assumes contiguous
    # sequential ids — a gap or out-of-range id would silently desync
    # ExperimentStateMachine.current_step_idx from the visible step list.
    expected_ids = set(range(1, len(steps) + 1))
    if seen_ids != expected_ids:
        raise ProcedurePackError(
            f"{path}: step ids must be exactly 1..{len(steps)} (contiguous, sequential — "
            f"pipeline/state_machine.py assumes this), got {sorted(seen_ids)}")

    return ProcedurePack(
        pack_version=data["pack_version"],
        experiment_id=data["experiment_id"],
        name=data["name"],
        steps=steps,
        camera_hints=data.get("camera_hints", {}),
        fsm=data.get("fsm", {}),
        alerts=data.get("alerts", {"voice": True}),
        source_path=str(p),
    )
