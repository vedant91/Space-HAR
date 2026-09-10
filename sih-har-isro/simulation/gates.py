"""Pass/fail gates for the end-to-end loop."""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.experiment_config import (
    E2E_LSTM_MIN_VAL_ACC, E2E_LSTM_MIN_TEST_ACC, E2E_HSV_MIN_RECALL,
    E2E_MAX_MEAN_LATENCY_MS, E2E_MAX_P95_LATENCY_MS, E2E_MIN_ORACLE_STEP_ACC,
    E2E_MIN_POSENET_PCK,
)

# Numeric thresholds come from config.experiment_config's E2E_* constants —
# this used to hardcode its own second copy of the same numbers, with
# nothing keeping them in sync (the E2E_* constants were defined but never
# actually read anywhere). The boolean FSM gates have no config equivalent
# (they're structural pass/fail expectations, not tunable thresholds).
DEFAULT_GATES = {
    "posenet_pck": E2E_MIN_POSENET_PCK,
    "lstm_val_acc": E2E_LSTM_MIN_VAL_ACC,
    "lstm_test_acc": E2E_LSTM_MIN_TEST_ACC,
    "hsv_red_recall": E2E_HSV_MIN_RECALL,
    "hsv_yellow_recall": E2E_HSV_MIN_RECALL,
    "mean_latency_ms": E2E_MAX_MEAN_LATENCY_MS,
    "p95_latency_ms": E2E_MAX_P95_LATENCY_MS,
    "oracle_step_acc": E2E_MIN_ORACLE_STEP_ACC,
    "sequence_complete": True,
    "skip_detected": True,
    "recovery_hold_at_expected_step": True,
    "recovery_confirmed": True,
    "sequence_complete_after_correction": True,
}

# Higher-is-better except latency keys
_LOWER_IS_BETTER = {"mean_latency_ms", "p95_latency_ms"}


def evaluate_gates(metrics: Dict, gates: Dict = None) -> Tuple[bool, List[Dict]]:
    """Return (all_passed, list of per-gate results)."""
    gates = gates or DEFAULT_GATES
    results = []
    all_ok = True
    for key, threshold in gates.items():
        value = metrics.get(key)
        if value is None:
            results.append({
                "gate": key, "threshold": threshold, "value": None,
                "passed": False, "reason": "metric missing",
            })
            all_ok = False
            continue
        if key in _LOWER_IS_BETTER:
            passed = float(value) <= float(threshold)
        elif isinstance(threshold, bool):
            passed = bool(value) is True if threshold is True else bool(value) == threshold
        else:
            passed = float(value) >= float(threshold)
        results.append({
            "gate": key, "threshold": threshold, "value": value, "passed": passed,
        })
        if not passed:
            all_ok = False
    return all_ok, results
