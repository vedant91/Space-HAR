"""Pass/fail gates for the end-to-end loop."""

from typing import Dict, List, Tuple

DEFAULT_GATES = {
    "lstm_val_acc": 0.90,
    "lstm_test_acc": 0.88,
    "hsv_red_recall": 0.85,
    "hsv_yellow_recall": 0.85,
    "mean_latency_ms": 80.0,
    "p95_latency_ms": 130.0,
    "oracle_step_acc": 0.85,
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
