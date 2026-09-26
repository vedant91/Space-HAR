"""tools/retrain_and_evaluate.sh: regression #4 (no dataset arg -> exit 2
with a clear message; --help -> exit 0) and an integration check of
regression #5 (a holdout tag matching zero takes fails the stage-0 gate,
so the script exits non-zero before doing any real work)."""
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "retrain_and_evaluate.sh"

# `bash` on PATH resolves inconsistently under a plain subprocess.run (it can
# land on a WSL launcher that wants /mnt/d/... paths instead of Git Bash's
# /d/...). Pin the actual Git Bash binary so path translation is predictable.
_GIT_BASH_CANDIDATES = (
    r"C:\Program Files\Git\usr\bin\bash.exe",
    r"C:\Program Files\Git\bin\bash.exe",
)
BASH = next((p for p in _GIT_BASH_CANDIDATES if Path(p).exists()),
            shutil.which("bash") or "bash")


def _posix(p) -> str:
    """Git Bash's bash.exe wants forward slashes even for absolute Windows
    paths passed as bare argv (no shell in between to translate them)."""
    return str(p).replace("\\", "/")


def _run(args, env=None):
    return subprocess.run([BASH, _posix(SCRIPT), *args], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=60)


def test_no_dataset_arg_exits_nonzero_with_usage():
    result = _run([])
    assert result.returncode == 2
    assert "dataset_dir is required" in result.stderr
    assert "Usage:" in result.stderr


def test_help_exits_zero():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "Usage:" in result.stdout


def test_dry_run_gate_fails_on_holdout_tag_matching_zero_takes(make_take, tmp_path):
    """Regression #5, exercised through the real script: DRY_RUN=1 stops
    after the stage-0 gate (tools/check_dataset.py), which must fail loudly
    -- non-zero exit -- when a holdout tag matches no take."""
    root = tmp_path / "ds"
    make_take(root, index=0, tag="nominal", n_frames=12, sequence=(1, 2))

    env = dict(os.environ)
    env["DRY_RUN"] = "1"
    env["TEMP"] = str(REPO.parent / ".tmp")
    env["TMP"] = str(REPO.parent / ".tmp")
    result = subprocess.run(
        [BASH, _posix(SCRIPT), _posix(root), "orient_180"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=60,
    )
    assert result.returncode != 0
    assert "matches no take" in result.stdout


def test_dry_run_gate_passes_and_stops_before_training(make_take, tmp_path):
    root = tmp_path / "ds"
    make_take(root, index=0, tag="orient_180", n_frames=12, sequence=(1, 2))
    make_take(root, index=1, tag="nominal", n_frames=12, sequence=(1, 2))

    env = dict(os.environ)
    env["DRY_RUN"] = "1"
    env["TEMP"] = str(REPO.parent / ".tmp")
    env["TMP"] = str(REPO.parent / ".tmp")
    result = subprocess.run(
        [BASH, _posix(SCRIPT), _posix(root), "orient_180"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=60,
    )
    assert result.returncode == 0
    assert "gate passed, stopping here" in result.stdout
    # Echoes the resolved dataset, take count/tags before doing any work
    # (Decision #3): train tags line names the non-holdout tag.
    assert "train   : nominal" in result.stdout
