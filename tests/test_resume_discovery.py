"""Pin resume discovery: the sibling lookup must match the start epoch exactly, never by substring."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.scripts.training.train_cliphba import (  # noqa: E402
    _find_previous_run_with_same_start,
    _resume_from_previous,
)


def _make_run(parent: Path, name: str, checkpoint_epochs=()):
    run = parent / name
    (run / "dora_params").mkdir(parents=True)
    (run / "random_states").mkdir()
    for epoch in checkpoint_epochs:
        (run / "dora_params" / f"epoch{epoch}_dora_params.pth").write_bytes(b"")
        (run / "random_states" / f"epoch{epoch}_random_states.pth").write_bytes(b"")
    (run / "training_res.csv").write_text("epoch,train_loss,val_loss\n")
    return run


def test_onset_prefix_collision_is_not_a_match(tmp_path):
    # "epoch1" is a substring of both; the old substring match returned one of them
    _make_run(tmp_path, "epoch14_length5")
    _make_run(tmp_path, "epoch19_length5")
    assert _find_previous_run_with_same_start(tmp_path / "epoch1_length10", 1, 10) == (None, None)


def test_picks_longest_shorter_sibling_with_exact_onset(tmp_path):
    for name in ("epoch1_length1", "epoch1_length5", "epoch1_length10", "epoch1_length20",
                 "epoch14_length8", "epoch1_length7_backup"):
        _make_run(tmp_path, name)
    (tmp_path / "epoch1_length8").write_text("a file, not a run dir")
    best_dir, best_length = _find_previous_run_with_same_start(tmp_path / "epoch1_length10", 1, 10)
    assert (best_dir.name, best_length) == ("epoch1_length5", 5)


def test_resume_from_previous_uses_exact_onset_sibling(tmp_path):
    # Under the old code the longer-length decoy at a colliding onset won the tie-break,
    # its checkpoint at epoch 9 was missing, and the resume was silently skipped.
    _make_run(tmp_path, "epoch2_length5", checkpoint_epochs=(2, 3, 4, 5, 6))
    _make_run(tmp_path, "epoch29_length8", checkpoint_epochs=range(29, 37))
    run_root = tmp_path / "epoch2_length10"
    config = _resume_from_previous({"perturb_epoch": 2, "perturb_length": 10}, run_root)
    assert config["resume_from_epoch"] == 6
    assert Path(config["resume_checkpoint_path"]) == tmp_path / "epoch2_length5"
    assert (run_root / "training_res.csv").exists()


def test_resume_falls_back_when_sibling_lacks_checkpoint(tmp_path, capsys):
    _make_run(tmp_path, "epoch2_length5", checkpoint_epochs=(2, 3))
    config = _resume_from_previous({"perturb_epoch": 2, "perturb_length": 10}, tmp_path / "epoch2_length10")
    assert "resume_checkpoint_path" not in config
    assert "falling back to the baseline checkpoint" in capsys.readouterr().out
