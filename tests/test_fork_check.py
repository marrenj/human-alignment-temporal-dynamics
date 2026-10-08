"""Pin fork_check.py: the fork runs the baseline's exact args, and the loss comparison is exact."""
import shlex
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.scripts.training import fork_check  # noqa: E402

DATA_KEYS = ("runs_root", "imagenet_train_dir", "imagenet_val_dir", "yfcc15m_parquet_dir", "val_dataset_root")


def _point_paths_at(tmp_path, monkeypatch):
    for key in DATA_KEYS:
        monkeypatch.setenv(f"TP_{key.upper()}", str(tmp_path / key))


def test_fork_args_are_baseline_args_plus_bookkeeping(tmp_path, monkeypatch):
    _point_paths_at(tmp_path, monkeypatch)
    runs = yaml.safe_load(open(fork_check.SCRATCH_RUNS))["runs"]
    for family, (prefix, _) in fork_check.FAMILIES.items():
        for onset in fork_check.ONSETS:
            gpus, tokens, resume, baseline_name = fork_check.build_fork(family, onset, 2)
            baseline = shlex.split(runs[baseline_name]["args"].format_map(fork_check._PathsLookup()))
            perturbed = shlex.split(runs[f"{prefix}_label_shuffle_{onset}"]["args"].format_map(fork_check._PathsLookup()))

            assert gpus == runs[baseline_name]["gpus"] == runs[f"{prefix}_label_shuffle_{onset}"]["gpus"]
            assert resume == perturbed[perturbed.index("--resume_ckpt_path") + 1]
            assert "--perturb_type" not in tokens
            # Everything but the output location and the stop/logging flags is the baseline's.
            added = {"--experiment_name", "--save_dir", "--resume_ckpt_path", "--stop_epoch", "--log_every_n_steps"}
            strip = lambda t: [x for i, x in enumerate(t) if x not in added and (i == 0 or t[i - 1] not in added)]
            assert strip(tokens) == strip(baseline)
            onset_epoch = int(perturbed[perturbed.index("--perturb_epoch") + 1])
            assert tokens[tokens.index("--stop_epoch") + 1] == str(onset_epoch + 1)
            assert resume.endswith(f"epoch_{onset_epoch - 1:03d}.ckpt")
            assert tokens[tokens.index("--save_dir") + 1] == str(tmp_path / "runs_root" / "fork_checks")


def test_offset_comes_from_the_resume_step_and_is_confirmed_by_lr():
    resume = 1000
    lr = lambda s: 0.1 * (1 - s / 5000)
    # Baseline logged every 50 steps on the absolute axis; the fork logged every step from 0.
    base_lr = {"lr-SGD": {s: [lr(s)] for s in range(49, 2000, 50)}}
    fork_lr = {"lr-SGD": {s: [lr(s + resume)] for s in range(100)}}
    assert fork_check._choose_offset(base_lr, fork_lr, resume, list(range(100))) == (resume, "expected")


def test_offset_found_by_lr_when_baseline_axis_is_shifted():
    # Baseline was itself resumed earlier, so its axis is 300 steps behind the checkpoint's global_step.
    resume, shift = 1000, 300
    lr = lambda s: 0.1 * (1 - s / 5000)
    base_lr = {"lr-SGD": {s: [lr(s + shift)] for s in range(49, 2000, 50)}}
    fork_lr = {"lr-SGD": {s: [lr(s + resume)] for s in range(100)}}
    assert fork_check._choose_offset(base_lr, fork_lr, resume, list(range(100))) == (resume - shift, "lr-search")


def test_compare_is_exact_equality():
    base = {1049: [2.5], 1099: [2.25]}
    assert fork_check.compare_losses(base, {49: [2.5], 99: [2.25]}, 1000)[0] == "EXACT"
    verdict, rows = fork_check.compare_losses(base, {49: [2.5], 99: [2.2500002]}, 1000)
    assert verdict == "DIFFERS" and [r["exact"] for r in rows] == [True, False]
    assert fork_check.compare_losses(base, {0: [2.5]}, 1000)[0] == "NO_OVERLAP"


def test_rows_record_absolute_and_relative_differences():
    _, rows = fork_check.compare_losses({1049: [2.0], 1099: [4.0, 5.0]}, {49: [2.0], 99: [4.5, 5.0]}, 1000)
    assert (rows[0]["abs_diff"], rows[0]["rel_diff"]) == (0.0, 0.0)
    # CLIP-style multi-value step: the largest per-value gap
    assert (rows[1]["abs_diff"], rows[1]["rel_diff"]) == (0.5, 0.125)
    _, rows = fork_check.compare_losses({1049: [2.0, 3.0]}, {49: [2.0]}, 1000)
    assert (rows[0]["abs_diff"], rows[0]["rel_diff"]) == (None, None)


def _rows_with_gaps(gaps):
    base = {1000 + 50 * i: [5.0] for i in range(len(gaps))}
    fork = {50 * i: [5.0 + g] for i, g in enumerate(gaps)}
    return fork_check.compare_losses(base, fork, 1000)[1]


def test_summary_reports_largest_gap():
    gaps = [0.0] * 4 + [1e-7 * 2 ** i for i in range(16)]  # exact, then diverging
    summary = fork_check.summarize_differences(_rows_with_gaps(gaps))
    assert summary["n_differing"] == 16
    assert summary["first"]["baseline_step"] == 1000 + 50 * 4
    assert summary["max_abs_diff"]["baseline_step"] == 1000 + 50 * 19
    assert summary["max_rel_diff"]["baseline_step"] == 1000 + 50 * 19
    assert fork_check.summarize_differences(_rows_with_gaps([0.0] * 5)) == {"n_differing": 0}


def test_growth_counts_increases_and_periods_of_two_or_more():
    #        idx: 0    1    2    3    4    5    6    7    8
    gaps = [0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0]
    g = fork_check.growth_periods(_rows_with_gaps([x * 1e-6 for x in gaps]), "abs_diff")
    assert (g["increases"], g["transitions"]) == (6, 8)
    # 0<1<2<3 (3 increases) and 1<2<3 at the end; the single rise 1->2 at idx 4-5 is not a period
    assert [(p["start_fork_step"] // 50, p["end_fork_step"] // 50, p["consecutive_increases"]) for p in g["periods"]] \
        == [(0, 3, 3), (6, 8, 2)]
    assert fork_check.growth_periods(_rows_with_gaps([1e-6, 1e-6, 1e-6]), "rel_diff") \
        == {"increases": 0, "transitions": 2, "periods": []}


def test_growth_period_is_broken_by_a_step_without_a_gap():
    rows = _rows_with_gaps([1e-6, 2e-6, 3e-6, 4e-6])
    rows[2]["abs_diff"] = None  # e.g. the runs logged a different number of values there
    g = fork_check.growth_periods(rows, "abs_diff")
    assert (g["increases"], g["transitions"], g["periods"]) == (1, 1, [])
