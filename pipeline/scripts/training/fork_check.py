"""Check that forking from a baseline checkpoint is exact: an unperturbed fork must reproduce
the baseline's training loss bit-for-bit.

For one family and onset, the fork resumes from the same baseline checkpoint the perturbed
runs use (`--resume_ckpt_path` of `<prefix>_image_noise_<onset>` in scratch_runs.yaml) and
runs the baseline's exact arguments from scratch_runs.yaml, with no perturbation, for
`--epochs` whole epochs (default 2: the onset epoch E and E+1). The only added arguments are
bookkeeping that doesn't touch the training math:
  --experiment_name / --save_dir   write to <runs_root>/fork_checks/<name>, not the baseline
  --stop_epoch E+epochs-1          stop at the end of that epoch
  --log_every_n_steps 1            log every step, so the fork covers every baseline log point

`compare` then reads the per-step training loss from both runs' TensorBoard event files and
checks it for exact (float32) equality at every step both runs logged. The baseline logged
every 50 steps (the paper setting), so the comparison runs every 50 steps across both epochs. A
resume continues the baseline's step axis (RestoreEpochCallback restores global_step); a fork
whose axis starts near 0 instead (code before that fix) is offset by the checkpoint's
global_step. Either way the logged learning rate, a deterministic function of the step,
confirms the alignment. Each compared step records the absolute and relative difference
(|baseline - fork| and that over |baseline|); a summary reports the largest of each, how many
times each is larger than at the previous compared step, and the periods where it rises at
least twice in a row.

    python pipeline/scripts/training/fork_check.py run     --family resnet50 --onset early
    python pipeline/scripts/training/fork_check.py compare --family resnet50 --onset early

On a Slurm cluster, pipeline/slurm/submit_fork_checks.sh submits one job per family. Dataset paths are
read like every other path: `TP_<KEY>` env var or pipeline/paths.yaml, with keys
runs_root, imagenet_train_dir, imagenet_val_dir, yfcc15m_parquet_dir, val_dataset_root.
"""
import argparse
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from temporal_perturbations import paths  # noqa: E402

TRAIN_SCRIPT = REPO_ROOT / "pipeline" / "scripts" / "training" / "train_from_scratch.py"
SCRATCH_RUNS = REPO_ROOT / "pipeline" / "configs" / "scratch_runs.yaml"

# family -> (run-name prefix in scratch_runs.yaml, per-step loss tag in TensorBoard)
FAMILIES = {
    "resnet50": ("resnet50_imagenet", "train/ce_loss_step"),
    "vitb16": ("vitb16_imagenet", "train/ce_loss_step"),
    "convnext_b": ("convnext_b_imagenet", "train/ce_loss_step"),
    "clip_vitb32": ("clip_vitb32", "train/clip_loss_step"),
}
ONSETS = ("early", "mid", "late")


class _PathsLookup(dict):
    """format_map source that resolves scratch_runs.yaml placeholders through paths.py."""

    def __missing__(self, key):
        return str(paths.path(key))


def _load_runs():
    with open(SCRATCH_RUNS) as f:
        return yaml.safe_load(f)["runs"]


def _arg_value(tokens, flag):
    return tokens[tokens.index(flag) + 1]


def _set_arg(tokens, flag, value):
    tokens = list(tokens)
    if flag in tokens:
        tokens[tokens.index(flag) + 1] = value
    else:
        tokens += [flag, value]
    return tokens


def fork_check_name(family, onset):
    return f"fork_check_{family}_{onset}"


def fork_check_root():
    return paths.path("runs_root") / "fork_checks"


def build_fork(family, onset, epochs):
    """Return (gpus, train_from_scratch.py args, resume checkpoint, baseline run name)."""
    prefix, _ = FAMILIES[family]
    runs = _load_runs()
    baseline_name = f"{prefix}_baseline"
    baseline = runs[baseline_name]
    # The perturbed runs at this onset all fork from the same checkpoint; take it from one.
    perturbed = shlex.split(runs[f"{prefix}_image_noise_{onset}"]["args"].format_map(_PathsLookup()))
    resume_ckpt = _arg_value(perturbed, "--resume_ckpt_path")
    onset_epoch = int(_arg_value(perturbed, "--perturb_epoch"))

    tokens = shlex.split(baseline["args"].format_map(_PathsLookup()))
    for flag in ("--resume_ckpt_path", "--perturb_type", "--stop_epoch", "--log_every_n_steps"):
        assert flag not in tokens, f"baseline args unexpectedly contain {flag}"
    tokens = _set_arg(tokens, "--experiment_name", fork_check_name(family, onset))
    tokens = _set_arg(tokens, "--save_dir", str(fork_check_root()))
    tokens += ["--resume_ckpt_path", resume_ckpt,
               "--stop_epoch", str(onset_epoch + epochs - 1),
               "--log_every_n_steps", "1"]
    return baseline["gpus"], tokens, resume_ckpt, baseline_name


def _visible_gpus():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([d for d in cvd.split(",") if d.strip()])
    return None


def cmd_run(args):
    gpus, tokens, resume_ckpt, baseline_name = build_fork(args.family, args.onset, args.epochs)
    out_dir = fork_check_root() / fork_check_name(args.family, args.onset)
    if not Path(resume_ckpt).is_file():
        sys.exit(f"Baseline checkpoint not found: {resume_ckpt}")
    if out_dir.exists():
        sys.exit(f"{out_dir} already exists; remove it first so the comparison reads only this fork.")
    visible = _visible_gpus()
    if visible is not None and visible != gpus:
        # The GPU count sets DDP sharding and grad accumulation, so it must match the baseline.
        sys.exit(f"{baseline_name} was trained on {gpus} GPUs but {visible} are visible.")

    cmd = ["torchrun", "--standalone", "--nproc_per_node", str(gpus), str(TRAIN_SCRIPT), *tokens]
    print("$", shlex.join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    if rc == 0 and not args.keep_checkpoints:
        # The comparison reads only the logs; the fork's checkpoints are large and
        # would otherwise sit next to the real runs looking like one.
        ckpt_dir = out_dir / "checkpoints"
        if ckpt_dir.is_dir() and fork_check_root() in ckpt_dir.parents:
            shutil.rmtree(ckpt_dir)
    return rc


# ---------------------------------------------------------------------------- comparison

def read_scalars(run_dir, loss_tag):
    """{tag: {step: [values in logging order]}} for the loss tag and every lr-* tag.

    Each event file is read separately and without orphan purging, so a step logged by more
    than one file (a resumed run re-logging steps) is kept and flagged, not silently dropped.
    Returns (scalars, steps_in_multiple_files).
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    scalars, owner, multi = {}, {}, set()
    for f in sorted(Path(run_dir).rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(f), size_guidance={"scalars": 0}, purge_orphaned_data=False)
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if tag != loss_tag and not tag.startswith("lr-"):
                continue
            for ev in acc.Scalars(tag):
                if owner.setdefault((tag, ev.step), f) != f:
                    multi.add((tag, ev.step))
                scalars.setdefault(tag, {}).setdefault(ev.step, []).append(ev.value)
    return scalars, multi


def _f32_hex(x):
    return struct.pack(">f", x).hex()


def _choose_offset(base_lr, fork_lr, resume_step, fork_steps):
    """Offset d with baseline_step = fork_step + d, confirmed by exact lr agreement.

    Tries the expected offset first (resume_step if the fork's counter restarted at 0,
    else 0), then any offset under which every overlapping lr value agrees.
    Returns (offset, how) where how is 'expected', 'lr-search' or 'unverified'.
    """
    expected = resume_step if fork_steps and min(fork_steps) < resume_step else 0
    tags = sorted(set(base_lr) & set(fork_lr))

    def agrees(d):
        n = 0
        for tag in tags:
            for s, vals in fork_lr[tag].items():
                if s + d in base_lr[tag]:
                    if base_lr[tag][s + d] != vals:
                        return False
                    n += 1
        return n > 0

    if not tags:
        return expected, "unverified"
    if agrees(expected):
        return expected, "expected"
    tag = tags[0]
    candidates = {b - s for s, v in fork_lr[tag].items() for b, bv in base_lr[tag].items() if bv == v}
    good = sorted(d for d in candidates if agrees(d))
    if len(good) == 1:
        return good[0], "lr-search"
    return expected, "unverified"


def _diffs(bv, fv):
    """(abs_diff, rel_diff): largest |baseline - fork| and |baseline - fork| / |baseline| over
    the values logged at one step (CLIP logs one per accumulated micro-batch). None if the two
    runs logged a different number of values there."""
    if len(bv) != len(fv) or not bv:
        return None, None
    abs_d = [abs(x - y) for x, y in zip(bv, fv)]
    rel_d = [d / abs(x) if x else (0.0 if d == 0 else float("inf")) for d, x in zip(abs_d, bv)]
    return max(abs_d), max(rel_d)


def compare_losses(base_loss, fork_loss, offset, steps_in_multiple_files=()):
    """Compare the loss at every step both runs logged. Pure function, testable without TB."""
    rows = []
    for s in sorted(fork_loss):
        b = s + offset
        if b not in base_loss:
            continue
        bv, fv = base_loss[b], fork_loss[s]
        abs_diff, rel_diff = _diffs(bv, fv)
        rows.append({
            "baseline_step": b, "fork_step": s,
            "baseline": bv, "fork": fv,
            "baseline_hex": [_f32_hex(v) for v in bv], "fork_hex": [_f32_hex(v) for v in fv],
            "exact": bv == fv,
            "abs_diff": abs_diff, "rel_diff": rel_diff,
            "ambiguous_baseline": b in steps_in_multiple_files,
        })
    if not rows:
        verdict = "NO_OVERLAP"
    elif all(r["exact"] for r in rows):
        verdict = "EXACT"
    else:
        verdict = "DIFFERS"
    return verdict, rows


def growth_periods(rows, key):
    """Count where the `key` gap ('abs_diff' or 'rel_diff') is larger than at the previous
    compared step, and collect the periods where it rises at least twice in a row
    (gap[t] < gap[t+1] < gap[t+2] ...). A step with no gap value breaks a period."""
    increases, transitions, periods, run = 0, 0, [], []
    prev = None
    for r in rows:
        d = r[key]
        if d is None or prev is None or prev[key] is None:
            run = [r] if d is not None else []
            prev = r
            continue
        transitions += 1
        if d > prev[key]:
            increases += 1
            run = (run or [prev]) + [r]
        else:
            if len(run) >= 3:
                periods.append(run)
            run = [r]
        prev = r
    if len(run) >= 3:
        periods.append(run)
    return {
        "increases": increases,
        "transitions": transitions,
        "periods": [{"start_baseline_step": p[0]["baseline_step"], "end_baseline_step": p[-1]["baseline_step"],
                     "start_fork_step": p[0]["fork_step"], "end_fork_step": p[-1]["fork_step"],
                     "consecutive_increases": len(p) - 1,
                     "start": p[0][key], "end": p[-1][key]} for p in periods],
    }


def summarize_differences(rows):
    """First, largest and last differences, and how often the gap grows from step to step."""
    diffs = [r for r in rows if not r["exact"]]
    if not diffs:
        return {"n_differing": 0}
    sized = [r for r in diffs if r["abs_diff"] is not None]
    summary = {
        "n_differing": len(diffs),
        "n_unmatched_value_counts": len(diffs) - len(sized),
        "first": {k: diffs[0][k] for k in ("baseline_step", "fork_step", "abs_diff", "rel_diff")},
        "last_compared": {k: rows[-1][k] for k in ("baseline_step", "fork_step", "abs_diff", "rel_diff")},
        "abs_diff_growth": growth_periods(rows, "abs_diff"),
        "rel_diff_growth": growth_periods(rows, "rel_diff"),
    }
    if sized:
        worst_abs = max(sized, key=lambda r: r["abs_diff"])
        worst_rel = max(sized, key=lambda r: r["rel_diff"])
        summary["max_abs_diff"] = {"value": worst_abs["abs_diff"], "baseline_step": worst_abs["baseline_step"],
                                   "fork_step": worst_abs["fork_step"]}
        summary["max_rel_diff"] = {"value": worst_rel["rel_diff"], "baseline_step": worst_rel["baseline_step"],
                                   "fork_step": worst_rel["fork_step"]}
    return summary


def _fmt(x):
    return "" if x is None else f"{x:.3g}"


def cmd_compare(args):
    import torch

    _, loss_tag = FAMILIES[args.family]
    _, _, resume_ckpt, baseline_name = build_fork(args.family, args.onset, args.epochs)
    baseline_dir = paths.path("runs_root") / baseline_name
    fork_dir = fork_check_root() / fork_check_name(args.family, args.onset)

    meta = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
    resume_step, resume_epoch = int(meta["global_step"]), int(meta["epoch"])
    del meta

    base, base_multi = read_scalars(baseline_dir, loss_tag)
    fork, _ = read_scalars(fork_dir, loss_tag)
    if loss_tag not in fork:
        sys.exit(f"No '{loss_tag}' scalars under {fork_dir}; did the fork run?")
    if loss_tag not in base:
        sys.exit(f"No '{loss_tag}' scalars under {baseline_dir}.")

    base_lr = {t: v for t, v in base.items() if t.startswith("lr-")}
    fork_lr = {t: v for t, v in fork.items() if t.startswith("lr-")}
    offset, how = _choose_offset(base_lr, fork_lr, resume_step, list(fork[loss_tag]))
    multi = {s for (t, s) in base_multi if t == loss_tag}
    verdict, rows = compare_losses(base[loss_tag], fork[loss_tag], offset, multi)

    print(f"\nFork check: {args.family}, onset {args.onset} "
          f"(resumed {resume_ckpt}: epoch {resume_epoch}, global_step {resume_step})")
    fork_steps = sorted(fork[loss_tag])
    in_range = [b for b in base[loss_tag] if fork_steps[0] + offset <= b <= fork_steps[-1] + offset]
    # Steps into the fork, from the fork's own axis (absolute once global_step is restored).
    fork_base = resume_step if fork_steps[0] >= resume_step else 0
    print(f"Fork logged steps {fork_steps[0]}-{fork_steps[-1]} of '{loss_tag}'; step offset {offset} ({how})")
    print(f"{'into fork':>10} {'base step':>10} {'baseline':>14} {'fork':>14} {'abs diff':>10} {'rel diff':>10}  exact")
    for r in rows:
        flag = "yes" if r["exact"] else "NO"
        if r["ambiguous_baseline"]:
            flag += " (baseline logged this step in more than one file)"
        print(f"{r['fork_step'] - fork_base:>10} {r['baseline_step']:>10} "
              f"{', '.join(f'{v:.9g}' for v in r['baseline']):>14} "
              f"{', '.join(f'{v:.9g}' for v in r['fork']):>14} "
              f"{_fmt(r['abs_diff']):>10} {_fmt(r['rel_diff']):>10}  {flag}")
    first_diff = next((r for r in rows if not r["exact"]), None)
    summary = summarize_differences(rows)
    print(f"\nVERDICT: {verdict} ({sum(r['exact'] for r in rows)}/{len(rows)} compared steps bit-identical; "
          f"{len(rows)} of {len(in_range)} baseline log points in the fork's range compared)")
    if first_diff is not None:
        into = lambda d: d["fork_step"] - fork_base
        print(f"First difference at baseline step {first_diff['baseline_step']} "
              f"({into(first_diff)} steps into the fork): abs {_fmt(first_diff['abs_diff'])}, "
              f"rel {_fmt(first_diff['rel_diff'])}.")
        if "max_abs_diff" in summary:
            ma, mr = summary["max_abs_diff"], summary["max_rel_diff"]
            print(f"Largest abs diff {_fmt(ma['value'])} at baseline step {ma['baseline_step']} "
                  f"({into(ma)} steps into the fork); largest rel diff {_fmt(mr['value'])} at baseline "
                  f"step {mr['baseline_step']} ({into(mr)} steps into the fork).")
        last = summary["last_compared"]
        print(f"At the last compared step ({into(last)} steps into the fork): abs {_fmt(last['abs_diff'])}, "
              f"rel {_fmt(last['rel_diff'])}.")
        for key, label in (("abs_diff", "Abs diff"), ("rel_diff", "Rel diff")):
            g = summary[f"{key}_growth"]
            print(f"{label} larger than at the previous compared step: {g['increases']} of "
                  f"{g['transitions']} times; {len(g['periods'])} periods of 2+ increases in a row"
                  + (":" if g["periods"] else "."))
            for p in g["periods"]:
                print(f"    {p['start_fork_step'] - fork_base}-{p['end_fork_step'] - fork_base} steps into the fork "
                      f"(baseline steps {p['start_baseline_step']}-{p['end_baseline_step']}): "
                      f"{p['consecutive_increases']} increases, {_fmt(p['start'])} -> {_fmt(p['end'])}")
        if summary["n_unmatched_value_counts"]:
            print(f"{summary['n_unmatched_value_counts']} differing steps logged a different number of "
                  f"values in the two runs; they have no diff.")
    if how == "unverified":
        print("WARNING: the step alignment could not be confirmed from the learning rate.")

    report = {"family": args.family, "onset": args.onset, "resume_ckpt": resume_ckpt,
              "resume_epoch": resume_epoch, "resume_global_step": resume_step,
              "loss_tag": loss_tag, "offset": offset, "alignment": how,
              "verdict": verdict,
              "first_differing_baseline_step": first_diff and first_diff["baseline_step"],
              "baseline_points_in_range": len(in_range), "summary": summary, "rows": rows}
    report_path = fork_dir / "fork_check_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report: {report_path}")
    return {"EXACT": 0, "DIFFERS": 1}.get(verdict, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "compare", "print"):
        p = sub.add_parser(name)
        p.add_argument("--family", required=True, choices=sorted(FAMILIES))
        p.add_argument("--onset", default="early", choices=ONSETS)
        p.add_argument("--epochs", type=int, default=2,
                       help="Whole epochs to run the fork for, starting at the onset epoch.")
        if name == "run":
            p.add_argument("--keep_checkpoints", action="store_true",
                           help="Keep the fork's (partial-epoch) checkpoints.")
    args = parser.parse_args()

    if args.command == "print":
        gpus, tokens, _, _ = build_fork(args.family, args.onset, args.epochs)
        print(shlex.join(["torchrun", "--standalone", "--nproc_per_node", str(gpus), str(TRAIN_SCRIPT), *tokens]))
        return 0
    return cmd_run(args) if args.command == "run" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
