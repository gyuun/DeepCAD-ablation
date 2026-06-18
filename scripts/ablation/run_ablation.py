import argparse
import importlib.util
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.ablation.collect_metrics import collect_run_metrics, parse_acc, refresh_summary
from scripts.ablation.utils import (
    ARCH_FLAGS,
    acc_stat_path,
    checkpoint_path,
    command_to_str,
    git_commit,
    git_dirty,
    iter_runs,
    load_manifest,
    load_meta,
    metrics_path,
    pc_stat_path,
    print_alignment_report,
    python_executable,
    reconstruction_count,
    reconstruction_dir,
    run_command,
    save_meta,
    sha256_file,
    test_split_count,
    update_meta_status,
    utc_now,
)


def _manifest_bool(manifest, key):
    return bool(manifest.get(key, False))


def _run_identity(run):
    exp = run["experiment"]
    return exp["id"], run["run_name"]


def _proj_args(manifest, run):
    exp_id, seed_name = _run_identity(run)
    return os.path.join(manifest["proj_dir"], exp_id), seed_name


def _train_command(manifest, run, force_train, resume_train=False):
    proj_dir, exp_name = _proj_args(manifest, run)
    cmd = [
        python_executable(),
        "train.py",
        "--proj_dir", proj_dir,
        "--exp_name", exp_name,
        "--data_root", manifest["data_root"],
        "--gpu_ids", str(manifest.get("gpu_ids", "0")),
        "--batch_size", str(manifest.get("batch_size", 512)),
        "--num_workers", str(manifest.get("num_workers", 8)),
        "--nr_epochs", str(manifest.get("nr_epochs", 1000)),
        "--seed", str(run["seed"]),
    ]
    if resume_train:
        cmd.extend(["--cont", "--ckpt", "latest"])
    elif force_train:
        cmd.append("--overwrite")
    for key in ARCH_FLAGS:
        if key in run["config"]:
            cmd.extend(["--" + key, str(run["config"][key])])
    for key in ["save_frequency", "val_frequency", "vis_frequency", "lr", "grad_clip", "warmup_step"]:
        if key in manifest:
            cmd.extend(["--" + key, str(manifest[key])])
    if _manifest_bool(manifest, "augment"):
        cmd.append("--augment")
    return cmd


def _test_command(manifest, run):
    proj_dir, exp_name = _proj_args(manifest, run)
    return [
        python_executable(),
        "test.py",
        "--proj_dir", proj_dir,
        "--exp_name", exp_name,
        "--data_root", manifest["data_root"],
        "--gpu_ids", str(manifest.get("gpu_ids", "0")),
        "--batch_size", str(manifest.get("batch_size", 512)),
        "--num_workers", str(manifest.get("num_workers", 8)),
        "--seed", str(run["seed"]),
        "--ckpt", str(manifest.get("ckpt", "latest")),
        "--mode", "rec",
    ] + _arch_args(run["config"])


def _arch_args(config):
    args = []
    for key in ARCH_FLAGS:
        if key in config:
            args.extend(["--" + key, str(config[key])])
    return args


def _acc_command(manifest, run):
    return [
        python_executable(),
        "evaluation/evaluate_ae_acc.py",
        "--src", reconstruction_dir(run["run_dir"], manifest.get("ckpt", "latest")),
    ]


def _cd_command(manifest, run):
    cmd = [
        python_executable(),
        "evaluation/evaluate_ae_cd.py",
        "--src", reconstruction_dir(run["run_dir"], manifest.get("ckpt", "latest")),
    ]
    if manifest.get("cd_parallel", True):
        cmd.append("--parallel")
    if "n_points" in manifest:
        cmd.extend(["--n_points", str(manifest["n_points"])])
    if "cd_num" in manifest:
        cmd.extend(["--num", str(manifest["cd_num"])])
    return cmd


def _new_meta(manifest, run):
    exp = run["experiment"]
    split_path = os.path.join(manifest["data_root"], "train_val_test_split.json")
    train_cmd = _train_command(manifest, run, force_train=False)
    return {
        "experiment_id": exp["id"],
        "run_name": run["run_name"],
        "run_dir": run["run_dir"],
        "component": exp["component"],
        "variant": exp["variant"],
        "seed": run["seed"],
        "config": run["config"],
        "command": command_to_str(train_cmd),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "data_split_sha256": sha256_file(split_path),
        "config_sha256": None,
        "started_at": utc_now(),
        "finished_at": None,
        "status": "planned",
    }


def _meta_path(run):
    return os.path.join(run["run_dir"], "run_meta.json")


def _ensure_meta(manifest, run):
    path = _meta_path(run)
    meta = load_meta(path)
    if meta is None:
        os.makedirs(run["run_dir"], exist_ok=True)
        meta = _new_meta(manifest, run)
        save_meta(path, meta)
    return meta


def _assert_config_matches_manifest(manifest, run):
    config_path = os.path.join(run["run_dir"], "config.txt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, "r") as fp:
        config = json.load(fp)
    expected = dict(run["config"])
    expected["seed"] = run["seed"]
    mismatches = {}
    for key, value in expected.items():
        if config.get(key) != value:
            mismatches[key] = {"expected": value, "actual": config.get(key)}
    if mismatches:
        raise RuntimeError("config.txt does not match manifest: {}".format(mismatches))


def _force(force_list, stage):
    return "all" in force_list or stage in force_list


def _mark_failed(run, reason):
    update_meta_status(_meta_path(run), "failed", failure_reason=reason, finished_at=utc_now())


def run_train(manifest, run, force_list, dry_run=False, resume_train=False):
    ckpt = manifest.get("ckpt", "latest")
    latest_ckpt_exists = os.path.exists(checkpoint_path(run["run_dir"], "latest"))
    should_resume_train = resume_train and latest_ckpt_exists
    if dry_run:
        cmd = _train_command(
            manifest,
            run,
            force_train=_force(force_list, "train"),
            resume_train=should_resume_train,
        )
        run_command(cmd, cwd=REPO_ROOT, dry_run=True)
        return

    if os.path.exists(checkpoint_path(run["run_dir"], ckpt)) and not _force(force_list, "train") and not should_resume_train:
        _ensure_meta(manifest, run)
        update_meta_status(_meta_path(run), "trained")
        return

    if os.path.exists(run["run_dir"]) and not _force(force_list, "train") and not should_resume_train:
        if not os.path.exists(checkpoint_path(run["run_dir"], ckpt)):
            _ensure_meta(manifest, run)
            _mark_failed(run, "existing_incomplete_run_requires_force_train")
            raise RuntimeError("{} exists without checkpoint; rerun with --force train".format(run["run_dir"]))

    cmd = _train_command(
        manifest,
        run,
        force_train=_force(force_list, "train"),
        resume_train=should_resume_train,
    )
    code = run_command(cmd, cwd=REPO_ROOT, dry_run=dry_run)
    if code != 0:
        _ensure_meta(manifest, run)
        _mark_failed(run, "train_command_failed")
        raise RuntimeError("train failed: {}".format(command_to_str(cmd)))

    _ensure_meta(manifest, run)
    _assert_config_matches_manifest(manifest, run)
    config_hash = sha256_file(os.path.join(run["run_dir"], "config.txt"))
    update_meta_status(_meta_path(run), "trained", config_sha256=config_hash)
    if not os.path.exists(checkpoint_path(run["run_dir"], ckpt)):
        _mark_failed(run, "checkpoint_missing")
        raise RuntimeError("missing checkpoint {}".format(checkpoint_path(run["run_dir"], ckpt)))


def run_reconstruct(manifest, run, force_list, dry_run=False):
    ckpt = manifest.get("ckpt", "latest")
    expected_count = test_split_count(manifest["data_root"])
    current_count = reconstruction_count(run["run_dir"], ckpt)
    if expected_count is not None and current_count == expected_count and not _force(force_list, "reconstruct"):
        if not dry_run:
            update_meta_status(_meta_path(run), "reconstructed")
        return

    cmd = _test_command(manifest, run)
    code = run_command(cmd, cwd=REPO_ROOT, log_path=os.path.join(run["run_dir"], "run_ablation.log"), dry_run=dry_run)
    if code != 0:
        _mark_failed(run, "reconstruct_command_failed")
        raise RuntimeError("reconstruct failed: {}".format(command_to_str(cmd)))
    if dry_run:
        return

    current_count = reconstruction_count(run["run_dir"], ckpt)
    if expected_count is not None and current_count != expected_count:
        _mark_failed(run, "reconstruction_count_mismatch")
        raise RuntimeError("expected {} h5 files, got {}".format(expected_count, current_count))
    update_meta_status(_meta_path(run), "reconstructed")


def run_acc(manifest, run, force_list, dry_run=False):
    ckpt = manifest.get("ckpt", "latest")
    path = acc_stat_path(run["run_dir"], ckpt)
    if os.path.exists(path) and not _force(force_list, "acc"):
        current = parse_acc(path)
        if current["acc_cmd"] is not None and current["acc_param"] is not None:
            if not dry_run:
                update_meta_status(_meta_path(run), "acc_evaluated")
            return
        print("existing accuracy stat has non-finite metrics; recomputing {}".format(path))
    cmd = _acc_command(manifest, run)
    code = run_command(cmd, cwd=REPO_ROOT, log_path=os.path.join(run["run_dir"], "run_ablation.log"), dry_run=dry_run)
    if code != 0:
        _mark_failed(run, "acc_command_failed")
        raise RuntimeError("accuracy evaluation failed: {}".format(command_to_str(cmd)))
    if not dry_run:
        update_meta_status(_meta_path(run), "acc_evaluated")


def run_cd(manifest, run, force_list, dry_run=False):
    if importlib.util.find_spec("OCC") is None and not dry_run:
        message = (
            "CD evaluation requires pythonocc-core/OCC. "
            "Install it in the active environment, or run --stage eval_acc to collect accuracy-only metrics."
        )
        _mark_failed(run, "occ_dependency_missing")
        raise RuntimeError(message)

    ckpt = manifest.get("ckpt", "latest")
    path = pc_stat_path(run["run_dir"], ckpt)
    if os.path.exists(path) and not _force(force_list, "cd"):
        if not dry_run:
            update_meta_status(_meta_path(run), "cd_evaluated")
        return
    if os.path.exists(path) and _force(force_list, "cd"):
        os.remove(path)
    cmd = _cd_command(manifest, run)
    code = run_command(cmd, cwd=REPO_ROOT, log_path=os.path.join(run["run_dir"], "run_ablation.log"), dry_run=dry_run)
    if code != 0:
        _mark_failed(run, "cd_command_failed")
        raise RuntimeError("CD evaluation failed: {}".format(command_to_str(cmd)))
    if not dry_run:
        update_meta_status(_meta_path(run), "cd_evaluated")


def run_collect(manifest, run, dry_run=False, allow_missing_cd=False):
    if dry_run:
        print("collect {}".format(run["run_dir"]))
        return
    metric = collect_run_metrics(run["run_dir"], manifest.get("ckpt", "latest"), manifest=manifest,
                                 experiment=run["experiment"], seed=run["seed"],
                                 allow_missing_cd=allow_missing_cd)
    update_meta_status(_meta_path(run), "collected")
    if metric["acc_cmd"] is None or metric["acc_param"] is None:
        _mark_failed(run, "non_finite_accuracy_metric")
        raise RuntimeError("non-finite accuracy metric in {}".format(metrics_path(run["run_dir"])))
    if not metric["cd_available"] and allow_missing_cd:
        update_meta_status(_meta_path(run), "acc_collected", finished_at=utc_now())
        return
    if metric["valid"] == 0 or metric["invalid_ratio"] == 1.0:
        _mark_failed(run, "no_valid_cd_samples")
        print("no valid CD samples in {}; marked failed and continuing".format(
            pc_stat_path(run["run_dir"], manifest.get("ckpt", "latest"))
        ))
        return
    update_meta_status(_meta_path(run), "completed", finished_at=utc_now())


def stages_for(stage):
    if stage == "all":
        return ["train", "reconstruct", "acc", "cd", "collect"]
    if stage == "eval":
        return ["reconstruct", "acc", "cd", "collect"]
    if stage == "eval_acc":
        return ["reconstruct", "acc", "collect_acc"]
    return [stage]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(SCRIPT_DIR, "experiments.yaml"))
    parser.add_argument("--stage", default="all",
                        choices=["all", "eval", "eval_acc", "check", "train", "reconstruct", "acc", "cd", "collect", "collect_acc"])
    parser.add_argument("--force", action="append", default=[],
                        choices=["all", "train", "reconstruct", "acc", "cd", "collect"])
    parser.add_argument("--only", action="append", default=[],
                        help="experiment id to run; may be passed multiple times")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume_train", action="store_true",
                        help="continue train.py from run_dir/model/latest.pth instead of skipping")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if not print_alignment_report(manifest):
        raise SystemExit(1)
    if args.stage == "check":
        return

    for run in iter_runs(manifest, only_ids=args.only):
        for stage in stages_for(args.stage):
            if stage != "train" and not args.dry_run:
                _ensure_meta(manifest, run)
            print("[{} seed {}] {}".format(run["experiment"]["id"], run["seed"], stage))
            if stage == "train":
                run_train(manifest, run, args.force, dry_run=args.dry_run, resume_train=args.resume_train)
            elif stage == "reconstruct":
                run_reconstruct(manifest, run, args.force, dry_run=args.dry_run)
            elif stage == "acc":
                run_acc(manifest, run, args.force, dry_run=args.dry_run)
            elif stage == "cd":
                run_cd(manifest, run, args.force, dry_run=args.dry_run)
            elif stage == "collect":
                run_collect(manifest, run, dry_run=args.dry_run)
            elif stage == "collect_acc":
                run_collect(manifest, run, dry_run=args.dry_run, allow_missing_cd=True)

    if not args.dry_run:
        summary_path, count = refresh_summary(manifest)
        print("wrote {} collected metrics to {}".format(count, summary_path))
    print_alignment_report(manifest)


if __name__ == "__main__":
    main()
