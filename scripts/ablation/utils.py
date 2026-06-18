import datetime
import glob
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys


ARCH_FLAGS = [
    "model_type",
    "pos_encoding",
    "n_heads",
    "n_layers",
    "n_layers_decode",
    "decoder_conditioning",
    "dim_feedforward",
]

BASELINE_DEFAULTS = {
    "model_type": "ablation",
    "pos_encoding": "learned",
    "n_heads": 8,
    "n_layers": 4,
    "n_layers_decode": 4,
    "decoder_conditioning": "global_add",
    "dim_feedforward": 512,
}

ALLOWED_VALUES = {
    "pos_encoding": {"none", "sincos", "learned"},
    "n_heads": {1, 2, 4, 8},
    "n_layers": {1, 2, 4},
    "n_layers_decode": {1, 2, 4},
    "decoder_conditioning": {"global_add", "cross_attn"},
    "dim_feedforward": {128, 256, 512},
}

EXPECTED_COVERAGE = {
    "pos_encoding": {"none", "sincos", "learned"},
    "n_heads": {1, 2, 4, 8},
    "layers": {(1, 1), (2, 2), (4, 4)},
    "decoder_conditioning": {"global_add", "cross_attn"},
    "dim_feedforward": {128, 256, 512},
}

STAGE_ORDER = [
    "planned",
    "trained",
    "reconstructed",
    "acc_evaluated",
    "cd_evaluated",
    "collected",
    "completed",
]


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_manifest(path):
    with open(path, "r") as fp:
        text = fp.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "{} is not JSON and PyYAML is not installed. "
                "Use JSON-compatible YAML or install PyYAML.".format(path)
            )
        return yaml.safe_load(text)


def dump_json(path, data):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)


def append_jsonl(path, records):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fp:
        for record in records:
            fp.write(json.dumps(record, sort_keys=True) + "\n")


def sha256_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True).strip()
    except Exception:
        return None


def git_dirty():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root(), text=True)
        return bool(out.strip())
    except Exception:
        return None


def exp_config(manifest, experiment):
    cfg = dict(manifest.get("defaults", {}))
    cfg.update(experiment.get("overrides", {}))
    return cfg


def run_name(manifest, experiment, seed):
    template = experiment.get("run_name_template")
    if template is None:
        template = experiment.get("run_name")
    if template is None:
        template = manifest.get("run_name_template", "seed_{seed}")
    return str(template).format(
        seed=seed,
        experiment_id=experiment["id"],
        component=experiment.get("component", ""),
        variant=experiment.get("variant", ""),
    )


def run_dir(manifest, experiment_id, seed, experiment=None):
    if experiment is None:
        experiment = {"id": experiment_id}
    return os.path.join(manifest["proj_dir"], experiment_id, run_name(manifest, experiment, seed))


def iter_runs(manifest, only_ids=None):
    only = set(only_ids or [])
    for experiment in manifest.get("experiments", []):
        if only and experiment["id"] not in only:
            continue
        for seed in manifest.get("seeds", []):
            yield {
                "experiment": experiment,
                "seed": int(seed),
                "config": exp_config(manifest, experiment),
                "run_name": run_name(manifest, experiment, seed),
                "run_dir": run_dir(manifest, experiment["id"], seed, experiment),
            }


def load_meta(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as fp:
        return json.load(fp)


def save_meta(path, meta):
    dump_json(path, meta)


def update_meta_status(meta_path, status, **updates):
    meta = load_meta(meta_path) or {}
    meta.update(updates)
    meta["status"] = status
    meta["updated_at"] = utc_now()
    save_meta(meta_path, meta)
    return meta


def checkpoint_path(run_path, ckpt):
    name = "latest.pth" if str(ckpt) == "latest" else "ckpt_epoch{}.pth".format(ckpt)
    return os.path.join(run_path, "model", name)


def reconstruction_dir(run_path, ckpt):
    return os.path.join(run_path, "results", "test_{}".format(ckpt))


def acc_stat_path(run_path, ckpt):
    return reconstruction_dir(run_path, ckpt) + "_acc_stat.txt"


def pc_stat_path(run_path, ckpt):
    return reconstruction_dir(run_path, ckpt) + "_pc_stat.txt"


def metrics_path(run_path):
    return os.path.join(run_path, "metrics.json")


def test_split_count(data_root):
    path = os.path.join(data_root, "train_val_test_split.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as fp:
        split = json.load(fp)
    return len(split["test"])


def reconstruction_count(run_path, ckpt):
    return len(glob.glob(os.path.join(reconstruction_dir(run_path, ckpt), "*.h5")))


def command_to_str(cmd):
    return shlex.join([str(x) for x in cmd])


def run_command(cmd, cwd, log_path=None, dry_run=False):
    cmd_str = command_to_str(cmd)
    if dry_run:
        print(cmd_str)
        return 0
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as log:
            log.write("\n[{}] {}\n".format(utc_now(), cmd_str))
            proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, cwd=cwd, text=True)
    return proc.returncode


def finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def validate_manifest_against_plan(manifest):
    errors = []
    warnings = []

    defaults = manifest.get("defaults", {})
    for key, value in BASELINE_DEFAULTS.items():
        if defaults.get(key) != value:
            errors.append("default {} must be baseline value {!r}, got {!r}".format(key, value, defaults.get(key)))

    if not manifest.get("seeds"):
        errors.append("manifest must define at least one seed")

    experiment_ids = [exp.get("id") for exp in manifest.get("experiments", [])]
    if len(experiment_ids) != len(set(experiment_ids)):
        errors.append("experiment ids must be unique")
    if "baseline" not in set(experiment_ids):
        errors.append("manifest must include a baseline experiment")

    coverage = {
        "pos_encoding": set(),
        "n_heads": set(),
        "layers": set(),
        "decoder_conditioning": set(),
        "dim_feedforward": set(),
    }

    for exp in manifest.get("experiments", []):
        cfg = exp_config(manifest, exp)
        for key, allowed in ALLOWED_VALUES.items():
            if cfg.get(key) not in allowed:
                errors.append("{} has invalid {}={!r}".format(exp.get("id"), key, cfg.get(key)))
        coverage["pos_encoding"].add(cfg.get("pos_encoding"))
        coverage["n_heads"].add(cfg.get("n_heads"))
        coverage["layers"].add((cfg.get("n_layers"), cfg.get("n_layers_decode")))
        coverage["decoder_conditioning"].add(cfg.get("decoder_conditioning"))
        coverage["dim_feedforward"].add(cfg.get("dim_feedforward"))
        if exp.get("id") != "baseline" and not exp.get("overrides"):
            warnings.append("{} has no overrides and duplicates baseline".format(exp.get("id")))

    if manifest.get("allow_partial_manifest"):
        warnings.append("partial manifest: A1-A5 coverage check skipped")
    else:
        for key, expected in EXPECTED_COVERAGE.items():
            missing = expected - coverage[key]
            if missing:
                errors.append("{} coverage missing {}".format(key, sorted(missing)))

    return errors, warnings


def print_alignment_report(manifest):
    errors, warnings = validate_manifest_against_plan(manifest)
    if errors:
        print("Alignment check failed:")
        for error in errors:
            print("  - {}".format(error))
    elif manifest.get("allow_partial_manifest"):
        print("Alignment check passed: partial manifest is valid; A1-A5 coverage check skipped.")
    else:
        print("Alignment check passed: manifest covers A1-A5 baseline and variants.")
    for warning in warnings:
        print("Warning: {}".format(warning))
    return not errors


def python_executable():
    return sys.executable or "python3"
