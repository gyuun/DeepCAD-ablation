import argparse
import csv
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.ablation.utils import (
    acc_stat_path,
    finite_or_none,
    load_manifest,
    load_meta,
    metrics_path,
    pc_stat_path,
    run_dir,
    dump_json,
    append_jsonl,
)


COMMANDS = ["Line", "Arc", "Circle", "EOS", "SOL", "Ext"]


def _read(path):
    with open(path, "r") as fp:
        return fp.read()


def _float_after(label, text):
    match = re.search(re.escape(label) + r"\s*:\s*([-+0-9.eE]+|nan|inf|-inf)", text)
    if not match:
        return None
    return finite_or_none(match.group(1))


def _array_after(label, text):
    match = re.search(re.escape(label) + r"\s*:\s*\[([^\]]*)\]", text, flags=re.S)
    if not match:
        return []
    return [finite_or_none(x) for x in re.findall(r"[-+0-9.eE]+|nan|inf|-inf", match.group(1))]


def parse_acc(path):
    text = _read(path)
    command_acc_values = _array_after("each command acc", text)
    command_acc = {
        COMMANDS[i]: command_acc_values[i]
        for i in range(min(len(COMMANDS), len(command_acc_values)))
    }

    line = _array_after("Line param acc", text)
    arc = _array_after("Arc param acc", text)
    circle = _array_after("Circle param acc", text)
    ext = _array_after("Ext param acc", text)

    return {
        "acc_cmd": _float_after("avg command acc (ACC_cmd)", text),
        "acc_param": _float_after("avg param acc (ACC_param)", text),
        "command_acc": command_acc,
        "arg_group_acc": {
            "line": _mean(line),
            "arc": _mean(arc),
            "circle": _mean(circle),
            "plane": _mean(ext[:3]),
            "transform": _mean(ext[3:7]),
            "extent": _mean(ext[7:]),
        },
    }


def _mean(values):
    finite = [x for x in values if x is not None]
    if not finite:
        return None
    return sum(finite) / len(finite)


def parse_cd(path, per_item_csv=None):
    text = _read(path)
    total_match = re.search(
        r"total:\s*(\d+)\s+invalid:\s*(\d+)\s+invalid ratio:\s*([-+0-9.eE]+|nan|inf|-inf)",
        text,
    )
    dist_match = re.search(
        r"avg dist:\s*([-+0-9.eE]+|nan|inf|-inf)\s+"
        r"trim_avg_dist:\s*([-+0-9.eE]+|nan|inf|-inf)\s+"
        r"med dist:\s*([-+0-9.eE]+|nan|inf|-inf)",
        text,
    )

    per_item = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        idx, data_id, value = parts
        if not idx.isdigit():
            continue
        per_item.append({
            "index": int(idx),
            "data_id": data_id,
            "chamfer": None if value == "None" else finite_or_none(value),
        })

    if per_item_csv is not None and per_item:
        with open(per_item_csv, "w") as fp:
            writer = csv.DictWriter(fp, fieldnames=["index", "data_id", "chamfer"])
            writer.writeheader()
            writer.writerows(per_item)

    total = int(total_match.group(1)) if total_match else None
    invalid = int(total_match.group(2)) if total_match else None
    invalid_ratio = finite_or_none(total_match.group(3)) if total_match else None
    valid = None if total is None or invalid is None else total - invalid

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "invalid_ratio": invalid_ratio,
        "chamfer_mean": finite_or_none(dist_match.group(1)) if dist_match else None,
        "chamfer_trimmed_mean": finite_or_none(dist_match.group(2)) if dist_match else None,
        "chamfer_median": finite_or_none(dist_match.group(3)) if dist_match else None,
    }


def collect_run_metrics(run_path, ckpt, manifest=None, experiment=None, seed=None):
    meta = load_meta(os.path.join(run_path, "run_meta.json")) or {}
    acc_path = acc_stat_path(run_path, ckpt)
    cd_path = pc_stat_path(run_path, ckpt)
    if not os.path.exists(acc_path):
        raise FileNotFoundError(acc_path)
    if not os.path.exists(cd_path):
        raise FileNotFoundError(cd_path)

    acc = parse_acc(acc_path)
    cd = parse_cd(cd_path, per_item_csv=os.path.join(run_path, "per_item_cd.csv"))

    metric = {
        "experiment_id": meta.get("experiment_id") or (experiment or {}).get("id"),
        "component": meta.get("component") or (experiment or {}).get("component"),
        "variant": meta.get("variant") or (experiment or {}).get("variant"),
        "seed": meta.get("seed") if meta.get("seed") is not None else seed,
        "ckpt": ckpt,
        "acc_cmd": acc["acc_cmd"],
        "acc_param": acc["acc_param"],
        "invalid_ratio": cd["invalid_ratio"],
        "chamfer_mean": cd["chamfer_mean"],
        "chamfer_trimmed_mean": cd["chamfer_trimmed_mean"],
        "chamfer_median": cd["chamfer_median"],
        "command_acc": acc["command_acc"],
        "arg_group_acc": acc["arg_group_acc"],
        "total": cd["total"],
        "valid": cd["valid"],
        "invalid": cd["invalid"],
    }
    dump_json(metrics_path(run_path), metric)
    return metric


def refresh_summary(manifest):
    metric_records = []
    run_records = []
    for exp in manifest.get("experiments", []):
        for seed in manifest.get("seeds", []):
            this_run_dir = run_dir(manifest, exp["id"], seed, exp)
            meta_path = os.path.join(this_run_dir, "run_meta.json")
            meta = load_meta(meta_path)
            if meta is not None:
                run_records.append(meta)

            path = metrics_path(this_run_dir)
            if os.path.exists(path):
                with open(path, "r") as fp:
                    metric_records.append(json.load(fp))
    summary_path = os.path.join(manifest["proj_dir"], "summary", "metrics.jsonl")
    runs_path = os.path.join(manifest["proj_dir"], "summary", "runs.jsonl")
    append_jsonl(summary_path, metric_records)
    append_jsonl(runs_path, run_records)
    return summary_path, len(metric_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(SCRIPT_DIR, "experiments.yaml"))
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--refresh_summary", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    ckpt = args.ckpt if args.ckpt is not None else manifest.get("ckpt")

    if args.run_dir is not None:
        metric = collect_run_metrics(args.run_dir, ckpt)
        print(json.dumps(metric, indent=2, sort_keys=True))

    if args.refresh_summary or args.run_dir is not None:
        summary_path, count = refresh_summary(manifest)
        print("wrote {} records to {}".format(count, summary_path))


if __name__ == "__main__":
    main()
