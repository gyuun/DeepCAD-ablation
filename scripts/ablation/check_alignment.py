import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.ablation.utils import load_manifest, print_alignment_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(SCRIPT_DIR, "experiments.yaml"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    ok = print_alignment_report(manifest)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
