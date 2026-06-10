# Ablation Scripts Quick Use

이 디렉터리는 `ablation_study_strategy.md`의 Part 1 자동화 범위를 실행한다.

## 1. Manifest 확인

실험 matrix는 `experiments.yaml`에서 수정한다. 현재 파일은 JSON-compatible YAML 형식이므로 PyYAML 없이도 실행된다.

```bash
python3 scripts/ablation/check_alignment.py \
  --manifest scripts/ablation/experiments.yaml
```

이 검사는 A1-A5 ablation variant와 baseline 값이 계획과 맞는지 확인한다.

## 2. Dry Run

실제 학습 전에 생성될 명령을 확인한다.

```bash
python3 scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --stage all \
  --dry_run
```

특정 experiment만 확인하려면:

```bash
python3 scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --only A4_cross_attn \
  --stage all \
  --dry_run
```

## 3. 실행

전체 flow:

```bash
python3 scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --stage all
```

단계별 실행:

```bash
python3 scripts/ablation/run_ablation.py --stage train
python3 scripts/ablation/run_ablation.py --stage reconstruct
python3 scripts/ablation/run_ablation.py --stage acc
python3 scripts/ablation/run_ablation.py --stage cd
python3 scripts/ablation/run_ablation.py --stage collect
```

## 4. 재실행

이미 완료된 stage는 기본적으로 skip된다. 강제로 다시 실행하려면:

```bash
python3 scripts/ablation/run_ablation.py \
  --stage train \
  --only A1_pos_none \
  --force train
```

전체를 강제 재실행하려면:

```bash
python3 scripts/ablation/run_ablation.py --stage all --force all
```

## 5. 산출물

기본 출력 위치:

```text
proj_log/ablation/<experiment_id>/seed_<seed>/
```

주요 파일:

```text
config.txt
run_meta.json
model/
results/test_<ckpt>/
results/test_<ckpt>_acc_stat.txt
results/test_<ckpt>_pc_stat.txt
metrics.json
per_item_cd.csv
```

전체 요약:

```text
proj_log/ablation/summary/runs.jsonl
proj_log/ablation/summary/metrics.jsonl
```
