# Ablation Scripts Quick Use

이 디렉터리는 `ablation_study_strategy.md`의 Part 1 자동화 범위를 실행한다.

## 1. Manifest 설정

실험 matrix는 `experiments.yaml`에서 수정한다. 현재 파일은 JSON-compatible YAML 형식이므로 PyYAML 없이도 실행된다.

중간중간 학습을 끊고 이어서 진행하는 방식을 기본 사용법으로 둔다. 이 경우 checkpoint는 `latest`를 사용한다.

```json
"ckpt": "latest",
"nr_epochs": 101,
"seeds": [2026]
```

`train.py`는 `nr_epochs`보다 1 적게 도는 구조라서, 실제 100 epoch를 돌리려면 `nr_epochs`를 `101`로 둔다. 200 epoch까지 이어서 돌리려면 `201`, 300 epoch는 `301`, 400 epoch는 `401`로 올린다.

전체 실험을 돌릴 때는 `seeds`를 원하는 seed 목록으로 둔다. 서버에서 시간을 확인하는 동안에는 seed 하나만 두는 것이 편하다.

## 2. Manifest 확인

```bash
uv run python scripts/ablation/check_alignment.py \
  --manifest scripts/ablation/experiments.yaml
```

이 검사는 A1-A5 ablation variant와 baseline 값이 계획과 맞는지 확인한다.

## 3. Dry Run

실제 학습 전에 생성될 명령을 확인한다.

```bash
uv run python scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --stage train \
  --resume_train \
  --dry_run
```

특정 experiment만 확인하려면:

```bash
uv run python scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --only A4_cross_attn \
  --stage train \
  --resume_train \
  --dry_run
```

`run_dir/model/latest.pth`가 이미 있으면 dry run 명령에 `--cont --ckpt latest`가 포함된다. 없으면 처음 학습 명령이 생성된다.

## 4. 학습

학습은 100 epoch 단위로 끊어서 확인하고 이어서 진행한다.

1차 100 epoch:

```json
"nr_epochs": 101,
"ckpt": "latest"
```

```bash
CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python -u scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --stage train \
  --resume_train
```

2차 200 epoch까지 이어서:

```json
"nr_epochs": 201,
"ckpt": "latest"
```

```bash
CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python -u scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --stage train \
  --resume_train
```

같은 방식으로 `nr_epochs`만 `301`, `401`처럼 올리면 기존 `latest.pth`부터 이어서 학습한다.

특정 experiment만 학습하려면:

```bash
CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python -u scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --stage train \
  --only baseline \
  --resume_train
```

`--resume_train`은 `run_dir/model/latest.pth`가 있을 때 `train.py --cont --ckpt latest`를 붙인다. 없으면 처음부터 학습한다.

## 5. 평가

학습이 원하는 epoch까지 끝난 뒤 평가 stage를 실행한다.

가져온 1 epoch baseline checkpoint
`proj_log/ablation/baseline/seed_2026_test1epoch/model/latest.pth`는 전용 manifest로 바로 평가할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python -u scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments_baseline_1epoch.yaml \
  --stage eval \
  --only baseline
```

위 명령은 학습을 실행하지 않고 `reconstruct -> acc -> cd -> collect`만 수행한다. `cd` 평가는 `data/pc_cad`가 필요하다.

전체 ablation manifest의 특정 실험을 평가하려면:

```bash
CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python -u scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --stage reconstruct

CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python -u scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --stage acc

CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python -u scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --stage cd

uv run python scripts/ablation/run_ablation.py \
  --manifest scripts/ablation/experiments.yaml \
  --stage collect
```

`acc`와 `cd`는 autoencoder reconstruction 평가다. LGAN 학습은 필요 없다.

`cd` 평가는 `data/pc_cad`가 필요하다. 없으면 `cad_json`에서 point cloud를 생성해야 한다.

## 6. 재실행

이미 완료된 stage는 기본적으로 skip된다. 강제로 다시 실행하려면:

```bash
uv run python scripts/ablation/run_ablation.py \
  --stage train \
  --only A1_pos_none \
  --force train
```

주의: `--force train`만 쓰면 기존 학습을 덮어쓸 수 있다. 이어서 학습하려면 `--resume_train`을 같이 사용한다.

전체를 강제 재실행하려면:

```bash
uv run python scripts/ablation/run_ablation.py --stage all --force all
```

## 7. LGAN과 생성 평가

LGAN은 autoencoder reconstruction ablation에는 필요 없다. 다음 경우에만 필요하다.

```text
random generation 평가: COV, MMD, JSD
```

이 경우 AE checkpoint별로 latent code를 먼저 만들고, 그 latent distribution에 대해 LGAN을 따로 학습해야 한다.

```bash
# encode all data to latent space
CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python -u test.py \
  --proj_dir proj_log/ablation/baseline \
  --exp_name seed_2026 \
  --data_root data \
  --gpu_ids 0 \
  --batch_size 512 \
  --num_workers 8 \
  --mode enc \
  --ckpt latest \
  --model_type ablation \
  --pos_encoding learned \
  --n_heads 8 \
  --n_layers 4 \
  --n_layers_decode 4 \
  --decoder_conditioning global_add \
  --dim_feedforward 512

# train latent GAN
CUDA_VISIBLE_DEVICES=0 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run python -u lgan.py \
  --proj_dir proj_log/ablation/baseline \
  --exp_name seed_2026 \
  --ae_ckpt latest \
  --gpu_ids 0
```

단, ablation 실험에서 AE 구조가 바뀌면 latent distribution도 바뀐다. 논문 pretrained LGAN을 새로 학습한 ablation AE에 그대로 붙여 쓰는 것은 공정한 비교가 아니다. 생성 평가까지 할 계획이면 각 AE variant/seed마다 LGAN을 따로 맞춰야 한다.

## 8. 산출물

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
