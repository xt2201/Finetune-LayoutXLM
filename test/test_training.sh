#!/bin/bash
# Quick test training với 1 epoch để verify setup

echo "=================================================="
echo "DRY RUN TEST - Training với 1 epoch"
echo "=================================================="

# Backup config gốc
cp config.yml config.yml.backup

# Tạo temporary config với 1 epoch
cat > config_test.yml << 'EOF'
# Test Configuration - Chỉ chạy 1 epoch để test
wandb:
  api_key: 137834a14d24a94f1371552f73fd1e8c913b3862
  project: doclayout-test
  entity: thanhnx
  job_type: test
  name: layoutxlm-dryrun

model:
  pretrained_model_name: microsoft/layoutxlm-base
  num_labels: 8
  id2label:
    '0': 'O'
    '1': '1'
    '2': '2'
    '3': '3'
    '4': '4'
    '5': '5'
    '6': '6'
    '7': '7'

data:
  train: data/train.txt
  validation: data/val.txt
  test: data/test.txt
  preprocessing_num_workers: 2
  pad_to_max_length: true
  batch_size: 2
  max_length: 512
  image_size: [224, 224]

training:
  num_epochs: 1                    # Chỉ 1 epoch để test
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  fp16: false
  seed: 42
  
optimizer:
  name: adamw
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  name: linear
  num_warmup_steps: null

checkpoint:
  save_dir: checkpoints_test
  save_every_n_epochs: 1
  save_best_only: false
  metric_for_best_model: eval_loss
  greater_is_better: false

logging:
  log_dir: log_test
  log_level: INFO
  log_steps: 10                    # Log thường xuyên hơn
  eval_steps: 100
  save_steps: 100
EOF

echo ""
echo "✓ Created test config: config_test.yml"
echo "  - 1 epoch only"
echo "  - Batch size: 2"
echo "  - Log every 10 steps"
echo ""
echo "Starting dry run..."
echo ""

# Chạy training với config test
python train.py --config config_test.yml

echo ""
echo "=================================================="
echo "Dry run completed!"
echo "Check log_test/ for logs"
echo "Check checkpoints_test/ for saved models"
echo "=================================================="
