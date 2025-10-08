# 🚀 Hướng dẫn Training LayoutXLM

## ✅ Pre-requisites (Đã sẵn sàng)

- ✓ Model code đã được sửa (thêm image parameter)
- ✓ Dataset có sẵn (10,105 train, ~900 val samples)
- ✓ Config file đã được setup
- ✓ Tất cả dependencies đã install

## 📝 Các file quan trọng

```
train.py          # Main training script
config.yml        # Configuration chính (30 epochs)
dataset.py        # Dataset loader
utils.py          # Utility functions
test_components.py # Test các components
```

## 🎯 Cách chạy Training

### Option 1: Training đầy đủ (30 epochs)

```bash
# Chạy với config mặc định
python train.py

# Hoặc chỉ định config cụ thể
python train.py --config config.yml
```

### Option 2: Test nhanh (1-2 epochs để test)

**Cách 1: Tạo config test tạm thời**

```bash
# Copy và sửa config
cp config.yml config_test.yml
nano config_test.yml  # Sửa num_epochs: 1 hoặc 2
python train.py --config config_test.yml
```

**Cách 2: Quick test script**

Tạo file `quick_test.py`:

```python
from train import LayoutXLMTrainer

# Tạo trainer và override config
trainer = LayoutXLMTrainer(config_path="config.yml")

# Sửa num_epochs thành 1 để test
trainer.config['training']['num_epochs'] = 1
trainer.config['logging']['log_steps'] = 10

print("Running quick test with 1 epoch...")
trainer.train()
```

Chạy:
```bash
python quick_test.py
```

### Option 3: Resume từ checkpoint

```bash
# Sẽ cần implement resume functionality
# (Hiện tại chưa có trong code)
```

## 📊 Monitoring

Training sẽ tự động log lên **Weights & Biases**:

- Project: `doclayout`
- Entity: `thanhnx`
- Link: https://wandb.ai/thanhnx/doclayout

Hoặc xem log local trong thư mục `log/`

## 💾 Checkpoints

Checkpoints sẽ được lưu tại:
```
checkpoints/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_15.pt
├── ...
└── best_model.pt  # Model có val_loss thấp nhất
```

## ⚙️ Config quan trọng

### Trong `config.yml`:

```yaml
training:
  num_epochs: 30              # Số epochs
  learning_rate: 5e-5         # Learning rate
  batch_size: 4               # Batch size (trong data section)
  gradient_accumulation_steps: 2  # Effective batch = 4 * 2 = 8
  
checkpoint:
  save_every_n_epochs: 5      # Lưu mỗi 5 epochs
  
logging:
  log_steps: 50               # Log mỗi 50 steps
```

### Điều chỉnh để training nhanh hơn:

```yaml
training:
  num_epochs: 5               # Giảm epochs
  
data:
  batch_size: 8               # Tăng batch (nếu GPU đủ VRAM)
  preprocessing_num_workers: 8  # Tăng workers
```

## 📈 Expected Training Time

**Với config hiện tại:**
- Dataset: ~10,000 samples
- Batch size: 4
- Gradient accumulation: 2 (effective batch = 8)
- Steps per epoch: ~1,263 steps
- **1 epoch ≈ 30-60 phút** (tùy GPU)
- **30 epochs ≈ 15-30 giờ**

**Khuyến nghị:**
- Test với 1-2 epochs trước (~ 1-2 giờ)
- Xem loss có giảm không
- Nếu OK, chạy full 30 epochs

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Giảm batch size trong config.yml
data:
  batch_size: 2  # Thay vì 4
```

### Wandb login error
```bash
# Login lại wandb
wandb login 137834a14d24a94f1371552f73fd1e8c913b3862
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

## 🎬 Quick Start Commands

```bash
# 1. Test components trước
python test_components.py

# 2. Test training loop
python test_training_loop.py

# 3. Chạy training thật (1 epoch test)
# Sửa config.yml: num_epochs = 1
python train.py

# 4. Nếu OK, chạy full training
# Sửa config.yml: num_epochs = 30
python train.py
```

## 📝 Notes

**LƯU Ý QUAN TRỌNG:**
- Code hiện tại đang dùng **DUMMY INPUTS** (random tokens, random bbox)
- **Không phải real OCR data!**
- Training sẽ chạy nhưng kết quả không có ý nghĩa thực tế
- Cần implement OCR integration để có kết quả thực sự

**Next Steps để training thật sự:**
1. Integrate OCR (Tesseract/PaddleOCR) để extract text từ images
2. Align OCR boxes với ground truth labels
3. Sử dụng processor.encode() thay vì dummy inputs
4. Implement proper evaluation metrics

---

Tạo bởi: Training Setup Assistant
Date: October 8, 2025
