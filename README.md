# Finetune LayoutXLM for Document Layout Analysis# LayoutXLM Fine-tuning for Document Layout Analysis



Fine-tuning LayoutXLM model for Document Layout Analysis using real OCR data from Tesseract.Dự án fine-tune LayoutXLM cho bài toán phân tích layout tài liệu (Document Layout Analysis).



## 🎯 Project Overview## Cấu trúc dự án



This project fine-tunes the LayoutXLM model for document layout analysis tasks. It uses **real OCR data** extracted via Tesseract to train the model on document structure understanding.```

dlo/

### Key Features├── config.yml              # File cấu hình training

├── train.py               # Script training chính

- ✅ **Real OCR Integration**: Uses Tesseract OCR to extract text and bounding boxes├── dataset.py             # Module xử lý dataset

- ✅ **LayoutXLM Model**: Fine-tuning microsoft/layoutxlm-base for token classification├── utils.py               # Các hàm tiện ích

- ✅ **8-Class Document Layout**: Classifies document regions into 8 layout categories├── requirements.txt       # Dependencies

- ✅ **Mixed Precision Training**: Supports FP16 for faster training├── LayoutXLM.md          # Tài liệu về LayoutXLM

- ✅ **Wandb Integration**: Full experiment tracking and logging├── Data_Description.md   # Mô tả dataset

- ✅ **Early Stopping**: Prevents overfitting with patience-based stopping├── data/                 # Thư mục dữ liệu

- ✅ **Comprehensive Testing**: Component tests for validation before training├── log/                  # Thư mục logs

└── checkpoints/          # Thư mục lưu checkpoints

## 📁 Project Structure```



```## Cài đặt

.

├── config.yml                  # Training configuration### 1. Tạo môi trường Python

├── train.py                   # Main training script

├── dataset_ocr.py             # OCR-integrated dataset```bash

├── utils.py                   # Utility functions# Tạo virtual environment

├── test_components.py         # Component validation testspython -m venv venv

├── requirements.txt           # Python dependencies

├── verify_data_pipeline.py    # Data pipeline verification script# Kích hoạt environment

├── doc/                       # Documentationsource venv/bin/activate  # Linux/Mac

│   ├── DATA_PIPELINE_STATUS.md# hoặc

│   ├── VERIFICATION_COMPLETE.mdvenv\Scripts\activate  # Windows

│   ├── XAC_NHAN_DU_LIEU_THAT.md```

│   ├── TRAINING_GUIDE.md

│   └── ...### 2. Cài đặt dependencies

├── data/                      # Dataset directory

│   ├── train.txt             # Training image list```bash

│   ├── val.txt               # Validation image listpip install -r requirements.txt

│   └── test.txt              # Test image list```

└── checkpoints/               # Model checkpoints (gitignored)

```## Cấu hình



## 🚀 Quick StartTất cả cấu hình được quản lý trong file `config.yml`:



### 1. Prerequisites- **Wandb**: API key, project name, entity

- **Model**: Pretrained model, số labels, mapping labels

- Python 3.8+- **Data**: Đường dẫn datasets, batch size, preprocessing workers

- CUDA-capable GPU (recommended)- **Training**: Learning rate, epochs, optimizer, scheduler

- Tesseract OCR installed- **Checkpoint**: Tần suất lưu checkpoint, best model

- **Logging**: Log directory, log level, logging frequency

Install Tesseract:

```bashXem chi tiết trong file `config.yml` với comments đầy đủ.

# Ubuntu/Debian

sudo apt-get install tesseract-ocr## Dataset



# macOSDataset bao gồm 12,636 ảnh tài liệu với annotations bounding boxes:

brew install tesseract

- **Train**: 10,105 samples

# Windows- **Validation**: 1,262 samples  

# Download from: https://github.com/UB-Mannheim/tesseract/wiki- **Test**: 1,269 samples

```

Format labels: YOLO format với 8 classes (0-7)

### 2. Installation

Chi tiết xem file `Data_Description.md`.

```bash

# Clone repository## Huấn luyện

git clone https://github.com/xt2201/Finetune-LayoutXLM.git

cd Finetune-LayoutXLM### Chạy training



# Create virtual environment```bash

python -m venv venvpython train.py

source venv/bin/activate  # Linux/Mac```

# or: venv\Scripts\activate  # Windows

Script sẽ:

# Install dependencies- ✓ Load config từ `config.yml`

pip install -r requirements.txt- ✓ Khởi tạo wandb logging

```- ✓ Load LayoutXLM pretrained model

- ✓ Load và preprocess dataset

### 3. Data Preparation- ✓ Train model với progress bars đẹp (Rich)

- ✓ Lưu checkpoint mỗi 5 epochs

Prepare your dataset with the following structure:- ✓ Lưu best model dựa trên validation loss

- ✓ Log metrics vào wandb và file logs

```

data/### Theo dõi training

├── train.txt              # List of training image paths

├── val.txt                # List of validation image paths**1. Console Output:**

├── test.txt               # List of test image paths- Progress bars với Rich library

├── train/- Bảng thông tin model, dataset

│   ├── images/           # Training images- Metrics mỗi epoch

│   └── labels/           # YOLO format labels

├── val/**2. Wandb Dashboard:**

│   ├── images/- Truy cập: https://wandb.ai/thanhnx/doclayout

│   └── labels/- Xem real-time metrics, charts

└── test/- So sánh các runs

    ├── images/

    └── labels/**3. Log Files:**

```- Logs được lưu trong `log/train_YYYYMMDD_HHMMSS.log`



Label format (YOLO): `class_id x_center y_center width height`## Checkpoints



### 4. ConfigurationCheckpoints được lưu trong thư mục `checkpoints/`:



Edit `config.yml` to customize training:- `checkpoint_epoch_5.pt`: Checkpoint epoch 5

- `checkpoint_epoch_10.pt`: Checkpoint epoch 10

```yaml- `checkpoint_epoch_15.pt`: Checkpoint epoch 15

data:- ...

  use_ocr: true              # Use real OCR (Tesseract)- `best_model.pt`: Best model theo validation loss

  batch_size: 8

  max_length: 512## Đánh giá



training:Sau khi training, evaluate model trên test set:

  num_epochs: 30

  learning_rate: 0.00005```bash

  gradient_accumulation_steps: 4python evaluate.py --checkpoint checkpoints/best_model.pt

  early_stopping_patience: 5```



wandb:## Tính năng nổi bật

  project: your-project-name

  api_key: your-api-key1. **Clean & Modular Code**: Code tổ chức rõ ràng, dễ maintain

```2. **Rich Logging**: Progress bars, tables đẹp với Rich library

3. **Wandb Integration**: Track experiments chuyên nghiệp

### 5. Verify Setup4. **Flexible Config**: Tất cả config trong YAML, dễ thay đổi

5. **Checkpoint Management**: Auto save checkpoints và best model

Run component tests to ensure everything is configured correctly:6. **Comprehensive Logging**: Logs đầy đủ vào files



```bash## Tham khảo

python test_components.py

```- [LayoutXLM Documentation](LayoutXLM.md)

- [Dataset Description](Data_Description.md)

Expected output:- [Hugging Face LayoutXLM](https://huggingface.co/docs/transformers/model_doc/layoutxlm)

```

✓ PASS   - Imports## License

✓ PASS   - Config

✓ PASS   - DatasetMIT License

✓ PASS   - Model
✓ PASS   - Forward Pass
✓ PASS   - Utilities

Total: 6/6 tests passed
```

### 6. Verify Data Pipeline

Confirm that real OCR data is being used:

```bash
python verify_data_pipeline.py
```

### 7. Start Training

```bash
python train.py
```

## 📊 Model Architecture

- **Base Model**: microsoft/layoutxlm-base
- **Task**: Token Classification
- **Input**: Text tokens + Bounding boxes + Document images
- **Output**: 8-class layout predictions per token
- **Parameters**: ~368M total, ~368M trainable

### Label Classes

```
0: O (Other/Background)
1-7: Document layout regions (configurable in config.yml)
```

## 🔧 Training Configuration

### Key Hyperparameters

```yaml
# Optimizer
- AdamW with weight decay 0.01
- Learning rate: 5e-5
- Warmup ratio: 6%

# Training
- Batch size: 8 (effective 32 with gradient accumulation)
- Max epochs: 30
- Gradient clipping: 1.0
- Mixed precision: FP16 (optional)

# Regularization
- Early stopping patience: 5 epochs
- Weight decay: 0.01
```

## 📈 Monitoring

Training metrics are logged to Weights & Biases:

- Training loss
- Validation loss and accuracy
- Learning rate schedule
- GPU utilization
- Sample predictions

## 🧪 Data Pipeline

### Real OCR Processing

1. **Image Loading**: Load document images
2. **OCR Extraction**: Tesseract extracts text and word-level bounding boxes
3. **Coordinate Normalization**: Convert to [0, 1000] scale for LayoutXLM
4. **Label Alignment**: Align layout labels to OCR words using IoU
5. **Tokenization**: LayoutXLMProcessor tokenizes with spatial information
6. **Batching**: Custom collate function for proper batching

### Fallback Behavior

When OCR is disabled (`use_ocr: false`):
- Used only for quick component testing
- Returns dummy tokens to verify pipeline without OCR dependency
- **NOT used in actual training** (training uses `use_ocr: true`)

## 📝 Output

### Checkpoints

Saved in `checkpoints/` directory:
- `best_model.pt`: Best model based on validation loss
- `checkpoint_epoch_N.pt`: Periodic checkpoints every N epochs

### Logs

- **Console**: Rich formatted progress bars and metrics
- **File**: Detailed logs in `log/` directory
- **Wandb**: Interactive dashboard with all metrics

## 🔍 Verification

The project includes comprehensive verification:

1. **Component Tests** (`test_components.py`)
   - Validates all components load correctly
   - Tests model forward pass
   - Checks dataset pipeline

2. **Data Pipeline Verification** (`verify_data_pipeline.py`)
   - Confirms real OCR is being used
   - Checks data quality
   - Validates token diversity

3. **Documentation**
   - `DATA_PIPELINE_STATUS.md`: Technical details
   - `VERIFICATION_COMPLETE.md`: Full verification report
   - `XAC_NHAN_DU_LIEU_THAT.md`: Vietnamese verification

## 🛠️ Troubleshooting

### Common Issues

1. **Tesseract not found**
   ```bash
   # Install Tesseract OCR
   sudo apt-get install tesseract-ocr
   ```

2. **CUDA out of memory**
   - Reduce `batch_size` in config.yml
   - Increase `gradient_accumulation_steps`
   - Reduce `max_length`

3. **Slow training**
   - Enable FP16: Set `fp16: true` in config.yml
   - Increase `num_workers` for data loading

## 📚 Documentation

- `doc/TRAINING_GUIDE.md`: Detailed training guide
- `doc/DATA_PIPELINE_STATUS.md`: Data pipeline documentation
- `doc/VERIFICATION_COMPLETE.md`: Verification report
- `doc/PROJECT_SUMMARY.md`: Project overview

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [LayoutXLM](https://github.com/microsoft/unilm/tree/master/layoutxlm) by Microsoft
- [Detectron2](https://github.com/facebookresearch/detectron2) by Facebook Research
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

## 📧 Contact

- GitHub: [@xt2201](https://github.com/xt2201)
- Email: xuanthanh2201.work@gmail.com

---

**Status**: ✅ Production Ready - All components verified with real OCR data
