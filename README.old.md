# LayoutXLM Fine-tuning for Document Layout Analysis

Dự án fine-tune LayoutXLM cho bài toán phân tích layout tài liệu (Document Layout Analysis).

## Cấu trúc dự án

```
dlo/
├── config.yml              # File cấu hình training
├── train.py               # Script training chính
├── dataset.py             # Module xử lý dataset
├── utils.py               # Các hàm tiện ích
├── requirements.txt       # Dependencies
├── LayoutXLM.md          # Tài liệu về LayoutXLM
├── Data_Description.md   # Mô tả dataset
├── data/                 # Thư mục dữ liệu
├── log/                  # Thư mục logs
└── checkpoints/          # Thư mục lưu checkpoints
```

## Cài đặt

### 1. Tạo môi trường Python

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt environment
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Cấu hình

Tất cả cấu hình được quản lý trong file `config.yml`:

- **Wandb**: API key, project name, entity
- **Model**: Pretrained model, số labels, mapping labels
- **Data**: Đường dẫn datasets, batch size, preprocessing workers
- **Training**: Learning rate, epochs, optimizer, scheduler
- **Checkpoint**: Tần suất lưu checkpoint, best model
- **Logging**: Log directory, log level, logging frequency

Xem chi tiết trong file `config.yml` với comments đầy đủ.

## Dataset

Dataset bao gồm 12,636 ảnh tài liệu với annotations bounding boxes:

- **Train**: 10,105 samples
- **Validation**: 1,262 samples  
- **Test**: 1,269 samples

Format labels: YOLO format với 8 classes (0-7)

Chi tiết xem file `Data_Description.md`.

## Huấn luyện

### Chạy training

```bash
python train.py
```

Script sẽ:
- ✓ Load config từ `config.yml`
- ✓ Khởi tạo wandb logging
- ✓ Load LayoutXLM pretrained model
- ✓ Load và preprocess dataset
- ✓ Train model với progress bars đẹp (Rich)
- ✓ Lưu checkpoint mỗi 5 epochs
- ✓ Lưu best model dựa trên validation loss
- ✓ Log metrics vào wandb và file logs

### Theo dõi training

**1. Console Output:**
- Progress bars với Rich library
- Bảng thông tin model, dataset
- Metrics mỗi epoch

**2. Wandb Dashboard:**
- Truy cập: https://wandb.ai/thanhnx/doclayout
- Xem real-time metrics, charts
- So sánh các runs

**3. Log Files:**
- Logs được lưu trong `log/train_YYYYMMDD_HHMMSS.log`

## Checkpoints

Checkpoints được lưu trong thư mục `checkpoints/`:

- `checkpoint_epoch_5.pt`: Checkpoint epoch 5
- `checkpoint_epoch_10.pt`: Checkpoint epoch 10
- `checkpoint_epoch_15.pt`: Checkpoint epoch 15
- ...
- `best_model.pt`: Best model theo validation loss

## Đánh giá

Sau khi training, evaluate model trên test set:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

## Tính năng nổi bật

1. **Clean & Modular Code**: Code tổ chức rõ ràng, dễ maintain
2. **Rich Logging**: Progress bars, tables đẹp với Rich library
3. **Wandb Integration**: Track experiments chuyên nghiệp
4. **Flexible Config**: Tất cả config trong YAML, dễ thay đổi
5. **Checkpoint Management**: Auto save checkpoints và best model
6. **Comprehensive Logging**: Logs đầy đủ vào files

## Tham khảo

- [LayoutXLM Documentation](LayoutXLM.md)
- [Dataset Description](Data_Description.md)
- [Hugging Face LayoutXLM](https://huggingface.co/docs/transformers/model_doc/layoutxlm)

## License

MIT License
