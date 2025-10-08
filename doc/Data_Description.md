# Dataset Description

## Cấu trúc thư mục
```
data/
├── train.txt          # Danh sách đường dẫn ảnh training (10,105 samples)
├── val.txt            # Danh sách đường dẫn ảnh validation (1,262 samples)
├── test.txt           # Danh sách đường dẫn ảnh test (1,269 samples)
├── train/
│   ├── images/        # 10,105 ảnh tài liệu (.png)
│   └── labels/        # 10,105 file nhãn (.txt)
├── val/
│   ├── images/        # 1,262 ảnh tài liệu
│   └── labels/        # 1,262 file nhãn
└── test/
    ├── images/        # 1,269 ảnh tài liệu
    └── labels/        # 1,269 file nhãn
```

## Định dạng dữ liệu

**Ảnh:** Các tài liệu đã được scan/chuyển đổi thành ảnh PNG, chủ yếu là văn bản tiếng Việt (Thông tư, Quyết định, hợp đồng, biểu mẫu).

**Nhãn:** Mỗi file label tương ứng với một ảnh, chứa nhiều dòng theo format YOLO:
```
<label> <x_center> <y_center> <width> <height>
```
- `label`: Class ID (0-7), tổng 8 classes
- `x_center, y_center`: Tọa độ tâm bounding box (normalized 0-1)
- `width, height`: Chiều rộng và cao bounding box (normalized 0-1)

**Ví dụ:**
```
4 0.5323488248274629 0.12990555570697557 0.8446667562395134 0.08497909582142592
```

## Mục đích sử dụng
Dataset phục vụ bài toán **Document Layout Analysis** - phát hiện và phân loại các vùng layout trong tài liệu (tiêu đề, đoạn văn, bảng, hình ảnh, v.v.). Dữ liệu được sử dụng để fine-tune LayoutXLM cho tác vụ object detection/segmentation trên tài liệu.
