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
- `label`: Class ID (0-8), tổng 9 classes
- `x_center, y_center`: Tọa độ tâm bounding box (normalized 0-1)
- `width, height`: Chiều rộng và cao bounding box (normalized 0-1)

## Label Classes và Phân Bố

Dataset bao gồm 9 classes document layout:

| Class ID | Tên Class | Mô tả | Số lượng boxes | Tỷ lệ |
|----------|-----------|-------|----------------|-------|
| 0 | useless | Vùng không sử dụng/Background | 489 | 0.31% |
| 1 | form | Form/Biểu mẫu | 1,141 | 0.71% |
| 2 | figure | Hình ảnh/Biểu đồ/Charts | 6,995 | 4.36% |
| 3 | title | Tiêu đề chính tài liệu | 22,394 | 13.97% |
| 4 | table | Bảng biểu | 6,294 | 3.93% |
| 5 | list-item | Danh sách/List items | 71,991 | 44.91% |
| 6 | text | Văn bản chính/Đoạn văn | 45,064 | 28.11% |
| 7 | header | Đầu trang/Headers | 5,722 | 3.57% |
| 8 | footnote | Chú thích/Footnotes/References | 213 | 0.13% |

**Tổng cộng:** 160,303 bounding boxes

**Đặc điểm:**
- Class 5 (list-item) chiếm đa số với 44.91% - danh sách và list items
- Class 6 (text) chiếm 28.11% - văn bản chính và đoạn văn
- Class 3 (title) chiếm 13.97% - tiêu đề tài liệu
- Classes khác (useless, form, figure, table, header, footnote) chiếm tỷ lệ nhỏ
- Dataset imbalanced - cần xem xét weighted loss hoặc sampling strategies

**Ví dụ:**
```
4 0.5323488248274629 0.12990555570697557 0.8446667562395134 0.08497909582142592
```

## Mục đích sử dụng
Dataset phục vụ bài toán **Document Layout Analysis** - phát hiện và phân loại các vùng layout trong tài liệu (tiêu đề, đoạn văn, bảng, hình ảnh, v.v.). Dữ liệu được sử dụng để fine-tune LayoutXLM cho tác vụ object detection/segmentation trên tài liệu.
