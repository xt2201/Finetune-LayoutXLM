
1. **Tìm hiểu và tài liệu hóa về LayoutXLM**

   * Nghiên cứu trên internet về model open-source **LayoutXLM**: kiến trúc, cách finetune, input/output.
   * Kết hợp tham khảo thêm nội dung trong file **`LayoutXLM.md`**.

2. **Phân tích dataset**
   * Khảo sát dữ liệu trong thư mục **`data/`**.
   * Viết file **`Data_Description.md`** mô tả dataset ngắn gọn, xúc tích (≤ 30 dòng), bao gồm:

     * Cấu trúc thư mục và định dạng dữ liệu
     * Kích thước và đặc điểm quan trọng (số lượng mẫu, nhãn, kiểu dữ liệu)
     * Mục đích và cách dùng trong finetune.
     * **Lưu ý:** Ở mỗi dòng trong các file label, giá trị **đầu tiên** là *nhãn*, **4 giá trị tiếp theo** là *tọa độ bounding box* theo format:
       ```
       <label> <x_center> <y_center> <width> <height>
       ```

       Ví dụ:

       ```
       4 0.5323488248274629 0.12990555570697557 0.8446667562395134 0.08497909582142592
       ```

3. **Phân tích cấu hình**

   * Đọc file **`config.yml`** để hiểu các tham số cấu hình và có thể bố sung thêm các cấu hình cần thiết  nếu còn thiếu(batch size, learning rate, optimizer, số epoch, v.v.)
   * Comment ý nghĩa của từng cấu hình trong tiến trình huấn luyện.

4. **Triển khai huấn luyện (finetune) LayoutXLM**

   * Viết code finetune model **LayoutXLM** với các yêu cầu sau:
     * Sử dụng mọi cấu hình ở trong file **`config.yml`**
     * Lưu **checkpoint mỗi 5 epoch** và **best model**.
     * Logging progress chi tiết và chuyên nghiệp:
       * Sử dụng **Rich** để hiển thị tiến trình, bảng kết quả, thanh progress.
       * Ghi log đầy đủ vào thư mục **`log/`**.
     * Theo dõi metrics qua **Weights & Biases (wandb)**.
   * Đảm bảo code rõ ràng, clean, ngắn gọn, dễ tái sử dụng, có cấu trúc tốt.