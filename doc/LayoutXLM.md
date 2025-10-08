## Giới thiệu LayoutXLM
LayoutXLM là model mã nguồn mở được phát triển để xử lý hiểu biết văn bản có bố cục trong tài liệu đa ngôn ngữ. Nó kế thừa và mở rộng từ LayoutLM, bổ sung khả năng hỗ trợ nhiều ngôn ngữ và tương tác giữa văn bản cùng bố cục hình ảnh tài liệu. Model này giúp kết hợp thông tin về văn bản, vị trí trên trang và hình ảnh để phục vụ các tác vụ như nhận diện thông tin, trích xuất dữ liệu, phân loại hình ảnh tài liệu.

## Cách tải về LayoutXLM
LayoutXLM có sẵn trên kho Hugging Face như các model LayoutLM khác. Có thể tải về sử dụng thư viện transformers của Hugging Face với lệnh đơn giản:

```python
from transformers import LayoutXLMTokenizer, LayoutXLMForTokenClassification

tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
model = LayoutXLMForTokenClassification.from_pretrained("microsoft/layoutxlm-base")
```

## Cách fine-tune LayoutXLM
Việc fine-tune LayoutXLM thường xoay quanh các tác vụ như phân loại token (token classification) hoặc nhận dạng thông tin trong tài liệu.

- Chuẩn bị dữ liệu đầu vào gồm:
  - Văn bản đã được đưa qua OCR (như Tesseract) để có danh sách từ (words).
  - Tọa độ bounding box mỗi từ trên trang tài liệu (định dạng chuẩn: (x0, y0, x1, y1), scale chuẩn hóa trên thang 0-1000).
  - Nhãn mục tiêu cho các từ (ví dụ nhãn cho các trường thông tin cần trích xuất).

- Chuẩn bị dữ liệu đầu vào cho tokenizer bao gồm text, bounding boxes và attention mask.

- Sử dụng API transformers, gọi model với input_ids, bbox và labels để huấn luyện:

```python
encoding = tokenizer(" ".join(words), return_tensors="pt", truncation=True, padding="max_length")
bbox = torch.tensor([normalized_bounding_boxes])
labels = torch.tensor([token_labels])  # nhãn cho token

outputs = model(input_ids=encoding.input_ids, bbox=bbox, attention_mask=encoding.attention_mask, labels=labels)
loss = outputs.loss
logits = outputs.logits
loss.backward()
```

- Tiến hành huấn luyện trên tập dữ liệu phù hợp với tác vụ của bạn.

## Định dạng input/output

- Input:
  - input_ids: mã hóa token của văn bản.
  - bbox: tọa độ bounding box chuẩn hóa của mỗi token trên tài liệu (thang 0-1000).
  - attention_mask: mặt nạ attention cho token.
  - labels (chỉ cho training): nhãn phân loại cho từng token.

- Output:
  - logits: xác suất phân loại của từng token.
  - loss: giá trị mất mát nếu input có labels.

## Tóm tắt
LayoutXLM là model mạnh mẽ hỗ trợ đa ngôn ngữ, tích hợp văn bản và bố cục tài liệu, phù hợp cho các tác vụ trích xuất thông tin từ tài liệu số. Có sẵn trên Hugging Face, dễ dàng tải về và fine-tune theo nhu cầu với dữ liệu kèm thông tin vị trí bounding box của từ trên trang tài liệu để khai thác tối đa thông tin bố cục.

Nếu cần, có thể cung cấp thêm ví dụ code cụ thể cho từng bước fine-tune. 

Thông tin tham khảo lấy từ nhiều nguồn Hugging Face, bài tutorial fine-tune LayoutLM và tài liệu Microsoft.[1][2][3][4]

[1](https://huggingface.co/docs/transformers/v4.51.1/model_doc/layoutlm)
[2](https://huggingface.co/docs/transformers/en/model_doc/layoutlm)
[3](https://nanonets.com/blog/layoutlm-explained/)
[4](https://github.com/cydal/LayoutLM_pytorch)
[5](https://www.philschmid.de/fine-tuning-layoutlm)
[6](https://stackoverflow.com/questions/69910822/input-data-format-for-simpletransformers-ai-layoutlm-models)
[7](https://ubiai.tools/fine-tuning-layoutlm-for-document-information-extraction/)
[8](https://github.com/microsoft/unilm/issues/286)
[9](https://www.nitorinfotech.com/blog/how-can-layoutlm-transform-text-extraction/)
[10](https://wandb.ai/wandb-data-science/layoutlm_sroie_demo/reports/Fine-tuning-LayoutLM-on-SROIE-Information-Extraction-from-Scanned-Receipts-Internal---VmlldzoxMTY5MzU5)
[11](https://datasaur.ai/blog-posts/layoutlm-invoice-extraction)
[12](https://towardsdatascience.com/fine-tuning-layoutlm-v2-for-invoice-recognition-91bf2546b19e/)
[13](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout?view=doc-intel-4.0.0)
[14](https://www.philschmid.de/fine-tuning-layoutlm-keras)
[15](https://informediq.com/architecting-ai-for-documents-a-deep-dive-into-layoutlm/)
[16](https://ubiai.tools/fine-tuning-question-answering-model-with-layoutlm-v2/)
[17](https://radar.elyadata.com/data-engineering/LayoutLM.html)
[18](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb)
[19](https://dataloop.ai/library/model/impira_layoutlm-document-qa/)
[20](https://www.kdnuggets.com/how-to-layoutlm-document-understanding-information-extraction-hugging-face-transformers)