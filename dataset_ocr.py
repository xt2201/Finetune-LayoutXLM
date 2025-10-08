"""
Dataset with OCR Integration for LayoutXLM
This version integrates OCR (Tesseract/EasyOCR) to extract text and word boxes
"""

import os
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Small epsilon to avoid division by zero
IOU_EPS = 1e-6

import pytesseract

class LayoutXLMDataset(Dataset):
    """
    Dataset for LayoutXLM with OCR integration
    
    This dataset performs OCR on images to extract text and word-level bounding boxes,
    which are required inputs for LayoutXLM model.
    """
    
    def __init__(
        self,
        image_list_file: str,
        processor,
        max_seq_length: int = 512,
        use_ocr: bool = True
    ):
        """
        Args:
            image_list_file: Path to .txt file containing image paths
            processor: LayoutXLMProcessor instance (should have apply_ocr=False)
            max_seq_length: Maximum sequence length
            use_ocr: Whether to use OCR (if False, uses dummy text)
        """
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.use_ocr = use_ocr
        
        # Load image paths
        with open(image_list_file, 'r') as f:
            self.image_paths = [line.strip() for line in f if line.strip()]
        
        # Derive label paths
        self.label_paths = [
            path.replace('/images/', '/labels/').replace('.png', '.txt')
            for path in self.image_paths
        ]
        
        print(f"Loaded {len(self.image_paths)} samples from {image_list_file}")
        if not self.use_ocr:
            print("Warning: OCR disabled or unavailable. Using dummy text.")
    
    def _extract_text_and_boxes(self, image: Image.Image) -> Tuple[List[str], List[List[int]]]:
        """
        Extract text and word-level bounding boxes using OCR
        
        Returns:
            words: List of words extracted from image
            boxes: List of bounding boxes [x0, y0, x1, y1] for each word
        """
        if not self.use_ocr:
            # Return minimal fallback if OCR is unavailable - use simple dummy text
            width, height = image.size
            # Create a single word token with a reasonable box
            return ["dummy"], [[0, 0, min(1000, int(1000 * width / height)), 1000]]
        
        # Perform OCR with Tesseract
        width, height = image.size
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        words = []
        boxes = []
        
        for i in range(len(ocr_data['text'])):
            word = ocr_data['text'][i].strip()
            if word:  # Skip empty strings
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                # Normalize to [0, 1000] scale (LayoutXLM requirement)
                box = [
                    int((x / width) * 1000),
                    int((y / height) * 1000),
                    int(((x + w) / width) * 1000),
                    int(((y + h) / height) * 1000)
                ]
                
                words.append(word)
                boxes.append(box)
        
        # If no text detected, return dummy
        if len(words) == 0:
            return ["empty"], [[0, 0, 1000, 1000]]
        
        return words, boxes
    
    def _load_labels(self, label_path: str) -> Dict[str, np.ndarray]:
        """
        Load document-level labels (for the whole image)
        
        Returns:
            Dictionary with layout labels and their bounding boxes
        """
        if not os.path.exists(label_path):
            return {'labels': np.array([]), 'boxes': np.array([])}
        
        labels = []
        boxes = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    label = int(parts[0])
                    box = [float(x) for x in parts[1:5]]
                    labels.append(label)
                    boxes.append(box)
        
        return {
            'labels': np.array(labels, dtype=np.int64),
            'boxes': np.array(boxes, dtype=np.float32)
        }
    
    @staticmethod
    def _yolo_to_layoutlm_boxes(boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
        """Convert YOLO normalized boxes to LayoutXLM coordinate system [0, 1000]."""
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.int64)

        boxes_pixel = boxes.copy()
        boxes_pixel[:, 0] = boxes[:, 0] * img_width   # x_center
        boxes_pixel[:, 1] = boxes[:, 1] * img_height  # y_center
        boxes_pixel[:, 2] = boxes[:, 2] * img_width   # width
        boxes_pixel[:, 3] = boxes[:, 3] * img_height  # height

        x0 = boxes_pixel[:, 0] - boxes_pixel[:, 2] / 2
        y0 = boxes_pixel[:, 1] - boxes_pixel[:, 3] / 2
        x1 = boxes_pixel[:, 0] + boxes_pixel[:, 2] / 2
        y1 = boxes_pixel[:, 1] + boxes_pixel[:, 3] / 2

        x0 = np.clip(x0, 0, img_width)
        y0 = np.clip(y0, 0, img_height)
        x1 = np.clip(x1, 0, img_width)
        y1 = np.clip(y1, 0, img_height)

        boxes_xyxy = np.stack([
            (x0 / img_width) * 1000,
            (y0 / img_height) * 1000,
            (x1 / img_width) * 1000,
            (y1 / img_height) * 1000
        ], axis=1)

        return boxes_xyxy.astype(np.int64)

    def _align_labels_to_words(
        self,
        word_boxes: List[List[int]],
        layout_boxes: np.ndarray,
        layout_labels: np.ndarray
    ) -> List[int]:
        """Align layout region labels to OCR word boxes."""
        if len(word_boxes) == 0:
            return []

        default_label = 0

        if layout_boxes.size == 0 or layout_labels.size == 0:
            return [default_label] * len(word_boxes)

        word_labels: List[int] = []

        for word_box in word_boxes:
            x0, y0, x1, y1 = word_box
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2

            assigned = default_label

            # First try center-point containment
            for layout_box, label in zip(layout_boxes, layout_labels):
                lx0, ly0, lx1, ly1 = layout_box
                if lx0 <= cx <= lx1 and ly0 <= cy <= ly1:
                    assigned = int(label)
                    break

            # If center not contained, fallback to IoU-based closest match
            if assigned == default_label:
                word_area = max((x1 - x0) * (y1 - y0), IOU_EPS)
                best_iou = 0.0
                best_label = default_label

                for layout_box, label in zip(layout_boxes, layout_labels):
                    lx0, ly0, lx1, ly1 = layout_box
                    inter_x0 = max(x0, lx0)
                    inter_y0 = max(y0, ly0)
                    inter_x1 = min(x1, lx1)
                    inter_y1 = min(y1, ly1)

                    inter_w = max(0.0, inter_x1 - inter_x0)
                    inter_h = max(0.0, inter_y1 - inter_y0)
                    inter_area = inter_w * inter_h

                    layout_area = max((lx1 - lx0) * (ly1 - ly0), IOU_EPS)
                    union = word_area + layout_area - inter_area
                    iou = inter_area / union if union > 0 else 0.0

                    if iou > best_iou:
                        best_iou = iou
                        best_label = int(label)

                if best_iou >= 0.1:
                    assigned = best_label

            word_labels.append(assigned)

        return word_labels
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample prepared for LayoutXLM
        
        Returns:
            Dictionary with processor outputs ready for model input
        """
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Extract text and word boxes via OCR
        words, word_boxes = self._extract_text_and_boxes(image)

        # Load and convert layout labels
        layout_labels = self._load_labels(label_path)
        layout_boxes = self._yolo_to_layoutlm_boxes(
            layout_labels['boxes'],
            image.width,
            image.height
        )

        # Align layout labels to word tokens
        word_labels = self._align_labels_to_words(word_boxes, layout_boxes, layout_labels['labels'])
        word_labels = [int(label) if isinstance(label, (int, np.integer)) else 0 for label in word_labels]
        
        # Encode with LayoutXLM processor
        encoding = self.processor(
            image,
            words,
            boxes=word_boxes,
            word_labels=word_labels,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension (added by processor)
        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)
        
        # Add metadata
        encoding['image_path'] = img_path
        
        return encoding


def collate_fn_layoutxlm(batch: List[Dict]) -> Dict:
    """
    Custom collate function for LayoutXLM batches
    
    Args:
        batch: List of encoded samples
        
    Returns:
        Batched dictionary ready for model
    """
    # Stack tensors - LayoutXLMProcessor returns 'image' not 'pixel_values'
    keys = ['input_ids', 'bbox', 'attention_mask', 'labels', 'image']
    batched = {}
    
    for key in keys:
        if key in batch[0]:
            batched[key] = torch.stack([item[key] for item in batch])
    
    # Rename 'image' to 'pixel_values' for model compatibility
    if 'image' in batched:
        batched['pixel_values'] = batched.pop('image')
    
    # Add metadata
    batched['image_paths'] = [item.get('image_path', '') for item in batch]
    
    return batched
