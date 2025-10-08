"""
Script để visualize predictions từ model so với ground truth
Hiển thị side-by-side comparison: Ground Truth vs Predictions
"""

import os
import cv2
import torch
import argparse
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import pytesseract

from transformers import LayoutLMv2ForTokenClassification
from dataset_ocr import LayoutXLMProcessor

from visualize_labels import (
    load_config, 
    get_predefined_colors, 
    generate_colors,
    draw_boxes_on_image,
    add_legend,
    load_yolo_labels,
    yolo_to_bbox
)


def load_model(config, checkpoint_path):
    """
    Load trained model từ checkpoint
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        config['model']['pretrained_model_name'],
        num_labels=config['model']['num_labels']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Device: {device}")
    
    return model, device


def extract_ocr_data(image_path):
    """
    Extract OCR data (words và bounding boxes) từ ảnh sử dụng Tesseract
    Returns: (words, boxes) where boxes are normalized [0, 1000]
    """
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Run Tesseract OCR
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    words = []
    boxes = []
    
    for i in range(len(ocr_result['text'])):
        text = ocr_result['text'][i].strip()
        if text:
            x = ocr_result['left'][i]
            y = ocr_result['top'][i]
            w = ocr_result['width'][i]
            h = ocr_result['height'][i]
            
            # Normalize to [0, 1000] scale (LayoutXLM convention)
            x_norm = int((x / width) * 1000)
            y_norm = int((y / height) * 1000)
            x2_norm = int(((x + w) / width) * 1000)
            y2_norm = int(((y + h) / height) * 1000)
            
            words.append(text)
            boxes.append([x_norm, y_norm, x2_norm, y2_norm])
    
    return words, boxes


def predict_on_image(model, processor, image_path, device, config):
    """
    Chạy prediction trên một ảnh
    Returns: list of (word, box, predicted_label_id)
    """
    # Extract OCR data
    words, boxes = extract_ocr_data(image_path)
    
    if len(words) == 0:
        print(f"Warning: No text detected in {image_path}")
        return []
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Prepare inputs
    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding='max_length',
        max_length=config['data']['max_length'],
        return_tensors='pt'
    )
    
    # Move to device
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().cpu().numpy()
    
    # Get word-level predictions (skip special tokens)
    word_ids = encoding.word_ids(batch_index=0)
    
    word_predictions = []
    seen_words = set()
    
    for idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in seen_words:
            if word_id < len(words):
                predicted_label = predictions[idx]
                word_predictions.append({
                    'word': words[word_id],
                    'box': boxes[word_id],  # Normalized [0, 1000]
                    'label': int(predicted_label)
                })
                seen_words.add(word_id)
    
    return word_predictions


def draw_predictions_on_image(image, predictions, id2label, colors, show_label=True, thickness=2):
    """
    Vẽ predicted bounding boxes lên ảnh
    
    Args:
        image: numpy array (H, W, 3)
        predictions: List of dicts with 'word', 'box' [x1, y1, x2, y2] normalized [0,1000], 'label'
        id2label: Dictionary mapping class_id -> class_name
        colors: Dictionary mapping class_id -> (B, G, R)
    """
    img_height, img_width = image.shape[:2]
    
    for pred in predictions:
        box = pred['box']
        label = pred['label']
        word = pred['word']
        
        # Denormalize from [0, 1000] to pixel coordinates
        x1 = int((box[0] / 1000) * img_width)
        y1 = int((box[1] / 1000) * img_height)
        x2 = int((box[2] / 1000) * img_width)
        y2 = int((box[3] / 1000) * img_height)
        
        # Clamp coordinates
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # Get color for this class
        color = colors.get(label, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if show_label:
            label_text = id2label.get(str(label), f"Class {label}")
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )
            
            # Draw background for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                image,
                label_text,
                (x1 + 2, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness,
                cv2.LINE_AA
            )
    
    return image


def visualize_comparison(image_path, label_path, model, processor, config, device,
                        output_path=None, show=True, predefined_colors=True):
    """
    Visualize comparison giữa Ground Truth và Predictions
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    img_height, img_width = image.shape[:2]
    
    # Get colors and id2label
    id2label = config['model']['id2label']
    num_labels = config['model']['num_labels']
    
    if predefined_colors:
        colors = get_predefined_colors()
    else:
        colors = generate_colors(num_labels)
    
    # === Ground Truth ===
    gt_labels = load_yolo_labels(label_path)
    gt_image = draw_boxes_on_image(
        image.copy(),
        gt_labels,
        id2label,
        colors,
        show_label=True,
        thickness=2
    )
    
    # Add title
    cv2.putText(
        gt_image,
        "Ground Truth",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    
    # === Predictions ===
    predictions = predict_on_image(model, processor, image_path, device, config)
    pred_image = draw_predictions_on_image(
        image.copy(),
        predictions,
        id2label,
        colors,
        show_label=True,
        thickness=2
    )
    
    # Add title
    cv2.putText(
        pred_image,
        "Predictions",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )
    
    # === Combine side by side ===
    # Add separator
    separator = np.ones((img_height, 5, 3), dtype=np.uint8) * 255
    comparison = np.hstack([gt_image, separator, pred_image])
    
    # Add legend
    comparison = add_legend(comparison, id2label, colors)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, comparison)
        print(f"Saved comparison to {output_path}")
    
    # Show if requested
    if show:
        # Resize if too large
        max_width = 1920
        h, w = comparison.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_h = int(h * scale)
            comparison = cv2.resize(comparison, (max_width, new_h))
        
        cv2.imshow(f'GT vs Predictions: {os.path.basename(image_path)}', comparison)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions vs ground truth")
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                       help='Path to single image to visualize')
    parser.add_argument('--label', type=str,
                       help='Path to label file for single image (YOLO format)')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test'],
                       help='Visualize samples from dataset')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize from dataset')
    parser.add_argument('--output_dir', type=str, default='predictions_vis',
                       help='Directory to save visualizations')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display images (only save)')
    parser.add_argument('--random_colors', action='store_true',
                       help='Use random colors instead of predefined colors')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    model, device = load_model(config, args.checkpoint)
    
    # Initialize processor
    processor = LayoutXLMProcessor.from_pretrained(
        config['model']['pretrained_model_name'],
        apply_ocr=False
    )
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.image:
        # Visualize single image
        if not args.label:
            # Try to infer label path
            label_path = args.image.replace('/images/', '/labels/')
            label_path = os.path.splitext(label_path)[0] + '.txt'
        else:
            label_path = args.label
        
        output_path = None
        if args.output_dir:
            output_filename = f"pred_{os.path.basename(args.image)}"
            output_path = os.path.join(args.output_dir, output_filename)
        
        visualize_comparison(
            args.image,
            label_path,
            model,
            processor,
            config,
            device,
            output_path=output_path,
            show=not args.no_show,
            predefined_colors=not args.random_colors
        )
    
    elif args.dataset:
        # Visualize dataset samples
        import random
        
        dataset_key_map = {'train': 'train', 'val': 'validation', 'test': 'test'}
        data_list_path = config['data'][dataset_key_map[args.dataset]]
        with open(data_list_path, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        # Random sample
        if len(image_paths) > args.num_samples:
            image_paths = random.sample(image_paths, args.num_samples)
        
        print(f"Visualizing {len(image_paths)} samples from {args.dataset} set")
        
        for i, image_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] Processing {image_path}")
            
            # Get label path
            label_path = image_path.replace('/images/', '/labels/')
            label_path = os.path.splitext(label_path)[0] + '.txt'
            
            # Set output path
            output_path = None
            if args.output_dir:
                output_filename = f"pred_{os.path.basename(os.path.splitext(image_path)[0])}.jpg"
                output_path = os.path.join(args.output_dir, output_filename)
            
            # Visualize
            visualize_comparison(
                image_path,
                label_path,
                model,
                processor,
                config,
                device,
                output_path=output_path,
                show=not args.no_show,
                predefined_colors=not args.random_colors
            )
    
    else:
        print("Error: Please specify either --image or --dataset")
        parser.print_help()


if __name__ == '__main__':
    main()
