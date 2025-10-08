"""
Script để visualize bounding boxes và labels lên ảnh
Hỗ trợ YOLO format labels và hiển thị với màu sắc theo class
"""

import os
import cv2
import argparse
import yaml
import numpy as np
from pathlib import Path
import random


def load_config(config_path="config.yml"):
    """Load cấu hình từ config.yml"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_colors(num_classes):
    """
    Tạo màu sắc khác nhau cho mỗi class
    Returns: dict mapping class_id -> (B, G, R) color
    """
    np.random.seed(42)  # Đảm bảo màu sắc nhất quán
    colors = {}
    for i in range(num_classes):
        # Tạo màu sắc bright và dễ phân biệt
        colors[i] = tuple(np.random.randint(50, 255, 3).tolist())
    return colors


def get_predefined_colors():
    """
    Định nghĩa màu sắc cụ thể cho từng class (theo document layout semantics)
    0: useless, 1: form, 2: figure, 3: title, 4: table, 5: list-item, 6: text, 7: header, 8: footnote
    """
    colors = {
        0: (128, 128, 128),  # useless - Gray
        1: (255, 128, 0),    # form - Orange
        2: (128, 0, 128),    # figure - Purple
        3: (0, 0, 255),      # title - Red
        4: (255, 255, 0),    # table - Cyan
        5: (255, 0, 0),      # list-item - Blue
        6: (0, 255, 0),      # text - Green
        7: (0, 255, 255),    # header - Yellow
        8: (255, 0, 255),    # footnote - Magenta
    }
    return colors


def load_yolo_labels(label_path):
    """
    Đọc YOLO format labels
    Format: class_id x_center y_center width height (normalized 0-1)
    Returns: List of (class_id, x_center, y_center, width, height)
    """
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append((class_id, x_center, y_center, width, height))
    return labels


def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """
    Convert YOLO format (normalized) to pixel coordinates
    Returns: (x1, y1, x2, y2) - top-left and bottom-right corners
    """
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    return x1, y1, x2, y2


def draw_boxes_on_image(image, labels, id2label, colors, show_label=True, thickness=2):
    """
    Vẽ bounding boxes và labels lên ảnh
    
    Args:
        image: numpy array (H, W, 3)
        labels: List of (class_id, x_center, y_center, width, height)
        id2label: Dictionary mapping class_id -> class_name
        colors: Dictionary mapping class_id -> (B, G, R)
        show_label: Có hiển thị tên class hay không
        thickness: Độ dày của bbox
    """
    img_height, img_width = image.shape[:2]
    
    for class_id, x_center, y_center, width, height in labels:
        # Convert YOLO to pixel coordinates
        x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
        
        # Clamp coordinates
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # Get color for this class
        color = colors.get(class_id, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if show_label:
            label_text = id2label.get(str(class_id), f"Class {class_id}")
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
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


def visualize_single_image(image_path, label_path, config, output_path=None, 
                          show=True, predefined_colors=True):
    """
    Visualize một ảnh với bbox và labels
    
    Args:
        image_path: Path đến ảnh
        label_path: Path đến file label (YOLO format)
        config: Config dict từ config.yml
        output_path: Path để save ảnh (optional)
        show: Hiển thị ảnh hay không
        predefined_colors: Dùng màu định nghĩa sẵn hay random
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    # Load labels
    labels = load_yolo_labels(label_path)
    
    # Get id2label mapping
    id2label = config['model']['id2label']
    
    # Get colors
    num_labels = config['model']['num_labels']
    if predefined_colors:
        colors = get_predefined_colors()
    else:
        colors = generate_colors(num_labels)
    
    # Draw boxes
    image_with_boxes = draw_boxes_on_image(
        image.copy(), 
        labels, 
        id2label, 
        colors,
        show_label=True,
        thickness=2
    )
    
    # Add legend
    image_with_boxes = add_legend(image_with_boxes, id2label, colors)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image_with_boxes)
        print(f"Saved visualization to {output_path}")
    
    # Show if requested
    if show:
        # Resize if image too large
        max_height = 1000
        h, w = image_with_boxes.shape[:2]
        if h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            image_with_boxes = cv2.resize(image_with_boxes, (new_w, max_height))
        
        cv2.imshow(f'Visualization: {os.path.basename(image_path)}', image_with_boxes)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def add_legend(image, id2label, colors, legend_width=250):
    """
    Thêm legend (chú thích màu sắc) vào ảnh
    """
    h, w = image.shape[:2]
    
    # Create legend panel
    legend_height = len(id2label) * 30 + 20
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
    
    # Draw each class
    y_offset = 20
    for class_id_str, class_name in sorted(id2label.items(), key=lambda x: int(x[0])):
        class_id = int(class_id_str)
        color = colors.get(class_id, (255, 255, 255))
        
        # Draw color box
        cv2.rectangle(legend, (10, y_offset - 15), (30, y_offset - 5), color, -1)
        cv2.rectangle(legend, (10, y_offset - 15), (30, y_offset - 5), (0, 0, 0), 1)
        
        # Draw text
        cv2.putText(
            legend,
            f"{class_id}: {class_name}",
            (40, y_offset - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        
        y_offset += 30
    
    # Concatenate legend to image (on the right side)
    # Resize legend to match image height if needed
    if legend_height < h:
        padding = np.ones((h - legend_height, legend_width, 3), dtype=np.uint8) * 255
        legend = np.vstack([legend, padding])
    elif legend_height > h:
        legend = cv2.resize(legend, (legend_width, h))
    
    result = np.hstack([image, legend])
    return result


def visualize_dataset(data_list_path, config, output_dir=None, num_samples=5, 
                     show=True, predefined_colors=True):
    """
    Visualize random samples từ dataset
    
    Args:
        data_list_path: Path đến file .txt chứa list ảnh (train.txt, val.txt, test.txt)
        config: Config dict
        output_dir: Directory để save visualizations
        num_samples: Số lượng samples để visualize
        show: Hiển thị ảnh hay không
        predefined_colors: Dùng màu định nghĩa sẵn
    """
    # Load image paths
    with open(data_list_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    # Random sample
    if len(image_paths) > num_samples:
        image_paths = random.sample(image_paths, num_samples)
    
    print(f"Visualizing {len(image_paths)} samples from {data_list_path}")
    
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing {image_path}")
        
        # Get label path (replace /images/ with /labels/ and .jpg/.png with .txt)
        label_path = image_path.replace('/images/', '/labels/')
        label_path = os.path.splitext(label_path)[0] + '.txt'
        
        # Set output path if output_dir provided
        output_path = None
        if output_dir:
            output_filename = f"vis_{os.path.basename(os.path.splitext(image_path)[0])}.jpg"
            output_path = os.path.join(output_dir, output_filename)
        
        # Visualize
        visualize_single_image(
            image_path, 
            label_path, 
            config, 
            output_path=output_path,
            show=show,
            predefined_colors=predefined_colors
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize bounding boxes and labels on images")
    parser.add_argument('--config', type=str, default='config.yml', 
                       help='Path to config file')
    parser.add_argument('--image', type=str, 
                       help='Path to single image to visualize')
    parser.add_argument('--label', type=str, 
                       help='Path to label file for single image (YOLO format)')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test'],
                       help='Visualize samples from dataset (train/val/test)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of random samples to visualize from dataset')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display images (only save)')
    parser.add_argument('--random_colors', action='store_true',
                       help='Use random colors instead of predefined colors')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
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
            output_filename = f"vis_{os.path.basename(args.image)}"
            output_path = os.path.join(args.output_dir, output_filename)
        
        visualize_single_image(
            args.image,
            label_path,
            config,
            output_path=output_path,
            show=not args.no_show,
            predefined_colors=not args.random_colors
        )
    
    elif args.dataset:
        # Visualize dataset samples
        dataset_key_map = {'train': 'train', 'val': 'validation', 'test': 'test'}
        data_list_path = config['data'][dataset_key_map[args.dataset]]
        visualize_dataset(
            data_list_path,
            config,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            show=not args.no_show,
            predefined_colors=not args.random_colors
        )
    
    else:
        print("Error: Please specify either --image or --dataset")
        parser.print_help()


if __name__ == '__main__':
    main()

# python visualize_labels.py --dataset val --num_samples 50 --no_show