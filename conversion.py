import os
import cv2
import numpy as np
from pathlib import Path
import shutil

def convert_icdar2015_to_yolo():
    """
    Convert ICDAR 2015 dataset to YOLO format
    ICDAR 2015 uses 8-point polygon annotations, we'll convert to bounding boxes
    """
    
    # Paths
    icdar_base = 'icdar2015'
    images_dir = os.path.join(icdar_base, 'images')
    gt_dir = os.path.join(icdar_base, 'gt')
    
    # YOLO output paths
    yolo_base = 'yolo_dataset'
    yolo_images = os.path.join(yolo_base, 'images')
    yolo_labels = os.path.join(yolo_base, 'labels')
    
    # Create YOLO directories
    os.makedirs(yolo_images, exist_ok=True)
    os.makedirs(yolo_labels, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).rglob(f'*{ext}'))
        image_files.extend(Path(images_dir).rglob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    converted_count = 0
    
    for image_path in image_files:
        try:
            # Get corresponding ground truth file
            image_name = image_path.stem
            
            # ICDAR 2015 ground truth files are named gt_img_*.txt
            gt_file = None
            possible_gt_names = [
                f"gt_{image_name}.txt",
                f"gt_{image_name}.TXT",
                f"{image_name}.txt",
                f"{image_name}.TXT"
            ]
            
            for gt_name in possible_gt_names:
                gt_path = Path(gt_dir) / gt_name
                if gt_path.exists():
                    gt_file = gt_path
                    break
                    
                # Also check in subdirectories
                for gt_subpath in Path(gt_dir).rglob(gt_name):
                    gt_file = gt_subpath
                    break
                if gt_file:
                    break
            
            if not gt_file:
                print(f"Warning: No ground truth file found for {image_name}")
                continue
            
            # Read image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            img_height, img_width = image.shape[:2]
            
            # Parse ground truth file
            yolo_annotations = []
            
            with open(gt_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # ICDAR 2015 format: x1,y1,x2,y2,x3,y3,x4,y4,transcription
                    parts = line.split(',')
                    
                    if len(parts) < 8:
                        continue
                    
                    # Extract coordinates (8 points for 4 corners)
                    coords = []
                    for i in range(8):
                        coords.append(float(parts[i]))
                    
                    # Convert 8-point polygon to bounding box
                    x_coords = [coords[i] for i in range(0, 8, 2)]
                    y_coords = [coords[i] for i in range(1, 8, 2)]
                    
                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)
                    
                    # Convert to YOLO format (normalized center_x, center_y, width, height)
                    center_x = (x_min + x_max) / (2 * img_width)
                    center_y = (y_min + y_max) / (2 * img_height)
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    
                    # Ensure values are within [0, 1]
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    # Class 0 for text (single class detection)
                    yolo_annotations.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line in {gt_file}: {line}")
                    continue
            
            # Copy image to YOLO images directory
            yolo_image_path = os.path.join(yolo_images, f"{image_name}.jpg")
            shutil.copy2(str(image_path), yolo_image_path)
            
            # Write YOLO annotation file
            yolo_label_path = os.path.join(yolo_labels, f"{image_name}.txt")
            with open(yolo_label_path, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(annotation + '\n')
            
            converted_count += 1
            print(f"Converted: {image_name} ({len(yolo_annotations)} annotations)")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"\nConversion completed!")
    print(f"Converted {converted_count} images")
    print(f"YOLO dataset saved to: {yolo_base}")
    
    # Create classes.txt file
    classes_file = os.path.join(yolo_base, 'classes.txt')
    with open(classes_file, 'w') as f:
        f.write('text\n')
    
    # Create dataset.yaml file for YOLO training
    yaml_content = f"""# ICDAR 2015 Text Detection Dataset in YOLO format
path: {os.path.abspath(yolo_base)}
train: images
val: images

# Classes
nc: 1  # number of classes
names: ['text']  # class names
"""
    
    yaml_file = os.path.join(yolo_base, 'dataset.yaml')
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created classes.txt and dataset.yaml files")
    
    # Create train/val split
    create_train_val_split(yolo_base)

def create_train_val_split(yolo_base, train_ratio=0.8):
    """Create train/validation split for YOLO dataset"""
    
    images_dir = os.path.join(yolo_base, 'images')
    labels_dir = os.path.join(yolo_base, 'labels')
    
    # Create train/val directories
    train_images = os.path.join(yolo_base, 'train', 'images')
    train_labels = os.path.join(yolo_base, 'train', 'labels')
    val_images = os.path.join(yolo_base, 'val', 'images')
    val_labels = os.path.join(yolo_base, 'val', 'labels')
    
    for dir_path in [train_images, train_labels, val_images, val_labels]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle and split
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Move files to train/val directories
    for file_list, img_dest, lbl_dest in [(train_files, train_images, train_labels), 
                                         (val_files, val_images, val_labels)]:
        for img_file in file_list:
            # Move image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(img_dest, img_file)
            shutil.move(src_img, dst_img)
            
            # Move corresponding label
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            src_lbl = os.path.join(labels_dir, label_file)
            dst_lbl = os.path.join(lbl_dest, label_file)
            
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
    
    # Remove original directories
    shutil.rmtree(images_dir)
    shutil.rmtree(labels_dir)
    
    # Update dataset.yaml
    yaml_content = f"""# ICDAR 2015 Text Detection Dataset in YOLO format
path: {os.path.abspath(yolo_base)}
train: train/images
val: val/images

# Classes
nc: 1  # number of classes
names: ['text']  # class names
"""
    
    yaml_file = os.path.join(yolo_base, 'dataset.yaml')
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created train/val split:")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

def verify_conversion(yolo_base):
    """Verify the YOLO conversion by checking a few samples"""
    
    train_images = os.path.join(yolo_base, 'train', 'images')
    train_labels = os.path.join(yolo_base, 'train', 'labels')
    
    if not os.path.exists(train_images):
        print("Train images directory not found")
        return
    
    image_files = os.listdir(train_images)[:3]  # Check first 3 images
    
    for img_file in image_files:
        img_path = os.path.join(train_images, img_file)
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(train_labels, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                annotations = f.readlines()
            print(f"{img_file}: {len(annotations)} text regions")
        else:
            print(f"{img_file}: No label file found")

if __name__ == "__main__":
    print("Converting ICDAR 2015 dataset to YOLO format...")
    convert_icdar2015_to_yolo()
    
    print("\nVerifying conversion...")
    verify_conversion('yolo_dataset')
    
    print("\nConversion complete! You can now use this dataset with YOLO models.")
    print("Dataset structure:")
    print("yolo_dataset/")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── val/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── dataset.yaml")
    print("└── classes.txt")
