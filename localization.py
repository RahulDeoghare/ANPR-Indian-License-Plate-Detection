import cv2
import os
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
import re

# Load license plate YOLO model
plate_model = YOLO("truck.pt")  # Replace with your trained license plate model

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang='en')

# Video input
video_path = "anpr_vid.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# Create output directory
output_dir = "output_plates"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
plate_count = 0

# Tracking variables
saved_plates = {}  # Dictionary to store already saved plates with their OCR text
plate_positions = {}  # Dictionary to track plate positions across frames
position_threshold = 50  # Distance threshold for considering plates as the same
min_confidence = 0.3  # Reduced minimum OCR confidence (was 0.6)
frame_skip = 5  # Process every 5th frame to reduce computation
min_plate_area = 500  # Minimum plate area in pixels
saved_plate_hashes = set()  # Store image hashes to avoid visual duplicates

def clean_text(text):
    """Clean OCR text by removing special characters and spaces"""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def get_image_hash(image):
    """Calculate a simple hash of the image for duplicate detection"""
    # Resize to standard size and convert to grayscale
    resized = cv2.resize(image, (64, 32))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    # Calculate mean and create binary hash
    mean = np.mean(gray)
    binary = (gray > mean).astype(np.uint8)
    return hash(binary.tobytes())

def enhance_plate_image(plate_img):
    """Enhance plate image for better OCR"""
    # Convert to grayscale
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img
    
    # Resize if too small
    height, width = gray.shape
    if height < 30 or width < 100:
        scale_factor = max(30/height, 100/width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply image enhancement
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Convert back to BGR for OCR
    return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

def is_duplicate_plate(current_text, current_pos, current_hash, saved_plates, plate_positions, saved_plate_hashes):
    """Check if current plate is a duplicate based on OCR text, position, and image hash"""
    # Check image hash first (fastest)
    if current_hash in saved_plate_hashes:
        return True
    
    # Check position-based duplicates
    for saved_text, saved_pos in plate_positions.items():
        if calculate_distance(current_pos, saved_pos) < position_threshold:
            return True
    
    # Check text-based duplicates (if we have text)
    if current_text:
        current_text_clean = clean_text(current_text)
        if current_text_clean in saved_plates and len(current_text_clean) > 2:
            return True
    
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    
    # Skip frames to reduce processing load
    if frame_count % frame_skip != 0:
        continue

    results = plate_model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # Check minimum plate area
        plate_area = (x2 - x1) * (y2 - y1)
        if plate_area < min_plate_area:
            print(f"[SKIP] Plate too small: {plate_area} pixels")
            continue

        plate_img = frame[y1:y2, x1:x2]

        if plate_img.size == 0:
            continue

        # Calculate center position and image hash
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        current_position = (center_x, center_y)
        current_hash = get_image_hash(plate_img)

        # Check for duplicates early
        if is_duplicate_plate("", current_position, current_hash, saved_plates, plate_positions, saved_plate_hashes):
            print(f"[DUPLICATE] Skipping visually similar plate at position {current_position}")
            continue

        # Enhance plate image for better OCR
        enhanced_plate = enhance_plate_image(plate_img)

        # Run OCR on enhanced image
        extracted_text = ""
        max_confidence = 0
        
        try:
            result = ocr.ocr(enhanced_plate, cls=False)
            if result and isinstance(result[0], list):
                for line in result[0]:
                    text, score = line[1]
                    if score > max_confidence:
                        max_confidence = score
                        extracted_text = text
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")
            # Still save the plate even if OCR fails
            max_confidence = 0.1  # Give it a minimal confidence to allow saving

        # More lenient confidence check - save plates even with low OCR confidence
        if max_confidence >= min_confidence:
            confidence_status = "HIGH"
        elif max_confidence > 0:
            confidence_status = "LOW"
            extracted_text = f"UNREADABLE_{plate_count}"  # Give unreadable plates a unique identifier
        else:
            confidence_status = "NONE"
            extracted_text = f"NOTEXT_{plate_count}"  # Give plates without text a unique identifier

        # Final duplicate check with OCR text
        if max_confidence >= min_confidence and is_duplicate_plate(extracted_text, current_position, current_hash, saved_plates, plate_positions, saved_plate_hashes):
            print(f"[DUPLICATE] Skipping already saved plate: {extracted_text}")
            continue

        # Save the plate (unique plates, even with poor OCR)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plate_filename = f"{output_dir}/plate_{frame_count}_{plate_count}_{timestamp}.jpg"
        cv2.imwrite(plate_filename, plate_img)

        # Store the plate information
        clean_extracted_text = clean_text(extracted_text) if extracted_text else f"UNKNOWN_{plate_count}"
        saved_plates[clean_extracted_text] = {
            'filename': plate_filename,
            'timestamp': timestamp,
            'confidence': max_confidence,
            'original_text': extracted_text
        }
        plate_positions[extracted_text] = current_position
        saved_plate_hashes.add(current_hash)

        plate_count += 1

cap.release()
print(f"[DONE] Processing complete.")
print(f"[SUMMARY] Total unique plates saved: {plate_count}")
print(f"[SUMMARY] Saved plates: {list(saved_plates.keys())}")
