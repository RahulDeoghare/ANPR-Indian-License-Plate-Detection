import cv2
import torch
import numpy as np
import time
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR
from difflib import SequenceMatcher
import logging
from datetime import datetime

# --- Helper Functions from trial.py ---

def is_valid_indian_plate(plate_text):
    if not plate_text or len(plate_text) < 8:
        return False
    clean_plate = plate_text.replace(" ", "").upper()
    if len(clean_plate) < 9 or len(clean_plate) > 11:
        return False
    pattern1 = r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4}$'
    pattern2 = r'^\d{2}BH\d{4}[A-Z]{2}$'
    pattern3 = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
    import re
    if re.match(pattern1, clean_plate) or re.match(pattern2, clean_plate) or re.match(pattern3, clean_plate):
        state_codes = [
            'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CT', 'DN', 'DD', 'DL', 'GA', 'GJ', 'HR', 'HP', 'JH', 'JK',
            'KA', 'KL', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TG',
            'TR', 'UP', 'UT', 'WB', 'LA', 'PY'
        ]
        if clean_plate.startswith(tuple(state_codes)) or 'BH' in clean_plate[:4]:
            return True
    return False

def format_indian_plate(plate_text):
    clean_plate = plate_text.replace(" ", "").upper()
    if len(clean_plate) == 10:
        if clean_plate[2:4] == 'BH':
            return f"{clean_plate[:2]} {clean_plate[2:4]} {clean_plate[4:8]} {clean_plate[8:]}"
        else:
            return f"{clean_plate[:2]} {clean_plate[2:4]} {clean_plate[4:6]} {clean_plate[6:]}"
    return plate_text

def calculate_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def point_line_distance(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    line_vec = np.array([x2 - x1, y2 - y1])
    point_vec = np.array([px - x1, py - y1])
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)
    t = max(0, min(1, np.dot(point_vec, line_vec) / (line_len ** 2)))
    projection = np.array([x1, y1]) + t * line_vec
    return np.linalg.norm(np.array([px, py]) - projection)

def is_point_above_line(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    return ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) > 0

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two texts using SequenceMatcher"""
    if not text1 or not text2:
        return 0
    return SequenceMatcher(None, text1, text2).ratio()

def should_update_plate(current_plate, new_plate, new_conf):
    """Determine if we should update the plate based on confidence and similarity"""
    if not current_plate:
        return True
    
    current_text = current_plate["ocr_text"]
    current_conf = current_plate["conf"]
    similarity = calculate_text_similarity(current_text, new_plate)
    
    # If texts are very similar (>0.8), only update if new confidence is higher
    if similarity > 0.8:
        return new_conf > current_conf
    
    # If texts are different, require significantly higher confidence to update
    return new_conf > (current_conf + 0.15)

def associate_plate_to_vehicle(plate_box, vehicle_boxes):
    plate_centroid = calculate_centroid(plate_box)
    min_dist = float("inf")
    associated_vehicle_id = None
    for vid, (vbox, _) in vehicle_boxes.items():
        vehicle_centroid = calculate_centroid(vbox)
        dist = np.linalg.norm(np.array(plate_centroid) - np.array(vehicle_centroid))
        if dist < min_dist and dist < 200:
            min_dist = dist
            associated_vehicle_id = vid
    return associated_vehicle_id

# --- Main Logic ---

def main():
    # --- Setup Logging ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('anpr_log.txt'),
            logging.StreamHandler()
        ]
    )

    # --- Model Paths ---
    VEHICLE_MODEL = "Front.pt"  # Change to your vehicle YOLO model
    PLATE_MODEL = "truck.pt"    # Change to your plate YOLO model
    VIDEO_SOURCE = "vms_test3.mp4"            # 0 for webcam, or path to video file
    
    # --- Constants ---
    PLATE_DETECTION_TIMEOUT = 3.0  # seconds to wait for plate detection
    MIN_CONFIDENCE_THRESHOLD = 0.5  # minimum confidence for plate detection

    # --- Load Models ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vehicle_model = YOLO(VEHICLE_MODEL)
    plate_model = YOLO(PLATE_MODEL)
    vehicle_model.to(device)
    plate_model.to(device)

    # --- PaddleOCR ---
    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang='en', use_gpu=(device=='cuda'))
    except:
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')

    # --- Video Setup ---
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_merged.mp4", fourcc, fps, (w, h))
    os.makedirs("number_plates", exist_ok=True)

    # --- Tracking ---
    tracked_vehicles = {}  # {id: (bbox, has_crossed, first_seen_time)}
    last_plate_for_vehicle = {}  # {vehicle_id: {"plate_box":..., "ocr_text":..., "conf":...}}
    vehicle_warnings = set()  # Set to track vehicles that have been warned about
    next_id = 0
    crossing_count = 0
    vehicles_without_plates = 0

    # --- ROI Line ---
    roi_points = []
    roi_defined = False
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_points, roi_defined
        if event == cv2.EVENT_LBUTTONDOWN and not roi_defined:
            roi_points.append((x, y))
            if len(roi_points) == 2:
                roi_defined = True
    cv2.namedWindow("Vehicle ANPR")
    cv2.setMouseCallback("Vehicle ANPR", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Draw ROI ---
        if roi_defined:
            cv2.line(frame, roi_points[0], roi_points[1], (0, 255, 0), 2)
        elif roi_points:
            cv2.circle(frame, roi_points[0], 5, (0, 255, 0), -1)
            cv2.putText(frame, "Click second point for ROI line", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Click to set first ROI point", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Crossings: {crossing_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Detection and Tracking ---
        if roi_defined:
            # --- Vehicle Detection ---
            vehicle_results = vehicle_model.predict(frame, conf=0.5, verbose=False, device=device)[0]
            current_vehicles = {}
            for box in vehicle_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centroid = calculate_centroid((x1, y1, x2, y2))
                vehicle_id = None
                min_dist = float("inf")
                for vid, (prev_bbox, _) in tracked_vehicles.items():
                    prev_centroid = calculate_centroid(prev_bbox)
                    dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                    if dist < min_dist and dist < 100:
                        min_dist = dist
                        vehicle_id = vid
                if vehicle_id is None:
                    vehicle_id = next_id
                    next_id += 1
                old_data = tracked_vehicles.get(vehicle_id, (None, False, None))
                has_crossed = old_data[1] if old_data[0] is not None else False
                first_seen_time = old_data[2] if old_data[0] is not None else time.time()
                current_vehicles[vehicle_id] = ((x1, y1, x2, y2), has_crossed, first_seen_time)
                
                # Check for missing plate after timeout
                current_time = time.time()
                if (vehicle_id not in last_plate_for_vehicle and 
                    vehicle_id not in vehicle_warnings and 
                    current_time - first_seen_time > PLATE_DETECTION_TIMEOUT):
                    msg = f"WARNING: Vehicle {vehicle_id} detected for {current_time - first_seen_time:.1f}s without a license plate!"
                    logging.warning(msg)
                    vehicle_warnings.add(vehicle_id)
                
                # Draw vehicle box and info
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, centroid, 5, (0, 255, 255), -1)
                cv2.putText(frame, f"ID: {vehicle_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            tracked_vehicles = current_vehicles

            # --- Plate Detection ---
            plate_results = plate_model.predict(frame, conf=0.5, verbose=False, device=device)[0]
            for plate_box in plate_results.boxes:
                conf = float(plate_box.conf[0])
                if conf < 0.5:
                    continue
                px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                associated_vehicle_id = associate_plate_to_vehicle((px1, py1, px2, py2), tracked_vehicles)
                if associated_vehicle_id is not None:
                    # Process plate if vehicle doesn't have one or if enough time has passed
                    current_plate = last_plate_for_vehicle.get(associated_vehicle_id)
                    plate_img = frame[py1:py2, px1:px2]
                    # Preprocess for OCR (from trial.py)
                    if plate_img.shape[0] > 0 and plate_img.shape[1] > 0:
                        scale_factor = max(3.0, 100.0 / min(plate_img.shape[:2]))
                        new_width = int(plate_img.shape[1] * scale_factor)
                        new_height = int(plate_img.shape[0] * scale_factor)
                        plate_resized = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                        gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        enhanced = clahe.apply(gray)
                        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
                        processed_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
                    else:
                        processed_img = plate_img
                    # OCR
                    ocr_result = ocr.ocr(processed_img)
                    best_text = ""
                    best_confidence = 0
                    if ocr_result and len(ocr_result) > 0:
                        result = ocr_result[0]
                        if isinstance(result, list):
                            for item in result:
                                if len(item) == 2 and isinstance(item[1], tuple) and len(item[1]) == 2:
                                    text, confidence = item[1]
                                    cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                                    if (confidence > best_confidence and 
                                        len(cleaned_text) >= 8 and 
                                        is_valid_indian_plate(cleaned_text) and
                                        confidence > MIN_CONFIDENCE_THRESHOLD):
                                        best_text = cleaned_text
                                        best_confidence = confidence
                    # Process detected plate
                    if best_text:
                        formatted_plate = format_indian_plate(best_text)
                        new_plate_info = {
                            "plate_box": (px1, py1, px2, py2),
                            "ocr_text": formatted_plate,
                            "conf": best_confidence
                        }
                        
                        # Update plate only if it's better than existing one
                        if should_update_plate(current_plate, formatted_plate, best_confidence):
                            # Save plate image
                            plate_path = f"number_plates/plate_{int(time.time())}_{associated_vehicle_id}.jpg"
                            cv2.imwrite(plate_path, plate_img)
                            
                            last_plate_for_vehicle[associated_vehicle_id] = new_plate_info
                            logging.info(f"Vehicle {associated_vehicle_id}: Updated plate to {formatted_plate} (conf: {best_confidence:.2f})")
                # Draw plate box
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                cv2.putText(frame, f"Plate {conf:.2f}", (px1, py1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # --- Crossing Logic ---
            for vehicle_id, (bbox, has_crossed, first_seen_time) in tracked_vehicles.items():
                centroid = calculate_centroid(bbox)
                if not has_crossed and not is_point_above_line(centroid, roi_points[0], roi_points[1]):
                    dist_to_line = point_line_distance(centroid, roi_points[0], roi_points[1])
                    if dist_to_line < 10:
                        crossing_count += 1
                        tracked_vehicles[vehicle_id] = (bbox, True, first_seen_time)
                        
                        # Handle vehicle crossing
                        plate_info = last_plate_for_vehicle.get(vehicle_id, None)
                        if plate_info:
                            px1, py1, px2, py2 = plate_info["plate_box"]
                            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                            cv2.putText(frame, f"Plate: {plate_info['ocr_text']}", (px1, py1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            logging.info(f"Vehicle {vehicle_id} crossed with plate {plate_info['ocr_text']}")
                        else:
                            vehicles_without_plates += 1
                            msg = f"⚠️ ALERT: Vehicle {vehicle_id} crossed without a license plate!"
                            cv2.putText(frame, "NO PLATE DETECTED!", (centroid[0], centroid[1]),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            logging.warning(msg)

        # --- Display ---
        cv2.imshow("Vehicle ANPR", frame)
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

