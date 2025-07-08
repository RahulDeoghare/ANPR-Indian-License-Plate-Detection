import cv2
import torch
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import json

def is_valid_indian_plate(plate_text):
    """
    Enhanced validation for Indian number plates with better error tolerance.
    """
    if not plate_text or len(plate_text) < 8:
        return False
    
    # Remove spaces and convert to uppercase
    clean_plate = plate_text.replace(" ", "").upper()
    
    # Allow slight length variations (9-11 characters for flexibility)
    if len(clean_plate) < 9 or len(clean_plate) > 11:
        return False
    
    # Enhanced patterns with more flexibility
    # Pattern 1: Standard format - 2 letters + 2 digits + 2 letters + 4 digits
    pattern1 = r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4}$'
    
    # Pattern 2: BH Series format - 2 digits + BH + 4 digits + 2 letters
    pattern2 = r'^\d{2}BH\d{4}[A-Z]{2}$'
    
    # Pattern 3: New format variations
    pattern3 = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'  # Exact 10 char
    
    if re.match(pattern1, clean_plate) or re.match(pattern2, clean_plate) or re.match(pattern3, clean_plate):
        # Extended state codes list
        state_codes = [
            'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CT', 'DN', 'DD', 'DL', 'GA', 'GJ', 'HR', 'HP', 'JH', 'JK',
            'KA', 'KL', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TG',
            'TR', 'UP', 'UT', 'WB', 'LA', 'PY'
        ]
        
        # Check first two characters for state code or BH series
        if clean_plate.startswith(tuple(state_codes)) or 'BH' in clean_plate[:4]:
            return True
    
    return False

def format_indian_plate(plate_text):
    """
    Format the plate text in standard Indian format with spaces.
    """
    clean_plate = plate_text.replace(" ", "").upper()
    
    if len(clean_plate) == 10:
        if clean_plate[2:4] == 'BH':
            # BH series: 22 BH 1234 AB
            return f"{clean_plate[:2]} {clean_plate[2:4]} {clean_plate[4:8]} {clean_plate[8:]}"
        else:
            # Standard: MH 15 FV 8808
            return f"{clean_plate[:2]} {clean_plate[2:4]} {clean_plate[4:6]} {clean_plate[6:]}"
    
    return plate_text

# Load YOLOv8 model (custom model trained to detect number plates)
# Enable GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")
model = YOLO("truck.pt")
model.to(device)  # Move model to GPU if available

# Load PaddleOCR with better configuration
try:
    # Try to enable GPU for PaddleOCR if available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        try:
            ocr = PaddleOCR(use_textline_orientation=True, lang='en', use_gpu=True)
            print(f"[INFO] PaddleOCR initialized successfully with GPU: True")
        except:
            # Fallback to CPU if GPU initialization fails
            ocr = PaddleOCR(use_textline_orientation=True, lang='en')
            print(f"[INFO] PaddleOCR initialized successfully with GPU: False (fallback to CPU)")
    else:
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        print(f"[INFO] PaddleOCR initialized successfully with GPU: False")
except Exception as e:
    print(f"[ERROR] Failed to initialize PaddleOCR: {e}")
    exit(1)

# Initialize DeepSORT
tracker = DeepSort(max_age=30)

# Store results
results_dict = {}

# Create debug folder for plate images
debug_folder = "debug_plates"
os.makedirs(debug_folder, exist_ok=True)

# Video input and output
video_path = "vms_test3.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_annotated.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Print progress every 100 frames
    if frame_count % 100 == 0:
        print(f"[PROGRESS] Processing frame {frame_count}...")

    # Run YOLO detection with GPU acceleration
    results = model.predict(frame, verbose=False, device=device, half=False, imgsz=640)[0]

    detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))  # format for DeepSORT

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltwh()
        x1, y1, x2, y2 = int(l), int(t), int(l + w), int(t + h)

        # Skip very small plates
        if x2 - x1 < 30 or y2 - y1 < 15:
            continue

        # Expand bounding box more aggressively to capture complete license plate
        expansion = 20  # pixels to expand on each side (increased from 10)
        x1_expanded = max(0, x1 - expansion)
        y1_expanded = max(0, y1 - expansion) 
        x2_expanded = min(frame.shape[1], x2 + expansion)
        y2_expanded = min(frame.shape[0], y2 + expansion)

        # OCR optimization: limit attempts and only OCR when necessary
        if track_id not in results_dict:
            results_dict[track_id] = {
                "all_detections": [],  # Store all OCR attempts
                "best_plate_text": "",
                "formatted_plate_text": "",
                "best_confidence": 0,
                "frame_detected": frame_count,
                "bbox": [x1_expanded, y1_expanded, x2_expanded, y2_expanded],
                "method": "enhanced_processing",
                "last_ocr_frame": 0
            }
        
        # Significantly reduce OCR attempts for speed
        should_ocr = (
            len(results_dict[track_id]["all_detections"]) < 3 and  # Max 3 attempts only
            (frame_count - results_dict[track_id]["last_ocr_frame"]) > 10 and  # Wait 10 frames between attempts
            (results_dict[track_id]["best_confidence"] < 0.85 or not is_valid_indian_plate(results_dict[track_id]["best_plate_text"]))
        )
        
        if should_ocr:
            results_dict[track_id]["last_ocr_frame"] = frame_count
            plate_img = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
            
            # Simplified processing - only use the best method
            if plate_img.shape[0] > 0 and plate_img.shape[1] > 0:
                # Scale up for better OCR
                scale_factor = max(3.0, 100.0 / min(plate_img.shape[:2]))
                new_width = int(plate_img.shape[1] * scale_factor)
                new_height = int(plate_img.shape[0] * scale_factor)
                plate_resized = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                # Convert to grayscale and enhance
                gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                # Simple sharpening
                kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
                processed_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
                
                # Save debug image
                debug_path = os.path.join(debug_folder, f"plate_{track_id}_{frame_count}.jpg")
                cv2.imwrite(debug_path, processed_img)
            else:
                processed_img = plate_img

            # Single OCR attempt per frame
            try:
                ocr_result = ocr.ocr(processed_img)
                if ocr_result and len(ocr_result) > 0:
                    result = ocr_result[0]  # Get first result
                    
                    best_text = ""
                    best_confidence = 0
                    
                    # Handle different PaddleOCR output formats
                    if isinstance(result, list):
                        # Legacy format: list of [bbox, (text, confidence)]
                        for item in result:
                            if len(item) == 2 and isinstance(item[1], tuple) and len(item[1]) == 2:
                                text, confidence = item[1]
                                cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                                
                                # Only accept if it's a valid Indian plate format
                                if confidence > best_confidence and len(cleaned_text) >= 8 and is_valid_indian_plate(cleaned_text):
                                    best_text = cleaned_text
                                    best_confidence = confidence
                    
                    elif isinstance(result, dict) and 'rec_texts' in result and 'rec_scores' in result:
                        # New dictionary format
                        texts = result['rec_texts']
                        scores = result['rec_scores']
                        
                        if texts and scores:
                            # Try individual texts
                            for text, confidence in zip(texts, scores):
                                cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                                
                                # Only accept if it's a valid Indian plate format
                                if confidence > best_confidence and len(cleaned_text) >= 8 and is_valid_indian_plate(cleaned_text):
                                    best_text = cleaned_text
                                    best_confidence = confidence
                            
                            # Try merged text
                            merged_text = ''.join(''.join(c for c in text if c.isalnum()).upper() for text in texts)
                            avg_confidence = sum(scores) / len(scores)
                            
                            # Only accept if it's a valid Indian plate format
                            if len(merged_text) >= 8 and avg_confidence > best_confidence and is_valid_indian_plate(merged_text):
                                best_text = merged_text
                                best_confidence = avg_confidence
                    
                    # Only proceed if we found a valid Indian plate
                    if best_text and is_valid_indian_plate(best_text):
                        formatted_plate = format_indian_plate(best_text)
                        print(f"[OCR] ID {track_id}: '{formatted_plate}' (conf: {best_confidence:.3f}) - VALID INDIAN PLATE")
                        
                        # Store this detection attempt
                        current_detection = {
                            "text": best_text,
                            "formatted_text": formatted_plate,
                            "confidence": best_confidence,
                            "frame": frame_count
                        }
                        results_dict[track_id]["all_detections"].append(current_detection)
                        
                        # Update best result if this is better
                        if best_confidence > results_dict[track_id]["best_confidence"]:
                            results_dict[track_id]["best_plate_text"] = best_text
                            results_dict[track_id]["formatted_plate_text"] = formatted_plate
                            results_dict[track_id]["best_confidence"] = best_confidence
                            results_dict[track_id]["frame_detected"] = frame_count
                            results_dict[track_id]["bbox"] = [x1_expanded, y1_expanded, x2_expanded, y2_expanded]
                            print(f"[OCR] New best for ID {track_id}: {formatted_plate} (conf: {best_confidence:.3f})")
                    else:
                        # Log rejected detections that don't match Indian format
                        if result and isinstance(result, list) and len(result) > 0:
                            for item in result:
                                if len(item) == 2 and isinstance(item[1], tuple) and len(item[1]) == 2:
                                    text, confidence = item[1]
                                    cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                                    if cleaned_text:
                                        print(f"[OCR] ID {track_id}: '{cleaned_text}' - REJECTED (not valid Indian plate format)")
                        
                    # Simple combination logic - only try if we have 2+ detections
                    if len(results_dict[track_id]["all_detections"]) >= 2:
                        all_texts = [d["text"] for d in results_dict[track_id]["all_detections"] if len(d["text"]) >= 8]
                        
                        # Look for complementary parts
                        for i, text1 in enumerate(all_texts):
                            for j, text2 in enumerate(all_texts):
                                if i != j and len(text1) + len(text2) <= 12:
                                    combo = text1 + text2
                                    # Check if combination is valid Indian plate
                                    if is_valid_indian_plate(combo):
                                        conf1 = next((d["confidence"] for d in results_dict[track_id]["all_detections"] if d["text"] == text1), 0)
                                        conf2 = next((d["confidence"] for d in results_dict[track_id]["all_detections"] if d["text"] == text2), 0)
                                        combined_conf = (conf1 + conf2) / 2
                                        
                                        if combined_conf > results_dict[track_id]["best_confidence"]:
                                            formatted_combo = format_indian_plate(combo)
                                            results_dict[track_id]["best_plate_text"] = combo
                                            results_dict[track_id]["formatted_plate_text"] = formatted_combo
                                            results_dict[track_id]["best_confidence"] = combined_conf
                                            print(f"[OCR] Combined valid Indian plate for ID {track_id}: {formatted_combo} (conf: {combined_conf:.3f})")
                                            break
            
            except Exception as e:
                print(f"[OCR ERROR] ID {track_id}: {e}")
                
        else:
            if results_dict[track_id]["best_plate_text"] and is_valid_indian_plate(results_dict[track_id]["best_plate_text"]):
                formatted_text = results_dict[track_id].get("formatted_plate_text", format_indian_plate(results_dict[track_id]["best_plate_text"]))
                print(f"[OCR] Skipping OCR for ID {track_id} - already have valid Indian plate: {formatted_text}")

        # Draw boxes and overlay info
        cv2.rectangle(frame, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)
        label = f"ID: {track_id}"

        if track_id in results_dict and results_dict[track_id]["best_plate_text"] and is_valid_indian_plate(results_dict[track_id]["best_plate_text"]):
            # Display formatted Indian plate
            formatted_text = results_dict[track_id].get("formatted_plate_text", format_indian_plate(results_dict[track_id]["best_plate_text"]))
            label += f" | {formatted_text}"

        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame with annotations
    cv2.imshow('License Plate Detection - Press Q to quit', frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Stopping processing...")
        break

    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()  # Close all OpenCV windows

# Save final JSON results with simplified format - only valid Indian plates
final_results = {}
for tid, info in results_dict.items():
    if info["best_plate_text"] and is_valid_indian_plate(info["best_plate_text"]):  # Only include valid Indian plates
        final_results[tid] = {
            "plate_text": info["best_plate_text"],
            "formatted_plate_text": info.get("formatted_plate_text", format_indian_plate(info["best_plate_text"])),
            "frame_detected": info["frame_detected"],
            "bbox": info["bbox"],
            "confidence": info["best_confidence"],
            "method": info.get("method", "enhanced_processing"),
            "total_attempts": len(info["all_detections"])
        }

with open("detected_plates.json", "w") as f:
    json.dump(final_results, f, indent=4)

print("Detection complete. Valid Indian Plates:")
for tid, info in final_results.items():
    # Add color formatting for better visibility
    print(f"\033[92m✓ Vehicle {tid}: Plate = \033[1m{info['formatted_plate_text']}\033[0m\033[92m, Frame = {info['frame_detected']}, Confidence = {info['confidence']:.3f}, Method = {info['method']}, Attempts = {info['total_attempts']}\033[0m")

# Add summary with colors
print(f"\n\033[94m{'='*50}")
print(f"📊 SUMMARY: {len(final_results)} valid Indian license plates detected")
print(f"{'='*50}\033[0m")

