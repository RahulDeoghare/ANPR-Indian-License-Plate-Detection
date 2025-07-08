from ultralytics import YOLO
import torch

# Check for GPU availability
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load model
model = YOLO("ANPR2.pt")  # Make sure this file is in the same dir or give full path

# Train model
model.train(
    data="ANPR/data.yaml",   # Update if path is different
    epochs=100,
    imgsz=640,
    batch=16,
    device=device,
    plots=True,
    name="anpr_train"
)
