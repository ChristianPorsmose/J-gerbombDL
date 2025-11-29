import torch
from ultralytics.models import YOLO
from torchvision.io import read_image, write_jpeg
from torchvision.utils import draw_bounding_boxes
import os

def test_image_simple(weights_path, image_path, conf_threshold=0.6, output_path="output.jpg"):
    print(f"Loading model from {weights_path}...")

    # Correct way to load model
    model = YOLO(weights_path)

    print(f"Running inference on {image_path}...")
    
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=conf_threshold,
        save=False,
        verbose=True
    )

    result = results[0]

    print("\n=== Detections ===")
    if len(result.boxes) > 0:
        print(f"Found {len(result.boxes)} objects:")
        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            print(f"  Object {i+1}: Class {cls}, Conf: {conf:.3f}, Box: {xyxy}")

        annotated = result.plot()

        import cv2
        cv2.imwrite(output_path, annotated)
        print(f"Saved: {output_path}")

    else:
        print("No detections!")

    return result

if __name__ == "__main__":
    # Configuration
    WEIGHTS_PATH = "weights.pt"
    IMAGE_PATH = "../dataset_final_boxes_yolo/images/Adrian_20251107_094907.jpg"
    CONFIDENCE_THRESHOLD = 0.6
    OUTPUT_PATH = "detection_result.jpg"
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        exit(1)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}")
        exit(1)
    
    result = test_image_simple(
        weights_path=WEIGHTS_PATH,
        image_path=IMAGE_PATH,
        conf_threshold=CONFIDENCE_THRESHOLD,
        output_path=OUTPUT_PATH
    )