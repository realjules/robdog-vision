import cv2
import numpy as np
from ultralytics import YOLO

class PersonDetector:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('yolov8s.pt')
    
    def detect_people(self, image_path):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Detect objects
        results = self.model(image)
        
        # Draw bounding boxes
        output_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Only process person class (class_id = 0 in COCO dataset)
                if box.cls[0].item() == 0:  # person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].item()
                    
                    # Draw bounding box
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw confidence text
                    text = f"Person {conf:.2f}"
                    cv2.putText(output_image, text, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return output_image

def main():
    # Initialize detector
    detector = PersonDetector()
    
    # Process image
    output_image = detector.detect_people('test_image.jpg')
    
    # Save result
    cv2.imwrite('output_yolo.jpg', output_image)
    print("Result saved as 'output_yolo.jpg'")

if __name__ == "__main__":
    main()