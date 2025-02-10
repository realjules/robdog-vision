import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import depth_pro
from PIL import Image

class DepthEstimator:
    def __init__(self):
        # Initialize YOLO model
        self.yolo_model = YOLO('yolov8s.pt')
        
        # Initialize Depth Pro model
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.model.to('cuda')
    
    def detect_and_measure(self, image_path):
        # Load and preprocess image
        image, _, f_px = depth_pro.load_rgb(image_path)
        image_tensor = self.transform(image)
        
        # Get depth prediction
        with torch.no_grad():
            prediction = self.model.infer(image_tensor, f_px=f_px)
            depth_map = prediction["depth"].cpu().numpy()  # Depth in meters
            
        # Convert depth map to colored visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Convert PIL image to OpenCV format for YOLO
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect objects with YOLO
        results = self.yolo_model(image_cv)
        
        # Draw bounding boxes and distances
        output_image = image_cv.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Only process person class (class_id = 0 in COCO dataset)
                if box.cls[0].item() == 0:  # person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].item()
                    
                    # Get average depth in the bounding box
                    depth_roi = depth_map[y1:y2, x1:x2]
                    avg_depth = np.mean(depth_roi)
                    
                    # Draw bounding box
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw distance text
                    text = f"{avg_depth:.2f}m"
                    cv2.putText(output_image, text, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return output_image, depth_colored

def main():
    # Initialize depth estimator
    estimator = DepthEstimator()
    
    # Process sample image
    image_path = "sample.jpg"  # Replace with your image path
    output_image, depth_map = estimator.detect_and_measure(image_path)
    
    # Save results
    cv2.imwrite("output_depth.jpg", output_image)
    cv2.imwrite("depth_map.jpg", depth_map)

if __name__ == "__main__":
    main()