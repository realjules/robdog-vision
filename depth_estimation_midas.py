import cv2
import torch
import numpy as np
from ultralytics import YOLO

class DepthEstimator:
    def __init__(self):
        # Initialize YOLO model
        self.yolo_model = YOLO('yolov8s.pt')
        
        # Initialize MiDaS model
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.eval()
        
        # MiDaS transform
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.small_transform
        
        if torch.cuda.is_available():
            self.midas.to('cuda')
    
    def detect_and_measure(self, image_path):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Transform image for MiDaS
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img)
        
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        
        # Get depth prediction
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map for visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Detect objects with YOLO
        results = self.yolo_model(image)
        
        # Draw bounding boxes and distances
        output_image = image.copy()
        
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
                    text = f"Depth: {avg_depth:.2f}"
                    cv2.putText(output_image, text, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return output_image, depth_colored

def main():
    # Initialize depth estimator
    estimator = DepthEstimator()
    
    # Process image
    output_image, depth_map = estimator.detect_and_measure('test_image.jpg')
    
    # Save results
    cv2.imwrite('output_depth.jpg', output_image)
    cv2.imwrite('depth_map.jpg', depth_map)
    print("Results saved as 'output_depth.jpg' and 'depth_map.jpg'")

if __name__ == "__main__":
    main()