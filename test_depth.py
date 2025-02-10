import os
import cv2
from depth_estimation import DepthEstimator

def test_depth_estimation():
    # Initialize depth estimator
    estimator = DepthEstimator()
    
    # Get absolute path to test image
    image_path = os.path.abspath('test_image.jpg')
    if not os.path.exists(image_path):
        print(f"Please provide a test image at {image_path}")
        return
    
    # Process image
    output_image, depth_map = estimator.detect_and_measure(image_path)
    
    # Save results
    cv2.imwrite('output_depth.jpg', output_image)
    cv2.imwrite('depth_map.jpg', depth_map)
    print("Results saved as 'output_depth.jpg' and 'depth_map.jpg'")

if __name__ == "__main__":
    test_depth_estimation()