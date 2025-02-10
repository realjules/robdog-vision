from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class DepthAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model and processor
        print("Loading DepthAnything model...")
        self.processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.model.to(self.device)
        print("Model loaded successfully")

    def analyze_image(self, image_path):
        """
        Analyze an image and return depth information.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            dict: Dictionary containing depth analysis results
        """
        # Load and process image
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        
        # Get depth prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        # Convert to numpy
        depth_map = prediction.squeeze().cpu().numpy()
        
        # Normalize depth map for visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Analyze regions
        h, w = depth_map.shape
        regions = {
            "center": depth_map[h//3:2*h//3, w//3:2*w//3],
            "top": depth_map[:h//3, :],
            "bottom": depth_map[2*h//3:, :],
            "left": depth_map[:, :w//3],
            "right": depth_map[:, 2*w//3:]
        }
        
        # Calculate statistics for each region
        region_stats = {}
        for region_name, region_data in regions.items():
            stats = {
                "mean_depth": float(region_data.mean()),
                "min_depth": float(region_data.min()),
                "max_depth": float(region_data.max()),
                "relative_distance": "near" if region_data.mean() < depth_map.mean() else "far"
            }
            region_stats[region_name] = stats
        
        # Find potential objects (areas with significant depth differences)
        gradient_y = np.gradient(depth_map, axis=0)
        gradient_x = np.gradient(depth_map, axis=1)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Threshold for object detection
        threshold = np.percentile(gradient_magnitude, 90)
        object_mask = gradient_magnitude > threshold
        
        # Find connected components (potential objects)
        num_labels, labels = cv2.connectedComponents(object_mask.astype(np.uint8))
        
        # Analyze each potential object
        objects = []
        min_object_size = 100  # Minimum size to be considered an object
        
        for label in range(1, num_labels):  # Skip background (label 0)
            object_mask = labels == label
            if object_mask.sum() < min_object_size:
                continue
                
            # Get object properties
            y_indices, x_indices = np.where(object_mask)
            object_depth = depth_map[object_mask]
            
            object_info = {
                "position": {
                    "x": float(x_indices.mean() / w),  # Normalized x position (0-1)
                    "y": float(y_indices.mean() / h)   # Normalized y position (0-1)
                },
                "size": {
                    "width": float((x_indices.max() - x_indices.min()) / w),
                    "height": float((y_indices.max() - y_indices.min()) / h)
                },
                "depth": {
                    "mean": float(object_depth.mean()),
                    "relative": "near" if object_depth.mean() < depth_map.mean() else "far"
                }
            }
            objects.append(object_info)
        
        # Save visualizations
        plt.figure(figsize=(15, 5))
        
        # Original depth map
        plt.subplot(131)
        plt.imshow(normalized_depth, cmap='magma')
        plt.title('Depth Map')
        plt.colorbar(label='Normalized Depth')
        
        # Region visualization
        plt.subplot(132)
        region_vis = np.zeros_like(depth_map)
        for i, (name, data) in enumerate(regions.items()):
            mask = np.zeros_like(depth_map, dtype=bool)
            if name == "center":
                mask[h//3:2*h//3, w//3:2*w//3] = True
            elif name == "top":
                mask[:h//3, :] = True
            elif name == "bottom":
                mask[2*h//3:, :] = True
            elif name == "left":
                mask[:, :w//3] = True
            else:  # right
                mask[:, 2*w//3:] = True
            region_vis[mask] = i + 1
        plt.imshow(region_vis, cmap='tab10')
        plt.title('Regions')
        
        # Object visualization
        plt.subplot(133)
        plt.imshow(gradient_magnitude > threshold, cmap='gray')
        plt.title('Detected Objects')
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig('depth_analysis.png')
        plt.close()
        
        return {
            "depth_map": normalized_depth.tolist(),
            "region_analysis": region_stats,
            "detected_objects": objects,
            "visualization_path": "depth_analysis.png"
        }

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python depth_analyzer.py <image_path>")
        sys.exit(1)
        
    analyzer = DepthAnalyzer()
    results = analyzer.analyze_image(sys.argv[1])
    
    # Print results in a readable format
    print("\nRegion Analysis:")
    for region, stats in results["region_analysis"].items():
        print(f"\n{region.upper()}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print("\nDetected Objects:")
    for i, obj in enumerate(results["detected_objects"], 1):
        print(f"\nObject {i}:")
        print(f"  Position: x={obj['position']['x']:.2f}, y={obj['position']['y']:.2f}")
        print(f"  Size: {obj['size']['width']:.2f}x{obj['size']['height']:.2f}")
        print(f"  Depth: {obj['depth']['mean']:.2f} ({obj['depth']['relative']})")
    
    print(f"\nVisualization saved to: {results['visualization_path']}")