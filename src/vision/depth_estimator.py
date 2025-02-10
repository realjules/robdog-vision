import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from typing import Tuple, Dict, Any
import torch.nn.functional as F

class DepthEstimator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.model.to(self.device)
        
    def estimate_depth(self, image_path: str) -> Dict[str, Any]:
        """
        Estimate depth from an image and return depth information with object distance estimates.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing:
            - depth_map: Normalized depth map
            - object_distances: Dictionary of detected objects and their estimated distances
            - depth_stats: Statistical information about the depth map
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
        
        # Convert to numpy and normalize
        depth_map = prediction.squeeze().cpu().numpy()
        normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        # Calculate depth statistics
        depth_stats = {
            "min_depth": float(depth_map.min()),
            "max_depth": float(depth_map.max()),
            "mean_depth": float(depth_map.mean()),
            "median_depth": float(np.median(depth_map))
        }
        
        # Estimate rough object distances by dividing the image into regions
        h, w = depth_map.shape
        regions = {
            "center": depth_map[h//3:2*h//3, w//3:2*w//3],
            "top": depth_map[:h//3, :],
            "bottom": depth_map[2*h//3:, :],
            "left": depth_map[:, :w//3],
            "right": depth_map[:, 2*w//3:]
        }
        
        object_distances = {
            region: {
                "mean_depth": float(region_depth.mean()),
                "min_depth": float(region_depth.min()),
                "relative_distance": "near" if region_depth.mean() < depth_map.mean() else "far"
            }
            for region, region_depth in regions.items()
        }
        
        return {
            "depth_map": normalized_depth,
            "object_distances": object_distances,
            "depth_stats": depth_stats
        }

    def visualize_depth(self, depth_map: np.ndarray, save_path: str = None) -> np.ndarray:
        """
        Create a color visualization of the depth map
        
        Args:
            depth_map: Normalized depth map
            save_path: Optional path to save the visualization
            
        Returns:
            Colored depth map visualization as numpy array
        """
        colored_depth = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        
        if save_path:
            cv2.imwrite(save_path, colored_depth)
            
        return colored_depth