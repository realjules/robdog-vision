from transformers import AutoFeatureExtractor, ViTForImageClassification
import torch
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any, Optional, Tuple

class VisionProcessor:
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """Initialize the vision processor with ViT model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        # Define standard prompts
        self.prompts = {
            "navigation": """Analyze this scene for robot navigation:
            1. Identify objects and their approximate distances
            2. Find safe paths for navigation
            3. Note any obstacles or hazards
            4. Consider the best approach path
            
            Generate a JSON command with:
            - Linear velocity (-1.0 to 1.0 m/s)
            - Angular velocity (-1.0 to 1.0 rad/s)
            - Appropriate gait mode
            - Detailed reasoning for the decision""",
            
            "object_detection": """Identify and locate objects in the scene:
            1. List visible objects
            2. Estimate distances
            3. Note spatial relationships
            4. Identify potential obstacles""",
            
            "safety_check": """Analyze the scene for safety:
            1. Identify any hazards
            2. Check for moving objects
            3. Verify clear paths
            4. Note any unsafe conditions"""
        }

    def process_image(
        self,
        image: np.ndarray,
        prompt_type: str = "navigation",
        custom_prompt: Optional[str] = None,
        depth_map: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Process an image and generate navigation commands"""
        try:
            # Convert CV2 image to PIL
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Prepare prompt
            prompt = custom_prompt if custom_prompt else self.prompts[prompt_type]
            
            # Add depth information if available
            if depth_map is not None:
                depth_info = self._analyze_depth(depth_map)
                prompt += f"\n\nDepth information:\n{depth_info}"
            
            # Process inputs
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_label = logits.argmax(-1).item()
            
            # Convert prediction to navigation command
            if predicted_label in [0, 1, 2]:  # If object detected ahead
                command = {
                    "velocity_command": {
                        "linear_velocity_mps": 0.0,
                        "angular_velocity_radps": 0.5  # Turn right
                    },
                    "gait_mode": "walking",
                    "reasoning": "Object detected ahead, turning right to avoid"
                }
            else:
                command = {
                    "velocity_command": {
                        "linear_velocity_mps": 0.5,  # Move forward
                        "angular_velocity_radps": 0.0
                    },
                    "gait_mode": "trotting",
                    "reasoning": "Path appears clear, moving forward"
                }
            
            return command
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return self._get_safe_command(f"Error: {str(e)}")

    def _analyze_depth(self, depth_map: np.ndarray) -> str:
        """Analyze depth map for navigation"""
        h, w = depth_map.shape
        regions = {
            "center": depth_map[h//3:2*h//3, w//3:2*w//3].mean(),
            "left": depth_map[h//3:2*h//3, :w//3].mean(),
            "right": depth_map[h//3:2*h//3, 2*w//3:].mean(),
            "min": depth_map.min(),
            "max": depth_map.max()
        }
        
        return f"""
        Average depths:
        - Center: {regions['center']:.2f}m
        - Left: {regions['left']:.2f}m
        - Right: {regions['right']:.2f}m
        - Closest point: {regions['min']:.2f}m
        - Furthest point: {regions['max']:.2f}m
        """

    def _parse_command(self, response: str) -> Dict[str, Any]:
        """Parse and validate the model's response"""
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            command = eval(response[start_idx:end_idx])
            
            # Validate command structure
            required_fields = ['velocity_command', 'gait_mode']
            if not all(field in command for field in required_fields):
                raise ValueError("Missing required fields in command")
            
            # Validate and clip velocities
            command['velocity_command']['linear_velocity_mps'] = np.clip(
                command['velocity_command']['linear_velocity_mps'], -1.0, 1.0
            )
            command['velocity_command']['angular_velocity_radps'] = np.clip(
                command['velocity_command']['angular_velocity_radps'], -1.0, 1.0
            )
            
            # Validate gait mode
            valid_gaits = ["trotting", "walking", "standing", "jumping"]
            if command['gait_mode'] not in valid_gaits:
                command['gait_mode'] = "standing"
            
            return command
            
        except Exception as e:
            print(f"Error parsing command: {str(e)}")
            return self._get_safe_command(f"Parse error: {str(e)}")

    def _get_safe_command(self, reason: str) -> Dict[str, Any]:
        """Return a safe stop command"""
        return {
            "velocity_command": {
                "linear_velocity_mps": 0.0,
                "angular_velocity_radps": 0.0
            },
            "gait_mode": "standing",
            "reasoning": f"Safety stop: {reason}"
        }