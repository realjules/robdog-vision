import sys
import os
import cv2
import numpy as np
import torch
import pytest
from fastapi.testclient import TestClient
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vision.processor import VisionProcessor
from src.robot.controller import RobotController
from src.web.app import app

class TestSystem:
    @pytest.fixture
    def vision_processor(self):
        return VisionProcessor()
    
    @pytest.fixture
    def robot_controller(self):
        return RobotController()
    
    @pytest.fixture
    def test_client(self):
        return TestClient(app)
    
    @pytest.fixture
    def test_image(self):
        # Create a test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)
        return img
    
    @pytest.fixture
    def test_depth_map(self):
        # Create a test depth map
        depth = np.ones((480, 640), dtype=np.float32) * 2.0
        depth[100:200, 100:200] = 0.5  # Simulate closer object
        return depth

    def test_vision_processor(self, vision_processor, test_image):
        """Test vision processing"""
        command = vision_processor.process_image(test_image)
        
        assert isinstance(command, dict)
        assert "velocity_command" in command
        assert "gait_mode" in command
        
        vel_cmd = command["velocity_command"]
        assert -1.0 <= vel_cmd["linear_velocity_mps"] <= 1.0
        assert -1.0 <= vel_cmd["angular_velocity_radps"] <= 1.0

    def test_robot_controller(self, robot_controller):
        """Test robot controller"""
        test_command = {
            "velocity_command": {
                "linear_velocity_mps": 0.5,
                "angular_velocity_radps": 0.2
            },
            "gait_mode": "trotting"
        }
        
        # Test normal command
        result = robot_controller.process_command(test_command)
        assert result["velocity_command"]["linear_velocity_mps"] <= 1.0
        assert result["gait_mode"] == "trotting"
        
        # Test emergency stop
        robot_controller.current_state["is_emergency_stopped"] = True
        result = robot_controller.process_command(test_command)
        assert result["velocity_command"]["linear_velocity_mps"] == 0.0
        assert result["gait_mode"] == "standing"

    def test_safety_limits(self, robot_controller, test_depth_map):
        """Test safety features"""
        test_command = {
            "velocity_command": {
                "linear_velocity_mps": 2.0,  # Over limit
                "angular_velocity_radps": 1.5  # Over limit
            },
            "gait_mode": "trotting"
        }
        
        result = robot_controller.process_command(test_command, test_depth_map)
        
        # Check velocity limiting
        assert result["velocity_command"]["linear_velocity_mps"] <= 1.0
        assert result["velocity_command"]["angular_velocity_radps"] <= 1.0
        
        # Test with close obstacle
        close_depth = np.ones_like(test_depth_map) * 0.2  # Very close obstacle
        result = robot_controller.process_command(test_command, close_depth)
        assert result["velocity_command"]["linear_velocity_mps"] == 0.0
        assert result["gait_mode"] == "standing"

    def test_web_interface(self, test_client):
        """Test web interface"""
        # Test main page
        response = test_client.get("/")
        assert response.status_code == 200
        
        # Test status endpoint
        response = test_client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "is_processing" in data
        assert "robot_state" in data

    def test_full_pipeline(self, vision_processor, robot_controller, test_image, test_depth_map):
        """Test complete processing pipeline"""
        # Process image
        vision_command = vision_processor.process_image(test_image, depth_map=test_depth_map)
        
        # Process command through robot controller
        final_command = robot_controller.process_command(vision_command, test_depth_map)
        
        # Validate final command
        assert isinstance(final_command, dict)
        assert "velocity_command" in final_command
        assert "gait_mode" in final_command
        assert "timestamp" in final_command
        
        # Check velocity limits
        vel_cmd = final_command["velocity_command"]
        assert -1.0 <= vel_cmd["linear_velocity_mps"] <= 1.0
        assert -1.0 <= vel_cmd["angular_velocity_radps"] <= 1.0
        
        # Check gait mode
        assert final_command["gait_mode"] in ["trotting", "walking", "standing", "jumping"]

if __name__ == "__main__":
    pytest.main([__file__])