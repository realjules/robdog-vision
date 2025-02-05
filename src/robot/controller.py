from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime
import json

class RobotController:
    def __init__(self):
        """Initialize robot controller with safety parameters"""
        self.safety_limits = {
            "max_linear_velocity": 1.0,    # m/s
            "max_angular_velocity": 1.0,    # rad/s
            "min_obstacle_distance": 0.5,   # meters
            "emergency_stop_distance": 0.3  # meters
        }
        
        self.gait_modes = {
            "trotting": {
                "max_velocity": 1.0,
                "description": "Standard trotting gait"
            },
            "walking": {
                "max_velocity": 0.5,
                "description": "Stable walking gait"
            },
            "standing": {
                "max_velocity": 0.0,
                "description": "Stationary stance"
            },
            "jumping": {
                "max_velocity": 0.8,
                "description": "Dynamic jumping gait"
            }
        }
        
        self.current_state = {
            "velocity": {"linear": 0.0, "angular": 0.0},
            "gait_mode": "standing",
            "last_command_time": datetime.now(),
            "is_emergency_stopped": False
        }

    def process_command(self, command: Dict[str, Any], depth_info: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process and validate a command before sending to robot"""
        try:
            # Check if we're in emergency stop
            if self.current_state["is_emergency_stopped"]:
                return self._emergency_stop_command("Emergency stop active")
            
            # Validate command structure
            if not self._validate_command_structure(command):
                return self._emergency_stop_command("Invalid command structure")
            
            # Check depth information if available
            if depth_info is not None:
                if not self._check_safety_distances(depth_info):
                    return self._emergency_stop_command("Obstacle too close")
            
            # Apply safety limits
            safe_command = self._apply_safety_limits(command)
            
            # Update current state
            self._update_state(safe_command)
            
            # Add timestamp
            safe_command["timestamp"] = datetime.now().isoformat()
            
            return safe_command
            
        except Exception as e:
            return self._emergency_stop_command(f"Error: {str(e)}")

    def _validate_command_structure(self, command: Dict[str, Any]) -> bool:
        """Validate the command has all required fields"""
        required_fields = {
            "velocity_command": ["linear_velocity_mps", "angular_velocity_radps"],
            "gait_mode": None
        }
        
        try:
            for field, subfields in required_fields.items():
                if field not in command:
                    return False
                if subfields:
                    if not all(sf in command[field] for sf in subfields):
                        return False
            
            if command["gait_mode"] not in self.gait_modes:
                return False
                
            return True
        except:
            return False

    def _check_safety_distances(self, depth_map: np.ndarray) -> bool:
        """Check if there are any obstacles too close"""
        min_distance = np.min(depth_map[depth_map > 0])  # Ignore zero values
        
        if min_distance < self.safety_limits["emergency_stop_distance"]:
            self.current_state["is_emergency_stopped"] = True
            return False
            
        return min_distance >= self.safety_limits["min_obstacle_distance"]

    def _apply_safety_limits(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Apply safety limits to command velocities"""
        safe_command = command.copy()
        
        # Get maximum velocity for current gait
        max_vel = self.gait_modes[command["gait_mode"]]["max_velocity"]
        
        # Limit velocities
        safe_command["velocity_command"]["linear_velocity_mps"] = np.clip(
            command["velocity_command"]["linear_velocity_mps"],
            -max_vel,
            max_vel
        )
        
        safe_command["velocity_command"]["angular_velocity_radps"] = np.clip(
            command["velocity_command"]["angular_velocity_radps"],
            -self.safety_limits["max_angular_velocity"],
            self.safety_limits["max_angular_velocity"]
        )
        
        return safe_command

    def _update_state(self, command: Dict[str, Any]):
        """Update current robot state"""
        self.current_state.update({
            "velocity": {
                "linear": command["velocity_command"]["linear_velocity_mps"],
                "angular": command["velocity_command"]["angular_velocity_radps"]
            },
            "gait_mode": command["gait_mode"],
            "last_command_time": datetime.now()
        })

    def _emergency_stop_command(self, reason: str) -> Dict[str, Any]:
        """Generate emergency stop command"""
        return {
            "velocity_command": {
                "linear_velocity_mps": 0.0,
                "angular_velocity_radps": 0.0
            },
            "gait_mode": "standing",
            "reasoning": f"EMERGENCY STOP: {reason}",
            "timestamp": datetime.now().isoformat()
        }

    def reset_emergency_stop(self):
        """Reset emergency stop state"""
        self.current_state["is_emergency_stopped"] = False

    def get_current_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        return self.current_state