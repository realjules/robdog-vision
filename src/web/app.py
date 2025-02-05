from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import asyncio
from typing import Optional, Dict, Any

from ..vision.processor import VisionProcessor
from ..robot.controller import RobotController

class RobotVisionApp:
    def __init__(self):
        """Initialize the application components"""
        self.app = FastAPI(title="Robot Vision Demo")
        self.vision_processor = VisionProcessor()
        self.robot_controller = RobotController()
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Setup routes
        self.setup_routes()
        
        # WebSocket connections
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Processing state
        self.is_processing = False
        self.last_depth_frame: Optional[np.ndarray] = None

    def setup_routes(self):
        """Setup FastAPI routes"""
        @self.app.get("/")
        async def get_index():
            with open("static/index.html", "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket(websocket)

        @self.app.post("/reset_emergency")
        async def reset_emergency():
            self.robot_controller.reset_emergency_stop()
            return {"status": "success", "message": "Emergency stop reset"}

        @self.app.get("/status")
        async def get_status():
            return {
                "is_processing": self.is_processing,
                "robot_state": self.robot_controller.get_current_state(),
                "last_update": datetime.now().isoformat()
            }

    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        await websocket.accept()
        client_id = str(id(websocket))
        self.active_connections[client_id] = websocket
        
        try:
            while True:
                # Receive frame from client
                data = await websocket.receive_text()
                
                if not self.is_processing:
                    continue
                
                try:
                    # Process frame
                    frame = self._decode_frame(data)
                    
                    # Generate depth map (simulate with random data for demo)
                    # In real application, this would come from depth camera
                    depth_map = np.random.rand(*frame.shape[:2]) * 5
                    self.last_depth_frame = depth_map
                    
                    # Process image
                    vision_command = self.vision_processor.process_image(
                        frame,
                        depth_map=depth_map
                    )
                    
                    # Apply safety checks and get final command
                    robot_command = self.robot_controller.process_command(
                        vision_command,
                        depth_map
                    )
                    
                    # Send command back to client
                    await websocket.send_json(robot_command)
                    
                except Exception as e:
                    error_command = {
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_json(error_command)
                
        except Exception as e:
            print(f"WebSocket error: {str(e)}")
        finally:
            # Clean up
            del self.active_connections[client_id]
            await websocket.close()

    def _decode_frame(self, data: str) -> np.ndarray:
        """Decode base64 image data to numpy array"""
        try:
            image_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            raise ValueError(f"Error decoding frame: {str(e)}")

    def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the FastAPI application"""
        import uvicorn
        print(f"\nServer running at: http://localhost:{port}")
        uvicorn.run(self.app, host=host, port=port, access_log=False)

# Create application instance
app_instance = RobotVisionApp()
app = app_instance.app

if __name__ == "__main__":
    app_instance.start()