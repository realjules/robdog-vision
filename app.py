import cv2
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from starlette.websockets import WebSocketState
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import datetime
from PIL import Image
import io
import base64
import logging
import os
# No need for transformers import as we use depth_estimator directly

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Static files mounted successfully")
except Exception as e:
    logger.error(f"Error mounting static files: {str(e)}")

class RobotVisionSystem:
    def __init__(self):
        from src.vision.depth_estimator import DepthEstimator
        self.depth_estimator = DepthEstimator()
        logger.info("Depth estimation model initialized")

    def process_image(self, image, query=None):
        try:
            logger.info("Processing image...")
            # Convert CV2 image to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                logger.info("Image converted to PIL format")
            
            # Save temporary image for depth estimation
            temp_path = "temp_image.jpg"
            image.save(temp_path)
            
            # Get depth estimation
            results = self.depth_estimator.estimate_depth(temp_path)
            logger.info("Depth estimation completed")
            
            # Generate visualization
            depth_vis_path = os.path.join("static", "depth_output.png")
            self.depth_estimator.visualize_depth(results['depth_map'], depth_vis_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Convert depth information to navigation command
            center_depth = results['object_distances']['center']
            
            if center_depth['relative_distance'] == 'near':
                command = {
                    "velocity_command": {
                        "linear_velocity_mps": 0.0,
                        "angular_velocity_radps": 0.5  # Turn right
                    },
                    "gait_mode": "walking",
                    "reasoning": f"Object detected ahead at depth {center_depth['mean_depth']:.2f}, turning right to avoid"
                }
            else:
                command = {
                    "velocity_command": {
                        "linear_velocity_mps": 0.5,  # Move forward
                        "angular_velocity_radps": 0.0
                    },
                    "gait_mode": "trotting",
                    "reasoning": f"Path clear ahead, mean depth {center_depth['mean_depth']:.2f}"
                }
            
            # Add depth information and timestamp
            command['depth_info'] = results
            command['timestamp'] = datetime.utcnow().isoformat() + "Z"
            logger.info(f"Generated command: {command}")
            return command
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

# Initialize vision system
vision_system = RobotVisionSystem()

@app.get("/")
async def get_index():
    try:
        logger.info("Attempting to read index.html")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(current_dir, "static", "index.html")
        logger.info(f"Looking for index.html at: {index_path}")
        
        if not os.path.exists(index_path):
            logger.error(f"index.html not found at {index_path}")
            return HTMLResponse(
                content="<h1>Error: Page not found</h1><p>index.html is missing from static directory</p>",
                status_code=404
            )
        
        with open(index_path, "r") as f:
            html_content = f.read()
            logger.info("Successfully read index.html")
            return HTMLResponse(content=html_content, status_code=200)
            
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return HTMLResponse(
            content=f"<h1>Internal Server Error</h1><p>Error: {str(e)}</p>",
            status_code=500
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_info = f"{websocket.client.host}:{websocket.client.port}"
    logger.info("New WebSocket connection from %s", client_info)
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted for %s", client_info)
    except Exception as e:
        logger.error(f"Error accepting WebSocket connection from {client_info}: {str(e)}")
        return
    
    try:
        while True:
            try:
                # Receive base64 image from client
                logger.info("Waiting for image data from %s...", client_info)
                data = await websocket.receive_text()
                logger.info(f"Received image data from {client_info}: {data[:100]}...")
                
                # Decode base64 image
                try:
                    image_data = base64.b64decode(data.split(',')[1])
                    logger.info("Base64 image decoded from %s", client_info)
                except Exception as e:
                    logger.error(f"Error decoding base64 image from {client_info}: {str(e)}")
                    continue
                
                # Convert to CV2 image
                try:
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is None:
                        logger.error(f"Failed to decode image from {client_info}")
                        continue
                    logger.info("Image converted to CV2 format from %s", client_info)
                except Exception as e:
                    logger.error(f"Error converting image from {client_info}: {str(e)}")
                    continue
                
                # Process image
                try:
                    command = vision_system.process_image(image)
                    logger.info(f"Generated command for {client_info}: {command}")
                except Exception as e:
                    logger.error(f"Error processing image from {client_info}: {str(e)}")
                    continue
                
                # Send response
                try:
                    await websocket.send_json(command)
                    logger.info("Command sent to client %s", client_info)
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected while sending command to %s", client_info)
                    break
                except Exception as e:
                    logger.error(f"Error sending command to {client_info}: {str(e)}")
                    logger.exception(e)
                    break
                    
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected from %s", client_info)
                break
            except Exception as e:
                logger.error(f"Error in main WebSocket loop for {client_info}: {str(e)}")
                break
    finally:
        logger.info("WebSocket connection closed for %s", client_info)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    print("\nServer running at: http://localhost:52554")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=52554,
        access_log=True,
        log_level="info",
        reload=True
    )