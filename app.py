import cv2
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
from datetime import datetime
from PIL import Image
import io
import base64
from transformers import AutoFeatureExtractor, ViTForImageClassification

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class RobotVisionSystem:
    def __init__(self):
        # Initialize ViT model
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    def process_image(self, image, query=None):
        # Convert CV2 image to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        
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
        
        # Add timestamp
        command['timestamp'] = datetime.utcnow().isoformat() + "Z"
        return command

# Initialize vision system
vision_system = RobotVisionSystem()

@app.get("/")
async def get_index():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive base64 image from client
            data = await websocket.receive_text()
            image_data = base64.b64decode(data.split(',')[1])
            
            # Convert to CV2 image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process image
            command = vision_system.process_image(image)
            
            # Send response
            await websocket.send_json(command)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    print("\nServer running at: http://localhost:50232")
    uvicorn.run(app, host="0.0.0.0", port=50232, access_log=False)