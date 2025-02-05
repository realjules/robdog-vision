import cv2
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json
from datetime import datetime

def test_vision_system():
    print("1. Testing GPU availability...")
    if torch.cuda.is_available():
        print("✓ GPU is available")
        device = "cuda"
    else:
        print("! Running on CPU (this will be slower)")
        device = "cpu"

    print("\n2. Loading LLaVA model...")
    try:
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        model = model.to(device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"! Error loading model: {str(e)}")
        return

    print("\n3. Testing camera...")
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            print("✓ Camera is working")
            cv2.imwrite('test_frame.jpg', frame)
        else:
            print("! Cannot access camera")
            return
        cap.release()
    except Exception as e:
        print(f"! Camera error: {str(e)}")
        return

    print("\n4. Testing image processing...")
    try:
        # Process a single frame
        query = """Analyze this scene and generate a robot navigation command. 
        Generate a JSON response with velocity and gait commands."""
        
        # Prepare inputs
        inputs = processor(images=frame, text=query, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        print("\nModel Response:")
        print(response)
        
        print("\n✓ Image processing test complete")
    except Exception as e:
        print(f"! Processing error: {str(e)}")
        return

    print("\n5. Testing web server...")
    try:
        import uvicorn
        import asyncio
        import threading
        import requests
        import time
        
        def run_server():
            uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="error")
        
        # Start server in a thread
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Test connection
        response = requests.get("http://localhost:8000")
        if response.status_code == 200:
            print("✓ Web server is working")
        else:
            print("! Web server test failed")
    except Exception as e:
        print(f"! Server error: {str(e)}")
        return

    print("\nAll tests completed! You can now run the full demo:")
    print("1. Run 'python app.py'")
    print("2. Open http://localhost:8000 in your browser")
    print("3. Click 'Start Processing' to begin")

if __name__ == "__main__":
    test_vision_system()