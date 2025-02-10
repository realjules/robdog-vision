from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
from pathlib import Path
import os
import shutil
from depth_analyzer import DepthAnalyzer
import json
import base64

app = FastAPI()

# Create directories if they don't exist
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Mount the results directory for serving visualization images
app.mount("/results", StaticFiles(directory="results"), name="results")

# Initialize the depth analyzer
analyzer = DepthAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def get_upload_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Depth Analysis Interface</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }
            .panel {
                flex: 1;
                min-width: 300px;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .upload-form {
                margin-bottom: 20px;
            }
            .results {
                display: none;
            }
            .visualization img {
                max-width: 100%;
                height: auto;
                border-radius: 4px;
                margin-top: 10px;
            }
            .depth-region {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
            }
            .depth-region.near {
                background-color: #ffebee;
            }
            .depth-region.far {
                background-color: #e8f5e9;
            }
            .object-card {
                background: #f8f8f8;
                padding: 10px;
                margin: 5px 0;
                border-radius: 4px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            .loading {
                display: none;
                color: #666;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <h1>Depth Analysis Interface</h1>
        
        <div class="container">
            <div class="panel">
                <h2>Upload Image</h2>
                <div class="upload-form">
                    <input type="file" id="imageInput" accept="image/*">
                    <button onclick="analyzeImage()">Analyze Image</button>
                    <div id="loading" class="loading">Processing image...</div>
                </div>
                
                <div class="visualization">
                    <h3>Input Image</h3>
                    <img id="inputImage" style="display: none;">
                </div>
            </div>
            
            <div class="panel results" id="resultsPanel">
                <h2>Analysis Results</h2>
                
                <div class="visualization">
                    <h3>Depth Analysis</h3>
                    <img id="depthVisualization">
                </div>
                
                <div class="analysis">
                    <h3>Region Analysis</h3>
                    <div id="regionAnalysis"></div>
                    
                    <h3>Detected Objects</h3>
                    <div id="objectAnalysis"></div>
                </div>
            </div>
        </div>

        <script>
        async function analyzeImage() {
            const fileInput = document.getElementById('imageInput');
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select an image first.');
                return;
            }

            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultsPanel').style.display = 'none';

            // Display input image
            const inputImage = document.getElementById('inputImage');
            inputImage.src = URL.createObjectURL(file);
            inputImage.style.display = 'block';

            // Create form data
            const formData = new FormData();
            formData.append('file', file);

            try {
                // Send request
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Analysis failed');
                }

                const result = await response.json();

                // Update visualization
                document.getElementById('depthVisualization').src = 
                    `/results/${result.visualization_path}?t=${new Date().getTime()}`;

                // Update region analysis
                const regionAnalysis = document.getElementById('regionAnalysis');
                regionAnalysis.innerHTML = '';
                for (const [region, data] of Object.entries(result.region_analysis)) {
                    const div = document.createElement('div');
                    div.className = `depth-region ${data.relative_distance}`;
                    div.innerHTML = `
                        <strong>${region.toUpperCase()}</strong><br>
                        Mean Depth: ${data.mean_depth.toFixed(2)}<br>
                        Min Depth: ${data.min_depth.toFixed(2)}<br>
                        Max Depth: ${data.max_depth.toFixed(2)}<br>
                        Distance: ${data.relative_distance}
                    `;
                    regionAnalysis.appendChild(div);
                }

                // Update object analysis
                const objectAnalysis = document.getElementById('objectAnalysis');
                objectAnalysis.innerHTML = '';
                result.detected_objects.forEach((obj, index) => {
                    const div = document.createElement('div');
                    div.className = 'object-card';
                    div.innerHTML = `
                        <strong>Object ${index + 1}</strong><br>
                        Position: x=${obj.position.x.toFixed(2)}, y=${obj.position.y.toFixed(2)}<br>
                        Size: ${obj.size.width.toFixed(2)}x${obj.size.height.toFixed(2)}<br>
                        Depth: ${obj.depth.mean.toFixed(2)} (${obj.depth.relative})
                    `;
                    objectAnalysis.appendChild(div);
                });

                // Show results
                document.getElementById('resultsPanel').style.display = 'block';
            } catch (error) {
                alert('Error analyzing image: ' + error.message);
            } finally {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';
            }
        }
        </script>
    </body>
    </html>
    """

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Create a unique filename
        file_path = UPLOAD_DIR / f"temp_{file.filename}"
        vis_path = RESULTS_DIR / f"depth_analysis_{file.filename}.png"
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze image
        results = analyzer.analyze_image(str(file_path))
        
        # Move visualization to results directory
        if os.path.exists("depth_analysis.png"):
            shutil.move("depth_analysis.png", vis_path)
            results["visualization_path"] = f"depth_analysis_{file.filename}.png"
        
        # Clean up
        os.remove(file_path)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\nServer running at: http://localhost:53549")
    uvicorn.run(app, host="0.0.0.0", port=53549)