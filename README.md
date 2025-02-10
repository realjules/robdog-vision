# Robot Vision Demo Using LLaVA and DepthAnything

A simple demonstration of using LLaVA (Large Language and Vision Assistant) and DepthAnything for robot navigation. This demo shows how to process real-time camera feed, estimate depth information, and generate structured navigation commands for a robot.

## Features

- Real-time camera feed processing
- LLaVA-based scene understanding
- Depth estimation using DepthAnything
- Region-based depth analysis
- Object detection based on depth gradients
- JSON command generation for robot navigation
- Web-based visualization interface
- Real-time command visualization

## Depth Analysis

The system uses DepthAnything to provide detailed depth information:

### Region Analysis
- Divides the image into 5 regions (center, top, bottom, left, right)
- Provides depth statistics for each region:
  - Mean depth
  - Minimum depth
  - Maximum depth
  - Relative distance classification (near/far)

### Object Detection
- Detects objects based on depth gradients
- For each object provides:
  - Position (normalized x, y coordinates)
  - Size (width and height as proportion of image)
  - Depth information (mean depth and relative distance)

### Visualization
- Generates depth map visualization
- Shows region divisions
- Highlights detected objects

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Main Application
1. Start the server:
```bash
python app.py
```

2. Open your web browser to `http://localhost:53549/`

3. Allow camera access when prompted

4. Click "Start Processing" to begin real-time processing

### Depth Analysis Interface
1. Start the depth analysis interface:
```bash
python web_interface.py
```

2. Open your web browser to `http://localhost:53549/`

3. Use the interface to:
   - Upload images for analysis
   - View depth maps and visualizations
   - See region-based depth analysis
   - Inspect detected objects

### Command Line Usage
You can also use the depth analyzer directly from the command line:
```bash
python depth_analyzer.py path/to/image.jpg
```

This will:
- Generate a depth analysis
- Save visualization to 'depth_analysis.png'
- Print detailed analysis results

## Command Structure

The system generates JSON commands in the following format:
```json
{
    "velocity_command": {
        "linear_velocity_mps": 0.5,    // Forward/backward speed (-1.0 to 1.0)
        "angular_velocity_radps": 0.2   // Turning speed (-1.0 to 1.0)
    },
    "gait_mode": "trotting",           // Robot's movement style
    "reasoning": "Moving forward to approach the target object while avoiding the obstacle on the left",
    "timestamp": "2024-01-31T17:00:03Z"
}
```