# Robot Vision Demo Using LLaVA

A simple demonstration of using LLaVA (Large Language and Vision Assistant) for robot navigation. This demo shows how to process real-time camera feed and generate structured navigation commands for a robot.

## Features

- Real-time camera feed processing
- LLaVA-based scene understanding
- JSON command generation for robot navigation
- Web-based visualization interface
- Real-time command visualization

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

1. Start the server:
```bash
python app.py
```

2. Open your web browser to `http://localhost:50232/`

3. Allow camera access when prompted

4. Click "Start Processing" to begin real-time processing

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