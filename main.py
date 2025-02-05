import argparse
from src.web.app import RobotVisionApp

def main():
    parser = argparse.ArgumentParser(description='Robot Vision Demo')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=50232, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create and start application
    app = RobotVisionApp()
    app.start(host=args.host, port=args.port)

if __name__ == "__main__":
    main()