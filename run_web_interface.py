#!/usr/bin/env python3
"""
Launcher script for the CSV File Browser and Plotter web interface
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web_interface.app import app

if __name__ == '__main__':
    print("Starting CSV File Browser and Plotter...")
    print("Open your web browser and go to: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install Flask==3.0.2 plotly==5.19.0") 
