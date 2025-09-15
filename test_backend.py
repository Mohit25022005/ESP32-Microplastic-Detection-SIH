#!/usr/bin/env python3
"""
Simple test script to run the FastAPI backend directly
This allows testing without Docker containers
"""

import sys
import subprocess
import time
import requests
import webbrowser
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
    ]
    
    print("Installing required packages...")
    for req in requirements:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", req], 
                         check=True, capture_output=True)
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
            return False
    return True

def run_backend():
    """Run the FastAPI backend"""
    backend_path = Path("backend/app/main.py")
    
    if not backend_path.exists():
        print(f"❌ Backend file not found: {backend_path}")
        return False
    
    print("🚀 Starting FastAPI backend server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🎮 Demo Page: http://localhost:8000/demo")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    try:
        # Run uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "backend.app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], cwd=".")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except FileNotFoundError:
        print("❌ uvicorn not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uvicorn[standard]"])
        print("✅ Try running the script again")

def test_api():
    """Test API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing API endpoints...")
    
    endpoints = [
        "/",
        "/health", 
        "/api/devices",
        "/api/analytics/real-time-dashboard",
        "/api/analytics/pollution-forecast"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint} - OK")
            else:
                print(f"⚠️  {endpoint} - Status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint} - Connection failed (server not running?)")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")

def open_demo():
    """Open demo page in browser"""
    time.sleep(2)  # Wait for server to start
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("🌐 Opening demo page in browser...")
            webbrowser.open("http://localhost:8000/demo")
        else:
            print("❌ Server not responding, demo page not opened")
    except:
        print("❌ Could not connect to server")

def main():
    """Main function"""
    print("🔬 ESP32 Microplastic Detection System - Backend Test")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("backend").exists():
        print("❌ Please run this script from the project root directory")
        print("   Current directory should contain 'backend/' folder")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return
    
    print("\n" + "="*60)
    print("🚀 Starting backend server...")
    print("   Once started, you can:")
    print("   • Visit http://localhost:8000/demo for the demo page")
    print("   • Visit http://localhost:8000/docs for API documentation")
    print("   • Test endpoints manually or use the demo interface")
    print("="*60)
    
    # Run the backend
    run_backend()

if __name__ == "__main__":
    main()