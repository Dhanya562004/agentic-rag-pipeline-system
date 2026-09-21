"""
Script to start FastAPI server
"""

import subprocess
import sys
import time
import requests
from pathlib import Path
from fix_embeddings import fix_embeddings

def check_embeddings_integrity():
    """Check if the embeddings file is consistent with the data"""
    print("Checking embeddings integrity...")
    
    result = fix_embeddings()
    if result:
        print("✅ Embeddings check passed")
        return True
    else:
        print("❌ Embeddings check failed")
        return False

def start_fastapi():
    """Start the FastAPI server"""
    print("Starting FastAPI server...")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"],
        cwd=Path(__file__).parent
    )

def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API health check passed")
            return True
        else:
            print(f"❌ API health check failed: Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API health check failed: Connection error")
        return False


def main():
    """Main function to start FastAPI service"""
    print("🚀 RAG Pipeline Services")
    print("\nOptions:")
    print("1. Start FastAPI server")
    print("2. Exit")

    choice = input("\nEnter your choice (1-2): ")

    if choice == '2':
        print("Exiting...")
        sys.exit(0)
    
    
    if choice == '1':
        print("\n🚀 Starting RAG Pipeline FastAPI Service...")
        
        if not check_embeddings_integrity():
            print("⚠️ Embeddings integrity check failed. Service might not work correctly.")
            user_input = input("Do you want to continue anyway? (y/n): ")
            if user_input.lower() != 'y':
                print("Exiting...")
                sys.exit(1)
    
    fastapi_process = None
    try:
        fastapi_process = start_fastapi()
        
        print("Waiting for FastAPI service to start...")
        max_retries = 5
        for i in range(max_retries):
            time.sleep(3)
            print(f"Health check attempt {i+1}/{max_retries}...")
            if check_api_health():
                break
            if i == max_retries - 1:
                print("⚠️ Could not confirm API health after multiple attempts.")
                print("The service might still be starting up, please check the logs.")
        
        print("\n✅ FastAPI service started")
        print("📊 FastAPI server: http://localhost:8000")
        print("📚 API docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the service...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping service...")
        try:
            if fastapi_process:
                fastapi_process.terminate()
            print("✅ FastAPI service stopped successfully!")
        except:
            print("⚠️ Error stopping service. You may need to stop it manually.")
        sys.exit(0)

if __name__ == "__main__":
    main()
