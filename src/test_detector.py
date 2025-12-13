"""
Test script to verify detector setup
"""
import cv2
import sys
from pathlib import Path

print("TESTING EMOTION DETECTOR SETUP")

# Test 1: Check imports
print("\n[1/5] Testing imports...")
try:
    import torch
    import torchvision
    from PIL import Image
    import numpy as np
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -r ../requirements.txt")
    sys.exit(1)

# Test 2: Check OpenCV
print("\n[2/5] Testing OpenCV...")
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("❌ Haar Cascade not loaded")
    else:
        print("✅ OpenCV and Haar Cascade working")
except Exception as e:
    print(f"❌ OpenCV error: {e}")

# Test 3: Check CUDA availability
print("\n[3/5] Checking GPU availability...")
if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
else:
    print("⚠️  No GPU detected - will use CPU (slower)")

# Test 4: Check model file
print("\n[4/5] Checking for trained model...")
model_path = Path("C:/Users/PC/Desktop/yr3/Artificial intelligence/projects/ml_project/machine_learning/src/best_fer_resnet18.pth")
if model_path.exists():
    print(f"✅ Model found: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"   Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
else:
    print(f"❌ Model not found: {model_path}")
    print("   You need to train the model first!")
    print("   Run the notebook: notebooks/mood_detection.ipynb")

# Test 5: Check webcam
print("\n[5/5] Testing webcam access...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Webcam working ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print("⚠️  Webcam opened but can't read frames")
        cap.release()
    else:
        print("❌ Cannot open webcam")
        print("   Image detector will still work!")
except Exception as e:
    print(f"❌ Webcam error: {e}")

print("\n" + "="*60)
print("SETUP VERIFICATION COMPLETE")
print("="*60)

# Summary
if model_path.exists():
    print("\n✅ You're ready to run:")
    print("   python webcam_detector.py --model ../best_fer_resnet18.pth")
    print("   python image_detector.py --model ../best_fer_resnet18.pth --image test.jpg")
else:
    print("\n⚠️  Next steps:")
    print("   1. Train the model using notebooks/mood_detection.ipynb")
    print("   2. Then run the detectors")
