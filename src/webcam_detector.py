"""
Real-time Emotion Detection from Webcam
==========================================
Detects emotions from webcam feed in real-time using trained ResNet model.

Usage:
    python webcam_detector.py --model ../best_fer_resnet18.pth

Controls:
    'q' - Quit
    's' - Toggle showing probability bars
    'f' - Toggle FPS display
"""

import cv2
import torch
import argparse
import time
from pathlib import Path
from utils import (
    load_model, detect_faces, predict_emotion,
    draw_emotion_label, draw_emotion_bar
)


class WebcamEmotionDetector:
    """Real-time emotion detector using webcam"""
    
    def __init__(self, model_path, device='cpu', camera_id=0):
        """
        Initialize webcam detector
        
        Args:
            model_path: Path to trained model
            device: Device to run inference ('cpu' or 'cuda')
            camera_id: Camera device ID (0 for default)
        """
        self.device = device
        self.model = load_model(model_path, device)
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Display settings
        self.show_probabilities = True
        self.show_fps = True
        
        print("Webcam emotion detector initialized")
        print(f"Device: {device}")
        print(f"Model: {model_path}")
    
    def run(self):
        """Run real-time detection"""
        print("\nStarting webcam detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Toggle probability bars")
        print("  'f' - Toggle FPS display")
        print("-" * 50)
        
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = detect_faces(frame, self.face_cascade)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = frame[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, confidence, probabilities = predict_emotion(
                    self.model, face_img, self.device
                )
                
                # Draw results
                frame = draw_emotion_label(frame, x, y, w, h, emotion, confidence)
                
                # Draw probability bars if enabled
                if self.show_probabilities and len(faces) == 1:
                    frame = draw_emotion_bar(frame, probabilities, x=10, y=30)
            
            # Calculate FPS
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display FPS
            if self.show_fps:
                cv2.putText(frame, f"FPS: {fps}",
                           (frame.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 0), 2)
            
            # Display number of faces detected
            cv2.putText(frame, f"Faces: {len(faces)}",
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_probabilities = not self.show_probabilities
                print(f"Probability bars: {'ON' if self.show_probabilities else 'OFF'}")
            elif key == ord('f'):
                self.show_fps = not self.show_fps
                print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam detector stopped")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'cap'):
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(description='Real-time emotion detection from webcam')
    parser.add_argument('--model', type=str, 
                       default='../best_fer_resnet18.pth',
                       help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first using the notebook.")
        return
    
    # Determine device
    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run detector
    try:
        detector = WebcamEmotionDetector(args.model, device, args.camera)
        detector.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
