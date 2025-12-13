"""
Real-time Emotion Detection from Webcam
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
import numpy as np
from pathlib import Path
from utils import (
    load_model, detect_faces, predict_emotion,
    draw_emotion_label, draw_emotion_bar
)


class LivenessDetector:
    """Detects if face is live (not a photo/poster)"""
    
    def __init__(self, history_size=5):
        """
        Initialize liveness detector
        
        Args:
            history_size: Number of frames to track for movement detection
        """
        self.history_size = history_size
        self.face_history = []
        self.motion_threshold = 5  # Minimum pixel movement
        
    def is_live(self, face_rect, frame_shape):
        """
        Check if face is live based on movement patterns
        
        Args:
            face_rect: Tuple (x, y, w, h) of detected face
            frame_shape: Shape of current frame
            
        Returns:
            bool: True if face appears to be live
        """
        x, y, w, h = face_rect
        face_center = (x + w // 2, y + h // 2)
        
        # Add to history
        self.face_history.append({
            'center': face_center,
            'rect': face_rect,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.face_history) > self.history_size:
            self.face_history.pop(0)
        
        # Need at least 3 frames to detect movement
        if len(self.face_history) < 3:
            return None  # Not enough data yet
        
        # Calculate movement between frames
        movements = []
        for i in range(1, len(self.face_history)):
            prev_center = self.face_history[i-1]['center']
            curr_center = self.face_history[i]['center']
            
            dist = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                          (curr_center[1] - prev_center[1])**2)
            movements.append(dist)
        
        # Check for motion variance (live faces have varied movement)
        avg_motion = np.mean(movements) if movements else 0
        motion_variance = np.var(movements) if len(movements) > 1 else 0
        
        # Live detection criteria:
        # 1. Some consistent movement (avg_motion > threshold)
        # 2. Varied movement pattern (not static)
        is_moving = avg_motion > self.motion_threshold
        has_variance = motion_variance > 1.0
        
        # Face is considered live if it shows natural movement
        return is_moving or has_variance
    
    def reset(self):
        """Reset history"""
        self.face_history = []


class WebcamEmotionDetector:
    """Real-time emotion detector using webcam"""
    
    def __init__(self, model_path, device='cpu', camera_id=0, enable_liveness=True):
        """
        Initialize webcam detector
        
        Args:
            model_path: Path to trained model
            device: Device to run inference ('cpu' or 'cuda')
            camera_id: Camera device ID (0 for default)
            enable_liveness: Enable liveness detection
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
        self.enable_liveness = enable_liveness
        
        # Liveness detection
        self.liveness_detector = LivenessDetector(history_size=5)
        self.liveness_confirmed = {}  # Track liveness per face
        
        print("Webcam emotion detector initialized")
        print(f"Device: {device}")
        print(f"Model: {model_path}")
        print(f"Liveness detection: {'ENABLED' if enable_liveness else 'DISABLED'}")
    
    def run(self):
        """Run real-time detection"""
        print("\nStarting webcam detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Toggle probability bars")
        print("  'f' - Toggle FPS display")
        print("  'l' - Toggle liveness detection")
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
            
            # Only process if faces are detected
            if len(faces) == 0:
                # Display message when no faces detected
                cv2.putText(frame, "No faces detected",
                           (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 0, 255), 2)
                self.liveness_detector.reset()
            else:
                # Process each detected face
                for idx, (x, y, w, h) in enumerate(faces):
                    # Validate face region coordinates
                    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                        continue
                    
                    # Validate minimum face size (at least 20x20 pixels)
                    if w < 20 or h < 20:
                        continue
                    
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Validate face image is not empty
                    if face_img.size == 0:
                        continue
                    
                    # Check liveness if enabled
                    liveness_status = None
                    if self.enable_liveness:
                        liveness_status = self.liveness_detector.is_live((x, y, w, h), frame.shape)
                    
                    # Only predict emotion if liveness check passes or is inconclusive
                    if not self.enable_liveness or liveness_status is not False:
                        # Predict emotion
                        emotion, confidence, probabilities = predict_emotion(
                            self.model, face_img, self.device
                        )
                        
                        # Draw results
                        frame = draw_emotion_label(frame, x, y, w, h, emotion, confidence)
                        
                        # Draw probability bars if enabled
                        if self.show_probabilities and len(faces) == 1:
                            frame = draw_emotion_bar(frame, probabilities, x=10, y=30)
                        
                        # Draw liveness indicator
                        if self.enable_liveness:
                            if liveness_status is True:
                                cv2.putText(frame, "LIVE",
                                           (x, y - 25),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                           (0, 255, 0), 2)
                            elif liveness_status is False:
                                cv2.putText(frame, "PHOTO/POSTER",
                                           (x, y - 25),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                           (0, 0, 255), 2)
                            else:
                                cv2.putText(frame, "ANALYZING...",
                                           (x, y - 25),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                           (255, 165, 0), 2)
                    else:
                        # Draw warning for static/poster
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "STATIC IMAGE DETECTED",
                                   (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                   (0, 0, 255), 2)
            
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
            elif key == ord('l'):
                self.enable_liveness = not self.enable_liveness
                print(f"Liveness detection: {'ON' if self.enable_liveness else 'OFF'}")
        
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
                       default='C:/Users/PC/Desktop/yr3/Artificial intelligence/projects/ml_project/machine_learning/src/best_fer_resnet18.pth',
                       help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage (default: auto-detect)')
    parser.add_argument('--no-liveness', action='store_true',
                       help='Disable liveness detection')
    
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
        detector = WebcamEmotionDetector(args.model, device, args.camera, 
                                        enable_liveness=not args.no_liveness)
        detector.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
