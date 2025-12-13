"""
Emotion Detection from Images
==============================
Detect emotions from single images or batch process multiple images.

Usage:
    # Single image
    python image_detector.py --model ../best_fer_resnet18.pth --image photo.jpg
    
    # Batch process folder
    python image_detector.py --model ../best_fer_resnet18.pth --folder ./images --output ./results
"""

import cv2
import torch
import argparse
from pathlib import Path
from utils import (
    load_model, detect_faces, predict_emotion,
    draw_emotion_label, draw_emotion_bar
)


class ImageEmotionDetector:
    """Emotion detector for static images"""
    
    def __init__(self, model_path, device='cpu'):
        """
        Initialize image detector
        
        Args:
            model_path: Path to trained model
            device: Device to run inference ('cpu' or 'cuda')
        """
        self.device = device
        self.model = load_model(model_path, device)
        
        # Load Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        print("Image emotion detector initialized")
    
    def process_image(self, image_path, output_path=None, show_probabilities=True):
        """
        Process single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)
            show_probabilities: Whether to show probability bars
        
        Returns:
            results: List of detected emotions with confidence
        """
        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Detect faces
        faces = detect_faces(frame, self.face_cascade)
        
        results = []
        
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face
            face_img = frame[y:y+h, x:x+w]
            
            # Predict emotion
            emotion, confidence, probabilities = predict_emotion(
                self.model, face_img, self.device
            )
            
            # Store results
            results.append({
                'face_id': i + 1,
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probabilities,
                'bbox': (x, y, w, h)
            })
            
            # Draw annotations
            frame = draw_emotion_label(frame, x, y, w, h, emotion, confidence)
        
        # Draw probability bars for first face
        if show_probabilities and len(results) > 0:
            frame = draw_emotion_bar(frame, results[0]['probabilities'], x=10, y=30)
        
        # Add summary text
        cv2.putText(frame, f"Faces detected: {len(faces)}",
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2)
        
        # Save or display
        if output_path:
            cv2.imwrite(str(output_path), frame)
            print(f"Saved annotated image: {output_path}")
        else:
            # Display image
            cv2.imshow('Emotion Detection', frame)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def process_folder(self, input_folder, output_folder=None):
        """
        Process all images in a folder
        
        Args:
            input_folder: Path to folder containing images
            output_folder: Path to save annotated images (optional)
        
        Returns:
            all_results: Dictionary of results for each image
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise ValueError(f"Input folder not found: {input_folder}")
        
        # Create output folder if specified
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if len(image_files) == 0:
            print(f"No images found in {input_folder}")
            return {}
        
        print(f"Processing {len(image_files)} images...")
        all_results = {}
        
        for img_file in image_files:
            print(f"\nProcessing: {img_file.name}")
            
            try:
                # Determine output path
                out_path = None
                if output_folder:
                    out_path = output_path / f"annotated_{img_file.name}"
                
                # Process image
                results = self.process_image(img_file, out_path, show_probabilities=False)
                all_results[img_file.name] = results
                
                # Print results
                print(f"  Faces detected: {len(results)}")
                for result in results:
                    print(f"    Face {result['face_id']}: "
                          f"{result['emotion']} ({result['confidence']:.1f}%)")
            
            except Exception as e:
                print(f"  Error processing {img_file.name}: {e}")
                all_results[img_file.name] = []
        
        print(f"\nProcessed {len(image_files)} images")
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Emotion detection from images')
    parser.add_argument('--model', type=str,
                       default='C:/Users/PC/Desktop/yr3/Artificial intelligence/projects/ml_project/machine_learning/src/best_fer_resnet18.pth',
                       help='Path to trained model')
    parser.add_argument('--image', type=str,
                       help='Path to single image')
    parser.add_argument('--folder', type=str,
                       help='Path to folder containing images')
    parser.add_argument('--output', type=str,
                       help='Path to save annotated images')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.folder:
        parser.error("Please specify either --image or --folder")
    
    if args.image and args.folder:
        parser.error("Please specify only one of --image or --folder")
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Determine device
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize detector
    detector = ImageEmotionDetector(args.model, device)
    
    try:
        if args.image:
            # Process single image
            results = detector.process_image(args.image, args.output)
            
            print("\n" + "="*50)
            print("RESULTS")
            print("="*50)
            for result in results:
                print(f"\nFace {result['face_id']}:")
                print(f"  Emotion: {result['emotion']}")
                print(f"  Confidence: {result['confidence']:.2f}%")
                print(f"  Probabilities:")
                for emotion, prob in sorted(result['probabilities'].items(),
                                          key=lambda x: x[1], reverse=True):
                    print(f"    {emotion}: {prob:.2f}%")
        
        else:
            # Process folder
            results = detector.process_folder(args.folder, args.output)
            
            # Summary statistics
            total_faces = sum(len(r) for r in results.values())
            emotion_counts = {}
            
            for img_results in results.values():
                for result in img_results:
                    emotion = result['emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            print("\n" + "="*50)
            print("SUMMARY")
            print("="*50)
            print(f"Images processed: {len(results)}")
            print(f"Total faces detected: {total_faces}")
            print(f"\nEmotion distribution:")
            for emotion, count in sorted(emotion_counts.items(), 
                                        key=lambda x: x[1], reverse=True):
                print(f"  {emotion}: {count} ({100*count/total_faces:.1f}%)")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
