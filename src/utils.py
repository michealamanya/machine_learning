"""
Utility functions for emotion detection
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Color mapping for each emotion (BGR format for OpenCV)
EMOTION_COLORS = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 128, 0),    # Green
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 255),    # Yellow
    'Neutral': (128, 128, 128),# Gray
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (0, 165, 255)  # Orange
}


class EmotionResNet(nn.Module):
    """ResNet-18 model for emotion recognition"""
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(EmotionResNet, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def load_model(model_path, device='cpu'):
    """
    Load trained emotion detection model
    
    Args:
        model_path: Path to saved model (.pth file)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in eval mode
    """
    model = EmotionResNet(num_classes=len(EMOTIONS)).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def get_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def preprocess_face(face_img):
    """
    Preprocess face image for model input
    
    Args:
        face_img: Face image (numpy array, grayscale or RGB)
    
    Returns:
        Preprocessed tensor
    """
    # Convert to PIL Image
    if len(face_img.shape) == 2:  # Grayscale
        face_pil = Image.fromarray(face_img).convert('L')
        face_pil = Image.merge('RGB', (face_pil, face_pil, face_pil))
    else:  # RGB
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    
    # Apply transforms
    transform = get_transform()
    face_tensor = transform(face_pil).unsqueeze(0)
    
    return face_tensor


def predict_emotion(model, face_img, device='cpu'):
    """
    Predict emotion from face image
    
    Args:
        model: Trained emotion model
        face_img: Face image (numpy array)
        device: Device to run inference on
    
    Returns:
        emotion: Predicted emotion label
        confidence: Confidence score (0-100)
        probabilities: Dictionary of all emotion probabilities
    """
    face_tensor = preprocess_face(face_img).to(device)
    
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)
    
    emotion = EMOTIONS[predicted.item()]
    confidence_pct = confidence.item() * 100
    
    prob_dict = {EMOTIONS[i]: probabilities[i].item() * 100 
                 for i in range(len(EMOTIONS))}
    
    return emotion, confidence_pct, prob_dict


def draw_emotion_label(frame, x, y, w, h, emotion, confidence):
    """
    Draw bounding box and emotion label on frame
    
    Args:
        frame: Video frame
        x, y, w, h: Face bounding box coordinates
        emotion: Predicted emotion
        confidence: Confidence score
    
    Returns:
        Frame with annotations
    """
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Prepare label text
    label = f"{emotion}: {confidence:.1f}%"
    
    # Calculate text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    
    # Draw background rectangle for text
    cv2.rectangle(frame, 
                 (x, y - text_height - 10),
                 (x + text_width, y),
                 color, -1)
    
    # Draw text
    cv2.putText(frame, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def draw_emotion_bar(frame, probabilities, x=10, y=30):
    """
    Draw emotion probability bar chart on frame
    
    Args:
        frame: Video frame
        probabilities: Dictionary of emotion probabilities
        x, y: Starting position for bar chart
    """
    bar_width = 200
    bar_height = 20
    spacing = 5
    
    # Sort emotions by probability
    sorted_emotions = sorted(probabilities.items(), 
                            key=lambda item: item[1], 
                            reverse=True)
    
    for i, (emotion, prob) in enumerate(sorted_emotions):
        # Calculate bar position
        bar_y = y + i * (bar_height + spacing)
        
        # Draw background
        cv2.rectangle(frame, 
                     (x, bar_y),
                     (x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Draw filled bar
        filled_width = int(bar_width * prob / 100)
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        cv2.rectangle(frame,
                     (x, bar_y),
                     (x + filled_width, bar_y + bar_height),
                     color, -1)
        
        # Draw text
        text = f"{emotion}: {prob:.1f}%"
        cv2.putText(frame, text,
                   (x + bar_width + 10, bar_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1)
    
    return frame


def detect_faces(frame, face_cascade):
    """
    Detect faces in frame using Haar Cascade
    
    Args:
        frame: Video frame
        face_cascade: Loaded Haar Cascade classifier
    
    Returns:
        List of face bounding boxes (x, y, w, h)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces
