"""
Nexus Dataset Feature Prediction Script

This script loads the trained vehicle collision prediction model and runs inference
on images from the Nexus dataset to predict vehicle features and collision danger.

The script:
1. Loads the trained model from checkpoint
2. Processes Nexus dataset images
3. Predicts vehicle features and collision danger
4. Saves predictions to JSON files
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Import the model architecture from the training script
from featureExtractor_train import BoundingBoxFeaturePredictor, FEATURE_NAMES

# Constants
CHECKPOINT_PATH = "checkpoints/bounding_box_feature_predictor.pth"
NEXUS_DATA_ROOT = "datasets/Nexus_Data/train/no_accident"  # Update this path to your Nexus dataset location
OUTPUT_DIR = "datasets/nexus_predictions/no_accident"
IMAGE_SIZE = (224, 224)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    """
    Load the trained model from checkpoint.
    
    Returns:
        BoundingBoxFeaturePredictor: Loaded and initialized model
    """
    model = BoundingBoxFeaturePredictor().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def process_image(image_path, model):
    """
    Process a single image and predict features.
    
    Args:
        image_path (str): Path to the image file
        model (BoundingBoxFeaturePredictor): Loaded model
        
    Returns:
        tuple: (features, danger_score)
            - features: Dict mapping feature names to predicted values
            - danger_score: Predicted collision danger probability
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            pred_features, pred_danger = model(image_tensor)
        
        # Convert predictions to numpy and create feature dictionary
        features_dict = {}
        pred_features = pred_features.cpu().numpy()[0]
        for name, value in zip(FEATURE_NAMES, pred_features):
            features_dict[name] = float(value)
        
        danger_score = float(pred_danger.cpu().numpy()[0][0])
        
        return features_dict, danger_score
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def process_nexus_dataset():
    """
    Process all images in the Nexus dataset and save predictions.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model()
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(NEXUS_DATA_ROOT).rglob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        # Get relative path for output file structure
        rel_path = image_path.relative_to(NEXUS_DATA_ROOT)
        output_path = Path(OUTPUT_DIR) / rel_path.with_suffix('.json')
        
        # Create output directory if needed
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Process image
        features, danger_score = process_image(str(image_path), model)
        
        if features is not None:
            # Save predictions
            predictions = {
                'image_path': str(rel_path),
                'features': features,
                'danger_score': danger_score
            }
            
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)

def analyze_predictions():
    """
    Analyze the predictions across the dataset and generate statistics.
    """
    print("\nAnalyzing predictions...")
    
    # Collect all predictions
    predictions = []
    for json_file in Path(OUTPUT_DIR).rglob('*.json'):
        with open(json_file, 'r') as f:
            predictions.append(json.load(f))
    
    if not predictions:
        print("No predictions found!")
        return
    
    # Calculate statistics
    danger_scores = [p['danger_score'] for p in predictions]
    feature_values = {name: [] for name in FEATURE_NAMES}
    
    for pred in predictions:
        for name, value in pred['features'].items():
            feature_values[name].append(value)
    
    # Print summary statistics
    print(f"\nProcessed {len(predictions)} images")
    print(f"Danger Score Statistics:")
    print(f"  Mean: {np.mean(danger_scores):.4f}")
    print(f"  Std: {np.std(danger_scores):.4f}")
    print(f"  Min: {np.min(danger_scores):.4f}")
    print(f"  Max: {np.max(danger_scores):.4f}")
    
    print("\nFeature Statistics:")
    for name in FEATURE_NAMES:
        values = feature_values[name]
        print(f"\n{name}:")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std: {np.std(values):.4f}")
        print(f"  Min: {np.min(values):.4f}")
        print(f"  Max: {np.max(values):.4f}")

if __name__ == "__main__":
    print("Starting Nexus dataset prediction...")
    process_nexus_dataset()
    analyze_predictions()
    print("\nPrediction complete! Results saved in:", OUTPUT_DIR) 