"""
Evaluation Script for Vehicle Collision Predictions

This script:
1. Loads videos from the test set
2. Processes each video frame by frame
3. Generates and saves frame-level predictions as JSON files
4. Aggregates predictions into video-level scores
5. Outputs the final submission CSV file
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms
from PIL import Image

# Import the model architecture and feature names from the training script
from featureExtractor_train import BoundingBoxFeaturePredictor, FEATURE_NAMES

# Constants
TEST_CSV_PATH = "datasets/Nexus_Data/test.csv"
VIDEO_ROOT = "/Volumes/T7 Shield/nexar-collision-prediction/test"  # Directory containing test videos

if not os.path.exists(VIDEO_ROOT):
    raise FileNotFoundError(f"Video root directory not found: {VIDEO_ROOT}")
else: 
    print(f"Video root directory found: {VIDEO_ROOT}")

JSON_OUTPUT_DIR = "/Volumes/T7 Shield/nexar-collision-prediction/nexar_predictions"  # Directory for frame-level JSON predictions
CSV_OUTPUT_DIR = "evaluation_results"  # Directory for final submission CSV
OUTPUT_CSV = "submission.csv"
CHECKPOINT_PATH = "checkpoints/bounding_box_feature_predictor.pth"
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


def pad_video_id(video_id):
    """
    Pad video ID with zeros to ensure it's 5 digits.
    
    Args:
        video_id (str or int): Video identifier
        
    Returns:
        str: Zero-padded 5-digit video ID
    """
    # Convert to string and remove any existing leading zeros
    video_id_str = str(int(video_id))

    # Pad with zeros to 5 digits
    return video_id_str.zfill(5)

def process_frame(frame, model):
    """
    Process a single frame and generate predictions.
    
    Args:
        frame (numpy.ndarray): Input frame in BGR format
        model (BoundingBoxFeaturePredictor): Loaded model
        
    Returns:
        tuple: (features_dict, danger_score)
    """
    # Convert BGR to RGB and then to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    # Preprocess and predict
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_features, pred_danger = model(image_tensor)
    
    # Convert predictions to numpy
    features = pred_features.cpu().numpy()[0]
    danger_score = float(pred_danger.cpu().numpy()[0][0])
    
    # Create features dictionary using FEATURE_NAMES from training script
    features_dict = {
        name: float(value) 
        for name, value in zip(FEATURE_NAMES, features)
    }
    
    return features_dict, danger_score

def process_video(video_path, model, video_id):
    """
    Process all frames in a video and save predictions.
    
    Args:
        video_path (str): Path to the video file
        model (BoundingBoxFeaturePredictor): Loaded model
        video_id (str): Video identifier
        
    Returns:
        float: Aggregated danger score for the video
    """
    # Ensure video_id is padded and string
    padded_video_id = pad_video_id(str(video_id))
    
    # Create output directory for this video's JSON files
    json_output_dir = Path(JSON_OUTPUT_DIR) / padded_video_id
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0.0
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_danger_score = 0.0
    
    # Process each frame
    pbar = tqdm(range(frame_count), desc=f"Video {padded_video_id}", position=1, leave=False)
    for frame_num in pbar:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        features_dict, danger_score = process_frame(frame, model)
        max_danger_score = max(max_danger_score, danger_score)
        
        # Update progress bar with current frame
        pbar.set_postfix_str(f"Frame {frame_num}/{frame_count}")
        
        # Save frame predictions as JSON
        prediction = {
            'frame_number': frame_num,
            'features': features_dict,
            'danger_score': danger_score
        }
        
        json_path = json_output_dir / f"frame_{frame_num:04d}.json"
        with open(json_path, 'w') as f:
            json.dump(prediction, f, indent=2)
    
    cap.release()
    return max_danger_score

def generate_submission(json_dir=JSON_OUTPUT_DIR, csv_dir=CSV_OUTPUT_DIR):
    """
    Generate submission CSV file with video IDs and their danger scores.
    
    Args:
        json_dir (str): Directory to save frame-level JSON predictions
        csv_dir (str): Directory to save final submission CSV
    """
    # Create output directories
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model()
    
    # Load test data
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    # Process each video
    results = []
    print("\nProcessing videos...")
    
    # Main progress bar for videos
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Overall Progress", position=0):
        video_id = str(row['id'])  # Ensure video_id is string
        padded_video_id = pad_video_id(video_id)
        video_path = Path(VIDEO_ROOT) / f"{padded_video_id}.mp4"
        
        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            video_score = 0.0
        else:
            video_score = process_video(video_path, model, padded_video_id)
        
        results.append({
            'id': padded_video_id,
            'score': video_score
        })
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(results)
    submission_df = submission_df.sort_values('id')
    submission_df['score'] = submission_df['score'].map('{:.4f}'.format)
    
    # Save submission CSV
    output_path = os.path.join(csv_dir, OUTPUT_CSV)
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission file saved to: {output_path}")
    
    # Display first few rows
    print("\nFirst few rows of submission file:")
    print(submission_df.head())
    
    return submission_df

if __name__ == "__main__":
    print("Starting prediction and submission generation...")
    submission_df = generate_submission()
    print("\nProcess complete!") 