"""
Vehicle Collision Prediction Model Training Script

This script implements a deep learning model for predicting vehicle collisions using data collected from video games.
The model analyzes vehicle bounding boxes to predict various vehicle features and assess collision danger.

The script handles:
- Dataset loading and preprocessing
- Model training and validation
- Performance evaluation
- Result visualization

Author: [Your Name]
Date: [Current Date]
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Define paths
DATA_ROOT = "datasets/GTACrash/accident/GTACrash_accident_part1 imgs"  # Directory containing image frames
JSON_ROOT = "/Users/vishi/VSC Codes/Collision Detection/predicting-vehicle-collisions-using-data-collected-from-video-games/datasets/GTACrash/accident/GTACrash_accident_part1"  # Directory containing JSON files
 # Directory containing JSON files
MODEL_SAVE_PATH = "bounding_box_feature_predictor.pth"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
IMAGE_SIZE = (224, 224)

# Features to predict
FEATURE_NAMES = [
    'speed', 'acceleration', 
    'forwardV_x', 'forwardV_y', 'forwardV_z', 
    'objectSize_x', 'objectSize_y', 'objectSize_z',
    'position_x', 'position_y', 'position_z',
    'angularVelocity_x', 'angularVelocity_y', 'angularVelocity_z'
]
NUM_FEATURES_TO_PREDICT = len(FEATURE_NAMES)

class BoundingBoxDataset(Dataset):
    """
    Custom Dataset for loading and preprocessing vehicle bounding box data.
    
    This dataset handles:
    - Loading image and JSON pairs
    - Extracting bounding boxes and associated features
    - Preprocessing images and normalizing features
    
    Attributes:
        image_paths (list): List of paths to image files
        json_paths (list): List of paths to corresponding JSON annotation files
        transform (callable): Optional transform to be applied to images
        samples (list): Processed samples containing bounding boxes and features
    """
    
    def __init__(self, image_paths, json_paths, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): Paths to image files
            json_paths (list): Paths to corresponding JSON annotation files
            transform (callable, optional): Transform to apply to images
        """
        self.image_paths = image_paths
        self.json_paths = json_paths
        self.transform = transform
        self.samples = []
        
        # Pre-process to extract bounding boxes and features
        self._extract_bounding_boxes()
        
    def _extract_bounding_boxes(self):
        """
        Process image-JSON pairs to extract bounding boxes and features.
        
        For each vehicle in each image:
        - Extracts bounding box coordinates
        - Validates box dimensions
        - Extracts vehicle features (speed, acceleration, etc.)
        - Normalizes coordinates
        """
        for img_path, json_path in zip(self.image_paths, self.json_paths):
            try:
                # Load image
                image = Image.open(img_path).convert("RGB")
                img_width, img_height = image.size
                
                # Load JSON data
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract vehicles and their features
                for vehicle in data.get('vehicleInfo', []):
                    if 'x' in vehicle and 'y' in vehicle and 'width' in vehicle and 'height' in vehicle:
                        # Bounding box coordinates
                        x, y = vehicle['x'], vehicle['y']
                        width, height = vehicle['width'], vehicle['height']
                        
                        # Ensure bounding box is within image bounds
                        if x >= 0 and y >= 0 and x + width <= img_width and y + height <= img_height:
                            # Extract features to predict
                            features = [
                                vehicle.get('speed', 0.0),
                                vehicle.get('acceleration', 0.0),
                            ]
                            
                            # Extract vectors
                            forward_v = vehicle.get('forwardV', [0.0, 0.0, 0.0])
                            features.extend(forward_v[:3])  # Ensure we get exactly 3 values
                            
                            obj_size = vehicle.get('objectSize', [0.0, 0.0, 0.0])
                            features.extend(obj_size[:3])
                            
                            position = vehicle.get('position', [0.0, 0.0, 0.0])
                            features.extend(position[:3])
                            
                            ang_vel = vehicle.get('angularVelocity', [0.0, 0.0, 0.0])
                            features.extend(ang_vel[:3])
                            
                            # Calculate normalized bounding box coordinates (for visualization later)
                            bbox_norm = [x/img_width, y/img_height, width/img_width, height/img_height]
                            
                            # Store the sample
                            self.samples.append({
                                'image_path': img_path,
                                'bbox': [x, y, width, height],
                                'bbox_norm': bbox_norm,
                                'features': features,
                                'is_dangerous': 1 if vehicle.get('syntheticLabel', False) else 0
                            })
            except Exception as e:
                print(f"Error processing {img_path}, {json_path}: {e}")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (processed_image, features, is_dangerous)
                - processed_image: Transformed image of the bounding box
                - features: Tensor of vehicle features
                - is_dangerous: Binary indicator of collision danger
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert("RGB")
        
        # Extract bounding box
        x, y, width, height = sample['bbox']
        
        
        
        # Validate and adjust bounding box coordinates
        left, top = x, y
        right, bottom = x + width, y + height
        
        if right <= left or bottom <= top:
            # If invalid coordinates, use the entire image
            bbox_image = image.copy()
        else:
            # Crop the bounding box from the image
            bbox_image = image.crop((left, top, right, bottom))
        
        # Apply transformations
        if self.transform:
            bbox_image = self.transform(bbox_image)
        
        # Convert features to tensor
        features = torch.tensor(sample['features'], dtype=torch.float32)
        is_dangerous = torch.tensor([sample['is_dangerous']], dtype=torch.float32)
        
        return bbox_image, features, is_dangerous

# Feature prediction model based on VGG16
class BoundingBoxFeaturePredictor(nn.Module):
    """
    Neural network model for predicting vehicle features and collision danger.
    
    Architecture:
    - VGG16 backbone for feature extraction
    - Custom regression head for feature prediction
    - Custom classification head for danger prediction
    
    Features predicted:
    - Vehicle dynamics (speed, acceleration)
    - Spatial properties (position, size)
    - Motion characteristics (velocity, angular velocity)
    """
    
    def __init__(self, num_features=NUM_FEATURES_TO_PREDICT):
        """
        Initialize the model.
        
        Args:
            num_features (int): Number of features to predict
        """
        super(BoundingBoxFeaturePredictor, self).__init__()
        
        # Load pre-trained VGG16
        vgg16 = models.vgg16(pretrained=True)
        
        # Remove the classifier part
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        
        # Calculate feature vector size after VGG16 feature extraction
        vgg_output_size = 512 * 7 * 7  # 25088 for 224x224 input
        
        # Regression head for feature prediction
        self.feature_predictor = nn.Sequential(
            nn.Linear(vgg_output_size, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_features)
        )
        
        # Classification head for danger prediction
        self.danger_predictor = nn.Sequential(
            nn.Linear(vgg_output_size, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            tuple: (features, danger)
                - features: Predicted vehicle features
                - danger: Predicted collision danger probability
        """
        # Feature extraction
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Predict features
        features = self.feature_predictor(x)
        
        # Predict danger
        danger = torch.sigmoid(self.danger_predictor(x))
        
        return features, danger

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get all image and JSON paths
def get_paths():
    image_paths = []
    json_paths = []
    
    # Assuming image and JSON files have matching names (except extensions)
    for root, _, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                json_path = os.path.join(JSON_ROOT, base_name + '.json')
                
                # Only include pairs where both image and JSON exist
                if os.path.exists(json_path):
                    image_paths.append(img_path)
                    json_paths.append(json_path)
    
    return image_paths, json_paths

def train_bbox_model():
    """
    Train the bounding box feature prediction model.
    
    Process:
    1. Load and preprocess data
    2. Initialize model, optimizer, and loss functions
    3. Train for specified number of epochs
    4. Save best model and training curves
    
    Returns:
        BoundingBoxFeaturePredictor: Trained model
    """
    # Get data paths
    image_paths, json_paths = get_paths()
    print(f"Found {len(image_paths)} matching image-JSON pairs")
    
    # Split data
    train_img, val_img, train_json, val_json = train_test_split(
        image_paths, json_paths, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = BoundingBoxDataset(train_img, train_json, transform)
    val_dataset = BoundingBoxDataset(val_img, val_json, transform)
    
    print(f"Extracted {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    model = BoundingBoxFeaturePredictor().to(device)
    
    # Loss functions
    feature_criterion = nn.MSELoss()  # Regression loss for features
    danger_criterion = nn.BCELoss()   # Binary classification loss for danger
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Track metrics
    train_losses = []
    val_losses = []
    feature_losses = []
    danger_accuracies = []
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for bbox_images, features, is_dangerous in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            bbox_images = bbox_images.to(device)
            target_features = features.to(device)
            is_dangerous = is_dangerous.to(device)
            
            # Forward pass
            pred_features, pred_danger = model(bbox_images)
            
            # Calculate losses
            feature_loss = feature_criterion(pred_features, target_features)
            danger_loss = danger_criterion(pred_danger, is_dangerous)
            
            # Combined loss (you can adjust weights if needed)
            loss = feature_loss + danger_loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * bbox_images.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_feature_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for bbox_images, features, is_dangerous in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                bbox_images = bbox_images.to(device)
                target_features = features.to(device)
                is_dangerous = is_dangerous.to(device)
                
                # Forward pass
                pred_features, pred_danger = model(bbox_images)
                
                # Calculate losses
                feature_loss = feature_criterion(pred_features, target_features)
                danger_loss = danger_criterion(pred_danger, is_dangerous)
                loss = feature_loss + danger_loss
                
                val_loss += loss.item() * bbox_images.size(0)
                val_feature_loss += feature_loss.item() * bbox_images.size(0)
                
                # Calculate danger prediction accuracy
                predicted = (pred_danger >= 0.5).float()
                total += is_dangerous.size(0)
                correct += (predicted == is_dangerous).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_feature_loss = val_feature_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        feature_losses.append(val_feature_loss)
        
        danger_accuracy = correct / total
        danger_accuracies.append(danger_accuracy)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Feature Prediction Loss: {val_feature_loss:.4f}")
        print(f"Danger Prediction Accuracy: {danger_accuracy:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'feature_loss': val_feature_loss,
                'danger_accuracy': danger_accuracy
            }, MODEL_SAVE_PATH)
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(feature_losses, label='Feature Prediction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Feature Prediction Loss')
    
    plt.subplot(2, 2, 3)
    plt.plot(danger_accuracies, label='Danger Prediction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Danger Prediction Accuracy')
    
    plt.tight_layout()
    plt.savefig('bbox_training_curves.png')
    plt.close()
    
    return model

def evaluate_bbox_model(model_path):
    """
    Evaluate the trained model's performance.
    
    Computes and saves:
    - Feature prediction metrics (MAE, RMSE)
    - Danger prediction metrics (accuracy, precision, recall, F1)
    - Visualization of predictions
    
    Args:
        model_path (str): Path to the saved model checkpoint
        
    Returns:
        tuple: (feature_metrics, danger_metrics)
            - feature_metrics: Dict of feature-wise evaluation metrics
            - danger_metrics: Tuple of (accuracy, precision, recall, f1)
    """
    # Load the best model
    checkpoint = torch.load(model_path)
    model = BoundingBoxFeaturePredictor().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get validation data
    image_paths, json_paths = get_paths()
    _, val_img, _, val_json = train_test_split(
        image_paths, json_paths, test_size=0.2, random_state=42
    )
    
    val_dataset = BoundingBoxDataset(val_img, val_json, transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Evaluation metrics
    feature_errors = {name: [] for name in FEATURE_NAMES}
    all_pred_features = []
    all_true_features = []
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for bbox_images, features, is_dangerous in tqdm(val_loader, desc="Evaluating"):
            bbox_images = bbox_images.to(device)
            
            # Predict features and danger
            pred_features, pred_danger = model(bbox_images)
            
            # Move predictions to CPU for analysis
            pred_features = pred_features.cpu().numpy()
            true_features = features.numpy()
            
            all_pred_features.append(pred_features)
            all_true_features.append(true_features)
            
            # Calculate errors for each feature
            for i, name in enumerate(FEATURE_NAMES):
                errors = np.abs(pred_features[:, i] - true_features[:, i])
                feature_errors[name].extend(errors)
            
            # Collect danger predictions
            pred_danger = (pred_danger >= 0.5).cpu().numpy()
            y_true.extend(is_dangerous.numpy())
            y_pred.extend(pred_danger)
    
    # Calculate feature prediction metrics
    feature_metrics = {}
    for name in FEATURE_NAMES:
        mae = np.mean(feature_errors[name])
        rmse = np.sqrt(np.mean(np.square(feature_errors[name])))
        feature_metrics[name] = {
            'MAE': mae,
            'RMSE': rmse
        }
    
    # Calculate danger prediction metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("\nModel Evaluation Results")
    print("\nFeature Prediction Metrics:")
    for name, metrics in feature_metrics.items():
        print(f"{name}: MAE = {metrics['MAE']:.4f}, RMSE = {metrics['RMSE']:.4f}")
    
    print("\nDanger Prediction Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Save feature prediction results to CSV
    feature_results = []
    for name, metrics in feature_metrics.items():
        feature_results.append({
            'Feature': name,
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE']
        })
    
    pd.DataFrame(feature_results).to_csv('feature_prediction_results.csv', index=False)
    
    # Save danger prediction results to CSV
    danger_results = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    }
    
    pd.DataFrame(danger_results).to_csv('danger_prediction_results.csv', index=False)
    
    # Visualize feature predictions vs ground truth
    all_pred_features = np.vstack(all_pred_features)
    all_true_features = np.vstack(all_true_features)
    
    # Plot predicted vs actual for each feature
    plt.figure(figsize=(20, 15))
    for i, name in enumerate(FEATURE_NAMES):
        plt.subplot(4, 4, i+1)
        plt.scatter(all_true_features[:, i], all_pred_features[:, i], alpha=0.3)
        plt.plot([all_true_features[:, i].min(), all_true_features[:, i].max()], 
                [all_true_features[:, i].min(), all_true_features[:, i].max()], 'r--')
        plt.xlabel(f'Actual {name}')
        plt.ylabel(f'Predicted {name}')
        plt.title(f'{name} Prediction')
    
    plt.tight_layout()
    plt.savefig('feature_predictions.png')
    plt.close()
    
    return feature_metrics, (accuracy, precision, recall, f1)

def visualize_bbox_predictions(model_path, num_samples=5):
    """
    Generate visualizations of model predictions.
    
    Creates visualization showing:
    - Original image with bounding box
    - Cropped bounding box
    - Predicted vs actual features
    - Danger prediction
    
    Args:
        model_path (str): Path to the saved model checkpoint
        num_samples (int): Number of random samples to visualize
    """
    # Load the best model
    checkpoint = torch.load(model_path)
    model = BoundingBoxFeaturePredictor().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get validation data
    image_paths, json_paths = get_paths()
    _, val_img, _, val_json = train_test_split(
        image_paths, json_paths, test_size=0.2, random_state=42
    )
    
    val_dataset = BoundingBoxDataset(val_img, val_json, transform)
    
    # Randomly select samples to visualize
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 5*num_samples))
    
    for i, idx in enumerate(indices):
        # Get sample
        bbox_image, true_features, is_dangerous = val_dataset[idx]
        sample = val_dataset.samples[idx]
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            bbox_image_batch = bbox_image.unsqueeze(0).to(device)
            pred_features, pred_danger = model(bbox_image_batch)
            pred_features = pred_features.cpu().numpy()[0]
            pred_danger = pred_danger.cpu().numpy()[0][0]
        
        # Original image with bounding box
        orig_image = Image.open(sample['image_path'])
        axes[i, 0].imshow(orig_image)
        
        # Draw bounding box
        x, y, width, height = sample['bbox']
        import matplotlib.patches as patches
        rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                                edgecolor='r', facecolor='none')
        axes[i, 0].add_patch(rect)
        axes[i, 0].set_title(f"Original Image with Bounding Box\nDanger: {is_dangerous.item():.0f}, Predicted: {pred_danger:.4f}")
        axes[i, 0].axis('off')
        
        # Cropped bounding box
        # Convert from normalized tensor back to image
        inv_normalize = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
            transforms.ToPILImage()
        ])
        bbox_img_display = inv_normalize(bbox_image)
        
        axes[i, 1].imshow(bbox_img_display)
        axes[i, 1].set_title("Bounding Box Crop")
        axes[i, 1].axis('off')
        
        # Print feature comparisons
        feature_text = "Feature Predictions vs. Ground Truth:\n"
        for j, name in enumerate(FEATURE_NAMES):
            feature_text += f"{name}: {pred_features[j]:.4f} vs {true_features[j]:.4f}\n"
        
        axes[i, 1].text(1.05, 0.5, feature_text, transform=axes[i, 1].transAxes, 
                       fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('bbox_predictions_visualization.png')
    plt.close()

if __name__ == "__main__":
    print("\nTraining Bounding Box Feature Predictor...")
    model = train_bbox_model()
    
    print("\nEvaluating the model...")
    feature_metrics, danger_metrics = evaluate_bbox_model(MODEL_SAVE_PATH)
    
    print("\nGenerating prediction visualizations...")
    visualize_bbox_predictions(MODEL_SAVE_PATH)
    
    print("\nTraining and evaluation complete!")