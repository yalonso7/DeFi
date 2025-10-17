"""
Advanced ML Analysis Script for Election Fraud Detection
Features: PyTorch/TensorFlow models, statistical analysis, visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== PYTORCH NEURAL NETWORK MODEL ====================

class FraudDetectionNN(nn.Module):
    """Deep Neural Network for election fraud detection"""
    
    def __init__(self, input_size=10, hidden_sizes=[128, 64, 32, 16]):
        super(FraudDetectionNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ElectionDataset(Dataset):
    """Custom PyTorch Dataset for election data"""
    
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


# ==================== FEATURE ENGINEERING ====================

class AdvancedFeatureExtractor:
    """Extract sophisticated features for fraud detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def extract_all_features(self, election_df):
        """Extract comprehensive feature set"""
        features = []
        
        for _, row in election_df.iterrows():
            feature_vector = self.extract_region_features(row)
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_region_features(self, region_data):
        """Extract features from single region"""
        
        # Basic turnout features
        turnout_rate = region_data['votes_cast'] / max(region_data['registered_voters'], 1)
        turnout_deviation = abs(turnout_rate - 0.70)
        
        # Vote distribution features
        candidate_votes = [region_data.get(f'candidate_{i}_votes', 0) for i in range(1, 6)]
        vote_mean = np.mean(candidate_votes)
        vote_std = np.std(candidate_votes)
        vote_cv = vote_std / max(vote_mean, 1)
        vote_entropy = stats.entropy([v/sum(candidate_votes) for v in candidate_votes if v > 0])
        
        # Benford's Law analysis
        benford_score = self.calculate_benford_deviation(candidate_votes)
        
        # Statistical moments
        skewness = stats.skew(candidate_votes)
        kurtosis = stats.kurtosis(candidate_votes)
        
        # Temporal features (if available)
        time_variance = region_data.get('hourly_vote_variance', 0)
        
        # Demographic features
        population_density = region_data.get('population_density', 0)
        urban_ratio = region_data.get('urban_ratio', 0.5)
        
        # Anomaly flags
        overvote_flag = 1.0 if region_data['votes_cast'] > region_data['registered_voters'] else 0.0
        invalid_vote_ratio = region_data.get('invalid_votes', 0) / max(region_data['votes_cast'], 1)
        
        # Winner margin (potential indicator)
        sorted_votes = sorted(candidate_votes, reverse=True)
        winner_margin = (sorted_votes[0] - sorted_votes[1]) / sum(candidate_votes) if len(sorted_votes) > 1 else 0
        
        return [
            turnout_rate,
            turnout_deviation,
            vote_cv,
            vote_entropy,
            benford_score,
            skewness,
            kurtosis,
            time_variance,
            population_density,
            urban_ratio,
            overvote_flag,
            invalid_vote_ratio,
            winner_margin
        ]
    
    def calculate_benford_deviation(self, votes):
        """Calculate deviation from Benford's Law"""
        first_digits = [int(str(abs(int(v)))[0]) for v in votes if v > 0]
        if not first_digits:
            return 0.0
        
        # Expected Benford distribution
        benford_expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
        
        # Observed distribution
        observed = np.array([first_digits.count(d) / len(first_digits) for d in range(1, 10)])
        
        # Calculate Mean Absolute Deviation (MAD)
        mad = np.mean(np.abs(observed - benford_expected))
        return mad
    
    def fit_transform(self, data):
        """Fit scaler and transform data"""
        return self.scaler.fit_transform(data)
    
    def transform(self, data):
        """Transform data using fitted scaler"""
        return self.scaler.transform(data)


# ==================== TRAINING PIPELINE ====================

class FraudDetectionTrainer:
    """Complete training pipeline for fraud detection"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features).squeeze()
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = self.model(features).squeeze()
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50):
        """Complete training loop"""
        print("Starting training...")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        print("Training completed!")
        return self.history
    
    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {path}")


# ==================== ENSEMBLE METHODS ====================

class EnsembleFraudDetector:
    """Ensemble of multiple fraud detection methods"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.neural_net = None
    
    def fit(self, X, y=None):
        """Train ensemble models"""
        # Unsupervised anomaly detection
        self.isolation_forest.fit(X)
        
        # Supervised classification (if labels available)
        if y is not None:
            self.random_forest.fit(X, y)
    
    def predict(self, X):
        """Predict using ensemble voting"""
        # Get predictions from each model
        iso_pred = (self.isolation_forest.predict(X) == -1).astype(int)  # Anomalies = 1
        
        if hasattr(self.random_forest, 'classes_'):
            rf_pred = self.random_forest.predict(X)
        else:
            rf_pred = iso_pred
        
        # Simple voting
        ensemble_pred = ((iso_pred + rf_pred) >= 1).astype(int)
        return ensemble_pred
    
    def predict_proba(self, X):
        """Get fraud probability scores"""
        iso_scores = self.isolation_forest.score_samples(X)
        iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        
        if hasattr(self.random_forest, 'classes_'):
            rf_proba = self.random_forest.predict_proba(X)[:, 1]
        else:
            rf_proba = iso_scores
        
        # Average probabilities
        avg_proba = (1 - iso_scores + rf_proba) / 2
        return avg_proba


# ==================== VISUALIZATION FUNCTIONS ====================

class ElectionVisualizer:
    """Create comprehensive visualizations"""
    
    @staticmethod
    def plot_training_history(history):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss During Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Model Accuracy During Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved!")
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Fraud Detection')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved!")
    
    @staticmethod
    def plot_roc_curve(y_true, y_scores):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("ROC curve saved!")
    
    @staticmethod
    def plot_feature_importance(feature_names, importances):
        """Plot feature importance"""
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance for Fraud Detection')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved!")
    
    @staticmethod
    def plot_anomaly_distribution(predictions, scores):
        """Plot anomaly score distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Score distribution
        ax1.hist(scores[predictions == 0], bins=30, alpha=0.7, label='Normal', color='green')
        ax1.hist(scores[predictions == 1], bins=30, alpha=0.7, label='Fraud', color='red')
        ax1.set_xlabel('Fraud Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Fraud Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [scores[predictions == 0], scores[predictions == 1]]
        ax2.boxplot(data_to_plot, labels=['Normal', 'Fraud'])
        ax2.set_ylabel('Fraud Score')
        ax2.set_title('Fraud Score by Classification')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('anomaly_distribution.png', dpi=300, bbox_inches='tight')
        print("Anomaly distribution plot saved!")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=" * 60)
    print("ELECTION FRAUD DETECTION - ML ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Generate synthetic election data
    print("\n[1] Generating synthetic election data...")
    np.random.seed(42)
    n_regions = 200
    
    # Create synthetic dataset
    data = {
        'region_id': range(1, n_regions + 1),
        'registered_voters': np.random.randint(50000, 200000, n_regions),
        'votes_cast': np.random.randint(30000, 180000, n_regions),
        'candidate_1_votes': np.random.randint(10000, 60000, n_regions),
        'candidate_2_votes': np.random.randint(10000, 60000, n_regions),
        'candidate_3_votes': np.random.randint(5000, 40000, n_regions),
        'candidate_4_votes': np.random.randint(3000, 30000, n_regions),
        'invalid_votes': np.random.randint(100, 5000, n_regions),
        'hourly_vote_variance': np.random.uniform(0, 1, n_regions),
        'population_density': np.random.uniform(100, 10000, n_regions),
        'urban_ratio': np.random.uniform(0.3, 0.9, n_regions)
    }
    
    election_df = pd.DataFrame(data)
    
    # Inject some fraud cases
    fraud_indices = np.random.choice(n_regions, size=20, replace=False)
    election_df.loc[fraud_indices, 'votes_cast'] = election_df.loc[fraud_indices, 'registered_voters'] * 1.1
    
    # Create labels
    labels = np.zeros(n_regions)
    labels[fraud_indices] = 1
    
    print(f"Dataset created: {n_regions} regions, {labels.sum():.0f} fraudulent")
    
    # Extract features
    print("\n[2] Extracting features...")
    feature_extractor = AdvancedFeatureExtractor()
    features = feature_extractor.extract_all_features(election_df)
    features_scaled = feature_extractor.fit_transform(features)
    
    print(f"Feature matrix shape: {features_scaled.shape}")
    
    # Train PyTorch model
    print("\n[3] Training PyTorch Neural Network...")
    model = FraudDetectionNN(input_size=features_scaled.shape[1])
    
    # Create datasets
    train_size = int(0.8 * len(features_scaled))
    train_features, val_features = features_scaled[:train_size], features_scaled[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]
    
    train_dataset = ElectionDataset(train_features, train_labels)
    val_dataset = ElectionDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    trainer = FraudDetectionTrainer(model)
    history = trainer.train(train_loader, val_loader, epochs=30)
    
    # Train ensemble
    print("\n[4] Training Ensemble Models...")
    ensemble = EnsembleFraudDetector()
    ensemble.fit(features_scaled, labels)
    
    # Make predictions
    print("\n[5] Making Predictions...")
    ensemble_pred = ensemble.predict(features_scaled)
    ensemble_proba = ensemble.predict_proba(features_scaled)
    
    # Evaluation
    print("\n[6] Model Evaluation:")
    print("\nClassification Report:")
    print(classification_report(labels, ensemble_pred, target_names=['Normal', 'Fraud']))
    
    # Visualizations
    print("\n[7] Generating Visualizations...")
    visualizer = ElectionVisualizer()
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(labels, ensemble_pred)
    visualizer.plot_roc_curve(labels, ensemble_proba)
    visualizer.plot_anomaly_distribution(ensemble_pred, ensemble_proba)
    
    # Save model
    print("\n[8] Saving models...")
    trainer.save_model('fraud_detection_model.pth')
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - fraud_detection_model.pth")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - anomaly_distribution.png")
