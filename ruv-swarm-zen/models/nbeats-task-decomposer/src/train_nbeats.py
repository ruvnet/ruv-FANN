#!/usr/bin/env python3
"""
N-BEATS Task Decomposer Training Script

Trains the N-BEATS model for interpretable task decomposition with focus on:
- >88% decomposition accuracy (updated target)
- Clear interpretability with confidence intervals
- Support for multiple decomposition strategies (Agile, Waterfall, Feature-driven)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nbeats_model import NBEATSTaskDecomposer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskDecompositionDataset(Dataset):
    """Dataset for task decomposition training."""
    
    def __init__(self, data_path: str, max_subtasks: int = 16):
        self.max_subtasks = max_subtasks
        self.data = self._load_and_process_data(data_path)
        
    def _load_and_process_data(self, data_path: str):
        """Load and process training data."""
        data = []
        
        # Load all JSON files in data path
        data_dir = Path(data_path)
        for json_file in data_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                file_data = json.load(f)
                data.extend(file_data)
        
        logger.info(f"Loaded {len(data)} training examples")
        return data
    
    def _encode_task(self, task_data: Dict) -> torch.Tensor:
        """Encode task features into tensor."""
        features = []
        
        # Basic features
        features.append(task_data['complexity_score'])
        features.append(task_data['estimated_duration_hours'] / 1000.0)  # Normalize
        features.append(task_data['team_size'] / 10.0)  # Normalize
        
        # Task type encoding (one-hot)
        task_types = ['web_development', 'api_development', 'machine_learning', 
                      'data_processing', 'testing', 'enterprise_application', 
                      'saas_platform', 'mobile_development']
        task_type_vector = [0.0] * len(task_types)
        if task_data['task_type'] in task_types:
            task_type_vector[task_types.index(task_data['task_type'])] = 1.0
        features.extend(task_type_vector)
        
        # Strategy encoding (one-hot)
        strategies = ['waterfall', 'agile', 'feature_driven', 'component_based']
        strategy_vector = [0.0] * len(strategies)
        if task_data['decomposition_strategy'] in strategies:
            strategy_vector[strategies.index(task_data['decomposition_strategy'])] = 1.0
        features.extend(strategy_vector)
        
        # Text features (simplified - would use proper NLP in production)
        text_features = [
            len(task_data['original_task']) / 1000.0,  # Text length
            task_data['original_task'].count(' ') / 100.0,  # Word count
        ]
        features.extend(text_features)
        
        # Pad to fixed size
        while len(features) < 64:
            features.append(0.0)
        
        return torch.tensor(features[:64], dtype=torch.float32)
    
    def _create_targets(self, task_data: Dict) -> Dict[str, torch.Tensor]:
        """Create target tensors from task data."""
        subtasks = task_data['subtasks']
        num_subtasks = len(subtasks)
        
        # Complexity scores
        complexity_scores = torch.zeros(self.max_subtasks)
        for i, subtask in enumerate(subtasks[:self.max_subtasks]):
            complexity_scores[i] = subtask['complexity']
        
        # Duration estimates
        duration_estimates = torch.zeros(self.max_subtasks)
        for i, subtask in enumerate(subtasks[:self.max_subtasks]):
            duration_estimates[i] = subtask['duration_hours'] / 10.0  # Normalize
        
        # Dependency matrix
        dependency_matrix = torch.zeros(self.max_subtasks, self.max_subtasks)
        subtask_id_map = {subtask['id']: idx for idx, subtask in enumerate(subtasks)}
        for i, subtask in enumerate(subtasks[:self.max_subtasks]):
            for dep_id in subtask.get('dependencies', []):
                if dep_id in subtask_id_map and subtask_id_map[dep_id] < self.max_subtasks:
                    dependency_matrix[i, subtask_id_map[dep_id]] = 1.0
        
        # Subtask count
        subtask_count = torch.tensor([min(num_subtasks, self.max_subtasks) / self.max_subtasks], 
                                   dtype=torch.float32)
        
        return {
            'complexity_scores': complexity_scores,
            'duration_estimates': duration_estimates,
            'dependency_matrix': dependency_matrix,
            'subtask_count': subtask_count
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        task_data = self.data[idx]
        features = self._encode_task(task_data)
        targets = self._create_targets(task_data)
        
        return features, targets

class EnhancedTaskDecompositionDataset(Dataset):
    """Enhanced dataset that can handle both split data and task decomposition data."""
    
    def __init__(self, data_paths: List[str], max_subtasks: int = 16, use_split_format: bool = False):
        self.max_subtasks = max_subtasks
        self.use_split_format = use_split_format
        self.data = self._load_and_process_data(data_paths)
        
    def _load_and_process_data(self, data_paths: List[str]):
        """Load and process training data from multiple sources."""
        all_data = []
        
        for data_path in data_paths:
            if self.use_split_format:
                # Load split format data (CSV/JSON from splits directory)
                data_dir = Path(data_path)
                if data_dir.is_file() and data_path.endswith('.json'):
                    with open(data_path, 'r') as f:
                        split_data = json.load(f)
                        all_data.extend(split_data)
            else:
                # Load task decomposition format data
                data_dir = Path(data_path)
                if data_dir.is_dir():
                    for json_file in data_dir.glob("*.json"):
                        with open(json_file, 'r') as f:
                            file_data = json.load(f)
                            if isinstance(file_data, list):
                                all_data.extend(file_data)
                            else:
                                all_data.append(file_data)
                elif data_dir.is_file() and data_path.endswith('.json'):
                    with open(data_path, 'r') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            all_data.extend(file_data)
                        else:
                            all_data.append(file_data)
        
        logger.info(f"Loaded {len(all_data)} training examples from {len(data_paths)} sources")
        return all_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # For now, use the same logic as the original dataset
        # This is a simplified version - in production you'd adapt based on data format
        task_data = self.data[idx]
        
        # Create simple features and targets for training
        features = self._encode_task_simple(task_data)
        targets = self._create_targets_simple(task_data)
        
        return features, targets
    
    def _encode_task_simple(self, task_data: Dict) -> torch.Tensor:
        """Simple encoding for task features."""
        features = []
        
        # Basic features with defaults
        features.append(task_data.get('complexity_score', 0.5))
        features.append(task_data.get('estimated_duration_hours', 40) / 1000.0)
        features.append(task_data.get('team_size', 3) / 10.0)
        
        # Simple text length features
        text_len = len(task_data.get('original_task', '')) / 1000.0
        features.append(text_len)
        
        # Strategy encoding
        strategy = task_data.get('decomposition_strategy', 'agile')
        strategies = ['waterfall', 'agile', 'feature_driven', 'component_based']
        strategy_vector = [0.0] * len(strategies)
        if strategy in strategies:
            strategy_vector[strategies.index(strategy)] = 1.0
        features.extend(strategy_vector)
        
        # Task type encoding
        task_type = task_data.get('task_type', 'web_development')
        task_types = ['web_development', 'api_development', 'machine_learning', 
                      'data_processing', 'testing', 'enterprise_application', 
                      'saas_platform', 'mobile_development']
        task_type_vector = [0.0] * len(task_types)
        if task_type in task_types:
            task_type_vector[task_types.index(task_type)] = 1.0
        features.extend(task_type_vector)
        
        # Pad to 64 features
        while len(features) < 64:
            features.append(0.0)
        
        return torch.tensor(features[:64], dtype=torch.float32)
    
    def _create_targets_simple(self, task_data: Dict) -> Dict[str, torch.Tensor]:
        """Create simple targets from task data."""
        subtasks = task_data.get('subtasks', [])
        max_subtasks = self.max_subtasks
        
        # Complexity scores
        complexity_scores = torch.zeros(max_subtasks)
        for i, subtask in enumerate(subtasks[:max_subtasks]):
            complexity_scores[i] = subtask.get('complexity', 0.5)
        
        # Duration estimates
        duration_estimates = torch.zeros(max_subtasks)
        for i, subtask in enumerate(subtasks[:max_subtasks]):
            duration_estimates[i] = subtask.get('duration_hours', 8) / 10.0
        
        # Simple dependency matrix (random for now)
        dependency_matrix = torch.zeros(max_subtasks, max_subtasks)
        
        # Subtask count
        subtask_count = torch.tensor([min(len(subtasks), max_subtasks) / max_subtasks], 
                                   dtype=torch.float32)
        
        return {
            'complexity_scores': complexity_scores,
            'duration_estimates': duration_estimates,
            'dependency_matrix': dependency_matrix,
            'subtask_count': subtask_count
        }

class NBEATSTrainer:
    """Trainer for N-BEATS task decomposer."""
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        self.criterion = nn.MSELoss()
        self.bce_criterion = nn.BCELoss()
        self.ce_criterion = nn.CrossEntropyLoss()
        
        # Optimizer with improved parameters for interpretable training
        self.optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.accuracy_scores = []
        self.interpretability_scores = []
        self.best_val_loss = float('inf')
        self.best_accuracy = 0.0
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (features, targets) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            
            outputs = self.model(features)
            
            # Calculate loss
            loss, loss_components = self._calculate_loss(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
                if batch_idx % 50 == 0:  # Detailed loss breakdown every 50 batches
                    logger.info(f'  - Complexity: {loss_components["complexity_loss"]:.4f}')
                    logger.info(f'  - Duration: {loss_components["duration_loss"]:.4f}')
                    logger.info(f'  - Dependencies: {loss_components["dependency_loss"]:.4f}')
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                outputs = self.model(features)
                loss, _ = self._calculate_loss(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            return True  # New best model
        
        return False
    
    def _calculate_loss(self, outputs, targets):
        """Calculate combined loss with interpretability components."""
        # Core decomposition losses
        complexity_loss = self.criterion(outputs['complexity_scores'], targets['complexity_scores'])
        duration_loss = self.criterion(outputs['duration_estimates'], targets['duration_estimates'])
        dependency_loss = self.bce_criterion(outputs['dependency_matrix'], targets['dependency_matrix'])
        count_loss = self.criterion(outputs['subtask_count'], targets['subtask_count'])
        
        # Quality estimation loss
        if 'quality_estimates' in outputs and 'quality_targets' in targets:
            quality_loss = self.criterion(outputs['quality_estimates'], targets['quality_targets'])
        else:
            quality_loss = torch.tensor(0.0, device=self.device)
        
        # Interpretability losses (if targets available)
        strategy_loss = torch.tensor(0.0, device=self.device)
        task_type_loss = torch.tensor(0.0, device=self.device)
        confidence_loss = torch.tensor(0.0, device=self.device)
        
        if 'strategy_targets' in targets:
            strategy_loss = self.ce_criterion(outputs['strategy_probabilities'], targets['strategy_targets'])
        
        if 'task_type_targets' in targets:
            task_type_loss = self.ce_criterion(outputs['task_type_probabilities'], targets['task_type_targets'])
        
        if 'confidence_targets' in targets:
            confidence_loss = self.criterion(outputs['confidence_score'], targets['confidence_targets'])
        
        # Weighted combination of losses
        total_loss = (
            1.0 * complexity_loss +
            1.0 * duration_loss + 
            0.8 * dependency_loss +
            0.6 * count_loss +
            0.5 * quality_loss +
            0.3 * strategy_loss +
            0.3 * task_type_loss +
            0.2 * confidence_loss
        )
        
        return total_loss, {
            'complexity_loss': complexity_loss.item(),
            'duration_loss': duration_loss.item(),
            'dependency_loss': dependency_loss.item(),
            'count_loss': count_loss.item(),
            'quality_loss': quality_loss.item(),
            'strategy_loss': strategy_loss.item(),
            'task_type_loss': task_type_loss.item(),
            'confidence_loss': confidence_loss.item()
        }
    
    def train(self, num_epochs=50):
        """Train the model."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch()
            is_best = self.validate()
            val_loss = self.val_losses[-1]
            
            self.scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if is_best:
                logger.info("New best model!")
                self.save_model('best_model.pth')
        
        logger.info("Training completed!")
    
    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
        
    def evaluate_accuracy(self):
        """Evaluate comprehensive model accuracy on validation set."""
        self.model.eval()
        all_metrics = {
            'complexity_accuracies': [],
            'duration_accuracies': [],
            'dependency_accuracies': [],
            'count_accuracies': [],
            'overall_accuracies': []
        }
        
        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                outputs = self.model(features)
                
                # Complexity accuracy (MAE-based)
                complexity_mae = mean_absolute_error(
                    targets['complexity_scores'].cpu().numpy().flatten(),
                    outputs['complexity_scores'].cpu().numpy().flatten()
                )
                complexity_accuracy = max(0.0, 1.0 - complexity_mae)
                all_metrics['complexity_accuracies'].append(complexity_accuracy)
                
                # Duration accuracy (MAPE-based)  
                duration_targets = targets['duration_estimates'].cpu().numpy().flatten()
                duration_preds = outputs['duration_estimates'].cpu().numpy().flatten()
                # Avoid division by zero
                duration_targets_safe = np.where(duration_targets == 0, 1e-8, duration_targets)
                duration_mape = np.mean(np.abs((duration_targets - duration_preds) / duration_targets_safe))
                duration_accuracy = max(0.0, 1.0 - min(duration_mape, 1.0))
                all_metrics['duration_accuracies'].append(duration_accuracy)
                
                # Dependency accuracy (binary classification accuracy)
                dependency_targets = targets['dependency_matrix'].cpu().numpy().flatten()
                dependency_preds = (outputs['dependency_matrix'].cpu().numpy().flatten() > 0.5).astype(int)
                dependency_accuracy = accuracy_score(dependency_targets > 0.5, dependency_preds)
                all_metrics['dependency_accuracies'].append(dependency_accuracy)
                
                # Subtask count accuracy
                count_targets = targets['subtask_count'].cpu().numpy().flatten()
                count_preds = outputs['subtask_count'].cpu().numpy().flatten()
                count_mae = mean_absolute_error(count_targets, count_preds)
                count_accuracy = max(0.0, 1.0 - count_mae)
                all_metrics['count_accuracies'].append(count_accuracy)
                
                # Overall accuracy (weighted combination)
                overall_accuracy = (
                    0.35 * complexity_accuracy +
                    0.25 * duration_accuracy +
                    0.25 * dependency_accuracy +
                    0.15 * count_accuracy
                )
                all_metrics['overall_accuracies'].append(overall_accuracy)
        
        # Calculate average metrics
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        
        logger.info("=== Model Accuracy Report ===")
        logger.info(f"Overall Accuracy: {avg_metrics['overall_accuracies']:.4f} ({avg_metrics['overall_accuracies']*100:.1f}%)")
        logger.info(f"Complexity Accuracy: {avg_metrics['complexity_accuracies']:.4f} ({avg_metrics['complexity_accuracies']*100:.1f}%)")
        logger.info(f"Duration Accuracy: {avg_metrics['duration_accuracies']:.4f} ({avg_metrics['duration_accuracies']*100:.1f}%)")
        logger.info(f"Dependency Accuracy: {avg_metrics['dependency_accuracies']:.4f} ({avg_metrics['dependency_accuracies']*100:.1f}%)")
        logger.info(f"Count Accuracy: {avg_metrics['count_accuracies']:.4f} ({avg_metrics['count_accuracies']*100:.1f}%)")
        
        return avg_metrics['overall_accuracies'], avg_metrics

def create_interpretability_report(model, test_loader, device, save_path):
    """Generate comprehensive interpretability report."""
    model.eval()
    interpretability_data = {
        'strategy_recommendations': [],
        'task_type_detections': [],
        'confidence_scores': [],
        'feature_importances': [],
        'trend_analysis': [],
        'seasonal_analysis': [],
        'generic_analysis': []
    }
    
    strategy_names = ['waterfall', 'agile', 'feature_driven', 'component_based']
    task_type_names = ['web_development', 'api_development', 'machine_learning', 
                      'data_processing', 'testing', 'enterprise_application',
                      'saas_platform', 'mobile_development']
    
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(test_loader):
            if batch_idx >= 10:  # Limit to first 10 batches for report
                break
                
            features = features.to(device)
            outputs = model(features)
            
            # Collect interpretability data
            batch_size = features.size(0)
            for i in range(batch_size):
                # Strategy recommendations
                strategy_probs = outputs['strategy_probabilities'][i].cpu().numpy()
                strategy_idx = np.argmax(strategy_probs)
                interpretability_data['strategy_recommendations'].append({
                    'recommended_strategy': strategy_names[strategy_idx],
                    'confidence': float(strategy_probs[strategy_idx]),
                    'probabilities': {name: float(prob) for name, prob in zip(strategy_names, strategy_probs)}
                })
                
                # Task type detection
                task_type_probs = outputs['task_type_probabilities'][i].cpu().numpy()
                task_type_idx = np.argmax(task_type_probs)
                interpretability_data['task_type_detections'].append({
                    'detected_type': task_type_names[task_type_idx],
                    'confidence': float(task_type_probs[task_type_idx]),
                    'probabilities': {name: float(prob) for name, prob in zip(task_type_names, task_type_probs)}
                })
                
                # Confidence scores
                interpretability_data['confidence_scores'].append(float(outputs['confidence_score'][i].cpu().item()))
                
                # Feature importance
                feature_importance = outputs['feature_importance'][i].cpu().numpy()
                interpretability_data['feature_importances'].append(feature_importance.tolist())
                
                # Component analysis
                interpretability_data['trend_analysis'].append(float(torch.mean(outputs['trend_forecast'][i]).cpu().item()))
                interpretability_data['seasonal_analysis'].append(float(torch.mean(outputs['seasonal_forecast'][i]).cpu().item()))
                interpretability_data['generic_analysis'].append(float(torch.mean(outputs['generic_forecast'][i]).cpu().item()))
    
    # Generate summary statistics
    summary = {
        'total_samples_analyzed': len(interpretability_data['confidence_scores']),
        'average_confidence': float(np.mean(interpretability_data['confidence_scores'])),
        'strategy_distribution': {},
        'task_type_distribution': {},
        'component_influence': {
            'trend_avg': float(np.mean(interpretability_data['trend_analysis'])),
            'seasonal_avg': float(np.mean(interpretability_data['seasonal_analysis'])),
            'generic_avg': float(np.mean(interpretability_data['generic_analysis']))
        }
    }
    
    # Calculate distributions
    for strategy_rec in interpretability_data['strategy_recommendations']:
        strategy = strategy_rec['recommended_strategy']
        summary['strategy_distribution'][strategy] = summary['strategy_distribution'].get(strategy, 0) + 1
    
    for task_type_det in interpretability_data['task_type_detections']:
        task_type = task_type_det['detected_type']
        summary['task_type_distribution'][task_type] = summary['task_type_distribution'].get(task_type, 0) + 1
    
    # Create full report
    report = {
        'interpretability_summary': summary,
        'detailed_analysis': interpretability_data,
        'generated_at': pd.Timestamp.now().isoformat(),
        'model_info': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_type': 'N-BEATS Task Decomposer',
            'interpretability_features': [
                'Strategy recommendation with confidence',
                'Task type detection',
                'Feature importance analysis',
                'Trend/Seasonal/Generic component analysis',
                'Confidence scoring for all predictions'
            ]
        }
    }
    
    # Save report
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Interpretability report saved to {save_path}")
    return report

def main():
    """Main training function with comprehensive evaluation."""
    logger.info("=== Starting N-BEATS Task Decomposer Training ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data from multiple sources
    data_paths = [
        "/workspaces/ruv-FANN/ruv-swarm/models/nbeats-task-decomposer/training-data",
        "/workspaces/ruv-FANN/ruv-swarm/training-data/splits/nbeats/train.json"
    ]
    
    # Try to load task decomposition data first
    try:
        dataset = EnhancedTaskDecompositionDataset(data_paths[:1], use_split_format=False)
        logger.info("Using task decomposition format data")
    except Exception as e:
        logger.warning(f"Failed to load task decomposition data: {e}")
        # Fallback to original dataset
        try:
            dataset = TaskDecompositionDataset(data_paths[0])
            logger.info("Using original dataset format")
        except Exception as e2:
            logger.error(f"Failed to load any training data: {e2}")
            return
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders with optimal batch size
    batch_size = min(16, len(train_dataset) // 10)  # Dynamic batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")
    logger.info(f"Batch size: {batch_size}")
    
    # Create enhanced N-BEATS model
    model = NBEATSTaskDecomposer(
        input_size=64,
        max_subtasks=16,
        hidden_size=256,
        num_strategies=4,
        num_task_types=8
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Train model with enhanced training
    trainer = NBEATSTrainer(model, train_loader, val_loader, device)
    
    # Train for more epochs with better monitoring
    num_epochs = 75  # Increased for better convergence
    trainer.train(num_epochs=num_epochs)
    
    # Comprehensive evaluation
    logger.info("=== Final Model Evaluation ===")
    overall_accuracy, detailed_metrics = trainer.evaluate_accuracy()
    
    # Check if target accuracy is achieved
    target_accuracy = 0.88  # Updated target
    if overall_accuracy >= target_accuracy:
        logger.info(f"✅ TARGET ACHIEVED: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%) ≥ {target_accuracy*100:.0f}%")
        target_met = True
    else:
        logger.info(f"❌ Target not achieved: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%) < {target_accuracy*100:.0f}%")
        target_met = False
    
    # Save model with metadata
    model_metadata = {
        'training_date': pd.Timestamp.now().isoformat(),
        'target_accuracy': target_accuracy,
        'achieved_accuracy': overall_accuracy,
        'target_met': target_met,
        'training_epochs': num_epochs,
        'model_parameters': total_params,
        'device_used': str(device)
    }
    
    model.save_model('best_model.pth', model_metadata)
    
    # Generate interpretability report
    interpretability_report = create_interpretability_report(
        model, val_loader, device, 'interpretability_report.json'
    )
    
    # Save comprehensive training results
    results = {
        'training_summary': {
            'final_accuracy': float(overall_accuracy),
            'target_accuracy': target_accuracy,
            'target_met': target_met,
            'training_epochs': num_epochs,
            'model_parameters': total_params,
            'training_samples': train_size,
            'validation_samples': val_size
        },
        'detailed_metrics': {k: float(v) for k, v in detailed_metrics.items()},
        'training_history': {
            'train_losses': [float(loss) for loss in trainer.train_losses],
            'val_losses': [float(loss) for loss in trainer.val_losses],
            'best_val_loss': float(trainer.best_val_loss)
        },
        'interpretability_summary': interpretability_report['interpretability_summary'],
        'model_config': {
            'input_size': 64,
            'max_subtasks': 16,
            'hidden_size': 256,
            'num_strategies': 4,
            'num_task_types': 8
        },
        'training_metadata': model_metadata
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    logger.info("=== Training Complete ===")
    logger.info(f"Final Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)")
    logger.info(f"Target Met: {'✅ YES' if target_met else '❌ NO'}")
    logger.info(f"Model saved: best_model.pth")
    logger.info(f"Results saved: training_results.json")
    logger.info(f"Interpretability report: interpretability_report.json")
    
    return {
        'accuracy': overall_accuracy,
        'target_met': target_met,
        'model_path': 'best_model.pth'
    }

if __name__ == "__main__":
    main()