#!/usr/bin/env python3
"""
Pattern Detector Trainer Utilities
Advanced training utilities for TCN pattern detection with optimization and monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    gradient_clip_norm: float = 1.0
    warmup_epochs: int = 5
    target_accuracy: float = 0.90
    target_inference_time_ms: float = 15.0

@dataclass 
class ModelConfig:
    """Model architecture configuration"""
    input_dim: int = 128
    vocab_size: int = 8192
    embed_dim: int = 128
    num_channels: List[int] = None
    kernel_size: int = 3
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.num_channels is None:
            self.num_channels = [64, 128, 256, 256, 128, 64, 32]

class PatternDetectorTrainer:
    """Advanced trainer for TCN pattern detection models"""
    
    def __init__(self, model, train_config: TrainingConfig, model_config: ModelConfig):
        self.model = model
        self.train_config = train_config
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.best_inference_time = float('inf')
        self.training_history = []
        
        # Optimization components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_functions()
        
    def _setup_optimizer(self):
        """Setup optimizer with advanced configuration"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
    def _setup_loss_functions(self):
        """Setup loss functions for multi-task learning"""
        self.design_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.anti_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))  # Handle class imbalance
        self.refactor_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.5))
        
        # Adaptive loss weighting
        self.loss_weights = nn.Parameter(torch.ones(3))
        
    def compute_multi_task_loss(self, outputs, targets):
        """Compute weighted multi-task loss"""
        design_loss = self.design_loss_fn(outputs['design_logits'], targets['design_labels'].argmax(dim=1))
        anti_loss = self.anti_loss_fn(outputs['anti_logits'], targets['anti_labels'])
        refactor_loss = self.refactor_loss_fn(outputs['refactor_logits'], targets['refactor_labels'])
        
        # Adaptive weighting with uncertainty
        weights = F.softmax(self.loss_weights, dim=0)
        total_loss = (weights[0] * design_loss + 
                     weights[1] * anti_loss + 
                     weights[2] * refactor_loss)
        
        return total_loss, {
            'design_loss': design_loss.item(),
            'anti_loss': anti_loss.item(), 
            'refactor_loss': refactor_loss.item(),
            'total_loss': total_loss.item(),
            'loss_weights': weights.detach().cpu().numpy()
        }
    
    def train_epoch(self, train_loader):
        """Train model for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'design_loss': [],
            'anti_loss': [],
            'refactor_loss': [],
            'total_loss': []
        }
        
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Move data to device
            inputs = self._get_batch_input(batch).to(self.device)
            targets = {
                'design_labels': batch['design_labels'].to(self.device),
                'anti_labels': batch['anti_labels'].to(self.device),
                'refactor_labels': batch['refactor_labels'].to(self.device)
            }
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            loss, loss_dict = self.compute_multi_task_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.gradient_clip_norm)
            self.optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            for key, value in loss_dict.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)
        
        # Update learning rate
        self.scheduler.step()
        
        # Compute epoch averages
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        avg_metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        return avg_metrics
    
    def evaluate_model(self, eval_loader):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_predictions = {
            'design': [], 'anti': [], 'refactor': []
        }
        all_labels = {
            'design': [], 'anti': [], 'refactor': []
        }
        inference_times = []
        
        with torch.no_grad():
            for batch in eval_loader:
                start_time = time.time()
                
                inputs = self._get_batch_input(batch).to(self.device)
                outputs = self.model(inputs)
                
                batch_inference_time = (time.time() - start_time) * 1000 / inputs.size(0)
                inference_times.append(batch_inference_time)
                
                # Collect predictions and labels
                all_predictions['design'].append(outputs['design_patterns'].cpu().numpy())
                all_predictions['anti'].append(outputs['anti_patterns'].cpu().numpy())
                all_predictions['refactor'].append(outputs['refactoring_ops'].cpu().numpy())
                
                all_labels['design'].append(batch['design_labels'].cpu().numpy())
                all_labels['anti'].append(batch['anti_labels'].cpu().numpy())
                all_labels['refactor'].append(batch['refactor_labels'].cpu().numpy())
        
        # Concatenate all predictions and labels
        predictions = {key: np.concatenate(vals, axis=0) for key, vals in all_predictions.items()}
        labels = {key: np.concatenate(vals, axis=0) for key, vals in all_labels.items()}
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, labels)
        metrics['avg_inference_time_ms'] = np.mean(inference_times)
        metrics['inference_time_std'] = np.std(inference_times)
        
        return metrics
    
    def _get_batch_input(self, batch):
        """Extract input tensor from batch (handles both token IDs and features)"""
        if 'input_ids' in batch:
            return batch['input_ids']
        elif 'input_features' in batch:
            return batch['input_features']
        else:
            raise ValueError("Batch must contain either 'input_ids' or 'input_features'")
    
    def _calculate_metrics(self, predictions, labels):
        """Calculate comprehensive evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        metrics = {}
        
        # Design patterns (multi-class)
        design_pred_classes = np.argmax(predictions['design'], axis=1)
        design_true_classes = np.argmax(labels['design'], axis=1)
        metrics['design_accuracy'] = accuracy_score(design_true_classes, design_pred_classes)
        
        design_prec, design_rec, design_f1, _ = precision_recall_fscore_support(
            design_true_classes, design_pred_classes, average='weighted', zero_division=0
        )
        metrics['design_precision'] = design_prec
        metrics['design_recall'] = design_rec
        metrics['design_f1'] = design_f1
        
        # Anti-patterns (multi-label)
        anti_pred_binary = (predictions['anti'] > 0.5).astype(int)
        anti_labels_binary = labels['anti'].astype(int)
        
        # Handle case where all labels are zero
        if anti_labels_binary.sum() > 0:
            metrics['anti_accuracy'] = accuracy_score(anti_labels_binary, anti_pred_binary)
            try:
                metrics['anti_auc'] = roc_auc_score(anti_labels_binary, predictions['anti'], average='macro')
            except ValueError:
                metrics['anti_auc'] = 0.0
        else:
            metrics['anti_accuracy'] = 1.0 if anti_pred_binary.sum() == 0 else 0.0
            metrics['anti_auc'] = 0.0
        
        # Refactoring (multi-label)
        refactor_pred_binary = (predictions['refactor'] > 0.5).astype(int)
        refactor_labels_binary = labels['refactor'].astype(int)
        
        if refactor_labels_binary.sum() > 0:
            metrics['refactor_accuracy'] = accuracy_score(refactor_labels_binary, refactor_pred_binary)
            try:
                metrics['refactor_auc'] = roc_auc_score(refactor_labels_binary, predictions['refactor'], average='macro')
            except ValueError:
                metrics['refactor_auc'] = 0.0
        else:
            metrics['refactor_accuracy'] = 1.0 if refactor_pred_binary.sum() == 0 else 0.0
            metrics['refactor_auc'] = 0.0
        
        # Overall metrics
        metrics['overall_accuracy'] = (metrics['design_accuracy'] + 
                                     metrics['anti_accuracy'] + 
                                     metrics['refactor_accuracy']) / 3
        
        return metrics
    
    def train(self, train_loader, val_loader=None, save_path=None):
        """Full training loop with validation and early stopping"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        self.model.to(self.device)
        epochs_without_improvement = 0
        
        for epoch in range(self.train_config.max_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            epoch_info = {
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': None
            }
            
            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate_model(val_loader)
                epoch_info['val_metrics'] = val_metrics
                
                current_accuracy = val_metrics['overall_accuracy']
                current_inference_time = val_metrics['avg_inference_time_ms']
                
                print(f"Epoch {epoch+1:3d}/{self.train_config.max_epochs}: "
                      f"Loss={train_metrics['total_loss']:.4f}, "
                      f"ValAcc={current_accuracy:.3f}, "
                      f"InfTime={current_inference_time:.2f}ms")
                
                # Check for improvement
                if current_accuracy > self.best_accuracy:
                    self.best_accuracy = current_accuracy
                    self.best_inference_time = current_inference_time
                    epochs_without_improvement = 0
                    
                    # Save best model
                    if save_path:
                        self.save_checkpoint(save_path, epoch_info)
                        print(f"     New best model saved! Accuracy: {self.best_accuracy:.3f}")
                else:
                    epochs_without_improvement += 1
                
                # Early stopping
                if epochs_without_improvement >= self.train_config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1:3d}/{self.train_config.max_epochs}: "
                      f"Loss={train_metrics['total_loss']:.4f}")
            
            # Store training history
            self.training_history.append(epoch_info)
        
        print("=" * 60)
        print("Training completed!")
        return self.training_history
    
    def save_checkpoint(self, save_path, epoch_info):
        """Save model checkpoint with training info"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'best_inference_time': self.best_inference_time,
            'train_config': self.train_config.__dict__,
            'model_config': self.model_config.__dict__,
            'training_history': self.training_history,
            'epoch_info': epoch_info
        }
        
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.best_inference_time = checkpoint['best_inference_time']
        self.training_history = checkpoint.get('training_history', [])
        
        return checkpoint
    
    def generate_performance_report(self, test_metrics, save_path=None):
        """Generate comprehensive performance report"""
        report = {
            'model_name': 'EnhancedTCNPatternDetector',
            'training_completed': True,
            'training_config': self.train_config.__dict__,
            'model_config': self.model_config.__dict__,
            'best_epoch': self.current_epoch,
            'best_metrics': {
                'accuracy': self.best_accuracy,
                'inference_time_ms': self.best_inference_time
            },
            'final_test_metrics': test_metrics,
            'targets_achieved': {
                'accuracy': test_metrics['overall_accuracy'] >= self.train_config.target_accuracy,
                'inference_time': test_metrics['avg_inference_time_ms'] <= self.train_config.target_inference_time_ms
            },
            'training_history': self.training_history,
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024),
            'total_parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Performance report saved to: {save_path}")
        
        return report

def optimize_for_inference(model, example_input, optimization_level='standard'):
    """Optimize model for fast inference"""
    model.eval()
    
    if optimization_level == 'standard':
        # Standard optimizations
        model = torch.jit.trace(model, example_input)
        model = torch.jit.optimize_for_inference(model)
        
    elif optimization_level == 'aggressive':
        # More aggressive optimizations
        model = torch.jit.trace(model, example_input)
        model = torch.jit.optimize_for_inference(model)
        
        # Freeze model
        model = torch.jit.freeze(model)
        
        # Additional graph optimizations
        torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
    
    return model

def benchmark_inference_speed(model, example_input, num_iterations=1000, warmup_iterations=100):
    """Benchmark model inference speed"""
    model.eval()
    device = next(model.parameters()).device
    example_input = example_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(example_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(example_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean_inference_time_ms': np.mean(times),
        'std_inference_time_ms': np.std(times),
        'min_inference_time_ms': np.min(times),
        'max_inference_time_ms': np.max(times),
        'p95_inference_time_ms': np.percentile(times, 95),
        'p99_inference_time_ms': np.percentile(times, 99)
    }