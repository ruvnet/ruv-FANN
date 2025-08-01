#!/usr/bin/env python3
"""
N-BEATS Task Decomposer Model Implementation

A comprehensive N-BEATS architecture specifically designed for interpretable
task decomposition with support for multiple decomposition strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class NBEATSBlock(nn.Module):
    """
    Individual N-BEATS block with interpretable basis expansion.
    """
    
    def __init__(self, input_size: int, theta_size: int, basis_size: int, 
                 num_layers: int = 4, hidden_size: int = 256):
        super().__init__()
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_size = basis_size
        
        # Fully connected layers for feature extraction
        layers = []
        layer_sizes = [input_size] + [hidden_size] * (num_layers - 1) + [theta_size]
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation on last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        
        self.theta_layer = nn.Sequential(*layers)
        
        # Basis expansion for interpretability
        self.basis_expansion = nn.Linear(theta_size, basis_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both forecast and backcast.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            forecast: Forward prediction
            backcast: Backward residual
        """
        theta = self.theta_layer(x)
        basis_output = self.basis_expansion(theta)
        
        # Split basis output into forecast and backcast
        forecast_size = basis_output.size(-1) // 2
        forecast = basis_output[..., :forecast_size]
        backcast = basis_output[..., forecast_size:]
        
        return forecast, backcast

class TrendStack(nn.Module):
    """
    Trend stack for capturing long-term complexity patterns and dependencies.
    """
    
    def __init__(self, input_size: int, num_blocks: int = 3, 
                 theta_size: int = 128, basis_size: int = None):
        super().__init__()
        
        # Set basis_size to match input_size for proper residual connections
        if basis_size is None:
            basis_size = input_size * 2  # Double for forecast/backcast split
            
        self.input_size = input_size
        self.basis_size = basis_size
        
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_size, theta_size, basis_size)
            for _ in range(num_blocks)
        ])
        
        # Polynomial trend basis functions
        self.trend_coefficients = nn.Parameter(torch.randn(input_size, 4))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input through trend stack.
        
        Returns:
            Dictionary containing trend forecasts and interpretable components
        """
        forecasts = []
        backcasts = []
        residual = x
        
        for block in self.blocks:
            forecast, backcast = block(residual)
            forecasts.append(forecast)
            backcasts.append(backcast)
            residual = residual - backcast
        
        # Combine forecasts
        trend_forecast = torch.stack(forecasts, dim=1).sum(dim=1)
        
        # Generate polynomial trends for interpretability
        batch_size = x.size(0)
        t = torch.linspace(0, 1, self.input_size, device=x.device)
        polynomial_basis = torch.stack([t**i for i in range(4)], dim=0)
        
        trend_components = torch.matmul(
            self.trend_coefficients.unsqueeze(0).expand(batch_size, -1, -1),
            polynomial_basis.unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        return {
            'forecast': trend_forecast,
            'backcast': torch.stack(backcasts, dim=1).sum(dim=1),
            'trend_components': trend_components,
            'residual': residual
        }

class SeasonalityStack(nn.Module):
    """
    Seasonality stack for identifying recurring patterns in task structures.
    """
    
    def __init__(self, input_size: int, num_blocks: int = 3, 
                 theta_size: int = 128, basis_size: int = None):
        super().__init__()
        
        # Set basis_size to match input_size for proper residual connections
        if basis_size is None:
            basis_size = input_size * 2  # Double for forecast/backcast split
            
        self.input_size = input_size
        self.basis_size = basis_size
        
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_size, theta_size, basis_size)
            for _ in range(num_blocks)
        ])
        
        # Fourier basis for seasonality
        self.num_harmonics = 8
        self.seasonality_coefficients = nn.Parameter(
            torch.randn(input_size, self.num_harmonics * 2)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input through seasonality stack.
        
        Returns:
            Dictionary containing seasonal forecasts and interpretable components
        """
        forecasts = []
        backcasts = []
        residual = x
        
        for block in self.blocks:
            forecast, backcast = block(residual)
            forecasts.append(forecast)
            backcasts.append(backcast)
            residual = residual - backcast
        
        # Combine forecasts
        seasonal_forecast = torch.stack(forecasts, dim=1).sum(dim=1)
        
        # Generate Fourier basis for interpretability
        batch_size = x.size(0)
        t = torch.linspace(0, 2 * np.pi, self.input_size, device=x.device)
        
        fourier_basis = []
        for i in range(1, self.num_harmonics + 1):
            fourier_basis.extend([torch.sin(i * t), torch.cos(i * t)])
        fourier_basis = torch.stack(fourier_basis, dim=0)
        
        seasonal_components = torch.matmul(
            self.seasonality_coefficients.unsqueeze(0).expand(batch_size, -1, -1),
            fourier_basis.unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        return {
            'forecast': seasonal_forecast,
            'backcast': torch.stack(backcasts, dim=1).sum(dim=1),
            'seasonal_components': seasonal_components,
            'residual': residual
        }

class GenericStack(nn.Module):
    """
    Generic stack for handling novel and irregular task patterns.
    """
    
    def __init__(self, input_size: int, num_blocks: int = 3, 
                 theta_size: int = 128, basis_size: int = None):
        super().__init__()
        
        # Set basis_size to match input_size for proper residual connections
        if basis_size is None:
            basis_size = input_size * 2  # Double for forecast/backcast split
            
        self.input_size = input_size
        self.basis_size = basis_size
        
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_size, theta_size, basis_size)
            for _ in range(num_blocks)
        ])
        
        # Adaptive basis functions
        self.adaptive_basis = nn.Parameter(torch.randn(input_size, input_size))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input through generic stack.
        
        Returns:
            Dictionary containing generic forecasts and adaptive components
        """
        forecasts = []
        backcasts = []
        residual = x
        
        for block in self.blocks:
            forecast, backcast = block(residual)
            forecasts.append(forecast)
            backcasts.append(backcast)
            residual = residual - backcast
        
        # Combine forecasts
        generic_forecast = torch.stack(forecasts, dim=1).sum(dim=1)
        
        # Apply adaptive basis transformation
        batch_size = x.size(0)
        adaptive_components = torch.matmul(
            generic_forecast.unsqueeze(1),
            self.adaptive_basis.unsqueeze(0).expand(batch_size, -1, -1)
        ).squeeze(1)
        
        return {
            'forecast': generic_forecast,
            'backcast': torch.stack(backcasts, dim=1).sum(dim=1),
            'adaptive_components': adaptive_components,
            'residual': residual
        }

class InterpretabilityModule(nn.Module):
    """
    Module for providing interpretable explanations of task decomposition decisions.
    """
    
    def __init__(self, feature_size: int, num_strategies: int = 4, 
                 num_task_types: int = 8):
        super().__init__()
        
        self.feature_size = feature_size
        self.num_strategies = num_strategies
        self.num_task_types = num_task_types
        
        # Attention mechanism for feature importance
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=feature_size, num_heads=8, batch_first=True
        )
        
        # Strategy recommendation head
        self.strategy_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_strategies),
            nn.Softmax(dim=-1)
        )
        
        # Task type classification head
        self.task_type_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_task_types),
            nn.Softmax(dim=-1)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate interpretability outputs.
        
        Args:
            features: Input features tensor
            
        Returns:
            Dictionary containing interpretability metrics
        """
        # Self-attention for feature importance
        attended_features, attention_weights = self.feature_attention(
            features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Generate interpretability outputs
        strategy_probs = self.strategy_head(attended_features)
        task_type_probs = self.task_type_head(attended_features)
        confidence = self.confidence_head(attended_features)
        
        return {
            'strategy_probabilities': strategy_probs,
            'task_type_probabilities': task_type_probs,
            'confidence_score': confidence,
            'feature_attention': attention_weights.squeeze(1),
            'feature_importance': torch.mean(attention_weights, dim=1).squeeze(1)
        }

class NBEATSTaskDecomposer(nn.Module):
    """
    Complete N-BEATS Task Decomposer with interpretable task breakdown capabilities.
    """
    
    def __init__(self, input_size: int = 64, max_subtasks: int = 16, 
                 hidden_size: int = 256, num_strategies: int = 4, 
                 num_task_types: int = 8):
        super().__init__()
        
        self.input_size = input_size
        self.max_subtasks = max_subtasks
        self.hidden_size = hidden_size
        
        # Input encoding (removed batch norm for small datasets)
        self.input_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),  # Use LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # N-BEATS stacks
        self.trend_stack = TrendStack(hidden_size)
        self.seasonality_stack = SeasonalityStack(hidden_size)
        self.generic_stack = GenericStack(hidden_size)
        
        # Interpretability module
        self.interpretability = InterpretabilityModule(
            hidden_size, num_strategies, num_task_types
        )
        
        # Task decomposition heads
        self.complexity_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, max_subtasks),
            nn.Sigmoid()
        )
        
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, max_subtasks),
            nn.ReLU()
        )
        
        self.dependency_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, max_subtasks * max_subtasks),
            nn.Sigmoid()
        )
        
        self.subtask_count_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Quality estimation heads
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # decomposition_accuracy, time_accuracy, dependency_accuracy
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete N-BEATS Task Decomposer.
        
        Args:
            x: Input features tensor of shape (batch_size, input_size)
            
        Returns:
            Dictionary containing all model outputs
        """
        # Encode input features
        encoded = self.input_encoder(x)
        
        # Process through N-BEATS stacks
        trend_output = self.trend_stack(encoded)
        seasonal_output = self.seasonality_stack(trend_output['residual'])
        generic_output = self.generic_stack(seasonal_output['residual'])
        
        # Combine stack outputs
        combined_features = (
            trend_output['forecast'] + 
            seasonal_output['forecast'] + 
            generic_output['forecast']
        )
        
        # Generate interpretability outputs
        interpretability_outputs = self.interpretability(combined_features)
        
        # Generate task decomposition outputs
        batch_size = x.size(0)
        complexity_scores = self.complexity_head(combined_features)
        duration_estimates = self.duration_head(combined_features)
        dependency_matrix = self.dependency_head(combined_features).view(
            batch_size, self.max_subtasks, self.max_subtasks
        )
        subtask_count = self.subtask_count_head(combined_features)
        quality_estimates = self.quality_head(combined_features)
        
        return {
            # Core outputs
            'complexity_scores': complexity_scores,
            'duration_estimates': duration_estimates,
            'dependency_matrix': dependency_matrix,
            'subtask_count': subtask_count,
            'quality_estimates': quality_estimates,
            
            # N-BEATS components (for interpretability)
            'trend_components': trend_output['trend_components'],
            'seasonal_components': seasonal_output['seasonal_components'],
            'adaptive_components': generic_output['adaptive_components'],
            'trend_forecast': trend_output['forecast'],
            'seasonal_forecast': seasonal_output['forecast'],
            'generic_forecast': generic_output['forecast'],
            
            # Interpretability outputs
            'strategy_probabilities': interpretability_outputs['strategy_probabilities'],
            'task_type_probabilities': interpretability_outputs['task_type_probabilities'],
            'confidence_score': interpretability_outputs['confidence_score'],
            'feature_attention': interpretability_outputs['feature_attention'],
            'feature_importance': interpretability_outputs['feature_importance'],
            
            # Combined features for downstream analysis
            'combined_features': combined_features,
            'encoded_features': encoded
        }
    
    def get_interpretable_explanation(self, outputs: Dict[str, torch.Tensor],
                                    feature_names: Optional[List[str]] = None) -> Dict:
        """
        Generate human-readable interpretable explanation of the decomposition.
        
        Args:
            outputs: Model outputs dictionary
            feature_names: Optional list of feature names for explanation
            
        Returns:
            Dictionary containing interpretable explanations
        """
        strategy_names = ['waterfall', 'agile', 'feature_driven', 'component_based']
        task_type_names = ['web_development', 'api_development', 'machine_learning', 
                          'data_processing', 'testing', 'enterprise_application',
                          'saas_platform', 'mobile_development']
        
        # Get the first sample from batch for explanation
        strategy_probs = outputs['strategy_probabilities'][0].cpu().numpy()
        task_type_probs = outputs['task_type_probabilities'][0].cpu().numpy()
        confidence = outputs['confidence_score'][0].cpu().item()
        feature_importance = outputs['feature_importance'][0].cpu().numpy()
        
        # Recommended strategy
        recommended_strategy_idx = np.argmax(strategy_probs)
        recommended_strategy = strategy_names[recommended_strategy_idx]
        strategy_confidence = strategy_probs[recommended_strategy_idx]
        
        # Detected task type
        detected_task_type_idx = np.argmax(task_type_probs)
        detected_task_type = task_type_names[detected_task_type_idx]
        task_type_confidence = task_type_probs[detected_task_type_idx]
        
        # Top contributing features
        if feature_names:
            top_features_idx = np.argsort(feature_importance)[-5:][::-1]
            top_features = [(feature_names[i], feature_importance[i]) for i in top_features_idx]
        else:
            top_features_idx = np.argsort(feature_importance)[-5:][::-1]
            top_features = [(f"Feature_{i}", feature_importance[i]) for i in top_features_idx]
        
        explanation = {
            'recommended_strategy': {
                'name': recommended_strategy,
                'confidence': float(strategy_confidence),
                'explanation': f"Based on the task characteristics, {recommended_strategy} decomposition strategy is recommended with {strategy_confidence:.1%} confidence."
            },
            'detected_task_type': {
                'name': detected_task_type,
                'confidence': float(task_type_confidence),
                'explanation': f"The task appears to be {detected_task_type.replace('_', ' ')} with {task_type_confidence:.1%} confidence."
            },
            'overall_confidence': {
                'score': float(confidence),
                'explanation': f"The model is {confidence:.1%} confident in this decomposition."
            },
            'key_factors': {
                'features': top_features,
                'explanation': "The most important factors influencing this decomposition are: " + 
                              ", ".join([f"{name} ({importance:.3f})" for name, importance in top_features[:3]])
            },
            'decomposition_insights': {
                'trend_influence': float(torch.mean(outputs['trend_forecast'][0]).cpu().item()),
                'seasonal_influence': float(torch.mean(outputs['seasonal_forecast'][0]).cpu().item()),
                'generic_influence': float(torch.mean(outputs['generic_forecast'][0]).cpu().item()),
                'explanation': "The decomposition is influenced by trend patterns (long-term complexity), seasonal patterns (recurring structures), and novel/irregular patterns."
            }
        }
        
        return explanation
    
    def load_config(self, config_path: str) -> Dict:
        """Load model configuration from TOML file."""
        try:
            import toml
            with open(config_path, 'r') as f:
                config = toml.load(f)
            return config
        except ImportError:
            logger.warning("TOML library not available. Using default configuration.")
            return {}
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def save_model(self, path: str, metadata: Optional[Dict] = None):
        """Save the complete model with metadata."""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'max_subtasks': self.max_subtasks,
                'hidden_size': self.hidden_size
            },
            'metadata': metadata or {}
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """Load a saved model."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        model = cls(
            input_size=config['input_size'],
            max_subtasks=config['max_subtasks'],
            hidden_size=config['hidden_size']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from {path}")
        return model, checkpoint.get('metadata', {})