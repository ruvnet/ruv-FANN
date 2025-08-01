#!/usr/bin/env python3
"""
RUV-Swarm Ensemble Coordinator Training Script - Improved Version

This enhanced version includes:
- Advanced data augmentation techniques
- Curriculum learning for better convergence
- Improved loss functions and regularization
- Better architecture designs
- Sophisticated training strategies

Target: 95%+ coordination accuracy with cognitive diversity optimization
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import SAGEConv, global_mean_pool, GATConv
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import pickle
import warnings
import random
from collections import deque
import math
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class AdvancedCoordinatorDataset(Dataset):
    """Enhanced dataset with data augmentation and better preprocessing"""
    
    def __init__(self, data_path: str, augment=True, augmentation_factor=3):
        self.data = self._load_data(data_path)
        self.augment = augment
        self.augmentation_factor = augmentation_factor
        self.scaler = RobustScaler()  # More robust to outliers
        self._preprocess_data()
        if augment:
            self._augment_data()
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load training data from JSON file"""
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def _preprocess_data(self):
        """Enhanced preprocessing with better normalization"""
        features = np.array([item['features'] for item in self.data])
        labels = np.array([item['labels'] for item in self.data])
        
        # Normalize features using RobustScaler
        self.features = self.scaler.fit_transform(features)
        
        # Normalize labels to [0, 1] range for better training stability
        self.label_min = labels.min(axis=0)
        self.label_max = labels.max(axis=0)
        self.labels = (labels - self.label_min) / (self.label_max - self.label_min + 1e-8)
        
        # Extract metadata for cognitive diversity modeling
        self.metadata = [item.get('metadata', {}) for item in self.data]
        
    def _augment_data(self):
        """Data augmentation to increase training diversity"""
        original_features = self.features.copy()
        original_labels = self.labels.copy()
        original_metadata = self.metadata.copy()
        
        augmented_features = []
        augmented_labels = []
        augmented_metadata = []
        
        for _ in range(self.augmentation_factor):
            # Add noise augmentation
            noise_std = 0.05
            noise = np.random.normal(0, noise_std, original_features.shape)
            aug_features = original_features + noise
            
            # Add label noise (smaller scale)
            label_noise = np.random.normal(0, 0.01, original_labels.shape)
            aug_labels = np.clip(original_labels + label_noise, 0, 1)
            
            augmented_features.append(aug_features)
            augmented_labels.append(aug_labels)
            augmented_metadata.extend(original_metadata)
        
        # Combine original and augmented data
        self.features = np.vstack([original_features] + augmented_features)
        self.labels = np.vstack([original_labels] + augmented_labels)
        self.metadata = original_metadata + augmented_metadata
        
    def denormalize_labels(self, normalized_labels):
        """Convert labels back to original scale"""
        return normalized_labels * (self.label_max - self.label_min) + self.label_min
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'labels': torch.FloatTensor(self.labels[idx]),
            'metadata': self.metadata[idx]
        }

class ImprovedGraphSAGE(nn.Module):
    """Enhanced GraphSAGE with attention and residual connections"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        
        # Use GAT for better attention mechanism
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
        
        self.dropout = nn.Dropout(0.3)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers-1)])
        self.residual_connections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers-1)
        ])
        
    def forward(self, x, edge_index, batch=None):
        identity = x
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.layer_norms[i](x)
            
            # Residual connection
            if i == 0 and identity.size(-1) != x.size(-1):
                identity = self.residual_connections[i](identity)
            elif i > 0:
                identity = self.residual_connections[i](identity)
            
            x = x + identity
            x = F.gelu(x)  # GELU activation for better gradients
            x = self.dropout(x)
            identity = x
        
        x = self.convs[-1](x, edge_index)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        return x

class AdvancedTransformerSelector(nn.Module):
    """Enhanced Transformer with better architecture and regularization"""
    
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._create_positional_encoding(d_model, 1000)
        
        # Enhanced transformer with layer normalization improvements
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Improved cognitive profiling heads with attention
        self.cognitive_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.problem_solving_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        self.info_processing_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        self.decision_making_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        self.communication_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        self.learning_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        
        self.diversity_predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.selection_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def _create_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        seq_len = x.size(1)
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        
        # Transformer encoding
        encoded = self.transformer(x)
        
        # Self-attention for cognitive profiling
        attended, _ = self.cognitive_attention(encoded, encoded, encoded)
        pooled = attended.mean(dim=1)  # Global average pooling
        
        # Cognitive profile predictions with improved architecture
        cognitive_profiles = {
            'problem_solving': F.softmax(self.problem_solving_head(pooled), dim=-1),
            'info_processing': F.softmax(self.info_processing_head(pooled), dim=-1),
            'decision_making': F.softmax(self.decision_making_head(pooled), dim=-1),
            'communication': F.softmax(self.communication_head(pooled), dim=-1),
            'learning': F.softmax(self.learning_head(pooled), dim=-1)
        }
        
        diversity_score = self.diversity_predictor(pooled)
        selection_prob = self.selection_head(pooled)
        
        return {
            'cognitive_profiles': cognitive_profiles,
            'diversity_score': diversity_score,
            'selection_prob': selection_prob,
            'encoded_features': pooled
        }

class ImprovedDQN(nn.Module):
    """Enhanced DQN with better architecture and training stability"""
    
    def __init__(self, state_dim: int, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        
        # Improved network architecture with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        self.residual_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Noisy networks for better exploration
        self.noisy_value = NoisyLinear(hidden_dim, 1)
        self.noisy_advantage = NoisyLinear(hidden_dim, action_dim)
        
    def forward(self, state):
        features = self.feature_extractor(state)
        
        # Residual connection
        residual = self.residual_block(features)
        features = features + residual
        
        # Noisy dueling architecture
        value = self.noisy_value(features)
        advantage = self.noisy_advantage(features)
        
        # Dueling DQN formula
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return {
            'q_values': q_values,
            'features': features,
            'value': value,
            'advantage': advantage
        }

class NoisyLinear(nn.Module):
    """Noisy linear layer for better exploration in DQN"""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
        
    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                          self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class EnhancedVAE(nn.Module):
    """Improved VAE with better architecture and regularization"""
    
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Enhanced encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Enhanced decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Improved diversity predictor
        self.diversity_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        diversity_score = self.diversity_predictor(z)
        
        return {
            'recon_x': recon_x,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'diversity_score': diversity_score
        }

class ImprovedEnsembleCoordinator(nn.Module):
    """Enhanced multi-model ensemble coordinator"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        
        # Initialize improved component models
        self.task_distributor = ImprovedGraphSAGE(input_dim)
        self.agent_selector = AdvancedTransformerSelector(input_dim)
        self.load_balancer = ImprovedDQN(input_dim)
        self.diversity_optimizer = EnhancedVAE(input_dim)
        
        # Simplified but effective meta-learner
        self.meta_learner = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        
        # Enhanced fusion layer with attention
        self.fusion_layer = None  # Will be initialized dynamically
        
    def forward(self, x, edge_index=None, batch=None):
        # Get predictions from all models
        
        # Task distribution (with synthetic graph if needed)
        if edge_index is not None:
            task_features = self.task_distributor(x, edge_index, batch)
        else:
            # Create more meaningful synthetic graph
            batch_size = x.size(0)
            num_nodes = min(batch_size * 3, 50)  # Reasonable graph size
            edge_index = self._create_synthetic_graph(num_nodes, x.device)
            node_features = x.repeat(num_nodes // batch_size + 1, 1)[:num_nodes]
            task_features = self.task_distributor(node_features, edge_index)
            task_features = task_features[:batch_size]  # Take only needed features
        
        # Agent selection
        agent_output = self.agent_selector(x)
        agent_features = agent_output['encoded_features']
        
        # Load balancing
        load_output = self.load_balancer(x)
        load_features = load_output['features']
        
        # Cognitive diversity
        diversity_output = self.diversity_optimizer(x)
        diversity_features = diversity_output['z']
        
        # Meta-learning
        meta_features = self.meta_learner(x)
        
        # Concatenate all features
        ensemble_features = torch.cat([
            task_features,
            agent_features, 
            load_features,
            diversity_features,
            meta_features
        ], dim=-1)
        
        # Initialize fusion layer with attention mechanism
        if self.fusion_layer is None:
            fusion_input_dim = ensemble_features.size(-1)
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 2)  # Final coordination output
            ).to(x.device)
        
        # Final fusion
        coordination_output = self.fusion_layer(ensemble_features)
        
        return {
            'coordination_output': coordination_output,
            'task_features': task_features,
            'agent_output': agent_output,
            'load_output': load_output,
            'diversity_output': diversity_output,
            'meta_features': meta_features
        }
    
    def _create_synthetic_graph(self, num_nodes, device):
        """Create a more meaningful synthetic graph structure"""
        # Create a small-world network structure
        edges = []
        
        # Ring topology
        for i in range(num_nodes):
            edges.append([i, (i + 1) % num_nodes])
        
        # Add random long-range connections
        num_random_edges = min(num_nodes // 2, 10)
        for _ in range(num_random_edges):
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src != dst:
                edges.append([src, dst])
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        return edge_index

class CurriculumLearningScheduler:
    """Curriculum learning to progressively increase task difficulty"""
    
    def __init__(self, total_epochs=100):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def get_difficulty_weight(self):
        """Get current difficulty weight (0.0 to 1.0)"""
        return min(1.0, self.current_epoch / (self.total_epochs * 0.7))
    
    def get_loss_weights(self):
        """Get adaptive loss weights based on curriculum"""
        difficulty = self.get_difficulty_weight()
        return {
            'coordination': 1.0,
            'diversity': 0.1 + 0.4 * difficulty,  # Gradually increase diversity importance
            'reconstruction': 0.1 + 0.3 * difficulty,  # Gradually increase VAE importance
            'regularization': 0.05 * difficulty  # Add regularization later in training
        }
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch

class AdvancedEnsembleTrainer:
    """Enhanced trainer with better training strategies"""
    
    def __init__(self, model: ImprovedEnsembleCoordinator, train_loader: DataLoader, val_loader: DataLoader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Use different learning rates for different components
        self.optimizers = {
            'task': optim.AdamW(model.task_distributor.parameters(), lr=2e-4, weight_decay=1e-5),
            'agent': optim.AdamW(model.agent_selector.parameters(), lr=1e-4, weight_decay=1e-5),
            'load': optim.AdamW(model.load_balancer.parameters(), lr=5e-4, weight_decay=1e-5),
            'diversity': optim.AdamW(model.diversity_optimizer.parameters(), lr=3e-4, weight_decay=1e-5),
            'meta': optim.AdamW(model.meta_learner.parameters(), lr=2e-4, weight_decay=1e-5),
            'fusion': None  # Will be initialized later
        }
        
        # Learning rate schedulers
        self.schedulers = {
            'task': optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizers['task'], T_0=10),
            'agent': optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizers['agent'], T_0=10),
            'load': optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizers['load'], T_0=10),
            'diversity': optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizers['diversity'], T_0=10),
            'meta': optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizers['meta'], T_0=10),
        }
        
        # Curriculum learning
        self.curriculum = CurriculumLearningScheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'coordination_accuracy': [],
            'diversity_score': [],
            'learning_rates': []
        }
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Enhanced training epoch with curriculum learning"""
        self.model.train()
        self.curriculum.update_epoch(epoch)
        
        epoch_losses = {
            'total': 0.0,
            'coordination': 0.0,
            'diversity': 0.0,
            'reconstruction': 0.0,
            'regularization': 0.0
        }
        
        loss_weights = self.curriculum.get_loss_weights()
        
        for batch_idx, batch in enumerate(self.train_loader):
            features = batch['features']
            labels = batch['labels']
            
            # Forward pass
            output = self.model(features)
            
            # Initialize fusion optimizer after first forward pass
            if self.optimizers['fusion'] is None and self.model.fusion_layer is not None:
                self.optimizers['fusion'] = optim.AdamW(
                    self.model.fusion_layer.parameters(), lr=1e-4, weight_decay=1e-5
                )
                self.schedulers['fusion'] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizers['fusion'], T_0=10
                )
            
            # Coordination loss with label smoothing
            coord_loss = self._smooth_l1_loss(output['coordination_output'], labels)
            
            # Enhanced diversity loss
            agent_profiles = output['agent_output']['cognitive_profiles']
            diversity_score = self._calculate_enhanced_diversity_score(agent_profiles)
            diversity_target = 0.85 + 0.1 * torch.sin(torch.tensor(epoch * 0.1))  # Dynamic target
            diversity_loss = F.mse_loss(diversity_score, torch.ones_like(diversity_score) * diversity_target)
            
            # Enhanced VAE loss with KL annealing
            vae_output = output['diversity_output']
            recon_loss = F.mse_loss(vae_output['recon_x'], features)
            kl_weight = min(1.0, epoch / 20.0)  # KL annealing
            kl_loss = -0.5 * torch.sum(1 + vae_output['logvar'] - vae_output['mu'].pow(2) - vae_output['logvar'].exp())
            vae_loss = recon_loss + kl_weight * 0.1 * kl_loss
            
            # Regularization losses
            reg_loss = self._compute_regularization_loss()
            
            # Weighted total loss
            total_loss = (
                loss_weights['coordination'] * coord_loss +
                loss_weights['diversity'] * diversity_loss +
                loss_weights['reconstruction'] * vae_loss +
                loss_weights['regularization'] * reg_loss
            )
            
            # Backward pass with gradient clipping
            for optimizer in self.optimizers.values():
                if optimizer is not None:
                    optimizer.zero_grad()
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer steps
            for optimizer in self.optimizers.values():
                if optimizer is not None:
                    optimizer.step()
            
            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['coordination'] += coord_loss.item()
            epoch_losses['diversity'] += diversity_loss.item()
            epoch_losses['reconstruction'] += vae_loss.item()
            epoch_losses['regularization'] += reg_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: Loss = {total_loss.item():.4f}, '
                      f'Coord = {coord_loss.item():.4f}, Diversity = {diversity_score.item():.4f}')
        
        # Update learning rate schedulers
        for scheduler in self.schedulers.values():
            if scheduler is not None:
                scheduler.step()
        
        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def _smooth_l1_loss(self, predictions, targets):
        """Smooth L1 loss for better training stability"""
        return F.smooth_l1_loss(predictions, targets)
    
    def _calculate_enhanced_diversity_score(self, profiles: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enhanced diversity calculation with multiple metrics"""
        diversity_scores = []
        
        for dimension, profile in profiles.items():
            # Shannon entropy
            entropy = -torch.sum(profile * torch.log(profile + 1e-8), dim=-1)
            
            # Gini coefficient for inequality measure
            gini = self._gini_coefficient(profile)
            
            # Combine entropy and gini
            combined_diversity = 0.7 * entropy + 0.3 * gini
            diversity_scores.append(combined_diversity)
        
        # Weighted average based on importance
        weights = torch.tensor([0.25, 0.2, 0.2, 0.15, 0.2], device=list(profiles.values())[0].device)
        total_diversity = torch.stack(diversity_scores).T @ weights
        
        return total_diversity.mean()
    
    def _gini_coefficient(self, x):
        """Calculate Gini coefficient for diversity measurement"""
        sorted_x, _ = torch.sort(x, dim=-1)
        n = x.size(-1)
        index = torch.arange(1, n + 1, device=x.device).float()
        gini = 2 * torch.sum(index * sorted_x, dim=-1) / (n * torch.sum(sorted_x, dim=-1)) - (n + 1) / n
        return gini
    
    def _compute_regularization_loss(self):
        """Compute various regularization losses"""
        reg_loss = 0.0
        
        # L2 regularization for fusion layer
        if self.model.fusion_layer is not None:
            for param in self.model.fusion_layer.parameters():
                reg_loss += torch.norm(param, p=2)
        
        return reg_loss * 1e-6  # Small weight
    
    def validate(self) -> Dict[str, float]:
        """Enhanced validation with better metrics"""
        self.model.eval()
        val_losses = {'total': 0.0, 'coordination': 0.0, 'diversity': 0.0}
        diversity_scores = []
        coordination_accuracies = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features']
                labels = batch['labels']
                
                output = self.model(features)
                
                # Coordination loss and accuracy
                coord_loss = F.mse_loss(output['coordination_output'], labels)
                coord_accuracy = self._calculate_enhanced_coordination_accuracy(
                    output['coordination_output'], labels
                )
                
                # Diversity metrics
                agent_profiles = output['agent_output']['cognitive_profiles']
                diversity_score = self._calculate_enhanced_diversity_score(agent_profiles)
                
                val_losses['coordination'] += coord_loss.item()
                val_losses['diversity'] += diversity_score.item()
                diversity_scores.append(diversity_score.item())
                coordination_accuracies.append(coord_accuracy)
        
        # Average metrics
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        val_losses['coordination_accuracy'] = np.mean(coordination_accuracies)
        val_losses['avg_diversity'] = np.mean(diversity_scores)
        
        return val_losses
    
    def _calculate_enhanced_coordination_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Enhanced coordination accuracy calculation"""
        # Multiple tolerance levels for more nuanced accuracy
        tolerances = [0.05, 0.1, 0.2]  # 5%, 10%, 20%
        weights = [0.6, 0.3, 0.1]  # Higher weight for stricter tolerance
        
        total_accuracy = 0.0
        for tolerance, weight in zip(tolerances, weights):
            accurate_predictions = torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8) <= tolerance
            accuracy = accurate_predictions.float().mean().item()
            total_accuracy += weight * accuracy
        
        return total_accuracy
    
    def train(self, num_epochs: int = 100) -> Dict[str, List[float]]:
        """Enhanced training loop with early stopping and model checkpointing"""
        best_val_accuracy = 0.0
        patience = 15
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_losses = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_metrics['coordination'])
            self.history['coordination_accuracy'].append(val_metrics['coordination_accuracy'])
            self.history['diversity_score'].append(val_metrics['avg_diversity'])
            
            # Learning rates
            current_lrs = {name: scheduler.get_last_lr()[0] if scheduler else 0.0 
                          for name, scheduler in self.schedulers.items()}
            self.history['learning_rates'].append(current_lrs)
            
            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_losses["total"]:.4f}')
            print(f'  Val Coordination Accuracy: {val_metrics["coordination_accuracy"]:.4f}')
            print(f'  Diversity Score: {val_metrics["avg_diversity"]:.4f}')
            print(f'  Learning Rates: {current_lrs}')
            
            # Save best model and early stopping
            if val_metrics['coordination_accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['coordination_accuracy']
                patience_counter = 0
                self.save_model('/workspaces/ruv-FANN/ruv-swarm/models/swarm-coordinator/best_improved_ensemble.bin')
                print(f'  🎯 New best model saved! Accuracy: {best_val_accuracy:.4f}')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            # Target reached
            if best_val_accuracy >= 0.95:
                print(f'🎉 TARGET ACHIEVED: 95%+ coordination accuracy!')
                break
        
        return self.history
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'curriculum_epoch': self.curriculum.current_epoch
        }
        
        for name, optimizer in self.optimizers.items():
            if optimizer is not None:
                save_dict[f'{name}_optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, filepath)
        print(f'Model saved to {filepath}')

def main_improved():
    """Main training function with improved architecture"""
    print("🚀 Starting RUV-Swarm Enhanced Ensemble Coordinator Training...")
    
    # Load data with augmentation
    print("📊 Loading and augmenting training data...")
    train_dataset = AdvancedCoordinatorDataset(
        '/workspaces/ruv-FANN/ruv-swarm/training-data/splits/coordinator/train.json',
        augment=True, augmentation_factor=5
    )
    val_dataset = AdvancedCoordinatorDataset(
        '/workspaces/ruv-FANN/ruv-swarm/training-data/splits/coordinator/validation.json',
        augment=False
    )
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create data loaders (adjust batch size for small validation set)
    train_batch_size = min(32, len(train_dataset))
    val_batch_size = min(16, len(val_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False)
    
    # Initialize improved model
    input_dim = len(train_dataset.features[0])
    print(f"🧠 Initializing Enhanced Ensemble Coordinator (input_dim={input_dim})...")
    model = ImprovedEnsembleCoordinator(input_dim)
    
    # Initialize trainer
    trainer = AdvancedEnsembleTrainer(model, train_loader, val_loader)
    
    # Training
    print("🎯 Starting enhanced ensemble training...")
    training_history = trainer.train(num_epochs=100)
    
    # Final evaluation
    print("📊 Final evaluation...")
    final_metrics = trainer.validate()
    
    # Save final model
    print("💾 Saving final enhanced model...")
    trainer.save_model('/workspaces/ruv-FANN/ruv-swarm/models/swarm-coordinator/enhanced_coordinator_weights.bin')
    
    # Generate report
    final_report = {
        'training_completed': True,
        'final_coordination_accuracy': final_metrics['coordination_accuracy'],
        'final_diversity_score': final_metrics['avg_diversity'],
        'training_history': training_history,
        'model_improvements': [
            'Advanced data augmentation (5x training data)',
            'Curriculum learning with adaptive loss weights',
            'Enhanced architectures with attention mechanisms',
            'Improved regularization and gradient clipping',
            'Noisy networks for better exploration',
            'Cosine annealing learning rate schedules',
            'Early stopping with patience',
            'Multiple accuracy tolerance levels'
        ]
    }
    
    with open('/workspaces/ruv-FANN/ruv-swarm/models/swarm-coordinator/enhanced_training_report.json', 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    # Print summary
    print("✅ Enhanced training completed!")
    print(f"Final Coordination Accuracy: {final_metrics['coordination_accuracy']:.3f}")
    print(f"Final Diversity Score: {final_metrics['avg_diversity']:.3f}")
    
    # Check if target achieved
    if final_metrics['coordination_accuracy'] >= 0.95:
        print("🎉 TARGET ACHIEVED: 95%+ coordination accuracy with cognitive diversity optimization!")
        return True
    else:
        print(f"⚠️  Target not fully reached. Best accuracy: {final_metrics['coordination_accuracy']:.3f}")
        print("📈 Significant improvements made with enhanced architecture and training strategies.")
        return False

if __name__ == "__main__":
    success = main_improved()