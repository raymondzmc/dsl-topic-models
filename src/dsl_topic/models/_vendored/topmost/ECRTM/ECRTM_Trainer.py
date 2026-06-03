"""ECRTM (Effective Neural Topic Modeling with Embedding Clustering Regularization) Trainer.

Reference: Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023
Authors: Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from dsl_topic.models._vendored.topmost.ECRTM.ECRTM import ECRTM


class ECRTMTrainer:
    """Trainer wrapper for ECRTM model."""
    
    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        vocab: list[str],
        pretrained_WE: np.ndarray = None,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 0.002,
        en_units: int = 200,
        dropout: float = 0.0,
        embed_size: int = 200,
        beta_temp: float = 0.2,
        weight_loss_ECR: float = 100.0,
        sinkhorn_alpha: float = 20.0,
        sinkhorn_max_iter: int = 1000,
        device: str = None,
    ):
        """
        Initialize ECRTM trainer.
        
        Args:
            vocab_size: Size of vocabulary
            num_topics: Number of topics
            vocab: List of vocabulary words (for extracting top words)
            pretrained_WE: Pre-trained word embeddings (V x D), optional
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            en_units: Encoder hidden units
            dropout: Dropout rate
            embed_size: Embedding size (ignored if pretrained_WE provided)
            beta_temp: Temperature for beta (topic-word) matrix
            weight_loss_ECR: Weight for ECR loss
            sinkhorn_alpha: Sinkhorn algorithm alpha parameter
            sinkhorn_max_iter: Sinkhorn algorithm max iterations
            device: Device to use ('cuda' or 'cpu')
        """
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.vocab = vocab
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize ECRTM model
        self.model = ECRTM(
            vocab_size=vocab_size,
            num_topics=num_topics,
            en_units=en_units,
            dropout=dropout,
            pretrained_WE=pretrained_WE,
            embed_size=embed_size,
            beta_temp=beta_temp,
            weight_loss_ECR=weight_loss_ECR,
            sinkhorn_alpha=sinkhorn_alpha,
            sinkhorn_max_iter=sinkhorn_max_iter,
        )
        
        self.model = self.model.to(self.device)
    
    def train(self, train_data: np.ndarray, verbose: bool = True):
        """
        Train the ECRTM model.
        
        Args:
            train_data: Training data as BoW matrix (N x V)
            verbose: Whether to print training progress
            
        Returns:
            beta: Topic-word matrix (K x V)
        """
        # Convert to tensor and create dataloader
        train_tensor = torch.from_numpy(train_data).float().to(self.device)
        data_loader = DataLoader(train_tensor, batch_size=self.batch_size, shuffle=True)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        data_size = len(train_tensor)
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            loss_rst_dict = defaultdict(float)
            
            for batch_data in data_loader:
                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key].item() * len(batch_data)
            
            if verbose and (epoch % 10 == 0 or epoch == 1):
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    output_log += f' {key}: {loss_rst_dict[key] / data_size:.3f}'
                print(output_log)
        
        # Get beta matrix
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta
    
    def get_theta(self, input_data: np.ndarray):
        """
        Get document-topic distribution (theta).
        
        Args:
            input_data: Input BoW data (N x V)
            
        Returns:
            theta: Document-topic matrix (N x K)
        """
        data_size = input_data.shape[0]
        theta = np.zeros((data_size, self.num_topics))
        all_idx = torch.split(torch.arange(data_size), self.batch_size)
        
        input_tensor = torch.from_numpy(input_data).float().to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_tensor[idx]
                batch_theta = self.model.get_theta(batch_input)
                theta[idx.cpu().numpy()] = batch_theta.cpu().numpy()
        
        return theta
    
    def get_topics(self, beta: np.ndarray, top_words: int = 15):
        """
        Extract top words for each topic from beta matrix.
        
        Args:
            beta: Topic-word matrix (K x V)
            top_words: Number of top words per topic
            
        Returns:
            topics: List of lists of top words for each topic
        """
        topics = []
        for k in range(self.num_topics):
            top_indices = beta[k].argsort()[-top_words:][::-1]
            topic_words = [self.vocab[idx] for idx in top_indices]
            topics.append(topic_words)
        return topics

