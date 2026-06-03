"""Generative ETM: ETM variant that uses LLM hidden states as encoder input
and KL divergence against LLM next-token logits as the reconstruction target."""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class DSLETMModel(nn.Module):
    """ETM architecture with LLM embedding input and teacher-distillation loss."""

    def __init__(self, num_topics, vocab_size, embedding_size, t_hidden_size,
                 rho_size, theta_act='softplus',
                 embeddings=None, train_embeddings=True, enc_drop=0.5,
                 temperature=3.0, loss_weight=1e3, sparsity_ratio=1.0,
                 loss_type='KL'):
        super().__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.sparsity_ratio = sparsity_ratio
        self.loss_type = loss_type
        self.enc_drop = enc_drop

        self.t_drop = nn.Dropout(enc_drop)
        self.theta_act = self._get_activation(theta_act)

        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            self.rho = embeddings.clone().float()

        self.alphas = nn.Linear(rho_size, num_topics, bias=False)

        # Encoder takes LLM hidden states (embedding_size) instead of BoW (vocab_size)
        self.q_theta = nn.Sequential(
            nn.Linear(embedding_size, t_hidden_size), self.theta_act,
            nn.Linear(t_hidden_size, t_hidden_size), self.theta_act,
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    @staticmethod
    def _get_activation(act):
        activations = {
            'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'softplus': nn.Softplus(),
            'sigmoid': nn.Sigmoid(), 'leakyrelu': nn.LeakyReLU(),
            'elu': nn.ELU(), 'selu': nn.SELU(),
        }
        return activations.get(act, nn.Tanh())

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        return mu

    def encode(self, x):
        q_theta = self.q_theta(x)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu = self.mu_q_theta(q_theta)
        logsigma = self.logsigma_q_theta(q_theta)
        kl = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(), dim=-1).mean()
        return mu, logsigma, kl

    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight)
        except AttributeError:
            logit = self.alphas(self.rho)
        return F.softmax(logit, dim=0).transpose(1, 0)

    def get_theta(self, x):
        mu, logsigma, kl = self.encode(x)
        z = self.reparameterize(mu, logsigma)
        theta = F.softmax(z, dim=-1)
        return theta, kl

    def forward(self, x_embeddings, teacher_logits):
        theta, kl = self.get_theta(x_embeddings)
        beta = self.get_beta()
        student_probs = F.softmax(torch.mm(theta, beta), dim=-1)

        k = math.ceil(self.sparsity_ratio * teacher_logits.size(1))
        topk_indices = torch.topk(teacher_logits, k=k, dim=1)[1]
        mask = torch.zeros_like(teacher_logits)
        mask.scatter_(1, topk_indices, 1.0)
        masked_logits = teacher_logits * mask

        if self.loss_type == 'CE':
            rl = -torch.sum(masked_logits * torch.log(student_probs + 1e-10), dim=1)
        elif self.loss_type == 'KL':
            teacher_probs = torch.softmax(masked_logits / self.temperature, dim=-1).clamp_min(1e-9)
            student_probs = student_probs.clamp_min(1e-9)
            rl = torch.sum(teacher_probs * torch.log(teacher_probs / student_probs), dim=1)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        loss = kl + self.loss_weight * rl.mean()
        return loss, kl, rl.mean()


class DSLETM:
    """Trainer for DSLETMModel using CTMDataset (LLM embeddings + logits)."""

    def __init__(self, vocab_size, embedding_size, num_topics=25,
                 t_hidden_size=800, rho_size=300,
                 activation='softplus', dropout=0.5, lr=0.002,
                 batch_size=64, num_epochs=100, clip=0.0,
                 temperature=3.0, loss_weight=1e3, sparsity_ratio=1.0,
                 loss_type='KL', top_words=15,
                 embeddings=None, train_embeddings=True):
        self.num_topics = num_topics
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.clip = clip
        self.top_words = top_words
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = DSLETMModel(
            num_topics=num_topics,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            t_hidden_size=t_hidden_size,
            rho_size=rho_size,
            theta_act=activation,
            embeddings=embeddings,
            train_embeddings=train_embeddings,
            enc_drop=dropout,
            temperature=temperature,
            loss_weight=loss_weight,
            sparsity_ratio=sparsity_ratio,
            loss_type=loss_type,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, ctm_dataset):
        loader = DataLoader(ctm_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss, n = 0, 0
            for batch in loader:
                x = batch['x_embeddings'].to(self.device)
                y = batch['y'].to(self.device)
                self.optimizer.zero_grad()
                loss, kl, rl = self.model(x, y)
                loss.backward()
                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'DSLETM Epoch [{epoch+1}/{self.num_epochs}] Loss: {total_loss/n:.4f}')

    def get_info(self, idx2token=None):
        self.model.eval()
        info = {}
        with torch.no_grad():
            beta = self.model.get_beta().cpu().numpy()

        topics = []
        for k in range(self.num_topics):
            if np.isnan(beta[k]).any():
                topics = None
                break
            top_indices = beta[k].argsort()[-self.top_words:][::-1]
            if idx2token is not None:
                topics.append([idx2token[i] for i in top_indices])
            else:
                topics.append(list(top_indices))

        info['topic-word-matrix'] = beta
        info['topics'] = topics
        return info

    def get_theta(self, ctm_dataset):
        loader = DataLoader(ctm_dataset, batch_size=self.batch_size, shuffle=False)
        all_theta = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch['x_embeddings'].to(self.device)
                theta, _ = self.model.get_theta(x)
                all_theta.append(theta.cpu().numpy())
        return np.concatenate(all_theta, axis=0)
