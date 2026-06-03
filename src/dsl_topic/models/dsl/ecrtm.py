"""Generative ECRTM: ECRTM variant that uses LLM hidden states as encoder input
and KL divergence against LLM next-token logits as the reconstruction target.

Keeps ECRTM's word/topic embeddings and ECR regularization intact, only
changes the encoder input dimension and reconstruction loss."""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from dsl_topic.models._vendored.topmost.ECRTM.ECR import ECR


class DSLECRTMModel(nn.Module):
    def __init__(self, vocab_size, num_topics=50, embedding_size=3072,
                 en_units=200, dropout=0., pretrained_WE=None, embed_size=200,
                 beta_temp=0.2, weight_loss_ECR=100.0,
                 sinkhorn_alpha=20.0, sinkhorn_max_iter=1000,
                 temperature=3.0, loss_weight=1e3, sparsity_ratio=1.0,
                 loss_type='KL'):
        super().__init__()

        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.sparsity_ratio = sparsity_ratio
        self.loss_type = loss_type

        # Dirichlet prior (same as ECRTM)
        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor(
            (np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor(
            (((1.0 / self.a) * (1 - (2.0 / num_topics))).T +
             (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))
        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        # Encoder: takes LLM hidden states (embedding_size) instead of BoW (vocab_size)
        self.fc11 = nn.Linear(embedding_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        # Word and topic embeddings (same as ECRTM)
        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.ECR = ECR(weight_loss_ECR, sinkhorn_alpha, sinkhorn_max_iter)

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        return mu

    def encode(self, x):
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        loss_KL = self.compute_loss_KL(mu, logvar)
        return theta, loss_KL

    def get_theta(self, x):
        theta, loss_KL = self.encode(x)
        if self.training:
            return theta, loss_KL
        return theta

    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        return KLD.mean()

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        return self.ECR(cost)

    def pairwise_euclidean_distance(self, x, y):
        return torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())

    def forward(self, x_embeddings, teacher_logits):
        theta, loss_KL = self.encode(x_embeddings)
        beta = self.get_beta()

        student_probs = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)

        # Teacher-distillation reconstruction loss (replaces BoW NLL)
        k = math.ceil(self.sparsity_ratio * teacher_logits.size(1))
        topk_indices = torch.topk(teacher_logits, k=k, dim=1)[1]
        mask = torch.zeros_like(teacher_logits)
        mask.scatter_(1, topk_indices, 1.0)
        masked_logits = teacher_logits * mask

        if self.loss_type == 'CE':
            recon_loss = -torch.sum(masked_logits * torch.log(student_probs + 1e-10), dim=1).mean()
        elif self.loss_type == 'KL':
            teacher_probs = torch.softmax(masked_logits / self.temperature, dim=-1).clamp_min(1e-9)
            student_probs = student_probs.clamp_min(1e-9)
            recon_loss = torch.sum(teacher_probs * torch.log(teacher_probs / student_probs), dim=1).mean()
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        loss_TM = self.loss_weight * recon_loss + loss_KL
        loss_ECR = self.get_loss_ECR()
        loss = loss_TM + loss_ECR

        return {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_ECR': loss_ECR,
            'recon_loss': recon_loss,
            'kl_loss': loss_KL,
        }


class DSLECRTM:
    """Trainer for DSLECRTMModel using CTMDataset (LLM embeddings + logits)."""

    def __init__(self, vocab_size, embedding_size, num_topics=25, vocab=None,
                 en_units=200, dropout=0., pretrained_WE=None, embed_size=200,
                 beta_temp=0.2, weight_loss_ECR=100.0,
                 sinkhorn_alpha=20.0, sinkhorn_max_iter=1000,
                 epochs=200, batch_size=64, lr=0.002,
                 temperature=3.0, loss_weight=1e3, sparsity_ratio=1.0,
                 loss_type='KL', top_words=15):
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.vocab = vocab
        self.epochs = epochs
        self.batch_size = batch_size
        self.top_words = top_words
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = DSLECRTMModel(
            vocab_size=vocab_size,
            num_topics=num_topics,
            embedding_size=embedding_size,
            en_units=en_units,
            dropout=dropout,
            pretrained_WE=pretrained_WE,
            embed_size=embed_size,
            beta_temp=beta_temp,
            weight_loss_ECR=weight_loss_ECR,
            sinkhorn_alpha=sinkhorn_alpha,
            sinkhorn_max_iter=sinkhorn_max_iter,
            temperature=temperature,
            loss_weight=loss_weight,
            sparsity_ratio=sparsity_ratio,
            loss_type=loss_type,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, ctm_dataset):
        loader = DataLoader(ctm_dataset, batch_size=self.batch_size, shuffle=True)
        n_total = len(ctm_dataset)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            loss_accum = 0.0

            for batch in loader:
                x = batch['x_embeddings'].to(self.device)
                y = batch['y'].to(self.device)
                rst = self.model(x, y)
                batch_loss = rst['loss']

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                loss_accum += batch_loss.item() * x.size(0)

            if epoch % 10 == 0 or epoch == 1:
                print(f'DSLECRTM Epoch: {epoch:03d} loss: {loss_accum / n_total:.3f}')

    def get_info(self, idx2token=None):
        self.model.eval()
        info = {}
        with torch.no_grad():
            beta = self.model.get_beta().detach().cpu().numpy()

        topics = []
        for k in range(self.num_topics):
            if np.isnan(beta[k]).any():
                topics = None
                break
            top_indices = beta[k].argsort()[-self.top_words:][::-1]
            if idx2token is not None:
                topics.append([idx2token[i] for i in top_indices])
            elif self.vocab is not None:
                topics.append([self.vocab[i] for i in top_indices])
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
                theta = self.model.get_theta(x)
                all_theta.append(theta.cpu().numpy())
        return np.concatenate(all_theta, axis=0)
