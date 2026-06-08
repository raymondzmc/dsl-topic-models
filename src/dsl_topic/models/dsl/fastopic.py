"""DSL FASTopic: FASTopic variant that uses LLM hidden states as document
embeddings and a sparse top-k LLM distribution as the reconstruction target
in the Dual Semantic-relation Reconstruction (DSR) loss.

The optimal-transport components (DT-ETP, TW-ETP) remain the same.
Doc embeddings are swapped from SentenceTransformer to LLM hidden states.
The reconstruction target is built by zeroing out non-top-k logits and then
applying softmax with temperature over the full vocabulary, producing a proper
distribution that is naturally sparse."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dsl_topic.models.baselines.fastopic._fastopic import fastopic
from dsl_topic.models.dsl._objective import topk_target, extract_topics, theta_inference


class DSLFASTopicModel(fastopic):
    """fastopic subclass with sparse top-k LLM targets for the DSR loss."""

    def __init__(self, num_topics, theta_temp=1.0, DT_alpha=3.0, TW_alpha=2.0,
                 temperature=3.0, topk=20):
        super().__init__(num_topics, theta_temp, DT_alpha, TW_alpha)
        self.temperature = temperature
        self.topk = topk

    def get_theta(self, doc_embeddings, train_doc_embeddings):
        """Override parent's get_theta which uses exp(-dist) and underflows
        with high-norm LLM embeddings. Use the Sinkhorn transport plan directly,
        consistent with how theta is computed during training."""
        with torch.no_grad():
            _, transp = self.DT_ETP(doc_embeddings, self.topic_embeddings)
            theta = transp * transp.shape[0]
            theta = theta / theta.sum(1, keepdim=True)
            return theta

    def forward(self, teacher_logits, doc_embeddings):
        loss_DT, transp_DT = self.DT_ETP(doc_embeddings, self.topic_embeddings)
        loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)

        loss_ETP = loss_DT + loss_TW

        theta = transp_DT * transp_DT.shape[0]
        beta = transp_TW * transp_TW.shape[0]

        recon = torch.matmul(theta, beta)

        # Zero out non-top-k logits, then softmax with temperature over full vocab
        masked_logits = topk_target(teacher_logits, k=self.topk, mask_mode="neg_inf")
        sparse_target = F.softmax(masked_logits / self.temperature, dim=-1)

        loss_DSR = -(sparse_target * (recon + self.epsilon).log()).sum(axis=1).mean()

        loss = loss_DSR + loss_ETP

        return {'loss': loss, 'loss_DSR': loss_DSR, 'loss_ETP': loss_ETP}


class DSLFASTopic:
    """Trainer for DSLFASTopicModel using CTMDataset (LLM embeddings + logits).

    Uses full-batch training like the original FASTopic (no encoder/decoder,
    only OT on embedding matrices)."""

    def __init__(self, vocab_size, embedding_size, num_topics=25,
                 epochs=200, batch_size=None, lr=0.002,
                 DT_alpha=3.0, TW_alpha=2.0, theta_temp=1.0,
                 temperature=3.0, topk=20, top_words=15, vocab=None,
                 **kwargs):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_topics = num_topics
        self.epochs = epochs
        self.top_words = top_words
        self.vocab = vocab
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = DSLFASTopicModel(
            num_topics=num_topics,
            theta_temp=theta_temp,
            DT_alpha=DT_alpha,
            TW_alpha=TW_alpha,
            temperature=temperature,
            topk=topk,
        )
        self.model.init(vocab_size, embedding_size)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_doc_embeddings = None

    def fit(self, ctm_dataset):
        n_total = len(ctm_dataset)

        # Load all data onto GPU as single batch (same as original FASTopic)
        all_emb, all_logits = [], []
        loader = DataLoader(ctm_dataset, batch_size=256, shuffle=False)
        for batch in loader:
            all_emb.append(batch['x_embeddings'])
            all_logits.append(batch['y'])
        train_emb = torch.cat(all_emb, dim=0).to(self.device)
        train_logits = torch.cat(all_logits, dim=0).to(self.device)
        self.train_doc_embeddings = train_emb

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            rst = self.model(train_logits, train_emb)
            loss = rst['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0 or epoch == 1:
                print(f'DSLFASTopic Epoch: {epoch:03d} loss: {loss.item() / n_total:.3f}')

    def get_info(self, idx2token=None):
        self.model.eval()
        with torch.no_grad():
            beta = self.model.get_beta().detach().cpu().numpy()
        return {
            'topic-word-matrix': beta,
            'topics': extract_topics(beta, self.top_words, idx2token=idx2token, vocab=self.vocab),
        }

    def get_theta(self, ctm_dataset):
        return theta_inference(
            self.model, ctm_dataset, device=self.device, batch_size=256,
            theta_fn=lambda x: self.model.get_theta(x, self.train_doc_embeddings),
        )
