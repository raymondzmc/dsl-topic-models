import os
import math
import datetime
import torch
import numpy as np
from collections import defaultdict
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from dsl_topic.models._vendored.octis.contextualized_topic_models.inference_network import ContextualInferenceNetwork
from dsl_topic.models._vendored.octis.early_stopping.pytorchtools import EarlyStopping


class Autoencoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, n_components=10, hidden_sizes=(100,100),
                 activation='softplus', dropout=0.2, learn_priors=True,
                 topic_prior_mean=0.0, topic_prior_variance=None, temperature=1.0):
        """
        Args
            vocab_size : int, vocabulary size (output dimension for decoder)
            embedding_size : int, dimension of input embeddings
            n_components : int, number of topic components, (default 10)
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            dropout : float, dropout for theta (default 0.2)
            learn_priors : bool, make priors learnable parameter
            topic_prior_mean: double, mean parameter of the prior
            topic_prior_variance: double, variance parameter of the prior
            temperature: double, temperature parameter of the softmax
        """
        super(Autoencoder, self).__init__()
        assert isinstance(vocab_size, int), "vocab_size must by type int."
        assert isinstance(embedding_size, int), "embedding_size must by type int."
        assert (isinstance(n_components, int) or isinstance(n_components, np.int64)) and n_components > 0, \
            "n_components must be type int > 0."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'tanh', 'leakyrelu',
                              'rrelu', 'elu', 'selu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'leakyrelu'," \
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(topic_prior_mean, float), \
            "topic_prior_mean must be type float"
        assert isinstance(temperature, float), \
            "temperature must be type float"

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.encoder = ContextualInferenceNetwork(
            embedding_size, n_components, hidden_sizes, activation)

        self.prior_mean = torch.tensor([topic_prior_mean] * n_components)
        if topic_prior_variance is None:
            topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor([topic_prior_variance] * n_components)
        self.beta = torch.Tensor(n_components, vocab_size)

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.prior_mean = self.prior_mean.cuda()
            self.prior_variance = self.prior_variance.cuda()
            self.beta = self.beta.cuda()

        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)

        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta_batchnorm = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop_theta = nn.Dropout(p=self.dropout)   # dropout on theta
        self.temperature = temperature

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """Forward pass."""
        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.encoder(None, x)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = torch.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        topic_doc = theta
        theta = self.drop_theta(theta)
        word_dist = torch.softmax(
            self.beta_batchnorm(torch.matmul(theta, self.beta)) / self.temperature, dim=1)
        topic_word = self.beta
        return self.prior_mean, self.prior_variance, \
            posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, topic_word, topic_doc

    def get_theta(self, x):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.encoder(None, x)

            # generate samples from theta
            theta = torch.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta


class DSLProdLDA(object):
    def __init__(
        self, vocab_size, embedding_size, num_topics=10, hidden_sizes=(100, 100),
        activation='softplus', dropout=0.2, learn_priors=True, batch_size=64,
        lr=2e-3, momentum=0.99, solver='adam', num_epochs=100, num_samples=10,
        reduce_on_plateau=False, topic_prior_mean=0.0, top_words=10,
        topic_prior_variance=None, num_data_loader_workers=0, loss_weight=1.0,
        sparsity_ratio=1.0, topk=None, temperature=1.0, loss_type='KL'):

        """
        :param vocab_size: int, vocabulary size (target dimension for reconstruction)
        :param embedding_size: int, dimension of input embeddings from LLM
        :param num_topics: int, number of topic components, (default 10)
        :param model_type: string, 'prodLDA' or 'LDA' (default 'prodLDA')
        :param hidden_sizes: tuple, length = n_layers, (default (100, 100))
        :param activation: string, 'softplus', 'relu', 'sigmoid', 'swish',
            'tanh', 'leakyrelu', 'rrelu', 'elu', 'selu' (default 'softplus')
        :param dropout: float, dropout to use (default 0.2)
        :param learn_priors: bool, make priors a learnable parameter (default
            True)
        :param batch_size: int, size of batch to use for training (default 64)
        :param lr: float, learning rate to use for training (default 2e-3)
        :param momentum: float, momentum to use for training (default 0.99)
        :param solver: string, optimizer 'adam' or 'sgd' (default 'adam')
        :param num_samples: int, number of times theta needs to be sampled
        :param num_epochs: int, number of epochs to train for, (default 100)
        :param reduce_on_plateau: bool, reduce learning rate by 10x on plateau
            of 10 epochs (default False)
        :param num_data_loader_workers: int, number of data loader workers
            (default cpu_count). set it to 0 if you are using Windows
        """

        assert isinstance(vocab_size, int) and vocab_size > 0, \
            "vocab_size must by type int > 0."
        assert isinstance(embedding_size, int) and embedding_size > 0, \
            "embedding_size must by type int > 0."
        assert (isinstance(num_topics, int) or isinstance(
            num_topics, np.int64)) and num_topics > 0, \
            "num_topics must by type int > 0."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in [
            'softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu',
            'rrelu', 'elu', 'selu'], \
            ("activation must be 'softplus', 'relu', 'sigmoid', 'swish', "
             "'leakyrelu', 'rrelu', 'elu', 'selu' or 'tanh'.")
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size must be int > 0."
        assert lr > 0, "lr must be > 0."
        assert isinstance(
            momentum, float) and momentum > 0 and momentum <= 1, \
            "momentum must be 0 < float <= 1."
        assert solver in ['adagrad', 'adam', 'sgd', 'adadelta', 'rmsprop'], \
            "solver must be 'adam', 'adadelta', 'sgd', 'rmsprop' or 'adagrad'"
        assert isinstance(reduce_on_plateau, bool), \
            "reduce_on_plateau must be type bool."
        assert isinstance(topic_prior_mean, float), \
            "topic_prior_mean must be type float"
        assert isinstance(loss_weight, float), \
            "loss_weight must be type float"
        assert isinstance(sparsity_ratio, float), \
            "sparsity_ratio must be type float"
        assert isinstance(temperature, float), \
            "temperature must be type float"
        assert loss_type in ['CE', 'KL'], \
            "loss_type must be 'CE' or 'KL'"

        # and topic_prior_variance >= 0, \
        # assert isinstance(topic_prior_variance, float), \
        #    "topic prior_variance must be type float"

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_topics = num_topics
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.num_samples = num_samples
        self.top_words = top_words
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers
        self.topic_prior_mean = topic_prior_mean
        self.topic_prior_variance = topic_prior_variance
        self.loss_weight = loss_weight
        self.sparsity_ratio = sparsity_ratio
        self.topk = topk
        self.temperature = temperature
        self.loss_type = loss_type

        # init encoder-decoder network
        self.model = Autoencoder(
            vocab_size, embedding_size, num_topics, hidden_sizes, activation,
            dropout, self.learn_priors, self.topic_prior_mean,
            self.topic_prior_variance, self.temperature)
        self.early_stopping = EarlyStopping(patience=5, verbose=False)
        # init optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(
                self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum)
        elif self.solver == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        elif self.solver == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        elif self.solver == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=lr, momentum=self.momentum)
        # init lr scheduler
        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        # performance attributes
        self.best_loss_train = float('inf')

        # training attributes
        self.model_dir = None
        self.train_dataset = None
        self.nn_epoch = None

        # learned topics
        self.best_components = None

        # Use cuda if available
        self.USE_CUDA = torch.cuda.is_available()
        if self.USE_CUDA:
            self.model = self.model.cuda()

    def _loss(self, teacher_logits, student_probs, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):
        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = 0.5 * (
            var_division + diff_term - self.num_topics + logvar_det_division)

        # Reconstruction term: sparse top-k target with -inf masking
        if self.topk is not None:
            k = self.topk
        else:
            k = math.ceil(self.sparsity_ratio * teacher_logits.size(1))
        topk_vals, topk_idx = torch.topk(teacher_logits, k=k, dim=1)
        masked_logits = torch.full_like(teacher_logits, float('-inf'))
        masked_logits.scatter_(1, topk_idx, topk_vals)

        if self.loss_type == 'CE':
            teacher_probs = torch.softmax(masked_logits / self.temperature, dim=-1)
            RL = -torch.sum(teacher_probs * torch.log(student_probs + 1e-10), dim=1)
        elif self.loss_type == 'KL':
            teacher_probs = torch.softmax(masked_logits / self.temperature, dim=-1)
            teacher_probs = teacher_probs.clamp_min(1e-9)
            student_probs = student_probs.clamp_min(1e-9)
            RL = torch.sum(teacher_probs * torch.log(teacher_probs / student_probs), dim=1)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        loss = KL + self.loss_weight * RL
        return loss.sum()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0
        topic_doc_list = []
        for batch_samples in loader:
            x = batch_samples['x_embeddings']
            teacher_logits = batch_samples['y']
            if self.USE_CUDA:
                x = x.cuda()
                teacher_logits = teacher_logits.cuda()

            # forward pass
            self.model.zero_grad()
            (prior_mean, prior_variance,
             posterior_mean, posterior_variance, posterior_log_variance,
             word_dists, topic_word, topic_document) = self.model(x)
            topic_doc_list.extend(topic_document)

            # backward pass
            loss = self._loss(
                teacher_logits, word_dists, prior_mean, prior_variance,
                posterior_mean, posterior_variance, posterior_log_variance)
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += x.size(0)
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss, topic_word, topic_doc_list

    def _validation(self, loader):
        """Train epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            x_embeddings = batch_samples['x_embeddings']
            teacher_logits = batch_samples['y']

            if self.USE_CUDA:
                x_embeddings = x_embeddings.cuda()
                teacher_logits = teacher_logits.cuda()

            # forward pass
            self.model.zero_grad()
            (prior_mean, prior_variance,
             posterior_mean, posterior_variance, posterior_log_variance,
             word_dists, topic_word, topic_document) = self.model(x_embeddings)

            loss = self._loss(
                teacher_logits, word_dists, prior_mean, prior_variance,
                posterior_mean, posterior_variance, posterior_log_variance)

            # compute train loss
            samples_processed += x_embeddings.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss

    def fit(self, train_dataset, validation_dataset=None,
            save_dir=None, verbose=True):
        """
        Train the CTM model.

        :param train_dataset: PyTorch Dataset class for training data.
        :param validation_dataset: PyTorch Dataset class for validation data
        :param save_dir: directory to save checkpoint models to.
        :param verbose: verbose
        """
        # Print settings to output file
        if verbose:
            print("Settings: \n\
                   N Components: {}\n\
                   Topic Prior Mean: {}\n\
                   Topic Prior Variance: {}\n\
                   Hidden Sizes: {}\n\
                   Activation: {}\n\
                   Dropout: {}\n\
                   Learn Priors: {}\n\
                   Learning Rate: {}\n\
                   Momentum: {}\n\
                   Reduce On Plateau: {}\n\
                   Save Dir: {}".format(
                self.num_topics, self.topic_prior_mean,
                self.topic_prior_variance,
                self.hidden_sizes, self.activation, self.dropout,
                self.learn_priors, self.lr, self.momentum,
                self.reduce_on_plateau, save_dir))

        self.model_dir = save_dir
        self.train_dataset = train_dataset
        self.validation_data = validation_dataset
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_data_loader_workers)

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss, topic_word, topic_document = self._train_epoch(
                train_loader)
            samples_processed += sp
            e = datetime.datetime.now()

            if verbose:
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                    epoch + 1, self.num_epochs, samples_processed,
                    len(self.train_dataset) * self.num_epochs, train_loss, e - s))

            self.best_components = self.model.beta
            self.final_topic_word = topic_word
            self.final_topic_document = topic_document
            self.best_loss_train = train_loss
            if self.validation_data is not None:
                validation_loader = DataLoader(
                    self.validation_data, batch_size=self.batch_size,
                    shuffle=True, num_workers=self.num_data_loader_workers)
                # train epoch
                s = datetime.datetime.now()
                val_samples_processed, val_loss = self._validation(
                    validation_loader)
                e = datetime.datetime.now()

                if verbose:
                    print(
                        "Epoch: [{}/{}]\tSamples: [{}/{}]"
                        "\tValidation Loss: {}\tTime: {}".format(
                            epoch + 1, self.num_epochs, val_samples_processed,
                            len(self.validation_data) * self.num_epochs,
                            val_loss, e - s))

                if np.isnan(val_loss) or np.isnan(train_loss):
                    break
                else:
                    self.early_stopping(val_loss, self.model)
                    if self.early_stopping.early_stop:
                        if verbose:
                            print("Early stopping")
                        if save_dir is not None:
                            self.save(save_dir)
                        break

    def predict(self, dataset):
        """Predict input."""
        self.model.eval()

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_data_loader_workers)

        topic_document_mat = []
        with torch.no_grad():
            for batch_samples in loader:
                x_embeddings = batch_samples['x_embeddings']

                if self.USE_CUDA:
                    x_embeddings = x_embeddings.cuda()
                # forward pass
                self.model.zero_grad()
                _, _, _, _, _, _, _, topic_document = self.model(x_embeddings)
                topic_document_mat.append(topic_document)

        results = self.get_info()
        results['test-topic-document-matrix'] = np.asarray(
            self.get_thetas(dataset)).T

        return results

    def get_topic_word_mat(self):
        top_wor = self.final_topic_word.cpu().detach().numpy()
        return top_wor

    def get_topic_document_mat(self):
        top_doc = self.final_topic_document
        top_doc_arr = np.array([i.cpu().detach().numpy() for i in top_doc])
        return top_doc_arr

    def get_topics(self):
        """
        Retrieve topic words.

        """
        assert self.top_words <= self.vocab_size, "top_words must be <= vocab size."  # noqa
        component_dists = self.best_components
        topics = defaultdict(list)
        topics_list = []
        if self.num_topics is not None:
            for i in range(self.num_topics):
                _, idxs = torch.topk(component_dists[i], self.top_words)
                component_words = [self.train_dataset.idx2token[idx]
                                   for idx in idxs.cpu().numpy()]
                topics[i] = component_words
                topics_list.append(component_words)

        return topics_list

    def get_info(self):
        info = {}
        topic_word = self.get_topics()
        topic_word_dist = self.get_topic_word_mat()
        topic_document_dist = self.get_topic_document_mat()
        info['topics'] = topic_word

        info['topic-document-matrix'] = np.asarray(
            self.get_thetas(self.train_dataset)).T

        info['topic-word-matrix'] = topic_word_dist
        return info

    def _format_file(self):
        model_dir = (
            "DSLProdLDA_nc_{}_tpm_{}_tpv_{}_hs_{}_ac_{}_do_{}_"
            "lr_{}_mo_{}_rp_{}".format(
                self.num_topics, 0.0, 1 - (1. / self.num_topics),
                self.hidden_sizes, self.activation,
                self.dropout, self.lr, self.momentum,
                self.reduce_on_plateau))
        return model_dir

    def save(self, models_dir=None):
        """
        Save model.

        :param models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = self._format_file()
            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)

    def load(self, model_dir, epoch):
        """
        Load a previously trained model.

        :param model_dir: directory where models are saved.
        :param epoch: epoch of model to load.
        """
        epoch_file = "epoch_" + str(epoch) + ".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict)

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self.model.load_state_dict(checkpoint['state_dict'])

    def get_thetas(self, dataset):
        """
        Get the document-topic distribution for a dataset of topics. 
        Includes multiple sampling to reduce variation via
        the parameter num_samples.
        :param dataset: a PyTorch Dataset containing the documents
        """
        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_data_loader_workers)
        final_thetas = []
        for sample_index in range(self.num_samples):
            with torch.no_grad():
                collect_theta = []
                for batch_samples in loader:
                    x_embeddings = batch_samples['x_embeddings']
                    if self.USE_CUDA:
                        x_embeddings = x_embeddings.cuda()
                    self.model.zero_grad()
                    collect_theta.extend(
                        self.model.get_theta(x_embeddings).cpu().numpy().tolist())
                final_thetas.append(np.array(collect_theta))
        return np.sum(final_thetas, axis=0) / self.num_samples
