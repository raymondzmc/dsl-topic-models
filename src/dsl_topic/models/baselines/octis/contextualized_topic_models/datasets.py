import torch
from torch.utils.data import Dataset
import scipy.sparse


class CTMDataset(Dataset):

    """Class to load BOW dataset."""

    def __init__(self, x_bow, x_embeddings, idx2token, y=None):
        """
        Args
            X : array-like, shape=(n_samples, n_features)
                Document word matrix.
        """
        if x_bow is not None and x_bow.shape[0] != len(x_embeddings):
            raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
                            "You might want to check if the BoW preparation method has removed some documents. ")
        if x_bow is None and y is None:
            raise Exception("x_bow and y cannot be None at the same time")

        self.x_bow = x_bow
        self.x_embeddings = x_embeddings
        self.idx2token = idx2token
        self.y = y

    def __len__(self):
        """Return length of dataset."""
        return self.x_embeddings.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        x_embeddings = torch.FloatTensor(self.x_embeddings[i])
        example = {'x_embeddings': x_embeddings}
        if self.x_bow is not None:
            if type(self.x_bow[i]) == scipy.sparse.csr.csr_matrix:
                x_bow = torch.FloatTensor(self.x_bow[i].todense())
            else:
                x_bow = torch.FloatTensor(self.x_bow[i])
            example['x_bow'] = x_bow

        if self.y is not None:
            y = torch.FloatTensor(self.y[i])
            example['y'] = y

        return example
