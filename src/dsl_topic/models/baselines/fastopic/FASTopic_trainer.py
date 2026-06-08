from dsl_topic.models.baselines.topmost.utils.logger import Logger
from dsl_topic.models.baselines.topmost.preprocess import Preprocess
from dsl_topic.models.baselines.fastopic.FASTopic import FASTopic

logger = Logger("WARNING")


class FASTopicTrainer:
    def __init__(self,
                 dataset,
                 num_topics=50,
                 num_top_words=15,
                 preprocess=None,
                 doc_embed_model="all-MiniLM-L6-v2",
                 epochs=200,
                 DT_alpha=3.0,
                 TW_alpha=2.0,
                 theta_temp=1.0,
                 verbose=False,
                 low_memory=False,
                 low_memory_batch_size=None,
                ):
        self.dataset = dataset
        self.num_top_words = num_top_words

        preprocess = Preprocess(stopwords=[]) if preprocess is None else preprocess
        self.model = FASTopic(num_topics=num_topics,
                              preprocess=preprocess,
                              num_top_words=num_top_words,
                              doc_embed_model=doc_embed_model,
                              normalize_embeddings=True,  # Required for custom embedding models like gte-large-en-v1.5
                              DT_alpha=DT_alpha,
                              TW_alpha=TW_alpha,
                              theta_temp=theta_temp,
                              verbose=verbose,
                              low_memory=low_memory,
                              low_memory_batch_size=low_memory_batch_size,
                            )

        self.epochs = epochs

        if verbose:
            logger.set_level("DEBUG")
        else:
            logger.set_level("WARNING")

    def train(self):
        return self.model.fit_transform(self.dataset.train_texts, epochs=self.epochs)

    def test(self, texts):
        theta = self.model.transform(texts)
        return theta

    def get_beta(self):
        beta = self.model.get_beta()
        return beta

    def get_top_words(self, num_top_words=None):
        if num_top_words is None:
            num_top_words = self.num_top_words
        return self.model.get_top_words(num_top_words)

    def export_theta(self):
        train_theta = self.test(self.dataset.train_texts)
        test_theta = self.test(self.dataset.test_texts)
        return train_theta, test_theta
