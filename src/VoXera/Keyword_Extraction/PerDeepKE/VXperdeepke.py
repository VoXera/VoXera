from parsivar import Normalizer, Tokenizer
from embedding import Embedding
from text_segmentation import segmentor
import numpy as np
class KeywordExtraction():
    def __init__(self):
        pass
    
    def load_model(self):
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer()
        self.embeder = Embedding()
    
    def preprocess(self, text):
        normal_text = self.normalizer.normalize(text)
        return normal_text

    def train(self):
        pass
    
    def infer(self, text, segment_num= 2, top_n= 5):
        tokens = self.tokenizer.tokenize_words(text)
        words_type = list(set(tokens))

        text_embedding = self.embeder.sentence_embedding([text])
        segments_embedding = self.embeder.sentence_embedding(segmentor(tokens, segment_num)[0])
        words_type_embedding = self.embeder.sentence_embedding(words_type)

        words_text_sim = np.matmul(text_embedding, words_type_embedding.T)
        words_segments_sim = np.matmul(segments_embedding, words_type_embedding.T)

        similar_matrix = np.concatenate((words_text_sim, words_segments_sim), axis = 0)

        weights = np.ones(similar_matrix.shape[0])
        weights[0] *= 3 

        word_score = zip(words_type, np.apply_along_axis(self._semantic_score_up, 0, similar_matrix))

        sorted_word_score = sorted(dict(word_score).items(), key=lambda y: y[1], reverse=True)

        return sorted_word_score[:top_n+1]


    def postprocess():
        pass