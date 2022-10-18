from sentence_transformers import SentenceTransformer

class Embedding():
    def __init__(self):
        self.sentence_encode =  SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    def sentence_embedding(self, text):
        return self.sentence_encode.encode(text)



