from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def generate_embeddings_list(
            self,
            texts: list):
        embeddings = self.model.encode(texts)
        return embeddings

    def generate_embeddings_string(
            self,
            texts: str):
        embeddings = self.model.encode(texts)
        return embeddings
