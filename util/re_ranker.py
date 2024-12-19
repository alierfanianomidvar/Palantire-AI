from transformers import pipeline, AutoModel, AutoTokenizer
import numpy as np
import torch

class ReRanker(object):

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the LLMModel using Hugging Face's Transformers library.

        Args:
            model_name (str): The Hugging Face model to use (default is a lightweight SentenceTransformer model).
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts):
        """
        Generate embeddings for a list of texts.

        Args:
            texts (list of str): List of input texts to embed.

        Returns:
            np.array: Array of embeddings for the input texts.
        """
        # Ensure texts is a list of strings
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input texts must be a string or a list of strings.")

        # Tokenize and process
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Mean pooling
        embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def batch_evaluate(self, user_input, candidates):
        """
        Compute relevance scores for user input against candidate texts.

        Args:
            user_input (str): The user's query/input.
            candidates (list of dict): List of candidate texts with "chunk" key.

        Returns:
            list of float: Relevance scores for each candidate.
        """
        if not isinstance(user_input, str):
            raise ValueError("user_input must be a string.")
        if not isinstance(candidates, list) or not all(
                "chunk" in c and isinstance(c["chunk"], str) for c in candidates):
            raise ValueError("candidates must be a list of dictionaries with a 'chunk' key containing strings.")

        # Generate embeddings
        query_embedding = self.encode(user_input)[0]
        candidate_embeddings = self.encode([c["chunk"] for c in candidates])
        
        # Compute cosine similarities
        scores = np.dot(candidate_embeddings, query_embedding) / (
                np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Attach scores to the chunks
        results = [
            {"chunk": candidates[i]["chunk"], "cosine_similarity": score}
            for i, score in enumerate(scores)
        ]

        return results

