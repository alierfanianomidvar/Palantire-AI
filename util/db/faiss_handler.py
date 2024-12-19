import faiss
import pickle
import numpy as np

from ..embedding_model.embeddings_model import EmbeddingModel
from ..re_ranker import ReRanker

class FaissHandler:

    @staticmethod
    def save_to_vector_db(
            embeddings,
            chunks,
            faiss_index_path,
            metadata_path):

        if len(embeddings) == 0:
            raise ValueError("Embeddings list is empty.")
        if len(chunks) == 0:
            raise ValueError("Chunks list is empty.")
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length.")

        # Save embeddings to FAISS
        dimension = len(embeddings[0])  # Length of each vector
        index = faiss.IndexFlatL2(dimension)  # Create FAISS index for L2 distance
        index.add(np.array(embeddings).astype(np.float32))  # Add embeddings
        faiss.write_index(index, faiss_index_path)  # Save FAISS index

        # Save metadata (chunks) to a separate file
        with open(metadata_path, "wb") as metadata_file:
            pickle.dump(chunks, metadata_file)

        print(f"FAISS index saved to {faiss_index_path}.")
        print(f"Metadata saved to {metadata_path}.")

    @staticmethod
    def query_vector_db(
            query_embedding,
            faiss_index_path,
            metadata_path,
            top_k=5):
        if not isinstance(query_embedding, (list, np.ndarray)):
            raise ValueError("Query embedding must be a list or numpy array.")

        # Load FAISS index
        index = faiss.read_index(faiss_index_path)

        # Load metadata (chunks)
        with open(metadata_path, "rb") as metadata_file:
            chunks = pickle.load(metadata_file)

        # Ensure query embedding has the correct format
        query_embedding = np.array(query_embedding).astype(np.float32)

        # Perform search
        distances, indices = index.search(np.array([query_embedding]), top_k)

        # Retrieve results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):  # Ensure index is valid
                results.append({
                    "rank": i + 1,
                    "chunk": chunks[idx].page_content,  # Adjust based on your chunk data structure
                    "distance": distances[0][i]
                })
            else:
                print(f"Warning: Index {idx} is out of range for chunks.")

        return results

    @staticmethod
    def get_first_value(faiss_index_path, metadata_path):
        """
        Extract the first embedding and its associated metadata from the FAISS database.

        Args:
            faiss_index_path (str): Path to the FAISS index file.
            metadata_path (str): Path to the metadata (chunks) file.

        Returns:
            dict: A dictionary containing the first embedding and its associated chunk.
        """
        # Load FAISS index
        index = faiss.read_index(faiss_index_path)

        # Load metadata (chunks)
        with open(metadata_path, "rb") as metadata_file:
            chunks = pickle.load(metadata_file)

        # Check if the index and metadata are not empty
        if index.ntotal == 0:
            raise ValueError("FAISS index is empty.")
        if len(chunks) == 0:
            raise ValueError("Metadata is empty.")

        # Extract the first embedding and associated metadata
        first_embedding = index.reconstruct(0)  # Retrieve the first vector
        first_chunk = chunks[0]  # Retrieve the first chunk

        print("we are here")
        print(first_chunk)
        print(first_embedding)

        return first_chunk

    @staticmethod
    def get_closest_result(
            user_input,
            faiss_index_path,
            metadata_path):

        reRanker = ReRanker()
        """
        Find the closest result to the user input using cosine similarity and Euclidean distance.

        Args:
            user_input (str): The user's query/input.
            embedding_model: Pre-trained embedding model (e.g., SentenceTransformer).
            faiss_index_path (str): Path to the FAISS index file.
            metadata_path (str): Path to the metadata (chunks) file.

        Returns:
            dict: Closest result with cosine similarity and Euclidean distance scores.
        """
        embedding_model = EmbeddingModel()

        # Load FAISS index
        index = faiss.read_index(faiss_index_path)

        # Load metadata (chunks)
        with open(metadata_path, "rb") as metadata_file:
            chunks = pickle.load(metadata_file)

        # Check if the index and metadata are not empty
        if index.ntotal == 0:
            raise ValueError("FAISS index is empty.")
        if len(chunks) == 0:
            raise ValueError("Metadata is empty.")

        # Convert the user input into a vector
        user_vector = embedding_model.generate_embeddings_list([user_input])[0]
        user_vector = user_vector / np.linalg.norm(user_vector)

        distances, indices = index.search(np.expand_dims(user_vector.astype('float32'), axis=0), 5)


        # Retrieve metadata for the closest results
        closest_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                chunk_content = chunks[idx].page_content if hasattr(chunks[idx], 'page_content') else str(chunks[idx])

                closest_results.append({
                    "rank": i + 1,
                    "chunk": chunk_content,
                    "cosine_similarity": float(distances[0][i])  # Distances are cosine similarities
                })

        closest_results_re_ranked = reRanker.batch_evaluate(user_input, closest_results)

        return closest_results_re_ranked

