import faiss
import pickle
import numpy as np


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
