import os
import textract
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from db.faiss_handler import FaissHandler
from embedding_model.embeddings_model import EmbeddingModel

class PdfReader:

    def __init__(self):
        self.text = ''


    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in the text using GPT2 tokenizer."""
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokens = tokenizer(text)['input_ids']
        return len(tokens)



    def read_pdf(self, pdf_path: str, output_txt_path: str):
        """Reads a PDF, extracts text, splits into chunks, and generates embeddings."""
        # Extract text from the PDF
        print("Extracting text from PDF...")
        doc = textract.process(pdf_path)

        # Save the extracted text to a file
        print("Saving extracted text...")
        with open(output_txt_path, "w", encoding="utf-8") as outfile:
            outfile.write(doc.decode("utf-8"))

        # Read the text from the saved file
        with open(output_txt_path, "r", encoding="utf-8") as infile:
            text = infile.read()

        # Initialize the text splitter
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=24,
            length_function=len  # Use character length for splitting
        )
        chunks = text_splitter.create_documents([text])

        print(f"Number of chunks: {len(chunks)}")

        # Prepare text chunks for embedding
        chunk_texts = [chunk.page_content for chunk in chunks]

        # Generate embeddings in batches
        print("Generating embeddings for chunks...")
        embeddings = EmbeddingModel.generate_embeddings_list(chunk_texts)

        print(f"Generated {len(embeddings)} embeddings.")
        return embeddings, chunks


# Example usage
if __name__ == "__main__":
    pdf_reader = PdfReader()
    pdf_path = "../asset/the-lord-of-the-rings-the-fellowship-of-the-ring-2001.pdf"
    output_txt_path = "../asset/the-lord-of-the-rings-the-fellowship.txt"

    # Read the PDF, split into chunks, and generate embeddings
    embeddings, chunks = pdf_reader.read_pdf(pdf_path, output_txt_path)

    FaissHandler.save_to_vector_db(
        embeddings,
        chunks,
        'db/faiss_index.faiss',
        'db/metadata.pkl'
    )

    #value  = FaissHandler.get_first_value('db/faiss_index.faiss','db/metadata.pkl')
    # Print details of the first chunk
    #print(f"First value: {value}")
    #print(f"First embedding: {embeddings[0]}")
