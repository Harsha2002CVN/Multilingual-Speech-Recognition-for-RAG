from transformers import RagRetriever

# Load pre-trained RAG retriever
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")

def initialize_dummy_document():
    dummy_document = """
    RAG (Retrieval-Augmented Generation) is a framework that combines retrieval-based and generation-based methods.
    It retrieves relevant documents from a knowledge base and generates responses based on the retrieved documents.
    """

    # Add the dummy document to the retriever
    retriever.indexer.add_texts([dummy_document])
