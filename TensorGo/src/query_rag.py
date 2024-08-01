from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Load pre-trained RAG model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

def query_rag(transcription):
    inputs = tokenizer(transcription, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
