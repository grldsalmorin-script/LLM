# RAG Implementation Notes

Retrieval augmented generation has four core phases: load documents, split them into chunks, embed or vectorize the chunks, then retrieve the most relevant chunks for a question. The final answer should be grounded in retrieved context and include citations.

This local demo uses a deterministic bag-of-words vectorizer instead of external embeddings. In production, replace it with OpenAI embeddings, Vertex AI embeddings, or another embedding model, then store vectors in a database such as pgvector, Qdrant, Pinecone, Weaviate, or Elasticsearch.
