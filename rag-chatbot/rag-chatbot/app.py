from sentence_transformers import SentenceTransformer
import numpy as np

# sample text
documents = [
    "Machine learning is a subset of AI.",
    "Python is used for data science.",
    "Vector databases store embeddings."
]

# load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# create embeddings
vectors = model.encode(documents)

print("Embeddings created successfully!")
