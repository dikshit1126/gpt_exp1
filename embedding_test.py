from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Two sentences
sentences = [
    "How do I track my order?",
    "Where is my parcel?"
]

# Convert sentences into embeddings
embeddings = model.encode(sentences)

print("Number of sentences:", len(embeddings))
print("Vector size:", len(embeddings[0]))

print("\nFirst sentence embedding (first 10 values):")
print(embeddings[0][:10])

print("\nSecond sentence embedding (first 10 values):")
print(embeddings[1][:10])
