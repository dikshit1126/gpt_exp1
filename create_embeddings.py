import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os


# Load FAQ dataset
faqs = pd.read_csv("data/faqs.csv")

print("Loading FAQ dataset...")
print("Number of FAQs:", len(faqs))


# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# Generate embeddings for FAQ questions
print("\nGenerating FAQ embeddings...")
embeddings = model.encode(faqs["Question"].tolist())


# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)


# Save embeddings
np.save("models/faq_embeddings.npy", embeddings)


print("\nFAQ embeddings generated successfully!")
print("Embedding shape:", embeddings.shape)
print("Saved to: models/faq_embeddings.npy")
