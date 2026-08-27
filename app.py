import pandas as pd
import ollama

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. LOAD DATASETS
# ==========================================

products = pd.read_csv("data/products.csv")
faq = pd.read_csv("data/faq.csv")
reviews = pd.read_csv("data/reviews.csv")

print("Datasets loaded successfully!")


# ==========================================
# 2. CREATE TF-IDF VECTORS FOR FAQ
# ==========================================

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(faq["question"])


# ==========================================
# 3. GET USER QUESTION
# ==========================================

question = input("\nYou: ")


# ==========================================
# 4. CONVERT USER QUESTION INTO A VECTOR
# ==========================================

question_vector = vectorizer.transform([question])


# ==========================================
# 5. CALCULATE COSINE SIMILARITY
# ==========================================

similarities = cosine_similarity(question_vector, faq_vectors)


# ==========================================
# 6. FIND MOST SIMILAR FAQ
# ==========================================

best_index = similarities.argmax()

best_score = similarities[0][best_index]

best_question = faq.iloc[best_index]["question"]
best_answer = faq.iloc[best_index]["answer"]


# ==========================================
# 7. DISPLAY SEARCH RESULT
# ==========================================

print("\nMost relevant FAQ:", best_question)
print("Similarity score:", round(best_score, 2))
print("FAQ answer:", best_answer)


# ==========================================
# 8. SEND FAQ ANSWER TO QWEN
# ==========================================

prompt = f"""
You are an e-commerce customer support chatbot.

Answer the user's question using the information provided below.

User question:
{question}

Relevant FAQ question:
{best_question}

Relevant FAQ answer:
{best_answer}

Give a short and clear answer.
"""

response = ollama.chat(
    model="qwen:0.5b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ==========================================
# 9. DISPLAY FINAL ANSWER
# ==========================================

print("\nQwen:", response["message"]["content"])
