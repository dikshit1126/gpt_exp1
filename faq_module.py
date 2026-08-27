import pandas as pd
import numpy as np
import requests

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen:0.5b"


# ============================================================
# LOAD FAQ DATASET
# ============================================================

print("Loading FAQ dataset...")

df = pd.read_csv("data/faqs.csv")

print("Number of FAQs:", len(df))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# LOAD FAQ EMBEDDINGS
# ============================================================

faq_embeddings = np.load(
    "models/faq_embeddings.npy"
)


# ============================================================
# OLLAMA FUNCTION
# ============================================================

def ask_ollama(prompt):

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=120
        )

        response.raise_for_status()

        return response.json()["response"].strip()

    except Exception as e:

        print("Ollama Error:", e)

        return None


# ============================================================
# FAQ SEARCH FUNCTION
# ============================================================

def search_faq(question, threshold=0.5):

    # --------------------------------------------------------
    # CREATE QUESTION EMBEDDING
    # --------------------------------------------------------

    question_embedding = model.encode(
        [question]
    )


    # --------------------------------------------------------
    # CALCULATE COSINE SIMILARITY
    # --------------------------------------------------------

    similarities = cosine_similarity(
        question_embedding,
        faq_embeddings
    )[0]


    # --------------------------------------------------------
    # FIND BEST FAQ
    # --------------------------------------------------------

    best_index = np.argmax(
        similarities
    )

    best_score = float(
        similarities[best_index]
    )


    # --------------------------------------------------------
    # CHECK THRESHOLD
    # --------------------------------------------------------

    if best_score < threshold:

        return None


    # --------------------------------------------------------
    # GET FAQ
    # --------------------------------------------------------

    best_question = str(
        df.iloc[best_index]["Question"]
    )

    best_answer = str(
        df.iloc[best_index]["Answer"]
    )


    # --------------------------------------------------------
    # OLLAMA PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an e-commerce customer support assistant.

Answer the customer's question using ONLY the FAQ answer.

Customer Question:
{question}

FAQ Question:
{best_question}

FAQ Answer:
{best_answer}

IMPORTANT RULES:
- Use only the information in the FAQ answer.
- Do not invent information.
- Do not add unrelated information.
- Keep the answer short and clear.
- Do not mention that you are an AI.
- Do not mention limitations.
"""


    # --------------------------------------------------------
    # GET OLLAMA RESPONSE
    # --------------------------------------------------------

    response = ask_ollama(
        prompt
    )


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not response:

        response = best_answer


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {

        "question": best_question,

        "answer": best_answer,

        "score": best_score,

        "response": response

    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("FAQ MODULE")
    print("=" * 60)

    question = input(
        "\nAsk Question : "
    ).strip()


    if not question:

        print("\nPlease enter a question.")

        exit()


    result = search_faq(
        question
    )


    if result is None:

        print(
            "\nNo suitable FAQ found."
        )

    else:

        print(
            "\nBest Matching FAQ :",
            result["question"]
        )

        print(
            "Similarity Score :",
            round(result["score"], 4)
        )

        print(
            "\nFAQ matched!"
        )

        print(
            "\nChatbot:"
        )

        print(
            result["response"]
        )
