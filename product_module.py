import pandas as pd
import subprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD PRODUCT DATASET
# ============================================================

print("Loading product dataset...")

products = pd.read_csv(
    "data/products.csv"
)

print(
    "Number of products:",
    len(products)
)


# ============================================================
# CREATE SEARCH TEXT
# ============================================================

products["search_text"] = (

    products["ProductName"].fillna("") + " " +

    products["Category"].fillna("") + " " +

    products["Brand"].fillna("") + " " +

    products["Description"].fillna("")
)


# ============================================================
# CREATE TF-IDF MODEL
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)

product_vectors = vectorizer.fit_transform(
    products["search_text"]
)


# ============================================================
# OLLAMA FUNCTION
# ============================================================

def ask_ollama(prompt):

    result = subprocess.run(

        [
            "ollama",
            "run",
            "qwen:0.5b",
            prompt
        ],

        capture_output=True,

        text=True
    )

    return result.stdout.strip()


# ============================================================
# PRODUCT QUESTION ANSWERING
# ============================================================

def answer_product_question(query):

    query_lower = query.lower()

    # --------------------------------------------------------
    # FIND EXACT PRODUCT NAME
    # --------------------------------------------------------

    exact_match = None

    for index, product in products.iterrows():

        product_name = str(
            product["ProductName"]
        ).lower()

        if product_name in query_lower:

            exact_match = index

            break

    # --------------------------------------------------------
    # RETRIEVE PRODUCT
    # --------------------------------------------------------

    if exact_match is not None:

        best_index = exact_match

        similarity_score = 1.0

    else:

        query_vector = vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            product_vectors
        )[0]

        best_index = similarities.argmax()

        similarity_score = float(
            similarities[best_index]
        )

    # --------------------------------------------------------
    # GET PRODUCT
    # --------------------------------------------------------

    product = products.iloc[
        best_index
    ]

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = f"""
Product Name: {product['ProductName']}
Brand: {product['Brand']}
Category: {product['Category']}
Price: ₹{product['Price']}
Rating: {product['Rating']}
Description: {product['Description']}
"""

    # --------------------------------------------------------
    # OLLAMA PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an e-commerce product assistant.

Answer the user's question using ONLY the product
information provided below.

IMPORTANT RULES:

1. Do not invent information.
2. Do not change the product name.
3. Do not change the brand.
4. Do not change the price.
5. Do not change the rating.
6. Use the exact price from the context.
7. Use ₹ for the price.
8. Give a short answer.

User Question:
{query}

Product Context:
{context}
"""

    # --------------------------------------------------------
    # SEND TO OLLAMA
    # --------------------------------------------------------

    response = ask_ollama(
        prompt
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "product_name":
            product["ProductName"],

        "brand":
            product["Brand"],

        "category":
            product["Category"],

        "price":
            product["Price"],

        "rating":
            product["Rating"],

        "similarity":
            similarity_score,

        "response":
            response
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PRODUCT QUESTION ANSWERING MODULE")
    print("=" * 60)

    query = input(
        "\nAsk : "
    ).strip()

    result = answer_product_question(
        query
    )

    if result is None:

        print("\nProduct not found.")

    else:

        print("\nRetrieved Product")

        print(
            "Product Name :",
            result["product_name"]
        )

        print(
            "Brand        :",
            result["brand"]
        )

        print(
            "Category     :",
            result["category"]
        )

        print(
            "Price        : ₹",
            result["price"]
        )

        print(
            "Rating       :",
            result["rating"]
        )

        print(
            "Similarity   :",
            round(
                result["similarity"],
                4
            )
        )

        print("\nAssistant:")

        print(
            result["response"]
        )
