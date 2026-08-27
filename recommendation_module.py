import pandas as pd
import subprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD PRODUCT DATASET
# ============================================================

print("Loading product dataset...")

products = pd.read_csv("data/products.csv")

print("Number of products:", len(products))


# ============================================================
# CREATE SEARCH TEXT
# ============================================================

products["search_text"] = (
    products["ProductName"].fillna("").astype(str) + " " +
    products["Category"].fillna("").astype(str) + " " +
    products["Brand"].fillna("").astype(str) + " " +
    products["Description"].fillna("").astype(str)
)


# ============================================================
# CREATE TF-IDF VECTORS
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
        ["ollama", "run", "qwen:0.5b", prompt],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Ollama Error:")
        print(result.stderr)

        return "No recommendation explanation available."

    return result.stdout.strip()


# ============================================================
# PRODUCT RECOMMENDATION
# ============================================================

def recommend_products(query, top_k=3):

    # --------------------------------------------------------
    # CONVERT QUERY TO VECTOR
    # --------------------------------------------------------

    query_vector = vectorizer.transform([query])


    # --------------------------------------------------------
    # CALCULATE SIMILARITY
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_vector,
        product_vectors
    )[0]


    # --------------------------------------------------------
    # GET TOP PRODUCTS
    # --------------------------------------------------------

    top_indices = similarities.argsort()[-top_k:][::-1]


    # --------------------------------------------------------
    # CREATE RESULTS
    # --------------------------------------------------------

    results = []

    for index in top_indices:

        product = products.iloc[index]

        results.append({
            "ProductName": product["ProductName"],
            "Brand": product["Brand"],
            "Category": product["Category"],
            "Price": product["Price"],
            "Rating": product["Rating"],
            "Description": product["Description"],
            "Similarity": similarities[index]
        })


    # ========================================================
    # IMPORTANT:
    # PYTHON CHOOSES THE BEST PRODUCT
    # ========================================================

    best_product = results[0]


    # ========================================================
    # OLLAMA ONLY GENERATES THE REASON
    # ========================================================

    prompt = f"""
You are an e-commerce product assistant.

The Python system has already selected this product:

Product Name: {best_product['ProductName']}
Brand: {best_product['Brand']}
Price: ₹{best_product['Price']}
Rating: {best_product['Rating']}
Category: {best_product['Category']}
Description: {best_product['Description']}

User request:
{query}

Give ONE short reason why this product is suitable.

IMPORTANT:
- Use ONLY the information provided above.
- Do NOT invent features.
- Do NOT mention any other product.
- Do NOT change the product name.
- Do NOT change the brand.
- Do NOT change the price.
- Do NOT use dollars.
- Do NOT mention information that is not in the context.
- Give only ONE short sentence.
"""


    reason = ask_ollama(prompt)


    # --------------------------------------------------------
    # STORE FINAL ANSWER
    # --------------------------------------------------------

    response = f"""Product: {best_product['ProductName']}
Brand: {best_product['Brand']}
Price: ₹{best_product['Price']}
Rating: {best_product['Rating']}
Reason: {reason}"""


    return results, response


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("PRODUCT RECOMMENDATION MODULE")
    print("=" * 60)

    print("\nSearch Product : ", end="")

    query = input().strip()


    # --------------------------------------------------------
    # EMPTY QUERY
    # --------------------------------------------------------

    if not query:

        print("\nPlease enter a product request.")

        exit()


    # --------------------------------------------------------
    # GET RECOMMENDATIONS
    # --------------------------------------------------------

    results, response = recommend_products(query)


    # --------------------------------------------------------
    # DISPLAY RETRIEVED PRODUCTS
    # --------------------------------------------------------

    print("\nRetrieved Products")

    print("-" * 60)

    for product in results:

        print(
            f"{product['ProductName']} | "
            f"Brand: {product['Brand']} | "
            f"Price: ₹{product['Price']} | "
            f"Rating: {product['Rating']} | "
            f"Similarity: {product['Similarity']:.4f}"
        )


    # --------------------------------------------------------
    # DISPLAY FINAL RECOMMENDATION
    # --------------------------------------------------------

    print("\nAssistant:")

    print(response)
