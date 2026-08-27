import requests
import pandas as pd

from intent_module import detect_intent
from faq_module import search_faq
from recommendation_module import recommend_products
from product_module import answer_product_question
from review_module import analyze_product_reviews


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen:0.5b"


# ============================================================
# OLLAMA FUNCTION
# ============================================================

def ollama_response(prompt):

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

        return ""


# ============================================================
# FIND PRODUCT NAME IN QUESTION
# ============================================================

def find_product_name(question):

    """
    Search for a product name inside the user's question.

    Example:

    Question:
    What do customers think about Lenovo Laptop 1?

    Returns:
    Lenovo Laptop 1
    """

    try:

        reviews = pd.read_csv("data/reviews.csv")

    except Exception as e:

        print("Error loading reviews dataset:", e)

        return None


    question_lower = question.lower()


    # --------------------------------------------------------
    # GET UNIQUE PRODUCT NAMES
    # --------------------------------------------------------

    product_names = reviews["ProductName"].dropna().unique()


    # --------------------------------------------------------
    # EXACT PRODUCT NAME SEARCH
    # --------------------------------------------------------

    for product in product_names:

        product_name = str(product).strip()

        if product_name.lower() in question_lower:

            return product_name


    # --------------------------------------------------------
    # PRODUCT NAME NOT FOUND
    # --------------------------------------------------------

    return None


# ============================================================
# MAIN CHATBOT
# ============================================================

def main():

    print("=" * 60)
    print("E-COMMERCE AI CHATBOT")
    print("=" * 60)

    print("\nType 'exit' to stop the chatbot.")


    while True:

        # ====================================================
        # GET USER QUESTION
        # ====================================================

        question = input("\nYou : ").strip()


        # ====================================================
        # EXIT
        # ====================================================

        if question.lower() == "exit":

            print(
                "\nThank you for using the "
                "E-Commerce AI Chatbot!"
            )

            break


        # ====================================================
        # EMPTY INPUT
        # ====================================================

        if not question:

            continue


        # ====================================================
        # INTENT DETECTION
        # ====================================================

        intent = detect_intent(question)

        print("\nDetected Intent :", intent)


        # ====================================================
        # FAQ MODULE
        # ====================================================

        if intent == "FAQ":

            print("\nLoading FAQ Module...")


            result = search_faq(question)


            if result is None:

                print("\nNo suitable FAQ found.")

                continue


            # ------------------------------------------------
            # DISPLAY MATCHED FAQ
            # ------------------------------------------------

            print(
                "\nBest Matching FAQ :",
                result["question"]
            )

            print(
                "Similarity Score :",
                round(result["score"], 4)
            )


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            print("\nAssistant:")

            print(result["response"])


        # ====================================================
        # RECOMMENDATION MODULE
        # ====================================================

        elif intent == "RECOMMENDATION":

            print("\nLoading Recommendation Module...")


            results, response = recommend_products(question)


            if not results:

                print("\nNo suitable products found.")

                continue


            # ------------------------------------------------
            # DISPLAY PRODUCTS
            # ------------------------------------------------

            print("\nRetrieved Products")

            print("-" * 60)


            for product in results:

                print(
                    f"{product['ProductName']} | "
                    f"Brand: {product['Brand']} | "
                    f"Price: ₹{product['Price']} | "
                    f"Rating: {product['Rating']} | "
                    f"Similarity: "
                    f"{product['Similarity']:.4f}"
                )


            # ------------------------------------------------
            # DISPLAY OLLAMA RECOMMENDATION
            # ------------------------------------------------

            print("\nAssistant:")

            print(response)


        # ====================================================
        # PRODUCT MODULE
        # ====================================================

        elif intent == "PRODUCT":

            print("\nLoading Product QA Module...")


            result = answer_product_question(question)


            if result is None:

                print("\nProduct not found.")

                continue


            # ------------------------------------------------
            # DISPLAY PRODUCT
            # ------------------------------------------------

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
                round(result["similarity"], 4)
            )


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            print("\nAssistant:")

            print(result["response"])


        # ====================================================
        # REVIEW MODULE
        # ====================================================

        elif intent == "REVIEW":

            print("\nLoading Review Module...")


            # ------------------------------------------------
            # FIND PRODUCT NAME IN QUESTION
            # ------------------------------------------------

            selected_product = find_product_name(question)


            # ------------------------------------------------
            # IF PRODUCT NAME NOT FOUND
            # ------------------------------------------------

            if selected_product is None:

                print(
                    "\nI could not find a product name "
                    "in your question."
                )

                selected_product = input(
                    "\nEnter Product Name : "
                ).strip()


            # ------------------------------------------------
            # EMPTY PRODUCT NAME
            # ------------------------------------------------

            if not selected_product:

                print("\nProduct name cannot be empty.")

                continue


            # ------------------------------------------------
            # ANALYZE REVIEWS
            # ------------------------------------------------

            analyze_product_reviews(selected_product)


        # ====================================================
        # UNKNOWN INTENT
        # ====================================================

        else:

            print(
                "\nSorry, I could not understand your request."
            )

            print(
                "Try asking about orders, products, "
                "recommendations, or reviews."
            )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()
