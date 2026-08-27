import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen:0.5b"


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(question):

    question_lower = question.lower().strip()


    # ========================================================
    # RULE 1: FAQ KEYWORDS
    # ========================================================

    faq_keywords = [
        "where is my parcel",
        "where is my order",
        "track my order",
        "track order",
        "tracking my order",
        "delivery",
        "shipping",
        "return",
        "refund",
        "cancel my order",
        "cancellation",
        "payment",
        "order status"
    ]

    for keyword in faq_keywords:

        if keyword in question_lower:
            return "FAQ"


    # ========================================================
    # RULE 2: RECOMMENDATION KEYWORDS
    # ========================================================

    recommendation_keywords = [
        "recommend",
        "recommendation",
        "suggest",
        "suggestion",
        "which laptop should i buy",
        "which phone should i buy",
        "which mobile should i buy",
        "which product should i buy",
        "what should i buy",
        "best laptop",
        "best phone",
        "best mobile",
        "best product",
        "help me choose"
    ]

    for keyword in recommendation_keywords:

        if keyword in question_lower:
            return "RECOMMENDATION"


    # ========================================================
    # RULE 3: REVIEW KEYWORDS
    # ========================================================

    review_keywords = [
        "review",
        "reviews",
        "customer reviews",
        "customer opinion",
        "customers think",
        "customer feedback",
        "feedback",
        "sentiment",
        "pros and cons",
        "strengths and weaknesses",
        "is it good according to customers"
    ]

    for keyword in review_keywords:

        if keyword in question_lower:
            return "REVIEW"


    # ========================================================
    # RULE 4: SPECIFIC PRODUCT QUESTIONS
    # ========================================================

    product_keywords = [
        "price",
        "cost",
        "brand",
        "rating",
        "category",
        "description",
        "tell me about",
        "information about",
        "details about"
    ]

    for keyword in product_keywords:

        if keyword in question_lower:
            return "PRODUCT"


    # ========================================================
    # OLLAMA FALLBACK
    # ========================================================

    prompt = f"""
Classify the question into exactly one of these four labels:

FAQ
RECOMMENDATION
PRODUCT
REVIEW

Rules:

FAQ:
Questions about orders, delivery, shipping, tracking,
returns, refunds, payments and cancellations.

RECOMMENDATION:
Questions asking to recommend, suggest, choose, or find
a product to buy.

PRODUCT:
Questions about a specific product's price, brand,
rating, category or description.

REVIEW:
Questions about customer reviews, opinions, feedback,
sentiment, strengths or weaknesses.

IMPORTANT:
Return ONLY one of these exact words:
FAQ
RECOMMENDATION
PRODUCT
REVIEW

Never return a product name.

Question:
{question}
"""


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

        ollama_answer = response.json()["response"].strip()

        print("\nOllama Response:", repr(ollama_answer))

        answer = ollama_answer.upper()


        # ====================================================
        # EXTRACT VALID CATEGORY
        # ====================================================

        if "RECOMMENDATION" in answer:
            return "RECOMMENDATION"

        elif "FAQ" in answer:
            return "FAQ"

        elif "REVIEW" in answer:
            return "REVIEW"

        elif "PRODUCT" in answer:
            return "PRODUCT"

        else:
            return "UNKNOWN"


    except Exception as e:

        print("\nOllama error:", e)

        return "UNKNOWN"


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("INTENT DETECTION MODULE")
    print("=" * 60)

    question = input("\nQuestion : ")

    intent = detect_intent(question)

    print("\nDetected Intent :", intent)
