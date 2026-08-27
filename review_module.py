import pandas as pd
import requests

from transformers import pipeline


# ============================================================
# LOAD REVIEW DATASET
# ============================================================

print("Loading review dataset...")

reviews = pd.read_csv("data/reviews.csv")

print("Number of reviews:", len(reviews))


# ============================================================
# LOAD SENTIMENT MODEL
# ============================================================

print("\nLoading pre-trained sentiment analysis model...")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

print("Sentiment model loaded successfully!")


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(prompt):

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen:0.5b",
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

        return "Unable to generate AI summary."


# ============================================================
# FIND PRODUCT
# ============================================================

def find_product(product_name):

    product_name = product_name.strip().lower()

    for name in reviews["ProductName"].dropna().unique():

        if str(name).strip().lower() == product_name:

            return name

    return None


# ============================================================
# ANALYZE PRODUCT REVIEWS
# ============================================================

def analyze_product_reviews(product_name):

    # --------------------------------------------------------
    # FIND PRODUCT
    # --------------------------------------------------------

    actual_product = find_product(product_name)

    if actual_product is None:

        print("\nProduct not found in reviews dataset.")

        return None


    # --------------------------------------------------------
    # GET PRODUCT REVIEWS
    # --------------------------------------------------------

    product_reviews = reviews[
        reviews["ProductName"].astype(str).str.lower()
        == str(actual_product).lower()
    ]


    print(
        "\nNumber of reviews for",
        actual_product,
        ":",
        len(product_reviews)
    )


    if len(product_reviews) == 0:

        print("\nNo reviews available for this product.")

        return None


    # --------------------------------------------------------
    # SENTIMENT COUNTS
    # --------------------------------------------------------

    positive = 0
    neutral = 0
    negative = 0

    analyzed_reviews = []


    print("\nAnalyzing reviews...")


    # --------------------------------------------------------
    # ANALYZE EACH REVIEW
    # --------------------------------------------------------

    for review in product_reviews["Review"].astype(str):

        result = sentiment_model(review)[0]

        label = result["label"].lower()
        confidence = result["score"]


        # Convert labels to our categories
        if "positive" in label:

            sentiment = "positive"
            positive += 1

        elif "negative" in label:

            sentiment = "negative"
            negative += 1

        else:

            sentiment = "neutral"
            neutral += 1


        analyzed_reviews.append({
            "review": review,
            "sentiment": sentiment,
            "confidence": confidence
        })


    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SENTIMENT SUMMARY")
    print("=" * 60)

    print("Positive :", positive)
    print("Neutral  :", neutral)
    print("Negative :", negative)


    # --------------------------------------------------------
    # PRINT DETAILS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DETAILED REVIEWS")
    print("=" * 60)


    for item in analyzed_reviews:

        print("\nReview:", item["review"])
        print("Sentiment:", item["sentiment"])
        print(
            "Confidence:",
            round(item["confidence"], 3)
        )

        print("-" * 40)


    # --------------------------------------------------------
    # CREATE REVIEW CONTEXT
    # --------------------------------------------------------

    review_context = ""

    for item in analyzed_reviews:

        review_context += f"""
Review: {item['review']}
Sentiment: {item['sentiment']}
Confidence: {item['confidence']:.3f}

"""


    # --------------------------------------------------------
    # OLLAMA PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an e-commerce review analysis assistant.

Analyze the customer reviews for:

Product: {actual_product}

Sentiment summary:

Positive reviews: {positive}
Neutral reviews: {neutral}
Negative reviews: {negative}

Customer reviews:

{review_context}

IMPORTANT RULES:

1. Use ONLY the information provided above.
2. Do not invent product features.
3. Do not invent complaints.
4. Do not invent strengths.
5. Do not mention information that is not present.
6. Do not say you need more information.
7. Give a practical buying suggestion.

Your answer must contain:

Overall Opinion:
Give a short summary of the overall customer sentiment.

Strengths:
Mention only strengths actually supported by the reviews.

Weaknesses:
Mention only weaknesses actually supported by the reviews.

Buying Suggestion:
Say whether the product appears to be a good choice based ONLY
on the review sentiment and review content.

Keep the answer concise.
"""


    # --------------------------------------------------------
    # GET AI SUMMARY
    # --------------------------------------------------------

    ai_summary = ask_ollama(prompt)


    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("OLLAMA REVIEW SUMMARY")
    print("=" * 60)

    print(ai_summary)


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "ProductName": actual_product,
        "Positive": positive,
        "Neutral": neutral,
        "Negative": negative,
        "Reviews": analyzed_reviews,
        "Summary": ai_summary
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("REVIEW MODULE")
    print("=" * 60)

    product_name = input("\nEnter Product : ").strip()

    analyze_product_reviews(product_name)
