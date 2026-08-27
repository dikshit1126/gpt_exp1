# E-Commerce AI Chatbot

An AI-powered e-commerce chatbot that can understand customer questions and provide answers using FAQ retrieval, product recommendation, product information retrieval, and customer review sentiment analysis.

## Features

### 1. Intent Detection

The chatbot identifies the user's intent and routes the question to the appropriate module.

Supported intents:

- FAQ
- Product
- Recommendation
- Review
- Unknown

### 2. FAQ Question Answering

The chatbot uses semantic similarity to find the most relevant FAQ.

Example:

User:
Where is my parcel?

Chatbot:
Track Order page with your order ID.

### 3. Product Question Answering

The chatbot retrieves product information using TF-IDF and cosine similarity.

Example:

User:
What is the price of Lenovo Laptop 3?

Chatbot:
The price of Lenovo Laptop 3 is ₹7081.

### 4. Product Recommendation

The recommendation module retrieves the most relevant products based on the user's request.

Example:

User:
Recommend a laptop

The chatbot retrieves suitable laptops and recommends one based on the available product information.

### 5. Review Sentiment Analysis

The chatbot analyzes customer reviews using a pre-trained sentiment analysis model.

It classifies reviews as:

- Positive
- Neutral
- Negative

It also generates an overall review summary using Ollama.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- Transformers
- PyTorch
- Ollama
- Qwen 0.5B
- TF-IDF
- Cosine Similarity
- Sentiment Analysis

## Project Structure

```text
ecommerce-chatbot/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── chatbot.py
├── intent_module.py
├── faq_module.py
├── recommendation_module.py
├── product_module.py
├── review_module.py
├── create_embeddings.py
├── embedding_test.py
├── dataset_info.py
├── app.py
│
├── data/
│   ├── faqs.csv
│   ├── products.csv
│   └── reviews.csv
│
└── models/
    ├── faq_embeddings.npy
    ├── product_tfidf.pkl
    └── product_vectors.pkl
