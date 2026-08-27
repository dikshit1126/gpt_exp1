import pandas as pd


# Load datasets
faqs = pd.read_csv("data/faqs.csv")
products = pd.read_csv("data/products.csv")
reviews = pd.read_csv("data/reviews.csv")


def display_dataset(name, df):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # First five records
    print("\nFirst 5 Records:")
    print(df.head().to_string(index=False))

    # Number of rows and columns
    print("\nNumber of Rows:", df.shape[0])
    print("Number of Columns:", df.shape[1])

    # Attributes
    print("\nAttributes:")
    for column in df.columns:
        print("-", column)


# Display all datasets
display_dataset("FAQ DATASET", faqs)
display_dataset("PRODUCT DATASET", products)
display_dataset("REVIEWS DATASET", reviews)
