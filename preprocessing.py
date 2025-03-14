import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def clean_text(text):
    """Cleans text by removing unwanted symbols and digits."""
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_data(df):
    df['review'] = df['review'].apply(clean_text)

    stop_words = set(stopwords.words("english"))
    #stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    # df["processed_review"] = "
    # Tokenize, process, and store each review
    for i, row in df.iterrows():
        review = row["review"]
        tokens = word_tokenize(review)
        tokens = [token for token in tokens if token not in stop_words]

        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Choose either stemmed or lemmatized tokens
        processed_tokens = stemmed_tokens  # Or use lemmatized_tokens

        # Combine processed tokens back into a string
        processed_review = " ".join(processed_tokens)

        # Store the processed review in the new column
        # df.loc[i, "processed_review"] = processed_review

    return df
