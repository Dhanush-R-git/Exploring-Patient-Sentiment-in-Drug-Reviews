import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("drugslib.csv")  # Replace with actual file
    return df

data = load_data()

# Sentiment Analysis Model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def get_sentiment_score(text):
    if pd.isna(text) or text.strip() == "":
        return 0  # Neutral
    result = sentiment_pipeline(text[:512])[0]  # Limit to 512 tokens
    if result["label"] == "LABEL_0":
        return -1  # Negative
    elif result["label"] == "LABEL_1":
        return 0  # Neutral
    else:
        return 1  # Positive

# Apply sentiment analysis
@st.cache_data
def process_sentiments(df):
    df['benefits_sentiment'] = df['benefitsReview'].apply(get_sentiment_score)
    df['sideEffects_sentiment'] = df['sideEffectsReview'].apply(get_sentiment_score)
    df['comments_sentiment'] = df['commentsReview'].apply(get_sentiment_score)
    df['avg_sentiment'] = df[['benefits_sentiment', 'sideEffects_sentiment', 'comments_sentiment']].mean(axis=1)
    return df

data = process_sentiments(data)

# Group by DrugName
drug_summary = data.groupby("DrugName").agg({
    "rating": "mean",
    "effectiveness": "mean",
    "avg_sentiment": "mean",
    "sideEffectsReview": lambda x: ' '.join(str(i) for i in x)
}).reset_index()

def plot_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig

# Streamlit UI
st.title("📊 Drug Analysis Report")
selected_drug = st.selectbox("Select a Drug:", drug_summary["DrugName"].unique())

drug_data = drug_summary[drug_summary["DrugName"] == selected_drug]

st.subheader(f"Overview of {selected_drug}")
st.metric("Average Rating", round(drug_data["rating"].values[0], 2))
st.metric("Effectiveness", round(drug_data["effectiveness"].values[0], 2))
st.metric("Sentiment Score", round(drug_data["avg_sentiment"].values[0], 2))

# Sentiment Distribution
sentiment_counts = data[data["DrugName"] == selected_drug]["avg_sentiment"].value_counts().reset_index()
fig_sentiment = px.pie(sentiment_counts, names='index', values='avg_sentiment', title="Sentiment Distribution")
st.plotly_chart(fig_sentiment)

# Word Cloud for Side Effects
st.subheader("Common Side Effects")
side_effects_text = drug_data["sideEffectsReview"].values[0]
st.pyplot(plot_wordcloud(side_effects_text))

# Average Rating Bar Chart
fig_rating = px.bar(drug_summary, x="DrugName", y="rating", title="Average Rating per Drug", height=500)
st.plotly_chart(fig_rating)

st.success("Analysis Completed!")
