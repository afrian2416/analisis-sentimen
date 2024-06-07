import re
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Initialize Sastrawi stopword remover
factory = StopWordRemoverFactory()
indonesian_stopwords = set(factory.get_stop_words())

# Load model and tokenizer from Hugging Face
model_name = "Edelweisse/bert-sentiment"
auth_token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=auth_token)
label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

# Define function for single sentence analysis
def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    sentiment = label_index[model.config.id2label[predicted_class]]
    score = torch.softmax(outputs.logits, dim=1)[0][predicted_class].item()
    return {'label': sentiment, 'score': score}

# Define function for cleaning text
def clean_text(text):
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphanumeric characters and symbols
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Define function for batch analysis
def analyze_sentiment_batch(texts):
    cleaned_texts = [clean_text(text) for text in texts]
    cleaned_texts = [text for text in cleaned_texts if len(text.split()) >= 5]
    sentiments = []
    for text in cleaned_texts:
        result = sentiment_analysis(text)
        sentiments.append(result['label'])
    return cleaned_texts, sentiments

# Main function
def main():
    st.title("Sentiment Analysis")
    tabs = st.tabs(["Single Analysis", "Batch Analysis"])

    with tabs[0]:
        st.title("Single Sentiment Analysis")
        sentence = st.text_input("Enter a sentence in Indonesian:")
        if st.button("Single Analyze"):
            if sentence:
                result = sentiment_analysis(sentence)
                status = result['label']
                score = result['score']
                st.write(f'Text: {sentence} | Label : {status} ({score * 100:.3f}%)')

    with tabs[1]:
        st.title("Batch Sentiment Analysis")
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            # Read the file into a DataFrame
            delimiter = st.text_input("Enter the delimiter (e.g., ',' , ';' , '\\t' , etc.):", ',')
            df = pd.read_csv(uploaded_file, delimiter=delimiter)

            # Display uploaded data
            st.subheader("Uploaded Data")
            st.write(df)

            # Get text columns
            text_columns = [col for col in df.columns if df[col].dtype == 'object']

            if len(text_columns) == 0:
                st.write("No text columns found in the uploaded file.")
            else:
                # Let user select a text column
                selected_column = st.selectbox("Select the column to analyze:", text_columns)

                if st.button("Batch Analyze"):
                    # Get text data from the selected column
                    texts = df[selected_column].tolist()

                    # Analyze sentiment for batch of sentences
                    cleaned_texts, sentiments = analyze_sentiment_batch(texts)

                    # Create a DataFrame with cleaned text and sentiments
                    results_df = pd.DataFrame({
                        'text': cleaned_texts,
                        'sentiment': sentiments
                    })

                    # Display sentiment distribution
                    sentiment_counts = Counter(sentiments)
                    st.subheader("Sentiment Distribution")
                    fig = px.bar(
                        x=list(sentiment_counts.keys()),
                        y=list(sentiment_counts.values()),
                        labels={'x': 'Sentiment', 'y': 'Count'},
                        color=list(sentiment_counts.keys()),
                        color_discrete_map={
                            'positive': 'blue',
                            'neutral': 'gray',
                            'negative': 'red'
                        }
                    )
                    st.plotly_chart(fig)

                    # Generate word cloud for each sentiment
                    st.subheader("Word Cloud")
                    custom_stopwords = set(STOPWORDS)
                    custom_stopwords.update(['bpjs', 'muhammadiyah'])
                    custom_stopwords.update(indonesian_stopwords)

                    for sentiment in ["positive", "neutral", "negative"]:
                        sentiment_texts = [text for text, sent in zip(cleaned_texts, sentiments) if sent == sentiment]
                        all_texts = ' '.join(sentiment_texts)
                        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords).generate(all_texts)
                        st.write(f"{sentiment.capitalize()} Word Cloud")
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        st.pyplot(plt)

                    # Display the results DataFrame
                    st.subheader("Analyzed Data")
                    st.write(results_df)

if __name__ == "__main__":
    main()
