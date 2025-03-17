import streamlit as st
import requests
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import os
from PIL import Image
import time

import nltk
nltk.download('vader_lexicon')

# API endpoints
API_URL = "http://localhost:8000/api"

def get_news(company_name, num_articles=10):
    response = requests.post(
        f"{API_URL}/extract-news",
        json={"company_name": company_name, "num_articles": num_articles}
    )
    return response.json()

def analyze_sentiment(company_name, num_articles=10):
    response = requests.post(
        f"{API_URL}/analyze-sentiment",
        json={"company_name": company_name, "num_articles": num_articles}
    )
    return response.json()

def generate_speech(text):
    response = requests.post(
        f"{API_URL}/text-to-speech",
        json={"text": text}
    )
    return response.json()

def display_chart(chart_base64):
    if chart_base64:
        image_data = base64.b64decode(chart_base64)
        image = Image.open(BytesIO(image_data))
        st.image(image, caption="Sentiment Distribution")

def main():
    st.set_page_config(
        page_title="Company News Sentiment Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Company News Sentiment Analyzer")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Extract News", "Analyze Sentiment"])
    
    # Common inputs that will be used in both tabs
    company_name = st.sidebar.text_input("Enter Company Name", "Enter the company name here")
    num_articles = st.sidebar.number_input("Number of Articles", min_value=1, max_value=20, value=10)
    
    # Tab 1: Extract News Articles
    with tab1:
        st.header("News Article Extraction")
        st.write("Extract news articles for a company without sentiment analysis")
        
        extract_button = st.button("Extract News Articles")
        
        if extract_button:
            with st.spinner(f"Extracting news for {company_name}..."):
                try:
                    # Call the extract-news endpoint
                    articles = get_news(company_name, num_articles)
                    
                    if articles and len(articles) > 0:
                        st.success(f"Extraction completed! Found {len(articles)} articles")
                        
                        # Display articles in a table format
                        st.subheader("Extracted Articles")
                        
                        # Create a more compact summary table
                        article_data = []
                        for i, article in enumerate(articles):
                            article_data.append({
                                "No.": i+1,
                                "Title": article['title'],
                                "Source": article['source'],
                                "Date": article['date']
                            })
                        
                        article_df = pd.DataFrame(article_data)
                        st.dataframe(article_df)
                        
                        # Display detailed articles with expandable sections
                        st.subheader("Detailed Article Information")
                        for i, article in enumerate(articles):
                            with st.expander(f"{i+1}. {article['title']}"):
                                st.write(f"**Source:** {article['source']}")
                                st.write(f"**Date:** {article['date']}")
                                # st.write(f"**URL:** {article['url']}")
                                
                                # Summary
                                if 'summary' in article:
                                    st.write(f"**Summary:** {article['summary']}")
                                
                                st.write(f"**Original Link:** [View Article]({article['url']})")
                                # Keywords
                                if "keywords" in article and article["keywords"]:
                                    st.write("**Keywords:** " + ", ".join(article["keywords"]))
                                
                                st.write(f"**Relevance:** {article.get('relevance', 'N/A')}")

                                st.write("**Original Text:**")
                                st.write(article['raw_content'])  # Assuming 'raw_content' contains the original text
        
                    else:
                        st.error(f"No articles found for {company_name}")
                
                except Exception as e:
                    st.error(f"Error during extraction: {str(e)}")
    
    # Tab 2: Analyze Sentiment
    with tab2:
        st.header("News Sentiment Analysis")
        st.write("Analyze sentiment from news articles about a company")
        
        analyze_button = st.button("Analyze News Sentiment")
        
        if analyze_button:
            with st.spinner(f"Analyzing news for {company_name}..."):
                try:
                    # Get analysis results
                    results = analyze_sentiment(company_name, num_articles)
                    
                    if "articles" in results and len(results["articles"]) > 0:
                        st.success(f"Analysis completed! Found {len(results['articles'])} articles")
                        
                        # Display articles
                        st.header("News Articles with Sentiment")
                        
                        # Display the advanced sentiment information in the article expander
                        for i, article in enumerate(results["articles"]):
                            with st.expander(f"{i+1}. {article['title']}"):
                             
                                
                                # Sentiment information
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Sentiment", article.get('sentiment', 'Unknown'))
                                with col2:
                                    st.metric("Certainty Type", article.get('certainty_type', 'Unknown'))
                                with col3:
                                    st.metric("Intensity", article.get('intensity', 'Unknown'))
                                
                                # Sentiment details
                                st.write("**Sentiment Details:**")
                                sentiment_cols = st.columns(4)
                                with sentiment_cols[0]:
                                    st.metric("Polarity", f"{article.get('polarity', 0):.2f}")
                                with sentiment_cols[1]:
                                    st.metric("Subjectivity", f"{article.get('subjectivity', 0):.2f}")
                                with sentiment_cols[2]:
                                    st.metric("VADER Score", f"{article.get('vader_compound', 0):.2f}")
                                with sentiment_cols[3]:
                                    st.metric("Speculation", f"{article.get('speculation_score', 0):.2f}")
                                
                                # If speculative, show hedge words
                                if article.get('certainty_type') == 'speculative':
                                    hedge_words = article.get('hedge_words', [])
                                    if hedge_words:
                                        st.write("**Speculative language detected:**", ", ".join(hedge_words))
                                
                                # Summary and other info
                                st.write(f"**Summary:** {article['summary']}")
                        
                                # Link
                                st.write(f"**Original Link:** [View Article]({article['url']})")
                        
                    else:
                        st.error(f"No articles found for {company_name}")
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()