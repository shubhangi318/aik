from typing import Dict, List, Any, Tuple
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import base64
from io import BytesIO
import statistics

# Download VADER lexicon if not already installed
import ssl
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Then try to download VADER lexicon
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# List of speculative/hedging words and phrases
HEDGE_WORDS = [
    'may', 'might', 'could', 'can', 'possibly', 'perhaps', 'allegedly', 
    'reportedly', 'rumored', 'expected to', 'likely to', 'potentially',
    'supposedly', 'apparent', 'alleged', 'rumored', 'claimed', 'possible',
    'probable', 'speculate', 'predict', 'forecast', 'anticipate', 'estimate',
    'projected', 'provisional', 'preliminary', 'unconfirmed', 'not verified',
    'uncertain', 'unclear', 'unproven', 'suggest', 'indicate', 'imply',
    'consider', 'believe', 'think', 'assume', 'suspect', 'presume'
]

# Compile hedge word patterns for faster matching
HEDGE_PATTERNS = [re.compile(r'\b' + word + r'\b', re.IGNORECASE) for word in HEDGE_WORDS]
HEDGE_PHRASE_PATTERNS = [
    re.compile(r'\b' + phrase.replace(' ', r'\s+') + r'\b', re.IGNORECASE) 
    for phrase in HEDGE_WORDS if ' ' in phrase
]

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Comprehensive sentiment analysis including basic classification,
    speculation detection, and intensity analysis.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # Initialize VADER for intensity analysis
    sia = SentimentIntensityAnalyzer()
    
    # Basic TextBlob sentiment analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Determine basic sentiment category
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    # VADER intensity analysis (provides more granular scores)
    vader_scores = sia.polarity_scores(text)
    
    # Detect speculation/hedging
    speculation_score, hedge_words_found = detect_speculation(text)
    
    # Determine if the content is speculative or confirmed
    speculation_threshold = 0.15  # Adjust based on testing
    is_speculative = speculation_score > speculation_threshold
    certainty_type = "speculative" if is_speculative else "confirmed"
    
    # Sentiment intensity categorization
    intensity = categorize_sentiment_intensity(vader_scores['compound'])
    
    return {
        # Basic sentiment
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": subjectivity,
        
        # VADER detailed scores
        "vader_compound": vader_scores['compound'],
        "vader_pos": vader_scores['pos'],
        "vader_neg": vader_scores['neg'],
        "vader_neu": vader_scores['neu'],
        
        # Speculation analysis
        "speculation_score": speculation_score,
        "hedge_words": hedge_words_found,
        "certainty_type": certainty_type,
        
        # Sentiment intensity
        "intensity": intensity,
    }

def detect_speculation(text: str) -> Tuple[float, List[str]]:
    """
    Detect speculative language in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple of (speculation_score, list_of_hedge_words_found)
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    sentences = nltk.sent_tokenize(text)
    total_sentences = len(sentences)
    
    # Count sentences with hedge words
    speculative_sentences = 0
    hedge_words_found = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        found_hedge = False
        
        # Check for hedge words/phrases
        for pattern in HEDGE_PATTERNS:
            matches = pattern.findall(sentence_lower)
            if matches:
                hedge_words_found.extend(matches)
                found_hedge = True
        
        # Check for hedge phrases
        for pattern in HEDGE_PHRASE_PATTERNS:
            if pattern.search(sentence_lower):
                phrase = pattern.pattern.replace(r'\b', '').replace(r'\s+', ' ')
                hedge_words_found.append(phrase)
                found_hedge = True
        
        if found_hedge:
            speculative_sentences += 1
    
    # Calculate speculation score (proportion of sentences with hedging)
    speculation_score = speculative_sentences / total_sentences if total_sentences > 0 else 0
    
    # Remove duplicates from hedge_words_found
    hedge_words_found = list(set(hedge_words_found))
    
    return speculation_score, hedge_words_found

def categorize_sentiment_intensity(compound_score: float) -> str:
    """
    Categorize sentiment intensity based on VADER compound score.
    
    Args:
        compound_score: VADER compound sentiment score (-1 to 1)
        
    Returns:
        String describing the sentiment intensity
    """
    if compound_score >= 0.75:
        return "extremely positive"
    elif compound_score >= 0.5:
        return "very positive"
    elif compound_score >= 0.25:
        return "moderately positive"
    elif compound_score > 0.1:
        return "slightly positive"
    elif compound_score > -0.1:
        return "neutral"
    elif compound_score > -0.25:
        return "slightly negative"
    elif compound_score > -0.5:
        return "moderately negative"
    elif compound_score > -0.75:
        return "very negative"
    else:
        return "extremely negative"

# Make sure this function is properly defined in sentiment.py
def compare_sentiments(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform comparative analysis across multiple articles.
    
    Args:
        articles: List of article dictionaries with sentiment data
        
    Returns:
        Dictionary with comparative analysis results
    """
    if not articles:
        return {
            "average_sentiment": "N/A",
            "charts": {}
        }
    
    # Extract sentiment data
    polarities = [article.get('polarity', 0) for article in articles]
    sentiments = [article.get('sentiment', 'neutral') for article in articles]
    
    # Calculate statistics
    avg_polarity = statistics.mean(polarities) if polarities else 0
    
    # Determine overall sentiment
    if avg_polarity > 0.1:
        overall_sentiment = "positive"
    elif avg_polarity < -0.1:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"
    
    # Count sentiment distribution
    sentiment_count = {
        "positive": sentiments.count("positive"),
        "neutral": sentiments.count("neutral"),
        "negative": sentiments.count("negative")
    }
    
    # Generate only the news categories bubble chart
    charts = {}
    charts["news_categories_chart"] = generate_news_categories_bubble_chart(articles)
    
    return {
        "average_sentiment": overall_sentiment,
        "average_polarity": avg_polarity,
        "sentiment_distribution": sentiment_count,
        "charts": charts
    }


def generate_news_categories_bubble_chart(articles: List[Dict[str, Any]]) -> str:
    """
    Generate a bubble chart categorizing articles into business-specific categories:
    - Bullish News (Positive Growth)
    - Bearish News (Negative Outlook)
    - Mergers & Acquisitions
    - Reputation Damage (Backlash News)
    - Brand Love (Positive Perception)
    - Government & Policy Framing
    
    Args:
        articles: List of article dictionaries with sentiment data
        
    Returns:
        Base64 encoded image of the chart
    """
    if not articles:
        return ""
    
    # Define the categories
    categories = [
        "Bullish News", 
        "Bearish News", 
        "Mergers & Acquisitions", 
        "Reputation Damage", 
        "Brand Love", 
        "Government & Policy"
    ]
    
    # Keywords to help categorize articles
    category_keywords = {
        "Bullish News": ["growth", "profit", "revenue", "expansion", "success", "funding", "investment", 
                         "increase", "record", "earnings", "exceed", "beat", "up", "rise", "soar", "gain"],
        "Bearish News": ["loss", "decline", "scandal", "layoff", "bankruptcy", "plunge", "drop", "below", 
                         "miss", "fall", "decrease", "struggle", "shrink", "poor", "down", "negative"],
        "Mergers & Acquisitions": ["merger", "acquisition", "acquire", "buy", "purchase", "takeover", "partnership", 
                                   "collaborate", "joint venture", "deal", "alliance", "stake"],
        "Reputation Damage": ["controversy", "scandal", "backlash", "protest", "lawsuit", "toxic", "criticism", 
                              "resign", "apologize", "accusation", "failure", "problem", "issue", "risk"],
        "Brand Love": ["donate", "philanthropy", "award", "recognition", "customer satisfaction", "positive", 
                       "approval", "sustainable", "impact", "community", "charity", "responsible"],
        "Government & Policy": ["regulation", "compliance", "law", "policy", "government", "tax", "legal", 
                               "official", "authorities", "regulator", "legislation", "ban", "approve"]
    }
    
    # Initialize data structure to hold article data by category
    category_data = {cat: [] for cat in categories}
    
    # Categorize each article
    for i, article in enumerate(articles):
        title = article.get('title', '').lower()
        summary = article.get('summary', '').lower()
        content = article.get('raw_content', '').lower()
        
        # Combine text for category matching
        full_text = title + " " + summary
        
        # Get sentiment metrics for sizing/coloring
        sentiment_score = article.get('vader_compound', 0)
        
        # Determine article size (by content length)
        content_length = len(content)
        
        # Determine best matching category
        best_category = None
        max_matches = 0
        
        for category, keywords in category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in full_text)
            if matches > max_matches:
                max_matches = matches
                best_category = category
        
        # If no keywords matched, use sentiment to categorize
        if max_matches == 0:
            if sentiment_score >= 0.3:
                best_category = "Bullish News"
            elif sentiment_score <= -0.3:
                best_category = "Bearish News"
            else:
                best_category = "Government & Policy"  # Default fallback
        
        # Add the article to the appropriate category
        if best_category:
            # Store article data: (article index, sentiment score, content length)
            category_data[best_category].append((i+1, sentiment_score, content_length))
    
    # Prepare for plotting
    plt.figure(figsize=(14, 10), facecolor='white')
    plt.grid(True, color='#EEEEEE', linestyle='-', linewidth=0.5)
    
    # Colors for different categories
    colors = {
        "Bullish News": '#4CAF50',      # Green
        "Bearish News": '#F44336',      # Red
        "Mergers & Acquisitions": '#2196F3',  # Blue
        "Reputation Damage": '#FF9800',  # Orange
        "Brand Love": '#9C27B0',        # Purple
        "Government & Policy": '#607D8B'  # Gray
    }
    
    # Create bubble chart
    for y, category in enumerate(categories):
        for article_data in category_data[category]:
            article_id, sentiment_score, content_length = article_data
            
            # Use sentiment score for x-position (normalized to 0-1 scale)
            x_pos = (sentiment_score + 1) / 2  # Convert -1 to 1 scale to 0 to 1
            
            # Size based on content length, with minimum to ensure visibility
            size = 1000 + (content_length / 100)
            
            # Plot the bubble
            plt.scatter(
                x_pos, 
                y, 
                s=size, 
                color=colors[category], 
                alpha=0.7,
                edgecolors='white'
            )
            
            # Add article number label
            plt.annotate(
                str(article_id),
                (x_pos, y),
                ha='center',
                va='center',
                color='white',
                fontweight='bold'
            )
    
    # Add labels and title
    plt.title('Article Sentiment Comparison by Category', fontsize=18)
    plt.xlabel('Sentiment Score (Negative to Positive)', fontsize=12)
    plt.yticks(range(len(categories)), categories, fontsize=12)
    
    # Add category descriptions in a legend box
    category_desc = {
        "Bullish News": "Expansion, profits, funding, acquisitions",
        "Bearish News": "Losses, scandals, layoffs, bankruptcies",
        "Mergers & Acquisitions": "Partnerships or buyouts",
        "Reputation Damage": "Controversies, PR disasters, scandals",
        "Brand Love": "Goodwill, customer satisfaction, philanthropy",
        "Government & Policy": "Political influence, regulatory perspectives"
    }
    
    # Create a custom legend
    legend_elements = []
    for cat, color in colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=10, 
                              label=f"{cat}: {category_desc[cat]}"))
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=2, fontsize=10)
    
    plt.tight_layout()
    
    # Save chart as base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Generate base64 string
    chart_str = base64.b64encode(image_png).decode('utf-8')
    
    return chart_str
