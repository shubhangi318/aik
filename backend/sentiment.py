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
            "sentiment_distribution": {},
            "speculative_vs_confirmed": {},
            "intensity_distribution": {},
            "charts": {}
        }
    
    # Extract sentiment data
    polarities = [article.get('polarity', 0) for article in articles]
    sentiments = [article.get('sentiment', 'neutral') for article in articles]
    speculation_scores = [article.get('speculation_score', 0) for article in articles]
    certainty_types = [article.get('certainty_type', 'unknown') for article in articles]
    intensities = [article.get('intensity', 'neutral') for article in articles]
    
    # Calculate statistics
    avg_polarity = statistics.mean(polarities) if polarities else 0
    avg_speculation = statistics.mean(speculation_scores) if speculation_scores else 0
    
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
    
    # Count speculative vs. confirmed
    certainty_count = {
        "speculative": certainty_types.count("speculative"),
        "confirmed": certainty_types.count("confirmed")
    }
    
    # Count intensity distribution
    intensity_count = {}
    for intensity in intensities:
        intensity_count[intensity] = intensity_count.get(intensity, 0) + 1
    
    # Generate charts
    charts = {}
    charts["sentiment_distribution"] = generate_sentiment_chart(sentiment_count)
    charts["speculation_chart"] = generate_speculation_chart(certainty_count)
    charts["intensity_chart"] = generate_intensity_chart(intensity_count)
    charts["sentiment_timeline"] = generate_sentiment_timeline(articles)
    
    return {
        "average_sentiment": overall_sentiment,
        "average_polarity": avg_polarity,
        "average_speculation": avg_speculation,
        "sentiment_distribution": sentiment_count,
        "speculative_vs_confirmed": certainty_count,
        "intensity_distribution": intensity_count,
        "charts": charts
    }

def generate_sentiment_chart(sentiment_count: Dict[str, int]) -> str:
    """
    Generate a bar chart showing sentiment distribution.
    
    Args:
        sentiment_count: Dictionary with sentiment counts
        
    Returns:
        Base64 encoded image of the chart
    """
    labels = ['Positive', 'Neutral', 'Negative']
    values = [sentiment_count['positive'], sentiment_count['neutral'], sentiment_count['negative']]
    colors = ['#4CAF50', '#9E9E9E', '#F44336']
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors)
    plt.title('Sentiment Distribution Across Articles')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Articles')
    
    # Save chart as base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Generate base64 string
    chart_str = base64.b64encode(image_png).decode('utf-8')
    
    return chart_str

def generate_speculation_chart(certainty_count: Dict[str, int]) -> str:
    """
    Generate a pie chart showing speculative vs. confirmed distribution.
    
    Args:
        certainty_count: Dictionary with certainty type counts
        
    Returns:
        Base64 encoded image of the chart
    """
    labels = ['Confirmed', 'Speculative']
    values = [certainty_count['confirmed'], certainty_count['speculative']]
    colors = ['#2196F3', '#FF9800']
    
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Speculative vs. Confirmed Content')
    plt.axis('equal')
    
    # Save chart as base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Generate base64 string
    chart_str = base64.b64encode(image_png).decode('utf-8')
    
    return chart_str

def generate_intensity_chart(intensity_count: Dict[str, int]) -> str:
    """
    Generate a horizontal bar chart showing sentiment intensity distribution.
    
    Args:
        intensity_count: Dictionary with intensity counts
        
    Returns:
        Base64 encoded image of the chart
    """
    # Order intensities from most negative to most positive
    ordered_intensities = [
        "extremely negative", "very negative", "moderately negative", 
        "slightly negative", "neutral", "slightly positive", 
        "moderately positive", "very positive", "extremely positive"
    ]
    
    # Filter only intensities that exist in the data
    labels = [intensity for intensity in ordered_intensities if intensity in intensity_count]
    values = [intensity_count.get(intensity, 0) for intensity in labels]
    
    # Generate color gradient from red to gray to green
    colors = []
    for intensity in labels:
        if "negative" in intensity:
            if "extremely" in intensity:
                colors.append('#B71C1C')  # Dark red
            elif "very" in intensity:
                colors.append('#E53935')  # Red
            elif "moderately" in intensity:
                colors.append('#EF5350')  # Light red
            else:
                colors.append('#FFCDD2')  # Very light red
        elif "neutral" in intensity:
            colors.append('#9E9E9E')  # Gray
        else:  # positive
            if "extremely" in intensity:
                colors.append('#1B5E20')  # Dark green
            elif "very" in intensity:
                colors.append('#43A047')  # Green
            elif "moderately" in intensity:
                colors.append('#66BB6A')  # Light green
            else:
                colors.append('#C8E6C9')  # Very light green
    
    plt.figure(figsize=(12, 8))
    plt.barh(labels, values, color=colors)
    plt.title('Sentiment Intensity Distribution')
    plt.xlabel('Number of Articles')
    plt.tight_layout()
    
    # Save chart as base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Generate base64 string
    chart_str = base64.b64encode(image_png).decode('utf-8')
    
    return chart_str

def generate_sentiment_timeline(articles: List[Dict[str, Any]]) -> str:
    """
    Generate a timeline showing sentiment shifts across articles.
    
    Args:
        articles: List of article dictionaries with sentiment data
        
    Returns:
        Base64 encoded image of the chart
    """
    # Extract data for timeline
    article_indices = list(range(1, len(articles) + 1))
    polarities = [article.get('polarity', 0) for article in articles]
    
    # Create figure with two subplots
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot sentiment polarity line
    ax.plot(article_indices, polarities, 'o-', color='blue', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Article Number')
    ax.set_ylabel('Sentiment Polarity (-1 to +1)')
    ax.set_title('Sentiment Shifts Across Articles')
    
    # Add annotation for positive/negative regions
    ax.fill_between(article_indices, polarities, 0, where=[p > 0 for p in polarities], 
                   color='green', alpha=0.3, label='Positive')
    ax.fill_between(article_indices, polarities, 0, where=[p < 0 for p in polarities], 
                   color='red', alpha=0.3, label='Negative')
    
    ax.set_xticks(article_indices)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save chart as base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # Generate base64 string
    chart_str = base64.b64encode(image_png).decode('utf-8')
    
    return chart_str