from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from news_extractor import extract_company_news
from sentiment import analyze_sentiment, compare_sentiments
import os
import base64
from io import BytesIO
from gtts import gTTS

app = FastAPI(title="Company News Analysis API")

class CompanyRequest(BaseModel):
    company_name: str
    num_articles: Optional[int] = 10
    force_refresh: Optional[bool] = False  # This can be removed if not needed

@app.post("/api/extract-news", response_model=List[Dict[str, Any]])
async def get_news(request: CompanyRequest):
    try:
        company_name = request.company_name
        num_articles = request.num_articles
        
        print(f"Received request to extract news for: {company_name}, articles: {num_articles}")
        
        # Extract new articles directly (no database check)
        print(f"Extracting data for {company_name}")
        articles = extract_company_news(company_name, num_articles)
        
        if not articles:
            print("No articles found. Returning 404 error.")
            raise HTTPException(status_code=404, detail="No news articles found for this company")
        
        return articles
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error extracting news: {str(e)}")

@app.post("/api/analyze-sentiment")
async def analyze_news_sentiment(request: CompanyRequest):
    try:
        company_name = request.company_name
        num_articles = request.num_articles
        
        # Extract articles directly (no database check)
        articles = extract_company_news(company_name, num_articles)
        if not articles:
            raise HTTPException(status_code=404, detail="No news articles found for this company")
        
        # Analyze sentiment for each article
        for article in articles:
            if 'raw_content' in article:
                print(f"Analyzing article: {article.get('title', 'Unknown')}")
                sentiment_result = analyze_sentiment(article['raw_content'])
                article.update(sentiment_result)
            else:
                print(f"Missing raw_content for article: {article.get('title', 'Unknown')}")
                article.update({
                    'sentiment': 'Unknown',
                    'polarity': 0,
                    'subjectivity': 0,
                    'vader_compound': 0,
                    'speculation_score': 0,
                    'certainty_type': 'Unknown',
                    'intensity': 'Unknown'
                })
        
        return {"articles": articles}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")


@app.post("/api/compare-sentiment")
async def compare_news_sentiment(request: CompanyRequest):
    try:
        company_name = request.company_name
        num_articles = request.num_articles
        
        # Extract articles
        articles = extract_company_news(company_name, num_articles)
        if not articles:
            raise HTTPException(status_code=404, detail="No news articles found for this company")
        
        # Analyze sentiment for each article
        for article in articles:
            if 'raw_content' in article:
                sentiment_result = analyze_sentiment(article['raw_content'])
                article.update(sentiment_result)
            else:
                article.update({
                    'sentiment': 'Unknown',
                    'polarity': 0,
                    'subjectivity': 0,
                    'vader_compound': 0,
                    'speculation_score': 0,
                    'certainty_type': 'Unknown',
                    'intensity': 'Unknown'
                })
        
        # Perform comparative analysis
        comparison_results = compare_sentiments(articles)
        
        return comparison_results
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error comparing sentiment: {str(e)}")
    
def generate_hindi_audio_content(text: str) -> str:
    """Generate Hindi audio and return base64 encoded content"""
    try:
        audio_bytes = BytesIO()
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
        return audio_base64
    except Exception as e:
        print(f"Error generating audio content: {e}")
        return None
    
    
@app.post("/api/final-analysis")
async def generate_final_analysis(request: CompanyRequest):
    try:
        company_name = request.company_name
        num_articles = request.num_articles
        
        # Extract articles
        articles = extract_company_news(company_name, num_articles)
        if not articles:
            raise HTTPException(status_code=404, detail="No news articles found for this company")
        
        # Analyze sentiment for each article
        analyzed_articles = []
        for article in articles:
            if 'raw_content' in article:
                sentiment_result = analyze_sentiment(article['raw_content'])
                article.update(sentiment_result)
            else:
                article.update({
                    'sentiment': 'Unknown',
                    'polarity': 0,
                    'subjectivity': 0,
                    'vader_compound': 0,
                    'speculation_score': 0,
                    'certainty_type': 'Unknown',
                    'intensity': 'Unknown'
                })
            analyzed_articles.append(article)
        
        # Generate comparative analysis
        comparison_results = compare_sentiments(analyzed_articles)
        
        # Create the final report structure
        final_report = create_final_report(company_name, analyzed_articles, comparison_results)
        
        # Generate Hindi TTS for the final sentiment analysis
        try:
            hindi_summary = translate_to_hindi(final_report["Final Sentiment Analysis"])
            audio_content = generate_hindi_audio_content(hindi_summary)
            final_report["AudioContent"] = audio_content  # This contains the actual audio bytes
            final_report["Audio"] = "Generated"  # Just a marker
        except Exception as e:
            print(f"Error generating Hindi TTS: {e}")
            final_report["Audio"] = "Audio generation failed"
            final_report["AudioContent"] = None
        
        return final_report

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating final analysis: {str(e)}")

def create_final_report(company_name: str, articles: List[Dict[str, Any]], comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a structured final report with comparative analysis.
    """
    # Format articles for the report
    formatted_articles = []
    for article in articles:
        formatted_article = {
            "Title": article.get('title', 'No title available'),
            "Summary": article.get('summary', 'No summary available'),
            "Sentiment": article.get('sentiment', 'neutral').capitalize(),
            "Topics": article.get('keywords', [])
        }
        formatted_articles.append(formatted_article)
    
    # Count sentiment distribution
    sentiment_counts = {
        "Positive": sum(1 for a in articles if a.get('sentiment') == 'positive'),
        "Negative": sum(1 for a in articles if a.get('sentiment') == 'negative'),
        "Neutral": sum(1 for a in articles if a.get('sentiment') == 'neutral')
    }
    
    # Generate coverage differences (pairwise comparisons of articles)
    coverage_differences = []
    if len(articles) >= 2:
        for i in range(min(3, len(articles)-1)):  # Limit to 3 comparisons
            article1 = articles[i]
            article2 = articles[i+1]
            
            comparison = {
                "Comparison": f"Article {i+1} focuses on {', '.join(article1.get('keywords', [])[:2])}, " +
                             f"while Article {i+2} discusses {', '.join(article2.get('keywords', [])[:2])}.",
                "Impact": f"The first article presents a {article1.get('sentiment', 'neutral')} perspective, " +
                         f"while the second offers a {article2.get('sentiment', 'neutral')} viewpoint on {company_name}."
            }
            coverage_differences.append(comparison)
    
    # Find topic overlap between articles
    all_topics = [set(article.get('keywords', [])) for article in articles]
    common_topics = set.intersection(*all_topics) if all_topics else set()
    
    # Find unique topics per article
    unique_topics = {}
    for i, article in enumerate(articles):
        article_topics = set(article.get('keywords', []))
        other_topics = set().union(*[topics for j, topics in enumerate(all_topics) if j != i])
        unique_to_article = article_topics - other_topics
        unique_topics[f"Unique Topics in Article {i+1}"] = list(unique_to_article)
    
    # Generate final sentiment analysis
    dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0] if sentiment_counts else "Neutral"
    final_analysis = f"{company_name}'s latest news coverage is {dominant_sentiment.lower()}. "
    
    if dominant_sentiment == "Positive":
        final_analysis += "The articles generally highlight strengths and opportunities for growth."
    elif dominant_sentiment == "Negative":
        final_analysis += "The articles generally highlight challenges and concerns facing the company."
    else:
        final_analysis += "The articles present a balanced view of the company's current situation."
    
    # Create the final report structure
    final_report = {
        "Company": company_name,
        "Articles": formatted_articles,
        "Comparative Sentiment Score": {
            "Sentiment Distribution": sentiment_counts,
            "Coverage Differences": coverage_differences,
            "Topic Overlap": {
                "Common Topics": list(common_topics),
                **unique_topics
            }
        },
        "Final Sentiment Analysis": final_analysis,
        "Audio": "Not yet generated"
    }
    
    return final_report

def translate_to_hindi(text: str) -> str:
    """
    Translate English text to Hindi using OpenAI.
    """
    try:
        import openai
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a translator. Translate the following text from English to Hindi."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        hindi_text = response['choices'][0]['message']['content'].strip()
        return hindi_text
    except Exception as e:
        print(f"Error translating to Hindi: {e}")
        return text  # Return original text if translation fails
    
# Add a simple root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Company News Analysis API"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
