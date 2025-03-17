from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from news_extractor import extract_company_news
from sentiment import analyze_sentiment
from tts import convert_to_hindi_speech
import os

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

@app.post("/api/text-to-speech")
async def generate_speech(text: str):
    try:
        audio_path = convert_to_hindi_speech(text)
        return {"audio_path": audio_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

# Add a simple root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Company News Analysis API"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)