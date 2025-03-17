import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any
import time
import random
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import statistics
from urllib.parse import urlparse  # Add this line
import json

import openai


def extract_company_news(company_name: str, num_articles: int = 10) -> List[Dict[str, Any]]:
    # List to store extracted news articles
    news_articles = []
    
    # User agent to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Search query for Google News
    query = f"{company_name} news"
    encoded_query = query.replace(' ', '+')
    
    # Google News URL
    url = f"https://www.google.com/search?q={encoded_query}&tbm=nws"
    
    try:
        # Send request to Google News
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find news article elements
        article_elements = soup.find('div', id='center_col')
        i = 0
        if article_elements:    
            divs = article_elements.find_all('div', class_='SoaBEf')
            for div in divs:
                i = i + 1
                if len(news_articles) >= num_articles:
                    break
                    
                try:
                    # Extract link
                    link_element = div.find('a')
                    url = link_element['href'] if link_element and 'href' in link_element.attrs else ""

                    # Check if the URL is valid
                    if not url or not url.startswith('http'):
                        print("Invalid URL found, skipping this article.")
                        continue  # Skip to the next iteration if the URL is invalid
                    
                    # Extract actual URL from Google redirect
                    if url.startswith('/url?'):
                        url_match = re.search(r'url\?q=([^&]+)', url)
                        if url_match:
                            url = url_match.group(1)
                    
                    # Skip if no valid URL
                    if not url:
                        continue
                    
                    # Extract article content
                    article_data = extract_article_content(i, url, headers)
                    
                    # Check if article_data is empty (which means access was denied or another error)
                    if not article_data or not article_data.get('content'):
                        print(f"No content retrieved for {url}, skipping to next article.")
                        continue
                        
                    # Generate metadata
                    metadata = generate_article_metadata(article_data['content'], company_name)
                    
                    # Combine all data
                    article_info = {
                        'url': url,
                        'raw_content': article_data.get('content', 'No content available'),
                        'title': metadata.get('title', article_data.get('title', 'No title available')),
                        'summary': metadata.get('summary', 'No summary available'),
                        'source': article_data.get('source', 'Unknown'),
                        'date': article_data.get('date', 'Unknown'),
                        'keywords': metadata.get('keywords', []),
                        'relevance': metadata.get('relevance', 'Unknown'),
                    }
                    
                    # Add to list if not already present (check by URL)
                    if not any(article['url'] == url for article in news_articles):
                        news_articles.append(article_info)
                        print(f"Extracted article {len(news_articles)}/{num_articles}: {article_info['title']}")
                    
                    # Add a small delay to be respectful to servers
                    time.sleep(random.uniform(1.0, 2.0))
                    
                except Exception as e:
                    print(f"Error processing article: {e}")
                    continue  # Continue to the next article on any error
    
    except Exception as e:
        print(f"Error during news extraction: {e}")
    
    # Modify the final return statement to include display_news functionality
    combined_article_info = display_news(news_articles)
    return combined_article_info
        
def extract_article_content(i, url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract content from a news article URL.
    
    Args:
        url: The URL of the article
        headers: HTTP headers for the request
        
    Returns:
        Dictionary with article content and metadata or empty dict if access denied
    """
    try:
        # Enhance headers to appear more like a legitimate browser
        enhanced_headers = headers.copy()
        enhanced_headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        })
        
        # Add a referer if possible
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        enhanced_headers['Referer'] = domain
        
        # Rotate user agents to avoid detection
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        enhanced_headers['User-Agent'] = random.choice(user_agents)
        
        # Use a session to maintain cookies
        session = requests.Session()
        
        # Add a longer timeout and retry mechanism
        max_retries = 3
        access_granted = False
        
        for attempt in range(max_retries):
            try:
                # Increase timeout and add a randomized delay between attempts
                response = session.get(url, headers=enhanced_headers, timeout=15)
                
                # Check if we got a successful response
                if response.status_code == 200:
                    access_granted = True
                    break
                elif response.status_code == 403:
                    print(f"Access forbidden for {url}, attempt {attempt+1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(random.uniform(2.0, 5.0))
                    else:
                        print(f"Max retries reached for {url}. Skipping this article.")
                        return {}  # Return empty dict to signal skipping this article
                else:
                    response.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    wait_time = random.uniform(3.0, 7.0)
                    print(f"Request failed, retrying in {wait_time:.1f} seconds... ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to access {url} after {max_retries} attempts. Error: {str(e)}")
                    return {}  # Return empty dict to signal skipping this article
        
        # If we didn't get access after all retries, return empty dict
        if not access_granted:
            return {}
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract title
        title = soup.title.text.strip() if soup.title else "No title available"
        
        # Extract source from domain
        source = url.split('/')[2] if len(url.split('/')) > 2 else "Unknown"
        
        # Try to find date in common formats
        date = "Unknown"
        date_patterns = [
            # Look for time elements
            soup.find('time'),
            # Look for meta tags with date information
            soup.find('meta', property='article:published_time'),
            soup.find('meta', property='og:published_time'),
            # Look for spans or divs with date-related classes
            soup.find(['span', 'div'], class_=lambda c: c and any(date_term in c.lower() for date_term in ['date', 'time', 'published']))
        ]
        
        for pattern in date_patterns:
            if pattern:
                if pattern.name == 'meta':
                    date = pattern.get('content', '')
                else:
                    date = pattern.text.strip()
                if date:
                    break
        
        # Extract main content
        # Strategy 1: Look for article tag
        content = ""
        article_tag = soup.find('article')
        
        if article_tag:
            # Get all paragraphs within the article
            paragraphs = article_tag.find_all('p')
            content = ' '.join([p.text.strip() for p in paragraphs])
        
        # Strategy 2: Look for common content containers
        if not content:
            content_containers = soup.find_all(['div', 'section'], class_=lambda c: c and any(
                content_term in c.lower() for content_term in ['content', 'article', 'story', 'body', 'text', 'main']
            ))
            
            for container in content_containers:
                paragraphs = container.find_all('p')
                if paragraphs:
                    content = ' '.join([p.text.strip() for p in paragraphs])
                    break
        
        # Strategy 3: Just get all paragraphs if nothing else worked
        if not content or len(content) < 300:
            # Exclude navigation, footer, sidebar elements
            for nav in soup.find_all(['nav', 'footer', 'aside']):
                nav.decompose()
                
            paragraphs = soup.find_all('p')
            content = ' '.join([p.text.strip() for p in paragraphs])
        
        # Clean up content
        content = re.sub(r'\s+', ' ', content).strip()

        try:
            # Only write to file if content was successfully extracted
            if content:
                with open(f'output/article_{i}.json', 'w') as file:
                    json.dump({
                        'url': url,
                        'title': title,
                        'source': source,
                        'date': date,
                        'content': content
                    }, file)
        except Exception as file_error:
            print(f"Error writing article to file: {file_error}")
        
        return {
            'url': str(url),
            'title': title,
            'source': source,
            'date': date,
            'content': content
        }
    
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return {}  # Return empty dict on any error
            

def generate_article_metadata(article_content: str, company_name: str) -> Dict[str, Any]:
    """
    Generate metadata for an article using OpenAI API.
    
    Args:
        article_content: The content of the article
        company_name: The name of the company
        
    Returns:
        Dictionary with generated metadata
    """
    try:
        
        openai.api_key = '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'
        
        # Truncate content if too long (OpenAI has token limits)
        max_content_length = 4000  # Adjust based on your OpenAI model
        truncated_content = article_content[:max_content_length] if len(article_content) > max_content_length else article_content
        
        # Create prompt for OpenAI
        prompt = f"""
        Analyze the following news article about {company_name} and extract:
        1. A brief summary about what the article states
        2. 5 key keywords or topics that are relevant to the article
        3. Relevance to {company_name} (high, medium, or low)
        
        Article content:
        {truncated_content}
        
        Format your response as JSON with the following structure:
        {{
           
            "summary": "Brief summary of the article",
            "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
            "relevance": "high/medium/low"
        }}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or another appropriate model
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Extract and parse the response
        result = response['choices'][0]['message']['content'].strip()
        
        
        
        # Find JSON in the response
        json_match = re.search(r'({.*})', result, re.DOTALL)
        if json_match:
            import json
            metadata = json.loads(json_match.group(1))
            return metadata
        else:
            print("Failed to parse OpenAI response as JSON")
            return {}
    
    except Exception as e:
        print(f"Error generating metadata with OpenAI: {e}")
        return {}

def display_news(news_articles: List[Dict[str, Any]]) -> None:
    """
    Display the extracted news articles in a readable format.
    
    Args:
        news_articles: List of news article dictionaries
    """
    if not news_articles:
        print("No news articles found.")
        return
        
    print(f"\n{'=' * 80}")
    print(f"Found {len(news_articles)} news articles about the company:")
    print(f"{'=' * 80}\n")
    
    combined = []
    for i, article in enumerate(news_articles, 1):
        # Extract data from the article
        url = article.get('url', 'No url available')
        title = article.get('title', 'No title available')
        source = article.get('source', 'Unknown')
        date = article.get('date', 'Unknown')
        original_content = article.get('raw_content', 'No content available')
        
        # Extract metadata from the article
        summary = article.get('summary', 'No summary available')
        keywords = article.get('keywords', [])
        relevance = article.get('relevance', 'Unknown')
        
        # Combine all data into a single dictionary
        combined_info = {
            'url': url,
            'title': title,
            'source': source,
            'date': date,
            'summary': summary,
            'keywords': keywords,
            'relevance': relevance,
            'raw_content': original_content
        }

        combined.append(combined_info)

    return combined
