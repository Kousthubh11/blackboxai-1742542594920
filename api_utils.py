import requests
import logging
from cachetools import TTLCache
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Union
from config import Config
from nlp_utils import nlp_processor

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize cache with 30-minute TTL
news_cache = TTLCache(maxsize=100, ttl=1800)  # 30 minutes cache

class NewsAPIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.NEWS_API_KEY
        self.base_url = Config.BASE_URL
        if not self.api_key:
            raise ValueError("News API key is required")

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make HTTP request to News API with error handling and logging
        """
        params['apiKey'] = self.api_key
        cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"

        # Check cache first
        if cache_key in news_cache:
            logger.info("Returning cached response")
            return news_cache[cache_key]

        try:
            response = requests.get(
                self.base_url + endpoint,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 'ok':
                # Cache successful response
                news_cache[cache_key] = data
                return data
            else:
                logger.error(f"API Error: {data.get('message', 'Unknown error')}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return None

    def enrich_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Enrich articles with NLP analysis
        """
        enriched = []
        for article in articles:
            try:
                # Basic text for analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                # Add NLP enrichments
                article['sentiment'] = nlp_processor.analyze_sentiment(text)
                article['keywords'] = nlp_processor.extract_keywords(text, top_n=5)
                article['named_entities'] = nlp_processor.get_named_entities(text)
                
                # Get article summary if URL exists
                if article.get('url'):
                    article['summary'] = nlp_processor.summarize_article(article['url'])
                
                enriched.append(article)
            except Exception as e:
                logger.error(f"Error enriching article: {str(e)}")
                enriched.append(article)  # Add original article without enrichment
                
        return enriched

    def get_top_headlines(self, 
                         country: str = None,
                         category: str = None,
                         q: str = None,
                         page_size: int = 20,
                         page: int = 1) -> List[Dict]:
        """
        Fetch top headlines with optional filters
        """
        params = {
            'pageSize': min(page_size, 100),  # API limit is 100
            'page': page
        }
        
        if country:
            params['country'] = country
        if category:
            params['category'] = category
        if q:
            params['q'] = q

        response = self._make_request('', params)
        if response and 'articles' in response:
            articles = response['articles']
            return self.enrich_articles(articles)
        return []

    def search_everything(self,
                         q: str,
                         from_date: datetime = None,
                         to_date: datetime = None,
                         language: str = 'en',
                         sort_by: str = 'publishedAt',
                         page_size: int = 20,
                         page: int = 1) -> List[Dict]:
        """
        Search all articles with various filters
        """
        params = {
            'q': q,
            'language': language,
            'sortBy': sort_by,
            'pageSize': min(page_size, 100),
            'page': page
        }

        if from_date:
            params['from'] = from_date.isoformat()
        if to_date:
            params['to'] = to_date.isoformat()

        response = self._make_request('/everything', params)
        if response and 'articles' in response:
            articles = response['articles']
            return self.enrich_articles(articles)
        return []

    def get_sources(self, 
                    category: str = None,
                    language: str = 'en',
                    country: str = None) -> List[Dict]:
        """
        Get available news sources with optional filters
        """
        params = {'language': language}
        
        if category:
            params['category'] = category
        if country:
            params['country'] = country

        response = self._make_request('/sources', params)
        return response.get('sources', []) if response else []

class APIError(Exception):
    """Custom exception for API-related errors"""
    pass

# Initialize global API client
try:
    news_api = NewsAPIClient()
except ValueError as e:
    logger.error(f"Failed to initialize NewsAPIClient: {str(e)}")
    news_api = None