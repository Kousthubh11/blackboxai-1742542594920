import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from newspaper import Article
from wordcloud import WordCloud
import logging
from collections import Counter
from googletrans import Translator
from cachetools import TTLCache

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize cache
cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes cache

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize translator
translator = Translator()

class NLPProcessor:
    def __init__(self):
        # Ensure NLTK data is downloaded
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob
        Returns: 'positive', 'negative', or 'neutral'
        """
        if not text:
            return "neutral"
        
        cache_key = f"sentiment_{hash(text)}"
        if cache_key in cache:
            return cache[cache_key]
        
        try:
            analysis = TextBlob(str(text))
            polarity = analysis.sentiment.polarity
            
            if polarity > 0.1:
                result = 'positive'
            elif polarity < -0.1:
                result = 'negative'
            else:
                result = 'neutral'
                
            cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return "neutral"

    def extract_keywords(self, text, top_n=10):
        """
        Extract important keywords from text using spaCy
        """
        if not text:
            return []
            
        cache_key = f"keywords_{hash(text)}"
        if cache_key in cache:
            return cache[cache_key]
            
        try:
            doc = nlp(text)
            keywords = [token.text.lower() for token in doc 
                       if not token.is_stop and not token.is_punct and token.is_alpha]
            
            # Get frequency distribution
            freq_dist = Counter(keywords)
            top_keywords = freq_dist.most_common(top_n)
            
            cache[cache_key] = top_keywords
            return top_keywords
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return []

    def summarize_article(self, url):
        """
        Generate article summary using newspaper3k
        """
        cache_key = f"summary_{url}"
        if cache_key in cache:
            return cache[cache_key]
            
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            
            summary = article.summary
            cache[cache_key] = summary
            return summary
        except Exception as e:
            logger.error(f"Article summarization error: {e}")
            return None

    def generate_wordcloud(self, text):
        """
        Generate WordCloud from text
        """
        if not text:
            return None
            
        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=self.stop_words
            ).generate(text)
            return wordcloud
        except Exception as e:
            logger.error(f"WordCloud generation error: {e}")
            return None

    def translate_text(self, text, target_lang='en'):
        """
        Translate text to target language
        """
        if not text or target_lang == 'en':
            return text
            
        cache_key = f"trans_{hash(text)}_{target_lang}"
        if cache_key in cache:
            return cache[cache_key]
            
        try:
            translation = translator.translate(text, dest=target_lang)
            cache[cache_key] = translation.text
            return translation.text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

    def get_named_entities(self, text):
        """
        Extract named entities using spaCy
        """
        if not text:
            return []
            
        cache_key = f"ner_{hash(text)}"
        if cache_key in cache:
            return cache[cache_key]
            
        try:
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            cache[cache_key] = entities
            return entities
        except Exception as e:
            logger.error(f"Named entity recognition error: {e}")
            return []

    def compute_text_similarity(self, text1, text2):
        """
        Compute cosine similarity between two texts
        """
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            return (tfidf_matrix * tfidf_matrix.T).toarray()[0][1]
        except Exception as e:
            logger.error(f"Text similarity computation error: {e}")
            return 0.0

# Initialize global NLP processor
nlp_processor = NLPProcessor()